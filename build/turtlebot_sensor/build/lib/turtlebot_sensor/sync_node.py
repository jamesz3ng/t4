import cv2
import rclpy
import os
import numpy as np
import math
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

class CubeDetectionNode(Node):
    def __init__(self):
        super().__init__("cube_detection_node")
        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID')
        
        # Declare parameters
        self.declare_parameter(
            name="image_sub_topic",
            value=f"/T{self.robot_id_str}/oakd/rgb/image_raw/compressed",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
            ),
        )
        
        # Golden color HSV ranges (adjust these based on your cube's specific color)
        self.declare_parameter("hue_min", 16)
        self.declare_parameter("hue_max", 39)
        self.declare_parameter("sat_min", 119)
        self.declare_parameter("sat_max", 183)
        self.declare_parameter("val_min", 123)
        self.declare_parameter("val_max", 202)
        
        # Shape detection parameters
        self.declare_parameter("min_contour_area", 500)
        self.declare_parameter("max_contour_area", 30000)
        self.declare_parameter("epsilon_factor", 0.02)  # For polygon approximation
        
        # Temporal smoothing parameters (add these for stability)
        self.declare_parameter("temporal_buffer_size", 4)
        self.declare_parameter("min_consistent_detections", 2)  # Need 2/4 frames to start detecting
        self.declare_parameter("confidence_threshold", 40.0)  # Lower threshold for your tuned params
        
        image_sub_topic = (
            self.get_parameter("image_sub_topic").get_parameter_value().string_value
        )
        
        self.get_logger().info(f"{image_sub_topic=}")
        
        self.image_sub = Subscriber(
            self, CompressedImage, image_sub_topic, qos_profile=qos_profile_sensor_data
        )
        
        self.cv_bridge = CvBridge()
        self.image_sub.registerCallback(self.image_callback)
        
        # Tracking variables for temporal smoothing
        self.detection_history = []
        self.max_history = 5
        self.detection_buffer = []
        self.buffer_size = 4  # Match the parameter
        self.min_consistent_detections = 2  # Match the parameter
        self.last_valid_detection = None  # This will store a list of detection dictionaries
        
        self.get_logger().info("Cube detection node initialized and waiting for images...")

    def get_color_mask(self, image):
        """Create a mask for golden/yellow colors"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get HSV parameters
        h_min = self.get_parameter("hue_min").get_parameter_value().integer_value
        h_max = self.get_parameter("hue_max").get_parameter_value().integer_value
        s_min = self.get_parameter("sat_min").get_parameter_value().integer_value
        s_max = self.get_parameter("sat_max").get_parameter_value().integer_value
        v_min = self.get_parameter("val_min").get_parameter_value().integer_value
        v_max = self.get_parameter("val_max").get_parameter_value().integer_value
        
        lower_golden = np.array([h_min, s_min, v_min])
        upper_golden = np.array([h_max, s_max, v_max])
        
        mask = cv2.inRange(hsv, lower_golden, upper_golden)
        
        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def is_square_like(self, contour):
        """Check if a contour is square-like"""
        epsilon = self.get_parameter("epsilon_factor").get_parameter_value().double_value * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # A square should have 4 vertices
        if len(approx) != 4:
            return False, approx
        
        # Check if the shape is convex
        if not cv2.isContourConvex(approx):
            return False, approx
        
        # Calculate side lengths
        sides = []
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % 4][0]
            side_length = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            sides.append(side_length)
        
        # Check if all sides are approximately equal (within 20% tolerance)
        avg_side = np.mean(sides)
        for side in sides:
            if abs(side - avg_side) / avg_side > 0.3:  # 30% tolerance
                return False, approx
        
        # Check angles (should be close to 90 degrees)
        angles = []
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % 4][0]
            p3 = approx[(i + 2) % 4][0]
            
            # Calculate vectors
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
            angle = math.degrees(math.acos(abs(cos_angle)))
            angles.append(angle)
        
        # Check if angles are close to 90 degrees (within 25 degrees tolerance)
        for angle in angles:
            if abs(angle - 90) > 25:
                return False, approx
        
        return True, approx

    def detect_cubes(self, image):
        """Main cube detection function"""
        # Get color mask
        color_mask = self.get_color_mask(image)
        
        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cube_candidates = []
        min_area = self.get_parameter("min_contour_area").get_parameter_value().integer_value
        max_area = self.get_parameter("max_contour_area").get_parameter_value().integer_value
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Check if it's square-like
            is_square, approx = self.is_square_like(contour)
            
            if is_square:
                # Calculate center and bounding box
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate aspect ratio
                    aspect_ratio = float(w) / h
                    
                    # Squares should have aspect ratio close to 1
                    if 0.7 <= aspect_ratio <= 1.4:
                        cube_candidates.append({
                            'contour': contour,
                            'approx': approx,
                            'center': (cx, cy),
                            'area': area,
                            'bbox': (x, y, w, h),
                            'aspect_ratio': aspect_ratio
                        })
        
        return cube_candidates

    def calculate_confidence(self, contour, area, aspect_ratio, approx):
        """Calculate confidence score for detection stability"""
        confidence = 100.0
        
        # Penalize deviation from square aspect ratio
        aspect_penalty = abs(aspect_ratio - 1.0) * 50
        confidence -= aspect_penalty
        
        # Penalize very small or very large areas (adjusted for your area range)
        if area < 800:
            confidence -= (800 - area) / 10
        elif area > 20000:
            confidence -= (area - 20000) / 100
        
        # Check how well the approximation fits the original contour
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:  # Avoid division by zero
            approx_perimeter = cv2.arcLength(approx, True)
            fit_penalty = abs(perimeter - approx_perimeter) / perimeter * 30
            confidence -= fit_penalty
        
        # Boost confidence for areas in your sweet spot (1500-2500 based on logs)
        if 1500 <= area <= 2500:
            confidence += 10
        
        return max(0, min(100, confidence))

    def draw_detections(self, image, cube_candidates):
        """Draw detection results on the image"""
        result_image = image.copy()
        
        for i, cube in enumerate(cube_candidates):
            # Draw contour
            cv2.drawContours(result_image, [cube['contour']], -1, (0, 255, 0), 2)
            
            # Draw approximated polygon
            cv2.drawContours(result_image, [cube['approx']], -1, (255, 0, 0), 2)
            
            # Draw center point
            cv2.circle(result_image, cube['center'], 5, (0, 0, 255), -1)
            
            # Draw bounding box
            x, y, w, h = cube['bbox']
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            
            # Add text information with confidence
            confidence = cube.get('confidence', 0)
            text = f"Cube {i+1}: Area={int(cube['area'])}, Conf={confidence:.1f}"
            cv2.putText(result_image, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add detection status
        status_text = f"Buffer: {self.detection_buffer}, Detections: {len(cube_candidates)}"
        cv2.putText(result_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return result_image

    def smooth_detections(self, current_detections):
        """Apply temporal smoothing to reduce flickering"""
        # Add current detection count to buffer
        self.detection_buffer.append(len(current_detections))
        if len(self.detection_buffer) > self.buffer_size:
            self.detection_buffer.pop(0)
        
        # Only report detections if we have consistent detection over multiple frames
        if len(self.detection_buffer) >= self.min_consistent_detections:
            consistent_detections = sum(1 for count in self.detection_buffer if count > 0)
            
            # For reporting a detection, need consistency
            if consistent_detections >= self.min_consistent_detections:
                if current_detections:
                    self.last_valid_detection = current_detections.copy()
                    return current_detections
            
            # For stopping detection, be more responsive
            # If the last 2 frames have no detections, stop reporting
            recent_detections = sum(1 for count in self.detection_buffer[-2:] if count > 0)
            if recent_detections == 0:
                self.last_valid_detection = None
                return []
            
            # Otherwise, continue with last valid detection if available
            if self.last_valid_detection is not None:
                return self.last_valid_detection
        
        return []  # Not enough consistent detections

    def are_detections_similar(self, det1, det2, max_distance=50):
        """Check if two detections are likely the same object"""
        if det1 is None or det2 is None:
            return False
        
        center1 = det1.get('center', (0, 0))
        center2 = det2.get('center', (0, 0))
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance < max_distance
        """Draw detection results on the image"""
        result_image = image.copy()
        
        for i, cube in enumerate(cube_candidates):
            # Draw contour
            cv2.drawContours(result_image, [cube['contour']], -1, (0, 255, 0), 2)
            
            # Draw approximated polygon
            cv2.drawContours(result_image, [cube['approx']], -1, (255, 0, 0), 2)
            
            # Draw center point
            cv2.circle(result_image, cube['center'], 5, (0, 0, 255), -1)
            
            # Draw bounding box
            x, y, w, h = cube['bbox']
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            
            # Add text information
            text = f"Cube {i+1}: Area={int(cube['area'])}"
            cv2.putText(result_image, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image

    def image_callback(self, image_msg: CompressedImage):
        try:
            self.get_logger().info("Processing image for cube detection...")
            image = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg)
            
            # Detect cubes
            raw_detections = self.detect_cubes(image)
            
            # Apply temporal smoothing
            stable_detections = self.smooth_detections(raw_detections)
            
            # Log detection results
            if stable_detections:
                self.get_logger().info(f"STABLE: Detected {len(stable_detections)} cube(s)")
                for i, cube in enumerate(stable_detections):
                    cx, cy = cube['center']
                    area = cube['area']
                    confidence = cube.get('confidence', 0)
                    self.get_logger().info(f"  Cube {i+1}: Center=({cx}, {cy}), Area={area}, Conf={confidence:.1f}")
            else:
                if raw_detections:
                    self.get_logger().info(f"UNSTABLE: {len(raw_detections)} raw detection(s) but not stable enough")
                else:
                    self.get_logger().info("No cube candidates detected")
            
            # Draw results (show stable detections)
            result_image = self.draw_detections(image, stable_detections)
            
            # Also show the color mask for debugging
            color_mask = self.get_color_mask(image)
            mask_colored = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
            
            # Create a side-by-side display
            combined = np.hstack([result_image, mask_colored])
            
            cv2.imshow("Cube Detection (Original | Color Mask)", combined)
            cv2.waitKey(1)
            
            # Store detection history for tracking
            self.detection_history.append(len(stable_detections))
            if len(self.detection_history) > self.max_history:
                self.detection_history.pop(0)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    cube_detector = CubeDetectionNode()
    
    try:
        rclpy.spin(cube_detector)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        cv2.destroyAllWindows()
        cube_detector.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
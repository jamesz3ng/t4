import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import Subscriber
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from collections import deque

class GoldenCubeDetectorNode(Node):
    def __init__(self):
        super().__init__("sync_node")
        
        # Parameters
        self.declare_parameter(
            name="image_sub_topic",
            value="/T7/oakd/rgb/image_raw/compressed",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
            ),
        )
        
        # Default parametres
        self.declare_parameter("hue_min", 20)
        self.declare_parameter("hue_max", 40)
        self.declare_parameter("sat_min", 100)
        self.declare_parameter("sat_max", 255)
        self.declare_parameter("val_min", 100)
        self.declare_parameter("val_max", 255)
        
        # Stability parameters
        self.declare_parameter("confidence_threshold", 0.6)
        self.declare_parameter("smoothing_window", 5)
        self.declare_parameter("min_detection_area", 500)
        
        # Get parameters
        image_sub_topic = self.get_parameter("image_sub_topic").get_parameter_value().string_value
        self.hue_min = self.get_parameter("hue_min").get_parameter_value().integer_value
        self.hue_max = self.get_parameter("hue_max").get_parameter_value().integer_value
        self.sat_min = self.get_parameter("sat_min").get_parameter_value().integer_value
        self.sat_max = self.get_parameter("sat_max").get_parameter_value().integer_value
        self.val_min = self.get_parameter("val_min").get_parameter_value().integer_value
        self.val_max = self.get_parameter("val_max").get_parameter_value().integer_value
        
        # self.confidence_threshold = 0.30
        self.confidence_threshold = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        self.smoothing_window_size = self.get_parameter("smoothing_window").get_parameter_value().integer_value
        self.min_detection_area = self.get_parameter("min_detection_area").get_parameter_value().integer_value
        
        self.get_logger().info(f"{image_sub_topic=}")
        self.get_logger().info(f"HSV thresholds: H({self.hue_min}-{self.hue_max}), S({self.sat_min}-{self.sat_max}), V({self.val_min}-{self.val_max})")
        self.get_logger().info(f"Confidence threshold: {self.confidence_threshold}")
        self.get_logger().info(f"Smoothing window size: {self.smoothing_window_size}")
        
        # Create subscribers
        self.image_sub = Subscriber(
            self, CompressedImage, image_sub_topic, qos_profile=qos_profile_sensor_data
        )
        
        # Create publishers
        self.cube_position_pub = self.create_publisher(Point, '/golden_cube/position', 10)
        self.cube_properties_pub = self.create_publisher(Float32MultiArray, '/golden_cube/properties', 10)
        
        self.cv_bridge = CvBridge()
        
        # Register callback
        self.image_sub.registerCallback(self.image_callback)
        self.get_logger().info("Golden Cube Detector Node initialized and waiting for images...")
        
        # For tracking processing performance
        self.frame_count = 0
        self.last_log_time = self.get_clock().now()
        
        # Tracking variables for stability
        self.last_detections = deque(maxlen=self.smoothing_window_size)
        self.last_valid_detection = None
        self.detection_counter = 0
        self.no_detection_counter = 0
        
    def image_callback(self, image_msg: CompressedImage):
        try:
            # Performance tracking
            self.frame_count += 1
            current_time = self.get_clock().now()
            if (current_time.nanoseconds - self.last_log_time.nanoseconds) > 5e9:  # Log every 5 seconds
                fps = self.frame_count / ((current_time.nanoseconds - self.last_log_time.nanoseconds) / 1e9)
                self.get_logger().info(f"Processing at {fps:.2f} FPS")
                self.frame_count = 0
                self.last_log_time = current_time
            
            # Convert compressed image to OpenCV format
            image = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg)
            
            # Create a copy for visualization
            display_image = image.copy()
            
            # Process the image to detect golden cubes
            detection_result = self.detect_golden_cube(image, display_image)
            raw_detection, cube_center, cube_size, raw_confidence = detection_result
            
            # Apply temporal smoothing for stability
            stable_detection, stable_center, stable_size, stable_confidence = self.stabilize_detection(
                raw_detection, cube_center, cube_size, raw_confidence
            )
            
            # Publish results if a cube is detected
            if stable_detection:
                # Publish cube position
                position_msg = Point()
                position_msg.x = float(stable_center[0])  # Pixel X
                position_msg.y = float(stable_center[1])  # Pixel Y
                position_msg.z = 0.0  # No depth information
                self.cube_position_pub.publish(position_msg)
                
                # Publish cube properties
                properties_msg = Float32MultiArray()
                properties_msg.data = [float(stable_size), stable_confidence]
                self.cube_properties_pub.publish(properties_msg)
                
                # Draw stable detection results on the display image (in blue)
                x = int(stable_center[0] - stable_size/2)
                y = int(stable_center[1] - stable_size/2)
                w = int(stable_size)
                h = int(stable_size)
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(display_image, (int(stable_center[0]), int(stable_center[1])), 5, (255, 0, 0), -1)
                cv2.putText(display_image, f"Stable: {stable_confidence:.2f}", (x, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                self.get_logger().info(f"Golden cube detected at ({stable_center[0]:.1f}, {stable_center[1]:.1f}) with size {stable_size:.1f} and confidence {stable_confidence:.2f}")
            
            # Display the image with detection overlay
            cv2.imshow("Golden Cube Detection", display_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
    
    def stabilize_detection(self, detection, center, size, confidence):
        """Apply temporal smoothing to detection results for stability"""
        
        # Add current detection to history
        if detection:
            self.last_detections.append((True, center, size, confidence))
            self.detection_counter += 1
            self.no_detection_counter = 0
            self.last_valid_detection = (center, size, confidence)
        else:
            self.last_detections.append((False, (0, 0), 0, 0))
            self.no_detection_counter += 1
            self.detection_counter = 0
        
        # Only consider detection valid if we have multiple consecutive detections
        if len(self.last_detections) < 3:
            return detection, center, size, confidence
        
        # Count recent detections
        recent_detections = sum(1 for d in self.last_detections if d[0])
        
        # Require at least half of recent frames to have detections
        if recent_detections < len(self.last_detections) / 2:
            return False, (0, 0), 0, 0
        
        # If we have enough detections, compute average of valid values
        valid_centers_x = [d[1][0] for d in self.last_detections if d[0]]
        valid_centers_y = [d[1][1] for d in self.last_detections if d[0]]
        valid_sizes = [d[2] for d in self.last_detections if d[0]]
        valid_confidences = [d[3] for d in self.last_detections if d[0]]
        
        if not valid_centers_x:  # No valid detections in window
            return False, (0, 0), 0, 0
        
        # Calculate median values (more robust than mean)
        stable_center_x = np.median(valid_centers_x)
        stable_center_y = np.median(valid_centers_y)
        stable_size = np.median(valid_sizes)
        stable_confidence = np.median(valid_confidences)
        
        return True, (stable_center_x, stable_center_y), stable_size, stable_confidence
    
    def detect_golden_cube(self, image, display_image):
        """
        Process the image to detect golden cubes
        Returns: (detection_flag, center_point, size, confidence)
        """
        # Convert to HSV for better color filtering
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for golden color
        lower_gold = np.array([self.hue_min, self.sat_min, self.val_min])
        upper_gold = np.array([self.hue_max, self.sat_max, self.val_max])
        mask = cv2.inRange(hsv_image, lower_gold, upper_gold)
        
        # Apply morphological operations to remove noise and fill holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Display the mask for debugging
        cv2.imshow("Gold Mask", mask)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize return values
        cube_detected = False
        cube_center = (0, 0)
        cube_size = 0
        confidence = 0.0
        
        # Process contours to find potential cubes
        if contours:
            # Find the largest contour (assumption: the cube is the largest golden object)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Only process if the contour is large enough (to filter out noise)
            if area > self.min_detection_area:
                # Get rotated rectangle for better orientation handling
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Get the center, width, and height
                (center_x, center_y), (width, height), angle = rect
                
                # Get bounding rectangle (for visualization)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Calculate aspect ratio to check if it's roughly square (cube face)
                aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 999
                
                # Calculate solidity (area of contour / area of convex hull)
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Approximate the contour to check if it's a rectangle (cube face)
                epsilon = 0.04 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Calculate confidence based on multiple factors
                # 1. Aspect ratio (closer to 1 is better)
                ar_confidence = 1.0 - min(abs(1.0 - aspect_ratio), 1.0)
                
                # 2. Shape approximation (4 corners for a rectangle/square)
                vertex_diff = abs(len(approx) - 4)
                shape_confidence = max(0.0, 1.0 - (vertex_diff / 4.0))
                
                # 3. Solidity (higher is better, indicates how "solid" or filled the shape is)
                solidity_confidence = min(1.0, solidity * 1.25)  # Scale to give more weight
                
                # Combined confidence
                confidence = (0.4 * ar_confidence + 0.4 * shape_confidence + 0.2 * solidity_confidence)
                
                # Calculate size consistently (average of width and height)
                cube_size = (width + height) / 2
                
                # Store detection info
                cube_center = (center_x, center_y)
                
                # If confidence is high enough, consider it a cube
                if confidence > self.confidence_threshold:
                    cube_detected = True
                    
                    # Draw raw detection results on the display image (in green)
                    cv2.drawContours(display_image, [box], 0, (0, 255, 0), 2)
                    cv2.circle(display_image, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
                    cv2.putText(display_image, f"Raw: {confidence:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Debug information
                    cv2.putText(display_image, f"AR: {aspect_ratio:.2f}", (x, y + h + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(display_image, f"Corners: {len(approx)}", (x, y + h + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(display_image, f"Solidity: {solidity:.2f}", (x, y + h + 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    # Draw rejected contour
                    cv2.drawContours(display_image, [box], 0, (0, 165, 255), 2)
                    cv2.putText(display_image, f"Rejected: {confidence:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        return cube_detected, cube_center, cube_size, confidence

def main(args=None):
    rclpy.init(args=args)
    detector_node = GoldenCubeDetectorNode()
    try:
        rclpy.spin(detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        cv2.destroyAllWindows()
        detector_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()  
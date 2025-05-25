import cv2
import rclpy
import os
import numpy as np
import math
from cv_bridge import CvBridge
from message_filters import Subscriber
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
import traceback # Needed for detailed error logging

class CubeDetectionNode(Node):
    def __init__(self):
        super().__init__("cube_detection_node")
        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID')

        if not self.robot_id_str:
            self.get_logger().warning(
                "ROS_DOMAIN_ID environment variable not set or empty. Using default '0'."
            )
            self.robot_id_str = "0" 
        
        self.declare_parameter(
            name="image_sub_topic",
            value=f"/T{self.robot_id_str}/oakd/rgb/image_raw/compressed",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
            ),
        )
        
        self.declare_parameter("hue_min", 16)
        self.declare_parameter("hue_max", 39)
        self.declare_parameter("sat_min", 119)
        self.declare_parameter("sat_max", 240)
        self.declare_parameter("val_min", 123)
        self.declare_parameter("val_max", 202)
        
        self.declare_parameter("min_contour_area", 500)
        self.declare_parameter("max_contour_area", 30000)
        self.declare_parameter("epsilon_factor", 0.02) # Key parameter for approxPolyDP
        
        self.declare_parameter("temporal_buffer_size", 4)
        self.declare_parameter("min_consistent_detections", 2)
        self.declare_parameter("confidence_threshold", 40.0) 
        
        image_sub_topic = (
            self.get_parameter("image_sub_topic").get_parameter_value().string_value
        )
        
        self.get_logger().info(f"Subscribing to image topic: {image_sub_topic}")
        
        self.image_sub = Subscriber(
            self, CompressedImage, image_sub_topic, qos_profile=qos_profile_sensor_data
        )
        
        self.cv_bridge = CvBridge()
        self.image_sub.registerCallback(self.image_callback)
        
        self.detection_history = [] 
        self.max_history = 5        

        self.detection_buffer = []
        self.buffer_size = self.get_parameter("temporal_buffer_size").get_parameter_value().integer_value
        self.min_detections_for_consistency_val = self.get_parameter("min_consistent_detections").get_parameter_value().integer_value
        
        self.last_valid_detection = None 
        
        self.get_logger().info("Cube detection node initialized and waiting for images...")

    def get_color_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        h_min = self.get_parameter("hue_min").get_parameter_value().integer_value
        h_max = self.get_parameter("hue_max").get_parameter_value().integer_value
        s_min = self.get_parameter("sat_min").get_parameter_value().integer_value
        s_max = self.get_parameter("sat_max").get_parameter_value().integer_value
        v_min = self.get_parameter("val_min").get_parameter_value().integer_value
        v_max = self.get_parameter("val_max").get_parameter_value().integer_value
        
        lower_golden = np.array([h_min, s_min, v_min])
        upper_golden = np.array([h_max, s_max, v_max])
        
        mask = cv2.inRange(hsv, lower_golden, upper_golden)
        
        # Consider making kernel size and iterations parameters if tuning is needed often
        kernel = np.ones((3, 3), np.uint8) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return mask

    def is_square_like(self, contour):
        epsilon_factor = self.get_parameter("epsilon_factor").get_parameter_value().double_value
        arc_length = cv2.arcLength(contour, True)

        if arc_length < 1e-3: # Contour too small
            return False, None 

        epsilon = epsilon_factor * arc_length
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # CRITICAL CHECK: The rest of this function assumes a 4-vertex polygon.
        if not (approx is not None and len(approx) == 4):
            # Enable for debugging if too many shapes are rejected:
            # if approx is not None:
            #     self.get_logger().debug(f"Shape rejected: Has {len(approx)} points after approxPolyDP, expected 4.")
            # else:
            #     self.get_logger().debug(f"Shape rejected: approx is None after approxPolyDP.")
            return False, approx 

        # --- From this point on, we know len(approx) == 4 ---

        if not cv2.isContourConvex(approx):
            # self.get_logger().debug("Shape rejected: Not convex.")
            return False, approx
        
        sides = []
        for i in range(4): # This loop is now safe
            p1 = approx[i][0].astype(float) # Ensure points are float for calculations
            p2 = approx[(i + 1) % 4][0].astype(float)
            side_length = np.sqrt(np.sum((p1 - p2)**2))
            if side_length < 1e-3: # Degenerate side
                # self.get_logger().debug("Shape rejected: Degenerate side.")
                return False, approx
            sides.append(side_length)
        
        avg_side = np.mean(sides)
        if avg_side < 1e-3: # All sides are tiny
            # self.get_logger().debug("Shape rejected: Average side too small.")
            return False, approx
            
        side_tolerance = 0.40 # 40% tolerance for side variation
        for side in sides:
            if abs(side - avg_side) / avg_side > side_tolerance:
                # self.get_logger().debug(f"Shape rejected: Side variation too large. Side: {side}, Avg: {avg_side}")
                return False, approx
        
        angles = []
        angle_tolerance_degrees = 25.0 # Degrees
        for i in range(4): # This loop is now safe
            p_current = approx[i][0].astype(float)
            p_prev = approx[(i - 1 + 4) % 4][0].astype(float) 
            p_next = approx[(i + 1) % 4][0].astype(float)
            
            v1 = p_prev - p_current 
            v2 = p_next - p_current 
            
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 < 1e-3 or norm_v2 < 1e-3: # Degenerate vectors for angle calculation
                # self.get_logger().debug("Shape rejected: Degenerate vector for angle calc.")
                return False, approx

            cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
            # Clip to handle potential floating point inaccuracies leading to values slightly outside [-1, 1]
            cos_theta = np.clip(cos_theta, -1.0, 1.0) 
            
            angle = math.degrees(math.acos(cos_theta))
            angles.append(angle)
        
        for angle_val in angles:
            # Check if angle is around 90 degrees
            if not ( (90.0 - angle_tolerance_degrees) <= angle_val <= (90.0 + angle_tolerance_degrees) ):
                # self.get_logger().debug(f"Shape rejected: Angle {angle_val} not close to 90.")
                return False, approx
        
        # If all checks pass
        # self.get_logger().debug("Shape accepted as square-like.")
        return True, approx

    def detect_cubes(self, image):
        color_mask = self.get_color_mask(image)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cube_candidates = []
        min_area = self.get_parameter("min_contour_area").get_parameter_value().integer_value
        max_area = self.get_parameter("max_contour_area").get_parameter_value().integer_value
        confidence_thresh = self.get_parameter("confidence_threshold").get_parameter_value().double_value

        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < min_area or area > max_area:
                continue
            
            is_square, approx_shape = self.is_square_like(contour)
            
            # approx_shape could be None if arc_length was too small, or it could be a non-4-sided polygon
            # if is_square is False. We only proceed if is_square is True.
            if is_square: 
                # At this point, approx_shape is guaranteed to have 4 vertices due to checks in is_square_like
                M = cv2.moments(contour) # Use original contour for moments, not approx_shape
                if M["m00"] > 1e-5: # Avoid division by zero for center calculation
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour) # Bbox from original contour
                    aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0 
                    
                    # Aspect ratio check (can be tuned or moved into is_square_like if preferred)
                    # For now, keeping it separate as it's on the bounding box, not the approx shape sides.
                    if 0.7 <= aspect_ratio <= 1.4: 
                        confidence = self.calculate_confidence(contour, area, aspect_ratio, approx_shape)
                        if confidence >= confidence_thresh:
                            cube_candidates.append({
                                'contour': contour,
                                'approx': approx_shape, # This is the 4-sided approximation
                                'center': (cx, cy),
                                'area': area,
                                'bbox': (x_rect, y_rect, w_rect, h_rect),
                                'aspect_ratio': aspect_ratio,
                                'confidence': confidence
                            })
        
        return cube_candidates

    def calculate_confidence(self, contour, area, aspect_ratio, approx):
        # approx is guaranteed to be 4-sided here if called from detect_cubes path
        confidence = 100.0
        
        aspect_penalty = abs(aspect_ratio - 1.0) * 50 # Penalty based on bounding box aspect ratio
        confidence -= aspect_penalty
        
        if area < 800: 
            confidence -= (800 - area) / 20 
        elif area > 20000: 
            confidence -= (area - 20000) / 200
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 1e-3: 
            approx_perimeter = cv2.arcLength(approx, True) # Perimeter of the 4-sided approx
            fit_penalty = abs(perimeter - approx_perimeter) / perimeter * 30
            confidence -= fit_penalty
        
        if 1000 <= area <= 5000: 
            confidence += 10
        
        return max(0.0, min(100.0, confidence))

    def draw_detections(self, image, cube_candidates):
        result_image = image.copy()
        
        for i, cube in enumerate(cube_candidates):
            cv2.drawContours(result_image, [cube['contour']], -1, (0, 255, 0), 2)
            cv2.drawContours(result_image, [cube['approx']], -1, (255, 0, 0), 2) # approx is 4-sided
            cv2.circle(result_image, cube['center'], 5, (0, 0, 255), -1)
            
            x, y, w, h = cube['bbox']
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            
            confidence = cube.get('confidence', 0.0) 
            text = f"C{i+1} A:{int(cube['area'])} C:{confidence:.1f}"
            cv2.putText(result_image, text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1) 
        
        status_text = f"Buf:{len(self.detection_buffer)}/{self.buffer_size} MinConsist:{self.min_detections_for_consistency_val} Det:{len(cube_candidates)}"
        cv2.putText(result_image, status_text, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return result_image

    def smooth_detections(self, current_detections):
        self.detection_buffer.append(len(current_detections))
        if len(self.detection_buffer) > self.buffer_size:
            self.detection_buffer.pop(0)
        
        if len(self.detection_buffer) < self.min_detections_for_consistency_val:
            self.last_valid_detection = None # Clear last valid if buffer not full enough for consistency
            return [] 

        consistent_detection_frames_in_buffer = sum(1 for count in self.detection_buffer if count > 0)
            
        if consistent_detection_frames_in_buffer >= self.min_detections_for_consistency_val:
            # Enough frames in buffer showed detections, now check current frame
            if current_detections: # If current frame has detections, these are the stable ones
                self.last_valid_detection = current_detections.copy()
                return current_detections
            else: # Consistent history, but current frame is empty. Hold last valid detection.
                if self.last_valid_detection:
                    return self.last_valid_detection 
                else: # No current, no last valid, despite history.
                    return []
        else: # Not enough consistent frames in the buffer recently
            self.last_valid_detection = None
            return []


    def are_detections_similar(self, det1, det2, max_distance=50): # Unused currently
        if det1 is None or det2 is None:
            return False
        center1 = det1.get('center', (0, 0))
        center2 = det2.get('center', (0, 0))
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance < max_distance

    def image_callback(self, image_msg: CompressedImage):
        try:
            image = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg)
            
            raw_detections = self.detect_cubes(image)
            stable_detections = self.smooth_detections(raw_detections)
            
            if stable_detections:
                self.get_logger().info(f"STABLE: Detected {len(stable_detections)} cube(s)")
            elif raw_detections: # Log if there were raw detections but they weren't stable
                 self.get_logger().info(f"UNSTABLE: {len(raw_detections)} raw candidate(s), 0 stable.")
            # else: No raw, no stable - can be logged if needed but might be noisy
            
            result_image = self.draw_detections(image, stable_detections) # Draw stable ones
            
            # For debugging, you might want to see the raw mask
            # color_mask_display = self.get_color_mask(image)
            # mask_colored = cv2.cvtColor(color_mask_display, cv2.COLOR_GRAY2BGR)
            
            # For debugging raw detections, you could draw them on a separate image or the main one
            # raw_result_image = self.draw_detections(image.copy(), raw_detections) # Draw raw ones
            # combined_debug = np.hstack([result_image, raw_result_image])
            # cv2.imshow("Stable Detections | Raw Detections", combined_debug)
            # Or just show the stable ones with the mask
            
            color_mask_for_display = self.get_color_mask(image)
            mask_colored_for_display = cv2.cvtColor(color_mask_for_display, cv2.COLOR_GRAY2BGR)


            # Resize for display if images are large
            combined_display_scale = 0.6 # Adjusted scale
            final_h, final_w = result_image.shape[:2]
            scaled_w = max(1, int(final_w * combined_display_scale))
            scaled_h = max(1, int(final_h * combined_display_scale))

            display_result = cv2.resize(result_image, (scaled_w, scaled_h))
            display_mask = cv2.resize(mask_colored_for_display, (scaled_w, scaled_h))
            
            combined_display = np.hstack([display_result, display_mask])
            cv2.imshow("Cube Detection (Stable | Color Mask)", combined_display)
            cv2.waitKey(1)
            
            self.detection_history.append(len(stable_detections))
            if len(self.detection_history) > self.max_history:
                self.detection_history.pop(0)
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.get_logger().error(f"Error processing image: {str(e)}\nTraceback:\n{tb_str}")

def main(args=None):
    rclpy.init(args=args)
    # Set logger level for your node if you want to see DEBUG messages from is_square_like
    # rclpy.logging.set_logger_level("cube_detection_node", rclpy.logging.LoggingSeverity.DEBUG)
    cube_detector = CubeDetectionNode()
    
    try:
        rclpy.spin(cube_detector)
    except KeyboardInterrupt:
        cube_detector.get_logger().info("Keyboard interrupt, shutting down...")
    finally:
        cv2.destroyAllWindows()
        if cube_detector.get_node_names(): # Check if node is still valid
             cube_detector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
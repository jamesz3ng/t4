import cv2
import rclpy
import os
import numpy as np
import math
from cv_bridge import CvBridge, CvBridgeError # Added CvBridgeError
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo, Image # Added Image for publishing
import traceback

class CubeDetectionNode(Node):
    def __init__(self):
        super().__init__("cube_detection_node")
        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID')

        if not self.robot_id_str:
            self.get_logger().warning(
                "ROS_DOMAIN_ID environment variable not set or empty. Using default '0'."
            )
            self.robot_id_str = "0"
        
        # --- Parameters ---
        self.declare_parameter(
            name="image_sub_topic",
            value=f"/T{self.robot_id_str}/oakd/rgb/image_raw/compressed",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter(
            name="camera_info_sub_topic",
            value=f"/T{self.robot_id_str}/oakd/rgb/camera_info",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter("cube_physical_width_m", 0.08, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        
        # HSV
        self.declare_parameter("hue_min", 16, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("hue_max", 39, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("sat_min", 119, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("sat_max", 240, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("val_min", 123, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("val_max", 202, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        
        # Contour
        self.declare_parameter("min_contour_area", 500, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("max_contour_area", 30000, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("epsilon_factor", 0.02, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        
        # Temporal Smoothing
        self.declare_parameter("temporal_buffer_size", 4, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("min_consistent_detections", 2, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("confidence_threshold", 35.0, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))

        # Debug Display Parameters
        self.declare_parameter("publish_debug_image", True, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter("use_cv_imshow_debug", True, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter("debug_display_every_n_frames", 5, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))

        # --- Cached Parameters (will be loaded in image_callback or relevant methods) ---
        # This section is for clarity; actual loading happens dynamically.

        # --- Subscriptions ---
        image_sub_topic = self.get_parameter("image_sub_topic").get_parameter_value().string_value
        camera_info_sub_topic = self.get_parameter("camera_info_sub_topic").get_parameter_value().string_value
        
        self.get_logger().info(f"Subscribing to image topic: {image_sub_topic}")
        self.get_logger().info(f"Subscribing to camera info topic: {camera_info_sub_topic}")
        
        self.image_sub = self.create_subscription(
            CompressedImage, image_sub_topic, self.image_callback, qos_profile_sensor_data
        )
        
        self.camera_info_received = False
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        camera_info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_sub_topic, self.camera_info_callback, camera_info_qos
        )
        
        # --- Publishers ---
        self.debug_image_pub = self.create_publisher(Image, "~/debug_image/processed", 10)
        self.debug_mask_pub = self.create_publisher(Image, "~/debug_image/mask", 10)

        # --- Other Member Variables ---
        self.cv_bridge = CvBridge()
        self.detection_history = [] 
        self.max_history = 5        
        self.detection_buffer = []
        self.last_valid_detection = None
        self.frame_count_for_display = 0
        
        self.get_logger().info("Cube detection node initialized. Waiting for camera info and images...")

    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_info_received = True
            self.get_logger().info(
                f"Camera intrinsics received: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}"
            )

    def get_color_mask(self, image, h_min, h_max, s_min, s_max, v_min, v_max):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def is_square_like(self, contour, epsilon_factor_val):
        arc_length = cv2.arcLength(contour, True)
        if arc_length < 1e-3: return False, None
        epsilon = epsilon_factor_val * arc_length
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if not (approx is not None and len(approx) == 4): return False, approx
        if not cv2.isContourConvex(approx): return False, approx
        
        sides = [np.sqrt(np.sum((approx[i][0] - approx[(i + 1) % 4][0])**2)) for i in range(4)]
        if any(s < 1e-3 for s in sides): return False, approx
        avg_side = np.mean(sides)
        if avg_side < 1e-3: return False, approx
        side_tolerance = 0.40
        if any(abs(s - avg_side) / avg_side > side_tolerance for s in sides): return False, approx
        
        angles = []
        angle_tolerance_degrees = 25.0
        for i in range(4):
            p_curr = approx[i][0].astype(float)
            p_prev = approx[(i - 1 + 4) % 4][0].astype(float)
            p_next = approx[(i + 1) % 4][0].astype(float)
            v1, v2 = p_prev - p_curr, p_next - p_curr
            norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm_v1 < 1e-3 or norm_v2 < 1e-3: return False, approx
            cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
            angles.append(math.degrees(math.acos(cos_theta)))
        
        if any(not ((90.0 - angle_tolerance_degrees) <= angle <= (90.0 + angle_tolerance_degrees)) for angle in angles):
            return False, approx
        return True, approx

    def detect_cubes(self, image, hsv_params, contour_params, confidence_thresh_val, epsilon_factor_val):
        # Unpack parameters
        h_min, h_max, s_min, s_max, v_min, v_max = hsv_params
        min_area, max_area = contour_params

        color_mask = self.get_color_mask(image, h_min, h_max, s_min, s_max, v_min, v_max)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cube_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (min_area <= area <= max_area):
                continue
            
            is_square, approx_shape = self.is_square_like(contour, epsilon_factor_val)
            if is_square:
                M = cv2.moments(contour)
                if M["m00"] > 1e-5:
                    cx_pixel = int(M["m10"] / M["m00"])
                    cy_pixel = int(M["m01"] / M["m00"])
                    x_r, y_r, w_r, h_r = cv2.boundingRect(contour)
                    aspect_ratio = float(w_r) / h_r if h_r > 0 else 0
                    
                    if 0.7 <= aspect_ratio <= 1.4:
                        confidence = self.calculate_confidence(contour, area, aspect_ratio, approx_shape)
                        if confidence >= confidence_thresh_val:
                            cube_candidates.append({
                                'contour': contour, 'approx': approx_shape, 'center': (cx_pixel, cy_pixel),
                                'area': area, 'bbox': (x_r, y_r, w_r, h_r),
                                'aspect_ratio': aspect_ratio, 'confidence': confidence
                            })
        return cube_candidates, color_mask # Return mask for debug display

    def calculate_confidence(self, contour, area, aspect_ratio, approx):
        confidence = 100.0
        confidence -= abs(aspect_ratio - 1.0) * 50
        if area < 800: confidence -= (800 - area) / 20
        elif area > 20000: confidence -= (area - 20000) / 200
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 1e-3:
            approx_perimeter = cv2.arcLength(approx, True)
            confidence -= abs(perimeter - approx_perimeter) / perimeter * 30
        if 1000 <= area <= 5000: confidence += 10
        return max(0.0, min(100.0, confidence))

    def draw_detections(self, image, cube_candidates):
        result_image = image.copy()
        for i, cube in enumerate(cube_candidates):
            cv2.drawContours(result_image, [cube['contour']], -1, (0, 255, 0), 2)
            cv2.drawContours(result_image, [cube['approx']], -1, (255, 0, 0), 2)
            cv2.circle(result_image, cube['center'], 5, (0, 0, 255), -1)
            x, y, w, h = cube['bbox']
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            text = f"C{i+1} A:{int(cube['area'])} Cf:{cube.get('confidence', 0.0):.1f}"
            cv2.putText(result_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if 'position_in_camera' in cube:
                pos_cam = cube['position_in_camera']
                pos_text = f"D: {pos_cam[2]:.2f}m"
                cv2.putText(result_image, pos_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        return result_image

    def smooth_detections(self, current_detections, buffer_size_val, min_detections_for_consistency_val):
        self.detection_buffer.append(len(current_detections))
        if len(self.detection_buffer) > buffer_size_val:
            self.detection_buffer.pop(0)
        
        if len(self.detection_buffer) < min_detections_for_consistency_val:
            self.last_valid_detection = None
            return []

        consistent_detection_frames_in_buffer = sum(1 for count in self.detection_buffer if count > 0)
        if consistent_detection_frames_in_buffer >= min_detections_for_consistency_val:
            if current_detections:
                self.last_valid_detection = current_detections.copy()
                return current_detections
            elif self.last_valid_detection:
                return self.last_valid_detection
            else:
                return []
        else:
            self.last_valid_detection = None
            return []

    def image_callback(self, image_msg: CompressedImage):
        # --- Parameter Caching for this callback execution ---
        # HSV
        h_min_val = self.get_parameter("hue_min").get_parameter_value().integer_value
        h_max_val = self.get_parameter("hue_max").get_parameter_value().integer_value
        s_min_val = self.get_parameter("sat_min").get_parameter_value().integer_value
        s_max_val = self.get_parameter("sat_max").get_parameter_value().integer_value
        v_min_val = self.get_parameter("val_min").get_parameter_value().integer_value
        v_max_val = self.get_parameter("val_max").get_parameter_value().integer_value
        hsv_params = (h_min_val, h_max_val, s_min_val, s_max_val, v_min_val, v_max_val)

        # Contour
        min_contour_area_val = self.get_parameter("min_contour_area").get_parameter_value().integer_value
        max_contour_area_val = self.get_parameter("max_contour_area").get_parameter_value().integer_value
        contour_params = (min_contour_area_val, max_contour_area_val)
        epsilon_factor_val = self.get_parameter("epsilon_factor").get_parameter_value().double_value
        
        # Smoothing & Confidence
        buffer_size_val = self.get_parameter("temporal_buffer_size").get_parameter_value().integer_value
        min_consistent_detections_val = self.get_parameter("min_consistent_detections").get_parameter_value().integer_value
        confidence_threshold_val = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        
        # Distance
        W_physical_val = self.get_parameter("cube_physical_width_m").get_parameter_value().double_value

        # Debug Display
        publish_debug_image_val = self.get_parameter("publish_debug_image").get_parameter_value().bool_value
        use_cv_imshow_debug_val = self.get_parameter("use_cv_imshow_debug").get_parameter_value().bool_value
        debug_display_every_n_frames_val = self.get_parameter("debug_display_every_n_frames").get_parameter_value().integer_value
        # --- End Parameter Caching ---

        if not self.camera_info_received:
            self.get_logger().warn("Camera info not yet received. Skipping image processing.")
            return
            
        try:
            image = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg)
            
            raw_detections, color_mask_for_debug = self.detect_cubes(
                image, hsv_params, contour_params, confidence_threshold_val, epsilon_factor_val
            )
            stable_detections = self.smooth_detections(
                raw_detections, buffer_size_val, min_consistent_detections_val
            )
            
            if stable_detections:
                for i in range(len(stable_detections)):
                    cube_data = stable_detections[i]
                    P_pixels = float(cube_data['bbox'][2])
                    if self.fx is None or self.fx < 1e-3 or P_pixels < 1.0: continue
                    
                    Z_camera = (W_physical_val * self.fx) / P_pixels
                    u_pixel, v_pixel = cube_data['center']
                    X_camera = (u_pixel - self.cx) * Z_camera / self.fx
                    Y_camera = (v_pixel - self.cy) * Z_camera / self.fy
                    stable_detections[i]['position_in_camera'] = (X_camera, Y_camera, Z_camera)
                    
                    self.get_logger().info(
                        f"STABLE Cube {i+1} at camera (X,Y,Z): ({X_camera:.2f}m, {Y_camera:.2f}m, {Z_camera:.2f}m). "
                        f"P={P_pixels:.1f}px, Center=({u_pixel},{v_pixel})"
                    )
            elif raw_detections:
                 self.get_logger().info(f"UNSTABLE: {len(raw_detections)} raw candidate(s), 0 stable.")
            
            # --- Debug Display Logic ---
            self.frame_count_for_display += 1
            display_this_frame = (self.frame_count_for_display % debug_display_every_n_frames_val == 0)

            if display_this_frame:
                result_image_for_display = self.draw_detections(image, stable_detections)
                
                if publish_debug_image_val:
                    try:
                        self.debug_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(result_image_for_display, "bgr8"))
                        if color_mask_for_debug is not None:
                             mask_colored_for_display = cv2.cvtColor(color_mask_for_debug, cv2.COLOR_GRAY2BGR)
                             self.debug_mask_pub.publish(self.cv_bridge.cv2_to_imgmsg(mask_colored_for_display, "bgr8"))
                    except CvBridgeError as e:
                        self.get_logger().error(f"CvBridge Error for debug publishing: {e}")
                
                if use_cv_imshow_debug_val:
                    # This part will still be somewhat slow if enabled.
                    mask_colored = cv2.cvtColor(color_mask_for_debug, cv2.COLOR_GRAY2BGR)
                    combined_display_scale = 0.6
                    h_orig, w_orig = result_image_for_display.shape[:2]
                    scaled_w = max(1, int(w_orig * combined_display_scale))
                    scaled_h = max(1, int(h_orig * combined_display_scale))
                    display_result_scaled = cv2.resize(result_image_for_display, (scaled_w, scaled_h))
                    display_mask_scaled = cv2.resize(mask_colored, (scaled_w, scaled_h))
                    combined_cv_display = np.hstack([display_result_scaled, display_mask_scaled])
                    cv2.imshow("Cube Detection (Stable | Color Mask) - CV_IMSHOW", combined_cv_display)
                    cv2.waitKey(1)
            # --- End Debug Display Logic ---
            
            self.detection_history.append(len(stable_detections))
            if len(self.detection_history) > self.max_history: self.detection_history.pop(0)
            
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error in image_callback: {e}")
        except Exception as e:
            tb_str = traceback.format_exc()
            self.get_logger().error(f"Error processing image: {str(e)}\nTraceback:\n{tb_str}")

def main(args=None):
    rclpy.init(args=args)
    cube_detector = CubeDetectionNode()
    try:
        rclpy.spin(cube_detector)
    except KeyboardInterrupt:
        cube_detector.get_logger().info("Keyboard interrupt, shutting down...")
    finally:
        if cube_detector.get_parameter("use_cv_imshow_debug").get_parameter_value().bool_value:
            cv2.destroyAllWindows() # Only if cv.imshow was used
        if rclpy.ok() and cube_detector.get_node_names():
             cube_detector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
import cv2
import rclpy
import os
import numpy as np
import math
from cv_bridge import CvBridge, CvBridgeError
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from visualization_msgs.msg import Marker  # For RViz markers
from geometry_msgs.msg import PointStamped  # For TF transformations
import tf2_ros  # TF2 specific imports
import tf2_geometry_msgs  # ← ADD THIS LINE! This registers PointStamped transforms
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import traceback

class CubeDetectionNode(Node):
    def __init__(self):
        super().__init__("cube_detection_node")
        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID')

        if not self.robot_id_str:
            self.get_logger().warning("ROS_DOMAIN_ID not set, using default '0'.")
            self.robot_id_str = "0"

        # --- Parameters ---
        self.declare_parameter(
            "image_sub_topic",
            f"/T{self.robot_id_str}/oakd/rgb/image_raw/compressed",
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter(
            "camera_info_sub_topic",
            f"/T{self.robot_id_str}/oakd/rgb/camera_info",
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.declare_parameter(
            "cube_physical_width_m",
            0.25,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        )

        # HSV
        self.declare_parameter("hue_min", 15, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("hue_max", 39, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("sat_min", 90, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("sat_max", 211, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("val_min", 123, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("val_max", 255, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))

        # Contour
        self.declare_parameter("min_contour_area", 500, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("max_contour_area", 30000, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("epsilon_factor", 0.02, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))

        # Temporal Smoothing
        self.declare_parameter("temporal_buffer_size", 4, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("min_consistent_detections", 2, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("confidence_threshold", 30.0, descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))

        # Debug Display Parameters
        self.declare_parameter("publish_debug_image", True, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter("use_cv_imshow_debug", True, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter("debug_display_every_n_frames", 5, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))

        # TF and Marker Parameters
        # IMPORTANT: Set this default to the actual TF frame name you identified
        self.declare_parameter(
            "camera_optical_frame_id",
            "oakd_rgb_camera_optical_frame",  # Adjusted based on /tf_static findings
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="TF frame ID of the camera's optical center."
            )
        )
        self.declare_parameter(
            "target_map_frame_id",
            "base_link",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Target TF frame for cube position (e.g., 'map')."
            )
        )
        self.declare_parameter("publish_rviz_marker", True, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))

        # --- Subscriptions ---
        image_sub_topic = self.get_parameter("image_sub_topic").get_parameter_value().string_value
        camera_info_sub_topic = self.get_parameter("camera_info_sub_topic").get_parameter_value().string_value

        self.get_logger().info(f"Subscribing to image topic: {image_sub_topic}")
        self.get_logger().info(f"Subscribing to camera info topic: {camera_info_sub_topic}")

        self.image_sub = self.create_subscription(
            CompressedImage,
            image_sub_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.camera_info_received = False
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        camera_info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_sub_topic,
            self.camera_info_callback,
            camera_info_qos
        )

        # --- Publishers ---
        self.debug_image_pub = self.create_publisher(Image, "~/debug_image/processed", 10)
        self.debug_mask_pub = self.create_publisher(Image, "~/debug_image/mask", 10)
        self.cube_marker_pub = self.create_publisher(Marker, "~/cube_marker", 10)

        # --- TF2 ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Other Member Variables ---
        self.cv_bridge = CvBridge()
        self.detection_history = []
        self.max_history = 5
        self.detection_buffer = []
        self.last_valid_detection = None
        self.frame_count_for_display = 0

        self.get_logger().info("Cube detection node initialized. Waiting for camera info and images...")

        # Add these variables for persistent marker tracking
        self.last_published_marker_position = None
        self.marker_position_threshold = 0.05  # 5cm threshold for republishing
        self.published_marker_ids = set()  # Track published markers
        self.next_marker_id = 0

    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.fx = msg.k[0]  # Focal length in x
            self.fy = msg.k[4]  # Focal length in y
            self.cx = msg.k[2]  # Principal point x
            self.cy = msg.k[5]  # Principal point y
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
        if arc_length < 1e-3:
            return False, None
        epsilon = epsilon_factor_val * arc_length
        approx = cv2.approxPolyDP(contour, epsilon, True)

        sides = [np.sqrt(np.sum((approx[i][0] - approx[(i + 1) % 4][0])**2)) for i in range(4)]
        if any(s < 1e-3 for s in sides):
            return False, approx
        avg_side = np.mean(sides)
        if avg_side < 1e-3:
            return False, approx

        side_tolerance = 0.60
        if any(abs(s - avg_side) / avg_side > side_tolerance for s in sides):
            return False, approx

        angles = []
        angle_tolerance_degrees = 25.0
        for i in range(4):
            p_curr = approx[i][0].astype(float)
            p_prev = approx[(i - 1 + 4) % 4][0].astype(float)
            p_next = approx[(i + 1) % 4][0].astype(float)
            v1, v2 = p_prev - p_curr, p_next - p_curr
            norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm_v1 < 1e-3 or norm_v2 < 1e-3:
                return False, approx
            cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
            angles.append(math.degrees(math.acos(cos_theta)))

        if any(not ((90.0 - angle_tolerance_degrees) <= angle <= (90.0 + angle_tolerance_degrees)) for angle in angles):
            return False, approx
        return True, approx

    def detect_cubes(self, image, hsv_params, contour_params, confidence_thresh_val, epsilon_factor_val):
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
                                'contour': contour,
                                'approx': approx_shape,
                                'center': (cx_pixel, cy_pixel),
                                'area': area,
                                'bbox': (x_r, y_r, w_r, h_r),
                                'aspect_ratio': aspect_ratio,
                                'confidence': confidence
                            })
        return cube_candidates, color_mask

    def calculate_confidence(self, contour, area, aspect_ratio, approx):
        confidence = 100.0
        confidence -= abs(aspect_ratio - 1.0) * 50
        if area < 800:
            confidence -= (800 - area) / 20
        elif area > 20000:
            confidence -= (area - 20000) / 200
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 1e-3:
            approx_perimeter = cv2.arcLength(approx, True)
            confidence -= abs(perimeter - approx_perimeter) / perimeter * 30
        
        if 1000 <= area <= 5000:
            confidence += 10
        return max(0.0, min(100.0, confidence))

    def draw_detections(self, image, cube_candidates):
        result_image = image.copy()
        for i, cube in enumerate(cube_candidates):
            cv2.drawContours(result_image, [cube['contour']], -1, (0, 255, 0), 2)
            cv2.circle(result_image, cube['center'], 5, (0, 0, 255), -1)
            x, y, w, h = cube['bbox']
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            text = f"Cf:{cube.get('confidence', 0.0):.0f}"
            cv2.putText(result_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if 'position_in_camera' in cube:
                pos_cam = cube['position_in_camera']
                cam_text = f"Cam D: {pos_cam[2]:.2f}m"
                cv2.putText(result_image, cam_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if 'position_in_map' in cube:
                pos_map = cube['position_in_map']
                map_text = f"Map X:{pos_map.x:.1f} Y:{pos_map.y:.1f}"
                cv2.putText(result_image, map_text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
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
        # Parameter Caching
        hsv_params = (
            self.get_parameter("hue_min").get_parameter_value().integer_value,
            self.get_parameter("hue_max").get_parameter_value().integer_value,
            self.get_parameter("sat_min").get_parameter_value().integer_value,
            self.get_parameter("sat_max").get_parameter_value().integer_value,
            self.get_parameter("val_min").get_parameter_value().integer_value,
            self.get_parameter("val_max").get_parameter_value().integer_value
        )
        contour_params = (
            self.get_parameter("min_contour_area").get_parameter_value().integer_value,
            self.get_parameter("max_contour_area").get_parameter_value().integer_value
        )
        epsilon_factor_val = self.get_parameter("epsilon_factor").get_parameter_value().double_value
        buffer_size_val = self.get_parameter("temporal_buffer_size").get_parameter_value().integer_value
        min_consistent_detections_val = self.get_parameter("min_consistent_detections").get_parameter_value().integer_value
        confidence_threshold_val = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        W_physical_val = self.get_parameter("cube_physical_width_m").get_parameter_value().double_value
        
        publish_debug_image_val = self.get_parameter("publish_debug_image").get_parameter_value().bool_value
        use_cv_imshow_debug_val = self.get_parameter("use_cv_imshow_debug").get_parameter_value().bool_value
        debug_display_every_n_frames_val = self.get_parameter("debug_display_every_n_frames").get_parameter_value().integer_value

        camera_optical_frame_id_val = self.get_parameter("camera_optical_frame_id").get_parameter_value().string_value
        target_map_frame_id_val = self.get_parameter("target_map_frame_id").get_parameter_value().string_value
        publish_rviz_marker_val = self.get_parameter("publish_rviz_marker").get_parameter_value().bool_value

        if not self.camera_info_received:
            self.get_logger().warn("Camera info not yet received. Skipping frame.")
            return

        try:
            image_cv = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg)
            raw_detections, color_mask_for_debug = self.detect_cubes(
                image_cv, hsv_params, contour_params, confidence_threshold_val, epsilon_factor_val
            )
            stable_detections = self.smooth_detections(
                raw_detections, buffer_size_val, min_consistent_detections_val
            )

            processed_stable_detections = []

            if stable_detections:
                # Process the first stable detection (or implement logic for multiple)
                best_detection = stable_detections[0]
                P_pixels = float(best_detection['bbox'][2])

                if self.fx is not None and self.fx >= 1e-3 and P_pixels >= 1.0:
                    Z_camera = (W_physical_val * self.fx) / P_pixels
                    u_pixel, v_pixel = best_detection['center']
                    X_camera = (u_pixel - self.cx) * Z_camera / self.fx
                    Y_camera = (v_pixel - self.cy) * Z_camera / self.fy

                    best_detection['position_in_camera'] = (X_camera, Y_camera, Z_camera)
                    self.get_logger().info(
                        f"Cube at cam_opt_frame: ({X_camera:.2f}, {Y_camera:.2f}, {Z_camera:.2f})m"
                    )

                    # --- TF Transformation ---
                    transform_successful = False
                    try:
                        # Use a slightly longer timeout for the initial can_transform check
                        # to give TF time to populate, but keep lookup_transform potentially shorter.
                        # However, image_msg.header.stamp is crucial for looking up past transforms.
                        # We might need rclpy.time.Time() if image_msg.header.stamp is too old.
                        
                        # Ensure the stamp is not excessively old for can_transform
                        current_time = self.get_clock().now()
                        time_diff = current_time.nanoseconds - rclpy.time.Time.from_msg(image_msg.header.stamp).nanoseconds
                        max_age_ns = 1 * 1e9 # 1 second
                        
                        source_time = image_msg.header.stamp
                        if time_diff > max_age_ns:
                            self.get_logger().warn(f"Image timestamp is old ({time_diff/1e9:.2f}s). Using current time for TF lookup instead.")
                            source_time = rclpy.time.Time().to_msg() # Use current time if image is too old.

                        can_transform_check = self.tf_buffer.can_transform(
                            target_map_frame_id_val,
                            camera_optical_frame_id_val,
                            source_time, # Use potentially adjusted time
                            timeout=rclpy.duration.Duration(seconds=0.2) # Slightly longer for can_transform
                        )

                        if can_transform_check:
                            point_in_camera = PointStamped()
                            point_in_camera.header.stamp = source_time # Use same time as can_transform
                            point_in_camera.header.frame_id = camera_optical_frame_id_val
                            point_in_camera.point.x = X_camera
                            point_in_camera.point.y = Y_camera
                            point_in_camera.point.z = Z_camera

                            point_in_map = self.tf_buffer.transform(
                                point_in_camera,
                                target_map_frame_id_val,
                                timeout=rclpy.duration.Duration(seconds=0.1) # Shorter for actual transform
                            )
                            best_detection['position_in_map'] = point_in_map.point
                            self.get_logger().info(
                                f"Cube at '{target_map_frame_id_val}' frame: "
                                f"(X:{point_in_map.point.x:.2f}, Y:{point_in_map.point.y:.2f}, Z:{point_in_map.point.z:.2f})m"
                            )
                            transform_successful = True
                        else:
                            self.get_logger().warn(
                                f"Cannot transform from '{camera_optical_frame_id_val}' to '{target_map_frame_id_val}'. "
                                f"TF not available or timeout (checked at time {source_time.sec}.{source_time.nanosec:09d})."
                            )
                    except (LookupException, ConnectivityException, ExtrapolationException) as e:
                        self.get_logger().warn(
                            f"TF transform error from '{camera_optical_frame_id_val}' to "
                            f"'{target_map_frame_id_val}': {e}"
                        )

                    # --- RViz Marker Publishing ---
                    if transform_successful and publish_rviz_marker_val:
                        marker = Marker()
                        marker.header.frame_id = target_map_frame_id_val
                        marker.header.stamp = self.get_clock().now().to_msg()
                        marker.ns = "cube_detector"
                        marker.id = 0  # Single, updating marker
                        marker.type = Marker.CUBE
                        marker.action = Marker.ADD

                        marker.pose.position.x = best_detection['position_in_map'].x
                        marker.pose.position.y = best_detection['position_in_map'].y
                        marker.pose.position.z = best_detection['position_in_map'].z # Or fixed height
                        
                        marker.pose.orientation.x = 0.0
                        marker.pose.orientation.y = 0.0
                        marker.pose.orientation.z = 0.0
                        marker.pose.orientation.w = 1.0

                        marker.scale.x = 0.3
                        marker.scale.y = 0.3
                        marker.scale.z = 0.3

                        marker.color.r = 1.0
                        marker.color.g = 0.843
                        marker.color.b = 0.0
                        marker.color.a = 0.7

                        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
                        self.cube_marker_pub.publish(marker)
                    
                    processed_stable_detections.append(best_detection)

            elif raw_detections:
                self.get_logger().info(f"UNSTABLE: {len(raw_detections)} raw candidate(s), 0 stable.")

            # Debug Display Logic
            self.frame_count_for_display += 1
            display_this_frame = (self.frame_count_for_display % debug_display_every_n_frames_val == 0)

            if display_this_frame:
                result_image_for_display = self.draw_detections(image_cv.copy(), processed_stable_detections)
                
                if publish_debug_image_val:
                    try:
                        self.debug_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(result_image_for_display, "bgr8"))
                        if color_mask_for_debug is not None:
                            mask_colored_for_display = cv2.cvtColor(color_mask_for_debug, cv2.COLOR_GRAY2BGR)
                            self.debug_mask_pub.publish(self.cv_bridge.cv2_to_imgmsg(mask_colored_for_display, "bgr8"))
                    except CvBridgeError as e:
                        self.get_logger().error(f"CvBridge Error for debug publishing: {e}")
                
                if use_cv_imshow_debug_val:
                    mask_colored = cv2.cvtColor(color_mask_for_debug, cv2.COLOR_GRAY2BGR)
                    combined_display_scale = 0.6
                    h_orig, w_orig = result_image_for_display.shape[:2]
                    scaled_w = max(1, int(w_orig * combined_display_scale))
                    scaled_h = max(1, int(h_orig * combined_display_scale))
                    display_result_scaled = cv2.resize(result_image_for_display, (scaled_w, scaled_h))
                    display_mask_scaled = cv2.resize(mask_colored, (scaled_w, scaled_h))
                    combined_cv_display = np.hstack([display_result_scaled, display_mask_scaled])
                    cv2.imshow("Cube Detection (CV_IMSHOW)", combined_cv_display)
                    cv2.waitKey(1)
            
            self.detection_history.append(len(processed_stable_detections))
            if len(self.detection_history) > self.max_history:
                self.detection_history.pop(0)

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
        if hasattr(cube_detector, 'get_parameter') and \
           cube_detector.get_node_names() and \
           cube_detector.get_parameter("use_cv_imshow_debug").get_parameter_value().bool_value:
            cv2.destroyAllWindows()
        
        if rclpy.ok() and hasattr(cube_detector, 'get_node_names') and cube_detector.get_node_names():
            cube_detector.destroy_node()
        
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
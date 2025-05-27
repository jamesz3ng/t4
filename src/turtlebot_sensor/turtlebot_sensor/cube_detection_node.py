import cv2
import rclpy
import os
import numpy as np
import math
from cv_bridge import CvBridge, CvBridgeError
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped, PoseStamped # PoseStamped is used for publishing
import tf2_ros
import tf2_geometry_msgs # Registers PointStamped transforms
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import traceback
from collections import deque # For detection_buffer, slightly more efficient pops from left

class CubeDetectionNode(Node):
    def __init__(self):
        super().__init__("cube_detection_node")
        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID')

        if not self.robot_id_str:
            self.get_logger().warning("ROS_DOMAIN_ID not set, using default '0'.")
            self.robot_id_str = "0"

        # --- Cached Parameters ---
        # Image/Camera Topics
        self.param_image_sub_topic: str = ""
        self.param_camera_info_sub_topic: str = ""
        # Cube Properties
        self.param_cube_physical_width_m: float = 0.0
        # HSV
        self.param_hue_min: int = 0
        self.param_hue_max: int = 0
        self.param_sat_min: int = 0
        self.param_sat_max: int = 0
        self.param_val_min: int = 0
        self.param_val_max: int = 0
        self._lower_hsv_bound: np.ndarray = np.array([0,0,0])
        self._upper_hsv_bound: np.ndarray = np.array([255,255,255])
        # Contour
        self.param_min_contour_area: int = 0
        self.param_max_contour_area: int = 0
        self.param_epsilon_factor: float = 0.0
        # Shape Leniency (is_square_like)
        self.param_sq_side_tolerance: float = 0.60 # default, from original code
        self.param_sq_angle_tolerance_deg: float = 35.0 # default, from original code
        # BBox Aspect Ratio (detect_cubes)
        self.param_bbox_aspect_min: float = 0.5 # Made more lenient
        self.param_bbox_aspect_max: float = 2.0 # Made more lenient
        # Temporal Smoothing
        self.param_temporal_buffer_size: int = 0
        self.param_min_consistent_detections: int = 0
        self.param_confidence_threshold: float = 0.0
        # Debug Display
        self.param_publish_debug_image: bool = True
        self.param_use_cv_imshow_debug: bool = True
        self.param_debug_display_every_n_frames: int = 0
        # TF and Marker
        self.param_camera_optical_frame_id: str = ""
        self.param_target_map_frame_id: str = ""
        self.param_publish_rviz_marker: bool = True

        self._declare_and_load_params()
        self.add_on_set_parameters_callback(self.parameters_callback)

        # --- Subscriptions ---
        self.get_logger().info(f"Subscribing to image topic: {self.param_image_sub_topic}")
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.param_image_sub_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.camera_info_received = False
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        # Use TRANSIENT_LOCAL for durability to ensure CameraInfo is received
        # even if published slightly before this node subscribes.
        camera_info_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        self.get_logger().info(f"Subscribing to camera info topic: {self.param_camera_info_sub_topic}")
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.param_camera_info_sub_topic,
            self.camera_info_callback,
            camera_info_qos
        )

        # --- Publishers ---
        self.debug_image_pub = self.create_publisher(Image, "~/debug_image/processed", 10)
        self.debug_mask_pub = self.create_publisher(Image, "~/debug_image/mask", 10)
        self.cube_marker_pub = self.create_publisher(Marker, "~/cube_marker", 10)
        # Publishing format for cube_pose is PoseStamped
        self.cube_pose_pub = self.create_publisher(PoseStamped, f"/T{self.robot_id_str}/cube_pose", 10)


        # --- TF2 ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Other Member Variables ---
        self.cv_bridge = CvBridge()
        self.detection_buffer = deque(maxlen=self.param_temporal_buffer_size)
        self.last_valid_detection_data = None # Store processed data of last valid detection for smoothing
        self.frame_count_for_display = 0
        self.morph_kernel = np.ones((3, 3), np.uint8) # Pre-create kernel for morphology

        self.get_logger().info("Cube detection node initialized. Waiting for camera info and images...")

    def _declare_and_load_params(self):
        self.declare_parameter("image_sub_topic", f"/T{self.robot_id_str}/oakd/rgb/image_raw/compressed", ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("camera_info_sub_topic", f"/T{self.robot_id_str}/oakd/rgb/camera_info", ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("cube_physical_width_m", 0.25, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter("hue_min", 15, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("hue_max", 39, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("sat_min", 90, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("sat_max", 255, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("val_min", 123, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("val_max", 255, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("min_contour_area", 500, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("max_contour_area", 30000, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("epsilon_factor", 0.02, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter("sq_side_tolerance", 0.60, ParameterDescriptor(description="Tolerance for side length variation in square check (0.0-1.0)", type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter("sq_angle_tolerance_deg", 35.0, ParameterDescriptor(description="Tolerance for angle variation in square check (degrees)", type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter("bbox_aspect_min", 0.5, ParameterDescriptor(description="Min bounding box aspect ratio for cube candidates", type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter("bbox_aspect_max", 2.0, ParameterDescriptor(description="Max bounding box aspect ratio for cube candidates", type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter("temporal_buffer_size", 4, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("min_consistent_detections", 2, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("confidence_threshold", 30.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter("publish_debug_image", True, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter("use_cv_imshow_debug", True, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter("debug_display_every_n_frames", 5, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("camera_optical_frame_id", "oakd_rgb_camera_optical_frame", ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("target_map_frame_id", "base_link", ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("publish_rviz_marker", True, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))

        # Load initial values
        self.param_image_sub_topic = self.get_parameter("image_sub_topic").get_parameter_value().string_value
        self.param_camera_info_sub_topic = self.get_parameter("camera_info_sub_topic").get_parameter_value().string_value
        self.param_cube_physical_width_m = self.get_parameter("cube_physical_width_m").get_parameter_value().double_value
        self.param_hue_min = self.get_parameter("hue_min").get_parameter_value().integer_value
        self.param_hue_max = self.get_parameter("hue_max").get_parameter_value().integer_value
        self.param_sat_min = self.get_parameter("sat_min").get_parameter_value().integer_value
        self.param_sat_max = self.get_parameter("sat_max").get_parameter_value().integer_value
        self.param_val_min = self.get_parameter("val_min").get_parameter_value().integer_value
        self.param_val_max = self.get_parameter("val_max").get_parameter_value().integer_value
        self._update_hsv_bounds()
        self.param_min_contour_area = self.get_parameter("min_contour_area").get_parameter_value().integer_value
        self.param_max_contour_area = self.get_parameter("max_contour_area").get_parameter_value().integer_value
        self.param_epsilon_factor = self.get_parameter("epsilon_factor").get_parameter_value().double_value
        self.param_sq_side_tolerance = self.get_parameter("sq_side_tolerance").get_parameter_value().double_value
        self.param_sq_angle_tolerance_deg = self.get_parameter("sq_angle_tolerance_deg").get_parameter_value().double_value
        self.param_bbox_aspect_min = self.get_parameter("bbox_aspect_min").get_parameter_value().double_value
        self.param_bbox_aspect_max = self.get_parameter("bbox_aspect_max").get_parameter_value().double_value
        self.param_temporal_buffer_size = self.get_parameter("temporal_buffer_size").get_parameter_value().integer_value
        self.param_min_consistent_detections = self.get_parameter("min_consistent_detections").get_parameter_value().integer_value
        self.param_confidence_threshold = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        self.param_publish_debug_image = self.get_parameter("publish_debug_image").get_parameter_value().bool_value
        self.param_use_cv_imshow_debug = self.get_parameter("use_cv_imshow_debug").get_parameter_value().bool_value
        self.param_debug_display_every_n_frames = self.get_parameter("debug_display_every_n_frames").get_parameter_value().integer_value
        self.param_camera_optical_frame_id = self.get_parameter("camera_optical_frame_id").get_parameter_value().string_value
        self.param_target_map_frame_id = self.get_parameter("target_map_frame_id").get_parameter_value().string_value
        self.param_publish_rviz_marker = self.get_parameter("publish_rviz_marker").get_parameter_value().bool_value
        
        if hasattr(self, 'detection_buffer') and self.detection_buffer.maxlen != self.param_temporal_buffer_size:
            self.detection_buffer = deque(list(self.detection_buffer), maxlen=self.param_temporal_buffer_size)

    def _update_hsv_bounds(self):
        self._lower_hsv_bound = np.array([self.param_hue_min, self.param_sat_min, self.param_val_min])
        self._upper_hsv_bound = np.array([self.param_hue_max, self.param_sat_max, self.param_val_max])

    def parameters_callback(self, params):
        for param in params:
            if param.name == "image_sub_topic": self.param_image_sub_topic = param.value
            elif param.name == "camera_info_sub_topic": self.param_camera_info_sub_topic = param.value
            elif param.name == "cube_physical_width_m": self.param_cube_physical_width_m = param.value
            elif param.name == "hue_min": self.param_hue_min = param.value
            elif param.name == "hue_max": self.param_hue_max = param.value
            elif param.name == "sat_min": self.param_sat_min = param.value
            elif param.name == "sat_max": self.param_sat_max = param.value
            elif param.name == "val_min": self.param_val_min = param.value
            elif param.name == "val_max": self.param_val_max = param.value
            elif param.name == "min_contour_area": self.param_min_contour_area = param.value
            elif param.name == "max_contour_area": self.param_max_contour_area = param.value
            elif param.name == "epsilon_factor": self.param_epsilon_factor = param.value
            elif param.name == "sq_side_tolerance": self.param_sq_side_tolerance = param.value
            elif param.name == "sq_angle_tolerance_deg": self.param_sq_angle_tolerance_deg = param.value
            elif param.name == "bbox_aspect_min": self.param_bbox_aspect_min = param.value
            elif param.name == "bbox_aspect_max": self.param_bbox_aspect_max = param.value
            elif param.name == "temporal_buffer_size":
                self.param_temporal_buffer_size = param.value
                if self.detection_buffer.maxlen != self.param_temporal_buffer_size:
                    self.detection_buffer = deque(list(self.detection_buffer), maxlen=self.param_temporal_buffer_size)
            elif param.name == "min_consistent_detections": self.param_min_consistent_detections = param.value
            elif param.name == "confidence_threshold": self.param_confidence_threshold = param.value
            elif param.name == "publish_debug_image": self.param_publish_debug_image = param.value
            elif param.name == "use_cv_imshow_debug": self.param_use_cv_imshow_debug = param.value
            elif param.name == "debug_display_every_n_frames": self.param_debug_display_every_n_frames = param.value
            elif param.name == "camera_optical_frame_id": self.param_camera_optical_frame_id = param.value
            elif param.name == "target_map_frame_id": self.param_target_map_frame_id = param.value
            elif param.name == "publish_rviz_marker": self.param_publish_rviz_marker = param.value

        if any(p.name in ["hue_min", "hue_max", "sat_min", "sat_max", "val_min", "val_max"] for p in params):
            self._update_hsv_bounds()

        self.get_logger().info("Parameters updated.")
        return SetParametersResult(successful=True)

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
            # Optional: Unsubscribe if CameraInfo is truly static and QoS is TRANSIENT_LOCAL
            # self.destroy_subscription(self.camera_info_sub)
            # self.camera_info_sub = None

    def get_color_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower_hsv_bound, self._upper_hsv_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=1)
        return mask

    def is_square_like(self, contour):
        arc_length = cv2.arcLength(contour, True)
        if arc_length < 1e-3: return False, None
        epsilon = self.param_epsilon_factor * arc_length
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) != 4: return False, approx

        sides = [np.sqrt(np.sum((approx[i][0] - approx[(i + 1) % 4][0])**2)) for i in range(4)]
        if any(s < 1e-3 for s in sides): return False, approx
        avg_side = np.mean(sides)
        if avg_side < 1e-3: return False, approx
        
        # Use parameterized side tolerance
        if any(abs(s - avg_side) / avg_side > self.param_sq_side_tolerance for s in sides):
            return False, approx

        angles = []
        # Use parameterized angle tolerance
        angle_tolerance_degrees = self.param_sq_angle_tolerance_deg
        for i in range(4):
            p_curr = approx[i][0].astype(float)
            p_prev = approx[(i - 1 + 4) % 4][0].astype(float)
            p_next = approx[(i + 1) % 4][0].astype(float)
            v1, v2 = p_prev - p_curr, p_next - p_curr
            norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm_v1 < 1e-6 or norm_v2 < 1e-6: return False, approx
            cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
            angles.append(math.degrees(math.acos(cos_theta)))

        if any(not ((90.0 - angle_tolerance_degrees) <= angle <= (90.0 + angle_tolerance_degrees)) for angle in angles):
            return False, approx
        return True, approx

    def detect_cubes(self, image):
        color_mask = self.get_color_mask(image)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cube_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.param_min_contour_area <= area <= self.param_max_contour_area):
                continue

            is_square, approx_shape = self.is_square_like(contour)
            if is_square and approx_shape is not None and len(approx_shape) == 4:
                M = cv2.moments(contour)
                if M["m00"] > 1e-5:
                    cx_pixel = int(M["m10"] / M["m00"])
                    cy_pixel = int(M["m01"] / M["m00"])
                    x_r, y_r, w_r, h_r = cv2.boundingRect(contour)
                    aspect_ratio = float(w_r) / h_r if h_r > 0 else 0

                    # Use parameterized and more lenient bounding box aspect ratio check
                    if self.param_bbox_aspect_min <= aspect_ratio <= self.param_bbox_aspect_max:
                        confidence = self.calculate_confidence(contour, area, aspect_ratio, approx_shape)
                        if confidence >= self.param_confidence_threshold:
                            cube_candidates.append({
                                'contour': contour, 'approx': approx_shape, 'center': (cx_pixel, cy_pixel),
                                'area': area, 'bbox': (x_r, y_r, w_r, h_r),
                                'aspect_ratio': aspect_ratio, 'confidence': confidence
                            })
        # Sort by confidence (highest first) or area if multiple candidates are common
        cube_candidates.sort(key=lambda c: c['confidence'], reverse=True)
        return cube_candidates, color_mask

    def calculate_confidence(self, contour, area, aspect_ratio, approx):
        confidence = 100.0
        # Reduced penalty for aspect ratio deviation
        confidence -= abs(aspect_ratio - 1.0) * 30.0 # Was 50.0

        # Area penalties/bonuses (literals could become params if needed)
        # Reduced penalties for area deviations
        if area < 800: confidence -= (800.0 - area) / 40.0 # Was / 20.0
        elif area > 20000: confidence -= (area - 20000.0) / 400.0 # Was / 200.0
        elif 1000 <= area <= 5000: confidence += 10.0 # Bonus for "ideal" size

        perimeter = cv2.arcLength(contour, True)
        if perimeter > 1e-3:
            approx_perimeter = cv2.arcLength(approx, True)
            # Reduced penalty for perimeter difference
            confidence -= abs(perimeter - approx_perimeter) / perimeter * 15.0 # Was 30.0
        return max(0.0, min(100.0, confidence))

    def draw_detections(self, image, cube_candidates_with_info):
        # Draw only the top candidate if that's what's processed, or all stable ones
        # For this example, it draws what's passed in (typically one processed detection)
        for cube in cube_candidates_with_info:
            cv2.drawContours(image, [cube['contour']], -1, (0, 255, 0), 2)
            cv2.circle(image, cube['center'], 5, (0, 0, 255), -1)
            x, y, w, h = cube['bbox']
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            text = f"Cf:{cube['confidence']:.0f}" # Use direct access if confidence is guaranteed
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if 'position_in_camera' in cube:
                pos_cam = cube['position_in_camera']
                cam_text = f"Cam D: {pos_cam[2]:.2f}m"
                cv2.putText(image, cam_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if 'position_in_map' in cube: # position_in_map is a Point, not tuple
                pos_map = cube['position_in_map']
                map_text = f"Map X:{pos_map.x:.1f} Y:{pos_map.y:.1f}"
                cv2.putText(image, map_text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        return image

    def smooth_detections(self, current_raw_detections):
        self.detection_buffer.append(len(current_raw_detections) > 0) # Store True if any detection, False otherwise

        # Not enough data in buffer yet
        if len(self.detection_buffer) < self.param_temporal_buffer_size : # Require full buffer for stability
            self.last_valid_detection_data = None
            return None # Indicate no stable detection yet or buffer not full

        num_positive_detections_in_buffer = sum(self.detection_buffer)

        if num_positive_detections_in_buffer >= self.param_min_consistent_detections:
            if current_raw_detections: # If current frame has detections
                # Process the best raw detection (already sorted by confidence)
                best_current_detection = current_raw_detections[0]
                # (Further processing like 3D pose estimation will happen outside this function)
                self.last_valid_detection_data = best_current_detection # Cache it
                return best_current_detection
            elif self.last_valid_detection_data:
                # Current frame has no detections, but buffer was consistent, so reuse last valid
                # This adds persistence, but check if it still meets confidence criteria if needed
                return self.last_valid_detection_data
            else: # Buffer consistent, but no current and no last valid (edge case)
                self.last_valid_detection_data = None
                return None
        else: # Not enough consistent detections in buffer
            self.last_valid_detection_data = None
            return None


    def image_callback(self, image_msg: CompressedImage):
        if not self.camera_info_received:
            self.get_logger().warn("Camera info not yet received. Skipping frame.", throttle_duration_sec=5.0)
            return

        try:
            image_cv = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            
            raw_detections, color_mask_for_debug = self.detect_cubes(image_cv)
            stable_detection_candidate = self.smooth_detections(raw_detections) # Returns one best candidate or None

            processed_detection_for_publish_and_draw = None # Will hold the single detection to publish/draw

            if stable_detection_candidate:
                # stable_detection_candidate is a dict like {'contour': ..., 'center': ...}
                # Proceed to calculate 3D pose for this stable candidate
                P_pixels = float(stable_detection_candidate['bbox'][2]) # Width of bounding box

                if self.fx is not None and self.fx >= 1e-3 and P_pixels >= 1.0:
                    Z_camera = (self.param_cube_physical_width_m * self.fx) / P_pixels
                    u_pixel, v_pixel = stable_detection_candidate['center']
                    X_camera = (u_pixel - self.cx) * Z_camera / self.fx
                    Y_camera = (v_pixel - self.cy) * Z_camera / self.fy

                    stable_detection_candidate['position_in_camera'] = (X_camera, Y_camera, Z_camera)
                    
                    transform_successful = False
                    point_in_map_coords = None # Store Point geometry_msgs
                    try:
                        current_time = self.get_clock().now() # Use for target time if image stamp is problematic
                        source_time = image_msg.header.stamp
                        
                        # Check if image stamp is reasonable, otherwise use current time for TF
                        time_diff_s = (current_time.nanoseconds - rclpy.time.Time.from_msg(source_time).nanoseconds) / 1e9
                        source_time_for_tf = source_time
                        if abs(time_diff_s) > 0.5: # If image stamp is > 0.5s off from current time
                             self.get_logger().warn(
                                f"Image timestamp diff is {time_diff_s:.2f}s. Using current time for TF.", throttle_duration_sec=5.0)
                             source_time_for_tf = current_time.to_msg()


                        if self.tf_buffer.can_transform(
                            self.param_target_map_frame_id,
                            self.param_camera_optical_frame_id,
                            source_time_for_tf, # Use potentially adjusted time
                            timeout=rclpy.duration.Duration(seconds=0.05) # Reduced timeout
                        ):
                            point_in_camera_stamped = PointStamped()
                            point_in_camera_stamped.header.stamp = source_time_for_tf
                            point_in_camera_stamped.header.frame_id = self.param_camera_optical_frame_id
                            point_in_camera_stamped.point.x = X_camera
                            point_in_camera_stamped.point.y = Y_camera
                            point_in_camera_stamped.point.z = Z_camera

                            point_in_map_stamped = self.tf_buffer.transform(
                                point_in_camera_stamped,
                                self.param_target_map_frame_id,
                                timeout=rclpy.duration.Duration(seconds=0.05)
                            )
                            stable_detection_candidate['position_in_map'] = point_in_map_stamped.point
                            point_in_map_coords = point_in_map_stamped.point
                            transform_successful = True
                        else:
                            self.get_logger().warn(
                                f"Cannot transform: TF not available or timeout from '{self.param_camera_optical_frame_id}' to "
                                f"'{self.param_target_map_frame_id}'.",
                                throttle_duration_sec=5.0
                            )
                    except (LookupException, ConnectivityException, ExtrapolationException) as e:
                        self.get_logger().warn(f"TF transform error: {e}", throttle_duration_sec=5.0)

                    if transform_successful:
                        processed_detection_for_publish_and_draw = stable_detection_candidate
                        
                        # Publish PoseStamped
                        pose_stamped_msg = PoseStamped()
                        pose_stamped_msg.header.stamp = self.get_clock().now().to_msg() # Use current time for published pose
                        pose_stamped_msg.header.frame_id = self.param_target_map_frame_id
                        pose_stamped_msg.pose.position = point_in_map_coords
                        # Assuming cube is upright, orientation can be set to default (no rotation)
                        pose_stamped_msg.pose.orientation.x = 0.0
                        pose_stamped_msg.pose.orientation.y = 0.0
                        pose_stamped_msg.pose.orientation.z = 0.0
                        pose_stamped_msg.pose.orientation.w = 1.0
                        self.cube_pose_pub.publish(pose_stamped_msg)

                        if self.param_publish_rviz_marker:
                            marker = Marker()
                            marker.header.frame_id = self.param_target_map_frame_id
                            marker.header.stamp = pose_stamped_msg.header.stamp # Match pose stamp
                            marker.ns = "cube_detector"
                            marker.id = 0 # Single, updating marker
                            marker.type = Marker.CUBE
                            marker.action = Marker.ADD
                            marker.pose = pose_stamped_msg.pose # Use the same pose
                            marker.scale.x = self.param_cube_physical_width_m
                            marker.scale.y = self.param_cube_physical_width_m
                            marker.scale.z = self.param_cube_physical_width_m
                            marker.color.r = 1.0; marker.color.g = 0.843; marker.color.b = 0.0; marker.color.a = 0.8 # Slightly transparent
                            marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg() # Marker persists for 1s then auto-deletes
                            self.cube_marker_pub.publish(marker)
                else: # self.fx is None or P_pixels too small
                    if self.fx is None:
                        self.get_logger().warn("fx is None, cannot calculate 3D pose.", throttle_duration_sec=5.0)
                    if P_pixels < 1.0:
                         self.get_logger().warn(f"Detected pixel width {P_pixels} is too small.", throttle_duration_sec=5.0)


            # --- Debug Display Logic ---
            self.frame_count_for_display += 1
            display_this_frame = (self.param_debug_display_every_n_frames > 0 and \
                                  self.frame_count_for_display % self.param_debug_display_every_n_frames == 0)

            if display_this_frame:
                debug_image_content = None
                # Only create a copy for drawing if we are going to display/publish
                if self.param_publish_debug_image or self.param_use_cv_imshow_debug:
                    debug_image_content = image_cv.copy() # Draw on a fresh copy
                    if processed_detection_for_publish_and_draw: # If a valid detection was processed
                        self.draw_detections(debug_image_content, [processed_detection_for_publish_and_draw])
                    # Optionally, draw raw_detections if no stable one, for more info
                    # else:
                    #    self.draw_detections(debug_image_content, raw_detections)


                if self.param_publish_debug_image and debug_image_content is not None:
                    try:
                        self.debug_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_image_content, "bgr8"))
                        if color_mask_for_debug is not None:
                            mask_colored_for_display = cv2.cvtColor(color_mask_for_debug, cv2.COLOR_GRAY2BGR)
                            self.debug_mask_pub.publish(self.cv_bridge.cv2_to_imgmsg(mask_colored_for_display, "bgr8"))
                    except CvBridgeError as e:
                        self.get_logger().error(f"CvBridge Error for debug publishing: {e}")
                
                if self.param_use_cv_imshow_debug and debug_image_content is not None:
                    if color_mask_for_debug is not None:
                        mask_colored_cv = cv2.cvtColor(color_mask_for_debug, cv2.COLOR_GRAY2BGR)
                        combined_display_scale = 0.6
                        h_orig, w_orig = debug_image_content.shape[:2]
                        scaled_w = max(1, int(w_orig * combined_display_scale))
                        scaled_h = max(1, int(h_orig * combined_display_scale))
                        display_result_scaled = cv2.resize(debug_image_content, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                        display_mask_scaled = cv2.resize(mask_colored_cv, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                        combined_cv_display = np.hstack([display_result_scaled, display_mask_scaled])
                        cv2.imshow("Cube Detection (CV_IMSHOW)", combined_cv_display)
                    else: # No mask available
                        cv2.imshow("Cube Detection (CV_IMSHOW) - No Mask", debug_image_content)
                    cv2.waitKey(1)

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
        # Safely clean up OpenCV windows and ROS node
        if hasattr(cube_detector, 'param_use_cv_imshow_debug') and cube_detector.param_use_cv_imshow_debug:
            cv2.destroyAllWindows()
        
        if rclpy.ok() and cube_detector.get_node_names(): # Check if node still valid before destroying
            cube_detector.destroy_node()
        
        if rclpy.ok(): # Check if rclpy context is still valid
            rclpy.shutdown()

if __name__ == "__main__":
    main()
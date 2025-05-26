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
from geometry_msgs.msg import PointStamped
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
        # (Declare them first, then get initial values, then set up callback)
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
        self._lower_hsv_bound: np.ndarray = np.array([0,0,0]) # Cached numpy array
        self._upper_hsv_bound: np.ndarray = np.array([255,255,255]) # Cached numpy array
        # Contour
        self.param_min_contour_area: int = 0
        self.param_max_contour_area: int = 0
        self.param_epsilon_factor: float = 0.0
        # Temporal Smoothing
        self.param_temporal_buffer_size: int = 0
        self.param_min_consistent_detections: int = 0
        self.param_confidence_threshold: float = 0.0
        # Debug Display
        self.param_publish_debug_image: bool = False
        self.param_use_cv_imshow_debug: bool = False
        self.param_debug_display_every_n_frames: int = 0
        # TF and Marker
        self.param_camera_optical_frame_id: str = ""
        self.param_target_map_frame_id: str = ""
        self.param_publish_rviz_marker: bool = False

        self._declare_and_load_params() # Helper to declare and load initial values
        self.add_on_set_parameters_callback(self.parameters_callback) # To update cached params

        # --- Subscriptions ---
        self.get_logger().info(f"Subscribing to image topic: {self.param_image_sub_topic}")
        self.get_logger().info(f"Subscribing to camera info topic: {self.param_camera_info_sub_topic}")

        self.image_sub = self.create_subscription(
            CompressedImage,
            self.param_image_sub_topic, # Use cached value
            self.image_callback,
            qos_profile_sensor_data
        )

        self.camera_info_received = False
        self.fx, self.fy, self.cx, self.cy = None, None, None, None
        camera_info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE, # Changed from TRANSIENT_LOCAL
            history=HistoryPolicy.KEEP_LAST,
            depth=1 # Only need the latest
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.param_camera_info_sub_topic, # Use cached value
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
        self.detection_history = deque(maxlen=self.param_temporal_buffer_size) # Use deque
        self.max_history = 5 # This seems different from temporal_buffer_size, clarify if needed
        self.detection_buffer = deque(maxlen=self.param_temporal_buffer_size) # Use deque
        self.last_valid_detection = None
        self.frame_count_for_display = 0
        self.morph_kernel = np.ones((3, 3), np.uint8) # Pre-create kernel

        self.get_logger().info("Cube detection node initialized. Waiting for camera info and images...")

        # Marker tracking (seems fine)
        self.last_published_marker_position = None
        self.marker_position_threshold = 0.05
        self.published_marker_ids = set()
        self.next_marker_id = 0

    def _declare_and_load_params(self):
        # Helper to keep __init__ cleaner
        self.declare_parameter("image_sub_topic", f"/T{self.robot_id_str}/oakd/rgb/image_raw/compressed", ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("camera_info_sub_topic", f"/T{self.robot_id_str}/oakd/rgb/camera_info", ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("cube_physical_width_m", 0.25, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter("hue_min", 15, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("hue_max", 39, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("sat_min", 90, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("sat_max", 211, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("val_min", 123, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("val_max", 255, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("min_contour_area", 500, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("max_contour_area", 30000, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("epsilon_factor", 0.02, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
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
        self._update_hsv_bounds() # Update numpy arrays
        self.param_min_contour_area = self.get_parameter("min_contour_area").get_parameter_value().integer_value
        self.param_max_contour_area = self.get_parameter("max_contour_area").get_parameter_value().integer_value
        self.param_epsilon_factor = self.get_parameter("epsilon_factor").get_parameter_value().double_value
        self.param_temporal_buffer_size = self.get_parameter("temporal_buffer_size").get_parameter_value().integer_value
        self.param_min_consistent_detections = self.get_parameter("min_consistent_detections").get_parameter_value().integer_value
        self.param_confidence_threshold = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        self.param_publish_debug_image = self.get_parameter("publish_debug_image").get_parameter_value().bool_value
        self.param_use_cv_imshow_debug = self.get_parameter("use_cv_imshow_debug").get_parameter_value().bool_value
        self.param_debug_display_every_n_frames = self.get_parameter("debug_display_every_n_frames").get_parameter_value().integer_value
        self.param_camera_optical_frame_id = self.get_parameter("camera_optical_frame_id").get_parameter_value().string_value
        self.param_target_map_frame_id = self.get_parameter("target_map_frame_id").get_parameter_value().string_value
        self.param_publish_rviz_marker = self.get_parameter("publish_rviz_marker").get_parameter_value().bool_value

        # Update deque maxlen if params change
        if hasattr(self, 'detection_buffer') and self.detection_buffer.maxlen != self.param_temporal_buffer_size:
            self.detection_buffer = deque(list(self.detection_buffer), maxlen=self.param_temporal_buffer_size)
            self.detection_history = deque(list(self.detection_history), maxlen=self.param_temporal_buffer_size) # Assuming same size for history


    def _update_hsv_bounds(self):
        self._lower_hsv_bound = np.array([self.param_hue_min, self.param_sat_min, self.param_val_min])
        self._upper_hsv_bound = np.array([self.param_hue_max, self.param_sat_max, self.param_val_max])

    def parameters_callback(self, params):
        # This is called when parameters are changed (e.g., by ros2 param set)
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
            elif param.name == "temporal_buffer_size":
                self.param_temporal_buffer_size = param.value
                # Re-initialize deques with new maxlen if it changed
                if self.detection_buffer.maxlen != self.param_temporal_buffer_size:
                    self.detection_buffer = deque(list(self.detection_buffer), maxlen=self.param_temporal_buffer_size)
                    self.detection_history = deque(list(self.detection_history), maxlen=self.param_temporal_buffer_size) # Assuming same size
            elif param.name == "min_consistent_detections": self.param_min_consistent_detections = param.value
            elif param.name == "confidence_threshold": self.param_confidence_threshold = param.value
            elif param.name == "publish_debug_image": self.param_publish_debug_image = param.value
            elif param.name == "use_cv_imshow_debug": self.param_use_cv_imshow_debug = param.value
            elif param.name == "debug_display_every_n_frames": self.param_debug_display_every_n_frames = param.value
            elif param.name == "camera_optical_frame_id": self.param_camera_optical_frame_id = param.value
            elif param.name == "target_map_frame_id": self.param_target_map_frame_id = param.value
            elif param.name == "publish_rviz_marker": self.param_publish_rviz_marker = param.value

        # Update derived parameters like HSV bounds if relevant params changed
        if any(p.name in ["hue_min", "hue_max", "sat_min", "sat_max", "val_min", "val_max"] for p in params):
            self._update_hsv_bounds()

        self.get_logger().info("Parameters updated.")
        return SetParametersResult(successful=True)


    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received: # Process only once
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_info_received = True
            self.get_logger().info(
                f"Camera intrinsics received: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}"
            )
            # Unsubscribe if we only need it once and it's RELIABLE + VOLATILE
            # self.destroy_subscription(self.camera_info_sub)
            # self.camera_info_sub = None # Avoid issues if called multiple times

    def get_color_mask(self, image): # HSV params are now member variables for bounds
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower_hsv_bound, self._upper_hsv_bound) # Use cached np.array
        # Use pre-created kernel
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=1)
        return mask

    def is_square_like(self, contour): # epsilon_factor is now a member variable
        arc_length = cv2.arcLength(contour, True)
        if arc_length < 1e-3: # Avoid division by zero or tiny arc lengths
            return False, None
        epsilon = self.param_epsilon_factor * arc_length
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Optimization: Check number of vertices early
        if len(approx) != 4:
            return False, approx # Not a quadrilateral

        # The rest of your square checking logic seems fine
        # (side lengths, angles)
        # Ensure no division by zero if avg_side or norms are very small
        sides = [np.sqrt(np.sum((approx[i][0] - approx[(i + 1) % 4][0])**2)) for i in range(4)]
        if any(s < 1e-3 for s in sides): # Check for degenerate sides
            return False, approx
        avg_side = np.mean(sides)
        if avg_side < 1e-3: # Avoid division by zero if average side is tiny
             return False, approx

        side_tolerance = 0.60 # This is quite large, allowing for very non-square quads
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
            if norm_v1 < 1e-6 or norm_v2 < 1e-6: # Increased tolerance slightly for safety
                return False, approx
            cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
            angles.append(math.degrees(math.acos(cos_theta)))

        if any(not ((90.0 - angle_tolerance_degrees) <= angle <= (90.0 + angle_tolerance_degrees)) for angle in angles):
            return False, approx
        return True, approx

    def detect_cubes(self, image): # Pass only image, other params are members
        # min_area, max_area, confidence_thresh_val are now self.param_...
        color_mask = self.get_color_mask(image) # Uses cached HSV bounds
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cube_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Optimization: Check area before more expensive checks
            if not (self.param_min_contour_area <= area <= self.param_max_contour_area):
                continue

            is_square, approx_shape = self.is_square_like(contour) # Uses cached epsilon
            if is_square and approx_shape is not None and len(approx_shape) == 4: # Ensure approx_shape is valid
                M = cv2.moments(contour)
                if M["m00"] > 1e-5: # Avoid division by zero
                    cx_pixel = int(M["m10"] / M["m00"])
                    cy_pixel = int(M["m01"] / M["m00"])
                    x_r, y_r, w_r, h_r = cv2.boundingRect(contour)
                    aspect_ratio = float(w_r) / h_r if h_r > 0 else 0 # Avoid division by zero

                    # Looser aspect ratio for "square-like" might be fine if is_square_like is robust
                    if 0.7 <= aspect_ratio <= 1.4: # This is a common check
                        confidence = self.calculate_confidence(contour, area, aspect_ratio, approx_shape)
                        if confidence >= self.param_confidence_threshold:
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
        # This logic is specific to your needs. Minor optimization:
        # Avoid re-calculating perimeter if not needed or cache intermediate results
        # if computationally intensive. For now, it looks okay.
        confidence = 100.0
        confidence -= abs(aspect_ratio - 1.0) * 50.0 # Ensure float math

        # Area penalties/bonuses
        if area < 800: # Using literals, could be params
            confidence -= (800.0 - area) / 20.0
        elif area > 20000:
            confidence -= (area - 20000.0) / 200.0
        elif 1000 <= area <= 5000: # Bonus for "ideal" size
            confidence += 10.0

        # Perimeter difference penalty
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 1e-3: # Avoid division by zero
            approx_perimeter = cv2.arcLength(approx, True)
            confidence -= abs(perimeter - approx_perimeter) / perimeter * 30.0

        return max(0.0, min(100.0, confidence)) # Clamp confidence

    def draw_detections(self, image, cube_candidates):
        # This function is mostly for debugging. If not publishing debug image,
        # it might not need to be called or could be simplified.
        # No major performance bottlenecks here other than the drawing itself.
        result_image = image # Operate on the copy passed in
        for cube in cube_candidates: # Removed index 'i' as it wasn't used
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
        return result_image # Return the modified image

    def smooth_detections(self, current_detections):
        # Uses self.detection_buffer (deque) and self.param_... for buffer size etc.
        self.detection_buffer.append(len(current_detections))
        # No pop needed due to deque's maxlen

        # Check if buffer is filled enough for consistency check
        if len(self.detection_buffer) < self.param_min_consistent_detections: # or < self.param_temporal_buffer_size for full buffer
            self.last_valid_detection = None
            return []

        consistent_detection_frames_in_buffer = sum(1 for count in self.detection_buffer if count > 0)

        if consistent_detection_frames_in_buffer >= self.param_min_consistent_detections:
            if current_detections:
                # Select the best detection if multiple, or average, etc.
                # For now, assuming current_detections is a list of dicts, and we might want to smooth properties of the best one.
                # The current logic returns the whole list if consistent.
                self.last_valid_detection = current_detections # Potentially copy() if mutable and modified later
                return current_detections
            elif self.last_valid_detection: # If current frame has no detections but buffer was consistent
                return self.last_valid_detection
            else: # No current detections, no last valid detection
                return []
        else: # Not enough consistent frames in buffer
            self.last_valid_detection = None
            return []

    def image_callback(self, image_msg: CompressedImage):
        # Parameter Caching is now handled by member variables like self.param_hue_min etc.

        if not self.camera_info_received:
            self.get_logger().warn("Camera info not yet received. Skipping frame.", throttle_duration_sec=5.0) # Throttle warning
            return

        try:
            image_cv = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            
            # --- OPTIONAL: Image Downscaling for Performance ---
            # If enabled, ensure intrinsics (fx,fy,cx,cy) and detected pixel coordinates/widths
            # are scaled appropriately before 3D calculations.
            # E.g., if you scale image_cv by 0.5:
            # image_cv = cv2.resize(image_cv, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            # Then, self.fx * 0.5, self.cx * 0.5 etc. should be used, or scale P_pixels back up.
            # For simplicity, this is not added by default.

            raw_detections, color_mask_for_debug = self.detect_cubes(image_cv) # Pass only image
            
            stable_detections = self.smooth_detections(raw_detections)

            processed_stable_detections = [] # Detections with 3D info

            if stable_detections:
                # For simplicity, process the first stable detection.
                # If multiple cubes are expected, this loop needs to be more sophisticated.
                # Consider sorting by confidence or area if multiple stable_detections.
                best_detection = stable_detections[0] # Assuming at least one

                # Ensure detection has the necessary keys
                if 'bbox' not in best_detection or 'center' not in best_detection:
                    self.get_logger().warn("Stable detection missing bbox or center. Skipping.", throttle_duration_sec=5.0)
                    # Fall through to debug display logic, which will show no processed detections
                else:
                    P_pixels = float(best_detection['bbox'][2]) # Width of bounding box

                    # Ensure fx, fy, cx, cy are valid and P_pixels is reasonable
                    if self.fx is not None and self.fx >= 1e-3 and P_pixels >= 1.0:
                        Z_camera = (self.param_cube_physical_width_m * self.fx) / P_pixels
                        u_pixel, v_pixel = best_detection['center']
                        X_camera = (u_pixel - self.cx) * Z_camera / self.fx
                        Y_camera = (v_pixel - self.cy) * Z_camera / self.fy

                        best_detection['position_in_camera'] = (X_camera, Y_camera, Z_camera)
                        # self.get_logger().info( # Log less frequently or on change
                        #     f"Cube at cam_opt_frame: ({X_camera:.2f}, {Y_camera:.2f}, {Z_camera:.2f})m"
                        # )

                        # --- TF Transformation ---
                        transform_successful = False
                        try:
                            current_time = self.get_clock().now()
                            source_time = image_msg.header.stamp
                            time_diff_s = (current_time.nanoseconds - rclpy.time.Time.from_msg(source_time).nanoseconds) / 1e9
                            
                            if abs(time_diff_s) > 1.0: # If image stamp is too old or in future
                                self.get_logger().warn(
                                    f"Image timestamp is {'old' if time_diff_s > 0 else 'in future'} ({time_diff_s:.2f}s). "
                                    f"Using current time for TF lookup.", throttle_duration_sec=5.0)
                                source_time_for_tf = current_time.to_msg()
                            else:
                                source_time_for_tf = source_time

                            # Check transform feasibility with a reasonable timeout
                            if self.tf_buffer.can_transform(
                                self.param_target_map_frame_id,
                                self.param_camera_optical_frame_id,
                                source_time_for_tf,
                                timeout=rclpy.duration.Duration(seconds=0.1) # Reduced timeout
                            ):
                                point_in_camera = PointStamped()
                                point_in_camera.header.stamp = source_time_for_tf
                                point_in_camera.header.frame_id = self.param_camera_optical_frame_id
                                point_in_camera.point.x = X_camera
                                point_in_camera.point.y = Y_camera
                                point_in_camera.point.z = Z_camera

                                point_in_map = self.tf_buffer.transform(
                                    point_in_camera,
                                    self.param_target_map_frame_id,
                                    timeout=rclpy.duration.Duration(seconds=0.05) # Shorter for actual transform
                                )
                                best_detection['position_in_map'] = point_in_map.point
                                # self.get_logger().info( # Log less frequently
                                #     f"Cube at '{self.param_target_map_frame_id}' frame: "
                                #     f"(X:{point_in_map.point.x:.2f}, Y:{point_in_map.point.y:.2f}, Z:{point_in_map.point.z:.2f})m"
                                # )
                                transform_successful = True
                            else:
                                self.get_logger().warn(
                                    f"Cannot transform from '{self.param_camera_optical_frame_id}' to "
                                    f"'{self.param_target_map_frame_id}'. TF not available or timeout.",
                                    throttle_duration_sec=5.0
                                )
                        except (LookupException, ConnectivityException, ExtrapolationException) as e:
                            self.get_logger().warn(
                                f"TF transform error: {e}", throttle_duration_sec=5.0
                            )

                        # --- RViz Marker Publishing ---
                        if transform_successful and self.param_publish_rviz_marker:
                            # Simplified marker logic for now (single updating marker)
                            marker = Marker()
                            marker.header.frame_id = self.param_target_map_frame_id
                            marker.header.stamp = self.get_clock().now().to_msg()
                            marker.ns = "cube_detector"
                            marker.id = 0 # Single, updating marker for simplicity
                            marker.type = Marker.CUBE
                            marker.action = Marker.ADD
                            marker.pose.position = best_detection['position_in_map']
                            marker.pose.orientation.w = 1.0 # No orientation assumed
                            marker.scale.x = self.param_cube_physical_width_m # Use actual width
                            marker.scale.y = self.param_cube_physical_width_m
                            marker.scale.z = self.param_cube_physical_width_m
                            marker.color.r = 1.0; marker.color.g = 0.843; marker.color.b = 0.0; marker.color.a = 0.7
                            marker.lifetime = rclpy.duration.Duration(seconds=1).to_msg() # Marker persists for 1s
                            self.cube_marker_pub.publish(marker)
                        
                        processed_stable_detections.append(best_detection)

            elif raw_detections: # No stable detections, but raw ones found
                # self.get_logger().info(f"UNSTABLE: {len(raw_detections)} raw candidate(s), 0 stable.", throttle_duration_sec=2.0)
                pass


            # --- Debug Display Logic ---
            self.frame_count_for_display += 1
            display_this_frame = (self.param_debug_display_every_n_frames > 0 and \
                                  self.frame_count_for_display % self.param_debug_display_every_n_frames == 0)

            if display_this_frame:
                # Create a copy for drawing only if we are going to display/publish
                result_image_for_display = None
                if self.param_publish_debug_image or self.param_use_cv_imshow_debug:
                    result_image_for_display = self.draw_detections(image_cv.copy(), processed_stable_detections)

                if self.param_publish_debug_image and result_image_for_display is not None:
                    try:
                        self.debug_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(result_image_for_display, "bgr8"))
                        if color_mask_for_debug is not None: # Ensure mask was created
                            # Convert mask to BGR for Image msg if it's single channel
                            if len(color_mask_for_debug.shape) == 2 or color_mask_for_debug.shape[2] == 1:
                                mask_colored_for_display = cv2.cvtColor(color_mask_for_debug, cv2.COLOR_GRAY2BGR)
                            else:
                                mask_colored_for_display = color_mask_for_debug
                            self.debug_mask_pub.publish(self.cv_bridge.cv2_to_imgmsg(mask_colored_for_display, "bgr8"))
                    except CvBridgeError as e:
                        self.get_logger().error(f"CvBridge Error for debug publishing: {e}")
                
                if self.param_use_cv_imshow_debug and result_image_for_display is not None:
                    # Ensure mask is valid before trying to use it
                    if color_mask_for_debug is not None:
                        if len(color_mask_for_debug.shape) == 2 or color_mask_for_debug.shape[2] == 1:
                             mask_colored = cv2.cvtColor(color_mask_for_debug, cv2.COLOR_GRAY2BGR)
                        else:
                             mask_colored = color_mask_for_debug

                        combined_display_scale = 0.6 # Consider making this a parameter
                        h_orig, w_orig = result_image_for_display.shape[:2]
                        
                        # Ensure scaled dimensions are at least 1x1
                        scaled_w = max(1, int(w_orig * combined_display_scale))
                        scaled_h = max(1, int(h_orig * combined_display_scale))

                        display_result_scaled = cv2.resize(result_image_for_display, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                        display_mask_scaled = cv2.resize(mask_colored, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                        combined_cv_display = np.hstack([display_result_scaled, display_mask_scaled])
                        cv2.imshow("Cube Detection (CV_IMSHOW)", combined_cv_display)
                        cv2.waitKey(1) # Essential for imshow to refresh
                    else: # color_mask_for_debug was None
                        cv2.imshow("Cube Detection (CV_IMSHOW) - No Mask", result_image_for_display) # Show only result
                        cv2.waitKey(1)

            # Update history (using deque now, append is fine)
            self.detection_history.append(len(processed_stable_detections))
            # No pop needed due to deque's maxlen

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
        # Check if node and parameter still exist before accessing
        # This can be tricky if node is already partially destroyed
        try:
            if rclpy.ok() and cube_detector.get_node_names() and \
               cube_detector.param_use_cv_imshow_debug: # Use cached param
                cv2.destroyAllWindows()
        except Exception as e:
            if rclpy.ok() and cube_detector.get_node_names(): # Check if node still valid
                 cube_detector.get_logger().warn(f"Error during cv2.destroyAllWindows: {e}")


        if rclpy.ok() and hasattr(cube_detector, 'destroy_node') and cube_detector.get_node_names(): # Check if node still valid
            cube_detector.destroy_node()
        
        if rclpy.ok(): # Check if rclpy context is still valid
            rclpy.shutdown()

if __name__ == "__main__":
    main()
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
            value="/T6/oakd/rgb/image_raw/compressed",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
            ),
        )
        
        # Default parametres
        self.declare_parameter("hue_min", 0)
        self.declare_parameter("hue_max", 18)
        self.declare_parameter("sat_min", 102)
        self.declare_parameter("sat_max", 200)
        self.declare_parameter("val_min", 101)
        self.declare_parameter("val_max", 255)
        
        # Stability parameters
        self.declare_parameter("confidence_threshold", 0.6)
        self.declare_parameter("smoothing_window", 5)
        self.declare_parameter("min_detection_area", 500)
        
        # New parameters for cube validation
        self.declare_parameter("edge_threshold", 50)  # Canny edge detection threshold
        self.declare_parameter("min_edge_ratio", 0.15)  # Minimum ratio of edge pixels to total area
        self.declare_parameter("max_edge_ratio", 0.6)   # Maximum ratio (too many edges = noise)
        self.declare_parameter("perimeter_check_thickness", 10)  # How thick the perimeter band to check
        self.declare_parameter("min_perimeter_edge_ratio", 0.3)  # Min edge ratio in perimeter
        
        # Color difference validation parameters
        self.declare_parameter("color_diff_threshold", 30)  # Minimum color difference (HSV)
        self.declare_parameter("min_different_sides", 3)  # Minimum sides that must have different colors
        self.declare_parameter("side_sample_ratio", 0.3)  # Ratio of each side to sample for color
        
        # Get parameters
        image_sub_topic = self.get_parameter("image_sub_topic").get_parameter_value().string_value
        self.hue_min = self.get_parameter("hue_min").get_parameter_value().integer_value
        self.hue_max = self.get_parameter("hue_max").get_parameter_value().integer_value
        self.sat_min = self.get_parameter("sat_min").get_parameter_value().integer_value
        self.sat_max = self.get_parameter("sat_max").get_parameter_value().integer_value
        self.val_min = self.get_parameter("val_min").get_parameter_value().integer_value
        self.val_max = self.get_parameter("val_max").get_parameter_value().integer_value
        
        self.confidence_threshold = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        self.smoothing_window_size = self.get_parameter("smoothing_window").get_parameter_value().integer_value
        self.min_detection_area = self.get_parameter("min_detection_area").get_parameter_value().integer_value
        
        # New cube validation parameters
        self.edge_threshold = self.get_parameter("edge_threshold").get_parameter_value().integer_value
        self.min_edge_ratio = self.get_parameter("min_edge_ratio").get_parameter_value().double_value
        self.max_edge_ratio = self.get_parameter("max_edge_ratio").get_parameter_value().double_value
        self.perimeter_check_thickness = self.get_parameter("perimeter_check_thickness").get_parameter_value().integer_value
        self.min_perimeter_edge_ratio = self.get_parameter("min_perimeter_edge_ratio").get_parameter_value().double_value
        
        # Color difference validation parameters
        self.color_diff_threshold = self.get_parameter("color_diff_threshold").get_parameter_value().integer_value
        self.min_different_sides = self.get_parameter("min_different_sides").get_parameter_value().integer_value
        self.side_sample_ratio = self.get_parameter("side_sample_ratio").get_parameter_value().double_value
        
        self.get_logger().info(f"{image_sub_topic=}")
        self.get_logger().info(f"HSV thresholds: H({self.hue_min}-{self.hue_max}), S({self.sat_min}-{self.sat_max}), V({self.val_min}-{self.val_max})")
        self.get_logger().info(f"Confidence threshold: {self.confidence_threshold}")
        self.get_logger().info(f"Edge detection parameters: threshold={self.edge_threshold}, ratio=({self.min_edge_ratio}-{self.max_edge_ratio})")
        self.get_logger().info(f"Color validation: diff_threshold={self.color_diff_threshold}, min_sides={self.min_different_sides}")
        
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
        self.get_logger().info("Enhanced Golden Cube Detector Node initialized and waiting for images...")
        
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
            
            # Validate image
            if image is None:
                self.get_logger().error("Received None image from cv_bridge")
                return
                
            if len(image.shape) != 3 or image.shape[0] == 0 or image.shape[1] == 0:
                self.get_logger().error(f"Invalid image shape: {image.shape}")
                return
            
            # Create a copy for visualization
            display_image = image.copy()
            
            # Process the image to detect golden cubes
            try:
                detection_result = self.detect_golden_cube(image, display_image)
                if detection_result is None or len(detection_result) != 4:
                    self.get_logger().error("detect_golden_cube returned invalid result")
                    return
                    
                raw_detection, cube_center, cube_size, raw_confidence = detection_result
                
                # Validate detection result values
                if raw_detection is None: raw_detection = False
                if cube_center is None: cube_center = (0, 0)
                if cube_size is None: cube_size = 0
                if raw_confidence is None: raw_confidence = 0.0
                    
            except Exception as e:
                self.get_logger().error(f"Error in detect_golden_cube: {str(e)}")
                import traceback
                self.get_logger().error(f"Traceback: {traceback.format_exc()}")
                return
            
            # Apply temporal smoothing for stability
            try:
                stable_detection, stable_center, stable_size, stable_confidence = self.stabilize_detection(
                    raw_detection, cube_center, cube_size, raw_confidence
                )
            except Exception as e:
                self.get_logger().error(f"Error in stabilize_detection: {str(e)}")
                return
            
            # Publish results if a cube is detected
            if stable_detection:
                position_msg = Point()
                position_msg.x = float(stable_center[0])
                position_msg.y = float(stable_center[1])
                position_msg.z = 0.0
                self.cube_position_pub.publish(position_msg)
                
                properties_msg = Float32MultiArray()
                properties_msg.data = [float(stable_size), stable_confidence]
                self.cube_properties_pub.publish(properties_msg)
                
                if stable_size > 0: # Only draw if size is valid
                    x_s = int(stable_center[0] - stable_size/2)
                    y_s = int(stable_center[1] - stable_size/2)
                    w_s = int(stable_size)
                    h_s = int(stable_size)
                    cv2.rectangle(display_image, (x_s, y_s), (x_s + w_s, y_s + h_s), (255, 0, 0), 2) # Blue
                    cv2.circle(display_image, (int(stable_center[0]), int(stable_center[1])), 5, (255, 0, 0), -1)
                    cv2.putText(display_image, f"Stable: {stable_confidence:.2f}", (x_s, y_s - 25 if y_s > 25 else y_s + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # self.get_logger().info(f"Golden cube detected at ({stable_center[0]:.1f}, {stable_center[1]:.1f}) with size {stable_size:.1f} and confidence {stable_confidence:.2f}")
            
            cv2.imshow("Golden Cube Detection", display_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")
    
    def stabilize_detection(self, detection, center, size, confidence):
        if not isinstance(center, tuple) or len(center) != 2: center = (0.0, 0.0)
        if not isinstance(size, (int, float)): size = 0.0
        if not isinstance(confidence, (int, float)): confidence = 0.0

        if detection:
            self.last_detections.append((True, center, size, confidence))
            self.detection_counter += 1
            self.no_detection_counter = 0
            self.last_valid_detection = (center, size, confidence)
        else:
            self.last_detections.append((False, center, size, 0.0)) # Store last known even if not detected now
            self.no_detection_counter += 1
            self.detection_counter = 0
        
        if len(self.last_detections) < max(3, self.smoothing_window_size // 2): # Require at least 3 or half window
            return detection, center, size, confidence
        
        recent_positive_detections = [d for d in self.last_detections if d[0]]
        
        if not recent_positive_detections or len(recent_positive_detections) < len(self.last_detections) / 2.0:
            # If few positive detections, or if last valid detection is stale
            if self.no_detection_counter > self.smoothing_window_size and self.last_valid_detection:
                 # Gradually reduce confidence of last valid detection if no new one for a while
                _, _, last_conf = self.last_valid_detection
                # return False, (0,0), 0, last_conf * 0.5 # Option to decay
            return False, (0,0), 0, 0.0

        valid_centers_x = [d[1][0] for d in recent_positive_detections]
        valid_centers_y = [d[1][1] for d in recent_positive_detections]
        valid_sizes = [d[2] for d in recent_positive_detections]
        valid_confidences = [d[3] for d in recent_positive_detections]
        
        stable_center_x = np.median(valid_centers_x)
        stable_center_y = np.median(valid_centers_y)
        stable_size = np.median(valid_sizes)
        stable_confidence = np.median(valid_confidences)
        
        if stable_size < self.min_detection_area**0.5 / 5 : # If stable size is too small, likely noise
            return False, (0,0), 0, 0.0

        return True, (stable_center_x, stable_center_y), stable_size, stable_confidence
    
    def validate_cube_shape(self, image, contour, mask):
        try:
            x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(contour) # Bounding box of the contour
            
            if w_bbox <= 0 or h_bbox <= 0: return False, 0.0, "Invalid bbox dims"
            if y_bbox + h_bbox > image.shape[0] or x_bbox + w_bbox > image.shape[1]:
                return False, 0.0, "Bbox exceeds image" # Should be caught by ROI creation if image is passed correctly
            
            roi_color = image[y_bbox : y_bbox + h_bbox, x_bbox : x_bbox + w_bbox]
            roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            roi_mask = mask[y_bbox : y_bbox + h_bbox, x_bbox : x_bbox + w_bbox] # Mask specific to this ROI
            roi_hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            
            if roi_color.size == 0 or roi_hsv.size == 0 or roi_mask.size == 0:
                return False, 0.0, "Empty ROI"
            
            full_image_dims = image.shape[:2] # (height, width)

            try:
                color_validation_score, different_sides_count = self.check_perimeter_color_differences(
                    roi_hsv, roi_mask, x_bbox, y_bbox, w_bbox, h_bbox, full_image_dims
                )
            except Exception as e:
                self.get_logger().error(f"Error in check_perimeter_color_differences: {str(e)}")
                color_validation_score, different_sides_count = 0.0, 0
            
            edges = cv2.Canny(roi_gray, self.edge_threshold, self.edge_threshold * 2)
            # cv2.imshow("Edge Detection ROI", edges) # Debugging specific ROI
            
            golden_pixels_in_roi = np.sum(roi_mask > 0)
            if golden_pixels_in_roi == 0: return False, 0.0, "No golden pixels in ROI"
            
            edges_in_golden = cv2.bitwise_and(edges, roi_mask)
            edge_pixels_in_golden = np.sum(edges_in_golden > 0)
            edge_ratio_in_golden = edge_pixels_in_golden / golden_pixels_in_roi
            
            perimeter_score = self.check_perimeter_edges(roi_gray, roi_mask)
            corner_score = self.detect_cube_corners(roi_gray, roi_mask)
            
            image_area = full_image_dims[0] * full_image_dims[1]
            object_area = w_bbox * h_bbox
            size_ratio = object_area / image_area if image_area > 0 else 0
            size_score = max(0, 1.0 - (size_ratio - 0.5) * 2) if size_ratio > 0.5 else 1.0
            
            edge_score = 1.0 if (self.min_edge_ratio <= edge_ratio_in_golden <= self.max_edge_ratio) else 0.0
            
            cube_confidence = (0.5 * color_validation_score + 0.2 * perimeter_score + 
                              0.1 * corner_score + 0.1 * size_score + 0.1 * edge_score)
            
            reason = f"ClrSides:{different_sides_count}/{self.min_different_sides}, ClrScr:{color_validation_score:.2f}"
            
            is_valid_cube = (cube_confidence > self.confidence_threshold * 0.8 and # Slightly lower threshold for internal validation score
                            different_sides_count >= self.min_different_sides)
            
            return is_valid_cube, cube_confidence, reason
            
        except Exception as e:
            self.get_logger().error(f"Error in validate_cube_shape: {str(e)}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")
            return False, 0.0, f"Validation error: {str(e)}"

    def check_perimeter_color_differences(self, roi_hsv, roi_mask, 
                                          bbox_x, bbox_y, bbox_w, bbox_h, 
                                          full_image_dims):
        h_roi, w_roi = roi_mask.shape
        full_img_h, full_img_w = full_image_dims

        golden_pixels_hsv = roi_hsv[roi_mask > 0]
        if len(golden_pixels_hsv) == 0: return 0.0, 0
        
        avg_golden_color = np.mean(golden_pixels_hsv, axis=0)
        if avg_golden_color is None or len(avg_golden_color) != 3: return 0.0, 0
        
        sides = {
            'top': (0, int(h_roi * self.side_sample_ratio)),
            'bottom': (int(h_roi * (1 - self.side_sample_ratio)), h_roi),
            'left': (0, int(w_roi * self.side_sample_ratio)),
            'right': (int(w_roi * (1 - self.side_sample_ratio)), w_roi)
        }
        
        different_sides = 0
        debug_image = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        
        for side_name, (start_coord, end_coord) in sides.items():
            perimeter_hsv_region = None
            mask_region_for_side = None # Renamed to avoid conflict
            actual_start, actual_end = 0,0 

            # Determine if this side of the bounding box is at the edge of the full image
            is_at_full_image_edge = False
            if side_name == 'top' and bbox_y == 0:
                is_at_full_image_edge = True
            elif side_name == 'bottom' and (bbox_y + bbox_h) >= full_img_h:
                is_at_full_image_edge = True
            elif side_name == 'left' and bbox_x == 0:
                is_at_full_image_edge = True
            elif side_name == 'right' and (bbox_x + bbox_w) >= full_img_w:
                is_at_full_image_edge = True

            # Define sample regions within ROI
            if side_name in ['top', 'bottom']:
                actual_start = max(0, start_coord)
                actual_end = min(h_roi, end_coord)
                if actual_start >= actual_end: continue
                perimeter_hsv_region = roi_hsv[actual_start:actual_end, :]
                mask_region_for_side = roi_mask[actual_start:actual_end, :]
            else: # left, right
                actual_start = max(0, start_coord)
                actual_end = min(w_roi, end_coord)
                if actual_start >= actual_end: continue
                perimeter_hsv_region = roi_hsv[:, actual_start:actual_end]
                mask_region_for_side = roi_mask[:, actual_start:actual_end]

            # Draw rectangle for the sampled region first (default: yellow for "not processed yet")
            rect_color = (0, 255, 255) # Yellow

            if is_at_full_image_edge:
                rect_color = (255, 255, 0) # Cyan for AT CAMERA EDGE
                # This side automatically fails the color difference test
            elif perimeter_hsv_region is None or perimeter_hsv_region.size == 0 or \
                 mask_region_for_side is None or mask_region_for_side.size == 0:
                rect_color = (0, 165, 255) # Orange for invalid sample region
            else:
                background_pixels_hsv = perimeter_hsv_region[mask_region_for_side == 0]
                if len(background_pixels_hsv) > 10:
                    avg_background_color = np.mean(background_pixels_hsv, axis=0)
                    if avg_background_color is not None and len(avg_background_color) == 3:
                        color_diff = self.calculate_hsv_difference(avg_golden_color, avg_background_color)
                        if color_diff is not None and color_diff > self.color_diff_threshold:
                            different_sides += 1
                            rect_color = (0, 255, 0) # Green for different
                        else:
                            rect_color = (0, 0, 255) # Red for similar
                    else: # Should not happen
                        rect_color = (0, 165, 255) # Orange for avg_background_color error
                else: # Not enough background pixels
                    rect_color = (0, 165, 255) # Orange for not enough background

            # Draw the rectangle on debug image
            if side_name == 'top': cv2.rectangle(debug_image, (0, actual_start), (w_roi-1, actual_end-1), rect_color, 1)
            elif side_name == 'bottom': cv2.rectangle(debug_image, (0, actual_start), (w_roi-1, actual_end-1), rect_color, 1)
            elif side_name == 'left': cv2.rectangle(debug_image, (actual_start, 0), (actual_end-1, h_roi-1), rect_color, 1)
            else: cv2.rectangle(debug_image, (actual_start, 0), (actual_end-1, h_roi-1), rect_color, 1)

        cv2.imshow("Color Difference Analysis ROI", debug_image)
        
        validation_score = 0.0
        if self.min_different_sides > 0:
            if different_sides >= self.min_different_sides:
                validation_score = 0.75 + 0.25 * ( (different_sides - self.min_different_sides) / float(max(1, 4 - self.min_different_sides)) )
                validation_score = min(1.0, validation_score)
            else:
                validation_score = (different_sides / float(self.min_different_sides)) * 0.75
        elif different_sides > 0 : # if min_different_sides is 0, any diff side is a pass
             validation_score = 1.0

        return validation_score, different_sides
    
    def calculate_hsv_difference(self, color1, color2):
        if color1 is None or color2 is None or len(color1) !=3 or len(color2) !=3 :
            return 0 # Or some other indicator of error

        h_diff = min(abs(color1[0] - color2[0]), 180 - abs(color1[0] - color2[0]))
        s_diff = abs(color1[1] - color2[1])
        v_diff = abs(color1[2] - color2[2])
        total_diff = 2.0 * h_diff + 1.0 * s_diff + 1.0 * v_diff
        return total_diff
        
    def check_perimeter_edges(self, roi_gray, roi_mask):
        h_roi, w_roi = roi_mask.shape
        thickness = max(1, self.perimeter_check_thickness)
        kernel = np.ones((thickness, thickness), np.uint8)
        
        dilated_mask = cv2.dilate(roi_mask, kernel, iterations=1)
        perimeter_band = cv2.subtract(dilated_mask, roi_mask)
        
        if np.sum(perimeter_band > 0) == 0: return 0.0
        
        edges = cv2.Canny(roi_gray, self.edge_threshold, self.edge_threshold * 2)
        edges_in_perimeter = cv2.bitwise_and(edges, perimeter_band)
        edge_pixels_in_perimeter = np.sum(edges_in_perimeter > 0)
        perimeter_pixels_count = np.sum(perimeter_band > 0)
        
        perimeter_edge_ratio = edge_pixels_in_perimeter / perimeter_pixels_count if perimeter_pixels_count > 0 else 0
        
        score = 0.0
        if self.min_perimeter_edge_ratio > 0:
            if perimeter_edge_ratio >= self.min_perimeter_edge_ratio:
                score = min(1.0, (perimeter_edge_ratio / self.min_perimeter_edge_ratio) * 0.75 + 0.25)
            else:
                score = (perimeter_edge_ratio / self.min_perimeter_edge_ratio) * 0.75
        elif perimeter_edge_ratio > 0: # If min_ratio is 0, any edge is good
            score = 1.0
        return score
    
    def detect_cube_corners(self, roi_gray, roi_mask):
        if roi_gray.size == 0 or roi_mask.size == 0: return 0.0
        roi_gray_float = np.float32(roi_gray)
        corners = cv2.cornerHarris(roi_gray_float, 2, 3, 0.04) # blockSize, ksize, k
        # corners = cv2.dilate(corners, None) # Optional
        
        if corners.max() <= 0: return 0.0
        corner_threshold = 0.01 * corners.max()
        
        corner_pixels_mask = (corners > corner_threshold).astype(np.uint8) * 255
        corners_in_golden = cv2.bitwise_and(corner_pixels_mask, roi_mask)
        
        # Instead of pixel count, count distinct corner regions
        contours, _ = cv2.findContours(corners_in_golden, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_distinct_corners = len(contours)

        if 3 <= num_distinct_corners <= 7: # Cube faces usually have 3-4 visible corners, allow some error
            return 1.0
        elif 2 <= num_distinct_corners <= 8:
            return 0.7
        elif num_distinct_corners > 0:
            return 0.3
        else:
            return 0.0
    
    def detect_golden_cube(self, image, display_image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_gold = np.array([self.hue_min, self.sat_min, self.val_min])
        upper_gold = np.array([self.hue_max, self.sat_max, self.val_max])
        mask = cv2.inRange(hsv_image, lower_gold, upper_gold)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        cv2.imshow("Gold Mask Full", mask)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_detection = {"detected": False, "center": (0,0), "size": 0, "confidence": 0.0}

        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3] # Check top 3 largest

            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= self.min_detection_area: continue

                rect = cv2.minAreaRect(contour)
                box_points = cv2.boxPoints(rect)
                box_int = np.int0(box_points)
                
                (center_x, center_y), (width, height), angle = rect
                if width <=0 or height <=0: continue

                x_br, y_br, w_br, h_br = cv2.boundingRect(contour)
                
                aspect_ratio = max(width, height) / min(width, height)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                is_valid_cube, cube_validation_score, validation_reason = self.validate_cube_shape(
                    image, contour, mask # Pass full image, contour, and full mask
                )
                
                ar_confidence = max(0.0, 1.0 - abs(1.0 - aspect_ratio)) # Should be close to 1
                shape_confidence = max(0.0, 1.0 - (abs(len(approx) - 4) / 4.0))
                solidity_confidence = min(1.0, solidity * 1.1) # Solidity is often high for good shapes
                
                current_confidence = (0.20 * ar_confidence + 0.20 * shape_confidence + 
                                     0.10 * solidity_confidence + 0.50 * cube_validation_score) # Heavier on validation
                
                current_size = (width + height) / 2.0
                
                # Draw individual contour processing attempt
                temp_display = display_image.copy() if len(contours) > 1 else display_image # Avoid overdrawing if only one
                
                if current_confidence > self.confidence_threshold and is_valid_cube:
                    if current_confidence > best_detection["confidence"]:
                        best_detection = {
                            "detected": True, 
                            "center": (center_x, center_y), 
                            "size": current_size, 
                            "confidence": current_confidence
                        }
                    cv2.drawContours(temp_display, [box_int], 0, (0, 255, 0), 2)
                    cv2.putText(temp_display, f"VALID: {current_confidence:.2f}", (x_br, y_br - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.drawContours(temp_display, [box_int], 0, (0, 165, 255), 1) # Orange for rejected
                    reason_str = "LowConf" if not is_valid_cube else "NotCube"
                    cv2.putText(temp_display, f"REJ({reason_str}):{current_confidence:.2f} {validation_reason}", (x_br, y_br - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                
                if len(contours) > 1 : cv2.imshow(f"Contour Proc {contours.index(contour)}", temp_display)


        return best_detection["detected"], best_detection["center"], best_detection["size"], best_detection["confidence"]

def main(args=None):
    rclpy.init(args=args)
    detector_node = GoldenCubeDetectorNode()
    try:
        rclpy.spin(detector_node)
    except KeyboardInterrupt:
        detector_node.get_logger().info("KeyboardInterrupt, shutting down.")
    finally:
        cv2.destroyAllWindows()
        detector_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
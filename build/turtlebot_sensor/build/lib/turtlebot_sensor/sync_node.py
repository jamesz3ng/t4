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
                
            if len(image.shape) != 3:
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
                if raw_detection is None:
                    raw_detection = False
                if cube_center is None:
                    cube_center = (0, 0)
                if cube_size is None:
                    cube_size = 0
                if raw_confidence is None:
                    raw_confidence = 0.0
                    
            except Exception as e:
                self.get_logger().error(f"Error in detect_golden_cube: {str(e)}")
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
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")
    
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
    
    def validate_cube_shape(self, image, contour, mask):
        """
        Validate if the detected golden region is actually a cube by checking for:
        1. Different colored pixels around the perimeter (indicating complete cube edges)
        2. Proper edge structure around the perimeter
        3. Reasonable edge density within the region
        4. Shape characteristics that indicate a 3D cube rather than flat surface
        """
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Validate bounding rectangle
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                return False, 0.0, "Invalid bounding rectangle"
            
            if y + h > image.shape[0] or x + w > image.shape[1]:
                return False, 0.0, "Bounding rectangle exceeds image bounds"
            
            # Create ROI (Region of Interest)
            roi_color = image[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            roi_mask = mask[y:y+h, x:x+w]
            roi_hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            
            # Validate ROI
            if roi_color is None or roi_hsv is None or roi_mask is None:
                return False, 0.0, "Failed to create ROI"
            
            if roi_color.size == 0 or roi_hsv.size == 0 or roi_mask.size == 0:
                return False, 0.0, "Empty ROI"
            
            # 1. NEW: Check for color differences around perimeter (your main requirement)
            try:
                color_validation_score, different_sides_count = self.check_perimeter_color_differences(
                    roi_hsv, roi_mask
                )
            except Exception as e:
                self.get_logger().error(f"Error in check_perimeter_color_differences: {str(e)}")
                color_validation_score, different_sides_count = 0.0, 0
            
            # Apply Canny edge detection to the ROI
            edges = cv2.Canny(roi_gray, self.edge_threshold, self.edge_threshold * 2)
            
            # Show edge detection result for debugging
            edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Edge Detection", edge_display)
            
            # 2. Check edge density within the golden region
            golden_pixels = np.sum(roi_mask > 0)
            if golden_pixels == 0:
                return False, 0.0, "No golden pixels"
            
            # Count edges within the golden region
            edges_in_golden = cv2.bitwise_and(edges, roi_mask)
            edge_pixels_in_golden = np.sum(edges_in_golden > 0)
            edge_ratio_in_golden = edge_pixels_in_golden / golden_pixels
            
            # 3. Check perimeter edges (to detect cube boundaries)
            try:
                perimeter_score = self.check_perimeter_edges(roi_gray, roi_mask)
            except Exception as e:
                self.get_logger().error(f"Error in check_perimeter_edges: {str(e)}")
                perimeter_score = 0.0
            
            # 4. Check for corner detection (cubes should have distinct corners)
            try:
                corner_score = self.detect_cube_corners(roi_gray, roi_mask)
            except Exception as e:
                self.get_logger().error(f"Error in detect_cube_corners: {str(e)}")
                corner_score = 0.0
            
            # 5. Check size constraints (cubes shouldn't be too large relative to image)
            image_area = image.shape[0] * image.shape[1]
            object_area = w * h
            size_ratio = object_area / image_area
            
            # Size score: penalize objects that take up too much of the frame
            if size_ratio > 0.5:  # If object takes up more than 50% of frame
                size_score = max(0, 1.0 - (size_ratio - 0.5) * 2)  # Penalty for large objects
            else:
                size_score = 1.0
            
            # Combine all validation scores with heavy weight on color validation
            edge_score = 1.0 if (self.min_edge_ratio <= edge_ratio_in_golden <= self.max_edge_ratio) else 0.0
            
            # Overall cube validation score - color validation gets highest weight
            cube_confidence = (0.5 * color_validation_score + 0.2 * perimeter_score + 
                              0.1 * corner_score + 0.1 * size_score + 0.1 * edge_score)
            
            # Debug information
            reason = f"Color:{different_sides_count}/{self.min_different_sides} sides, Score:{color_validation_score:.2f}"
            
            # Return validation result - require color validation to pass
            is_valid_cube = (cube_confidence > 0.5 and 
                            different_sides_count >= self.min_different_sides)
            
            return is_valid_cube, cube_confidence, reason
            
        except Exception as e:
            self.get_logger().error(f"Error in validate_cube_shape: {str(e)}")
            return False, 0.0, f"Validation error: {str(e)}"
    
    def check_perimeter_color_differences(self, roi_hsv, roi_mask):
        """
        Check if the perimeter of the golden object has different colors compared to the object itself.
        This indicates we can see complete cube edges rather than just part of a large surface.
        Returns: (validation_score, number_of_different_sides)
        """
        h, w = roi_mask.shape
        
        # Get the average color of the golden object itself
        golden_pixels = roi_hsv[roi_mask > 0]
        if len(golden_pixels) == 0:
            return 0.0, 0
        
        avg_golden_color = np.mean(golden_pixels, axis=0)
        
        # Ensure we have valid color values
        if avg_golden_color is None or len(avg_golden_color) != 3:
            return 0.0, 0
        
        # Define the four sides of the bounding rectangle
        sides = {
            'top': (0, int(h * self.side_sample_ratio)),           # Top portion
            'bottom': (int(h * (1 - self.side_sample_ratio)), h), # Bottom portion  
            'left': (0, int(w * self.side_sample_ratio)),          # Left portion
            'right': (int(w * (1 - self.side_sample_ratio)), w)   # Right portion
        }
        
        different_sides = 0
        side_differences = {}
        
        # Create visualization image for debugging
        debug_image = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        
        for side_name, (start, end) in sides.items():
            if side_name in ['top', 'bottom']:
                # For top/bottom sides, sample the perimeter region
                if side_name == 'top':
                    perimeter_region = roi_hsv[start:end, :]
                    mask_region = roi_mask[start:end, :]
                else:
                    perimeter_region = roi_hsv[start:end, :]
                    mask_region = roi_mask[start:end, :]
            else:
                # For left/right sides, sample the perimeter region
                if side_name == 'left':
                    perimeter_region = roi_hsv[:, start:end]
                    mask_region = roi_mask[:, start:end]
                else:
                    perimeter_region = roi_hsv[:, start:end]
                    mask_region = roi_mask[:, start:end]
            
            # Get pixels that are NOT part of the golden object (background pixels)
            background_pixels = perimeter_region[mask_region == 0]
            
            if len(background_pixels) > 10:  # Need sufficient background pixels for reliable average
                # Calculate average background color for this side
                avg_background_color = np.mean(background_pixels, axis=0)
                
                # Ensure we have valid background color
                if avg_background_color is not None and len(avg_background_color) == 3:
                    # Calculate color difference in HSV space
                    color_diff = self.calculate_hsv_difference(avg_golden_color, avg_background_color)
                    side_differences[side_name] = color_diff
                    
                    # Check if the difference is significant enough
                    if color_diff > self.color_diff_threshold:
                        different_sides += 1
                        
                        # Mark this side as different in debug image
                        if side_name == 'top':
                            cv2.rectangle(debug_image, (0, start), (w-1, end-1), (0, 255, 0), 2)
                        elif side_name == 'bottom':
                            cv2.rectangle(debug_image, (0, start), (w-1, end-1), (0, 255, 0), 2)
                        elif side_name == 'left':
                            cv2.rectangle(debug_image, (start, 0), (end-1, h-1), (0, 255, 0), 2)
                        else:  # right
                            cv2.rectangle(debug_image, (start, 0), (end-1, h-1), (0, 255, 0), 2)
                    else:
                        # Mark this side as similar in debug image
                        if side_name == 'top':
                            cv2.rectangle(debug_image, (0, start), (w-1, end-1), (0, 0, 255), 2)
                        elif side_name == 'bottom':
                            cv2.rectangle(debug_image, (0, start), (w-1, end-1), (0, 0, 255), 2)
                        elif side_name == 'left':
                            cv2.rectangle(debug_image, (start, 0), (end-1, h-1), (0, 0, 255), 2)
                        else:  # right
                            cv2.rectangle(debug_image, (start, 0), (end-1, h-1), (0, 0, 255), 2)
                else:
                    side_differences[side_name] = 0
            else:
                side_differences[side_name] = 0
        
        # Show debug visualization
        cv2.imshow("Color Difference Analysis", debug_image)
        
        # Calculate validation score based on how many sides have different colors
        if different_sides >= self.min_different_sides:
            validation_score = min(1.0, different_sides / 4.0)  # Perfect score if all 4 sides different
        else:
            validation_score = different_sides / self.min_different_sides
        
        return validation_score, different_sides
    
    def calculate_hsv_difference(self, color1, color2):
        """
        Calculate the difference between two HSV colors.
        Takes into account the circular nature of hue.
        """
        # Hue difference (circular)
        h_diff = min(abs(color1[0] - color2[0]), 180 - abs(color1[0] - color2[0]))
        
        # Saturation and Value differences (linear)
        s_diff = abs(color1[1] - color2[1])
        v_diff = abs(color1[2] - color2[2])
        
        # Combined difference with different weights
        # Hue is most important for distinguishing colors
        total_diff = 2.0 * h_diff + 1.0 * s_diff + 1.0 * v_diff
        
        return total_diff
    def check_perimeter_edges(self, roi_gray, roi_mask):
        """Check for edges around the perimeter of the golden region"""
        h, w = roi_mask.shape
        
        # Create a perimeter band around the golden region
        kernel = np.ones((self.perimeter_check_thickness, self.perimeter_check_thickness), np.uint8)
        dilated_mask = cv2.dilate(roi_mask, kernel, iterations=1)
        perimeter_band = dilated_mask - roi_mask
        
        if np.sum(perimeter_band) == 0:
            return 0.0
        
        # Apply edge detection
        edges = cv2.Canny(roi_gray, self.edge_threshold, self.edge_threshold * 2)
        
        # Count edges in the perimeter band
        edges_in_perimeter = cv2.bitwise_and(edges, perimeter_band)
        edge_pixels_in_perimeter = np.sum(edges_in_perimeter > 0)
        perimeter_pixels = np.sum(perimeter_band > 0)
        
        perimeter_edge_ratio = edge_pixels_in_perimeter / perimeter_pixels if perimeter_pixels > 0 else 0
        
        # Score based on perimeter edge ratio
        if perimeter_edge_ratio >= self.min_perimeter_edge_ratio:
            return min(1.0, perimeter_edge_ratio / self.min_perimeter_edge_ratio)
        else:
            return perimeter_edge_ratio / self.min_perimeter_edge_ratio
    
    def detect_cube_corners(self, roi_gray, roi_mask):
        """Detect corners that would indicate a cube structure"""
        # Use Harris corner detection
        corners = cv2.cornerHarris(roi_gray, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        
        # Threshold for corner detection
        corner_threshold = 0.01 * corners.max() if corners.max() > 0 else 0
        corner_pixels = corners > corner_threshold
        
        # Count corners within the golden region
        corners_in_golden = cv2.bitwise_and(corner_pixels.astype(np.uint8) * 255, roi_mask)
        num_corners = np.sum(corners_in_golden > 0)
        
        # Score based on number of corners (expect 4 corners for a cube face, but allow some tolerance)
        if 3 <= num_corners <= 6:  # Reasonable range for cube corners
            return 1.0
        elif 2 <= num_corners <= 8:  # Acceptable range
            return 0.7
        elif num_corners > 0:
            return 0.3
        else:
            return 0.0
    
    def detect_golden_cube(self, image, display_image):
        """
        Process the image to detect golden cubes with enhanced validation
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
            
            # Validate area
            if area is None or area <= 0:
                return cube_detected, cube_center, cube_size, confidence
            
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
                
                # ENHANCED: Validate if this is actually a cube using edge detection
                try:
                    is_valid_cube, cube_validation_score, validation_reason = self.validate_cube_shape(
                        image, largest_contour, mask
                    )
                except Exception as e:
                    self.get_logger().error(f"Error in validate_cube_shape: {str(e)}")
                    is_valid_cube, cube_validation_score, validation_reason = False, 0.0, "Validation failed"
                
                # Calculate confidence based on multiple factors
                # 1. Aspect ratio (closer to 1 is better)
                ar_confidence = 1.0 - min(abs(1.0 - aspect_ratio), 1.0)
                
                # 2. Shape approximation (4 corners for a rectangle/square)
                vertex_diff = abs(len(approx) - 4)
                shape_confidence = max(0.0, 1.0 - (vertex_diff / 4.0))
                
                # 3. Solidity (higher is better, indicates how "solid" or filled the shape is)
                solidity_confidence = min(1.0, solidity * 1.25)  # Scale to give more weight
                
                # 4. NEW: Cube validation score from edge analysis
                cube_structure_confidence = cube_validation_score
                
                # Combined confidence with cube validation having significant weight
                confidence = (0.25 * ar_confidence + 0.25 * shape_confidence + 
                             0.15 * solidity_confidence + 0.35 * cube_structure_confidence)
                
                # Calculate size consistently (average of width and height)
                cube_size = (width + height) / 2
                
                # Store detection info
                cube_center = (center_x, center_y)
                
                # If confidence is high enough AND cube validation passes, consider it a cube
                if confidence > self.confidence_threshold and is_valid_cube:
                    cube_detected = True
                    
                    # Draw raw detection results on the display image (in green)
                    cv2.drawContours(display_image, [box], 0, (0, 255, 0), 2)
                    cv2.circle(display_image, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
                    cv2.putText(display_image, f"CUBE: {confidence:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Debug information
                    cv2.putText(display_image, f"AR: {aspect_ratio:.2f}", (x, y + h + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(display_image, f"Validation: {validation_reason}", (x, y + h + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                else:
                    # Draw rejected contour with reason
                    cv2.drawContours(display_image, [box], 0, (0, 165, 255), 2)
                    rejection_reason = "Low conf" if confidence <= self.confidence_threshold else "Not cube"
                    cv2.putText(display_image, f"{rejection_reason}: {confidence:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    cv2.putText(display_image, f"Val: {validation_reason}", (x, y + h + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
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
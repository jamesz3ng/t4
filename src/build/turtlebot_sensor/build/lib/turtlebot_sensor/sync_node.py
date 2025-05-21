import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class GoldenCubeDetectorNode(Node):

    def __init__(self):
        super().__init__("golden_cube_detector")

        # Parameters
        self.declare_parameter(
            name="image_sub_topic",
            value="/T8/oakd/rgb/image_raw/compressed",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
            ),
        )
        
        # HSV thresholds for golden color detection
        self.declare_parameter("hue_min", 20, 
                               descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("hue_max", 40, 
                               descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("sat_min", 100, 
                               descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("sat_max", 255, 
                               descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("val_min", 100, 
                               descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("val_max", 255, 
                               descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        
        # Get parameters
        image_sub_topic = self.get_parameter("image_sub_topic").get_parameter_value().string_value
        self.get_logger().info(f"{image_sub_topic=}")
        
        # Get HSV threshold values
        self.hue_min = self.get_parameter("hue_min").get_parameter_value().integer_value
        self.hue_max = self.get_parameter("hue_max").get_parameter_value().integer_value
        self.sat_min = self.get_parameter("sat_min").get_parameter_value().integer_value
        self.sat_max = self.get_parameter("sat_max").get_parameter_value().integer_value
        self.val_min = self.get_parameter("val_min").get_parameter_value().integer_value
        self.val_max = self.get_parameter("val_max").get_parameter_value().integer_value

        # Create subscribers
        self.image_sub = Subscriber(
            self, CompressedImage, image_sub_topic, qos_profile=qos_profile_sensor_data
        )

        # Create publishers
        self.cube_positions_pub = self.create_publisher(
            Point, "golden_cube_position", 10
        )
        
        self.debug_image_pub = self.create_publisher(
            CompressedImage, "golden_cube_detection/debug_image/compressed", 10
        )

        self.cv_bridge = CvBridge()

        # Set up time synchronizer
        queue_size = 100
        max_delay = 0.1
        self.time_sync = ApproximateTimeSynchronizer(
            [self.image_sub],
            queue_size,
            max_delay,
        )
        self.time_sync.registerCallback(self.process_image)

    def process_image(self, image_msg: CompressedImage):
        try:
            # Convert to OpenCV format
            image = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg)
            
            # Create a copy for visualization
            debug_image = image.copy()
            
            # Convert to HSV color space for better color segmentation
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Create a mask for golden color
            lower_golden = np.array([self.hue_min, self.sat_min, self.val_min])
            upper_golden = np.array([self.hue_max, self.sat_max, self.val_max])
            mask = cv2.inRange(hsv_image, lower_golden, upper_golden)
            
            # Apply morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour
            for contour in contours:
                # Filter small contours
                area = cv2.contourArea(contour)
                if area < 500:  # Adjust this threshold based on your cube size
                    continue
                
                # Approximate the contour to simplify shape detection
                epsilon = 0.05 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If it's approximately a square (4-sided polygon)
                if len(approx) >= 4 and len(approx) <= 6:
                    # Get the bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate aspect ratio
                    aspect_ratio = float(w) / h
                    
                    # Cubes should have an aspect ratio close to 1
                    if 0.8 <= aspect_ratio <= 1.2:
                        # Calculate center of the cube
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Publish cube position
                        cube_pos = Point()
                        cube_pos.x = float(center_x)
                        cube_pos.y = float(center_y)
                        cube_pos.z = 0.0  # No depth info in 2D image
                        self.cube_positions_pub.publish(cube_pos)
                        
                        # Draw on debug image
                        cv2.drawContours(debug_image, [approx], -1, (0, 255, 0), 3)
                        cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)
                        cv2.putText(debug_image, f"Cube: ({center_x}, {center_y})", 
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Publish debug image
            debug_msg = self.cv_bridge.cv2_to_compressed_imgmsg(debug_image)
            self.debug_image_pub.publish(debug_msg)
            
            # Display the image for debug purposes
            cv2.imshow("Original Image", image)
            cv2.imshow("Golden Mask", mask)
            cv2.imshow("Golden Cube Detection", debug_image)
            cv2.waitKey(10)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")


def main(args=None):
    rclpy.init(args=args)

    cube_detector = GoldenCubeDetectorNode()

    rclpy.spin(cube_detector)

    cube_detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
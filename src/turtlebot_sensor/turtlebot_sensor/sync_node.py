import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


class TimeSyncNode(Node):

    def __init__(self):
        super().__init__("sync_node")

        self.declare_parameter(
            name="image_sub_topic",
            value="/T7/oakd/rgb/image_raw/compressed",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
            ),
        )

        image_sub_topic = (
            self.get_parameter("image_sub_topic").get_parameter_value().string_value
        )

        self.get_logger().info(f"{image_sub_topic=}")

        self.image_sub = Subscriber(
            self, CompressedImage, image_sub_topic, qos_profile=qos_profile_sensor_data
        )

        self.cv_bridge = CvBridge()

        # Since we're only using one subscriber now, we don't need ApproximateTimeSynchronizer
        # Instead, we can just use a direct callback
        self.image_sub.registerCallback(self.image_callback)
        
        self.get_logger().info("Node initialized and waiting for images...")

    def image_callback(self, image_msg: CompressedImage):
        try:
            self.get_logger().info("Received image message")
            image = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg)

            # Get image height and width
            height, width = image.shape[:2]
            self.get_logger().info(f"Image dimensions: {width}x{height}")

            # Take centre of the image
            center_x = width // 2
            center_y = height // 2
            
            # Draw a marker at the center for visualization
            cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)

            cv2.imshow("Image", image)
            cv2.waitKey(10)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")


def main(args=None):
    rclpy.init(args=args)

    time_sync = TimeSyncNode()
    
    try:
        rclpy.spin(time_sync)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        cv2.destroyAllWindows()
        time_sync.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage


def convert_compressedDepth_to_cv2(compressed_depth, depth_header_size=12):
    """
    Convert a compressedDepth topic image into a cv2 image.
    compressed_depth must be from a topic /bla/compressedDepth
    as it's encoded in PNG
    Code from: https://answers.ros.org/question/249775/display-compresseddepth-image-python-cv2/
    """
    depth_fmt, compr_type = compressed_depth.format.split(";")
    # remove white space
    depth_fmt = depth_fmt.strip()
    compr_type = compr_type.strip()
    if compr_type != "compressedDepth":
        raise Exception(
            "Compression type is not 'compressedDepth'."
            "You probably subscribed to the wrong topic."
        )

    # remove header from raw data, if necessary

    # If it comes from a robot/sensor, it has 12 useless bytes apparently
    if "PNG" in compressed_depth.data[:12]:
        # If we compressed it with opencv, there is nothing to strip
        depth_header_size = 0

    raw_data = compressed_depth.data[depth_header_size:]

    depth_img_raw = cv2.imdecode(
        np.frombuffer(raw_data, np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if depth_img_raw is None:
        # probably wrong header size
        raise Exception(
            "Could not decode compressed depth image."
            "You may need to change 'depth_header_size'!"
        )
    return depth_img_raw


def pixels_to_mm(depth_image: np.ndarray, camera_info: CameraInfo, x: int, y: int):
    # Depth at pixel (x, y)
    Z = depth_image[y, x]

    # Camera Matrix (K)
    fx = camera_info.k[0]  # Focal length x
    fy = camera_info.k[4]  # Focal length y
    cx = camera_info.k[2]  # Principal point x
    cy = camera_info.k[5]  # Principal point y

    # Compute real-world coordinates (in mm)
    X = ((x - cx) * Z) / fx
    Y = ((y - cy) * Z) / fy

    return X, Y, Z


class TimeSyncNode(Node):

    def __init__(self):
        super().__init__("sync_node")

        self.declare_parameter(
            name="image_sub_topic",
            value="/T8/oakd/rgb/image_raw/compressed",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
            ),
        )
        self.declare_parameter(
            name="depth_sub_topic",
            value="oakd/stereo/image_raw/compressedDepth",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
            ),
        )
        self.declare_parameter(
            name="stereo_camera_info_topic",
            value="oakd/stereo/camera_info",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
            ),
        )

        image_sub_topic = (
            self.get_parameter("image_sub_topic").get_parameter_value().string_value
        )

        depth_sub_topic = (
            self.get_parameter("depth_sub_topic").get_parameter_value().string_value
        )

        stereo_camera_info_topic = (
            self.get_parameter("stereo_camera_info_topic")
            .get_parameter_value()
            .string_value
        )

        self.get_logger().info(f"{image_sub_topic=}")
        self.get_logger().info(f"{depth_sub_topic=}")
        self.get_logger().info(f"{stereo_camera_info_topic=}")

        self.image_sub = Subscriber(
            self, CompressedImage, image_sub_topic, qos_profile=qos_profile_sensor_data
        )
        # self.depth_image_sub = Subscriber(
        #     self, CompressedImage, depth_sub_topic, qos_profile=qos_profile_sensor_data
        # )
        # self.stereo_info_sub = Subscriber(
        #     self,
        #     CameraInfo,
        #     stereo_camera_info_topic,
        #     qos_profile=qos_profile_sensor_data,
        # )

        self.cv_bridge = CvBridge()

        queue_size = 100
        max_delay = 0.1
        self.time_sync = ApproximateTimeSynchronizer(
            [
                self.image_sub,
                #self.depth_image_sub,
                #self.stereo_info_sub,
            ],
            queue_size,
            max_delay,
        )
        self.time_sync.registerCallback(self.SyncCallback)

    def SyncCallback(
        self,
        image_msg: CompressedImage,
        #depth_image_msg: CompressedImage,
        #depth_camera_info_mg: CameraInfo,
    ):
        image = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg)

        # Convert compressed depth image to numpy array
        #depth_image = convert_compressedDepth_to_cv2(depth_image_msg)  # / 1000.0

        # Get image height and width
        height, width = image.shape[:2]

        # Take centre of the image
        center_x = width // 2
        center_y = height // 2

        # Using the Camera parameters and depth image caluclate the real world coordinates
        #X, Y, Z = pixels_to_mm(depth_image, depth_camera_info_mg, center_x, center_y)

        # self.get_logger().info(
        #     f"Position of Center pixel ({center_x}, {center_y}): ({X} mm, {Y} mm, {Z} mm)"
        # )

        cv2.imshow("Image", image)
        #cv2.imshow("Depth", depth_image)
        cv2.waitKey(10)


def main(args=None):
    rclpy.init(args=args)

    time_sync = TimeSyncNode()

    rclpy.spin(time_sync)

    time_sync.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

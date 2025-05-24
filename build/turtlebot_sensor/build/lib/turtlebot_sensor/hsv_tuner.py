import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import qos_profile_sensor_data

class HSVTuner(Node):
    def __init__(self):
        super().__init__('hsv_tuner')
        
        # ROS topic to subscribe to
        self.image_topic = "/T26/oakd/rgb/image_raw/compressed"
        self.get_logger().info(f"Subscribing to topic: {self.image_topic}")
        
        # Create subscriber
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )
        
        self.cv_bridge = CvBridge()
        
        # Create a window for the trackbars
        cv2.namedWindow("HSV Trackbars")
        
        # Initial HSV values for gold detection
        self.hue_min = 20
        self.hue_max = 40
        self.sat_min = 100
        self.sat_max = 255
        self.val_min = 100
        self.val_max = 255
        
        # Create trackbars
        cv2.createTrackbar("Hue Min", "HSV Trackbars", self.hue_min, 179, self.on_hue_min_change)
        cv2.createTrackbar("Hue Max", "HSV Trackbars", self.hue_max, 179, self.on_hue_max_change)
        cv2.createTrackbar("Sat Min", "HSV Trackbars", self.sat_min, 255, self.on_sat_min_change)
        cv2.createTrackbar("Sat Max", "HSV Trackbars", self.sat_max, 255, self.on_sat_max_change)
        cv2.createTrackbar("Val Min", "HSV Trackbars", self.val_min, 255, self.on_val_min_change)
        cv2.createTrackbar("Val Max", "HSV Trackbars", self.val_max, 255, self.on_val_max_change)
        
        # Store the latest image
        self.latest_image = None
        self.get_logger().info("HSV Tuner initialized. Waiting for images...")

    def on_hue_min_change(self, value):
        self.hue_min = value
        self.update_mask()
        
    def on_hue_max_change(self, value):
        self.hue_max = value
        self.update_mask()
        
    def on_sat_min_change(self, value):
        self.sat_min = value
        self.update_mask()
        
    def on_sat_max_change(self, value):
        self.sat_max = value
        self.update_mask()
        
    def on_val_min_change(self, value):
        self.val_min = value
        self.update_mask()
        
    def on_val_max_change(self, value):
        self.val_max = value
        self.update_mask()
        
    def update_mask(self):
        if self.latest_image is not None:
            self.process_image(self.latest_image)
            
    def image_callback(self, msg):
        try:
            image = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
            self.latest_image = image
            self.process_image(image)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
            
    def process_image(self, image):
        # Create a copy for display
        display_image = image.copy()
        
        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask using current HSV values
        lower_bound = np.array([self.hue_min, self.sat_min, self.val_min])
        upper_bound = np.array([self.hue_max, self.sat_max, self.val_max])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        
        # Apply mask to show what's being detected
        result = cv2.bitwise_and(image, image, mask=mask)
        
        # Log current values
        self.get_logger().info(f"HSV values: H({self.hue_min}-{self.hue_max}), S({self.sat_min}-{self.sat_max}), V({self.val_min}-{self.val_max})")
        
        # Add text with current values on the display image
        cv2.putText(display_image, f"H: {self.hue_min}-{self.hue_max}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_image, f"S: {self.sat_min}-{self.sat_max}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_image, f"V: {self.val_min}-{self.val_max}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display results
        cv2.imshow("Original", display_image)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    tuner = HSVTuner()
    
    try:
        rclpy.spin(tuner)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        cv2.destroyAllWindows()
        tuner.destroy_node()
        rclpy.shutdown()
        
    # Print final values for easy copy-paste
    print("\nFinal HSV Values:")
    print(f"hue_min: {tuner.hue_min}")
    print(f"hue_max: {tuner.hue_max}")
    print(f"sat_min: {tuner.sat_min}")
    print(f"sat_max: {tuner.sat_max}")
    print(f"val_min: {tuner.val_min}")
    print(f"val_max: {tuner.val_max}")

if __name__ == '__main__':
    main()
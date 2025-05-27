[33mcommit 3a77432e0755c9183e0c35ed9a34f7ab000c3acf[m[33m ([m[1;36mHEAD -> [m[1;32mmaster[m[33m)[m
Author: James Zeng <jzen379@en441503.uoa.auckland.ac.nz>
Date:   Mon May 26 20:06:48 2025 +1200

    misc: update file name

[1mdiff --git a/build/py_pubsub/prefix_override/__pycache__/sitecustomize.cpython-310.pyc b/build/py_pubsub/prefix_override/__pycache__/sitecustomize.cpython-310.pyc[m
[1mindex 5c62e18..9c7d25b 100644[m
Binary files a/build/py_pubsub/prefix_override/__pycache__/sitecustomize.cpython-310.pyc and b/build/py_pubsub/prefix_override/__pycache__/sitecustomize.cpython-310.pyc differ
[1mdiff --git a/src/turtlebot_sensor/turtlebot_sensor/sync_node.py b/build/turtlebot_sensor/build/lib/turtlebot_sensor/cube_detection_node.py[m
[1mold mode 100755[m
[1mnew mode 100644[m
[1msimilarity index 99%[m
[1mrename from src/turtlebot_sensor/turtlebot_sensor/sync_node.py[m
[1mrename to build/turtlebot_sensor/build/lib/turtlebot_sensor/cube_detection_node.py[m
[1mindex f456489..ccfda66[m
[1m--- a/src/turtlebot_sensor/turtlebot_sensor/sync_node.py[m
[1m+++ b/build/turtlebot_sensor/build/lib/turtlebot_sensor/cube_detection_node.py[m
[36m@@ -93,7 +93,7 @@[m [mclass CubeDetectionNode(Node):[m
         self.cube_marker_pub = self.create_publisher(Marker, "~/cube_marker", 10)[m
 [m
         # --- TF2 ---[m
[31m-        self.tf_buffer = tf2_ros.Buffer()[m
[32m+[m[32m        self.tf_buffer = tf2_ros.Buffer()[m[41m   [m
         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)[m
 [m
         # --- Other Member Variables ---[m
[36m@@ -113,6 +113,8 @@[m [mclass CubeDetectionNode(Node):[m
         self.published_marker_ids = set()[m
         self.next_marker_id = 0[m
 [m
[32m+[m[41m        [m
[32m+[m
     def _declare_and_load_params(self):[m
         # Helper to keep __init__ cleaner[m
         self.declare_parameter("image_sub_topic", f"/T{self.robot_id_str}/oakd/rgb/image_raw/compressed", ParameterDescriptor(type=ParameterType.PARAMETER_STRING))[m
[1mdiff --git a/build/turtlebot_sensor/install.log b/build/turtlebot_sensor/install.log[m
[1mindex 873b610..1ac5553 100644[m
[1m--- a/build/turtlebot_sensor/install.log[m
[1m+++ b/build/turtlebot_sensor/install.log[m
[36m@@ -1,22 +1,22 @@[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/__init__.py[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/explorer.py[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/hsv_tuner.py[m
[31m-/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/sync_node.py[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/test_explorer.py[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/plan_node.py[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/golden_cube_mission_node.py[m
[32m+[m[32m/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/cube_detection_node.py[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/__pycache__/__init__.cpython-310.pyc[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/__pycache__/explorer.cpython-310.pyc[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/__pycache__/hsv_tuner.cpython-310.pyc[m
[31m-/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/__pycache__/sync_node.cpython-310.pyc[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/__pycache__/test_explorer.cpython-310.pyc[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/__pycache__/plan_node.cpython-310.pyc[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/__pycache__/golden_cube_mission_node.cpython-310.pyc[m
[32m+[m[32m/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/__pycache__/cube_detection_node.cpython-310.pyc[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/share/ament_index/resource_index/packages/turtlebot_sensor[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/share/turtlebot_sensor/package.xml[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/share/turtlebot_sensor/launch/mapping.launch.py[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/share/turtlebot_sensor/launch/slam.launch.py[m
[31m-/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/share/turtlebot_sensor/launch/sync.launch.py[m
[32m+[m[32m/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/share/turtlebot_sensor/launch/detection.launch.py[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/share/turtlebot_sensor/launch/navigation.launch.py[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor-0.0.0-py3.10.egg-info/PKG-INFO[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor-0.0.0-py3.10.egg-info/SOURCES.txt[m
[36m@@ -25,7 +25,7 @@[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor-0.0.0-py3.10.egg-info/requires.txt[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor-0.0.0-py3.10.egg-info/top_level.txt[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor-0.0.0-py3.10.egg-info/zip-safe[m
[32m+[m[32m/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/turtlebot_sensor/cube_detection_node[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/turtlebot_sensor/explorer[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/turtlebot_sensor/plan_node[m
[31m-/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/turtlebot_sensor/sync_node[m
 /afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/turtlebot_sensor/value_test[m
[1mdiff --git a/build/turtlebot_sensor/prefix_override/__pycache__/sitecustomize.cpython-310.pyc b/build/turtlebot_sensor/prefix_override/__pycache__/sitecustomize.cpython-310.pyc[m
[1mindex 8684bb5..3d8327f 100644[m
Binary files a/build/turtlebot_sensor/prefix_override/__pycache__/sitecustomize.cpython-310.pyc and b/build/turtlebot_sensor/prefix_override/__pycache__/sitecustomize.cpython-310.pyc differ
[1mdiff --git a/build/turtlebot_sensor/turtlebot_sensor.egg-info/SOURCES.txt b/build/turtlebot_sensor/turtlebot_sensor.egg-info/SOURCES.txt[m
[1mindex 682f50a..17149a9 100644[m
[1m--- a/build/turtlebot_sensor/turtlebot_sensor.egg-info/SOURCES.txt[m
[1m+++ b/build/turtlebot_sensor/turtlebot_sensor.egg-info/SOURCES.txt[m
[36m@@ -9,18 +9,18 @@[m [msetup.py[m
 ../../build/turtlebot_sensor/turtlebot_sensor.egg-info/requires.txt[m
 ../../build/turtlebot_sensor/turtlebot_sensor.egg-info/top_level.txt[m
 ../../build/turtlebot_sensor/turtlebot_sensor.egg-info/zip-safe[m
[32m+[m[32mlaunch/detection.launch.py[m
 launch/mapping.launch.py[m
 launch/navigation.launch.py[m
 launch/slam.launch.py[m
[31m-launch/sync.launch.py[m
 resource/turtlebot_sensor[m
 test/test_copyright.py[m
 test/test_flake8.py[m
 test/test_pep257.py[m
 turtlebot_sensor/__init__.py[m
[32m+[m[32mturtlebot_sensor/cube_detection_node.py[m
 turtlebot_sensor/explorer.py[m
 turtlebot_sensor/golden_cube_mission_node.py[m
 turtlebot_sensor/hsv_tuner.py[m
 turtlebot_sensor/plan_node.py[m
[31m-turtlebot_sensor/sync_node.py[m
 turtlebot_sensor/test_explorer.py[m
\ No newline at end of file[m
[1mdiff --git a/build/turtlebot_sensor/turtlebot_sensor.egg-info/entry_points.txt b/build/turtlebot_sensor/turtlebot_sensor.egg-info/entry_points.txt[m
[1mindex 5989312..6ec7fb3 100644[m
[1m--- a/build/turtlebot_sensor/turtlebot_sensor.egg-info/entry_points.txt[m
[1m+++ b/build/turtlebot_sensor/turtlebot_sensor.egg-info/entry_points.txt[m
[36m@@ -1,6 +1,6 @@[m
 [console_scripts][m
[32m+[m[32mcube_detection_node = turtlebot_sensor.cube_detection_node:main[m
 explorer = turtlebot_sensor.explorer:main[m
 plan_node = turtlebot_sensor.plan_node:main[m
[31m-sync_node = turtlebot_sensor.sync_node:main[m
 value_test = turtlebot_sensor.hsv_tuner:main[m
 [m
[1mdiff --git a/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor-0.0.0-py3.10.egg-info/SOURCES.txt b/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor-0.0.0-py3.10.egg-info/SOURCES.txt[m
[1mindex 682f50a..17149a9 100644[m
[1m--- a/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor-0.0.0-py3.10.egg-info/SOURCES.txt[m
[1m+++ b/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor-0.0.0-py3.10.egg-info/SOURCES.txt[m
[36m@@ -9,18 +9,18 @@[m [msetup.py[m
 ../../build/turtlebot_sensor/turtlebot_sensor.egg-info/requires.txt[m
 ../../build/turtlebot_sensor/turtlebot_sensor.egg-info/top_level.txt[m
 ../../build/turtlebot_sensor/turtlebot_sensor.egg-info/zip-safe[m
[32m+[m[32mlaunch/detection.launch.py[m
 launch/mapping.launch.py[m
 launch/navigation.launch.py[m
 launch/slam.launch.py[m
[31m-launch/sync.launch.py[m
 resource/turtlebot_sensor[m
 test/test_copyright.py[m
 test/test_flake8.py[m
 test/test_pep257.py[m
 turtlebot_sensor/__init__.py[m
[32m+[m[32mturtlebot_sensor/cube_detection_node.py[m
 turtlebot_sensor/explorer.py[m
 turtlebot_sensor/golden_cube_mission_node.py[m
 turtlebot_sensor/hsv_tuner.py[m
 turtlebot_sensor/plan_node.py[m
[31m-turtlebot_sensor/sync_node.py[m
 turtlebot_sensor/test_explorer.py[m
\ No newline at end of file[m
[1mdiff --git a/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor-0.0.0-py3.10.egg-info/entry_points.txt b/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor-0.0.0-py3.10.egg-info/entry_points.txt[m
[1mindex 5989312..6ec7fb3 100644[m
[1m--- a/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor-0.0.0-py3.10.egg-info/entry_points.txt[m
[1m+++ b/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor-0.0.0-py3.10.egg-info/entry_points.txt[m
[36m@@ -1,6 +1,6 @@[m
 [console_scripts][m
[32m+[m[32mcube_detection_node = turtlebot_sensor.cube_detection_node:main[m
 explorer = turtlebot_sensor.explorer:main[m
 plan_node = turtlebot_sensor.plan_node:main[m
[31m-sync_node = turtlebot_sensor.sync_node:main[m
 value_test = turtlebot_sensor.hsv_tuner:main[m
 [m
[1mdiff --git a/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/__pycache__/cube_detection_node.cpython-310.pyc b/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/__pycache__/cube_detection_node.cpython-310.pyc[m
[1mnew file mode 100644[m
[1mindex 0000000..6867087[m
Binary files /dev/null and b/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/__pycache__/cube_detection_node.cpython-310.pyc differ
[1mdiff --git a/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/cube_detection_node.py b/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/cube_detection_node.py[m
[1mnew file mode 100644[m
[1mindex 0000000..ccfda66[m
[1m--- /dev/null[m
[1m+++ b/install/turtlebot_sensor/lib/python3.10/site-packages/turtlebot_sensor/cube_detection_node.py[m
[36m@@ -0,0 +1,599 @@[m
[32m+[m[32mimport cv2[m
[32m+[m[32mimport rclpy[m
[32m+[m[32mimport os[m
[32m+[m[32mimport numpy as np[m
[32m+[m[32mimport math[m
[32m+[m[32mfrom cv_bridge import CvBridge, CvBridgeError[m
[32m+[m[32mfrom rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult[m
[32m+[m[32mfrom rclpy.node import Node[m
[32m+[m[32mfrom rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy[m
[32m+[m[32mfrom sensor_msgs.msg import CompressedImage, CameraInfo, Image[m
[32m+[m[32mfrom visualization_msgs.msg import Marker[m
[32m+[m[32mfrom geometry_msgs.msg import PointStamped[m
[32m+[m[32mimport tf2_ros[m
[32m+[m[32mimport tf2_geometry_msgs # Registers PointStamped transforms[m
[32m+[m[32mfrom tf2_ros import LookupException, ConnectivityException, ExtrapolationException[m
[32m+[m[32mimport traceback[m
[32m+[m[32mfrom collections import deque # For detection_buffer, slightly more efficient pops from left[m
[32m+[m
[32m+[m[32mclass CubeDetectionNode(Node):[m
[32m+[m[32m    def __init__(self):[m
[32m+[m[32m        super().__init__("cube_detection_node")[m
[32m+[m[32m        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID')[m
[32m+[m
[32m+[m[32m        if not self.robot_id_str:[m
[32m+[m[32m            self.get_logger().warning("ROS_DOMAIN_ID not set, using default '0'.")[m
[32m+[m[32m            self.robot_id_str = "0"[m
[32m+[m
[32m+[m[32m        # --- Cached Parameters ---[m
[32m+[m[32m        # (Declare them first, then get initial values, then set up callback)[m
[32m+[m[32m        # Image/Camera Topics[m
[32m+[m[32m        self.param_image_sub_topic: str = ""[m
[32m+[m[32m        self.param_camera_info_sub_topic: str = ""[m
[32m+[m[32m        # Cube Properties[m
[32m+[m[32m        self.param_cube_physical_width_m: float = 0.0[m
[32m+[m[32m        # HSV[m
[32m+[m[32m        self.param_hue_min: int = 0[m
[32m+[m[32m        self.param_hue_max: int = 0[m
[32m+[m[32m        self.param_sat_min: int = 0[m
[32m+[m[32m        self.param_sat_max: int = 0[m
[32m+[m[32m        self.param_val_min: int = 0[m
[32m+[m[32m        self.param_val_max: int = 0[m
[32m+[m[32m        self._lower_hsv_bound: np.ndarray = np.array([0,0,0]) # Cached numpy array[m
[32m+[m[32m        self._upper_hsv_bound: np.ndarray = np.array([255,255,255]) # Cached numpy array[m
[32m+[m[32m        # Contour[m
[32m+[m[32m        self.param_min_contour_area: int = 0[m
[32m+[m[32m        self.param_max_contour_area: int = 0[m
[32m+[m[32m        self.param_epsilon_factor: float = 0.0[m
[32m+[m[32m        # Temporal Smoothing[m
[32m+[m[32m        self.param_temporal_buffer_size: int = 0[m
[32m+[m[32m        self.param_min_consistent_detections: int = 0[m
[32m+[m[32m        self.param_confidence_threshold: float = 0.0[m
[32m+[m[32m        # Debug Display[m
[32m+[m[32m        self.param_publish_debug_image: bool = False[m
[32m+[m[32m        self.param_use_cv_imshow_debug: bool = False[m
[32m+[m[32m        self.param_debug_display_every_n_frames: int = 0[m
[32m+[m[32m        # TF and Marker[m
[32m+[m[32m        self.param_camera_optical_frame_id: str = ""[m
[32m+[m[32m        self.param_target_map_frame_id: str = ""[m
[32m+[m[32m        self.param_publish_rviz_marker: bool = False[m
[32m+[m
[32m+[m[32m        self._declare_and_load_params() # Helper to declare and load initial values[m
[32m+[m[32m        self.add_on_set_parameters_callback(self.parameters_callback) # To update cached params[m
[32m+[m
[32m+[m[32m        # --- Subscriptions ---[m
[32m+[m[32m        self.get_logger().info(f"Subscribing to image topic: {self.param_image_sub_topic}")[m
[32m+[m[32m        self.get_logger().info(f"Subscribing to camera info topic: {self.param_camera_info_sub_topic}")[m
[32m+[m
[32m+[m[32m        self.image_sub = self.create_subscription([m
[32m+[m[32m            CompressedImage,[m
[32m+[m[32m            self.param_image_sub_topic, # Use cached value[m
[32m+[m[32m            self.image_callback,[m
[32m+[m[32m            qos_profile_sensor_data[m
[32m+[m[32m        )[m
[32m+[m
[32m+[m[32m        self.camera_info_received = False[m
[32m+[m[32m        self.fx, self.fy, self.cx, self.cy = None, None, None, None[m
[32m+[m[32m        camera_info_qos = QoSProfile([m
[32m+[m[32m            reliability=ReliabilityPolicy.RELIABLE,[m
[32m+[m[32m            durability=DurabilityPolicy.VOLATILE, # Changed from TRANSIENT_LOCAL[m
[32m+[m[32m            history=HistoryPolicy.KEEP_LAST,[m
[32m+[m[32m            depth=1 # Only need the latest[m
[32m+[m[32m        )[m
[32m+[m[32m        self.camera_info_sub = self.create_subscription([m
[32m+[m[32m            CameraInfo,[m
[32m+[m[32m            self.param_camera_info_sub_topic, # Use cached value[m
[32m+[m[32m            self.camera_info_callback,[m
[32m+[m[32m            camera_info_qos[m
[32m+[m[32m        )[m
[32m+[m
[32m+[m[32m        # --- Publishers ---[m
[32m+[m[32m        self.debug_image_pub = self.create_publisher(Image, "~/debug_image/processed", 10)[m
[32m+[m[32m        self.debug_mask_pub = self.create_publisher(Image, "~/debug_image/mask", 10)[m
[32m+[m[32m        self.cube_marker_pub = self.create_publisher(Marker, "~/cube_marker", 10)[m
[32m+[m
[32m+[m[32m        # --- TF2 ---[m
[32m+[m[32m        self.tf_buffer = tf2_ros.Buffer()[m[41m   [m
[32m+[m[32m        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)[m
[32m+[m
[32m+[m[32m        # --- Other Member Variables ---[m
[32m+[m[32m        self.cv_bridge = CvBridge()[m
[32m+[m[32m        self.detection_history = deque(maxlen=self.param_temporal_buffer_size) # Use deque[m
[32m+[m[32m        self.max_history = 5 # This seems different from temporal_buffer_size, clarify if needed[m
[32m+[m[32m        self.detection_buffer = deque(maxlen=self.param_temporal_buffer_size) # Use deque[m
[32m+[m[32m        self.last_valid_detection = None[m
[32m+[m[32m        self.frame_count_for_display = 0[m
[32m+[m[32m        self.morph_kernel = np.ones((3, 3), np.uint8) # Pre-create kernel[m
[32m+[m
[32m+[m[32m        self.get_logger().info("Cube detection node initialized. Waiting for camera info and images...")[m
[32m+[m
[32m+[m[32m        # Marker tracking (seems fine)[m
[32m+[m[32m        self.last_published_marker_position = None[m
[32m+[m[32m        self.marker_position_threshold = 0.05[m
[32m+[m[32m        self.published_marker_ids = set()[m
[32m+[m[32m        self.next_marker_id = 0[m
[32m+[m
[32m+[m[41m        [m
[32m+[m
[32m+[m[32m    def _declare_and_load_params(self):[m
[32m+[m[32m        # Helper to keep __init__ cleaner[m
[32m+[m[32m        self.declare_parameter("image_sub_topic", f"/T{self.robot_id_str}/oakd/rgb/image_raw/compressed", ParameterDescriptor(type=ParameterType.PARAMETER_STRING))[m
[32m+[m[32m        self.declare_parameter("camera_info_sub_topic", f"/T{self.robot_id_str}/oakd/rgb/camera_info", ParameterDescriptor(type=ParameterType.PARAMETER_STRING))[m
[32m+[m[32m        self.declare_parameter("cube_physical_width_m", 0.25, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))[m
[32m+[m[32m        self.declare_parameter("hue_min", 15, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))[m
[32m+[m[32m        self.declare_parameter("hue_max", 39, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))[m
[32m+[m[32m        self.declare_parameter("sat_min", 90, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))[m
[32m+[m[32m        self.declare_parameter("sat_max", 211, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))[m
[32m+[m[32m        self.declare_parameter("val_min", 123, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))[m
[32m+[m[32m        self.declare_parameter("val_max", 255, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))[m
[32m+[m[32m        self.declare_parameter("min_contour_area", 500, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))[m
[32m+[m[32m        self.declare_parameter("max_contour_area", 30000, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))[m
[32m+[m[32m        self.declare_parameter("epsilon_factor", 0.02, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))[m
[32m+[m[32m        self.declare_parameter("temporal_buffer_size", 4, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))[m
[32m+[m[32m        self.declare_parameter("min_consistent_detections", 2, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))[m
[32m+[m[32m        self.declare_parameter("confidence_threshold", 30.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))[m
[32m+[m[32m        self.declare_parameter("publish_debug_image", True, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))[m
[32m+[m[32m        self.declare_parameter("use_cv_imshow_debug", True, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))[m
[32m+[m[32m        self.declare_parameter("debug_display_every_n_frames", 5, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))[m
[32m+[m[32m        self.declare_parameter("camera_optical_frame_id", "oakd_rgb_camera_optical_frame", ParameterDescriptor(type=ParameterType.PARAMETER_STRING))[m
[32m+[m[32m        self.declare_parameter("target_map_frame_id", "base_link", ParameterDescriptor(type=ParameterType.PARAMETER_STRING))[m
[32m+[m[32m        self.declare_parameter("publish_rviz_marker", True, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))[m
[32m+[m
[32m+[m[32m        # Load initial values[m
[32m+[m[32m        self.param_image_sub_topic = self.get_parameter("image_sub_topic").get_parameter_value().string_value[m
[32m+[m[32m        self.param_camera_info_sub_topic = self.get_parameter("camera_info_sub_topic").get_parameter_value().string_value[m
[32m+[m[32m        self.param_cube_physical_width_m = self.get_parameter("cube_physical_width_m").get_parameter_value().double_value[m
[32m+[m[32m        self.param_hue_min = self.get_parameter("hue_min").get_parameter_value().integer_value[m
[32m+[m[32m        self.param_hue_max = self.get_parameter("hue_max").get_parameter_value().integer_value[m
[32m+[m[32m        self.param_sat_min = self.get_parameter("sat_min").get_parameter_value().integer_value[m
[32m+[m[32m        self.param_sat_max = self.get_parameter("sat_max").get_parameter_value().integer_value[m
[32m+[m[32m        self.param_val_min = self.get_parameter("val_min").get_parameter_value().integer_value[m
[32m+[m[32m        self.param_val_max = self.get_parameter("val_max").get_parameter_value().integer_value[m
[32m+[m[32m        self._update_hsv_bounds() # Update numpy arrays[m
[32m+[m[32m        self.param_min_contour_area = self.get_parameter("min_contour_area").get_parameter_value().integer_value[m
[32m+[m[32m        self.param_max_contour_area = self.get_parameter("max_contour_area").get_parameter_value().integer_value[m
[32m+[m[32m        self.param_epsilon_factor = self.get_parameter("epsilon_factor").get_parameter_value().double_value[m
[32m+[m[32m        self.param_temporal_buffer_size = self.get_parameter("temporal_buffer_size").get_parameter_value().integer_value[m
[32m+[m[32m        self.param_min_consistent_detections = self.get_parameter("min_consistent_detections").get_parameter_value().integer_value[m
[32m+[m[32m        self.param_confidence_threshold = self.get_parameter("confidence_threshold").get_parameter_value().double_value[m
[32m+[m[32m        self.param_publish_debug_image = self.get_parameter("publish_debug_image").get_parameter_value().bool_value[m
[32m+[m[32m        self.param_use_cv_imshow_debug = self.get_parameter("use_cv_imshow_debug").get_parameter_value().bool_value[m
[32m+[m[32m        self.param_debug_display_every_n_frames = self.get_parameter("debug_display_every_n_frames").get_parameter_value().integer_value[m
[32m+[m[32m        self.param_camera_optical_frame_id = self.get_parameter("camera_optical_frame_id").get_parameter_value().string_value[m
[32m+[m[32m        self.param_target_map_frame_id = self.get_parameter("target_map_frame_id").get_parameter_value().string_value[m
[32m+[m[32m        self.param_publish_rviz_marker = self.get_parameter("publish_rviz_marker").get_parameter_value().bool_value[m
[32m+[m
[32m+[m[32m        # Update deque maxlen if params change[m
[32m+[m[32m        if hasattr(self, 'detection_buffer') and self.detection_buffer.maxlen != self.param_temporal_buffer_size:[m
[32m+[m[32m            self.detection_buffer = deque(list(self.detection_buffer), maxlen=self.param_temporal_buffer_size)[m
[32m+[m[32m            self.detection_history = deque(list(self.detection_history), maxlen=self.param_temporal_buffer_size) # Assuming same size for history[m
[32m+[m
[32m+[m
[32m+[m[32m    def _update_hsv_bounds(self):[m
[32m+[m[32m        self._lower_hsv_bound = np.array([self.param_hue_min, self.param_sat_min, self.param_val_min])[m
[32m+[m[32m        self._upper_hsv_bound = np.array([self.param_hue_max, self.param_sat_max, self.param_val_max])[m
[32m+[m
[32m+[m[32m    def parameters_callback(self, params):[m
[32m+[m[32m        # This is called when parameters are changed (e.g., by ros2 param set)[m
[32m+[m[32m        for param in params:[m
[32m+[m[32m            if param.name == "image_sub_topic": self.param_image_sub_topic = param.value[m
[32m+[m[32m            elif param.name == "camera_info_sub_topic": self.param_camera_info_sub_topic = param.value[m
[32m+[m[32m            elif param.name == "cube_physical_width_m": self.param_cube_physical_width_m = param.value[m
[32m+[m[32m            elif param.name == "hue_min": self.param_hue_min = param.value[m
[32m+[m[32m            elif param.name == "hue_max": self.param_hue_max = param.value[m
[32m+[m[32m            elif param.name == "sat_min": self.param_sat_min = param.value[m
[32m+[m[32m            elif param.name == "sat_max": self.param_sat_max = param.value[m
[32m+[m[32m            elif param.name == "val_min": self.param_val_min = param.value[m
[32m+[m[32m            elif param.name == "val_max": self.param_val_max = param.value[m
[32m+[m[32m            elif param.name == "min_contour_area": self.param_min_contour_area = param.value[m
[32m+[m[32m            elif param.name == "max_contour_area": self.param_max_contour_area = param.value[m
[32m+[m[32m            elif param.name == "epsilon_factor": self.param_epsilon_factor = param.value[m
[32m+[m[32m            elif param.name == "temporal_buffer_size":[m
[32m+[m[32m                self.param_temporal_buffer_size = param.value[m
[32m+[m[32m                # Re-initialize deques with new maxlen if it changed[m
[32m+[m[32m                if self.detection_buffer.maxlen != self.param_temporal_buffer_size:[m
[32m+[m[32m                    self.detection_buffer = deque(list(self.detection_buffer), maxlen=self.param_temporal_buffer_size)[m
[32m+[m[32m                    self.detection_history = deque(list(self.detection_history), maxlen=self.param_temporal_buffer_size) # Assuming same size[m
[32m+[m[32m            elif param.name == "min_consistent_detections": self.param_min_consistent_detections = param.value[m
[32m+[m[32m            elif param.name == "confidence_threshold": self.param_confidence_threshold = param.value[m
[32m+[m[32m            elif param.name == "publish_debug_image": self.param_publish_debug_image = param.value[m
[32m+[m[32m            elif param.name == "use_cv_imshow_debug": self.param_use_cv_imshow_debug = param.value[m
[32m+[m[32m            elif param.name == "debug_display_every_n_frames": self.param_debug_display_every_n_frames = param.value[m
[32m+[m[32m            elif param.name == "camera_optical_frame_id": self.param_camera_optical_frame_id = param.value[m
[32m+[m[32m            elif param.name == "target_map_frame_id": self.param_target_map_frame_id = param.value[m
[32m+[m[32m            elif param.name == "publish_rviz_marker": self.param_publish_rviz_marker = param.value[m
[32m+[m
[32m+[m[32m        # Update derived parameters like HSV bounds if relevant params changed[m
[32m+[m[32m        if any(p.name in ["hue_min", "hue_max", "sat_min", "sat_max", "val_min", "val_max"] for p in params):[m
[32m+[m[32m            self._update_hsv_bounds()[m
[32m+[m
[32m+[m[32m        self.get_logger().info("Parameters updated.")[m
[32m+[m[32m        return SetParametersResult(successful=True)[m
[32m+[m
[32m+[m
[32m+[m[32m    def camera_info_callback(self, msg: CameraInfo):[m
[32m+[m[32m        if not self.camera_info_received: # Process only once[m
[32m+[m[32m            self.fx = msg.k[0][m
[32m+[m[32m            self.fy = msg.k[4][m
[32m+[m[32m            self.cx = msg.k[2][m
[32m+[m[32m            self.cy = msg.k[5][m
[32m+[m[32m            self.camera_info_received = True[m
[32m+[m[32m            self.get_logger().info([m
[32m+[m[32m                f"Camera intrinsics received: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}"[m
[32m+[m[32m            )[m
[32m+[m[32m            # Unsubscribe if we only need it once and it's RELIABLE + VOLATILE[m
[32m+[m[32m            # self.destroy_subscription(self.camera_info_sub)[m
[32m+[m[32m            # self.camera_info_sub = None # Avoid issues if called multiple times[m
[32m+[m
[32m+[m[32m    def get_color_mask(self, image): # HSV params are now member variables for bounds[m
[32m+[m[32m        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[m
[32m+[m[32m        mask = cv2.inRange(hsv, self._lower_hsv_bound, self._upper_hsv_bound) # Use cached np.array[m
[32m+[m[32m        # Use pre-created kernel[m
[32m+[m[32m        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)[m
[32m+[m[32m        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=1)[m
[32m+[m[32m        return mask[m
[32m+[m
[32m+[m[32m    def is_square_like(self, contour): # epsilon_factor is now a member variable[m
[32m+[m[32m        arc_length = cv2.arcLength(contour, True)[m
[32m+[m[32m        if arc_length < 1e-3: # Avoid division by zero or tiny arc lengths[m
[32m+[m[32m            return False, None[m
[32m+[m[32m        epsilon = self.param_epsilon_factor * arc_length[m
[32m+[m[32m        approx = cv2.approxPolyDP(contour, epsilon, True)[m
[32m+[m
[32m+[m[32m        # Optimization: Check number of vertices early[m
[32m+[m[32m        if len(approx) != 4:[m
[32m+[m[32m            return False, approx # Not a quadrilateral[m
[32m+[m
[32m+[m[32m        # The rest of your square checking logic seems fine[m
[32m+[m[32m        # (side lengths, angles)[m
[32m+[m[32m        # Ensure no division by zero if avg_side or norms are very small[m
[32m+[m[32m        sides = [np.sqrt(np.sum((approx[i][0] - approx[(i + 1) % 4][0])**2)) for i in range(4)][m
[32m+[m[32m        if any(s < 1e-3 for s in sides): # Check for degenerate sides[m
[32m+[m[32m            return False, approx[m
[32m+[m[32m        avg_side = np.mean(sides)[m
[32m+[m[32m        if avg_side < 1e-3: # Avoid division by zero if average side is tiny[m
[32m+[m[32m             return False, approx[m
[32m+[m
[32m+[m[32m        side_tolerance = 0.60 # This is quite large, allowing for very non-square quads[m
[32m+[m[32m        if any(abs(s - avg_side) / avg_side > side_tolerance for s in sides):[m
[32m+[m[32m            return False, approx[m
[32m+[m
[32m+[m[32m        angles = [][m
[32m+[m[32m        angle_tolerance_degrees = 25.0[m
[32m+[m[32m        for i in range(4):[m
[32m+[m[32m            p_curr = approx[i][0].astype(float)[m
[32m+[m[32m            p_prev = approx[(i - 1 + 4) % 4][0].astype(float)[m
[32m+[m[32m            p_next = approx[(i + 1) % 4][0].astype(float)[m
[32m+[m[32m            v1, v2 = p_prev - p_curr, p_next - p_curr[m
[32m+[m[32m            norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)[m
[32m+[m[32m            if norm_v1 < 1e-6 or norm_v2 < 1e-6: # Increased tolerance slightly for safety[m
[32m+[m[32m                return False, approx[m
[32m+[m[32m            cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)[m
[32m+[m[32m            angles.append(math.degrees(math.acos(cos_theta)))[m
[32m+[m
[32m+[m[32m        if any(not ((90.0 - angle_tolerance_degrees) <= angle <= (90.0 + angle_tolerance_degrees)) for angle in angles):[m
[32m+[m[32m            return False, approx[m
[32m+[m[32m        return True, approx[m
[32m+[m
[32m+[m[32m    def detect_cubes(self, image): # Pass only image, other params are members[m
[32m+[m[32m        # min_area, max_area, confidence_thresh_val are now self.param_...[m
[32m+[m[32m        color_mask = self.get_color_mask(image) # Uses cached HSV bounds[m
[32m+[m[32m        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[m
[32m+[m
[32m+[m[32m        cube_candidates = [][m
[32m+[m[32m        for contour in contours:[m
[32m+[m[32m            area = cv2.contourArea(contour)[m
[32m+[m[32m            # Optimization: Check area before more expensive checks[m
[32m+[m[32m            if not (self.param_min_contour_area <= area <= self.param_max_contour_area):[m
[32m+[m[32m                continue[m
[32m+[m
[32m+[m[32m            is_square, approx_shape = self.is_square_like(contour) # Uses cached epsilon[m
[32m+[m[32m            if is_square and approx_shape is not None and len(approx_shape) == 4: # Ensure approx_shape is valid[m
[32m+[m[32m                M = cv2.moments(contour)[m
[32m+[m[32m                if M["m00"] > 1e-5: # Avoid division by zero[m
[32m+[m[32m                    cx_pixel = int(M["m10"] / M["m00"])[m
[32m+[m[32m                    cy_pixel = int(M["m01"] / M["m00"])[m
[32m+[m[32m                    x_r, y_r, w_r, h_r = cv2.boundingRect(contour)[m
[32m+[m[32m                    aspect_ratio = float(w_r) / h_r if h_r > 0 else 0 # Avoid division by zero[m
[32m+[m
[32m+[m[32m                    # Looser aspect ratio for "square-like" might be fine if is_square_like is robust[m
[32m+[m[32m                    if 0.7 <= aspect_ratio <= 1.4: # This is a common check[m
[32m+[m[32m                        confidence = self.calculate_confidence(contour, area, aspect_ratio, approx_shape)[m
[32m+[m[32m                        if confidence >= self.param_confidence_threshold:[m
[32m+[m[32m                            cube_candidates.append({[m
[32m+[m[32m                                'contour': contour,[m
[32m+[m[32m                                'approx': approx_shape,[m
[32m+[m[32m                                'center': (cx_pixel, cy_pixel),[m
[32m+[m[32m                                'area': area,[m
[32m+[m[32m                                'bbox': (x_r, y_r, w_r, h_r),[m
[32m+[m[32m                                'aspect_ratio': aspect_ratio,[m
[32m+[m[32m                                'confidence': confidence[m
[32m+[m[32m                            })[m
[32m+[m[32m        return cube_candidates, color_mask[m
[32m+[m
[32m+[m[32m    def calculate_confidence(self, contour, area, aspect_ratio, approx):[m
[32m+[m[32m        # This logic is specific to your needs. Minor optimization:[m
[32m+[m[32m        # Avoid re-calculating perimeter if not needed or cache intermediate results[m
[32m+[m[32m        # if computationally intensive. For now, it looks okay.[m
[32m+[m[32m        confidence = 100.0[m
[32m+[m[32m        confidence -= abs(aspect_ratio - 1.0) * 50.0 # Ensure float math[m
[32m+[m
[32m+[m[32m        # Area penalties/bonuses[m
[32m+[m[32m        if area < 800: # Using literals, could be params[m
[32m+[m[32m            confidence -= (800.0 - area) / 20.0[m
[32m+[m[32m        elif area > 20000:[m
[32m+[m[32m            confidence -= (area - 20000.0) / 200.0[m
[32m+[m[32m        elif 1000 <= area <= 5000: # Bonus for "ideal" size[m
[32m+[m[32m            confidence += 10.0[m
[32m+[m
[32m+[m[32m        # Perimeter difference penalty[m
[32m+[m[32m        perimeter = cv2.arcLength(contour, True)[m
[32m+[m[32m        if perimeter > 1e-3: # Avoid division by zero[m
[32m+[m[32m            approx_perimeter = cv2.arcLength(approx, True)[m
[32m+[m[32m            confidence -= abs(perimeter - approx_perimeter) / perimeter * 30.0[m
[32m+[m
[32m+[m[32m        return max(0.0, min(100.0, confidence)) # Clamp confidence[m
[32m+[m
[32m+[m[32m    def draw_detections(self, image, cube_candidates):[m
[32m+[m[32m        # This function is mostly for debugging. If not publishing debug image,[m
[32m+[m[32m        # it might not need to be called or could be simplified.[m
[32m+[m[32m        # No major performance bottlenecks here other than the drawing itself.[m
[32m+[m[32m        result_image = image # Operate on the copy passed in[m
[32m+[m[32m        for cube in cube_candidates: # Removed index 'i' as it wasn't used[m
[32m+[m[32m            cv2.drawContours(result_image, [cube['contour']], -1, (0, 255, 0), 2)[m
[32m+[m[32m            cv2.circle(result_image, cube['center'], 5, (0, 0, 255), -1)[m
[32m+[m[32m            x, y, w, h = cube['bbox'][m
[32m+[m[32m            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 255, 0), 2)[m
[32m+[m[32m            text = f"Cf:{cube.get('confidence', 0.0):.0f}"[m
[32m+[m[32m            cv2.putText(result_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)[m
[32m+[m
[32m+[m[32m            if 'position_in_camera' in cube:[m
[32m+[m[32m                pos_cam = cube['position_in_camera'][m
[32m+[m[32m                cam_text = f"Cam D: {pos_cam[2]:.2f}m"[m
[32m+[m[32m                cv2.putText(result_image, cam_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)[m
[32m+[m
[32m+[m[32m            if 'position_in_map' in cube:[m
[32m+[m[32m                pos_map = cube['position_in_map'][m
[32m+[m[32m                map_text = f"Map X:{pos_map.x:.1f} Y:{pos_map.y:.1f}"[m
[32m+[m[32m                cv2.putText(result_image, map_text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)[m
[32m+[m[32m        return result_image # Return the modified image[m
[32m+[m
[32m+[m[32m    def smooth_detections(self, current_detections):[m
[32m+[m[32m        # Uses self.detection_buffer (deque) and self.param_... for buffer size etc.[m
[32m+[m[32m        self.detection_buffer.append(len(current_detections))[m
[32m+[m[32m        # No pop needed due to deque's maxlen[m
[32m+[m
[32m+[m[32m        # Check if buffer is filled enough for consistency check[m
[32m+[m[32m        if len(self.detection_buffer) < self.param_min_consistent_detections: # or < self.param_temporal_buffer_size for full buffer[m
[32m+[m[32m            self.last_valid_detection = None[m
[32m+[m[32m            return [][m
[32m+[m
[32m+[m[32m        consistent_detection_frames_in_buffer = sum(1 for count in self.detection_buffer if count > 0)[m
[32m+[m
[32m+[m[32m        if consistent_detection_frames_in_buffer >= self.param_min_consistent_detections:[m
[32m+[m[32m            if current_detections:[m
[32m+[m[32m                # Select the best detection if multiple, or average, etc.[m
[32m+[m[32m                # For now, assuming current_detections is a list of dicts, and we might want to smooth properties of the best one.[m
[32m+[m[32m                # The current logic returns the whole list if consistent.[m
[32m+[m[32m                self.last_valid_detection = current_detections # Potentially copy() if mutable and modified later[m
[32m+[m[32m                return current_detections[m
[32m+[m[32m            elif self.last_valid_detection: # If current frame has no detections but buffer was consistent[m
[32m+[m[32m                return self.last_valid_detection[m
[32m+[m[32m            else: # No current detections, no last valid detection[m
[32m+[m[32m                return [][m
[32m+[m[32m        else: # Not enough consistent frames in buffer[m
[32m+[m[32m            self.last_valid_detection = None[m
[32m+[m[32m            return [][m
[32m+[m
[32m+[m[32m    def image_callback(self, image_msg: CompressedImage):[m
[32m+[m[32m        # Parameter Caching is now handled by member variables like self.param_hue_min etc.[m
[32m+[m
[32m+[m[32m        if not self.camera_info_received:[m
[32m+[m[32m            self.get_logger().warn("Camera info not yet received. Skipping frame.", throttle_duration_sec=5.0) # Throttle warning[m
[32m+[m[32m            return[m
[32m+[m
[32m+[m[32m        try:[m
[32m+[m[32m            image_cv = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding="bgr8")[m
[32m+[m[41m            [m
[32m+[m[32m            # --- OPTIONAL: Image Downscaling for Performance ---[m
[32m+[m[32m            # If enabled, ensure intrinsics (fx,fy,cx,cy) and detected pixel coordinates/widths[m
[32m+[m[32m            # are scaled appropriately before 3D calculations.[m
[32m+[m[32m            # E.g., if you scale image_cv by 0.5:[m
[32m+[m[32m            # image_cv = cv2.resize(image_cv, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)[m
[32m+[m[32m            # Then, self.fx * 0.5, self.cx * 0.5 etc. should be used, or scale P_pixels back up.[m
[32m+[m[32m            # For simplicity, this is not added by default.[m
[32m+[m
[32m+[m[32m            raw_detections, color_mask_for_debug = self.detect_cubes(image_cv) # Pass only image[m
[32m+[m[41m            [m
[32m+[m[32m            stable_detections = self.smooth_detections(raw_detections)[m
[32m+[m
[32m+[m[32m            processed_stable_detections = [] # Detections with 3D info[m
[32m+[m
[32m+[m[32m            if stable_detections:[m
[32m+[m[32m                # For simplicity, process the first stable detection.[m
[32m+[m[32m                # If multiple cubes are expected, this loop needs to be more sophisticated.[m
[32m+[m[32m                # Consider sorting by confidence or area if multiple stable_detections.[m
[32m+[m[32m                best_detection = stable_detections[0] # Assuming at least one[m
[32m+[m
[32m+[m[32m                # Ensure detection has the necessary keys[m
[32m+[m[32m                if 'bbox' not in best_detection or 'center' not in best_detection:[m
[32m+[m[32m                    self.get_logger().warn("Stable detection missing bbox or center. Skipping.", throttle_duration_sec=5.0)[m
[32m+[m[32m                    # Fall through to debug display logic, which will show no processed detections[m
[32m+[m[32m                else:[m
[32m+[m[32m                    P_pixels = float(best_detection['bbox'][2]) # Width of bounding box[m
[32m+[m
[32m+[m[32m                    # Ensure fx, fy, cx, cy are valid and P_pixels is reasonable[m
[32m+[m[32m                    if self.fx is not None and self.fx >= 1e-3 and P_pixels >= 1.0:[m
[32m+[m[32m                        Z_camera = (self.param_cube_physical_width_m * self.fx) / P_pixels[m
[32m+[m[32m                        u_pixel, v_pixel = best_detection['center'][m
[32m+[m[32m                        X_camera = (u_pixel - self.cx) * Z_camera / self.fx[m
[32m+[m[32m                        Y_camera = (v_pixel - self.cy) * Z_camera / self.fy[m
[32m+[m
[32m+[m[32m                        best_detection['position_in_camera'] = (X_camera, Y_camera, Z_camera)[m
[32m+[m[32m                        # self.get_logger().info( # Log less frequently or on change[m
[32m+[m[32m                        #     f"Cube at cam_opt_frame: ({X_camera:.2f}, {Y_camera:.2f}, {Z_camera:.2f})m"[m
[32m+[m[32m                        # )[m
[32m+[m
[32m+[m[32m                        # --- TF Transformation ---[m
[32m+[m[32m                        transform_successful = False[m
[32m+[m[32m                        try:[m
[32m+[m[32m                            current_time = self.get_clock().now()[m
[32m+[m[32m                            source_time = image_msg.header.stamp[m
[32m+[m[32m                            time_diff_s = (current_time.nanoseconds - rclpy.time.Time.from_msg(source_time).nanoseconds) / 1e9[m
[32m+[m[41m                            [m
[32m+[m[32m                            if abs(time_diff_s) > 1.0: # If image stamp is too old or in future[m
[32m+[m[32m                                self.get_logger().warn([m
[32m+[m[32m                                    f"Image timestamp is {'old' if time_diff_s > 0 else 'in future'} ({time_diff_s:.2f}s). "[m
[32m+[m[32m                                    f"Using current time for TF lookup.", throttle_duration_sec=5.0)[m
[32m+[m[32m                                source_time_for_tf = current_time.to_msg()[m
[32m+[m[32m                            else:[m
[32m+[m[32m                                source_time_for_tf = source_time[m
[32m+[m
[32m+[m[32m                            # Check transform feasibility with a reasonable timeout[m
[32m+[m[32m                            if self.tf_buffer.can_transform([m
[32m+[m[32m                                self.param_target_map_frame_id,[m
[32m+[m[32m                                self.param_camera_optical_frame_id,[m
[32m+[m[32m                                source_time_for_tf,[m
[32m+[m[32m                                timeout=rclpy.duration.Duration(seconds=0.1) # Reduced timeout[m
[32m+[m[32m                            ):[m
[32m+[m[32m                                point_in_camera = PointStamped()[m
[32m+[m[32m                                point_in_camera.header.stamp = source_time_for_tf[m
[32m+[m[32m                                point_in_camera.header.frame_id = self.param_camera_optical_frame_id[m
[32m+[m[32m                                point_in_camera.point.x = X_camera[m
[32m+[m[32m                                point_in_camera.point.y = Y_camera[m
[32m+[m[32m                                point_in_camera.point.z = Z_camera[m
[32m+[m
[32m+[m[32m                                point_in_map = self.tf_buffer.transform([m
[32m+[m[32m                                    point_in_camera,[m
[32m+[m[32m                                    self.param_target_map_frame_id,[m
[32m+[m[32m                                    timeout=rclpy.duration.Duration(seconds=0.05) # Shorter for actual transform[m
[32m+[m[32m                                )[m
[32m+[m[32m                                best_detection['position_in_map'] = point_in_map.point[m
[32m+[m[32m                                # self.get_logger().info( # Log less frequently[m
[32m+[m[32m                                #     f"Cube at '{self.param_target_map_frame_id}' frame: "[m
[32m+[m[32m                                #     f"(X:{point_in_map.point.x:.2f}, Y:{point_in_map.point.y:.2f}, Z:{point_in_map.point.z:.2f})m"[m
[32m+[m[32m                                # )[m
[32m+[m[32m                                transform_successful = True[m
[32m+[m[32m                            else:[m
[32m+[m[32m                                self.get_logger().warn([m
[32m+[m[32m                                    f"Cannot transform from '{self.param_camera_optical_frame_id}' to "[m
[32m+[m[32m                                    f"'{self.param_target_map_frame_id}'. TF not available or timeout.",[m
[32m+[m[32m                                    throttle_duration_sec=5.0[m
[32m+[m[32m                                )[m
[32m+[m[32m                        except (LookupException, ConnectivityException, ExtrapolationException) as e:[m
[32m+[m[32m                            self.get_logger().warn([m
[32m+[m[32m                                f"TF transform error: {e}", throttle_duration_sec=5.0[m
[32m+[m[32m                            )[m
[32m+[m
[32m+[m[32m                        # --- RViz Marker Publishing ---[m
[32m+[m[32m                        if transform_successful and self.param_publish_rviz_marker:[m
[32m+[m[32m                            # Simplified marker logic for now (single updating marker)[m
[32m+[m[32m                            marker = Marker()[m
[32m+[m[32m                            marker.header.frame_id = self.param_target_map_frame_id[m
[32m+[m[32m                            marker.header.stamp = self.get_clock().now().to_msg()[m
[32m+[m[32m                            marker.ns = "cube_detector"[m
[32m+[m[32m                            marker.id = 0 # Single, updating marker for simplicity[m
[32m+[m[32m                            marker.type = Marker.CUBE[m
[32m+[m[32m                            marker.action = Marker.ADD[m
[32m+[m[32m                            marker.pose.position = best_detection['position_in_map'][m
[32m+[m[32m                            marker.pose.orientation.w = 1.0 # No orientation assumed[m
[32m+[m[32m                            marker.scale.x = self.param_cube_physical_width_m # Use actual width[m
[32m+[m[32m                            marker.scale.y = self.param_cube_physical_width_m[m
[32m+[m[32m                            marker.scale.z = self.param_cube_physical_width_m[m
[32m+[m[32m                            marker.color.r = 1.0; marker.color.g = 0.843; marker.color.b = 0.0; marker.color.a = 0.7[m
[32m+[m[32m                            marker.lifetime = rclpy.duration.Duration(seconds=1).to_msg() # Marker persists for 1s[m
[32m+[m[32m                            self.cube_marker_pub.publish(marker)[m
[32m+[m[41m                        [m
[32m+[m[32m                        processed_stable_detections.append(best_detection)[m
[32m+[m
[32m+[m[32m            elif raw_detections: # No stable detections, but raw ones found[m
[32m+[m[32m                # self.get_logger().info(f"UNSTABLE: {len(raw_detections)} raw candidate(s), 0 stable.", throttle_duration_sec=2.0)[m
[32m+[m[32m                pass[m
[32m+[m
[32m+[m
[32m+[m[32m            # --- Debug Display Logic ---[m
[32m+[m[32m            self.frame_count_for_display += 1[m
[32m+[m[32m            display_this_frame = (self.param_debug_display_every_n_frames > 0 and \[m
[32m+[m[32m                                  self.frame_count_for_display % self.param_debug_display_every_n_frames == 0)[m
[32m+[m
[32m+[m[32m            if display_this_frame:[m
[32m+[m[32m                # Create a copy for drawing only if we are going to display/publish[m
[32m+[m[32m                result_image_for_display = None[m
[32m+[m[32m                if self.param_publish_debug_image or self.param_use_cv_imshow_debug:[m
[32m+[m[32m                    result_image_for_display = self.draw_detections(image_cv.copy(), processed_stable_detections)[m
[32m+[m
[32m+[m[32m                if self.param_publish_debug_image and result_image_for_display is not None:[m
[32m+[m[32m                    try:[m
[32m+[m[32m                        self.debug_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(result_image_for_display, "bgr8"))[m
[32m+[m[32m                        if color_mask_for_debug is not None: # Ensure mask was created[m
[32m+[m[32m                            # Convert mask to BGR for Image msg if it's single channel[m
[32m+[m[32m                            if len(color_mask_for_debug.shape) == 2 or color_mask_for_debug.shape[2] == 1:[m
[32m+[m[32m                                mask_colored_for_display = cv2.cvtColor(color_mask_for_debug, cv2.COLOR_GRAY2BGR)[m
[32m+[m[32m                            else:[m
[32m+[m[32m                                mask_colored_for_display = color_mask_for_debug[m
[32m+[m[32m                            self.debug_mask_pub.publish(self.cv_bridge.cv2_to_imgmsg(mask_colored_for_display, "bgr8"))[m
[32m+[m[32m                    except CvBridgeError as e:[m
[32m+[m[32m                        self.get_logger().error(f"CvBridge Error for debug publishing: {e}")[m
[32m+[m[41m                [m
[32m+[m[32m                if self.param_use_cv_imshow_debug and result_image_for_display is not None:[m
[32m+[m[32m                    # Ensure mask is valid before trying to use it[m
[32m+[m[32m                    if color_mask_for_debug is not None:[m
[32m+[m[32m                        if len(color_mask_for_debug.shape) == 2 or color_mask_for_debug.shape[2] == 1:[m
[32m+[m[32m                             mask_colored = cv2.cvtColor(color_mask_for_debug, cv2.COLOR_GRAY2BGR)[m
[32m+[m[32m                        else:[m
[32m+[m[32m                             mask_colored = color_mask_for_debug[m
[32m+[m
[32m+[m[32m                        combined_display_scale = 0.6 # Consider making this a parameter[m
[32m+[m[32m                        h_orig, w_orig = result_image_for_display.shape[:2][m
[32m+[m[41m                        [m
[32m+[m[32m                        # Ensure scaled dimensions are at least 1x1[m
[32m+[m[32m                        scaled_w = max(1, int(w_orig * combined_display_scale))[m
[32m+[m[32m                        scaled_h = max(1, int(h_orig * combined_display_scale))[m
[32m+[m
[32m+[m[32m                        display_result_scaled = cv2.resize(result_image_for_display, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)[m
[32m+[m[32m                        display_mask_scaled = cv2.resize(mask_colored, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)[m
[32m+[m[32m                        combined_cv_display = np.hstack([display_result_scaled, display_mask_scaled])[m
[32m+[m[32m                        cv2.imshow("Cube Detection (CV_IMSHOW)", combined_cv_display)[m
[32m+[m[32m                        cv2.waitKey(1) # Essential for imshow to refresh[m
[32m+[m[32m                    else: # color_mask_for_debug was None[m
[32m+[m[32m                        cv2.imshow("Cube Detection (CV_IMSHOW) - No Mask", result_image_for_display) # Show only result[m
[32m+[m[32m                        cv2.waitKey(1)[m
[32m+[m
[32m+[m[32m            # Update history (using deque now, append is fine)[m
[32m+[m[32m            self.detection_history.append(len(processed_stable_detections))[m
[32m+[m[32m            # No pop needed due to deque's maxlen[m
[32m+[m
[32m+[m[32m        except CvBridgeError as e:[m
[32m+[m[32m            self.get_logger().error(f"CvBridge Error in image_callback: {e}")[m
[32m+[m[32m        except Exception as e:[m
[32m+[m[32m            tb_str = traceback.format_exc()[m
[32m+[m[32m            self.get_logger().error(f"Error processing image: {str(e)}\nTraceback:\n{tb_str}")[m
[32m+[m
[32m+[m
[32m+[m[32mdef main(args=None):[m
[32m+[m[32m    rclpy.init(args=args)[m
[32m+[m[32m    cube_detector = CubeDetectionNode()[m
[32m+[m[32m    try:[m
[32m+[m[32m        rclpy.spin(cube_detector)[m
[32m+[m[32m    except KeyboardInterrupt:[m
[32m+[m[32m        cube_detector.get_logger().info("Keyboard interrupt, shutting down...")[m
[32m+[m[32m    finally:[m
[32m+[m[32m        # Check if node and parameter still exist before accessing[m
[32m+[m[32m        # This can be tricky if node is already partially destroyed[m
[32m+[m[32m        try:[m
[32m+[m[32m            if rclpy.ok() and cube_detector.get_node_names() and \[m
[32m+[m[32m               cube_detector.param_use_cv_imshow_debug: # Use cached param[m
[32m+[m[32m                cv2.destroyAllWindows()[m
[32m+[m[32m        except Exception as e:[m
[32m+[m[32m            if rclpy.ok() and cube_detector.get_node_names(): # Check if node still valid[m
[32m+[m[32m                 cube_detector.get_logger().warn(f"Error during cv2.destroyAllWindows: {e}")[m
[32m+[m
[32m+[m
[32m+[m[32m        if rclpy.ok() and hasattr(cube_detector, 'destroy_node') and cube_detector.get_node_names(): # Check if node still valid[m
[32m+[m[32m            cube_detector.destroy_node()[m
[32m+[m[41m        [m
[32m+[m[32m        if rclpy.ok(): # Check if rclpy context is still valid[m
[32m+[m[32m            rclpy.shutdown()[m
[32m+[m
[32m+[m[32mif __name__ == "__main__":[m
[32m+[m[32m    main()[m
\ No newline at end of file[m
[1mdiff --git a/install/turtlebot_sensor/lib/turtlebot_sensor/cube_detection_node b/install/turtlebot_sensor/lib/turtlebot_sensor/cube_detection_node[m
[1mnew file mode 100755[m
[1mindex 0000000..413cf00[m
[1m--- /dev/null[m
[1m+++ b/install/turtlebot_sensor/lib/turtlebot_sensor/cube_detection_node[m
[36m@@ -0,0 +1,33 @@[m
[32m+[m[32m#!/usr/bin/python3[m
[32m+[m[32m# EASY-INSTALL-ENTRY-SCRIPT: 'turtlebot-sensor==0.0.0','console_scripts','cube_detection_node'[m
[32m+[m[32mimport re[m
[32m+[m[32mimport sys[m
[32m+[m
[32m+[m[32m# for compatibility with easy_install; see #2198[m
[32m+[m[32m__requires__ = 'turtlebot-sensor==0.0.0'[m
[32m+[m
[32m+[m[32mtry:[m
[32m+[m[32m    from importlib.metadata import distribution[m
[32m+[m[32mexcept ImportError:[m
[32m+[m[32m    try:[m
[32m+[m[32m        from importlib_metadata import distribution[m
[32m+[m[32m    except ImportError:[m
[32m+[m[32m        from pkg_resources import load_entry_point[m
[32m+[m
[32m+[m
[32m+[m[32mdef importlib_load_entry_point(spec, group, name):[m
[32m+[m[32m    dist_name, _, _ = spec.partition('==')[m
[32m+[m[32m    matches = ([m
[32m+[m[32m        entry_point[m
[32m+[m[32m        for entry_point in distribution(dist_name).entry_points[m
[32m+[m[32m        if entry_point.group == group and entry_point.name == name[m
[32m+[m[32m    )[m
[32m+[m[32m    return next(matches).load()[m
[32m+[m
[32m+[m
[32m+[m[32mglobals().setdefault('load_entry_point', importlib_load_entry_point)[m
[32m+[m
[32m+[m
[32m+[m[32mif __name__ == '__main__':[m
[32m+[m[32m    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])[m
[32m+[m[32m    sys.exit(load_entry_point('turtlebot-sensor==0.0.0', 'console_scripts', 'cube_detection_node')())[m
[1mdiff --git a/install/turtlebot_sensor/share/turtlebot_sensor/launch/__pycache__/detection.launch.cpython-310.pyc b/install/turtlebot_sensor/share/turtlebot_sensor/launch/__pycache__/detection.launch.cpython-310.pyc[m
[1mnew file mode 100644[m
[1mindex 0000000..b4e8550[m
Binary files /dev/null and b/install/turtlebot_sensor/share/turtlebot_sensor/launch/__pycache__/detection.launch.cpython-310.pyc differ
[1mdiff --git a/src/turtlebot_sensor/launch/sync.launch.py b/install/turtlebot_sensor/share/turtlebot_sensor/launch/detection.launch.py[m
[1msimilarity index 98%[m
[1mrename from src/turtlebot_sensor/launch/sync.launch.py[m
[1mrename to install/turtlebot_sensor/share/turtlebot_sensor/launch/detection.launch.py[m
[1mindex 4f02a64..dea6833 100644[m
[1m--- a/src/turtlebot_sensor/launch/sync.launch.py[m
[1m+++ b/install/turtlebot_sensor/share/turtlebot_sensor/launch/detection.launch.py[m
[36m@@ -77,7 +77,7 @@[m [mdef generate_launch_description():[m
         # Cube detection node with parameters[m
         Node([m
             package='turtlebot_sensor',[m
[31m-            executable='sync_node',[m
[32m+[m[32m            executable='cube_detection_node',[m
             name='cube_detection_node',[m
             parameters=[{[m
                 'cube_physical_width_m': cube_size,[m
[1mdiff --git a/log/build_2025-05-26_20-05-55/events.log b/log/build_2025-05-26_20-05-55/events.log[m
[1mnew file mode 100644[m
[1mindex 0000000..9b5760f[m
[1m--- /dev/null[m
[1m+++ b/log/build_2025-05-26_20-05-55/events.log[m
[36m@@ -0,0 +1,69 @@[m
[32m+[m[32m[0.000000] (-) TimerEvent: {}[m
[32m+[m[32m[0.000121] (py_pubsub) JobQueued: {'identifier': 'py_pubsub', 'dependencies': OrderedDict()}[m
[32m+[m[32m[0.000153] (turtlebot_sensor) JobQueued: {'identifier': 'turtlebot_sensor', 'dependencies': OrderedDict()}[m
[32m+[m[32m[0.000178] (py_pubsub) JobStarted: {'identifier': 'py_pubsub'}[m
[32m+[m[32m[0.012030] (turtlebot_sensor) JobStarted: {'identifier': 'turtlebot_sensor'}[m
[32m+[m[32m[0.097258] (-) TimerEvent: {}[m
[32m+[m[32m[0.197484] (-) TimerEvent: {}[m
[32m+[m[32m[0.298038] (-) TimerEvent: {}[m
[32m+[m[32m[0.398271] (-) TimerEvent: {}[m
[32m+[m[32m[0.498483] (-) TimerEvent: {}[m
[32m+[m[32m[0.526320] (py_pubsub) Command: {'cmd': ['/usr/bin/python3', '-W', 'ignore:setup.py install is deprecated', '-W', 'ignore:easy_install command is deprecated', 'setup.py', 'egg_info', '--egg-base', '../../build/py_pubsub', 'build', '--build-base', '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/build/py_pubsub/build', 'install', '--record', '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/build/py_pubsub/install.log', '--single-version-externally-managed', 'install_data'], 'cwd': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/src/py_pubsub', 'env': {'LANGUAGE': 'en_NZ:en', 'USER': 'jzen379', 'XDG_SESSION_TYPE': 'x11', 'GIT_ASKPASS': '/usr/share/code/resources/app/extensions/git/dist/askpass.sh', 'XDG_CACHE_HOME': '/tmp/.cache-jzen379/xdg', 'SHLVL': '2', 'LD_LIBRARY_PATH': '/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib:/usr/local/cuda/lib', 'PERL_LOCAL_LIB_ROOT': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/perl5', 'HOME': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome', 'CHROME_DESKTOP': 'code.desktop', 'OLDPWD': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome', 'TERM_PROGRAM_VERSION': '1.97.1', 'DESKTOP_SESSION': 'ubuntu', 'ROS_PYTHON_VERSION': '3', 'GNOME_SHELL_SESSION_MODE': 'ubuntu', 'GTK_MODULES': 'gail:atk-bridge:appmenu-gtk-module:appmenu-gtk-module', 'VSCODE_GIT_ASKPASS_MAIN': '/usr/share/code/resources/app/extensions/git/dist/askpass-main.js', 'VSCODE_GIT_ASKPASS_NODE': '/usr/share/code/code', 'PYDEVD_DISABLE_FILE_VALIDATION': '1', 'SYSTEMD_EXEC_PID': '4298', 'BUNDLED_DEBUGPY_PATH': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/.vscode/extensions/ms-python.debugpy-2025.8.0-linux-x64/bundled/libs/debugpy', 'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/236169613/bus', 'COLORTERM': 'truecolor', 'UBUNTU_MENUPROXY': '1', 'IM_CONFIG_PHASE': '1', 'COLCON_PREFIX_PATH': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install', 'ROS_DISTRO': 'humble', 'GTK_IM_MODULE': 'ibus', 'LOGNAME': 'jzen379', 'IGN_GAZEBO_MODEL_PATH': '/opt/ros/humble/share', '_': '/usr/bin/colcon', 'ROS_VERSION': '2', 'XDG_SESSION_CLASS': 'user', 'USERNAME': 'jzen379', 'TERM': 'xterm-256color', 'PERL_MB_OPT': '--install_base "/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/perl5"', 'GAZEBO_MODEL_PATH': '/opt/ros/humble/share', 'GNOME_DESKTOP_SESSION_ID': 'this-is-deprecated', 'ROS_LOCALHOST_ONLY': '0', 'WINDOWPATH': '2', 'PATH': '/opt/ros/humble/bin:/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/perl5/bin:/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/.local/bin:/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/.local/share/gem/ruby/3.0.0/bin:/usr/lib/jvm/java-11-amazon-corretto/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:/usr/local/cuda/bin:/opt/puppetlabs/bin:/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/.vscode/extensions/ms-python.debugpy-2025.8.0-linux-x64/bundled/scripts/noConfigScripts', 'SESSION_MANAGER': 'local/en441503.uoa.auckland.ac.nz:@/tmp/.ICE-unix/4275,unix/en441503.uoa.auckland.ac.nz:/tmp/.ICE-unix/4275', 'FASTRTPS_DEFAULT_PROFILES_FILE': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/super_client_configuration_file.xml', 'XDG_MENU_PREFIX': 'gnome-', 'GNOME_TERMINAL_SCREEN': '/org/gnome/Terminal/screen/6dec95de_2968_40e0_9c03_8e71fe056699', 'PERL5LIB': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/perl5/lib/perl5', 'XDG_RUNTIME_DIR': '/run/user/236169613', 'GDK_BACKEND': 'x11', 'DISPLAY': ':1', 'VSCODE_DEBUGPY_ADAPTER_ENDPOINTS': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/.vscode/extensions/ms-python.debugpy-2025.8.0-linux-x64/.noConfigDebugAdapterEndpoints/endpoint-0d6ba41ee6198c50.txt', 'LANG': 'en_NZ.UTF-8', 'XDG_CURRENT_DESKTOP': 'Unity', 'XMODIFIERS': '@im=ibus', 'XDG_SESSION_DESKTOP': 'ubuntu', 'XAUTHORITY': '/run/user/236169613/gdm/Xauthority', 'VSCODE_GIT_IPC_HANDLE': '/run/user/236169613/vscode-git-3b9bb59097.sock', 'GNOME_TERMINAL_SERVICE': ':1.103', 'TERM_PROGRAM': 'vscode', 'SSH_AGENT_LAUNCHER': 'gnome-keyring', 'SSH_AUTH_SOCK': '/run/user/236169613/keyring/ssh', 'ROS_DOMAIN_ID': '10', 'AMENT_PREFIX_PATH': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor:/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/py_pubsub:/opt/ros/humble', 'ORIGINAL_XDG_CURRENT_DESKTOP': 'ubuntu:GNOME', 'SHELL': '/bin/bash', 'KRB5CCNAME': 'FILE:/tmp/krb5cc_236169613_RQJ3P4', 'ROS_DISCOVERY_SERVER': '172.24.192.19:11811', 'QT_ACCESSIBILITY': '1', 'GDMSESSION': 'ubuntu', 'PERL_MM_OPT': 'INSTALL_BASE=/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/perl5', 'IGN_GAZEBO_RESOURCE_PATH': '/opt/ros/humble/share', 'GPG_AGENT_INFO': '/run/user/236169613/gnupg/S.gpg-agent:0:1', 'GAZEBO_RESOURCE_PATH': '/opt/ros/humble/share', 'VSCODE_GIT_ASKPASS_EXTRA_ARGS': '', 'QT_IM_MODULE': 'ibus', 'JAVA_HOME': '/usr/lib/jvm/java-11-amazon-corretto', 'PWD': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/build/py_pubsub', 'XDG_CONFIG_DIRS': '/etc/xdg/xdg-ubuntu:/etc/xdg', 'XDG_DATA_DIRS': '/usr/share/ubuntu:/usr/share/gnome:/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/.local/share/flatpak/exports/share:/var/lib/flatpak/exports/share:/usr/local/share/:/usr/share/:/var/lib/snapd/desktop', 'PYTHONPATH': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/build/py_pubsub/prefix_override:/usr/lib/python3/dist-packages/colcon_core/task/python/colcon_distutils_commands:/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/py_pubsub/lib/python3.10/site-packages:/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/turtlebot_sensor/lib/python3.10/site-packages:/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/install/py_pubsub/lib/python3.10/site-packages:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages', 'COLCON': '1', 'MANPATH': ':/usr/lib/jvm/java-11-amazon-corretto/man:/opt/puppetlabs/puppet/share/man', 'VTE_VERSION': '6800'}, 'shell': False}[m
[32m+[m[32m[0.543099] (turtlebot_sensor) Command: {'cmd': ['/usr/bin/python3', '-W', 'ignore:setup.py install is deprecated', '-W', 'ignore:easy_install command is deprecated', 'setup.py', 'egg_info', '--egg-base', '../../build/turtlebot_sensor', 'build', '--build-base', '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/build/turtlebot_sensor/build', 'install', '--record', '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/build/turtlebot_sensor/install.log', '--single-version-externally-managed', 'install_data'], 'cwd': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/ros2_ws/src/turtlebot_sensor', 'env': {'LANGUAGE': 'en_NZ:en', 'USER': 'jzen379', 'XDG_SESSION_TYPE': 'x11', 'GIT_ASKPASS': '/usr/share/code/resources/app/extensions/git/dist/askpass.sh', 'XDG_CACHE_HOME': '/tmp/.cache-jzen379/xdg', 'SHLVL': '2', 'LD_LIBRARY_PATH': '/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib:/usr/local/cuda/lib', 'PERL_LOCAL_LIB_ROOT': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome/perl5', 'HOME': '/afs/ec.auckland.ac.nz/users/j/z/jzen379/unixhome', 'CHROME_DESKTOP': 'code.desktop', 'OLD
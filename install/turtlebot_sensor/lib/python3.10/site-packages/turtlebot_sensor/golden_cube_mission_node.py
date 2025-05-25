import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Point # Point for distance calc
from nav_msgs.msg import Odometry
from std_msgs.msg import String as StringMsg # For explorer status
from enum import Enum
import os
import math
import time

from tf2_ros import Buffer, TransformListener
# import tf2_geometry_msgs # Not strictly needed if PoseStamped is already in map frame

class MissionStatus(Enum):
    INITIALIZING = "INITIALIZING"
    MARKING_START_POSE = "MARKING_START_POSE"
    SENDING_START_TO_EXPLORER = "SENDING_START_TO_EXPLORER"
    EXPLORING = "EXPLORING"
    CUBE_FOUND_AWAITING_RETURN_TRIGGER = "CUBE_FOUND_AWAITING_RETURN_TRIGGER" # Explorer knows cube, mission confirms
    ROBOT_RETURNING_HOME = "ROBOT_RETURNING_HOME" # Explorer has set goal to start
    MISSION_COMPLETE = "MISSION_COMPLETE"
    MISSION_FAILED_TIMEOUT = "MISSION_FAILED_TIMEOUT"
    MISSION_FAILED_EXPLORER = "MISSION_FAILED_EXPLORER" # If explorer signals an unrecoverable error


class GoldenCubeMissionNode(Node):
    def __init__(self):
        super().__init__('golden_cube_mission_node')

        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID')
        if self.robot_id_str is None:
            self.get_logger().error("ROS_DOMAIN_ID environment variable not set! Defaulting to 0.")
            self.robot_id_str = "0"
        self.robot_namespace = f"/T{self.robot_id_str}"
        self.get_logger().info(f"Golden Cube Mission Node initialized for namespace: {self.robot_namespace}")

        self.map_frame = "map"
        self.robot_odom_frame_base = "odom" # Base name, will be namespaced
        self.robot_odom_frame = f"{self.robot_namespace}/{self.robot_odom_frame_base}"

        self.current_mission_status = MissionStatus.INITIALIZING
        self.mission_start_pose_map_frame: PoseStamped = None
        self.current_robot_pose_map_frame: PoseStamped = None
        self.last_explorer_status_str: str = None

        # Parameters
        self.declare_parameter('odom_topic_base', 'odom')
        self.declare_parameter('explorer_status_topic_base', 'explorer_status')
        self.declare_parameter('mission_start_pose_topic_base', 'mission_start_pose')
        self.declare_parameter('exploration_timeout_sec', 300.0) # 5 minutes
        self.declare_parameter('arrival_threshold_m', 0.3) # meters

        odom_topic_base = self.get_parameter('odom_topic_base').get_parameter_value().string_value
        explorer_status_topic_base = self.get_parameter('explorer_status_topic_base').get_parameter_value().string_value
        mission_start_pose_topic_base = self.get_parameter('mission_start_pose_topic_base').get_parameter_value().string_value
        
        self.exploration_timeout = self.get_parameter('exploration_timeout_sec').get_parameter_value().double_value
        self.arrival_threshold = self.get_parameter('arrival_threshold_m').get_parameter_value().double_value

        self.odom_topic_actual = f"{self.robot_namespace}/{odom_topic_base}"
        self.explorer_status_topic_actual = f"{self.robot_namespace}/{explorer_status_topic_base}"
        self.mission_start_pose_topic_actual = f"{self.robot_namespace}/{mission_start_pose_topic_base}"

        self.get_logger().info(f"Subscribing to odom: {self.odom_topic_actual}")
        self.get_logger().info(f"Subscribing to explorer status: {self.explorer_status_topic_actual}")
        self.get_logger().info(f"Publishing mission start pose to: {self.mission_start_pose_topic_actual}")

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # QoS
        qos_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        qos_reliable_latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)


        # Subscriptions
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic_actual, self.odom_callback, qos_sensor)
        self.explorer_status_sub = self.create_subscription(
            StringMsg, self.explorer_status_topic_actual, self.explorer_status_callback, qos_reliable_latched
        )

        # Publishers
        self.mission_start_pose_pub = self.create_publisher(
            PoseStamped, self.mission_start_pose_topic_actual, qos_reliable_latched
        )

        self.mission_timer = self.create_timer(1.0, self.mission_logic_tick)
        self.exploration_start_time = None
        self.set_mission_status(MissionStatus.INITIALIZING)


    def set_mission_status(self, new_status: MissionStatus):
        if self.current_mission_status != new_status:
            self.current_mission_status = new_status
            self.get_logger().info(f"Mission status changed to: {new_status.value}")
            if new_status == MissionStatus.MISSION_COMPLETE or \
               new_status == MissionStatus.MISSION_FAILED_TIMEOUT or \
               new_status == MissionStatus.MISSION_FAILED_EXPLORER:
                self.get_logger().info("Mission ended. Node will remain idle.")
                # Consider stopping the timer or other actions if truly final.

    def odom_callback(self, msg: Odometry):
        pose_in_odom_frame = PoseStamped()
        pose_in_odom_frame.header = msg.header
        pose_in_odom_frame.pose = msg.pose.pose
        try:
            transformed_pose = self.tf_buffer.transform(
                pose_in_odom_frame, self.map_frame, timeout=rclpy.duration.Duration(seconds=0.1)
            )
            self.current_robot_pose_map_frame = transformed_pose

            if self.current_mission_status == MissionStatus.INITIALIZING and self.current_robot_pose_map_frame:
                self.set_mission_status(MissionStatus.MARKING_START_POSE)

        except Exception as e:
            if self.current_mission_status == MissionStatus.INITIALIZING:
                 self.get_logger().warn(f"Odom callback: Could not transform robot pose from '{msg.header.frame_id}' to '{self.map_frame}' during init: {e}", throttle_duration_sec=5.0)
            self.current_robot_pose_map_frame = None # Invalidate if transform fails

    def explorer_status_callback(self, msg: StringMsg):
        self.last_explorer_status_str = msg.data
        # self.get_logger().debug(f"Received explorer status: {self.last_explorer_status_str}", throttle_duration_sec=2.0)

        # Transition based on explorer status
        if self.current_mission_status == MissionStatus.EXPLORING:
            if self.last_explorer_status_str == "CUBE_ACQUIRED_CALCULATING_GLOBAL": # From ExplorerStatus Enum
                self.set_mission_status(MissionStatus.CUBE_FOUND_AWAITING_RETURN_TRIGGER)
        
        if self.current_mission_status == MissionStatus.CUBE_FOUND_AWAITING_RETURN_TRIGGER:
            if self.last_explorer_status_str == "RETURNING_HOME_GOAL_SET": # From ExplorerStatus Enum
                self.get_logger().info("Explorer has set goal to return home. Monitoring robot pose.")
                self.set_mission_status(MissionStatus.ROBOT_RETURNING_HOME)


    def mission_logic_tick(self):
        current_time = self.get_clock().now().seconds_nanoseconds()[0] # Time in seconds

        if self.current_mission_status == MissionStatus.MARKING_START_POSE:
            if self.current_robot_pose_map_frame:
                self.mission_start_pose_map_frame = self.current_robot_pose_map_frame
                # Make a copy to be safe, though PoseStamped should be fine
                self.mission_start_pose_map_frame.header.stamp = self.get_clock().now().to_msg() # Update stamp
                self.get_logger().info(
                    f"Mission start pose marked at map frame ({self.mission_start_pose_map_frame.pose.position.x:.2f}, "
                    f"{self.mission_start_pose_map_frame.pose.position.y:.2f})"
                )
                self.set_mission_status(MissionStatus.SENDING_START_TO_EXPLORER)
            else:
                self.get_logger().warn("In MARKING_START_POSE, but current_robot_pose_map_frame is None. Waiting.")
                return

        elif self.current_mission_status == MissionStatus.SENDING_START_TO_EXPLORER:
            if self.mission_start_pose_map_frame:
                self.mission_start_pose_pub.publish(self.mission_start_pose_map_frame)
                self.get_logger().info("Published mission start pose to explorer.")
                self.exploration_start_time = current_time
                self.set_mission_status(MissionStatus.EXPLORING)
            else:
                self.get_logger().error("Cannot send start pose to explorer, it's not set! Reverting to INITIALIZING.")
                self.set_mission_status(MissionStatus.INITIALIZING) # Should not happen if logic is correct
                return

        elif self.current_mission_status == MissionStatus.EXPLORING:
            if self.exploration_start_time and (current_time - self.exploration_start_time > self.exploration_timeout):
                self.get_logger().error("Exploration timed out!")
                self.set_mission_status(MissionStatus.MISSION_FAILED_TIMEOUT)
            # State transition to CUBE_FOUND_AWAITING_RETURN_TRIGGER is handled by explorer_status_callback

        elif self.current_mission_status == MissionStatus.ROBOT_RETURNING_HOME:
            if self.current_robot_pose_map_frame and self.mission_start_pose_map_frame:
                dist_sq = (self.current_robot_pose_map_frame.pose.position.x - self.mission_start_pose_map_frame.pose.position.x)**2 + \
                          (self.current_robot_pose_map_frame.pose.position.y - self.mission_start_pose_map_frame.pose.position.y)**2
                if dist_sq < (self.arrival_threshold**2):
                    self.get_logger().info("Robot has returned to start location!")
                    self.set_mission_status(MissionStatus.MISSION_COMPLETE)
                else:
                    self.get_logger().debug(f"Returning home: Current distance to start: {math.sqrt(dist_sq):.2f}m", throttle_duration_sec=5.0)
            else:
                self.get_logger().warn("In ROBOT_RETURNING_HOME, but current robot pose or start pose is None. Waiting.")

        elif self.current_mission_status in [MissionStatus.MISSION_COMPLETE, MissionStatus.MISSION_FAILED_TIMEOUT, MissionStatus.MISSION_FAILED_EXPLORER]:
            # Mission has ended, do nothing further in the tick.
            # Optional: could stop the timer self.mission_timer.destroy()
            pass


def main(args=None):
    rclpy.init(args=args)
    mission_node = GoldenCubeMissionNode()
    try:
        rclpy.spin(mission_node)
    except KeyboardInterrupt:
        mission_node.get_logger().info("Keyboard interrupt, shutting down Mission Node.")
    finally:
        if mission_node.get_node_names():
            mission_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
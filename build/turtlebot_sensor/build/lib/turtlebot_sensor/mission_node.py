import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
from visualization_msgs.msg import Marker, MarkerArray
import os
import math
from enum import Enum
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

class MissionState(Enum):
    IDLE = "idle"
    EXPLORING = "exploring" 
    CUBE_DETECTED = "cube_detected"
    RETURNING_HOME = "returning_home"
    MISSION_COMPLETE = "mission_complete"
    MISSION_FAILED = "mission_failed"

class MissionCoordinator(Node):
    def __init__(self):
        super().__init__('mission_coordinator')
        
        # Robot identification
        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID', "0")
        if self.robot_id_str == "0" and os.environ.get('ROS_DOMAIN_ID') is None:
            self.get_logger().warn("ROS_DOMAIN_ID environment variable not set! Defaulting to '0'.")
        self.robot_namespace = f"/T{self.robot_id_str}"
        
        self.declare_parameter('home_arrival_threshold', 0.2)
        self.declare_parameter('target_map_frame', 'odom')
        
        self.target_map_frame = self.get_parameter('target_map_frame').get_parameter_value().string_value
        
        # Check what frame odometry is published in
        self.get_logger().info(f"Subscribing to odometry: {self.robot_namespace}/odom")
        self.get_logger().info(f"Will record start position and track robot pose")
        self.get_logger().info(f"Target frame for operations: '{self.target_map_frame}'")
        
        # Mission state
        self.current_state = MissionState.IDLE
        self.start_pose = None
        self.cube_pose = None
        self.mission_start_time = None
        
        # Get parameter values
        self.home_arrival_threshold = self.get_parameter('home_arrival_threshold').get_parameter_value().double_value
        self.target_map_frame = self.get_parameter('target_map_frame').get_parameter_value().string_value
        
        # QoS Profiles
        qos_reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        qos_reliable_transient_local_for_odom = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,      # Match publisher
            history=HistoryPolicy.KEEP_LAST,           # Standard choice, depth should ideally match
            depth=10,                                  # Or whatever depth the odom publisher uses if known, 10 is a safe default
            durability=DurabilityPolicy.TRANSIENT_LOCAL # Match publisher
        )
        
        # Subscribers
        self.get_logger().info(f"Subscribing to odometry: {self.robot_namespace}/odom")
        marker_topic_name = "/cube_detection_node/cube_marker"
        
        self.odom_sub = self.create_subscription(
            Odometry, f"{self.robot_namespace}/odom", 
            self.odom_callback, 
            qos_reliable_transient_local_for_odom)
        
        self.cube_marker_sub = self.create_subscription(
            Marker, marker_topic_name, 
            self.cube_marker_callback, qos_reliable)
        
        # Publishers
        self.mission_state_pub = self.create_publisher(
            String, f"{self.robot_namespace}/mission_state", qos_reliable)
        
        self.exploration_enable_pub = self.create_publisher(
            Bool, f"{self.robot_namespace}/exploration_enable", qos_reliable)
        
        self.goal_pub = self.create_publisher(
            PoseStamped, f"{self.robot_namespace}/goal_pose", qos_reliable)
        
        self.cmd_vel_pub = self.create_publisher(
            Twist, f"{self.robot_namespace}/cmd_vel", qos_reliable)
        
        self.mission_marker_pub = self.create_publisher(
            MarkerArray, f"{self.robot_namespace}/mission_markers", qos_reliable)
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Timers
        self.state_machine_timer = self.create_timer(0.5, self.state_machine_update)
        self.marker_timer = self.create_timer(2.0, self.publish_mission_markers)
        
        # State tracking
        self.current_robot_pose = None
        
        self.get_logger().info("Mission Coordinator ready. Waiting for start command...")
    
    def get_transformed_pose(self, input_pose_stamped: PoseStamped, target_frame: str) -> PoseStamped | None:
        """Transform pose to target frame"""
        if input_pose_stamped is None:
            return None
        
        source_frame = input_pose_stamped.header.frame_id
        if not source_frame:
            self.get_logger().warn("Input pose has empty frame_id")
            return None
            
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                target_frame, source_frame, 
                rclpy.time.Time(seconds=0), 
                timeout=rclpy.duration.Duration(seconds=0.1))
            
            transformed_pose_msg = do_transform_pose(input_pose_stamped.pose, transform_stamped)
            
            result_pose_stamped = PoseStamped()
            result_pose_stamped.header.stamp = transform_stamped.header.stamp
            result_pose_stamped.header.frame_id = target_frame
            result_pose_stamped.pose = transformed_pose_msg
            
            return result_pose_stamped
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}", throttle_duration_sec=2.0)
            return None
    
    def odom_callback(self, msg: Odometry):
        """Track robot odometry and record start position"""
        # self.get_logger().info(f"Received odometry in frame: {msg.header.frame_id}", throttle_duration_sec=2.0)
        
        current_pose_stamped = PoseStamped()
        current_pose_stamped.header = msg.header
        current_pose_stamped.pose = msg.pose.pose
        
        # Use odometry poses directly since they're in the frame we want
        self.current_robot_pose = current_pose_stamped
        
        # Record start position on first successful pose
        if self.start_pose is None and self.current_robot_pose is not None:
            self.start_pose = self.current_robot_pose
            self.get_logger().info(f"Start position recorded: ({self.start_pose.pose.position.x:.2f}, {self.start_pose.pose.position.y:.2f}) in frame '{self.start_pose.header.frame_id}'")
            self.get_logger().info("Mission will start automatically on next timer cycle...")
    
    def cube_marker_callback(self, msg: Marker):
        """Handle cube detection from cube detection node via marker"""
        
        if self.current_state in [MissionState.EXPLORING, MissionState.CUBE_DETECTED]: # CUBE_DETECTED here allows re-detection if needed
            if msg.action == Marker.ADD and msg.type == Marker.CUBE:
                
                cube_pose_stamped_from_marker = PoseStamped()
                cube_pose_stamped_from_marker.header = msg.header # frame_id will be 'base_link' from your echo
                cube_pose_stamped_from_marker.pose = msg.pose
                
                self.get_logger().info(f"Attempting to transform cube pose from '{msg.header.frame_id}' to '{self.target_map_frame}'...")
                transformed_cube_pose = self.get_transformed_pose(cube_pose_stamped_from_marker, self.target_map_frame) # self.target_map_frame is 'odom'

                if transformed_cube_pose is not None:
                    self.cube_pose = transformed_cube_pose # Store the TRANSFORMED pose (now in 'odom')
                    self.get_logger().info(f"🎯 Cube detected! Transformed Location: ({self.cube_pose.pose.position.x:.2f}, {self.cube_pose.pose.position.y:.2f}) in frame '{self.cube_pose.header.frame_id}'")
                    self.get_logger().info("🏠 Cube registered, proceeding to go home!")
                    self.transition_to_state(MissionState.CUBE_DETECTED) # This will then immediately go to RETURNING_HOME
                else:
                    self.get_logger().warn(f"Failed to transform cube pose from '{msg.header.frame_id}' to '{self.target_map_frame}'. Cube not registered. Check TF tree (odom -> base_link). Mission continues exploring.")
                    # IMPORTANT: If transform fails, we do NOT transition. Robot keeps exploring.
            else:
                self.get_logger().info(f"Ignoring marker: action={msg.action} (expected {Marker.ADD}), type={msg.type} (expected {Marker.CUBE})", throttle_duration_sec=2.0)
        else:
            self.get_logger().info(f"Received marker in state {self.current_state.value} - ignoring", throttle_duration_sec=2.0)
    
    def start_mission(self):
        """Start the cube search mission"""
        if self.start_pose is None:
            self.get_logger().error("Cannot start mission: start position not recorded yet")
            return False
        
        self.get_logger().info("Starting cube search mission!")
        self.mission_start_time = self.get_clock().now()
        self.transition_to_state(MissionState.EXPLORING)
        return True
    
    def transition_to_state(self, new_state: MissionState):
        """Handle state transitions"""
        if new_state == self.current_state:
            return
        
        old_state = self.current_state
        self.current_state = new_state
        
        self.get_logger().info(f"Mission state: {old_state.value} -> {new_state.value}")
        
        # Publish state change
        state_msg = String()
        state_msg.data = new_state.value
        self.mission_state_pub.publish(state_msg)
        
        # Handle state entry actions
        if new_state == MissionState.EXPLORING:
            self.enable_exploration(True)
        
        elif new_state == MissionState.CUBE_DETECTED:
            self.enable_exploration(False)
            # Immediately transition to returning home
            self.transition_to_state(MissionState.RETURNING_HOME)
            
        elif new_state == MissionState.RETURNING_HOME:
            self.publish_home_goal()
            
        elif new_state == MissionState.MISSION_COMPLETE:
            self.stop_robot()
            self.get_logger().info("🎉 Mission completed successfully!")
            
        elif new_state == MissionState.MISSION_FAILED:
            self.stop_robot()
            self.get_logger().error("❌ Mission failed!")
    
    def enable_exploration(self, enable: bool):
        """Enable/disable exploration mode"""
        msg = Bool()
        msg.data = enable
        self.exploration_enable_pub.publish(msg)
        self.get_logger().info(f"Exploration {'enabled' if enable else 'disabled'}")
    
    def publish_home_goal(self):
        """Publish goal to return home"""
        if self.start_pose is None:
            self.get_logger().error("Cannot return home: start position not recorded")
            self.transition_to_state(MissionState.MISSION_FAILED)
            return
        
        home_goal = PoseStamped()
        home_goal.header.stamp = self.get_clock().now().to_msg()
        home_goal.header.frame_id = self.target_map_frame
        home_goal.pose = self.start_pose.pose
        
        self.goal_pub.publish(home_goal)
        self.get_logger().info(f"Published home goal: ({home_goal.pose.position.x:.2f}, {home_goal.pose.position.y:.2f})")
    
    def stop_robot(self):
        """Stop robot movement"""
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
    
    def check_goal_reached(self, target_pose: PoseStamped, threshold: float) -> bool:
        """Check if robot has reached the target pose"""
        if self.current_robot_pose is None or target_pose is None:
            return False
        
        dx = self.current_robot_pose.pose.position.x - target_pose.pose.position.x
        dy = self.current_robot_pose.pose.position.y - target_pose.pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        return distance < threshold
    
    def state_machine_update(self):
        """Main state machine logic"""
        if self.current_state == MissionState.IDLE:
            # Wait for start command or auto-start if ready
            if self.start_pose is not None:
                self.start_mission()
        
        elif self.current_state == MissionState.EXPLORING:
            # Exploration handled by explorer node
            # Transition handled by cube detection callback
            pass
        
        elif self.current_state == MissionState.CUBE_DETECTED:
            # This state should immediately transition to RETURNING_HOME
            # Handled in transition_to_state method
            pass
        
        elif self.current_state == MissionState.RETURNING_HOME:
            # Check if we've reached home
            if self.start_pose is not None and self.check_goal_reached(self.start_pose, self.home_arrival_threshold):
                self.transition_to_state(MissionState.MISSION_COMPLETE)
    
    def publish_mission_markers(self):
        """Publish visualization markers for mission waypoints"""
        marker_array = MarkerArray()
        
        # Delete all previous markers
        delete_marker = Marker()
        delete_marker.header.frame_id = self.target_map_frame
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = "mission_markers"
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Start position marker
        if self.start_pose is not None:
            start_marker = Marker()
            start_marker.header.frame_id = self.target_map_frame
            start_marker.header.stamp = self.get_clock().now().to_msg()
            start_marker.ns = "mission_markers"
            start_marker.id = 1
            start_marker.type = Marker.CYLINDER
            start_marker.action = Marker.ADD
            start_marker.pose = self.start_pose.pose
            start_marker.pose.position.z = 0.1
            start_marker.scale.x = 0.4
            start_marker.scale.y = 0.4
            start_marker.scale.z = 0.2
            start_marker.color.r = 0.0
            start_marker.color.g = 1.0
            start_marker.color.b = 0.0
            start_marker.color.a = 0.7
            start_marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
            marker_array.markers.append(start_marker)
        
        # Cube position marker
        if self.cube_pose is not None:
            cube_marker = Marker()
            cube_marker.header.frame_id = self.target_map_frame
            cube_marker.header.stamp = self.get_clock().now().to_msg()
            cube_marker.ns = "mission_markers"
            cube_marker.id = 2
            cube_marker.type = Marker.CUBE
            cube_marker.action = Marker.ADD
            cube_marker.pose = self.cube_pose.pose
            cube_marker.pose.position.z = 0.15
            cube_marker.scale.x = 0.3
            cube_marker.scale.y = 0.3
            cube_marker.scale.z = 0.3
            cube_marker.color.r = 1.0
            cube_marker.color.g = 0.8
            cube_marker.color.b = 0.0
            cube_marker.color.a = 0.8
            cube_marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
            marker_array.markers.append(cube_marker)
        
        self.mission_marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    coordinator = MissionCoordinator()
    
    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        coordinator.get_logger().info("Keyboard interrupt, shutting down Mission Coordinator.")
    finally:
        coordinator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
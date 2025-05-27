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
    APPROACHING_CUBE = "approaching_cube"
    CUBE_ACQUIRED = "cube_acquired"
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
        
        # Parameters - declare first, then get values
        self.declare_parameter('cube_approach_distance', 1.5)
        self.declare_parameter('home_arrival_threshold', 0.3)
        self.declare_parameter('cube_acquisition_time', 3.0)
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
        self.cube_approach_distance = self.get_parameter('cube_approach_distance').get_parameter_value().double_value
        self.home_arrival_threshold = self.get_parameter('home_arrival_threshold').get_parameter_value().double_value
        self.cube_acquisition_time = self.get_parameter('cube_acquisition_time').get_parameter_value().double_value
        self.target_map_frame = self.get_parameter('target_map_frame').get_parameter_value().string_value
        
        # QoS Profiles
        qos_reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        
        # Subscribers
        self.get_logger().info(f"Subscribing to odometry: {self.robot_namespace}/odom")
        self.odom_sub = self.create_subscription(
            Odometry, f"{self.robot_namespace}/odom", 
            self.odom_callback, qos_best_effort)
        
        self.cube_marker_sub = self.create_subscription(
            Marker, f"{self.robot_namespace}/cube_marker", 
            self.cube_marker_callback, qos_best_effort)
        
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
        self.cube_acquisition_start_time = None
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
        current_pose_stamped = PoseStamped()
        current_pose_stamped.header = msg.header
        current_pose_stamped.pose = msg.pose.pose
        
        # Use odometry poses directly since they're in the frame we want
        self.current_robot_pose = current_pose_stamped
        
        # Record start position on first successful pose
        if self.start_pose is None and self.current_robot_pose is not None:
            self.start_pose = self.current_robot_pose
            self.get_logger().info(f"Start position recorded: ({self.start_pose.pose.position.x:.2f}, {self.start_pose.pose.position.y:.2f}) in frame '{self.start_pose.header.frame_id}'")
    
    def cube_marker_callback(self, msg: Marker):
        """Handle cube detection from cube detection node via marker"""
        self.get_logger().info(f"Received marker: type={msg.type}, action={msg.action}, frame={msg.header.frame_id}")
        
        if self.current_state in [MissionState.EXPLORING, MissionState.CUBE_DETECTED]:
            if msg.action == Marker.ADD and msg.type == Marker.CUBE:
                # Extract cube pose from marker
                cube_pose = PoseStamped()
                cube_pose.header = msg.header
                cube_pose.pose = msg.pose
                
                self.cube_pose = cube_pose
                self.get_logger().info(f"Cube detected via marker at: ({self.cube_pose.pose.position.x:.2f}, {self.cube_pose.pose.position.y:.2f}) in frame '{msg.header.frame_id}'")
                self.transition_to_state(MissionState.CUBE_DETECTED)
            else:
                self.get_logger().info(f"Ignoring marker: action={msg.action}, type={msg.type}")
        else:
            self.get_logger().info(f"Received marker in state {self.current_state.value} - ignoring")
    
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
            
        elif new_state == MissionState.APPROACHING_CUBE:
            self.publish_cube_approach_goal()
            
        elif new_state == MissionState.CUBE_ACQUIRED:
            self.cube_acquisition_start_time = self.get_clock().now()
            
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
    
    def publish_cube_approach_goal(self):
        """Publish goal to approach the cube"""
        if self.cube_pose is None or self.current_robot_pose is None:
            self.get_logger().error("Cannot approach cube: missing cube or robot pose")
            self.transition_to_state(MissionState.MISSION_FAILED)
            return
        
        # Calculate approach position (stay at specified distance from cube)
        cube_x = self.cube_pose.pose.position.x
        cube_y = self.cube_pose.pose.position.y
        robot_x = self.current_robot_pose.pose.position.x
        robot_y = self.current_robot_pose.pose.position.y
        
        # Direction from cube to robot
        dx = robot_x - cube_x
        dy = robot_y - cube_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 0.1:  # Too close to cube to calculate direction
            self.get_logger().warn("Robot too close to cube to calculate approach direction")
            self.transition_to_state(MissionState.CUBE_ACQUIRED)
            return
        
        # Normalize direction and scale to approach distance
        approach_x = cube_x + (dx / distance) * self.cube_approach_distance
        approach_y = cube_y + (dy / distance) * self.cube_approach_distance
        
        # Create approach goal
        approach_goal = PoseStamped()
        approach_goal.header.stamp = self.get_clock().now().to_msg()
        approach_goal.header.frame_id = self.target_map_frame
        approach_goal.pose.position.x = approach_x
        approach_goal.pose.position.y = approach_y
        approach_goal.pose.position.z = 0.0
        
        # Face towards the cube
        angle_to_cube = math.atan2(cube_y - approach_y, cube_x - approach_x)
        approach_goal.pose.orientation.z = math.sin(angle_to_cube / 2.0)
        approach_goal.pose.orientation.w = math.cos(angle_to_cube / 2.0)
        
        self.goal_pub.publish(approach_goal)
        self.get_logger().info(f"Published cube approach goal: ({approach_x:.2f}, {approach_y:.2f})")
    
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
            # Transition to approaching cube
            self.transition_to_state(MissionState.APPROACHING_CUBE)
        
        elif self.current_state == MissionState.APPROACHING_CUBE:
            # Check if we've reached the approach position
            if self.cube_pose is not None:
                # Check if we're close enough to the cube
                if self.current_robot_pose is not None:
                    cube_x = self.cube_pose.pose.position.x
                    cube_y = self.cube_pose.pose.position.y
                    robot_x = self.current_robot_pose.pose.position.x
                    robot_y = self.current_robot_pose.pose.position.y
                    
                    distance_to_cube = math.sqrt((robot_x - cube_x)**2 + (robot_y - cube_y)**2)
                    
                    if distance_to_cube <= self.cube_approach_distance * 1.2:  # Allow some tolerance
                        self.transition_to_state(MissionState.CUBE_ACQUIRED)
        
        elif self.current_state == MissionState.CUBE_ACQUIRED:
            # Wait for acquisition time, then return home
            if self.cube_acquisition_start_time is not None:
                elapsed = (self.get_clock().now() - self.cube_acquisition_start_time).nanoseconds / 1e9
                if elapsed >= self.cube_acquisition_time:
                    self.get_logger().info("Cube acquired! Returning home...")
                    self.transition_to_state(MissionState.RETURNING_HOME)
        
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
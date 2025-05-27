import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist, Pose
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from collections import deque
from std_msgs.msg import Bool
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID', "0")
        if self.robot_id_str == "0" and os.environ.get('ROS_DOMAIN_ID') is None:
            self.get_logger().warn("ROS_DOMAIN_ID environment variable not set! Defaulting to '0'.")
        self.robot_namespace = f"/T{self.robot_id_str}"
        self.get_logger().info(f"Frontier Explorer node initialized for namespace: {self.robot_namespace}")

        self.map_data = None
        self.robot_pose_odom: PoseStamped | None = None
        self.cube_seen_counter = 0
        self.seeking_cube = False
        self.cube_x_global_map: float | None = None
        self.cube_y_global_map: float | None = None
        self.returning_home = False
        self.start_pose_odom: PoseStamped | None = None

        self.current_direction = 0 
        self.direction_vectors = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
        self.turn_direction = 1    
        self.is_shifting_lane = False

        # Visited grid attributes
        self.visited_grid: np.ndarray | None = None
        self.visited_map_resolution: float = 0.0
        self.visited_grid_width: int = 0
        self.visited_grid_height: int = 0
        self.map_origin_x: float = 0.0
        self.map_origin_y: float = 0.0


        # Parameters
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('goal_topic', 'goal_pose')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('waypoints_topic', 'waypoints')
        self.declare_parameter('cube_pose_topic', 'cube_pose')
        self.declare_parameter('target_map_frame', 'map')
        self.declare_parameter('step_distance', 1.5) 
        self.declare_parameter('min_sweep_step_distance', 1.0)
        self.declare_parameter('sweep_step_reduction_factor', 0.75)
        self.declare_parameter('qos_depth', 10)
        self.declare_parameter('max_specific_goal_failures', 2) 
        self.declare_parameter('max_general_failures_for_step_reduction', 2)
        self.declare_parameter('sweep_max_probe_multiplier', 3)
        self.declare_parameter('visited_grid_resolution_multiplier', 2) # e.g. visited cell is 2x2 map cells
        self.declare_parameter('visited_grid_update_radius_cells', 2)  # Radius in visited_grid cells

        map_topic_param = self.get_parameter('map_topic').get_parameter_value().string_value
        odom_topic_param = self.get_parameter('odom_topic').get_parameter_value().string_value
        goal_topic_param = self.get_parameter('goal_topic').get_parameter_value().string_value
        cmd_vel_topic_param = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        waypoints_topic_param = self.get_parameter('waypoints_topic').get_parameter_value().string_value
        cube_pose_topic_param = self.get_parameter('cube_pose_topic').get_parameter_value().string_value
        self.target_map_frame = self.get_parameter('target_map_frame').get_parameter_value().string_value
        
        self.initial_step_distance = self.get_parameter('step_distance').get_parameter_value().double_value
        self.step_distance = self.initial_step_distance 
        
        self.min_sweep_step_distance = self.get_parameter('min_sweep_step_distance').get_parameter_value().double_value
        self.sweep_step_reduction_factor = self.get_parameter('sweep_step_reduction_factor').get_parameter_value().double_value
        qos_depth = self.get_parameter('qos_depth').get_parameter_value().integer_value
        self.max_specific_goal_failures = self.get_parameter('max_specific_goal_failures').get_parameter_value().integer_value
        self.max_general_failures_for_step_reduction = self.get_parameter('max_general_failures_for_step_reduction').get_parameter_value().integer_value
        self.sweep_max_probe_multiplier = self.get_parameter('sweep_max_probe_multiplier').get_parameter_value().integer_value
        self.visited_grid_resolution_multiplier = self.get_parameter('visited_grid_resolution_multiplier').get_parameter_value().integer_value
        self.visited_grid_update_radius_cells = self.get_parameter('visited_grid_update_radius_cells').get_parameter_value().integer_value


        self.failed_goal_attempts: dict[tuple[float, float], int] = {} 
        self.recently_blacklisted_world_goals: deque[tuple[float, float]] = deque(maxlen=10) 
        self.last_published_sweep_goal: PoseStamped | None = None
        self.last_robot_pose_at_sweep_goal_pub: PoseStamped | None = None
        self.general_consecutive_planning_failures = 0 

        self.map_topic_actual = f"{self.robot_namespace}/{map_topic_param}"
        self.odom_topic_actual = f"{self.robot_namespace}/{odom_topic_param}"
        self.goal_topic_actual = f"{self.robot_namespace}/{goal_topic_param}"
        self.cmd_vel_topic_actual = f"{self.robot_namespace}/{cmd_vel_topic_param}"
        self.waypoints_topic_actual = f"{self.robot_namespace}/{waypoints_topic_param}" 
        self.cube_pose_topic_actual = f"{self.robot_namespace}/{cube_pose_topic_param}"
        
        self.get_logger().info(f"Initial sweep step_distance: {self.step_distance:.2f}m, probe_mult: {self.sweep_max_probe_multiplier}, visited_res_mult: {self.visited_grid_resolution_multiplier}")

        qos_profile_map = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        qos_profile_odom = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=qos_depth)
        qos_profile_cube = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=qos_depth)
        qos_profile_goal = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=qos_depth)

        self.create_subscription(OccupancyGrid, self.map_topic_actual, self.map_callback, qos_profile_map)
        self.create_subscription(Odometry, self.odom_topic_actual, self.odom_callback, qos_profile_odom)
        self.create_subscription(PoseStamped, self.cube_pose_topic_actual, self.cube_callback, qos_profile_cube)
        
        self.goal_pub = self.create_publisher(PoseStamped, self.goal_topic_actual, qos_profile_goal)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic_actual, qos_depth)
        self.marker_pub = self.create_publisher(MarkerArray, self.waypoints_topic_actual, qos_profile_map)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(5.0, self.timer_callback)

        self.declare_parameter('use_mission_coordinator', True)
        self.use_mission_coordinator = self.get_parameter('use_mission_coordinator').get_parameter_value().bool_value
        self.exploration_enabled = True

        if self.use_mission_coordinator:
            self.exploration_enable_sub = self.create_subscription(
                Bool, f"{self.robot_namespace}/exploration_enable", 
                self.exploration_enable_callback, qos_profile_goal)
            self.get_logger().info("Mission coordinator integration enabled")
        else:
            from visualization_msgs.msg import Marker as VisMarker 
            self.cube_marker_sub = self.create_subscription(
                VisMarker, f"{self.robot_namespace}/cube_marker",
                self.cube_marker_callback, qos_profile_cube)
            self.get_logger().info("Running in standalone mode - monitoring cube markers")            

    def map_callback(self, msg: OccupancyGrid):
        if msg.header.frame_id != self.target_map_frame:
            self.get_logger().warn(
                f"Received map in frame '{msg.header.frame_id}' but expected '{self.target_map_frame}'. "
                f"Map-based calculations might be incorrect."
            )
        self.map_data = msg
        self.map_origin_x = self.map_data.info.origin.position.x
        self.map_origin_y = self.map_data.info.origin.position.y

        # Initialize/Re-initialize visited_grid
        if self.map_data.info.resolution > 0 and self.visited_grid_resolution_multiplier > 0:
            self.visited_map_resolution = self.map_data.info.resolution * self.visited_grid_resolution_multiplier
            self.visited_grid_width = int(self.map_data.info.width / self.visited_grid_resolution_multiplier)
            self.visited_grid_height = int(self.map_data.info.height / self.visited_grid_resolution_multiplier)
            
            if self.visited_grid_width > 0 and self.visited_grid_height > 0:
                self.visited_grid = np.zeros((self.visited_grid_height, self.visited_grid_width), dtype=np.uint8)
            else:
                self.get_logger().warn("Visited grid dimensions are zero. Disabling visited grid functionality.")
                self.visited_grid = None
        else:
            self.get_logger().warn("Map resolution or visited_grid_resolution_multiplier is zero. Cannot init visited_grid.")
            self.visited_grid = None


    def odom_callback(self, msg: Odometry):
        current_pose_stamped = PoseStamped()
        current_pose_stamped.header = msg.header 
        current_pose_stamped.pose = msg.pose.pose
        
        if self.last_published_sweep_goal and self.last_robot_pose_at_sweep_goal_pub:
            current_pose_map = self.get_transformed_pose(current_pose_stamped, self.target_map_frame)
            if current_pose_map:
                dist_sq_moved = (current_pose_map.pose.position.x - self.last_robot_pose_at_sweep_goal_pub.pose.position.x)**2 + \
                                (current_pose_map.pose.position.y - self.last_robot_pose_at_sweep_goal_pub.pose.position.y)**2
                
                if dist_sq_moved > (0.25**2): 
                    self.get_logger().info(f"Robot moved {math.sqrt(dist_sq_moved):.2f}m since last sweep goal. Resetting some failure counts.")
                    goal_key_last = (round(self.last_published_sweep_goal.pose.position.x, 1), 
                                     round(self.last_published_sweep_goal.pose.position.y, 1))
                    if goal_key_last in self.failed_goal_attempts:
                        del self.failed_goal_attempts[goal_key_last]
                    
                    self.last_published_sweep_goal = None 
                    self.last_robot_pose_at_sweep_goal_pub = None
                    
                    if self.general_consecutive_planning_failures > 0:
                        self.general_consecutive_planning_failures = 0
                        self.get_logger().info("Reset general_consecutive_planning_failures.")
                    
                    if self.step_distance != self.initial_step_distance:
                        self.step_distance = self.initial_step_distance
                        self.get_logger().info(f"Reset step_distance to initial {self.step_distance:.2f}m.")
        
        self.robot_pose_odom = current_pose_stamped
        if self.start_pose_odom is None:
            self.start_pose_odom = current_pose_stamped
            self.get_logger().info(f"Stored start_pose in {self.start_pose_odom.header.frame_id} frame at T={self.start_pose_odom.header.stamp.sec}")


    def get_transformed_pose(self, input_pose_stamped: PoseStamped, target_frame: str) -> PoseStamped | None:
        if input_pose_stamped is None: self.get_logger().debug("Input pose for transformation is None.", throttle_duration_sec=5.0); return None
        if not hasattr(input_pose_stamped, 'pose') or not isinstance(input_pose_stamped.pose, Pose): self.get_logger().error(f"Invalid input_pose_stamped.pose. Expected Pose, got {type(input_pose_stamped.pose)}."); return None
        source_frame = input_pose_stamped.header.frame_id
        if not source_frame: self.get_logger().warn("Input pose_stamped has an empty frame_id.", throttle_duration_sec=5.0); return None
        try:
            transform_stamped = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time(seconds=0, nanoseconds=0), timeout=rclpy.duration.Duration(seconds=0.1))
            transformed_pose_msg = do_transform_pose(input_pose_stamped.pose, transform_stamped)
            result_pose_stamped = PoseStamped(); result_pose_stamped.header.stamp = transform_stamped.header.stamp; result_pose_stamped.header.frame_id = target_frame; result_pose_stamped.pose = transformed_pose_msg
            return result_pose_stamped
        except Exception as e: self.get_logger().warn(f"TF EXCEPTION transforming from '{source_frame}' to '{target_frame}': {e}. Input pose stamp: {input_pose_stamped.header.stamp.sec}.{input_pose_stamped.header.stamp.nanosec:09d}.", throttle_duration_sec=2.0); return None

    def cube_callback(self, msg: PoseStamped):
        if self.use_mission_coordinator:
            if self.seeking_cube:
                self.get_logger().info("Cube detected - coordinator will handle approach")
                self.seeking_cube = False; self.cube_seen_counter = 0; self.cmd_pub.publish(Twist())
            return
        if self.returning_home: return
        x_rel_cube, y_rel_cube = msg.pose.position.x, msg.pose.position.y 
        if x_rel_cube == 0.0 and y_rel_cube == 0.0: 
            if self.seeking_cube: self.get_logger().info("Cube lost (0,0 received).", throttle_duration_sec=5.0)
            self.seeking_cube = False; self.cube_seen_counter = 0; return
        self.seeking_cube = True; lateral_threshold = 0.15; forward_target_distance = 1.5; forward_stop_distance = 1.7
        current_robot_pose_map = self.get_transformed_pose(self.robot_pose_odom, self.target_map_frame)
        if abs(x_rel_cube) < lateral_threshold and y_rel_cube < forward_stop_distance and y_rel_cube > 0.2:
            self.cube_seen_counter += 1; self.get_logger().info(f"Cube aligned. Seen {self.cube_seen_counter} times.")
            if current_robot_pose_map: self.publish_goal(current_robot_pose_map, from_sweep=False)
            else: self.cmd_pub.publish(Twist()) 
        else: 
            self.cube_seen_counter = 0; cmd = Twist(); cmd.angular.z = -0.7 * x_rel_cube; forward_error = y_rel_cube - forward_target_distance
            if abs(x_rel_cube) < (lateral_threshold*2.5) or y_rel_cube > (forward_target_distance+0.8): cmd.linear.x = 0.4 * forward_error
            else: cmd.linear.x = 0.0
            cmd.angular.z = max(min(cmd.angular.z, 0.6), -0.6); cmd.linear.x = max(min(cmd.linear.x, 0.35), -0.35)
            if abs(forward_error) < 0.15 and abs(x_rel_cube) > lateral_threshold: cmd.linear.x = 0.05 * np.sign(forward_error)
            self.cmd_pub.publish(cmd)
        if self.cube_seen_counter >= 25: 
            if current_robot_pose_map is None: self.get_logger().warn("Cannot determine cube global position: robot pose in map frame not available."); return
            x_robot_map = current_robot_pose_map.pose.position.x; y_robot_map = current_robot_pose_map.pose.position.y; q_map = current_robot_pose_map.pose.orientation
            robot_yaw_map = math.atan2(2.0*(q_map.w*q_map.z + q_map.x*q_map.y), 1.0-2.0*(q_map.y*q_map.y+q_map.z*q_map.z))
            self.cube_x_global_map = x_robot_map + math.cos(robot_yaw_map)*y_rel_cube - math.sin(robot_yaw_map)*x_rel_cube
            self.cube_y_global_map = y_robot_map + math.sin(robot_yaw_map)*y_rel_cube + math.cos(robot_yaw_map)*x_rel_cube
            self.get_logger().info(f"Cube acquired at map global ({self.cube_x_global_map:.2f}, {self.cube_y_global_map:.2f}). Returning home.")
            self.returning_home = True; self.seeking_cube = False
            if self.start_pose_odom:
                start_pose_map = self.get_transformed_pose(self.start_pose_odom, self.target_map_frame)
                if start_pose_map: self.publish_goal(start_pose_map, from_sweep=False)
                else: self.get_logger().error("Failed to transform start_pose to map frame for returning home.")
            else: self.get_logger().warn("Start pose (odom) not set, cannot publish return home goal.")

    def publish_goal(self, goal_pose_stamped_in_target_frame: PoseStamped, from_sweep: bool = True):
        if goal_pose_stamped_in_target_frame.header.frame_id != self.target_map_frame:
            self.get_logger().error(f"CRITICAL: publish_goal frame mismatch. Expected '{self.target_map_frame}', got '{goal_pose_stamped_in_target_frame.header.frame_id}'. Goal NOT published.")
            return

        final_goal = PoseStamped()
        final_goal.header.stamp = self.get_clock().now().to_msg()
        final_goal.header.frame_id = self.target_map_frame 
        final_goal.pose = goal_pose_stamped_in_target_frame.pose
        
        self.goal_pub.publish(final_goal)
        self.get_logger().info(f"Published goal to ({final_goal.pose.position.x:.2f}, {final_goal.pose.position.y:.2f}) in '{final_goal.header.frame_id}' frame.")
        
        if from_sweep:
            self.last_published_sweep_goal = final_goal
            self.last_robot_pose_at_sweep_goal_pub = self.get_transformed_pose(self.robot_pose_odom, self.target_map_frame)
            goal_key = (round(final_goal.pose.position.x, 1), round(final_goal.pose.position.y, 1)) 
            current_attempts_for_this_goal = self.failed_goal_attempts.get(goal_key, 0) + 1
            self.failed_goal_attempts[goal_key] = current_attempts_for_this_goal
            self.get_logger().debug(f"Attempt {current_attempts_for_this_goal} for goal key {goal_key}.")
            if current_attempts_for_this_goal == 1:
                if self.general_consecutive_planning_failures > 0:
                    self.get_logger().info(f"Trying new sweep goal area {goal_key}. Resetting general_consecutive_planning_failures.")
                    self.general_consecutive_planning_failures = 0
                if self.step_distance != self.initial_step_distance:
                     self.step_distance = self.initial_step_distance
                     self.get_logger().info(f"Reset step_distance to initial {self.step_distance:.2f}m for new goal area.")

    def timer_callback(self):
        if self.use_mission_coordinator and not self.exploration_enabled: return  
        current_robot_pose_map = self.get_transformed_pose(self.robot_pose_odom, self.target_map_frame)
        if self.map_data is None or current_robot_pose_map is None: return
        if self.returning_home: 
            if self.cube_x_global_map is not None: marker = Marker(); marker.header.frame_id = self.target_map_frame; marker.header.stamp = self.get_clock().now().to_msg(); marker.ns = f"{self.robot_namespace}_cube_marker"; marker.id = 0; marker.type = Marker.CUBE; marker.action = Marker.ADD; marker.pose.position.x = self.cube_x_global_map; marker.pose.position.y = self.cube_y_global_map; marker.pose.position.z = 0.15; marker.pose.orientation.w = 1.0; marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3; marker.color.r = 1.0; marker.color.g = 0.5; marker.color.b = 0.0; marker.color.a = 0.95; marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg(); self.marker_pub.publish(MarkerArray(markers=[marker]))
            return
        if self.seeking_cube: return

        # Update visited grid based on current robot pose
        if self.visited_grid is not None and self.visited_map_resolution > 0:
            robot_x_vis = int((current_robot_pose_map.pose.position.x - self.map_origin_x) / self.visited_map_resolution)
            robot_y_vis = int((current_robot_pose_map.pose.position.y - self.map_origin_y) / self.visited_map_resolution)
            radius = self.visited_grid_update_radius_cells
            for r_offset in range(-radius, radius + 1):
                for c_offset in range(-radius, radius + 1):
                    # Optional: circular radius check: if r_offset**2 + c_offset**2 > radius**2: continue
                    curr_r, curr_c = robot_y_vis + r_offset, robot_x_vis + c_offset
                    if 0 <= curr_r < self.visited_grid_height and 0 <= curr_c < self.visited_grid_width:
                        self.visited_grid[curr_r, curr_c] = 1 # Mark as visited

        dist_sq_moved_since_last_goal = float('inf')
        if self.last_published_sweep_goal and self.last_robot_pose_at_sweep_goal_pub:
            dist_sq_moved_since_last_goal = (current_robot_pose_map.pose.position.x - self.last_robot_pose_at_sweep_goal_pub.pose.position.x)**2 + \
                                          (current_robot_pose_map.pose.position.y - self.last_robot_pose_at_sweep_goal_pub.pose.position.y)**2
        
        if self.last_published_sweep_goal and self.last_robot_pose_at_sweep_goal_pub:
            goal_key_last = (round(self.last_published_sweep_goal.pose.position.x, 1), 
                             round(self.last_published_sweep_goal.pose.position.y, 1))
            if dist_sq_moved_since_last_goal < (0.1**2) and goal_key_last in self.failed_goal_attempts:
                num_attempts_for_last_goal = self.failed_goal_attempts[goal_key_last]
                if num_attempts_for_last_goal >= self.max_specific_goal_failures:
                    self.get_logger().warn(f"Goal {goal_key_last} failed {num_attempts_for_last_goal} times. Blacklisting & forcing rotation.")
                    self.recently_blacklisted_world_goals.append(goal_key_last)
                    if goal_key_last in self.failed_goal_attempts: del self.failed_goal_attempts[goal_key_last] 
                    self.current_direction = (self.current_direction + self.turn_direction) % 4
                    self.get_logger().info(f"Forced rotation. New sweep direction idx: {self.current_direction}")
                    self.general_consecutive_planning_failures = 0
                    if self.step_distance != self.initial_step_distance:
                        self.step_distance = self.initial_step_distance
                        self.get_logger().info(f"Reset step_distance to initial {self.step_distance:.2f}m after blacklisting.")
                    self.last_published_sweep_goal = None 
                    goal_for_sweep = self.find_next_sweep_goal(current_robot_pose_map) # Try new direction
                    if goal_for_sweep: self.publish_goal(goal_for_sweep)
                    else: 
                        self.get_logger().warn("Sweep still cannot find goal after forced rotation. Incr general failure.")
                        self.general_consecutive_planning_failures +=1 
                        self.current_direction = (self.current_direction + self.turn_direction) % 4 # Rotate again
                    return 
            else: 
                if self.general_consecutive_planning_failures > 0 and (self.last_published_sweep_goal is None or dist_sq_moved_since_last_goal >= (0.1**2)):
                     self.get_logger().info("Robot moved/stuck resolved, resetting general_consecutive_planning_failures & step_distance.")
                     self.general_consecutive_planning_failures = 0
                     if self.step_distance != self.initial_step_distance:
                         self.step_distance = self.initial_step_distance
                         self.get_logger().info(f"Reset step_distance to initial {self.step_distance:.2f}m.")

        if self.general_consecutive_planning_failures >= self.max_general_failures_for_step_reduction and \
           self.step_distance > self.min_sweep_step_distance:
            self.step_distance *= self.sweep_step_reduction_factor
            self.step_distance = max(self.step_distance, self.min_sweep_step_distance)
            self.get_logger().warn(f"General planning failures ({self.general_consecutive_planning_failures}). Reducing sweep step_distance to {self.step_distance:.2f}m.")
            self.general_consecutive_planning_failures = 0 

        goal_for_sweep = self.find_next_sweep_goal(current_robot_pose_map)
        if goal_for_sweep:
            self.publish_goal(goal_for_sweep) 
        else:
            self.get_logger().warn("Zig-zag could not find valid move. Rotating main sweep direction.")
            self.current_direction = (self.current_direction + self.turn_direction) % 4 
            if self.last_robot_pose_at_sweep_goal_pub and dist_sq_moved_since_last_goal < (0.1**2) :
                self.get_logger().warn("Robot stuck AND sweep cannot find next step. Incrementing general failure.")
                self.general_consecutive_planning_failures +=1

    def _check_and_create_goal(self, robot_x_map: float, robot_y_map: float,
                               direction_idx: int, step_dist: float,
                               map_grid_data: np.ndarray, occupancy_map_resolution: float, 
                               map_width: int, map_height: int) -> tuple[PoseStamped | None, bool]: # bool is_unvisited
        dx_world, dy_world = self.direction_vectors[direction_idx]
        goal_x_map_frame = robot_x_map + dx_world * step_dist
        goal_y_map_frame = robot_y_map + dy_world * step_dist
        
        goal_key_check = (round(goal_x_map_frame, 1), round(goal_y_map_frame, 1))
        is_unvisited_in_visited_grid = False # Default to false

        if goal_key_check in self.recently_blacklisted_world_goals:
            self.get_logger().debug(f"Proposed goal {goal_key_check} in dir {direction_idx} is blacklisted.")
            return None, is_unvisited_in_visited_grid

        goal_mx_occ = int((goal_x_map_frame - self.map_origin_x) / occupancy_map_resolution)
        goal_my_occ = int((goal_y_map_frame - self.map_origin_y) / occupancy_map_resolution)
        
        if 0 <= goal_mx_occ < map_width and 0 <= goal_my_occ < map_height:
            map_val_at_goal = map_grid_data[goal_my_occ, goal_mx_occ]
            if map_val_at_goal == 0:  # Free space in occupancy grid
                # Check visited grid
                if self.visited_grid is not None and self.visited_map_resolution > 0:
                    goal_mx_vis = int((goal_x_map_frame - self.map_origin_x) / self.visited_map_resolution)
                    goal_my_vis = int((goal_y_map_frame - self.map_origin_y) / self.visited_map_resolution)
                    if 0 <= goal_mx_vis < self.visited_grid_width and 0 <= goal_my_vis < self.visited_grid_height:
                        if self.visited_grid[goal_my_vis, goal_mx_vis] == 0:
                            is_unvisited_in_visited_grid = True
                    # else: goal is outside visited_grid bounds, treat as not determined or visited.
                else: # Visited grid not available, consider it unvisited for decision making (or neutral)
                    is_unvisited_in_visited_grid = True # Effectively makes it prefer any valid spot if no visited_grid

                goal_pose = PoseStamped()
                goal_pose.header.stamp = self.get_clock().now().to_msg()
                goal_pose.header.frame_id = self.target_map_frame
                goal_pose.pose.position.x = goal_x_map_frame
                goal_pose.pose.position.y = goal_y_map_frame
                goal_pose.pose.position.z = 0.0
                target_yaw = math.atan2(dy_world, dx_world)
                goal_pose.pose.orientation.z = math.sin(target_yaw / 2.0)
                goal_pose.pose.orientation.w = math.cos(target_yaw / 2.0)
                return goal_pose, is_unvisited_in_visited_grid
            else:
                # self.get_logger().debug(f"Proposed goal dir {direction_idx} map ({goal_mx_occ},{goal_my_occ}) not free (val:{map_val_at_goal}).")
                return None, is_unvisited_in_visited_grid
        else:
            # self.get_logger().debug(f"Proposed goal dir {direction_idx} map ({goal_mx_occ},{goal_my_occ}) out of map bounds.")
            return None, is_unvisited_in_visited_grid

    def find_next_sweep_goal(self, current_robot_pose_map: PoseStamped) -> PoseStamped | None:
        if self.map_data is None: self.get_logger().warn("Map data not available for sweep goal finding."); return None
        
        map_info = self.map_data.info
        map_width_occ, map_height_occ = map_info.width, map_info.height
        resolution_occ = map_info.resolution
        
        try:
            map_grid_data = np.array(self.map_data.data, dtype=np.int8).reshape((map_height_occ, map_width_occ))
        except ValueError:
            self.get_logger().error(f"Map data shape mismatch for sweep."); return None
        
        robot_x_map = current_robot_pose_map.pose.position.x
        robot_y_map = current_robot_pose_map.pose.position.y

        if self.is_shifting_lane:
            self.get_logger().info("Sweep: Was shifting lane. Proceeding with new lane direction.")
            self.is_shifting_lane = False

        # 1. Probe forward
        furthest_valid_goal: PoseStamped | None = None
        furthest_valid_goal_dist = 0.0
        best_unvisited_goal: PoseStamped | None = None
        best_unvisited_goal_dist = 0.0

        for i in range(1, self.sweep_max_probe_multiplier + 1):
            current_eval_step_distance = self.step_distance * i
            probe_goal, is_unvisited = self._check_and_create_goal(
                robot_x_map, robot_y_map, self.current_direction, current_eval_step_distance,
                map_grid_data, resolution_occ, map_width_occ, map_height_occ
            )
            if probe_goal:
                furthest_valid_goal = probe_goal # Furthest physically reachable
                furthest_valid_goal_dist = current_eval_step_distance
                if is_unvisited:
                    if best_unvisited_goal is None or current_eval_step_distance > best_unvisited_goal_dist:
                         best_unvisited_goal = probe_goal
                         best_unvisited_goal_dist = current_eval_step_distance
            else: break # Hit obstacle or map edge along probe line
        
        if best_unvisited_goal:
            self.get_logger().info(f"Sweep: Found UNVISITED forward goal in dir {self.current_direction} at {best_unvisited_goal_dist:.2f}m.")
            return best_unvisited_goal
        if furthest_valid_goal:
            self.get_logger().info(f"Sweep: Found VISITED forward goal in dir {self.current_direction} at {furthest_valid_goal_dist:.2f}m (no unvisited).")
            return furthest_valid_goal

        # 2. Forward is blocked or no unvisited preferred. Try to shift lane.
        self.get_logger().info(f"Sweep: End of lane in dir {self.current_direction} or no preferred forward. Attempting shift.")
        original_sweep_direction = self.current_direction
        shift_step_dist = self.step_distance

        # Evaluate preferred shift
        preferred_shift_dir_idx = (original_sweep_direction + self.turn_direction + 4) % 4
        pref_shift_goal, pref_is_unvisited = self._check_and_create_goal(
            robot_x_map, robot_y_map, preferred_shift_dir_idx, shift_step_dist,
            map_grid_data, resolution_occ, map_width_occ, map_height_occ
        )
        # Evaluate alternate shift
        alternate_turn_dir = -self.turn_direction
        alternate_shift_dir_idx = (original_sweep_direction + alternate_turn_dir + 4) % 4
        alt_shift_goal, alt_is_unvisited = self._check_and_create_goal(
            robot_x_map, robot_y_map, alternate_shift_dir_idx, shift_step_dist,
            map_grid_data, resolution_occ, map_width_occ, map_height_occ
        )

        chosen_shift_goal: PoseStamped | None = None
        chosen_shift_dir_idx = -1
        chosen_turn_for_next_lane = 0

        if pref_shift_goal and alt_shift_goal:
            self.get_logger().debug(f"Sweep: Both shifts possible. Pref unvisited: {pref_is_unvisited}, Alt unvisited: {alt_is_unvisited}")
            if pref_is_unvisited and not alt_is_unvisited:
                chosen_shift_goal, chosen_shift_dir_idx, chosen_turn_for_next_lane = pref_shift_goal, preferred_shift_dir_idx, self.turn_direction
            elif not pref_is_unvisited and alt_is_unvisited:
                chosen_shift_goal, chosen_shift_dir_idx, chosen_turn_for_next_lane = alt_shift_goal, alternate_shift_dir_idx, alternate_turn_dir
            else: # Both same, or both false: use preferred turn_direction
                chosen_shift_goal, chosen_shift_dir_idx, chosen_turn_for_next_lane = pref_shift_goal, preferred_shift_dir_idx, self.turn_direction
        elif pref_shift_goal:
            self.get_logger().debug(f"Sweep: Only preferred shift possible. Unvisited: {pref_is_unvisited}")
            chosen_shift_goal, chosen_shift_dir_idx, chosen_turn_for_next_lane = pref_shift_goal, preferred_shift_dir_idx, self.turn_direction
        elif alt_shift_goal:
            self.get_logger().debug(f"Sweep: Only alternate shift possible. Unvisited: {alt_is_unvisited}")
            chosen_shift_goal, chosen_shift_dir_idx, chosen_turn_for_next_lane = alt_shift_goal, alternate_shift_dir_idx, alternate_turn_dir
        
        if chosen_shift_goal:
            self.get_logger().info(f"Sweep: Shifting to new lane via direction {chosen_shift_dir_idx}.")
            self.current_direction = (chosen_shift_dir_idx + chosen_turn_for_next_lane + 4) % 4 # U-turn complete
            self.turn_direction *= -1 # Alternate U-turn preference for next time
            self.is_shifting_lane = True
            return chosen_shift_goal

        self.get_logger().error(f"Sweep: All forward and shift options exhausted from this position.")
        return None

    def exploration_enable_callback(self, msg: Bool):
        old_state = self.exploration_enabled
        self.exploration_enabled = msg.data
        if old_state != self.exploration_enabled:
            self.get_logger().info(f"Exploration {'enabled' if self.exploration_enabled else 'disabled'} by coordinator")
            if not self.exploration_enabled: self.cmd_pub.publish(Twist())

    def cube_marker_callback(self, msg: Marker): 
        if msg.action == Marker.ADD and msg.type == Marker.CUBE: 
            if not self.returning_home and not self.seeking_cube : 
                 self.get_logger().info("Standalone: Cube marker detected, exploration phase ending.")
            self.returning_home = True 
            self.cmd_pub.publish(Twist())

def main(args=None): 
    rclpy.init(args=args)
    frontier_explorer_node = FrontierExplorer()
    try: rclpy.spin(frontier_explorer_node)
    except KeyboardInterrupt: frontier_explorer_node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        frontier_explorer_node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
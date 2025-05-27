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

        # Parameters - declare first
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('goal_topic', 'goal_pose')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('waypoints_topic', 'waypoints')
        self.declare_parameter('cube_pose_topic', 'cube_pose')
        self.declare_parameter('target_map_frame', 'map')
        self.declare_parameter('step_distance', 1.5) # Default initial step distance
        self.declare_parameter('min_sweep_step_distance', 0.5)
        self.declare_parameter('sweep_step_reduction_factor', 0.75)
        self.declare_parameter('qos_depth', 10)
        self.declare_parameter('max_specific_goal_failures', 2) # How many times to try a specific coordinate
        self.declare_parameter('max_general_failures_for_step_reduction', 2) # How many different failed goals before reducing step

        # Get Parameters - now fetch their values
        map_topic_param = self.get_parameter('map_topic').get_parameter_value().string_value
        odom_topic_param = self.get_parameter('odom_topic').get_parameter_value().string_value
        goal_topic_param = self.get_parameter('goal_topic').get_parameter_value().string_value
        cmd_vel_topic_param = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        waypoints_topic_param = self.get_parameter('waypoints_topic').get_parameter_value().string_value
        cube_pose_topic_param = self.get_parameter('cube_pose_topic').get_parameter_value().string_value
        self.target_map_frame = self.get_parameter('target_map_frame').get_parameter_value().string_value
        
        self.initial_step_distance = self.get_parameter('step_distance').get_parameter_value().double_value
        self.step_distance = self.initial_step_distance # Current step distance, can change
        
        self.min_sweep_step_distance = self.get_parameter('min_sweep_step_distance').get_parameter_value().double_value
        self.sweep_step_reduction_factor = self.get_parameter('sweep_step_reduction_factor').get_parameter_value().double_value
        qos_depth = self.get_parameter('qos_depth').get_parameter_value().integer_value
        self.max_specific_goal_failures = self.get_parameter('max_specific_goal_failures').get_parameter_value().integer_value
        self.max_general_failures_for_step_reduction = self.get_parameter('max_general_failures_for_step_reduction').get_parameter_value().integer_value


        # Failure handling attributes
        self.failed_goal_attempts: dict[tuple[float, float], int] = {} # Key: (goal_x_rounded, goal_y_rounded), Value: count
        self.recently_blacklisted_world_goals: deque[tuple[float, float]] = deque(maxlen=10) # Store (x,y) of world goals blacklisted
        self.last_published_sweep_goal: PoseStamped | None = None
        self.last_robot_pose_at_sweep_goal_pub: PoseStamped | None = None
        self.general_consecutive_planning_failures = 0 # Counts distinct sweep goals that seem to fail consecutively

        # Construct full topic names
        self.map_topic_actual = f"{self.robot_namespace}/{map_topic_param}"
        self.odom_topic_actual = f"{self.robot_namespace}/{odom_topic_param}"
        self.goal_topic_actual = f"{self.robot_namespace}/{goal_topic_param}"
        self.cmd_vel_topic_actual = f"{self.robot_namespace}/{cmd_vel_topic_param}"
        self.waypoints_topic_actual = f"{self.robot_namespace}/{waypoints_topic_param}" # Used for cube marker
        self.cube_pose_topic_actual = f"{self.robot_namespace}/{cube_pose_topic_param}"
        
        self.get_logger().info(f"Initial sweep step_distance: {self.step_distance:.2f}m")

        # QoS Profiles and Subscriptions/Publishers (same as before)
        qos_profile_map = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        # ... (rest of QoS and pub/sub setup) ...
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
            # In standalone mode, monitor cube markers to know when to stop exploring
            from visualization_msgs.msg import Marker
            self.cube_marker_sub = self.create_subscription(
                Marker, f"{self.robot_namespace}/cube_marker",
                self.cube_marker_callback, qos_profile_cube)
            self.get_logger().info("Running in standalone mode - monitoring cube markers")            

    # map_callback, get_transformed_pose, cube_callback mostly unchanged
    # odom_callback needs refined logic for resetting failure counts
    def map_callback(self, msg: OccupancyGrid):
        if msg.header.frame_id != self.target_map_frame:
            self.get_logger().warn(
                f"Received map in frame '{msg.header.frame_id}' but expected '{self.target_map_frame}'. "
                f"Map-based calculations might be incorrect if TF is not available/consistent for this frame."
            )
        self.map_data = msg

    def odom_callback(self, msg: Odometry):
        current_pose_stamped = PoseStamped()
        current_pose_stamped.header = msg.header 
        current_pose_stamped.pose = msg.pose.pose
        
        # Check for significant movement since last SWEEP goal publication
        if self.last_published_sweep_goal and self.last_robot_pose_at_sweep_goal_pub:
            # Transform current pose to map frame for consistent comparison
            current_pose_map = self.get_transformed_pose(current_pose_stamped, self.target_map_frame)
            if current_pose_map:
                dist_sq_moved = (current_pose_map.pose.position.x - self.last_robot_pose_at_sweep_goal_pub.pose.position.x)**2 + \
                                (current_pose_map.pose.position.y - self.last_robot_pose_at_sweep_goal_pub.pose.position.y)**2
                
                if dist_sq_moved > (0.25**2): # Moved > 25cm
                    self.get_logger().info(f"Robot moved {math.sqrt(dist_sq_moved):.2f}m since last sweep goal. Resetting failure counts.")
                    # Clear specific attempt counter for the last goal, as we've moved past its relevance for "stuck" detection
                    goal_key_last = (round(self.last_published_sweep_goal.pose.position.x, 2), 
                                     round(self.last_published_sweep_goal.pose.position.y, 2))
                    if goal_key_last in self.failed_goal_attempts:
                        del self.failed_goal_attempts[goal_key_last]
                    
                    self.last_published_sweep_goal = None # We've moved, so the "stuck" check for this goal is over
                    self.last_robot_pose_at_sweep_goal_pub = None
                    
                    if self.general_consecutive_planning_failures > 0:
                        self.general_consecutive_planning_failures = 0
                        self.get_logger().info("Reset general_consecutive_planning_failures.")
                    
                    if self.step_distance != self.initial_step_distance:
                        self.step_distance = self.initial_step_distance
                        self.get_logger().info(f"Reset step_distance to initial {self.step_distance:.2f}m.")
        
        self.robot_pose_odom = current_pose_stamped
        if self.start_pose_odom is None:
            self.start_pose_odom = current_pose_stamped # ... (log start pose)

    def get_transformed_pose(self, input_pose_stamped: PoseStamped, target_frame: str) -> PoseStamped | None:
        # ... (same as your previous working version) ...
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
        # Just stop seeking when coordinator takes over
            if self.seeking_cube:
                self.get_logger().info("Cube detected - coordinator will handle approach")
                self.seeking_cube = False
                self.cube_seen_counter = 0
                # Stop robot movement
                self.cmd_pub.publish(Twist())
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
        else: # ... (rest of cube alignment cmd_vel logic) ...
            self.cube_seen_counter = 0; cmd = Twist(); cmd.angular.z = -0.7 * x_rel_cube; forward_error = y_rel_cube - forward_target_distance
            if abs(x_rel_cube) < (lateral_threshold*2.5) or y_rel_cube > (forward_target_distance+0.8): cmd.linear.x = 0.4 * forward_error
            else: cmd.linear.x = 0.0
            cmd.angular.z = max(min(cmd.angular.z, 0.6), -0.6); cmd.linear.x = max(min(cmd.linear.x, 0.35), -0.35)
            if abs(forward_error) < 0.15 and abs(x_rel_cube) > lateral_threshold: cmd.linear.x = 0.05 * np.sign(forward_error)
            self.cmd_pub.publish(cmd)
        if self.cube_seen_counter >= 25: # ... (rest of cube acquisition logic) ...
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
            # Get current robot pose in map frame for accurate "stuck" detection reference
            self.last_robot_pose_at_sweep_goal_pub = self.get_transformed_pose(self.robot_pose_odom, self.target_map_frame)

            goal_key = (round(final_goal.pose.position.x, 1), round(final_goal.pose.position.y, 1)) # Round to 1 decimal place
            
            current_attempts_for_this_goal = self.failed_goal_attempts.get(goal_key, 0) + 1
            self.failed_goal_attempts[goal_key] = current_attempts_for_this_goal
            self.get_logger().debug(f"Attempt {current_attempts_for_this_goal} for goal key {goal_key}.")

            # If this is a *new distinct* sweep goal we're trying (or first attempt for this one after moving away)
            # and general failures were high, it implies we moved to a new area or are retrying.
            # Reset general_consecutive_planning_failures.
            # A goal is "new" if its attempt count is 1.
            if current_attempts_for_this_goal == 1:
                if self.general_consecutive_planning_failures > 0:
                    self.get_logger().info(f"Trying new sweep goal area {goal_key}. Resetting general_consecutive_planning_failures.")
                    self.general_consecutive_planning_failures = 0
                # If step distance was reduced, and this is a truly new area, reset step_distance
                if self.step_distance != self.initial_step_distance:
                     is_truly_new_area = True
                     # Check if other nearby goals were also failing (e.g. if recently_blacklisted_world_goals is not empty)
                     # For now, simple reset if attempt count is 1
                     self.step_distance = self.initial_step_distance
                     self.get_logger().info(f"Reset step_distance to initial {self.step_distance:.2f}m for new goal area.")


    def timer_callback(self):

        if self.use_mission_coordinator and not self.exploration_enabled:
            return  # Don't explore if coordinator has disabled exploration

        current_robot_pose_map = self.get_transformed_pose(self.robot_pose_odom, self.target_map_frame)
        if self.map_data is None or current_robot_pose_map is None: return
        if self.returning_home: # ... (cube marker logic) ...
            if self.cube_x_global_map is not None: marker = Marker(); marker.header.frame_id = self.target_map_frame; marker.header.stamp = self.get_clock().now().to_msg(); marker.ns = f"{self.robot_namespace}_cube_marker"; marker.id = 0; marker.type = Marker.CUBE; marker.action = Marker.ADD; marker.pose.position.x = self.cube_x_global_map; marker.pose.position.y = self.cube_y_global_map; marker.pose.position.z = 0.15; marker.pose.orientation.w = 1.0; marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3; marker.color.r = 1.0; marker.color.g = 0.5; marker.color.b = 0.0; marker.color.a = 0.95; marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg(); self.marker_pub.publish(MarkerArray(markers=[marker]))
            return
        if self.seeking_cube: return

        # --- Implicit Planner Failure Detection & Strategy Adjustment ---
        if self.last_published_sweep_goal and self.last_robot_pose_at_sweep_goal_pub:
            dist_sq_moved_since_last_goal = (current_robot_pose_map.pose.position.x - self.last_robot_pose_at_sweep_goal_pub.pose.position.x)**2 + \
                                          (current_robot_pose_map.pose.position.y - self.last_robot_pose_at_sweep_goal_pub.pose.position.y)**2
            
            # Key for the last published sweep goal
            goal_key_last = (round(self.last_published_sweep_goal.pose.position.x, 1), 
                             round(self.last_published_sweep_goal.pose.position.y, 1))
            
            # If robot hasn't moved significantly AND we've actually attempted the last goal
            if dist_sq_moved_since_last_goal < (0.1**2) and goal_key_last in self.failed_goal_attempts:
                num_attempts_for_last_goal = self.failed_goal_attempts[goal_key_last]
                
                # STRATEGY 1: Specific goal is repeatedly failing
                if num_attempts_for_last_goal >= self.max_specific_goal_failures:
                    self.get_logger().warn(
                        f"Goal {goal_key_last} failed {num_attempts_for_last_goal} times. Blacklisting and forcing rotation."
                    )
                    self.recently_blacklisted_world_goals.append(goal_key_last)
                    del self.failed_goal_attempts[goal_key_last] # Remove from attempts to allow retrying later if map changes
                    self.current_direction = (self.current_direction + self.turn_direction) % 4
                    self.get_logger().info(f"Forced rotation. New sweep direction idx: {self.current_direction}")
                    
                    # Reset general failure counter & step distance because we took a drastic action (rotation)
                    self.general_consecutive_planning_failures = 0
                    if self.step_distance != self.initial_step_distance:
                        self.step_distance = self.initial_step_distance
                        self.get_logger().info(f"Reset step_distance to initial {self.step_distance:.2f}m after blacklisting.")
                    
                    self.last_published_sweep_goal = None # Clear to avoid re-triggering this exact logic immediately
                    # Try to find a new goal right away in the new direction
                    goal_for_sweep = self.find_next_directional_goal(current_robot_pose_map)
                    if goal_for_sweep: self.publish_goal(goal_for_sweep) # from_sweep=True is default
                    return # Done with this timer call

                # STRATEGY 2: General planning issues, reduce step distance
                # This is triggered if num_attempts_for_last_goal > 1 (meaning it's at least the 2nd try for THIS goal)
                # AND the robot is stuck.
                # We also use general_consecutive_planning_failures to see if *different* goals are failing.
                # Increment general_consecutive_planning_failures only if this is a new failure for this goal.
                # This logic is tricky; let's simplify: if a specific goal fails > 1 time and robot is stuck,
                # that counts towards general failure.
                if num_attempts_for_last_goal > 1: # It's a retry for this specific goal, and we're stuck
                    # This check is now implicitly handled by how general_consecutive_planning_failures is incremented
                    # in publish_goal. If publish_goal is called for the same key again, it means the previous timer call
                    # led to finding the same problematic goal.
                    # Let's use a simple increment here and let publish_goal manage detailed counting.
                    # No, better to manage general_consecutive_planning_failures here based on distinct failed goals.
                    # This is already done by publish_goal: if current_attempts_for_this_goal == 1, it resets general.
                    # So, if we are here, it means general_consecutive_planning_failures *might* be increasing.
                    pass # general_consecutive_planning_failures is managed by publish_goal now.

            else: # Robot moved or no specific goal we are stuck on
                if self.general_consecutive_planning_failures > 0 and self.last_published_sweep_goal is None: # We moved or resolved the previous stuck state
                     self.get_logger().info("Robot moved or stuck state resolved, resetting general_consecutive_planning_failures.")
                     self.general_consecutive_planning_failures = 0
                     if self.step_distance != self.initial_step_distance:
                         self.step_distance = self.initial_step_distance
                         self.get_logger().info(f"Reset step_distance to initial {self.step_distance:.2f}m.")


        # If general planning failures are high, reduce step distance
        # This check should happen *before* find_next_directional_goal if we want the reduction to apply to the upcoming search.
        if self.general_consecutive_planning_failures >= self.max_general_failures_for_step_reduction and \
           self.step_distance > self.min_sweep_step_distance:
            self.step_distance *= self.sweep_step_reduction_factor
            self.step_distance = max(self.step_distance, self.min_sweep_step_distance)
            self.get_logger().warn(f"General planning failures ({self.general_consecutive_planning_failures}) detected. Reducing sweep step_distance to {self.step_distance:.2f}m.")
            self.general_consecutive_planning_failures = 0 # Reset after taking action

        goal_for_sweep = self.find_next_directional_goal(current_robot_pose_map)
        if goal_for_sweep:
            # publish_goal will handle incrementing failed_goal_attempts and general_consecutive_planning_failures
            self.publish_goal(goal_for_sweep) 
        # If find_next_directional_goal returns None, it already rotated.


    def find_next_directional_goal(self, current_robot_pose_map: PoseStamped) -> PoseStamped | None:
        # ... (map setup, robot_x_map, robot_y_map same as before) ...
        map_info = self.map_data.info; width, height = map_info.width, map_info.height; resolution = map_info.resolution
        origin_x_map = map_info.origin.position.x; origin_y_map = map_info.origin.position.y
        try: map_grid_data = np.array(self.map_data.data, dtype=np.int8).reshape((height, width))
        except ValueError: self.get_logger().error(f"Map data shape mismatch: {len(self.map_data.data)} vs {height*width}"); return None
        robot_x_map = current_robot_pose_map.pose.position.x; robot_y_map = current_robot_pose_map.pose.position.y

        for _ in range(4): # Try all 4 directions with current settings
            dx_world, dy_world = self.direction_vectors[self.current_direction]
            current_eval_step_distance = self.step_distance 
            goal_x_map_frame = robot_x_map + dx_world * current_eval_step_distance
            goal_y_map_frame = robot_y_map + dy_world * current_eval_step_distance
            goal_key_check = (round(goal_x_map_frame, 1), round(goal_y_map_frame, 1))

            if goal_key_check in self.recently_blacklisted_world_goals:
                self.get_logger().info(f"Direction {self.current_direction} leads to blacklisted goal {goal_key_check}. Rotating.")
                self.current_direction = (self.current_direction + self.turn_direction) % 4
                continue 

            goal_mx = int((goal_x_map_frame - origin_x_map) / resolution)
            goal_my = int((goal_y_map_frame - origin_y_map) / resolution)
            is_valid_grid_goal = False; map_val_at_goal = -2 
            if 0 <= goal_mx < width and 0 <= goal_my < height:
                map_val_at_goal = map_grid_data[goal_my, goal_mx]
                if map_val_at_goal == 0: is_valid_grid_goal = True
            
            if is_valid_grid_goal: # ... (create and return goal_pose) ...
                goal_pose = PoseStamped(); goal_pose.header.stamp = self.get_clock().now().to_msg(); goal_pose.header.frame_id = self.target_map_frame
                goal_pose.pose.position.x = goal_x_map_frame; goal_pose.pose.position.y = goal_y_map_frame; goal_pose.pose.position.z = 0.0
                target_yaw = math.atan2(dy_world, dx_world)
                goal_pose.pose.orientation.z = math.sin(target_yaw / 2.0); goal_pose.pose.orientation.w = math.cos(target_yaw / 2.0)
                self.get_logger().info(f"Found sweep goal at map ({goal_x_map_frame:.2f}, {goal_y_map_frame:.2f}) with step {current_eval_step_distance:.2f}m, yaw {math.degrees(target_yaw):.1f} deg.")
                return goal_pose
            else: # ... (log reason and rotate) ...
                log_reason = "out of map bounds" if not (0 <= goal_mx < width and 0 <= goal_my < height) else f"not free (val:{map_val_at_goal})"
                self.get_logger().info(f"Sweep step to grid ({goal_mx},{goal_my}) using step_dist {current_eval_step_distance:.2f}m is {log_reason}. Rotating. Old dir: {self.current_direction}")
                self.current_direction = (self.current_direction + self.turn_direction) % 4
        
        self.get_logger().warn(f"All 4 directions failed with current step_distance {self.step_distance:.2f}m.")
        # If all directions fail, it could be because step_distance is too large for a confined space,
        # OR because all reachable areas are blacklisted.
        # The step reduction logic in timer_callback should eventually handle the former.
        # If recently_blacklisted_world_goals fills up, it will eventually clear old ones.
        return None
    def exploration_enable_callback(self, msg: Bool):
        """Handle exploration enable/disable from mission coordinator"""
        old_state = self.exploration_enabled
        self.exploration_enabled = msg.data
        
        if old_state != self.exploration_enabled:
            self.get_logger().info(f"Exploration {'enabled' if self.exploration_enabled else 'disabled'} by coordinator")
            
            # If exploration is disabled, stop current movement
            if not self.exploration_enabled:
                stop_cmd = Twist()
                self.cmd_pub.publish(stop_cmd)

    def cube_marker_callback(self, msg: Marker):
        """Handle cube detection via marker (standalone mode only)"""
        if msg.action == Marker.ADD and msg.type == Marker.CUBE:
            self.get_logger().info("Cube marker detected - stopping exploration")
            self.seeking_cube = True
            # In standalone mode, could transition to cube approach behavior
            # For now, just stop exploring
            stop_cmd = Twist()
            self.cmd_pub.publish(stop_cmd)

def main(args=None): # Unchanged
    rclpy.init(args=args)
    frontier_explorer_node = FrontierExplorer()
    try: rclpy.spin(frontier_explorer_node)
    except KeyboardInterrupt: frontier_explorer_node.get_logger().info("Keyboard interrupt, shutting down FrontierExplorer.")
    finally:
        frontier_explorer_node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
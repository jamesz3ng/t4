import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener # Standard TF2 imports
import tf2_ros # For specific exception types if needed

import numpy as np
import heapq
from scipy.ndimage import maximum_filter # Ensure scipy is installed
import math

from std_msgs.msg import Header
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')

        # Namespace for TOPICS (not necessarily for TF frame IDs if they are published globally)
        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID')
        if self.robot_id_str is None:
            self.get_logger().error("ROS_DOMAIN_ID environment variable not set! Defaulting to 0.")
            self.robot_id_str = "0"
        self.robot_namespace = f"/T{self.robot_id_str}" # e.g., /T30

        # --- Define Expected TF Frame IDs ---
        # Based on typical SLAM setup and your echoed TF data:
        # Map frame is usually global (published by SLAM)
        self.map_tf_frame = "map"
        # Robot's frames, based on your TF echo, appear non-namespaced in the TF tree itself
        self.robot_odom_tf_frame = "odom"
        self.robot_base_footprint_tf_frame = "base_footprint" # Often the primary control frame for TB4
        self.robot_base_link_tf_frame = "base_link"       # Also common

        self.get_logger().info(f"Planning node initialized for ROS_DOMAIN_ID: {self.robot_id_str} (Namespace: {self.robot_namespace})")
        self.get_logger().info(f"Expecting TF Map Frame: '{self.map_tf_frame}'")
        self.get_logger().info(f"Expecting TF Robot Odom Frame: '{self.robot_odom_tf_frame}'")
        self.get_logger().info(f"Expecting TF Robot Base Footprint Frame: '{self.robot_base_footprint_tf_frame}'")
        self.get_logger().info(f"Expecting TF Robot Base Link Frame: '{self.robot_base_link_tf_frame}'")

        self.map_data_storage = None # Renamed to avoid conflict
        self.inflated_map_storage = None # Renamed
        self.map_info_storage = None # Renamed
        self.current_goal_pose = None # Renamed
        self.current_path = [] # Renamed

        # IMPORTANT: For the tf_listener to receive data from namespaced /T<ID>/tf topics,
        # this node (PlanningNode) must either be LAUNCHED INTO THE /T<ID> NAMESPACE,
        # or if launched globally, its /tf and /tf_static subscriptions MUST BE REMAPPED at launch time:
        # e.g., ros2 run your_pkg planning_node --ros-args --remap /tf:=/T<ID>/tf --remap /tf_static:=/T<ID>/tf_static
        self.tf_buffer = Buffer() # Default 10s cache
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True) # Spin listener in a separate thread

        # Parameters for TOPIC base names
        self.declare_parameter('map_topic_base', 'map')
        self.declare_parameter('goal_topic_base', 'goal_pose')
        self.declare_parameter('cmd_vel_topic_base', 'cmd_vel')
        self.declare_parameter('waypoints_topic_base', 'waypoints')

        map_topic_base = self.get_parameter('map_topic_base').get_parameter_value().string_value
        goal_topic_base = self.get_parameter('goal_topic_base').get_parameter_value().string_value
        cmd_vel_topic_base = self.get_parameter('cmd_vel_topic_base').get_parameter_value().string_value
        waypoints_topic_base = self.get_parameter('waypoints_topic_base').get_parameter_value().string_value

        # Construct full TOPIC names (these ARE namespaced)
        self.map_topic_actual = f"{self.robot_namespace}/{map_topic_base}"
        self.goal_topic_actual = f"{self.robot_namespace}/{goal_topic_base}"
        self.cmd_vel_topic_actual = f"{self.robot_namespace}/{cmd_vel_topic_base}"
        self.waypoints_topic_actual = f"{self.robot_namespace}/{waypoints_topic_base}"

        self.get_logger().info(f"Subscribing to map topic: {self.map_topic_actual}")
        self.get_logger().info(f"Subscribing to goal topic: {self.goal_topic_actual}")
        self.get_logger().info(f"Publishing cmd_vel to: {self.cmd_vel_topic_actual}")
        self.get_logger().info(f"Publishing waypoints to: {self.waypoints_topic_actual}")

        qos_map = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_goal = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic_actual, self.map_callback, qos_map)
        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic_actual, self.goal_callback, qos_goal)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic_actual, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.waypoints_topic_actual, 10)

        self.path_execution_timer = self.create_timer(0.1, self.path_following_update) # Renamed timer

    def quaternion_to_yaw(self, q_geometry_msg): # Parameter is geometry_msgs.msg.Quaternion
        # Standard conversion from quaternion to yaw (around Z)
        siny_cosp = 2 * (q_geometry_msg.w * q_geometry_msg.z + q_geometry_msg.x * q_geometry_msg.y)
        cosy_cosp = 1 - 2 * (q_geometry_msg.y * q_geometry_msg.y + q_geometry_msg.z * q_geometry_msg.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def map_callback(self, msg: OccupancyGrid):
        self.get_logger().info(f"Map data received on '{self.map_topic_actual}' (Size: {msg.info.width}x{msg.info.height}, Res: {msg.info.resolution:.3f})", throttle_duration_sec=10.0)
        self.map_data_storage = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info_storage = msg.info
        self.inflate_map_data() # Perform inflation when new map arrives

    def inflate_map_data(self):
        if self.map_data_storage is None or self.map_info_storage is None:
            self.get_logger().warn("Cannot inflate map, map_data_storage or map_info_storage is None.")
            return
        
        occupied_mask = (self.map_data_storage == 100).astype(np.uint8) # Assuming 100 is occupied
        
        # Calculate inflation radius in cells
        # Example: robot_radius = 0.2m, safety_margin = 0.1m. Total desired clearance = 0.3m
        # inflation_radius_meters = 0.3 
        # inflation_radius_cells = math.ceil(inflation_radius_meters / self.map_info_storage.resolution)
        # For scipy.ndimage.maximum_filter, 'size' is more like diameter or width of kernel
        # A disk of radius R cells has a bounding box of roughly (2R+1)x(2R+1)
        # Let's use a fixed cell size for inflation for now, e.g., 5 cells around obstacle
        inflation_kernel_size = 11 # Must be odd for symmetric filter around center; (2*radius_cells + 1)
                                   # size=18 from your original code is a large inflation.
                                   # Let's try a slightly smaller one, adjust as needed.

        self.get_logger().info(f"Inflating map with kernel size: {inflation_kernel_size}")
        # maximum_filter expands occupied regions.
        # Inflated regions will become 1 (from occupied_mask being 1)
        inflated_regions_mask = maximum_filter(occupied_mask, size=inflation_kernel_size)
        
        # Create the inflated map: copy original, then mark inflated regions as occupied (e.g., 100)
        # Be careful not to overwrite unknown space (-1) if you want to preserve it.
        self.inflated_map_storage = self.map_data_storage.copy()
        self.inflated_map_storage[inflated_regions_mask > 0] = 100 # Mark inflated areas as occupied
        
        self.get_logger().info("Map inflation complete.")


    def goal_callback(self, msg: PoseStamped):
        self.get_logger().info(f"Goal received in frame '{msg.header.frame_id}': Pos(x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f})")
        
        # Ensure goal is in the map_tf_frame
        if msg.header.frame_id != self.map_tf_frame:
            self.get_logger().warn(f"Goal is in frame '{msg.header.frame_id}'. Attempting to transform to '{self.map_tf_frame}'.")
            try:
                # Wait for transform to be available, needed for goal transformation
                self.tf_buffer.can_transform(self.map_tf_frame, msg.header.frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
                transformed_goal_pose_stamped = self.tf_buffer.transform(msg, self.map_tf_frame, timeout=rclpy.duration.Duration(seconds=1.0))
                self.current_goal_pose = transformed_goal_pose_stamped.pose
                self.get_logger().info(f"Goal transformed to '{self.map_tf_frame}': Pos(x={self.current_goal_pose.position.x:.2f}, y={self.current_goal_pose.position.y:.2f})")
            except Exception as e:
                self.get_logger().error(f"Failed to transform goal from '{msg.header.frame_id}' to '{self.map_tf_frame}': {e}")
                self.current_goal_pose = None
                self.current_path = [] # Clear path if goal is invalid
                return
        else:
            self.current_goal_pose = msg.pose

        self.plan_new_path() # Renamed

    def world_to_map_grid(self, world_x, world_y):
        if self.map_info_storage is None: return None, None
        origin = self.map_info_storage.origin.position
        res = self.map_info_storage.resolution
        if res == 0: return None, None
        grid_x = int((world_x - origin.x) / res)
        grid_y = int((world_y - origin.y) / res)
        return grid_x, grid_y

    def map_grid_to_world(self, grid_x, grid_y):
        if self.map_info_storage is None: return None, None
        origin = self.map_info_storage.origin.position
        res = self.map_info_storage.resolution
        # Return center of the grid cell
        world_x = grid_x * res + origin.x + res / 2.0
        world_y = grid_y * res + origin.y + res / 2.0
        return world_x, world_y

    def is_grid_cell_occupied(self, grid_x, grid_y):
        if self.inflated_map_storage is None: return True # Assume occupied if no map
        height, width = self.inflated_map_storage.shape
        if 0 <= grid_y < height and 0 <= grid_x < width:
            # For inflated_map_storage: 100 means occupied/inflated
            return self.inflated_map_storage[grid_y, grid_x] >= 90 # Threshold for occupied
        return True # Out of bounds

    def get_a_star_heuristic(self, pos_a, pos_b): # pos are (x,y) tuples
        return np.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1]) # Euclidean distance

    def find_nearest_free_grid_cell(self, goal_grid_xy, max_radius_cells=20):
        if goal_grid_xy[0] is None: return None
        gx, gy = goal_grid_xy
        if not self.is_grid_cell_occupied(gx, gy): return gx, gy # Goal is already free

        for r in range(1, max_radius_cells + 1):
            for dx_offset in range(-r, r + 1):
                for dy_offset in [-r, r]: # Check horizontal segments of square
                    check_x, check_y = gx + dx_offset, gy + dy_offset
                    if not self.is_grid_cell_occupied(check_x, check_y): return check_x, check_y
            for dy_offset in range(-r + 1, r): # Check vertical segments (excluding corners already checked)
                 for dx_offset in [-r, r]:
                    check_x, check_y = gx + dx_offset, gy + dy_offset
                    if not self.is_grid_cell_occupied(check_x, check_y): return check_x, check_y
        return None


    def a_star_planner(self, start_grid_xy, goal_grid_xy):
        if start_grid_xy[0] is None or goal_grid_xy[0] is None: return None
            
        open_set = [] # Priority queue: (f_score, grid_xy_tuple)
        heapq.heappush(open_set, (0, start_grid_xy))
        
        came_from = {} # To reconstruct path: child_xy -> parent_xy
        g_score = {start_grid_xy: 0} # Cost from start to node
        
        # f_score = g_score + heuristic. Initialize f_score for start node.
        f_score = {start_grid_xy: self.get_a_star_heuristic(start_grid_xy, goal_grid_xy)}

        while open_set:
            current_f, current_grid_xy = heapq.heappop(open_set)

            if current_grid_xy == goal_grid_xy:
                # Reconstruct path
                path_reconstructed = []
                temp_xy = current_grid_xy
                while temp_xy in came_from:
                    path_reconstructed.append(temp_xy)
                    temp_xy = came_from[temp_xy]
                path_reconstructed.append(start_grid_xy)
                return list(reversed(path_reconstructed))

            # Explore neighbors (8-connectivity)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                neighbor_grid_xy = (current_grid_xy[0] + dx, current_grid_xy[1] + dy)

                if self.is_grid_cell_occupied(neighbor_grid_xy[0], neighbor_grid_xy[1]):
                    continue # Skip occupied or out-of-bounds neighbors

                # Cost to move to this neighbor (1 for cardinal, sqrt(2) for diagonal)
                movement_cost = np.hypot(dx, dy)
                tentative_g_val = g_score[current_grid_xy] + movement_cost
                
                # If this path to neighbor is better than any previous one
                if tentative_g_val < g_score.get(neighbor_grid_xy, float('inf')):
                    came_from[neighbor_grid_xy] = current_grid_xy
                    g_score[neighbor_grid_xy] = tentative_g_val
                    f_val = tentative_g_val + self.get_a_star_heuristic(neighbor_grid_xy, goal_grid_xy)
                    heapq.heappush(open_set, (f_val, neighbor_grid_xy))
        
        self.get_logger().warn("A* planner could not find a path to the goal.")
        return None


    def plan_new_path(self): # Renamed from plan_path
        if self.map_data_storage is None or self.current_goal_pose is None or self.map_info_storage is None:
            self.get_logger().warn("Cannot plan path: Map data, goal pose, or map info is missing.")
            self.current_path = []
            return

        current_robot_pos_world = None
        # TF Lookup: Try preferred frames first
        # Source frames should be what your robot actually publishes in TF (e.g., "base_footprint")
        source_frames_to_try = [self.robot_base_footprint_tf_frame, self.robot_base_link_tf_frame, self.robot_odom_tf_frame]
        
        for source_frame_id in source_frames_to_try:
            try:
                # Target is self.map_tf_frame (e.g., "map")
                # Source is the robot's frame (e.g., "base_footprint")
                transform_stamped = self.tf_buffer.lookup_transform(
                    self.map_tf_frame, 
                    source_frame_id, 
                    rclpy.time.Time(), # Get latest available transform
                    timeout=rclpy.duration.Duration(seconds=0.2) # Slightly longer timeout for initial planning
                )
                current_robot_pos_world = (transform_stamped.transform.translation.x, transform_stamped.transform.translation.y)
                self.get_logger().info(f"TF lookup successful for planning: Target='{self.map_tf_frame}', Source='{source_frame_id}'")
                break # Use the first successful transform
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, tf2_ros.TimeoutException) as e:
                self.get_logger().debug(f"TF lookup failed for planning: Target='{self.map_tf_frame}', Source='{source_frame_id}'. Error: {e}")
        
        if current_robot_pos_world is None:
            self.get_logger().warn("All TF lookups failed for robot's current pose. Cannot plan path.")
            self.current_path = []
            return

        start_grid_xy = self.world_to_map_grid(current_robot_pos_world[0], current_robot_pos_world[1])
        # Goal pose is assumed to be in self.map_tf_frame from goal_callback
        goal_world_xy = (self.current_goal_pose.position.x, self.current_goal_pose.position.y)
        goal_grid_xy = self.world_to_map_grid(goal_world_xy[0], goal_world_xy[1])
        
        if start_grid_xy[0] is None or goal_grid_xy[0] is None:
            self.get_logger().warn("Failed to convert world start/goal to map grid coordinates.")
            self.current_path = []
            return
            
        if self.is_grid_cell_occupied(goal_grid_xy[0], goal_grid_xy[1]):
            self.get_logger().warn(f"Goal grid cell {goal_grid_xy} is occupied. Finding nearest free cell.")
            effective_goal_grid_xy = self.find_nearest_free_grid_cell(goal_grid_xy)
            if effective_goal_grid_xy is None:
                self.get_logger().warn("No free cell found near goal. Path planning aborted.")
                self.current_path = []
                return
            self.get_logger().info(f"Using new free goal grid cell: {effective_goal_grid_xy}")
            goal_grid_xy = effective_goal_grid_xy # Update goal to the new free cell

        self.get_logger().info(f"Planning path from grid {start_grid_xy} to grid {goal_grid_xy}")
        grid_path = self.a_star_planner(start_grid_xy, goal_grid_xy)
        
        if grid_path:
            self.current_path = [] # Clear previous path
            for gx, gy in grid_path:
                wx, wy = self.map_grid_to_world(gx, gy)
                if wx is not None:
                    self.current_path.append((wx, wy))
            
            if self.current_path:
                self.publish_path_markers()
                self.get_logger().info(f"Path planned with {len(self.current_path)} world waypoints.")
            else:
                self.get_logger().warn("A* found a grid path, but failed to convert all points to world coordinates.")
                self.current_path = [] # Ensure path is cleared if conversion fails
        else:
            self.get_logger().warn("A* planner failed to find a path.")
            self.current_path = [] # Ensure path is cleared

    def publish_path_markers(self):
        if not self.current_path: return

        marker_array = MarkerArray()
        # Header for all markers in the array, and for DELETEALL
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.map_tf_frame)

        # Add a DELETEALL marker to clear previous path markers
        delete_all_marker = Marker()
        delete_all_marker.header = header
        delete_all_marker.ns = f"path_waypoints_{self.robot_namespace}" # Use same namespace to delete correctly
        delete_all_marker.id = 0 # ID doesn't strictly matter for DELETEALL if ns is specific enough
        delete_all_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_all_marker)
        
        # Add new path markers
        for i, (world_x, world_y) in enumerate(self.current_path):
            marker = Marker()
            marker.header = header
            marker.ns = f"path_waypoints_{self.robot_namespace}"
            marker.id = i + 1 # Start IDs from 1 to avoid conflict with DELETEALL if it used ID 0 for deletion scope
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = world_x
            marker.pose.position.y = world_y
            marker.pose.position.z = 0.1  # Slightly above the ground
            marker.pose.orientation.w = 1.0 # Default orientation
            marker.scale.x = 0.1; marker.scale.y = 0.1; marker.scale.z = 0.1;
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8) # Green, slightly transparent
            marker.lifetime = rclpy.duration.Duration(seconds=60).to_msg() # Auto-delete after 60s
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

    def path_following_update(self): # Renamed from update
        if not self.current_path:
            # Optionally publish a zero Twist if path is empty and robot should stop
            # self.cmd_pub.publish(Twist()) 
            return

        current_robot_pos_world = None
        current_robot_yaw_world = None

        # TF Lookup for current robot pose
        source_frames_to_try = [self.robot_base_footprint_tf_frame, self.robot_base_link_tf_frame]

        for source_frame_id in source_frames_to_try:
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    self.map_tf_frame,
                    source_frame_id,
                    rclpy.time.Time(), # Latest transform
                    timeout=rclpy.duration.Duration(seconds=0.05) # Short timeout for control loop
                )
                current_robot_pos_world = (transform_stamped.transform.translation.x, transform_stamped.transform.translation.y)
                current_robot_yaw_world = self.quaternion_to_yaw(transform_stamped.transform.rotation)
                # self.get_logger().debug(f"TF successful for update: '{self.map_tf_frame}' <- '{source_frame_id}'")
                break # Success
            except Exception as e:
                self.get_logger().debug(f"TF lookup failed for update ('{self.map_tf_frame}' <- '{source_frame_id}'): {e}", throttle_duration_sec=2.0)
        
        if current_robot_pos_world is None:
            self.get_logger().warn("TF lookup failed in path following update. Sending zero velocity.", throttle_duration_sec=2.0)
            self.cmd_pub.publish(Twist()) # Stop robot if its pose is unknown
            return

        robot_x, robot_y = current_robot_pos_world
        
        # Get current target waypoint
        target_wp_x, target_wp_y = self.current_path[0]

        error_x = target_wp_x - robot_x
        error_y = target_wp_y - robot_y
        distance_to_wp = np.hypot(error_x, error_y)
        
        cmd = Twist()
        arrival_threshold = 0.15 # meters

        if distance_to_wp < arrival_threshold:
            self.current_path.pop(0)
            self.get_logger().info(f"Reached waypoint. {len(self.current_path)} remaining.")
            if not self.current_path:
                self.get_logger().info("Path successfully completed! Stopping.")
                self.cmd_pub.publish(Twist()) # Publish once to stop
                return # Path finished
            else:
                # Moving to next waypoint, publish stop to handle transition smoothly
                self.cmd_pub.publish(Twist())
                return # Let next cycle pick up new target

        # Proportional controller for path following
        angle_to_wp = math.atan2(error_y, error_x)
        heading_error = angle_to_wp - current_robot_yaw_world

        # Normalize heading_error to [-pi, pi]
        while heading_error > math.pi: heading_error -= 2 * math.pi
        while heading_error < -math.pi: heading_error += 2 * math.pi

        # --- Control Gains (Tune these!) ---
        k_angular = 1.0  # Proportional gain for angular velocity
        k_linear = 0.3   # Proportional gain for linear velocity (or fixed speed if preferred)
        max_lin_vel = 0.22 # m/s (TurtleBot 4 typical max is ~0.3)
        max_ang_vel = 0.8  # rad/s
        turn_in_place_threshold = math.radians(30) # If angle error > 30 deg, turn first

        if abs(heading_error) > turn_in_place_threshold:
            cmd.linear.x = 0.0 # Turn in place
        else:
            # Scale linear velocity by distance, but cap it
            cmd.linear.x = min(k_linear * distance_to_wp, max_lin_vel)
            # Reduce speed if turning sharply (optional)
            # cmd.linear.x *= max(0.0, 1.0 - abs(heading_error) / (math.pi/2))

        cmd.angular.z = k_angular * heading_error
        
        # Clamp velocities
        cmd.linear.x = max(0.0, min(cmd.linear.x, max_lin_vel)) # Ensure non-negative and capped
        cmd.angular.z = max(-max_ang_vel, min(cmd.angular.z, max_ang_vel))

        self.cmd_pub.publish(cmd)
        # self.get_logger().debug(f"DistToWP: {distance_to_wp:.2f}, HdgErr: {math.degrees(heading_error):.1f}, LinX: {cmd.linear.x:.2f}, AngZ: {cmd.angular.z:.2f}", throttle_duration_sec=0.5)

def main(args=None):
    rclpy.init(args=args)
    planning_node = PlanningNode()
    try:
        rclpy.spin(planning_node)
    except KeyboardInterrupt:
        planning_node.get_logger().info("Keyboard interrupt, shutting down PlanningNode.")
    except Exception as e:
        planning_node.get_logger().error(f"Unhandled exception in spin: {e}", exc_info=True)
    finally:
        planning_node.get_logger().info("Node shutdown sequence initiated.")
        if hasattr(planning_node, 'cmd_pub') and planning_node.cmd_pub is not None and rclpy.ok():
            try:
                # Ensure publisher is valid before publishing
                if hasattr(planning_node.cmd_pub, 'handle') and planning_node.cmd_pub.handle :
                    planning_node.get_logger().info("Publishing zero velocity as part of shutdown.")
                    planning_node.cmd_pub.publish(Twist()) # Send one last stop command
                else:
                    planning_node.get_logger().warn("cmd_pub handle is invalid during shutdown. Cannot send stop.")
            except Exception as e_pub:
                planning_node.get_logger().error(f"Error publishing stop Twist on shutdown: {e_pub}")
        
        if hasattr(planning_node, 'destroy_node'): # Check before calling
            planning_node.destroy_node()
            planning_node.get_logger().info("PlanningNode destroyed.")
        if rclpy.ok():
            rclpy.shutdown()
            planning_node.get_logger().info("RCLPY shutdown complete.")

if __name__ == '__main__':
    main()
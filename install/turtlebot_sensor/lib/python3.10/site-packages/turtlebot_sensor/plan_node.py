import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist, Pose # Added Pose for marker
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from sensor_msgs.msg import LaserScan # PointCloud2 removed
from tf2_ros import Buffer, TransformListener
# import tf2_ros # Not needed directly if Buffer and TransformListener are imported

import numpy as np
import heapq
from scipy.ndimage import maximum_filter
import math
import time
# from enum import Enum # Removed Enum

# Removed: from sensor_msgs_py import point_cloud2 as pc2
# Removed: from std_msgs.msg import Bool
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Removed ObstacleAvoidanceState Enum

class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')

        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID')
        if self.robot_id_str is None:
            self.get_logger().error("ROS_DOMAIN_ID environment variable not set! Defaulting to 0.")
            self.robot_id_str = "0"
        self.robot_namespace = f"/T{self.robot_id_str}"

        self.map_tf_frame = "map"
        self.robot_odom_tf_frame = "odom" # Keep for potential future use, not directly used in this version
        self.robot_base_footprint_tf_frame = "base_footprint"
        self.robot_base_link_tf_frame = "base_link" # Often the source frame for TF

        self.get_logger().info(f"Planning node initialized for ROS_DOMAIN_ID: {self.robot_id_str} (Namespace: {self.robot_namespace})")
        self.get_logger().info(f"Using map frame: '{self.map_tf_frame}'")
        self.get_logger().info(f"Expecting robot pose in TF from '{self.robot_base_footprint_tf_frame}' or '{self.robot_base_link_tf_frame}' to '{self.map_tf_frame}'")

        self.map_data_storage = None
        self.inflated_map_storage = None
        self.map_info_storage = None
        self.current_goal_pose = None # Stores geometry_msgs/Pose
        self.current_path = [] # List of (world_x, world_y) tuples

        # --- Obstacle Avoidance and Bumper Code Removed ---
        # self.obstacle_avoidance_state = ObstacleAvoidanceState.NORMAL # Removed
        # self.left_bumper_hit = False # Removed
        # self.right_bumper_hit = False # Removed
        # self.front_bumper_hit = False # Removed
        # self.obstacle_avoidance_start_time = 0.0 # Removed
        # self.robot_pose_when_obstacle_detected = None # Removed
        # self.replanning_attempts = 0 # Removed (can be re-added for general A* retries if needed)
        # self.max_replanning_attempts = 3 # Removed

        self.tf_buffer = Buffer()
        # spin_thread=True creates a daemon thread for TF listener.
        # This thread can sometimes cause unclean shutdown messages (RCLError)
        # if rclpy context is invalidated before it fully stops.
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.declare_parameter('map_topic_base', 'map')
        self.declare_parameter('goal_topic_base', 'goal_pose')
        self.declare_parameter('cmd_vel_topic_base', 'cmd_vel')
        self.declare_parameter('waypoints_topic_base', 'waypoints')
        # self.declare_parameter('bumper_topic_base', 'mobile_base/sensors/bumper_pointcloud') # Removed
        self.declare_parameter('scan_topic_base', 'scan')
        self.declare_parameter('a_star_max_nodes', 60000) # Added parameter for A*

        map_topic_base = self.get_parameter('map_topic_base').get_parameter_value().string_value
        goal_topic_base = self.get_parameter('goal_topic_base').get_parameter_value().string_value
        cmd_vel_topic_base = self.get_parameter('cmd_vel_topic_base').get_parameter_value().string_value
        waypoints_topic_base = self.get_parameter('waypoints_topic_base').get_parameter_value().string_value
        # bumper_topic_base = self.get_parameter('bumper_topic_base').get_parameter_value().string_value # Removed
        scan_topic_base = self.get_parameter('scan_topic_base').get_parameter_value().string_value
        self.a_star_max_nodes = self.get_parameter('a_star_max_nodes').get_parameter_value().integer_value


        self.map_topic_actual = f"{self.robot_namespace}/{map_topic_base}"
        self.goal_topic_actual = f"{self.robot_namespace}/{goal_topic_base}"
        self.cmd_vel_topic_actual = f"{self.robot_namespace}/{cmd_vel_topic_base}"
        self.waypoints_topic_actual = f"{self.robot_namespace}/{waypoints_topic_base}"
        # self.bumper_topic_actual = f"{self.robot_namespace}/{bumper_topic_base}" # Removed
        self.scan_topic_actual = f"{self.robot_namespace}/{scan_topic_base}"

        self.get_logger().info(f"Subscribing to map topic: {self.map_topic_actual}")
        self.get_logger().info(f"Subscribing to goal topic: {self.goal_topic_actual}")
        # self.get_logger().info(f"Subscribing to bumper (PointCloud2) topic: {self.bumper_topic_actual}") # Removed
        self.get_logger().info(f"Subscribing to scan topic: {self.scan_topic_actual}")
        self.get_logger().info(f"Publishing cmd_vel to: {self.cmd_vel_topic_actual}")
        self.get_logger().info(f"Publishing waypoints to: {self.waypoints_topic_actual}")
        self.get_logger().info(f"A* max nodes: {self.a_star_max_nodes}")


        qos_map = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_goal = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic_actual, self.map_callback, qos_map)
        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic_actual, self.goal_callback, qos_goal)
        
        # Bumper subscription removed
        # LaserScan for proximity detection (basic safety)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic_actual, self.scan_callback, qos_sensor)
        
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic_actual, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.waypoints_topic_actual, 10)

        self.path_execution_timer = self.create_timer(0.2, self.path_following_update) # Timer for path following logic

    def quaternion_to_yaw(self, q_geometry_msg):
        # q_geometry_msg is geometry_msgs.msg.Quaternion
        siny_cosp = 2 * (q_geometry_msg.w * q_geometry_msg.z + q_geometry_msg.x * q_geometry_msg.y)
        cosy_cosp = 1 - 2 * (q_geometry_msg.y * q_geometry_msg.y + q_geometry_msg.z * q_geometry_msg.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # --- Bumper related methods removed ---
    # reset_bumper_flags
    # bumper_pointcloud_callback
    # initiate_obstacle_avoidance
    # execute_obstacle_avoidance

    def scan_callback(self, msg: LaserScan):
        if not msg.ranges:
            self.get_logger().debug("Scan callback: Empty ranges.", throttle_duration_sec=5.0)
            return
        
        # This scan callback is for very close obstacles as a last-resort safety.
        # Should be tuned to be just in front of the robot, further than physical bumper but closer than typical planning avoidance.
        min_distance_laser_stop = 0.15  # e.g., 15cm - if an obstacle is this close, stop.

        # Consider a narrow frontal cone for this emergency stop.
        num_ranges = len(msg.ranges)
        center_index = num_ranges // 2
        # e.g., +/- 15 degrees from center. angle_increment is in radians.
        # Assuming laser scans symmetrically around 0 radians (forward).
        angle_fov_deg = 30 
        half_fov_rad = math.radians(angle_fov_deg / 2.0)
        
        if msg.angle_increment == 0: # Avoid division by zero
            self.get_logger().warn("Scan callback: angle_increment is zero.", throttle_duration_sec=5.0)
            return

        num_rays_half_fov = int(half_fov_rad / msg.angle_increment)
        
        start_idx = max(0, center_index - num_rays_half_fov)
        end_idx = min(num_ranges, center_index + num_rays_half_fov + 1) # +1 because slice end is exclusive
        
        front_ranges = msg.ranges[start_idx:end_idx]
        
        # Filter for valid ranges that are too close
        critically_close_ranges = [r for r in front_ranges if msg.range_min < r < msg.range_max and r < min_distance_laser_stop]
        
        if critically_close_ranges:
            self.get_logger().warn(
                f"LASERSCAN: Critically close obstacle detected at {min(critically_close_ranges):.2f}m (within {min_distance_laser_stop}m). "
                f"Stopping robot and clearing current path."
            )
            stop_cmd = Twist() # Zero velocities
            self.cmd_pub.publish(stop_cmd)
            self.current_path = []
            # self.current_goal_pose = None # Optionally clear the goal too, to prevent immediate replanning into the obstacle
            self.publish_path_markers() # Update markers to show no path

    def get_current_robot_pose_in_map(self):
        """Gets the robot's current pose (x, y, yaw) in the map frame."""
        source_frames_to_try = [self.robot_base_footprint_tf_frame, self.robot_base_link_tf_frame]
        for source_frame_id in source_frames_to_try:
            try:
                # Use rclpy.time.Time(seconds=0) for latest available transform
                transform_stamped = self.tf_buffer.lookup_transform(
                    self.map_tf_frame,
                    source_frame_id,
                    rclpy.time.Time(seconds=0), # Get the latest available transform
                    timeout=rclpy.duration.Duration(seconds=0.1) # Short timeout
                )
                return {
                    'x': transform_stamped.transform.translation.x,
                    'y': transform_stamped.transform.translation.y,
                    'yaw': self.quaternion_to_yaw(transform_stamped.transform.rotation)
                }
            except Exception as e:
                self.get_logger().debug(
                    f"get_current_robot_pose_in_map: TF lookup failed from '{source_frame_id}' to '{self.map_tf_frame}': {e}",
                    throttle_duration_sec=2.0
                )
        return None # Return None if TF lookup fails for all tried frames

    def map_callback(self, msg: OccupancyGrid):
        self.get_logger().info(f"Map data received (Size: {msg.info.width}x{msg.info.height}, Res: {msg.info.resolution:.3f})", throttle_duration_sec=10.0)
        self.map_data_storage = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info_storage = msg.info
        self.inflate_map_data() # Re-inflate map whenever new map data arrives

    def inflate_map_data(self):
        if self.map_data_storage is None or self.map_info_storage is None:
            self.get_logger().warn("Cannot inflate map, map_data_storage or map_info_storage is None.")
            return
        
        # Treat cells with high probability of being occupied (e.g., >= 90) as obstacles for inflation
        occupied_mask = (self.map_data_storage >= 90).astype(np.uint8)
        
        inflation_radius_meters = 0.27 # Define robot radius + safety margin
        if self.map_info_storage.resolution == 0:
            self.get_logger().error("Map resolution is zero, cannot calculate inflation radius in cells.")
            return
        
        inflation_radius_cells = math.ceil(inflation_radius_meters / self.map_info_storage.resolution)
        # Kernel size must be odd for maximum_filter symmetric behavior
        inflation_kernel_size = 2 * inflation_radius_cells + 1
        # Use maximum_filter to expand occupied regions
        inflated_regions_mask = maximum_filter(occupied_mask, size=inflation_kernel_size)
        
        self.inflated_map_storage = self.map_data_storage.copy()
        # Mark inflated areas as occupied (e.g., 100), but only where inflated_regions_mask is true.
        # This ensures we don't turn free space (0) into occupied (100) unless it's truly part of an inflation zone.
        # Also, don't overwrite unknown (-1) unless it's being inflated from a known obstacle.
        self.inflated_map_storage[inflated_regions_mask > 0] = 100 
        

    def goal_callback(self, msg: PoseStamped):
        self.get_logger().info(f"Goal received in frame '{msg.header.frame_id}': Pos(x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f})")
        
        # Reset relevant states for a new goal
        self.current_path = [] # Clear previous path

        transformed_goal_pose = msg.pose # Default to using the pose directly if already in map frame
        if msg.header.frame_id != self.map_tf_frame:
            self.get_logger().warn(f"Goal is in frame '{msg.header.frame_id}'. Attempting to transform to '{self.map_tf_frame}'.")
            try:
                # Ensure TF buffer can transform and then perform the transform
                # Use a timeout for can_transform as well, though lookup_transform will also timeout
                self.tf_buffer.can_transform(self.map_tf_frame, msg.header.frame_id, rclpy.time.Time(seconds=0), timeout=rclpy.duration.Duration(seconds=1.0))
                transformed_goal_stamped = self.tf_buffer.transform(msg, self.map_tf_frame, timeout=rclpy.duration.Duration(seconds=1.0))
                transformed_goal_pose = transformed_goal_stamped.pose
                self.get_logger().info(f"Goal transformed to '{self.map_tf_frame}': Pos(x={transformed_goal_pose.position.x:.2f}, y={transformed_goal_pose.position.y:.2f})")
            except Exception as e:
                self.get_logger().error(f"Failed to transform goal from '{msg.header.frame_id}' to '{self.map_tf_frame}': {e}")
                self.current_goal_pose = None # Invalidate goal
                self.current_path = []
                self.publish_path_markers() # Clear markers
                return
        
        self.current_goal_pose = transformed_goal_pose # Store geometry_msgs/Pose
        self.plan_new_path()

    def world_to_map_grid(self, world_x, world_y):
        if self.map_info_storage is None: 
            self.get_logger().error("world_to_map_grid: map_info_storage is None.")
            return None, None
        origin_x = self.map_info_storage.origin.position.x
        origin_y = self.map_info_storage.origin.position.y
        res = self.map_info_storage.resolution
        if res == 0: 
            self.get_logger().error("world_to_map_grid: map resolution is zero.")
            return None, None
        grid_x = int((world_x - origin_x) / res)
        grid_y = int((world_y - origin_y) / res)
        return grid_x, grid_y

    def map_grid_to_world(self, grid_x, grid_y):
        if self.map_info_storage is None: 
            self.get_logger().error("map_grid_to_world: map_info_storage is None.")
            return None, None
        origin_x = self.map_info_storage.origin.position.x
        origin_y = self.map_info_storage.origin.position.y
        res = self.map_info_storage.resolution
        # Calculate center of the grid cell
        world_x = grid_x * res + origin_x + res / 2.0
        world_y = grid_y * res + origin_y + res / 2.0
        return world_x, world_y

    def is_grid_cell_occupied(self, grid_x, grid_y):
        if self.inflated_map_storage is None:
            self.get_logger().warn("is_grid_cell_occupied: Inflated map is None, assuming occupied.", throttle_duration_sec=5.0)
            return True # Safety: assume occupied if map not available
        
        height, width = self.inflated_map_storage.shape
        if 0 <= grid_y < height and 0 <= grid_x < width:
            cell_value = self.inflated_map_storage[grid_y, grid_x]
            # Occupied if value is high (e.g., >=90, indicating obstacle or inflated region)
            # or if value is -1 (unknown space, treat as occupied for safety)
            return cell_value >= 90 or cell_value == -1 
        self.get_logger().debug(f"is_grid_cell_occupied: Cell ({grid_x}, {grid_y}) is out of map bounds ({width}x{height}). Assuming occupied.", throttle_duration_sec=5.0)
        return True # Out of bounds is treated as occupied

    def get_a_star_heuristic(self, pos_a_grid, pos_b_grid):
        """Euclidean distance heuristic for A*."""
        return np.hypot(pos_a_grid[0] - pos_b_grid[0], pos_a_grid[1] - pos_b_grid[1])

    def find_nearest_free_grid_cell(self, center_grid_xy, max_radius_cells=30):
        if center_grid_xy[0] is None or center_grid_xy[1] is None:
            self.get_logger().error("find_nearest_free_grid_cell: Invalid center_grid_xy input (None).")
            return None
        
        gx, gy = center_grid_xy
        if not self.is_grid_cell_occupied(gx, gy):
            self.get_logger().debug(f"Cell ({gx}, {gy}) is already free.")
            return gx, gy # Already free

        self.get_logger().info(f"Searching for free cell near ({gx}, {gy}) within radius {max_radius_cells}")
        for r in range(1, max_radius_cells + 1):
            # Iterate over cells on the perimeter of a square of radius r
            for dx_offset in range(-r, r + 1):
                for dy_offset in [-r, r] if abs(dx_offset) != r else range(-r, r + 1): # Avoid double-checking corners
                    if dx_offset == 0 and dy_offset == 0: continue # Skip center

                    check_x, check_y = gx + dx_offset, gy + dy_offset
                    if not self.is_grid_cell_occupied(check_x, check_y):
                        self.get_logger().info(f"Found free cell ({check_x}, {check_y}) at radius {r} (offset {dx_offset},{dy_offset}) from ({gx},{gy}).")
                        return check_x, check_y
        
        self.get_logger().warn(f"No free cell found within {max_radius_cells} cells of ({gx}, {gy}).")
        return None

    def a_star_planner(self, start_grid_xy, goal_grid_xy):
        self.get_logger().info(f"A* attempting to plan from {start_grid_xy} to {goal_grid_xy}")
        if start_grid_xy[0] is None or start_grid_xy[1] is None or \
           goal_grid_xy[0] is None or goal_grid_xy[1] is None:
            self.get_logger().error("A*: Start or goal grid coordinates are None. Cannot plan.")
            return None
        
        # Handle if start position is occupied (e.g., due to map updates or inflation)
        effective_start_grid_xy = start_grid_xy
        if self.is_grid_cell_occupied(start_grid_xy[0], start_grid_xy[1]):
            self.get_logger().warn(f"A*: Original start position {start_grid_xy} is occupied! Finding nearest free cell.")
            new_start = self.find_nearest_free_grid_cell(start_grid_xy, max_radius_cells=5) # Small radius for start
            if new_start is None:
                self.get_logger().error("A*: Could not find a free cell near occupied start. Planning failed.")
                return None
            self.get_logger().warn(f"A*: Using new start {new_start} instead of {start_grid_xy}.")
            effective_start_grid_xy = new_start
            
        self.get_logger().info(f"A* effective planning from {effective_start_grid_xy} to {goal_grid_xy}")

        open_set_pq = []  # Priority queue (min-heap) storing (f_score, node_xy)
        heapq.heappush(open_set_pq, (0 + self.get_a_star_heuristic(effective_start_grid_xy, goal_grid_xy), effective_start_grid_xy))
        
        came_from = {}  # Stores parent of a node in the path: {node_xy: parent_node_xy}
        g_score = {effective_start_grid_xy: 0}  # Cost from start to node_xy

        nodes_explored_count = 0
        
        # 8-way movement: (dx, dy)
        movements = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        while open_set_pq and nodes_explored_count < self.a_star_max_nodes:
            current_f_val, current_node_xy = heapq.heappop(open_set_pq)
            nodes_explored_count += 1

            # Optimization: If we re-added a node to open_set_pq with a higher f_score (shouldn't happen with correct g_score check),
            # and it's popped later, its g_score would be higher than the already processed one.
            # However, a simple check if g_score for current_node_xy is worse than current_f_val - heuristic can also work.
            # For simplicity, we rely on the g_score check before adding to ensure only better paths are considered.

            if current_node_xy == goal_grid_xy:
                # Goal reached, reconstruct path
                path_reconstructed = []
                temp_xy = goal_grid_xy
                while temp_xy in came_from:
                    path_reconstructed.append(temp_xy)
                    temp_xy = came_from[temp_xy]
                path_reconstructed.append(effective_start_grid_xy) # Add start node
                self.get_logger().info(f"A* found path with {len(path_reconstructed)} waypoints after exploring {nodes_explored_count} nodes.")
                return list(reversed(path_reconstructed))

            for dx, dy in movements:
                neighbor_node_xy = (current_node_xy[0] + dx, current_node_xy[1] + dy)
                
                if self.is_grid_cell_occupied(neighbor_node_xy[0], neighbor_node_xy[1]):
                    continue # Skip occupied or out-of-bounds neighbors
                
                # Cost to move from current_node_xy to neighbor_node_xy (1 for cardinal, sqrt(2) for diagonal)
                movement_cost = np.hypot(dx, dy) 
                
                # g_score for current_node_xy should always exist if it was popped from open_set_pq
                tentative_g_val_for_neighbor = g_score[current_node_xy] + movement_cost

                if tentative_g_val_for_neighbor < g_score.get(neighbor_node_xy, float('inf')):
                    # This path to neighbor is better than any previously found. Record it.
                    came_from[neighbor_node_xy] = current_node_xy
                    g_score[neighbor_node_xy] = tentative_g_val_for_neighbor
                    
                    heuristic_val_for_neighbor = self.get_a_star_heuristic(neighbor_node_xy, goal_grid_xy)
                    f_val_for_neighbor = tentative_g_val_for_neighbor + heuristic_val_for_neighbor
                    
                    heapq.heappush(open_set_pq, (f_val_for_neighbor, neighbor_node_xy))
        
        # If loop finishes, path not found or max nodes limit reached
        if nodes_explored_count >= self.a_star_max_nodes:
            self.get_logger().warn(f"A* failed: Reached max_nodes limit ({self.a_star_max_nodes}) after exploring {nodes_explored_count} nodes. Goal {goal_grid_xy} not reached.")
        else: # open_set_pq is empty
            self.get_logger().warn(f"A* failed: Open set became empty after exploring {nodes_explored_count} nodes. Goal {goal_grid_xy} not reached.")
        return None

    def smooth_path(self, path_grid_coords, min_dist_sq_world=0.09): # min_dist_world=0.3m
        if not path_grid_coords or len(path_grid_coords) < 2:
            return path_grid_coords # No smoothing needed for short or empty paths
        
        # Convert grid path to world coordinates first for distance check
        path_world_coords = []
        for gx, gy in path_grid_coords:
            wx, wy = self.map_grid_to_world(gx, gy)
            if wx is not None and wy is not None:
                path_world_coords.append((wx, wy))
            else:
                self.get_logger().warn(f"smooth_path: Failed to convert grid {gx},{gy} to world. Skipping point.")
        
        if len(path_world_coords) < 2:
            return path_world_coords # Return original (possibly truncated) if conversion failed badly

        smoothed_world_path = [path_world_coords[0]]
        for i in range(1, len(path_world_coords) - 1): # Iterate up to second to last point
            current_wp_world = path_world_coords[i]
            last_kept_wp_world = smoothed_world_path[-1]
            
            dx_world = current_wp_world[0] - last_kept_wp_world[0]
            dy_world = current_wp_world[1] - last_kept_wp_world[1]
            dist_sq_world = dx_world*dx_world + dy_world*dy_world
            
            if dist_sq_world >= min_dist_sq_world:
                smoothed_world_path.append(current_wp_world)
        
        smoothed_world_path.append(path_world_coords[-1]) # Always add the last point
        
        self.get_logger().info(f"Path smoothed from {len(path_world_coords)} to {len(smoothed_world_path)} world waypoints.")
        return smoothed_world_path


    def get_lookahead_point(self, robot_x_world, robot_y_world, lookahead_dist_world):
        if not self.current_path: # current_path stores world coordinates
            self.get_logger().debug("get_lookahead_point: No current path available.", throttle_duration_sec=5.0)
            return robot_x_world, robot_y_world # Return current robot position if no path

        closest_idx = 0
        min_dist_sq_to_path = float('inf')

        # Find the closest point on the path to the robot
        for i, (wp_x, wp_y) in enumerate(self.current_path):
            dist_sq = (wp_x - robot_x_world)**2 + (wp_y - robot_y_world)**2
            if dist_sq < min_dist_sq_to_path:
                min_dist_sq_to_path = dist_sq
                closest_idx = i
        
        # Starting from the closest point, find a point on the path at lookahead_dist_world
        accumulated_dist = 0.0
        # Project robot onto the segment starting from closest_idx might be more robust
        # For now, simpler: advance along path from closest_idx
        
        for i in range(closest_idx, len(self.current_path)):
            if i == len(self.current_path) - 1: # If this is the last waypoint
                return self.current_path[i] # Target the last waypoint

            current_wp_world = self.current_path[i]
            next_wp_world = self.current_path[i+1]
            
            segment_dx = next_wp_world[0] - current_wp_world[0]
            segment_dy = next_wp_world[1] - current_wp_world[1]
            segment_dist = np.hypot(segment_dx, segment_dy)
            
            if segment_dist < 1e-6: # Avoid division by zero for zero-length segments
                continue

            # If the robot is effectively "on" this segment or past it (relevant to accumulated_dist)
            # and the lookahead point falls within this segment
            if accumulated_dist + segment_dist >= lookahead_dist_world:
                remaining_dist_on_segment = lookahead_dist_world - accumulated_dist
                ratio = remaining_dist_on_segment / segment_dist
                lookahead_x = current_wp_world[0] + ratio * segment_dx
                lookahead_y = current_wp_world[1] + ratio * segment_dy
                return lookahead_x, lookahead_y
            
            accumulated_dist += segment_dist
            
        # If lookahead distance is beyond the end of the path from closest_idx, target the last waypoint
        return self.current_path[-1]


    def cleanup_passed_waypoints(self, robot_x_world, robot_y_world, arrival_threshold_sq_world):
        """Removes waypoints from the front of self.current_path if the robot is close enough."""
        while self.current_path:
            wp_x, wp_y = self.current_path[0] # First waypoint in world coordinates
            dist_sq_to_wp = (wp_x - robot_x_world)**2 + (wp_y - robot_y_world)**2
            if dist_sq_to_wp < arrival_threshold_sq_world:
                self.current_path.pop(0)
                self.get_logger().debug(f"Removed passed waypoint. {len(self.current_path)} remaining.")
            else:
                break # First waypoint is still ahead / not yet reached

    def plan_new_path(self):
        if self.map_data_storage is None or self.current_goal_pose is None or self.map_info_storage is None:
            self.get_logger().warn("Cannot plan new path: Missing map, goal, or map info.")
            self.current_path = []
            self.publish_path_markers() # Clear markers
            return

        current_robot_state = self.get_current_robot_pose_in_map()
        if current_robot_state is None:
            self.get_logger().warn("Cannot plan new path: Failed to get current robot pose in map.")
            self.current_path = []
            self.publish_path_markers() # Clear markers
            return
        
        current_robot_pos_world = (current_robot_state['x'], current_robot_state['y'])
        start_grid_xy = self.world_to_map_grid(current_robot_pos_world[0], current_robot_pos_world[1])
        
        goal_world_xy = (self.current_goal_pose.position.x, self.current_goal_pose.position.y)
        goal_grid_xy_orig = self.world_to_map_grid(goal_world_xy[0], goal_world_xy[1])
        
        if start_grid_xy[0] is None or start_grid_xy[1] is None or \
           goal_grid_xy_orig[0] is None or goal_grid_xy_orig[1] is None:
            self.get_logger().warn("Failed to convert world start/goal to map grid coordinates.")
            self.current_path = []
            self.publish_path_markers()
            return
        
        # Ensure goal grid cell is free, find nearest if not
        effective_goal_grid_xy = goal_grid_xy_orig
        if self.is_grid_cell_occupied(goal_grid_xy_orig[0], goal_grid_xy_orig[1]):
            self.get_logger().warn(f"Original goal grid cell {goal_grid_xy_orig} is occupied. Finding nearest free cell.")
            found_free_goal_grid = self.find_nearest_free_grid_cell(goal_grid_xy_orig, max_radius_cells=30)
            if found_free_goal_grid is None:
                self.get_logger().error("No free cell found near goal. Path planning aborted.")
                self.current_path = []
                self.publish_path_markers()
                return
            effective_goal_grid_xy = found_free_goal_grid
            self.get_logger().info(f"Using effective goal grid {effective_goal_grid_xy} instead of {goal_grid_xy_orig}.")

        # A* planner returns a path in grid coordinates
        grid_path = self.a_star_planner(start_grid_xy, effective_goal_grid_xy)
        
        if grid_path:
            # Smooth path (which also converts to world coordinates)
            self.current_path = self.smooth_path(grid_path) # self.current_path is now in world_coords
            if self.current_path:
                self.get_logger().info(f"Path planned with {len(self.current_path)} smoothed world waypoints.")
            else:
                self.get_logger().warn("A* found grid path, but smoothing/world conversion resulted in empty path.")
        else:
            self.get_logger().warn("A* planner failed to find a path. No path generated.")
            self.current_path = []
        
        self.publish_path_markers() # Publish new path (or clear markers if path failed)


    def publish_path_markers(self):
        if not rclpy.ok(): 
            self.get_logger().warn("publish_path_markers: RCLPY not OK, skipping publish.", throttle_duration_sec=5.0)
            return 
            
        marker_array = MarkerArray()
        # Use a fixed timestamp for all markers in this array for consistency
        now_msg = self.get_clock().now().to_msg()
        header = Header(stamp=now_msg, frame_id=self.map_tf_frame)

        # Marker to delete all previous markers in this namespace
        delete_all_marker = Marker(
            header=header, 
            ns=f"path_waypoints_{self.robot_namespace}", 
            id=0, # Special ID for DELETEALL
            type=Marker.SPHERE, # Type doesn't matter much for DELETEALL but must be valid
            action=Marker.DELETEALL
        )
        marker_array.markers.append(delete_all_marker)
        
        # Add new markers for the current path (if any)
        for i, (world_x, world_y) in enumerate(self.current_path): # current_path is in world coordinates
            marker = Marker(
                header=header, 
                ns=f"path_waypoints_{self.robot_namespace}", 
                id=i + 1, # Unique ID for each marker, starting from 1
                type=Marker.SPHERE, 
                action=Marker.ADD
            )
            # Set marker pose
            marker.pose = Pose() # Create a new Pose object
            marker.pose.position.x = world_x
            marker.pose.position.y = world_y
            marker.pose.position.z = 0.1 # Slightly above ground for visibility
            marker.pose.orientation.w = 1.0 # Default orientation

            # Set marker scale
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            
            # Set marker color (always green now, as obstacle avoidance FSM is removed)
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8) # Green
                
            marker.lifetime = rclpy.duration.Duration(seconds=15).to_msg() # How long marker persists if not updated
            marker_array.markers.append(marker)
        
        if self.marker_pub.get_subscription_count() > 0:
            self.marker_pub.publish(marker_array)
        elif not self.current_path: # If path is empty, still try to publish delete_all
             self.marker_pub.publish(marker_array)


    def path_following_update(self):
        # Obstacle avoidance FSM removed. Path following is always "normal".
        # The scan_callback handles emergency stops by clearing the path.

        if not self.current_path: # No path to follow
            # If no path, ensure robot is stopped (e.g., after completion or if planning failed)
            # Check if last command was non-zero to avoid spamming stop commands.
            # For now, a simple stop if no path.
            # self.cmd_pub.publish(Twist()) # Consider if this spams too much.
            return

        current_robot_state = self.get_current_robot_pose_in_map()
        if current_robot_state is None:
            self.get_logger().warn("Path following: TF lookup failed. Sending zero velocity.", throttle_duration_sec=2.0)
            self.cmd_pub.publish(Twist()) # Stop if pose is unknown
            return

        robot_x_world, robot_y_world = current_robot_state['x'], current_robot_state['y']
        robot_yaw_world = current_robot_state['yaw']
        
        # Check if close to waypoints and remove them
        arrival_threshold_at_waypoint = 0.20  # meters
        self.cleanup_passed_waypoints(robot_x_world, robot_y_world, arrival_threshold_at_waypoint**2)
        
        if not self.current_path: # Path might have been completed after cleanup
            self.get_logger().info("Path successfully completed or cleared by cleanup! Stopping.")
            self.cmd_pub.publish(Twist())
            self.current_goal_pose = None # Clear goal as path is done
            self.publish_path_markers() # Update markers to show no path
            return

        # Pure Pursuit-like lookahead point
        lookahead_distance = 0.4 # meters
        target_wp_x_world, target_wp_y_world = self.get_lookahead_point(robot_x_world, robot_y_world, lookahead_distance)
        
        # Calculate errors to the lookahead point
        error_x_world = target_wp_x_world - robot_x_world
        error_y_world = target_wp_y_world - robot_y_world
        
        cmd = Twist()
        angle_to_lookahead_point = math.atan2(error_y_world, error_x_world)
        heading_error = angle_to_lookahead_point - robot_yaw_world
        
        # Normalize heading error to [-pi, pi]
        while heading_error > math.pi: heading_error -= 2 * math.pi
        while heading_error < -math.pi: heading_error += 2 * math.pi

        # --- Proportional Control Parameters ---
        k_angular = 0.8  # Proportional gain for angular velocity
        k_linear = 0.5   # Proportional gain for linear velocity (can also be fixed max)
        max_linear_vel = 0.22 # Max linear velocity (m/s) for TurtleBot4/Create3
        max_angular_vel = 1.0 # Max angular velocity (rad/s) for TurtleBot4/Create3 (spec is ~1.9, but 1.0 is safer)

        # --- Control Logic ---
        # Angular velocity
        angular_deadband_rad = math.radians(2.0) # Small deadband to prevent oscillation
        if abs(heading_error) > angular_deadband_rad:
            cmd.angular.z = k_angular * heading_error
        else:
            cmd.angular.z = 0.0
        cmd.angular.z = np.clip(cmd.angular.z, -max_angular_vel, max_angular_vel)

        # Linear velocity: slow down or stop if not facing the lookahead point
        turn_threshold_stop_rad = math.radians(75) # If heading error > 75 deg, stop linear motion and only turn
        turn_threshold_slow_rad = math.radians(30) # If heading error > 30 deg, reduce linear speed

        if abs(heading_error) > turn_threshold_stop_rad:
            cmd.linear.x = 0.0
        elif abs(heading_error) > turn_threshold_slow_rad:
            # Scale linear velocity based on how far into the "slow zone" we are
            # factor = 1.0 - (abs(heading_error) - turn_threshold_slow_rad) / (turn_threshold_stop_rad - turn_threshold_slow_rad)
            # Simpler: just reduce speed significantly
            cmd.linear.x = max_linear_vel * 0.3 # Reduced speed
        else:
            # When heading error is small, move at max speed (or k_linear * distance_to_lookahead)
            # For simplicity, using max_linear_vel when aligned.
            cmd.linear.x = max_linear_vel

        cmd.linear.x = np.clip(cmd.linear.x, 0.0, max_linear_vel) # Ensure non-negative and capped

        # Special handling if very close to the final waypoint
        if len(self.current_path) == 1:
            dist_to_final_sq = (self.current_path[0][0] - robot_x_world)**2 + \
                               (self.current_path[0][1] - robot_y_world)**2
            # If within 1.5 times the general arrival threshold of the *final* point
            if dist_to_final_sq < (arrival_threshold_at_waypoint * 1.5)**2 : 
                cmd.linear.x *= 0.5 # Slow down significantly when approaching final destination
                # Optionally, prioritize aligning with goal orientation if current_goal_pose has one.
                # For now, just slow down.

        self.cmd_pub.publish(cmd)
        self.publish_path_markers() # Update path markers regularly


def main(args=None):
    rclpy.init(args=args)
    planning_node = PlanningNode()
    try:
        rclpy.spin(planning_node)
    except KeyboardInterrupt:
        planning_node.get_logger().info("Keyboard interrupt received, shutting down PlanningNode.")
    except Exception as e:
        planning_node.get_logger().error(f"Unhandled exception in spin: {e}", exc_info=True)
    finally:
        planning_node.get_logger().info("Node shutdown sequence initiated in finally block.")
        
        # Attempt to publish a stop command if the node and publisher are still valid
        if rclpy.ok() and hasattr(planning_node, 'cmd_pub') and planning_node.cmd_pub is not None:
            try:
                if hasattr(planning_node.cmd_pub, 'handle') and planning_node.cmd_pub.handle: # Check if publisher is valid
                    planning_node.get_logger().info("Publishing zero velocity as part of shutdown.")
                    planning_node.cmd_pub.publish(Twist())
                else:
                    planning_node.get_logger().warn("cmd_pub handle is invalid during shutdown, cannot publish stop Twist.")
            except Exception as e_pub:
                # Log with severity that won't be filtered easily if rosout is already down
                print(f"ERROR: PlanningNode: Error publishing stop Twist on shutdown: {e_pub}")
                planning_node.get_logger().error(f"Error publishing stop Twist on shutdown: {e_pub}")
        
        # Destroy the node
        # This should also trigger cleanup of resources like timers and subscriptions.
        # The TransformListener's thread (if daemon) should exit when the main program exits,
        # but destroying the node it's associated with should signal it to stop its executor.
        if hasattr(planning_node, 'destroy_node'):
            planning_node.get_logger().info("Destroying PlanningNode instance...")
            planning_node.destroy_node()
            planning_node.get_logger().info("PlanningNode instance destroyed.")
        
        # Shutdown RCLPY
        if rclpy.ok():
            # Logger might not work reliably after rclpy.shutdown()
            print("INFO: PlanningNode: Shutting down RCLPY context.")
            rclpy.shutdown()
            print("INFO: PlanningNode: RCLPY shutdown complete.")
        else:
            print("INFO: PlanningNode: RCLPY context already shutdown or invalid.")

if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA # Bool is no longer used for bumper
from sensor_msgs.msg import LaserScan, PointCloud2 # Added PointCloud2
from sensor_msgs_py import point_cloud2 as pc2 # Helper for PointCloud2
from tf2_ros import Buffer, TransformListener
import tf2_ros

import numpy as np
import heapq
from scipy.ndimage import maximum_filter
import math
import time
from enum import Enum

from std_msgs.msg import Header
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class ObstacleAvoidanceState(Enum):
    NORMAL = "normal"
    OBSTACLE_DETECTED = "obstacle_detected"
    BACKING_UP = "backing_up"
    TURNING = "turning"
    REPLANNING = "replanning"

class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')

        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID')
        if self.robot_id_str is None:
            self.get_logger().error("ROS_DOMAIN_ID environment variable not set! Defaulting to 0.")
            self.robot_id_str = "0"
        self.robot_namespace = f"/T{self.robot_id_str}"

        self.map_tf_frame = "map"
        self.robot_odom_tf_frame = "odom"
        self.robot_base_footprint_tf_frame = "base_footprint"
        self.robot_base_link_tf_frame = "base_link"

        self.get_logger().info(f"Planning node initialized for ROS_DOMAIN_ID: {self.robot_id_str} (Namespace: {self.robot_namespace})")
        # ... (other log messages for TF frames)

        self.map_data_storage = None
        self.inflated_map_storage = None
        self.map_info_storage = None
        self.current_goal_pose = None
        self.current_path = []

        self.obstacle_avoidance_state = ObstacleAvoidanceState.NORMAL
        # self.bumper_triggered = False # This flag is effectively managed by obstacle_avoidance_state != NORMAL
        self.left_bumper_hit = False
        self.right_bumper_hit = False
        self.front_bumper_hit = False
        self.obstacle_avoidance_start_time = 0.0
        self.robot_pose_when_obstacle_detected = None # Store pose for replanning
        self.replanning_attempts = 0
        self.max_replanning_attempts = 3
        # Backup/Turn parameters are now mostly managed within execute_obstacle_avoidance
        self.target_backup_distance = 0.3
        self.target_turn_angle = 0.0


        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.declare_parameter('map_topic_base', 'map')
        self.declare_parameter('goal_topic_base', 'goal_pose')
        self.declare_parameter('cmd_vel_topic_base', 'cmd_vel')
        self.declare_parameter('waypoints_topic_base', 'waypoints')
        self.declare_parameter('bumper_topic_base', 'mobile_base/sensors/bumper_pointcloud')
        self.declare_parameter('scan_topic_base', 'scan')

        map_topic_base = self.get_parameter('map_topic_base').get_parameter_value().string_value
        goal_topic_base = self.get_parameter('goal_topic_base').get_parameter_value().string_value
        cmd_vel_topic_base = self.get_parameter('cmd_vel_topic_base').get_parameter_value().string_value
        waypoints_topic_base = self.get_parameter('waypoints_topic_base').get_parameter_value().string_value
        bumper_topic_base = self.get_parameter('bumper_topic_base').get_parameter_value().string_value
        scan_topic_base = self.get_parameter('scan_topic_base').get_parameter_value().string_value

        self.map_topic_actual = f"{self.robot_namespace}/{map_topic_base}"
        self.goal_topic_actual = f"{self.robot_namespace}/{goal_topic_base}"
        self.cmd_vel_topic_actual = f"{self.robot_namespace}/{cmd_vel_topic_base}"
        self.waypoints_topic_actual = f"{self.robot_namespace}/{waypoints_topic_base}"
        self.bumper_topic_actual = f"{self.robot_namespace}/{bumper_topic_base}" # Will now be PointCloud2
        self.scan_topic_actual = f"{self.robot_namespace}/{scan_topic_base}"

        self.get_logger().info(f"Subscribing to map topic: {self.map_topic_actual}")
        self.get_logger().info(f"Subscribing to goal topic: {self.goal_topic_actual}")
        self.get_logger().info(f"Subscribing to bumper (PointCloud2) topic: {self.bumper_topic_actual}")
        self.get_logger().info(f"Subscribing to scan topic: {self.scan_topic_actual}")
        self.get_logger().info(f"Publishing cmd_vel to: {self.cmd_vel_topic_actual}")
        self.get_logger().info(f"Publishing waypoints to: {self.waypoints_topic_actual}")

        qos_map = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_goal = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic_actual, self.map_callback, qos_map)
        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic_actual, self.goal_callback, qos_goal)
        
        # Subscribe to PointCloud2 for bumper data
        self.bumper_sub = self.create_subscription(
            PointCloud2,
            self.bumper_topic_actual,
            self.bumper_pointcloud_callback, # New callback
            qos_sensor
        )
        # LaserScan for proximity detection (can complement bumper_pointcloud)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic_actual, self.scan_callback, qos_sensor)
        
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic_actual, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.waypoints_topic_actual, 10)

        self.path_execution_timer = self.create_timer(0.2, self.path_following_update)

    def quaternion_to_yaw(self, q_geometry_msg):
        siny_cosp = 2 * (q_geometry_msg.w * q_geometry_msg.z + q_geometry_msg.x * q_geometry_msg.y)
        cosy_cosp = 1 - 2 * (q_geometry_msg.y * q_geometry_msg.y + q_geometry_msg.z * q_geometry_msg.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def reset_bumper_flags(self):
        """Reset all bumper hit-type flags"""
        self.get_logger().debug("Resetting specific bumper hit flags (left, right, front).")
        self.left_bumper_hit = False
        self.right_bumper_hit = False
        self.front_bumper_hit = False

    def bumper_pointcloud_callback(self, msg: PointCloud2):
        # self.get_logger().debug(f"Bumper PointCloud received. Frame: {msg.header.frame_id}, Num points: {msg.height * msg.width}", throttle_duration_sec=5.0)

        # Check if we are already avoiding obstacles
        if self.obstacle_avoidance_state != ObstacleAvoidanceState.NORMAL:
            # self.get_logger().debug("Bumper PointCloud: Already in avoidance state, skipping detailed check.", throttle_duration_sec=5.0)
            return

        # --- IMPORTANT: TUNE THESE REGIONS CAREFULLY ---
        # These define detection boxes relative to msg.header.frame_id (likely 'base_link' or similar).
        # Points are (x, y, z). For base_link: +x forward, +y left, +z up.
        # Adjust these values based on your robot's geometry and sensor characteristics.
        # Small values for close "bumper-like" detection.

        # Front detection region:
        FRONT_MIN_X = 0.01    # Min X distance (just in front of sensor origin)
        FRONT_MAX_X = 0.10    # Max X distance (e.g., 10cm) for a "front hit"
        FRONT_MAX_Y_ABS = 0.12 # Max Y distance from center (e.g., +/- 12cm wide)
        FRONT_MAX_Z_ABS = 0.15 # Max Z distance from sensor height (e.g. +/- 15cm, to filter ground/high points)


        # Left detection region (positive Y is usually left):
        LEFT_MIN_Y = 0.05     # Min Y distance from center (e.g., 5cm left of robot's Y=0)
        LEFT_MAX_Y = 0.20     # Max Y distance (e.g., 20cm left)
        LEFT_MAX_X_ABS = 0.10 # Max X deviation for a "side hit" (e.g., +/-10cm along X-axis)
        LEFT_MAX_Z_ABS = 0.15 # Max Z deviation

        # Right detection region (negative Y is usually right):
        RIGHT_MIN_Y = -0.20   # Min Y distance (e.g., 20cm right, so -0.20)
        RIGHT_MAX_Y = -0.05   # Max Y distance (e.g., 5cm right, so -0.05)
        RIGHT_MAX_X_ABS = 0.10# Max X deviation
        RIGHT_MAX_Z_ABS = 0.15 # Max Z deviation

        detected_front = False
        detected_left = False
        detected_right = False

        # Iterate through points. field_names might need adjustment based on actual PointCloud2 structure.
        # Common fields are 'x', 'y', 'z'.
        try:
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                px, py, pz = point[0], point[1], point[2]

                # Check Front Region
                if FRONT_MIN_X < px < FRONT_MAX_X and \
                   abs(py) < FRONT_MAX_Y_ABS and \
                   abs(pz) < FRONT_MAX_Z_ABS: # Add Z check if needed
                    detected_front = True
                    # self.get_logger().info(f"Bumper PointCloud: FRONT HIT detected at x={px:.2f}, y={py:.2f}, z={pz:.2f}")
                    break # Prioritize front hit, stop checking

                # Check Left Region
                if LEFT_MIN_Y < py < LEFT_MAX_Y and \
                   abs(px) < LEFT_MAX_X_ABS and \
                   abs(pz) < LEFT_MAX_Z_ABS:
                    detected_left = True
                    # self.get_logger().info(f"Bumper PointCloud: LEFT HIT detected at x={px:.2f}, y={py:.2f}, z={pz:.2f}")
                    # Don't break here, could also be a front hit if regions overlap

                # Check Right Region
                if RIGHT_MIN_Y < py < RIGHT_MAX_Y and \
                   abs(px) < RIGHT_MAX_X_ABS and \
                   abs(pz) < RIGHT_MAX_Z_ABS:
                    detected_right = True
                    # self.get_logger().info(f"Bumper PointCloud: RIGHT HIT detected at x={px:.2f}, y={py:.2f}, z={pz:.2f}")
                    # Don't break here
            
            if detected_front or detected_left or detected_right:
                # Ensure we are in normal state before initiating avoidance
                if self.obstacle_avoidance_state == ObstacleAvoidanceState.NORMAL:
                    self.get_logger().warn(f"BUMPER POINTCLOUD: Obstacle detected! F:{detected_front}, L:{detected_left}, R:{detected_right}. Initiating avoidance.")
                    self.reset_bumper_flags() # Clear any old specific hit types

                    # Determine primary hit type
                    if detected_front:
                        self.front_bumper_hit = True
                    elif detected_left and detected_right: # Both sides implies front for simplicity
                        self.front_bumper_hit = True
                    elif detected_left:
                        self.left_bumper_hit = True
                    elif detected_right:
                        self.right_bumper_hit = True
                    else: # Should not happen if one of them was true
                        self.front_bumper_hit = True # Default to front if logic is odd

                    self.initiate_obstacle_avoidance()
        except Exception as e:
            self.get_logger().error(f"Error processing PointCloud2: {e}")


    def scan_callback(self, msg: LaserScan):
        if not msg.ranges:
            return
        
        # This scan callback is for slightly further obstacles than the bumper_pointcloud
        min_distance_laser = 0.20  # e.g., 20cm - adjust as needed, should be > bumper_pointcloud range

        # Check if we are already avoiding obstacles
        if self.obstacle_avoidance_state != ObstacleAvoidanceState.NORMAL:
            # self.get_logger().debug("LaserScan: Already in avoidance state, skipping check.", throttle_duration_sec=5.0)
            return

        front_angles = len(msg.ranges) // 6  # Front 60 degrees
        center = len(msg.ranges) // 2
        
        front_ranges = msg.ranges[center - front_angles : center + front_angles]
        valid_ranges = [r for r in front_ranges if msg.range_min < r < msg.range_max and r < min_distance_laser]
        
        if valid_ranges: # No need for min(valid_ranges) < min_distance_laser, already filtered
            if self.obstacle_avoidance_state == ObstacleAvoidanceState.NORMAL:
                self.get_logger().warn(f"LASERSCAN: Close obstacle detected at {min(valid_ranges):.2f}m by LaserScan! Initiating avoidance.")
                self.reset_bumper_flags()
                self.front_bumper_hit = True # Laser scan primarily indicates front obstacle for this simplified logic
                self.initiate_obstacle_avoidance()


    def initiate_obstacle_avoidance(self):
        if self.obstacle_avoidance_state == ObstacleAvoidanceState.NORMAL:
            self.obstacle_avoidance_state = ObstacleAvoidanceState.OBSTACLE_DETECTED
            self.obstacle_avoidance_start_time = time.time()
            
            current_pos = self.get_current_robot_pose()
            if current_pos:
                self.robot_pose_when_obstacle_detected = current_pos
            
            self.get_logger().info(f"INITIATING AVOIDANCE: State -> OBSTACLE_DETECTED. Hit flags: F:{self.front_bumper_hit}, L:{self.left_bumper_hit}, R:{self.right_bumper_hit}")
        else:
            self.get_logger().warn(f"Attempted to initiate_obstacle_avoidance, but state is already {self.obstacle_avoidance_state.value}")


    def get_current_robot_pose(self):
        source_frames_to_try = [self.robot_base_footprint_tf_frame, self.robot_base_link_tf_frame]
        for source_frame_id in source_frames_to_try:
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    self.map_tf_frame,
                    source_frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                return {
                    'x': transform_stamped.transform.translation.x,
                    'y': transform_stamped.transform.translation.y,
                    'yaw': self.quaternion_to_yaw(transform_stamped.transform.rotation)
                }
            except Exception as e:
                self.get_logger().debug(f"Get_current_robot_pose TF lookup failed: {e}", throttle_duration_sec=2.0)
                continue
        return None

    def execute_obstacle_avoidance(self):
        cmd = Twist()
        current_time = time.time()
        if self.obstacle_avoidance_start_time == 0.0 and self.obstacle_avoidance_state != ObstacleAvoidanceState.OBSTACLE_DETECTED:
             self.get_logger().warn(f"Obstacle avoidance start time is 0 for state {self.obstacle_avoidance_state.value}, resetting.")
             self.obstacle_avoidance_start_time = current_time

        elapsed_time = current_time - self.obstacle_avoidance_start_time
        
        # self.get_logger().debug(f"Execute Avoidance: State={self.obstacle_avoidance_state.value}, Elapsed={elapsed_time:.2f}s", throttle_duration_sec=1.0)

        if self.obstacle_avoidance_state == ObstacleAvoidanceState.OBSTACLE_DETECTED:
            self.obstacle_avoidance_state = ObstacleAvoidanceState.BACKING_UP
            # self.backup_distance = 0.0 # Not used anymore for control
            self.obstacle_avoidance_start_time = current_time # Reset timer for this sub-state
            self.get_logger().info("Avoidance: OBSTACLE_DETECTED -> BACKING_UP")
            
        elif self.obstacle_avoidance_state == ObstacleAvoidanceState.BACKING_UP:
            backup_speed = -0.1  # m/s
            backup_duration = 2.0  # seconds (adjust as needed)
            
            if elapsed_time < backup_duration:
                cmd.linear.x = backup_speed
                cmd.angular.z = 0.0
                self.get_logger().debug(f"Avoidance: BACKING_UP (elapsed: {elapsed_time:.2f}/{backup_duration:.2f})")
            else:
                self.obstacle_avoidance_state = ObstacleAvoidanceState.TURNING
                # self.turn_angle_completed = 0.0 # Not used for control
                
                if self.left_bumper_hit:
                    self.target_turn_angle = -math.pi / 2.5  # Turn right (approx 72 deg)
                    self.get_logger().info("Avoidance: BACKUP complete. Left hit -> Turning RIGHT")
                elif self.right_bumper_hit:
                    self.target_turn_angle = math.pi / 2.5   # Turn left
                    self.get_logger().info("Avoidance: BACKUP complete. Right hit -> Turning LEFT")
                elif self.front_bumper_hit:
                    self.target_turn_angle = math.pi / 2 if np.random.random() > 0.5 else -math.pi / 2 # Random for front
                    self.get_logger().info(f"Avoidance: BACKUP complete. Front hit -> Turning {'LEFT' if self.target_turn_angle > 0 else 'RIGHT'} randomly")
                else:  # Fallback if no specific hit flag was set (should not happen with new logic)
                    self.get_logger().warn("Avoidance: BACKUP complete. No specific hit flag, defaulting to random turn.")
                    self.target_turn_angle = math.pi / 2 if np.random.random() > 0.5 else -math.pi / 2
                
                self.obstacle_avoidance_start_time = current_time # Reset timer for TURNING
                
        elif self.obstacle_avoidance_state == ObstacleAvoidanceState.TURNING:
            if self.target_turn_angle == 0: # Safety check
                self.get_logger().error("Avoidance: TURNING state, but target_turn_angle is 0. Defaulting.")
                self.target_turn_angle = math.pi / 2 

            turn_speed_magnitude = 0.3  # rad/s
            turn_speed = turn_speed_magnitude * np.sign(self.target_turn_angle)
            
            turn_duration = abs(self.target_turn_angle / turn_speed) if turn_speed != 0 else 0.1 # Avoid div by zero
            
            if elapsed_time < turn_duration and turn_speed != 0:
                cmd.linear.x = 0.0
                cmd.angular.z = turn_speed
                self.get_logger().debug(f"Avoidance: TURNING (elapsed: {elapsed_time:.2f}/{turn_duration:.2f}, speed: {turn_speed:.2f})")
            else:
                self.obstacle_avoidance_state = ObstacleAvoidanceState.REPLANNING
                self.obstacle_avoidance_start_time = current_time # Reset timer for REPLANNING pause
                self.get_logger().info("Avoidance: TURN complete -> REPLANNING")
                
        elif self.obstacle_avoidance_state == ObstacleAvoidanceState.REPLANNING:
            replan_pause_duration = 1.0 # seconds to pause before replanning
            if elapsed_time > replan_pause_duration:
                self.get_logger().info(f"Avoidance: REPLANNING (pause over). Replanning attempts: {self.replanning_attempts}/{self.max_replanning_attempts}")
                
                if self.replanning_attempts < self.max_replanning_attempts:
                    if self.current_goal_pose: # Only replan if there's an active goal
                        self.get_logger().info("Attempting to plan new path...")
                        self.plan_new_path() # This function logs its own success/failure
                    else:
                        self.get_logger().info("No current goal, skipping replan.")
                    self.replanning_attempts += 1
                else:
                    self.get_logger().warn(f"Max replanning attempts ({self.max_replanning_attempts}) reached for current goal. Clearing goal and path.")
                    self.current_path = []
                    self.current_goal_pose = None
                    # self.replanning_attempts will be reset when a new goal is received.

                # Transition back to normal AFTER attempting replan or deciding not to
                self.obstacle_avoidance_state = ObstacleAvoidanceState.NORMAL
                self.reset_bumper_flags() # Clear specific hit types for the next event
                self.get_logger().info("Avoidance: REPLANNING actions complete -> NORMAL state.")
            else:
                # Stay still during pause
                self.get_logger().debug(f"Avoidance: REPLANNING (pausing {elapsed_time:.2f}/{replan_pause_duration:.2f})")
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
        
        return cmd

    def map_callback(self, msg: OccupancyGrid):
        self.get_logger().info(f"Map data received (Size: {msg.info.width}x{msg.info.height}, Res: {msg.info.resolution:.3f})", throttle_duration_sec=10.0)
        self.map_data_storage = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info_storage = msg.info
        self.inflate_map_data()

    def inflate_map_data(self):
        if self.map_data_storage is None or self.map_info_storage is None:
            self.get_logger().warn("Cannot inflate map, map_data_storage or map_info_storage is None.")
            return
        
        occupied_mask = (self.map_data_storage >= 90).astype(np.uint8) # Consider values >= 90 as occupied
        
        inflation_radius_meters = 0.25 # Robot radius + safety margin
        if self.map_info_storage.resolution == 0:
            self.get_logger().error("Map resolution is zero, cannot calculate inflation radius in cells.")
            return
        inflation_radius_cells = math.ceil(inflation_radius_meters / self.map_info_storage.resolution)
        inflation_kernel_size = 2 * inflation_radius_cells + 1 # Must be odd

        self.get_logger().info(f"Inflating map with kernel size: {inflation_kernel_size} ({inflation_radius_meters}m)")
        inflated_regions_mask = maximum_filter(occupied_mask, size=inflation_kernel_size)
        
        self.inflated_map_storage = self.map_data_storage.copy()
        # Mark inflated areas as occupied (100), but don't overwrite unknown (-1) with inflated free space
        # Only inflate known occupied cells or areas that become inflated from known occupied cells.
        self.inflated_map_storage[inflated_regions_mask > 0] = 100 
        
        self.get_logger().info("Map inflation complete.")

    def goal_callback(self, msg: PoseStamped):
        self.get_logger().info(f"Goal received in frame '{msg.header.frame_id}': Pos(x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f})")
        
        self.replanning_attempts = 0 # Reset for a new goal
        self.obstacle_avoidance_state = ObstacleAvoidanceState.NORMAL # Reset avoidance state for new goal
        self.reset_bumper_flags() # Clear any lingering bumper flags

        if msg.header.frame_id != self.map_tf_frame:
            self.get_logger().warn(f"Goal is in frame '{msg.header.frame_id}'. Attempting to transform to '{self.map_tf_frame}'.")
            try:
                self.tf_buffer.can_transform(self.map_tf_frame, msg.header.frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
                transformed_goal_pose_stamped = self.tf_buffer.transform(msg, self.map_tf_frame, timeout=rclpy.duration.Duration(seconds=1.0))
                self.current_goal_pose = transformed_goal_pose_stamped.pose
                self.get_logger().info(f"Goal transformed to '{self.map_tf_frame}': Pos(x={self.current_goal_pose.position.x:.2f}, y={self.current_goal_pose.position.y:.2f})")
            except Exception as e:
                self.get_logger().error(f"Failed to transform goal: {e}")
                self.current_goal_pose = None
                self.current_path = []
                return
        else:
            self.current_goal_pose = msg.pose

        self.plan_new_path()

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
        world_x = grid_x * res + origin.x + res / 2.0
        world_y = grid_y * res + origin.y + res / 2.0
        return world_x, world_y

    def is_grid_cell_occupied(self, grid_x, grid_y):
        if self.inflated_map_storage is None:
            self.get_logger().warn("is_grid_cell_occupied: Inflated map is None, assuming occupied.", throttle_duration_sec=5.0)
            return True 
        height, width = self.inflated_map_storage.shape
        if 0 <= grid_y < height and 0 <= grid_x < width:
            cell_value = self.inflated_map_storage[grid_y, grid_x]
            return cell_value >= 90 or cell_value == -1 # Treat 90-100 (occupied/inflated) and -1 (unknown) as obstacles
        return True # Out of bounds is occupied

    def get_a_star_heuristic(self, pos_a, pos_b):
        return np.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])

    def find_nearest_free_grid_cell(self, goal_grid_xy, max_radius_cells=30):
        if goal_grid_xy[0] is None or goal_grid_xy[1] is None:
            self.get_logger().error("find_nearest_free_grid_cell: Invalid goal_grid_xy input.")
            return None
        gx, gy = goal_grid_xy
        if not self.is_grid_cell_occupied(gx, gy): return gx, gy

        self.get_logger().info(f"Searching for free cell near ({gx}, {gy}) within radius {max_radius_cells}")
        for r in range(1, max_radius_cells + 1):
            for dx_offset in range(-r, r + 1):
                for dy_offset in range(-r, r + 1):
                    if abs(dx_offset) == r or abs(dy_offset) == r: # Check perimeter
                        check_x, check_y = gx + dx_offset, gy + dy_offset
                        if not self.is_grid_cell_occupied(check_x, check_y):
                            self.get_logger().info(f"Found free cell ({check_x}, {check_y}) at radius {r}")
                            return check_x, check_y
        self.get_logger().warn(f"No free cell found within {max_radius_cells} cells of goal ({gx}, {gy})")
        return None

    def a_star_planner(self, start_grid_xy, goal_grid_xy):
        if start_grid_xy[0] is None or goal_grid_xy[0] is None:
            self.get_logger().error("A*: Start or goal grid coordinates are None.")
            return None
        
        if self.is_grid_cell_occupied(start_grid_xy[0], start_grid_xy[1]):
            self.get_logger().error(f"A*: Start position {start_grid_xy} is occupied! Cannot plan.")
            # Attempt to find a nearby free cell for start
            new_start = self.find_nearest_free_grid_cell(start_grid_xy, max_radius_cells=5)
            if new_start is None:
                self.get_logger().error("A*: Could not find a free cell near occupied start. Planning failed.")
                return None
            self.get_logger().warn(f"A*: Original start occupied, using new start {new_start}")
            start_grid_xy = new_start
            
        self.get_logger().info(f"A* planning from {start_grid_xy} to {goal_grid_xy}")
        open_set = []
        heapq.heappush(open_set, (0, start_grid_xy))
        came_from = {}
        g_score = {start_grid_xy: 0}
        f_score = {start_grid_xy: self.get_a_star_heuristic(start_grid_xy, goal_grid_xy)}
        
        nodes_explored = 0
        max_nodes = 30000

        while open_set and nodes_explored < max_nodes:
            _, current_grid_xy = heapq.heappop(open_set)
            nodes_explored += 1

            if current_grid_xy == goal_grid_xy:
                path_reconstructed = []
                temp_xy = current_grid_xy
                while temp_xy in came_from:
                    path_reconstructed.append(temp_xy)
                    temp_xy = came_from[temp_xy]
                path_reconstructed.append(start_grid_xy)
                self.get_logger().info(f"A* found path with {len(path_reconstructed)} waypoints ({nodes_explored} nodes).")
                return list(reversed(path_reconstructed))

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                neighbor_grid_xy = (current_grid_xy[0] + dx, current_grid_xy[1] + dy)
                if self.is_grid_cell_occupied(neighbor_grid_xy[0], neighbor_grid_xy[1]):
                    continue
                movement_cost = np.hypot(dx, dy)
                tentative_g_val = g_score[current_grid_xy] + movement_cost
                if tentative_g_val < g_score.get(neighbor_grid_xy, float('inf')):
                    came_from[neighbor_grid_xy] = current_grid_xy
                    g_score[neighbor_grid_xy] = tentative_g_val
                    f_val = tentative_g_val + self.get_a_star_heuristic(neighbor_grid_xy, goal_grid_xy)
                    if neighbor_grid_xy not in [item[1] for item in open_set]: # Avoid duplicates, or update priority
                         heapq.heappush(open_set, (f_val, neighbor_grid_xy))
        
        self.get_logger().warn(f"A* failed to find path after exploring {nodes_explored} nodes.")
        return None

    def smooth_path(self, path, min_distance_sq=0.09): # min_distance=0.3m, so 0.3^2
        if not path or len(path) < 2:
            return path
        smoothed = [path[0]]
        for i in range(1, len(path) -1): # Exclude last point from this part of loop
            current = path[i]
            last_kept = smoothed[-1]
            dx = current[0] - last_kept[0]
            dy = current[1] - last_kept[1]
            if (dx*dx + dy*dy) >= min_distance_sq:
                smoothed.append(current)
        smoothed.append(path[-1]) # Always add last point
        self.get_logger().info(f"Path smoothed from {len(path)} to {len(smoothed)} waypoints.")
        return smoothed

    def get_lookahead_point(self, robot_x, robot_y, lookahead_dist):
        if not self.current_path: return robot_x, robot_y
        
        closest_idx = 0
        min_dist_sq = float('inf')
        for i, (wx, wy) in enumerate(self.current_path):
            dist_sq = (wx - robot_x)**2 + (wy - robot_y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_idx = i
        
        accumulated_dist = 0.0
        for i in range(closest_idx, len(self.current_path)):
            if i == len(self.current_path) - 1: return self.current_path[i]
            current_wp = self.current_path[i]
            next_wp = self.current_path[i + 1]
            segment_dist = np.hypot(next_wp[0] - current_wp[0], next_wp[1] - current_wp[1])
            
            if segment_dist < 1e-6: # Skip zero-length segments
                continue

            if accumulated_dist + segment_dist >= lookahead_dist:
                remaining_dist = lookahead_dist - accumulated_dist
                ratio = remaining_dist / segment_dist
                x = current_wp[0] + ratio * (next_wp[0] - current_wp[0])
                y = current_wp[1] + ratio * (next_wp[1] - current_wp[1])
                return x, y
            accumulated_dist += segment_dist
        return self.current_path[-1]

    def cleanup_passed_waypoints(self, robot_x, robot_y, threshold_sq): # Use squared threshold
        while self.current_path:
            wp_x, wp_y = self.current_path[0]
            dist_sq = (wp_x - robot_x)**2 + (wp_y - robot_y)**2
            if dist_sq < threshold_sq:
                self.current_path.pop(0)
                # self.get_logger().debug(f"Removed close waypoint. {len(self.current_path)} remaining.")
            else:
                break

    def plan_new_path(self):
        if self.map_data_storage is None or self.current_goal_pose is None or self.map_info_storage is None:
            self.get_logger().warn("Cannot plan path: Missing map, goal, or map info.")
            self.current_path = []
            return

        current_robot_pos_world = None
        source_frames_to_try = [self.robot_base_footprint_tf_frame, self.robot_base_link_tf_frame]
        for source_frame_id in source_frames_to_try:
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    self.map_tf_frame, source_frame_id, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2)
                )
                current_robot_pos_world = (transform_stamped.transform.translation.x, transform_stamped.transform.translation.y)
                break
            except Exception as e:
                self.get_logger().debug(f"TF lookup failed for planning ('{self.map_tf_frame}' <- '{source_frame_id}'): {e}")
        
        if current_robot_pos_world is None:
            self.get_logger().warn("All TF lookups failed for robot's current pose. Cannot plan path.")
            self.current_path = []
            return

        start_grid_xy = self.world_to_map_grid(current_robot_pos_world[0], current_robot_pos_world[1])
        goal_world_xy = (self.current_goal_pose.position.x, self.current_goal_pose.position.y)
        goal_grid_xy_orig = self.world_to_map_grid(goal_world_xy[0], goal_world_xy[1])
        
        if start_grid_xy[0] is None or goal_grid_xy_orig[0] is None:
            self.get_logger().warn("Failed to convert world start/goal to map grid.")
            self.current_path = []
            return
        
        goal_grid_xy = goal_grid_xy_orig
        if self.is_grid_cell_occupied(goal_grid_xy[0], goal_grid_xy[1]):
            self.get_logger().warn(f"Original goal grid cell {goal_grid_xy} is occupied. Finding nearest free cell.")
            effective_goal_grid_xy = self.find_nearest_free_grid_cell(goal_grid_xy)
            if effective_goal_grid_xy is None:
                self.get_logger().warn("No free cell found near goal. Path planning aborted.")
                self.current_path = []
                return
            goal_grid_xy = effective_goal_grid_xy

        self.get_logger().info(f"Planning path from grid {start_grid_xy} to effective grid {goal_grid_xy}")
        grid_path = self.a_star_planner(start_grid_xy, goal_grid_xy)
        
        if grid_path:
            self.current_path = [(self.map_grid_to_world(gx, gy)) for gx, gy in grid_path if self.map_grid_to_world(gx, gy)[0] is not None]
            if self.current_path:
                self.current_path = self.smooth_path(self.current_path) # Use squared distance in smooth_path
                self.publish_path_markers()
                self.get_logger().info(f"Path planned with {len(self.current_path)} smoothed waypoints.")
            else:
                self.get_logger().warn("A* found grid path, but world conversion failed or path empty.")
        else:
            self.get_logger().warn("A* planner failed to find a path.")
            self.current_path = []

    def publish_path_markers(self):
        if not rclpy.ok(): return # Don't try to publish if shutting down
        marker_array = MarkerArray()
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.map_tf_frame)

        delete_all_marker = Marker(header=header, ns=f"path_waypoints_{self.robot_namespace}", id=0, action=Marker.DELETEALL)
        marker_array.markers.append(delete_all_marker)
        
        for i, (world_x, world_y) in enumerate(self.current_path):
            marker = Marker(
                header=header, ns=f"path_waypoints_{self.robot_namespace}", id=i + 1,
                type=Marker.SPHERE, action=Marker.ADD,
                pose=PoseStamped(header=header, pose=PoseStamped().pose).pose # Initialize pose
            )
            marker.pose.position.x = world_x
            marker.pose.position.y = world_y
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1; marker.scale.y = 0.1; marker.scale.z = 0.1;
            
            if self.obstacle_avoidance_state != ObstacleAvoidanceState.NORMAL:
                marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.8) # Orange
            else:
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8) # Green
                
            marker.lifetime = rclpy.duration.Duration(seconds=15).to_msg() # Shorter lifetime
            marker_array.markers.append(marker)
        
        if self.marker_pub.get_subscription_count() > 0:
            self.marker_pub.publish(marker_array)

    def path_following_update(self):
        # self.get_logger().debug(f"Path Following Update: State={self.obstacle_avoidance_state.value}", throttle_duration_sec=1.0)
        if self.obstacle_avoidance_state != ObstacleAvoidanceState.NORMAL:
            cmd = self.execute_obstacle_avoidance()
            self.cmd_pub.publish(cmd)
            self.publish_path_markers() # Update marker color during avoidance
            return

        if not self.current_path:
            # If no path, but we are in NORMAL state, ensure robot is stopped
            # This can happen if a goal was cleared or planning failed.
            # Only publish if not already stopped to avoid spamming.
            # (Better: check if last command was non-zero)
            # self.cmd_pub.publish(Twist()) # Simplistic stop
            return

        current_robot_pos_world = None
        current_robot_yaw_world = None
        source_frames_to_try = [self.robot_base_footprint_tf_frame, self.robot_base_link_tf_frame]
        for source_frame_id in source_frames_to_try:
            try:
                transform_stamped = self.tf_buffer.lookup_transform(
                    self.map_tf_frame, source_frame_id, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05)
                )
                current_robot_pos_world = (transform_stamped.transform.translation.x, transform_stamped.transform.translation.y)
                current_robot_yaw_world = self.quaternion_to_yaw(transform_stamped.transform.rotation)
                break
            except Exception as e:
                self.get_logger().debug(f"TF lookup failed for update ('{self.map_tf_frame}' <- '{source_frame_id}'): {e}", throttle_duration_sec=2.0)
        
        if current_robot_pos_world is None or current_robot_yaw_world is None:
            self.get_logger().warn("TF lookup failed in path following. Sending zero velocity.", throttle_duration_sec=2.0)
            self.cmd_pub.publish(Twist())
            return

        robot_x, robot_y = current_robot_pos_world
        
        arrival_threshold = 0.20  # meters
        arrival_threshold_sq = arrival_threshold**2
        self.cleanup_passed_waypoints(robot_x, robot_y, arrival_threshold_sq) # Use squared
        
        if not self.current_path:
            self.get_logger().info("Path successfully completed or cleared! Stopping.")
            self.cmd_pub.publish(Twist())
            self.current_goal_pose = None # Clear goal as path is done
            return

        lookahead_distance = 0.4
        target_wp_x, target_wp_y = self.get_lookahead_point(robot_x, robot_y, lookahead_distance)
        
        error_x = target_wp_x - robot_x
        error_y = target_wp_y - robot_y
        # distance_to_wp = np.hypot(error_x, error_y) # Not strictly needed if using lookahead point primarily for angle

        cmd = Twist()
        angle_to_wp = math.atan2(error_y, error_x)
        heading_error = angle_to_wp - current_robot_yaw_world
        while heading_error > math.pi: heading_error -= 2 * math.pi
        while heading_error < -math.pi: heading_error += 2 * math.pi

        k_angular = 0.8
        k_linear = 0.4 
        max_lin_vel = 0.22 # Max linear velocity for TB4
        max_ang_vel = 0.8  # Max angular velocity for TB4 (can be up to 1.9 for Create3)

        # Stop-and-turn thresholds
        turn_threshold_stop_rad = math.radians(75) # If error > 75 deg, stop linear and only turn
        turn_threshold_slow_rad = math.radians(30) # If error > 30 deg, slow down linear

        if abs(heading_error) > turn_threshold_stop_rad:
            cmd.linear.x = 0.0
        elif abs(heading_error) > turn_threshold_slow_rad:
            # Scale linear velocity based on how far into the "slow zone" we are
            factor = (turn_threshold_stop_rad - abs(heading_error)) / (turn_threshold_stop_rad - turn_threshold_slow_rad)
            cmd.linear.x = max_lin_vel * factor
        else:
            cmd.linear.x = max_lin_vel # Full speed if heading error is small

        # Angular velocity control
        angular_deadband_rad = math.radians(3) # 3 degree deadband
        if abs(heading_error) > angular_deadband_rad:
            cmd.angular.z = k_angular * heading_error
        else:
            cmd.angular.z = 0.0

        # Clamp velocities
        cmd.linear.x = max(0.0, min(cmd.linear.x, max_lin_vel)) # Ensure non-negative for forward motion
        cmd.angular.z = max(-max_ang_vel, min(cmd.angular.z, max_ang_vel))

        # If very close to the final waypoint, reduce speed further or prioritize orientation
        if len(self.current_path) == 1:
            dist_to_final_sq = (self.current_path[0][0] - robot_x)**2 + (self.current_path[0][1] - robot_y)**2
            if dist_to_final_sq < (arrival_threshold * 1.5)**2 : # If within 1.5x arrival threshold
                cmd.linear.x *= 0.5 # Slow down significantly
                # Could add logic here to align with goal orientation if desired

        self.cmd_pub.publish(cmd)
        self.publish_path_markers() # Update path markers regularly


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
                if hasattr(planning_node.cmd_pub, 'handle') and planning_node.cmd_pub.handle:
                    planning_node.get_logger().info("Publishing zero velocity as part of shutdown.")
                    planning_node.cmd_pub.publish(Twist())
                else:
                    planning_node.get_logger().warn("cmd_pub handle is invalid during shutdown.")
            except Exception as e_pub:
                planning_node.get_logger().error(f"Error publishing stop Twist on shutdown: {e_pub}")
        
        if hasattr(planning_node, 'destroy_node'):
            planning_node.destroy_node()
            planning_node.get_logger().info("PlanningNode destroyed.")
        if rclpy.ok(): # Check if rclpy context is still valid
            rclpy.shutdown()
            # Logger might not work after rclpy.shutdown()
            print("RCLPY shutdown complete.") # Use print as logger might be gone

if __name__ == '__main__':
    main()
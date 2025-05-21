# turtlebot_sensor/turtlebot_sensor/improved_turtlebot4_explorer.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from visualization_msgs.msg import Marker
import random
import math
import os
import numpy as np
import time
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

class Turtlebot4EnhancedExplorer(Node):
    def __init__(self):
        super().__init__('turtlebot4_enhanced_explorer')

        # --- Robot configuration ---
        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID')
        if self.robot_id_str is None:
            self.get_logger().error("ROS_DOMAIN_ID environment variable not set! Using default '0'.")
            self.robot_id_str = "0"  # Fallback

        self.robot_namespace = f"/T{self.robot_id_str}"
        self.scan_topic_name = f"{self.robot_namespace}/scan"
        self.cmd_vel_topic_name = f"{self.robot_namespace}/cmd_vel"
        self.odom_topic_name = f"{self.robot_namespace}/odom"
        
        self.get_logger().info(f"Initializing for robot namespace: {self.robot_namespace}")
        self.get_logger().info(f"Subscribing to scan on: {self.scan_topic_name}")
        self.get_logger().info(f"Publishing commands to: {self.cmd_vel_topic_name}")
        
        # --- QoS profiles ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5 
        )
        
        # --- Publishers/Subscribers ---
        self.vel_pub = self.create_publisher(Twist, self.cmd_vel_topic_name, 10)
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic_name,
            self.scan_callback,
            sensor_qos
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic_name,
            self.odom_callback,
            10
        )
        
        # Optional: Publish the explored map for visualization
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            f"{self.robot_namespace}/exploration_map", 
            10
        )
        
        # Publish path home for visualization
        self.path_pub = self.create_publisher(
            Path,
            f"{self.robot_namespace}/path_home",
            10
        )
        
        # Publish marker for start position
        self.marker_pub = self.create_publisher(
            Marker,
            f"{self.robot_namespace}/home_marker",
            10
        )
        
        # --- State variables ---
        self.scan_data = None
        self.current_pose = None
        self.previous_poses = []  # Store recent positions to detect if stuck
        self.state = 'waiting_for_first_pose'  # Start by waiting to establish home position
        self.turn_direction = 1
        self.stuck_counter = 0
        
        # Return-to-home variables
        self.start_position = None  # Will store the initial pose
        self.start_orientation = None  # Will store the initial orientation
        self.exploration_start_time = None  # When exploration began
        self.exploration_duration = 60.0  # Duration in seconds before returning (1 minute)
        
        # --- Grid map for tracking visited areas ---
        self.map_resolution = 0.1  # meters per cell
        self.map_width = 1000  # cells
        self.map_height = 1000  # cells
        # Center the origin
        self.map_origin_x = -self.map_width * self.map_resolution / 2
        self.map_origin_y = -self.map_height * self.map_resolution / 2
        # Initialize occupancy grid (-1: unknown, 0: free, 100: occupied)
        self.occupancy_grid = np.ones((self.map_height, self.map_width), dtype=np.int8) * -1
        self.visit_count_grid = np.zeros((self.map_height, self.map_width), dtype=np.int32)
        
        # --- Parameters ---
        # Obstacle detection
        self.obstacle_threshold = 0.5  # meters
        self.critical_front_threshold = 0.35  # more sensitive threshold for front
        self.front_check_angle_degrees = 60  # +/- 30 degrees from front
        
        # Movement speeds
        self.linear_velocity_forward = 0.2
        self.angular_velocity_turn = 0.5
        
        # Turning behavior
        self.desired_turn_angle = 0.0
        self.turn_start_time = None
        
        # Visited area marking
        self.visited_radius = 5  # cells
        self.high_visit_threshold = 3  # consider area "well-explored" after this many visits
        
        # Return-to-home parameters
        self.at_home_threshold = 0.3  # meters - how close to consider "at home"
        self.path_smoothing = 0.8  # Higher values create smoother paths (0-1)
        self.return_lookahead = 5  # How far to look ahead on the path (cells)
        
        # --- Timers ---
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.map_update_timer = self.create_timer(1.0, self.update_map)
        self.return_check_timer = self.create_timer(1.0, self.check_return_time)
        
        self.get_logger().info('Turtlebot4 Enhanced Explorer initialized. Waiting for sensor data...')
    
    def scan_callback(self, msg):
        self.scan_data = msg
    
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        
        # Store position history to detect if robot is stuck
        # Only store positions every few callbacks to avoid noise and reduce false stuck detection
        current_time = self.get_clock().now()
        
        # Only record position every ~0.5 seconds to reduce noise
        if not hasattr(self, 'last_pose_record_time') or \
           (current_time - self.last_pose_record_time).nanoseconds / 1e9 > 0.5:
            
            if len(self.previous_poses) >= 10:
                self.previous_poses.pop(0)
            
            self.previous_poses.append((
                self.current_pose.position.x, 
                self.current_pose.position.y
            ))
            self.last_pose_record_time = current_time
        
        # Initialize starting position if this is the first pose received
        if self.state == 'waiting_for_first_pose' and self.current_pose is not None:
            self.start_position = (
                self.current_pose.position.x,
                self.current_pose.position.y
            )
            self.start_orientation = (
                self.current_pose.orientation.x,
                self.current_pose.orientation.y,
                self.current_pose.orientation.z,
                self.current_pose.orientation.w
            )
            self.exploration_start_time = self.get_clock().now()
            self.state = 'forward'
            self.publish_home_marker()
            self.get_logger().info(f"Starting position set to: ({self.start_position[0]:.2f}, {self.start_position[1]:.2f})")
            self.get_logger().info(f"Exploration started. Will return home after {self.exploration_duration} seconds.")
    
    # Coordinate conversion methods
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        grid_x = int((x - self.map_origin_x) / self.map_resolution)
        grid_y = int((y - self.map_origin_y) / self.map_resolution)
        # Ensure within grid bounds
        grid_x = max(0, min(grid_x, self.map_width - 1))
        grid_y = max(0, min(grid_y, self.map_height - 1))
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates"""
        x = grid_x * self.map_resolution + self.map_origin_x
        y = grid_y * self.map_resolution + self.map_origin_y
        return x, y
    
    # Visualization methods
    def publish_home_marker(self):
        """Publish a visual marker at the home/start position"""
        if self.start_position is None:
            return
            
        marker = Marker()
        marker.header.frame_id = "map"  # Adjust if needed
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "home"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = self.start_position[0]
        marker.pose.position.y = self.start_position[1]
        marker.pose.position.z = 0.1  # Slightly above ground
        
        # Orientation (identity quaternion)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Scale
        marker.scale.x = 0.3  # Diameter
        marker.scale.y = 0.3
        marker.scale.z = 0.1  # Height
        
        # Color (green)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque
        
        self.marker_pub.publish(marker)
    
    def publish_map(self):
        """Publish the exploration map for visualization"""
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"  # Adjust if needed
        
        grid_msg.info.resolution = self.map_resolution
        grid_msg.info.width = self.map_width
        grid_msg.info.height = self.map_height
        grid_msg.info.origin.position.x = self.map_origin_x
        grid_msg.info.origin.position.y = self.map_origin_y
        
        # Flatten the numpy array to a 1D list
        grid_msg.data = self.occupancy_grid.flatten().tolist()
        
        self.map_pub.publish(grid_msg)
    
    def publish_path_home(self):
        """Publish a path from current position to home"""
        if self.current_pose is None or self.start_position is None:
            return
            
        path_msg = Path()
        path_msg.header.frame_id = "map"  # Adjust if needed
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Add current position
        current_pose = PoseStamped()
        current_pose.header = path_msg.header
        current_pose.pose = self.current_pose
        path_msg.poses.append(current_pose)
        
        # Add home position
        home_pose = PoseStamped()
        home_pose.header = path_msg.header
        home_pose.pose.position.x = self.start_position[0]
        home_pose.pose.position.y = self.start_position[1]
        home_pose.pose.position.z = 0.0
        
        # Use the saved initial orientation
        home_pose.pose.orientation.x = self.start_orientation[0]
        home_pose.pose.orientation.y = self.start_orientation[1]
        home_pose.pose.orientation.z = self.start_orientation[2]
        home_pose.pose.orientation.w = self.start_orientation[3]
        
        path_msg.poses.append(home_pose)
        
        self.path_pub.publish(path_msg)
    
    # Grid update and path planning
    def update_map(self):
        """Update occupancy grid based on current position and laser scan"""
        if self.current_pose is None or self.scan_data is None:
            return
        
        # Mark current position as visited
        current_x, current_y = self.current_pose.position.x, self.current_pose.position.y
        grid_x, grid_y = self.world_to_grid(current_x, current_y)
        
        # Mark a circular area around the current position as visited
        for i in range(-self.visited_radius, self.visited_radius + 1):
            for j in range(-self.visited_radius, self.visited_radius + 1):
                if i*i + j*j <= self.visited_radius*self.visited_radius:
                    x = grid_x + i
                    y = grid_y + j
                    if 0 <= x < self.map_width and 0 <= y < self.map_height:
                        # Mark as free space
                        self.occupancy_grid[y, x] = 0
                        # Increment visit count
                        self.visit_count_grid[y, x] += 1
        
        # Update map with laser scan data
        ranges = np.array(self.scan_data.ranges)
        angles = np.arange(
            self.scan_data.angle_min, 
            self.scan_data.angle_max + self.scan_data.angle_increment, 
            self.scan_data.angle_increment
        )
        
        # Get orientation as Euler angles (yaw)
        qx = self.current_pose.orientation.x
        qy = self.current_pose.orientation.y
        qz = self.current_pose.orientation.z
        qw = self.current_pose.orientation.w
        
        # Convert quaternion to yaw (rotation around z-axis)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        for i, (r, angle) in enumerate(zip(ranges, angles)):
            # Skip invalid measurements
            if np.isnan(r) or np.isinf(r) or r > self.scan_data.range_max:
                continue
                
            # Calculate endpoint in robot frame
            endpoint_x = r * math.cos(angle)
            endpoint_y = r * math.sin(angle)
            
            # Transform to world frame
            world_x = current_x + endpoint_x * math.cos(yaw) - endpoint_y * math.sin(yaw)
            world_y = current_y + endpoint_x * math.sin(yaw) + endpoint_y * math.cos(yaw)
            
            # Convert to grid coordinates
            grid_endpoint_x, grid_endpoint_y = self.world_to_grid(world_x, world_y)
            
            # Mark endpoint as occupied
            if 0 <= grid_endpoint_x < self.map_width and 0 <= grid_endpoint_y < self.map_height:
                self.occupancy_grid[grid_endpoint_y, grid_endpoint_x] = 100
        
        # Publish map for visualization
        self.publish_map()
        
        # Also publish other visualizations
        self.publish_home_marker()
        if self.state == 'returning_home':
            self.publish_path_home()
    
    # State machine logic
    def check_return_time(self):
        """Check if it's time to return to the starting position"""
        if self.exploration_start_time is None or self.start_position is None:
            return
            
        # Skip if already returning home or at home
        if self.state == 'returning_home' or self.state == 'at_home':
            return
            
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.exploration_start_time).nanoseconds / 1e9
        
        if elapsed_time >= self.exploration_duration:
            self.get_logger().info(f"Exploration time complete ({elapsed_time:.1f} seconds). Returning to start position.")
            self.state = 'returning_home'
    
    def calculate_distance_to_home(self):
        """Calculate distance from current position to home position"""
        if self.current_pose is None or self.start_position is None:
            return float('inf')
            
        dx = self.current_pose.position.x - self.start_position[0]
        dy = self.current_pose.position.y - self.start_position[1]
        return math.sqrt(dx*dx + dy*dy)
        
    def calculate_angle_to_home(self):
        """Calculate angle from current position to home position"""
        if self.current_pose is None or self.start_position is None:
            return 0.0
            
        # Vector to home
        dx = self.start_position[0] - self.current_pose.position.x
        dy = self.start_position[1] - self.current_pose.position.y
        
        # Calculate angle to home in world frame
        angle_to_home = math.atan2(dy, dx)
        
        # Get current orientation as yaw
        qx = self.current_pose.orientation.x
        qy = self.current_pose.orientation.y
        qz = self.current_pose.orientation.z
        qw = self.current_pose.orientation.w
        
        # Convert quaternion to yaw
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Calculate the angle difference
        angle_diff = angle_to_home - current_yaw
        
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2.0 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2.0 * math.pi
            
        return angle_diff
    
    # Obstacle detection and navigation
    def check_obstacles(self):
        if self.scan_data is None or self.scan_data.ranges is None:
            return {'front': False, 'left': False, 'right': False, 'valid': False}
        
        # Ensure ranges is a numpy array
        ranges = np.array(self.scan_data.ranges, dtype=np.float32)
        num_ranges = len(ranges)

        if num_ranges == 0:
            self.get_logger().warn("Received empty scan ranges.")
            return {'front': False, 'left': False, 'right': False, 'valid': False}

        # Replace inf/nan values
        valid_range_max = self.scan_data.range_max
        ranges[np.isinf(ranges)] = valid_range_max + 1.0 
        ranges[np.isnan(ranges)] = valid_range_max + 2.0

        # Calculate front sector width
        if self.scan_data.angle_increment == 0.0:
            front_half_width_indices = 30
        else:
            half_front_angle_rad = math.radians(self.front_check_angle_degrees / 2.0)
            front_half_width_indices = int(half_front_angle_rad / abs(self.scan_data.angle_increment))
        
        # Get front sector indices
        front_indices_part1 = np.arange(max(0, num_ranges - front_half_width_indices), num_ranges)
        front_indices_part2 = np.arange(0, min(num_ranges, front_half_width_indices + 1))
        
        # Get front sector ranges
        if len(front_indices_part1) == 0 and len(front_indices_part2) == 0:
            front_sector_ranges = np.array([])
        elif len(front_indices_part1) == 0:
            front_sector_ranges = ranges[front_indices_part2]
        elif len(front_indices_part2) == 0:
            front_sector_ranges = ranges[front_indices_part1]
        else:
            front_sector_ranges = np.concatenate((ranges[front_indices_part1], ranges[front_indices_part2]))
        
        # Check for front obstacles
        if front_sector_ranges.size > 0:
            front_obstacle = np.any(front_sector_ranges < self.critical_front_threshold)
        else:
            front_obstacle = False

        # Side obstacle detection
        side_sector_angle_degrees = 40
        side_half_width_indices = int(math.radians(side_sector_angle_degrees / 2.0) / 
                                    abs(self.scan_data.angle_increment)) if self.scan_data.angle_increment != 0 else 20

        # Left sector (approx. 90 degrees from front)
        left_center_idx = num_ranges // 4
        left_start_idx = max(0, left_center_idx - side_half_width_indices)
        left_end_idx = min(num_ranges - 1, left_center_idx + side_half_width_indices)
        left_sector_ranges = ranges[left_start_idx : left_end_idx + 1]
        
        # Right sector (approx. -90 degrees from front)
        right_center_idx = 3 * num_ranges // 4
        right_start_idx = max(0, right_center_idx - side_half_width_indices)
        right_end_idx = min(num_ranges - 1, right_center_idx + side_half_width_indices)
        right_sector_ranges = ranges[right_start_idx : right_end_idx + 1]

        # Check for left and right obstacles
        if left_sector_ranges.size > 0:
            left_obstacle = np.any(left_sector_ranges < self.obstacle_threshold)
        else:
            left_obstacle = False

        if right_sector_ranges.size > 0:
            right_obstacle = np.any(right_sector_ranges < self.obstacle_threshold)
        else:
            right_obstacle = False

        return {
            'front': front_obstacle,
            'left': left_obstacle,
            'right': right_obstacle,
            'valid': True
        }
    
    def check_if_stuck(self):
        """Check if the robot is stuck by analyzing position history"""
        # Don't check if stuck during intentional turning or if we don't have enough position history
        if self.state == 'turning' or len(self.previous_poses) < 10:
            return False
            
        # Get oldest and newest positions from history
        oldest_x, oldest_y = self.previous_poses[0]
        newest_x, newest_y = self.previous_poses[-1]
        
        # Calculate distance traveled between oldest and newest positions
        distance_traveled = math.sqrt((newest_x - oldest_x)**2 + (newest_y - oldest_y)**2)
        
        # Check if we're making meaningful progress
        # Higher threshold (20cm) to avoid false positives
        return distance_traveled < 0.2 and self.state == 'forward'
    
    def find_least_visited_direction(self):
        """Find the least visited direction among possible candidate directions"""
        if self.current_pose is None:
            return random.choice([-1, 1])  # Random if no pose data
        
        # Current position in grid coordinates
        current_x, current_y = self.world_to_grid(
            self.current_pose.position.x, 
            self.current_pose.position.y
        )
        
        # Get orientation as yaw angle
        qx = self.current_pose.orientation.x
        qy = self.current_pose.orientation.y
        qz = self.current_pose.orientation.z
        qw = self.current_pose.orientation.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Check visit counts in different directions
        direction_scores = []
        num_directions = 8  # Check 8 different directions
        
        for i in range(num_directions):
            angle = current_yaw + (i * 2 * math.pi / num_directions)
            
            # Look ahead a certain distance in this direction
            look_ahead_distance = 20  # cells
            check_distance = 5  # cells
            
            total_visits = 0
            count = 0
            
            # Sample points along this direction
            for dist in range(check_distance, look_ahead_distance, check_distance):
                x = int(current_x + dist * math.cos(angle))
                y = int(current_y + dist * math.sin(angle))
                
                # Check if within bounds
                if 0 <= x < self.map_width and 0 <= y < self.map_height:
                    # Check if the cell has been visited
                    visits = self.visit_count_grid[y, x]
                    # Unknown areas (visits=0) are prioritized
                    total_visits += visits
                    count += 1
            
            # Calculate average visit count (lower is better)
            avg_visits = total_visits / max(1, count)
            direction_scores.append((i, avg_visits))
        
        # Sort by visit count (ascending)
        direction_scores.sort(key=lambda x: x[1])
        
        # Calculate turn direction based on the least visited direction
        best_direction_idx = direction_scores[0][0]
        best_angle = best_direction_idx * 2 * math.pi / num_directions
        
        # Normalize relative to current orientation
        relative_angle = best_angle - current_yaw
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi
        
        # Return the direction to turn (-1 for right, 1 for left)
        return 1 if relative_angle > 0 else -1
    
    def find_best_direction(self, obstacles):
        # Check if free in front
        if not obstacles['front']:
            return 'forward'
        
        can_turn_left = not obstacles['left']
        can_turn_right = not obstacles['right']

        if can_turn_left and can_turn_right:
            # Both directions are clear, use exploration bias
            exploration_direction = self.find_least_visited_direction()
            return 'left' if exploration_direction > 0 else 'right'
        elif can_turn_left:
            return 'left'
        elif can_turn_right:
            return 'right'
        else:
            # All directions blocked, try a larger turn
            self.get_logger().info("All directions appear blocked. Attempting 180° turn.")
            return 'back'
    
    # Robot behavior controllers
    def handle_stuck_state(self):
        """Special behavior for when the robot is stuck"""
        self.get_logger().info(f"Robot appears stuck! Counter: {self.stuck_counter}")
        
        cmd_vel = Twist()
        
        # Increase the stuck counter
        self.stuck_counter += 1
        
        if self.stuck_counter < 3:
            # First try: back up slightly
            cmd_vel.linear.x = -0.2
            self.vel_pub.publish(cmd_vel)
            time.sleep(1.5)  # Back up for 1.5 second
            
            # Then turn randomly but with a preference for unexplored areas
            cmd_vel.linear.x = 0.0
            direction = self.find_least_visited_direction()
            cmd_vel.angular.z = 0.8 * direction  # Faster turn to escape
            self.vel_pub.publish(cmd_vel)
            time.sleep(2.5)  # Turn for longer
        else:
            # More aggressive escape: spin 360 degrees
            cmd_vel.angular.z = 0.9
            self.vel_pub.publish(cmd_vel)
            time.sleep(8.0)  # Full spin
            
            # Then back up further
            cmd_vel.angular.z = 0.0
            cmd_vel.linear.x = -0.25
            self.vel_pub.publish(cmd_vel)
            time.sleep(2.0)
            
            # Reset stuck counter after full recovery
            self.stuck_counter = 0
        
        # Return to forward state
        cmd_vel = Twist()
        self.state = 'forward'
        
        # Clear position history to avoid re-triggering stuck detection
        self.previous_poses = []
        
        return cmd_vel
    
    def handle_returning_home(self):
        """Logic for returning to home position"""
        cmd_vel = Twist()
        
        # Calculate distance to home
        distance_to_home = self.calculate_distance_to_home()
        angle_to_home = self.calculate_angle_to_home()
        
        # Check if we're close enough to home to consider it reached
        if distance_to_home < self.at_home_threshold:
            self.get_logger().info(f"Reached home position! Distance: {distance_to_home:.2f}m")
            self.state = 'at_home'
            return cmd_vel  # Return zero velocity
        
        # If we need to turn significantly toward home before moving
        if abs(angle_to_home) > math.radians(30):  # More than 30 degrees off
            # Turn toward home
            turn_direction = 1.0 if angle_to_home > 0 else -1.0
            cmd_vel.angular.z = 0.4 * turn_direction  # Turn at slower speed
            self.get_logger().debug(f"Turning toward home. Angle: {math.degrees(angle_to_home):.1f} degrees")
        else:
            # We're facing approximately toward home, so move forward
            cmd_vel.linear.x = self.linear_velocity_forward * 0.8  # Slightly slower when returning
            
            # Apply small correction to heading
            cmd_vel.angular.z = 0.3 * angle_to_home  # Proportional control
            
            self.get_logger().debug(f"Moving toward home. Distance: {distance_to_home:.2f}m, Angle: {math.degrees(angle_to_home):.1f} degrees")
        
        # Check for obstacles while returning home
        obstacles = self.check_obstacles()
        if obstacles['valid'] and obstacles['front']:
            # Obstacle in the way - stop forward motion
            cmd_vel.linear.x = 0.0
            
            # Choose a direction to turn
            if not obstacles['left']:
                cmd_vel.angular.z = self.angular_velocity_turn  # Turn left
            elif not obstacles['right']:
                cmd_vel.angular.z = -self.angular_velocity_turn  # Turn right
            else:
                # Both sides blocked - turn more sharply in the direction least blocked
                cmd_vel.angular.z = self.angular_velocity_turn * 1.5  # Default to sharp left turn
            
            self.get_logger().info("Obstacle detected while returning home. Detouring.")
        
        return cmd_vel
        
    def handle_at_home(self):
        """Robot has reached home, execute final positioning"""
        # Create a zero velocity command
        cmd_vel = Twist()
        
        # Check if we need to perform a final orientation alignment
        # Get current orientation as yaw
        qx = self.current_pose.orientation.x
        qy = self.current_pose.orientation.y
        qz = self.current_pose.orientation.z
        qw = self.current_pose.orientation.w
        
        # Convert quaternion to yaw
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Get initial orientation
        qx_init = self.start_orientation[0]
        qy_init = self.start_orientation[1]
        qz_init = self.start_orientation[2]
        qw_init = self.start_orientation[3]
        
        # Convert to yaw
        siny_cosp = 2.0 * (qw_init * qz_init + qx_init * qy_init)
        cosy_cosp = 1.0 - 2.0 * (qy_init * qy_init + qz_init * qz_init)
        initial_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Calculate angle difference
        angle_diff = initial_yaw - current_yaw
        
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2.0 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2.0 * math.pi
            
        # If we need to align
        if abs(angle_diff) > math.radians(5):  # More than 5 degrees off
            turn_direction = 1.0 if angle_diff > 0 else -1.0
            cmd_vel.angular.z = 0.2 * turn_direction  # Slow turn
            self.get_logger().debug(f"Final alignment. Angle diff: {math.degrees(angle_diff):.1f} degrees")
        
        return cmd_vel

    def control_loop(self):
        if self.scan_data is None or self.current_pose is None:
            return
        
        cmd_vel = Twist()
        
        # State machine for robot behavior
        if self.state == 'waiting_for_first_pose':
            # Do nothing, waiting for first pose to establish home position
            return
            
        elif self.state == 'at_home':
            # We've reached the home position
            cmd_vel = self.handle_at_home()
            
        elif self.state == 'returning_home':
            # Return to home state
            cmd_vel = self.handle_returning_home()
            
        elif self.state == 'stuck':
            # Check if robot is stuck
            cmd_vel = self.handle_stuck_state()
            
        else:  # 'forward' or 'turning' states (normal exploration)
            # Only check for stuck when in forward state and not turning
            # This prevents false stuck detection during intentional turns
            if self.state == 'forward' and self.check_if_stuck():
                # Add an extra validation before declaring stuck
                # Ensure we're trying to move but not actually moving
                # This helps avoid false positive stuck detections
                if cmd_vel.linear.x > 0.05:  # Only if we're trying to move forward
                    self.state = 'stuck'
                    cmd_vel = self.handle_stuck_state()
                    self.vel_pub.publish(cmd_vel)
                    return
            
            obstacles = self.check_obstacles()
            if not obstacles['valid']:
                self.get_logger().warn("Obstacle check returned invalid data. Stopping.", throttle_duration_sec=1.0)
                self.vel_pub.publish(cmd_vel)
                return

            if self.state == 'forward':
                if obstacles['front']:
                    # Stop before turning
                    cmd_vel.linear.x = 0.0
                    self.vel_pub.publish(cmd_vel)
                    
                    # Decide which direction to turn
                    turn_decision = self.find_best_direction(obstacles)
                    self.state = 'turning'
                    
                    # Set turn parameters based on decision
                    if turn_decision == 'left':
                        self.turn_direction = 1
                        self.desired_turn_angle = math.pi / 2  # 90 degrees
                    elif turn_decision == 'right':
                        self.turn_direction = -1
                        self.desired_turn_angle = math.pi / 2
                    elif turn_decision == 'back':
                        self.turn_direction = 1  # Clockwise turn
                        self.desired_turn_angle = math.pi  # 180 degrees
                    
                    self.turn_start_time = self.get_clock().now()
                    self.get_logger().info(f"Obstacle detected. Turning {turn_decision}.")
                else:
                    # No obstacle ahead, continue forward
                    cmd_vel.linear.x = self.linear_velocity_forward
                    
                    # Add slight turning bias based on exploration goals
                    if random.random() < 0.3:  # 30% chance to apply bias
                        bias_direction = self.find_least_visited_direction()
                        cmd_vel.angular.z = 0.1 * bias_direction  # Slight turn toward less visited areas
            
            elif self.state == 'turning':
                if self.turn_start_time is None:
                    self.state = 'forward'
                    return
                
                # Calculate turn duration
                turn_duration_seconds = abs(self.desired_turn_angle / self.angular_velocity_turn)
                
                # Check if turn is complete
                current_time = self.get_clock().now()
                elapsed_turn_time = (current_time - self.turn_start_time).nanoseconds / 1e9
                
                if elapsed_turn_time < turn_duration_seconds:
                    # Still turning
                    cmd_vel.angular.z = self.angular_velocity_turn * self.turn_direction
                else:
                    # Turn complete
                    cmd_vel.angular.z = 0.0
                    self.state = 'forward'
                    self.turn_start_time = None
                    
                    # Clear position history when changing direction to avoid false stuck detection
                    self.previous_poses = []
                    
                    self.get_logger().info("Turn complete. Moving forward.")
        
        self.vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    
    domain_id_for_print = os.environ.get('ROS_DOMAIN_ID', 'NOT_SET')
    print(f"--- Enhanced Explorer starting with ROS_DOMAIN_ID: {domain_id_for_print} ---")
        
    explorer_node = Turtlebot4EnhancedExplorer()
    
    try:
        rclpy.spin(explorer_node)
    except KeyboardInterrupt:
        explorer_node.get_logger().info('Keyboard interrupt received, stopping explorer.')
    except Exception as e:
        import traceback
        explorer_node.get_logger().error(f"Unhandled exception: {e}")
        explorer_node.get_logger().error(f"Traceback: {traceback.format_exc()}")
    finally:
        explorer_node.get_logger().info('Shutting down node, sending zero velocity.')
        stop_msg = Twist()
        if rclpy.ok() and hasattr(explorer_node, 'vel_pub'):
            try:
                explorer_node.vel_pub.publish(stop_msg)
            except Exception as e:
                explorer_node.get_logger().error(f"Error publishing stop message: {e}")

        explorer_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
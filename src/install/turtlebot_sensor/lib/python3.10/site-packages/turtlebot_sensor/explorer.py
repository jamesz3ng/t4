# turtlebot_sensor/turtlebot_sensor/turtlebot4_explorer.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import random
import math

class Turtlebot4Explorer(Node):
    def __init__(self):
        super().__init__('explorer')
        
        # Create QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create publisher for velocity commands
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Create subscriber for laser scan data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            sensor_qos
        )
        
        # Create subscriber for map data
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )
        
        # Initialize state variables
        self.scan_data = None
        self.map_data = None
        self.state = 'forward'  # States: forward, turning, stopping
        self.turn_direction = 1  # 1 for left, -1 for right
        self.obstacle_threshold = 0.5  # in meters
        self.desired_turn_angle = 0.0  # Target angle to turn (radians)
        self.turn_start_time = None
        
        # Create timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Turtlebot4 Explorer node initialized')
    
    def scan_callback(self, msg):
        self.scan_data = msg
    
    def map_callback(self, msg):
        self.map_data = msg
    
    def check_obstacles(self):
        """Check for obstacles in different directions and return safe directions"""
        if self.scan_data is None:
            return {'front': False, 'left': False, 'right': False}
        
        ranges = self.scan_data.ranges
        num_ranges = len(ranges)
        
        # Define sectors (adjust these based on your LiDAR setup)
        front_sector = list(range(num_ranges - 20, num_ranges)) + list(range(0, 20))
        left_sector = list(range(num_ranges // 4 - 20, num_ranges // 4 + 20))
        right_sector = list(range(3 * num_ranges // 4 - 20, 3 * num_ranges // 4 + 20))
        
        # Filter out inf values
        front_ranges = [ranges[i] for i in front_sector if not math.isinf(ranges[i])]
        left_ranges = [ranges[i] for i in left_sector if not math.isinf(ranges[i])]
        right_ranges = [ranges[i] for i in right_sector if not math.isinf(ranges[i])]
        
        # Check if each direction has obstacles
        front_obstacle = len(front_ranges) > 0 and min(front_ranges) < self.obstacle_threshold
        left_obstacle = len(left_ranges) > 0 and min(left_ranges) < self.obstacle_threshold
        right_obstacle = len(right_ranges) > 0 and min(right_ranges) < self.obstacle_threshold
        
        return {
            'front': front_obstacle,
            'left': left_obstacle,
            'right': right_obstacle
        }
    
    def find_best_direction(self, obstacles):
        """Determine the best direction to turn based on obstacles"""
        if not obstacles['front']:
            return 'forward'
        elif not obstacles['left'] and obstacles['right']:
            return 'left'
        elif obstacles['left'] and not obstacles['right']:
            return 'right'
        elif not obstacles['left'] and not obstacles['right']:
            # Both directions are clear, choose randomly
            return random.choice(['left', 'right'])
        else:
            # All directions have obstacles, try to turn around
            return 'left'  # Default to left for a 180° turn
    
    def control_loop(self):
        if self.scan_data is None:
            return
        
        cmd_vel = Twist()
        obstacles = self.check_obstacles()
        
        if self.state == 'forward':
            if obstacles['front']:
                # Obstacle ahead, decide which way to turn
                turn_direction = self.find_best_direction(obstacles)
                self.state = 'turning'
                
                if turn_direction == 'left':
                    self.turn_direction = 1
                    # Choose a random turn angle between π/4 and π/2
                    self.desired_turn_angle = random.uniform(math.pi/4, math.pi/2)
                elif turn_direction == 'right':
                    self.turn_direction = -1
                    self.desired_turn_angle = random.uniform(math.pi/4, math.pi/2)
                else:
                    # If all directions are blocked, try a larger turn
                    self.turn_direction = 1
                    self.desired_turn_angle = math.pi  # 180° turn
                
                self.turn_start_time = self.get_clock().now()
                self.get_logger().info(f'Obstacle detected. Turning {turn_direction}...')
            else:
                # No obstacle, move forward
                cmd_vel.linear.x = 0.2  # m/s
        
        elif self.state == 'turning':
            # Calculate how long to turn based on desired angle and angular velocity
            angular_velocity = 0.5  # rad/s
            turn_time_seconds = self.desired_turn_angle / angular_velocity
            
            # Check if we've turned enough
            current_time = self.get_clock().now()
            elapsed = (current_time - self.turn_start_time).nanoseconds / 1e9
            
            if elapsed < turn_time_seconds:
                # Still turning
                cmd_vel.angular.z = angular_velocity * self.turn_direction
            else:
                # Finished turning, go forward
                self.state = 'forward'
                self.get_logger().info('Turn complete. Moving forward...')
        
        # Publish velocity command
        self.vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    explorer = Turtlebot4Explorer()
    
    try:
        rclpy.spin(explorer)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot
        stop_msg = Twist()
        explorer.vel_pub.publish(stop_msg)
        
        explorer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
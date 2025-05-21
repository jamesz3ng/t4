# turtlebot_sensor/turtlebot_sensor/turtlebot4_explorer.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import random
import math
import os # Import os to access environment variables

class Turtlebot4Explorer(Node):
    def __init__(self):
        super().__init__('turtlebot4_explorer') # Node name can be generic

        # --- Determine Robot ID and Topic Names ---
        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID')
        if self.robot_id_str is None:
            self.get_logger().error("ROS_DOMAIN_ID environment variable not set! Cannot determine robot namespace. Exiting.")
            # You might want to raise an exception or sys.exit(1) here
            # For now, let's default to a placeholder to avoid immediate crash, but log an error
            self.robot_id_str = "0" # Fallback, but this is bad
            # A better approach is to make it mandatory:
            # raise ValueError("ROS_DOMAIN_ID must be set.")

        self.robot_namespace = f"/T{self.robot_id_str}"
        self.scan_topic_name = f"{self.robot_namespace}/scan"
        self.cmd_vel_topic_name = f"{self.robot_namespace}/cmd_vel"

        self.get_logger().info(f"Initializing for robot namespace: {self.robot_namespace}")
        self.get_logger().info(f"Subscribing to scan on: {self.scan_topic_name}")
        self.get_logger().info(f"Publishing commands to: {self.cmd_vel_topic_name}")
        # --- End Determine Robot ID ---
        
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5 
        )
        
        self.vel_pub = self.create_publisher(Twist, self.cmd_vel_topic_name, 10)
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic_name,
            self.scan_callback,
            sensor_qos
        )
        
        # ... (rest of your __init__ variables) ...
        self.scan_data = None
        self.state = 'forward'
        self.turn_direction = 1
        self.obstacle_threshold = 0.5
        self.desired_turn_angle = 0.0
        self.turn_start_time = None
        self.angular_velocity_turn = 0.5
        self.linear_velocity_forward = 0.15
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.control_loop)
        
        self.get_logger().info('Turtlebot4 Explorer node instance initialized. Waiting for scan data...')
    
    # ... (scan_callback, is_valid_range, check_obstacles, find_best_direction remain the same) ...
    def scan_callback(self, msg):
        self.scan_data = msg
        # self.get_logger().info('Scan data received.', throttle_duration_sec=5.0) # Keep this minimal once it works
    
    def is_valid_range(self, r):
        return not math.isinf(r) and not math.isnan(r)

    def check_obstacles(self):
        if self.scan_data is None:
            return {'front': False, 'left': False, 'right': False, 'valid': False}
        
        ranges = self.scan_data.ranges
        num_ranges = len(ranges)

        if num_ranges == 0:
            # self.get_logger().warn("Received empty scan ranges.") # Can be noisy
            return {'front': False, 'left': False, 'right': False, 'valid': False}

        angular_width_indices = 20 

        front_sector_indices = list(range(num_ranges - angular_width_indices, num_ranges)) + \
                               list(range(0, angular_width_indices + 1))
        
        left_center_idx = num_ranges // 4
        left_sector_indices = list(range(left_center_idx - angular_width_indices, left_center_idx + angular_width_indices + 1))
        
        right_center_idx = 3 * num_ranges // 4
        right_sector_indices = list(range(right_center_idx - angular_width_indices, right_center_idx + angular_width_indices + 1))
        
        front_sector_indices = [idx % num_ranges for idx in front_sector_indices]
        left_sector_indices = [idx % num_ranges for idx in left_sector_indices]
        right_sector_indices = [idx % num_ranges for idx in right_sector_indices]

        front_ranges = [ranges[i] for i in front_sector_indices if self.is_valid_range(ranges[i])]
        left_ranges = [ranges[i] for i in left_sector_indices if self.is_valid_range(ranges[i])]
        right_ranges = [ranges[i] for i in right_sector_indices if self.is_valid_range(ranges[i])]
        
        front_obstacle = len(front_ranges) > 0 and min(front_ranges) < self.obstacle_threshold
        left_obstacle = len(left_ranges) > 0 and min(left_ranges) < self.obstacle_threshold
        right_obstacle = len(right_ranges) > 0 and min(right_ranges) < self.obstacle_threshold
        
        return {
            'front': front_obstacle,
            'left': left_obstacle,
            'right': right_obstacle,
            'valid': True
        }
    
    def find_best_direction(self, obstacles):
        if not obstacles['front']:
            return 'forward' 
        
        can_turn_left = not obstacles['left']
        can_turn_right = not obstacles['right']

        if can_turn_left and can_turn_right:
            # self.get_logger().info("Front blocked, left & right clear. Choosing randomly.")
            return random.choice(['left', 'right'])
        elif can_turn_left:
            # self.get_logger().info("Front blocked, only left clear. Turning left.")
            return 'left'
        elif can_turn_right:
            # self.get_logger().info("Front blocked, only right clear. Turning right.")
            return 'right'
        else: 
            # self.get_logger().info("Front, left, and right blocked. Initiating larger turn (default left).")
            return 'left' 

    def control_loop(self):
        if self.scan_data is None:
            # self.get_logger().warn('Control loop called but no scan data yet.', throttle_duration_sec=5.0)
            # Don't publish stop here repeatedly if scan is just slow to start
            return 
        
        # self.get_logger().info(f"Control loop. State: {self.state}", throttle_duration_sec=1.0)
        
        cmd_vel = Twist() 
        obstacles = self.check_obstacles()

        if not obstacles['valid']:
            self.get_logger().warn("Obstacle check returned invalid data. Stopping.", throttle_duration_sec=1.0)
            self.vel_pub.publish(cmd_vel)
            return

        # self.get_logger().info(
        #     f"Obstacles: F:{obstacles['front']}, L:{obstacles['left']}, R:{obstacles['right']}",
        #     throttle_duration_sec=1.0
        # )

        if self.state == 'forward':
            if obstacles['front']:
                cmd_vel.linear.x = 0.0 
                self.vel_pub.publish(cmd_vel) 

                turn_decision = self.find_best_direction(obstacles)
                self.state = 'turning'
                
                if turn_decision == 'left':
                    self.turn_direction = 1 
                    self.desired_turn_angle = math.pi / 2 if not (obstacles['left'] or obstacles['right']) else math.pi * random.uniform(0.4, 0.6)
                elif turn_decision == 'right':
                    self.turn_direction = -1 
                    self.desired_turn_angle = math.pi / 2 if not (obstacles['left'] or obstacles['right']) else math.pi * random.uniform(0.4, 0.6)
                
                self.turn_start_time = self.get_clock().now()
                self.get_logger().info(f"Obstacle detected. Transitioning to 'turning {turn_decision}'. Desired angle: {math.degrees(self.desired_turn_angle):.1f} deg.")
            else:
                cmd_vel.linear.x = self.linear_velocity_forward
                # self.get_logger().info(f"State: forward, No obstacle, moving linear.x: {cmd_vel.linear.x:.2f}", throttle_duration_sec=1.0)
        
        elif self.state == 'turning':
            if self.turn_start_time is None: 
                self.get_logger().error("In 'turning' state but turn_start_time is None. Resetting to 'forward'.")
                self.state = 'forward'
                self.vel_pub.publish(cmd_vel) 
                return

            turn_duration_seconds = abs(self.desired_turn_angle / self.angular_velocity_turn)
            
            current_time = self.get_clock().now()
            elapsed_turn_time = (current_time - self.turn_start_time).nanoseconds / 1e9
            
            if elapsed_turn_time < turn_duration_seconds:
                cmd_vel.angular.z = self.angular_velocity_turn * self.turn_direction
                # self.get_logger().info(f"State: turning, Elapsed: {elapsed_turn_time:.2f}/{turn_duration_seconds:.2f}s, angular.z: {cmd_vel.angular.z:.2f}", throttle_duration_sec=0.5)
            else:
                cmd_vel.angular.z = 0.0 
                self.state = 'forward'
                self.turn_start_time = None 
                self.get_logger().info(f"Turn complete ({elapsed_turn_time:.2f}s). Transitioning to 'forward'.")
        
        # Use the dynamic topic name in the log
        self.get_logger().debug(f"Publishing to {self.cmd_vel_topic_name}: linear.x={cmd_vel.linear.x:.2f}, angular.z={cmd_vel.angular.z:.2f}", throttle_duration_sec=0.2)
        self.vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    
    # --- Critical: Check for ROS_DOMAIN_ID before creating the node if it's essential for node init ---
    # However, it's cleaner to let the node handle it internally as done above.
    # If you wanted to prevent node creation entirely:
    # robot_id_str = os.environ.get('ROS_DOMAIN_ID')
    # if robot_id_str is None:
    #     print("ERROR: ROS_DOMAIN_ID environment variable not set! Exiting.", file=sys.stderr)
    #     sys.exit(1)
        
    explorer_node = Turtlebot4Explorer()
    
    try:
        rclpy.spin(explorer_node)
    except KeyboardInterrupt:
        explorer_node.get_logger().info('Keyboard interrupt received, stopping explorer.')
    except Exception as e:
        explorer_node.get_logger().error(f"Unhandled exception in spin: {e}") # Log other exceptions
    finally:
        explorer_node.get_logger().info('Shutting down node, sending zero velocity.')
        stop_msg = Twist()
        if rclpy.ok() and hasattr(explorer_node, 'vel_pub') and explorer_node.vel_pub is not None:
            try:
                # Check if the publisher itself is still valid
                if hasattr(explorer_node.vel_pub, 'handle') and explorer_node.vel_pub.handle:
                     # Optional: Check for subscribers before publishing the final stop
                    # if explorer_node.vel_pub.get_subscription_count() > 0:
                    explorer_node.vel_pub.publish(stop_msg)
                    explorer_node.get_logger().info(f"Final stop command published to {explorer_node.cmd_vel_topic_name}.")
                    # else:
                    #    explorer_node.get_logger().info(f"No subscribers to {explorer_node.cmd_vel_topic_name}, not sending final stop.")
                else:
                    explorer_node.get_logger().warn("vel_pub handle is invalid during shutdown, cannot send stop.")
            except rclpy.exceptions.RCLError as e:
                explorer_node.get_logger().warn(f"Could not publish stop message during shutdown: {e}")
            except Exception as e: # Catch any other unexpected error during this sensitive phase
                explorer_node.get_logger().error(f"Unexpected error publishing stop message: {e}")

        if hasattr(explorer_node, 'destroy_node'): # Check if destroy_node method exists
             explorer_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    print(os.environ.get('ROS_DOMAIN_ID'))
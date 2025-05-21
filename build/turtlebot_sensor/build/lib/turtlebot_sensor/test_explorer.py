#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import math
import numpy as np
import time

class ExplorerTester(Node):
    def __init__(self):
        super().__init__('explorer_tester')
        
        # Publishers
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.diagnostics_sub = self.create_subscription(
            String, '/explorer/diagnostics', self.diagnostics_callback, 10)
        
        # Timer for publishing fake scan data
        self.timer = self.create_timer(0.1, self.publish_fake_scan)
        
        # Test scenario: 0=clear path, 1=obstacle in front, 2=obstacle left, 3=obstacle right, 4=all blocked
        self.test_scenario = 0
        self.scenario_timer = self.create_timer(5.0, self.change_scenario)
        
        # Track robot's virtual position for visualization
        self.position_x = 0.0
        self.position_y = 0.0
        self.orientation = 0.0  # radians
        self.last_cmd_vel = None
        self.last_cmd_time = time.time()
        
        self.get_logger().info('Explorer Tester started. Use "ros2 param set /explorer_tester test_scenario X" to change scenarios')
        self.declare_parameter('test_scenario', 0)
        
    def cmd_vel_callback(self, msg):
        self.last_cmd_vel = msg
        current_time = time.time()
        dt = current_time - self.last_cmd_time
        self.last_cmd_time = current_time
        
        # Update virtual position based on velocity commands
        if msg.linear.x != 0.0:
            self.position_x += msg.linear.x * dt * math.cos(self.orientation)
            self.position_y += msg.linear.x * dt * math.sin(self.orientation)
        if msg.angular.z != 0.0:
            self.orientation += msg.angular.z * dt
            
        self.get_logger().info(f'Robot at: ({self.position_x:.2f}, {self.position_y:.2f}), Orientation: {math.degrees(self.orientation):.1f}°')
    
    def diagnostics_callback(self, msg):
        self.get_logger().info(f'Explorer Diagnostics: {msg.data}')
    
    def change_scenario(self):
        # Read parameter to update test scenario
        self.test_scenario = self.get_parameter('test_scenario').value
        scenarios = {
            0: "Clear path all around",
            1: "Obstacle in front",
            2: "Obstacle on left",
            3: "Obstacle on right",
            4: "Obstacles all around"
        }
        self.get_logger().info(f'Changing to scenario {self.test_scenario}: {scenarios.get(self.test_scenario, "Unknown")}')
    
    def publish_fake_scan(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_scan'
        
        # Configure fake laser scan to match your LIDAR
        msg.angle_min = -3.14159  # -π
        msg.angle_max = 3.14159   # π
        msg.angle_increment = 0.01745  # 1 degree in radians
        num_readings = int((msg.angle_max - msg.angle_min) / msg.angle_increment) + 1
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = 0.1
        msg.range_max = 10.0
        
        # Default: all clear (ranges set to max range)
        ranges = [msg.range_max] * num_readings
        
        # Add obstacles based on scenario
        if self.test_scenario >= 1:  # Obstacle in front
            # Front sector (340° to 20°)
            for i in range(num_readings - 20, num_readings):
                ranges[i] = 0.3
            for i in range(0, 20):
                ranges[i] = 0.3
                
        if self.test_scenario == 2 or self.test_scenario == 4:  # Obstacle on left
            # Left sector (70° to 110°)
            left_start = int((70 * math.pi / 180 - msg.angle_min) / msg.angle_increment)
            left_end = int((110 * math.pi / 180 - msg.angle_min) / msg.angle_increment)
            for i in range(left_start, left_end):
                ranges[i] = 0.3
                
        if self.test_scenario == 3 or self.test_scenario == 4:  # Obstacle on right
            # Right sector (250° to 290°)
            right_start = int((250 * math.pi / 180 - msg.angle_min) / msg.angle_increment)
            right_end = int((290 * math.pi / 180 - msg.angle_min) / msg.angle_increment)
            for i in range(right_start, right_end):
                ranges[i] = 0.3
        
        msg.ranges = ranges
        self.scan_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    tester = ExplorerTester()
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
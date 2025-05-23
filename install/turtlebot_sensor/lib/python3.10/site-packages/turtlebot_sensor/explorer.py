import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
import time
from collections import deque
import os # Added for ROS_DOMAIN_ID
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy # Added for QoS

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # Namespace setup
        self.robot_id_str = os.environ.get('ROS_DOMAIN_ID')
        if self.robot_id_str is None:
            self.get_logger().error("ROS_DOMAIN_ID environment variable not set! Defaulting to 0.")
            self.robot_id_str = "0"
        self.robot_namespace = f"/T{self.robot_id_str}"
        self.get_logger().info(f"Frontier Explorer node initialized for namespace: {self.robot_namespace}")

        self.map_data = None
        self.robot_pose = None # Stores the latest PoseStamped from odom
        self.cube_seen_counter = 0
        self.seeking_cube = False
        self.cube_x_global = None
        self.cube_y_global = None
        self.returning_home = False
        self.start_pose = None # Stores the first PoseStamped from odom

        # For directional sweeping
        self.current_direction = 0  # 0=N(positive Y), 1=E(positive X), 2=S(negative Y), 3=W(negative X)
        self.direction_vectors = [(0, 1), (1, 0), (0, -1), (-1, 0)] # (dx, dy) in world frame
        self.step_distance = 2.0  # meters
        self.turn_direction = 1  # 1 = right turns (clockwise N->E->S->W), -1 = left turns

        # Parameters (declare with base names)
        self.declare_parameter('map_topic', 'map')      # Default to global /map
        self.declare_parameter('odom_topic', 'odom')     # Base name for namespaced odom
        self.declare_parameter('goal_topic', 'goal_pose')# Base name for namespaced goal
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')# Base name for namespaced cmd_vel
        self.declare_parameter('waypoints_topic', 'waypoints') # Base name for namespaced waypoints
        self.declare_parameter('cube_pose_topic', 'cube_pose') # Base name for namespaced cube_pose
        self.declare_parameter('qos_depth', 10)

        map_topic_param = self.get_parameter('map_topic').get_parameter_value().string_value
        odom_topic_param = self.get_parameter('odom_topic').get_parameter_value().string_value
        goal_topic_param = self.get_parameter('goal_topic').get_parameter_value().string_value
        cmd_vel_topic_param = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        waypoints_topic_param = self.get_parameter('waypoints_topic').get_parameter_value().string_value
        cube_pose_topic_param = self.get_parameter('cube_pose_topic').get_parameter_value().string_value
        qos_depth = self.get_parameter('qos_depth').get_parameter_value().integer_value

        # Construct full topic names
        # Map topic is usually global, not namespaced per robot unless specifically intended
        self.map_topic_actual = f"{self.robot_namespace}/{map_topic_param}"
        self.odom_topic_actual = f"{self.robot_namespace}/{odom_topic_param}"
        self.goal_topic_actual = f"{self.robot_namespace}/{goal_topic_param}"
        self.cmd_vel_topic_actual = f"{self.robot_namespace}/{cmd_vel_topic_param}"
        self.waypoints_topic_actual = f"{self.robot_namespace}/{waypoints_topic_param}"
        self.cube_pose_topic_actual = f"{self.robot_namespace}/{cube_pose_topic_param}"

        self.get_logger().info(f"Subscribing to map: {self.map_topic_actual}")
        self.get_logger().info(f"Subscribing to odom: {self.odom_topic_actual}")
        self.get_logger().info(f"Subscribing to cube_pose: {self.cube_pose_topic_actual}")
        self.get_logger().info(f"Publishing goal to: {self.goal_topic_actual}")
        self.get_logger().info(f"Publishing cmd_vel to: {self.cmd_vel_topic_actual}")
        self.get_logger().info(f"Publishing waypoints to: {self.waypoints_topic_actual}")

        # QoS profile for map
        qos_profile_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # General QoS profile for other topics
        qos_profile_others = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # Odom and cube_pose often BEST_EFFORT
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth
        )

        self.create_subscription(OccupancyGrid, self.map_topic_actual, self.map_callback, qos_profile_map)
        self.create_subscription(Odometry, self.odom_topic_actual, self.odom_callback, qos_profile_others)
        self.create_subscription(PoseStamped, self.cube_pose_topic_actual, self.cube_callback, qos_profile_others)
        self.goal_pub = self.create_publisher(PoseStamped, self.goal_topic_actual, qos_depth)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic_actual, qos_depth)
        self.marker_pub = self.create_publisher(MarkerArray, self.waypoints_topic_actual, 10) # Standard for markers

        self.timer = self.create_timer(5.0, self.timer_callback) # Timer for exploration logic

    def map_callback(self, msg: OccupancyGrid):
        self.get_logger().info("Map callback triggered!", throttle_duration_sec=10.0)
        self.map_data = msg

    def odom_callback(self, msg: Odometry):
        self.get_logger().info("Odom callback triggered!", throttle_duration_sec=10.0)
        # Store the full PoseStamped message for convenience
        current_pose_stamped = PoseStamped()
        current_pose_stamped.header = msg.header # Odom frame_id usually T_X/odom
        current_pose_stamped.pose = msg.pose.pose
        self.robot_pose = current_pose_stamped

        if self.start_pose is None:
            self.start_pose = current_pose_stamped
            self.get_logger().info(f"Start pose captured at ({self.start_pose.pose.position.x:.2f}, {self.start_pose.pose.position.y:.2f}) in frame '{self.start_pose.header.frame_id}'")

    def cube_callback(self, msg: PoseStamped):
        self.get_logger().info("Cube callback triggered!", throttle_duration_sec=5.0)
        if self.returning_home:
            return

        # Assuming cube_pose (msg) is in a robot-local frame (e.g., camera_link or base_link_oriented_forward)
        # x: lateral offset (positive right), y: forward distance
        x_rel_cube, y_rel_cube = msg.pose.position.x, msg.pose.position.y

        if x_rel_cube == 0.0 and y_rel_cube == 0.0: # Convention for "no cube detected"
            self.seeking_cube = False
            self.cube_seen_counter = 0 # Reset if cube lost
            # Optionally stop the robot if it was moving towards a non-existent cube
            # self.cmd_pub.publish(Twist())
            return

        self.seeking_cube = True
        # Alignment thresholds (adjust as needed)
        lateral_threshold = 0.2 # meters
        forward_target_distance = 2.0 # meters
        forward_stop_distance = 2.5 # meters, stop if closer than this to target

        if abs(x_rel_cube) < lateral_threshold and y_rel_cube < forward_stop_distance : # aligned and close enough
            self.cube_seen_counter += 1
            self.get_logger().info(f"Cube aligned. Seen {self.cube_seen_counter} times.")
            # Stop the robot by publishing zero velocity or by setting a goal at current pose
            if self.robot_pose:
                 self.publish_goal(self.robot_pose) # Tell planner to hold position
            else:
                self.cmd_pub.publish(Twist()) # Simple stop
        else:
            self.cube_seen_counter = 0 # Reset if alignment lost
            cmd = Twist()
            # Proportional control for alignment
            cmd.angular.z = -0.5 * x_rel_cube  # Turn to reduce lateral offset
            forward_error = y_rel_cube - forward_target_distance
            cmd.linear.x = 0.2 * forward_error # Move to target forward distance

            # Clamp velocities
            cmd.angular.z = max(min(cmd.angular.z, 0.3), -0.3)
            cmd.linear.x = max(min(cmd.linear.x, 0.25), -0.25)
            if abs(forward_error) < 0.1: # If very close to target distance, prioritize turning
                cmd.linear.x = 0.0

            self.cmd_pub.publish(cmd)
            self.get_logger().info(f"Adjusting to cube: lin_x={cmd.linear.x:.2f}, ang_z={cmd.angular.z:.2f}")


        if self.cube_seen_counter >= 30: # Need fewer consistent sightings if alignment check is good
            if self.robot_pose is None:
                self.get_logger().warn("Cannot determine cube global position: robot_pose is None.")
                return

            # Transform cube's relative pose to global 'map' frame
            # self.robot_pose is in odom frame (e.g., T18/odom)
            # The cube marker and return goal need to be in 'map' frame.
            # This requires a TF transform from robot_pose.header.frame_id to 'map'.
            # For simplicity here, we assume robot_pose can be used directly *if* odom frame IS map frame,
            # or if a TF listener/buffer were added to transform it.
            # Let's proceed with the calculation assuming robot_pose IS in the 'map' frame for now.
            # A TODO for robustness: Add TF transformation if robot_pose.header.frame_id != 'map'

            x_robot = self.robot_pose.pose.position.x
            y_robot = self.robot_pose.pose.position.y
            q = self.robot_pose.pose.orientation
            # Extract yaw from robot's orientation
            robot_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z))

            # Transform local cube coordinates to global frame
            self.cube_x_global = x_robot + math.cos(robot_yaw) * y_rel_cube - math.sin(robot_yaw) * x_rel_cube
            self.cube_y_global = y_robot + math.sin(robot_yaw) * y_rel_cube + math.cos(robot_yaw) * x_rel_cube

            self.get_logger().info(f"Cube acquired at global approx ({self.cube_x_global:.2f}, {self.cube_y_global:.2f}). Returning to charging station.")
            self.returning_home = True
            self.seeking_cube = False # No longer seeking after acquiring
            if self.start_pose:
                # Ensure start_pose is also in 'map' frame if used as goal
                # TODO: Add TF transformation if start_pose.header.frame_id != 'map'
                return_goal = PoseStamped()
                return_goal.header.stamp = self.get_clock().now().to_msg()
                return_goal.header.frame_id = "map" # Explicitly set goal frame
                return_goal.pose = self.start_pose.pose # Copy pose data
                self.publish_goal(return_goal)
            else:
                self.get_logger().warn("Start pose not set, cannot publish return home goal.")

    def publish_goal(self, goal_pose_stamped: PoseStamped):
        # The planning_node expects goals in the 'map' frame.
        # Ensure the input goal_pose_stamped is already in, or transformed to, the 'map' frame.
        # The code here assumes that if goal_pose_stamped is self.robot_pose or self.start_pose,
        # a TF transform to 'map' might be needed if their original frame_id is not 'map'.
        # For now, we create a new PoseStamped and set its frame_id to 'map'.

        final_goal = PoseStamped()
        final_goal.header.stamp = self.get_clock().now().to_msg()
        final_goal.header.frame_id = "map" # Crucial: Planner expects 'map' frame
        final_goal.pose = goal_pose_stamped.pose # Copy pose data (position and orientation)

        # If the input goal_pose_stamped was NOT in 'map' frame, its pose data would be incorrect
        # relative to 'map' unless transformed. This is a common point of failure.
        # For this example, we proceed with the assumption that the pose data is appropriate for 'map'.

        self.goal_pub.publish(final_goal)
        self.get_logger().info(f"Published goal to ({final_goal.pose.position.x:.2f}, {final_goal.pose.position.y:.2f}) in '{final_goal.header.frame_id}' frame.")

    def timer_callback(self):
        if self.map_data is None or self.robot_pose is None:
            self.get_logger().info("Timer: Map data or robot pose not available yet.", throttle_duration_sec=5.0)
            return

        if self.returning_home:
            if self.cube_x_global is not None and self.cube_y_global is not None:
                marker = Marker()
                marker.header.frame_id = "map" # Cube marker in global map frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"{self.robot_namespace}_cube_marker"
                marker.id = 0
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = self.cube_x_global
                marker.pose.position.y = self.cube_y_global
                marker.pose.position.z = 0.1
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.2; marker.scale.y = 0.2; marker.scale.z = 0.2
                marker.color.r = 1.0; marker.color.g = 0.5; marker.color.b = 0.0; marker.color.a = 1.0 # Orange
                marker.lifetime.sec = 0 # Permanent

                marker_array = MarkerArray()
                marker_array.markers.append(marker)
                self.marker_pub.publish(marker_array)
                # self.get_logger().info(f"Publishing cube marker at ({self.cube_x_global:.2f}, {self.cube_y_global:.2f})")
            # Once returning home, this node primarily waits for the planner to reach the goal.
            # It could check if the goal is reached and then, e.g., shutdown or enter a final state.
            return

        if self.seeking_cube:
            # If actively seeking cube, cube_callback handles direct cmd_vel or goal setting.
            # Timer callback shouldn't interfere with sweep logic.
            return

        # If not returning home and not seeking cube, perform sweep exploration.
        goal_for_sweep = self.find_next_directional_goal()
        if goal_for_sweep:
            self.publish_goal(goal_for_sweep)

    def find_next_directional_goal(self):
        # This function assumes self.map_data and self.robot_pose are available (checked in timer_callback)
        # And that self.robot_pose is in a frame that can be used for world calculations (e.g. 'map' or 'T_X/odom' if TF is good)

        map_info = self.map_data.info
        width, height = map_info.width, map_info.height
        resolution = map_info.resolution
        origin_x, origin_y = map_info.origin.position.x, map_info.origin.position.y

        try:
            map_grid_data = np.array(self.map_data.data).reshape((height, width))
        except ValueError:
            self.get_logger().error(f"Map data shape mismatch: {len(self.map_data.data)} vs {height*width}")
            return None

        # Current robot position from self.robot_pose (e.g., in 'T18/odom' frame)
        # For sweep logic, we need positions relative to the map's origin if checking occupancy
        # Or, if self.robot_pose is already in 'map' frame, this is simpler.
        # Assume self.robot_pose.pose.position is in the 'map' frame for this calculation
        # TODO: Add TF transform if self.robot_pose.header.frame_id != 'map'
        robot_x_map_frame = self.robot_pose.pose.position.x
        robot_y_map_frame = self.robot_pose.pose.position.y

        # Calculate goal position in the same frame as robot_x_map_frame, robot_y_map_frame
        dx_world, dy_world = self.direction_vectors[self.current_direction]
        goal_x_world = robot_x_map_frame + dx_world * self.step_distance
        goal_y_world = robot_y_map_frame + dy_world * self.step_distance

        # Convert world goal (which is in map frame) to map grid coordinates for occupancy check
        goal_mx = int((goal_x_world - origin_x) / resolution)
        goal_my = int((goal_y_world - origin_y) / resolution)

        # Check if the target cell is within map bounds and is free (value 0)
        if 0 <= goal_mx < width and 0 <= goal_my < height and map_grid_data[goal_my, goal_mx] == 0:
            goal_pose = PoseStamped()
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.header.frame_id = "map" # Explicitly 'map' for the planner
            goal_pose.pose.position.x = goal_x_world
            goal_pose.pose.position.y = goal_y_world
            goal_pose.pose.position.z = 0.0 # Assuming planar movement

            # Set orientation to face the direction of movement
            target_yaw = math.atan2(dy_world, dx_world)
            goal_pose.pose.orientation.x = 0.0
            goal_pose.pose.orientation.y = 0.0
            goal_pose.pose.orientation.z = math.sin(target_yaw / 2.0)
            goal_pose.pose.orientation.w = math.cos(target_yaw / 2.0)

            self.get_logger().info(f"Found sweep goal at ({goal_x_world:.2f}, {goal_y_world:.2f}) map, yaw {math.degrees(target_yaw):.1f} deg.")
            return goal_pose
        else:
            # Obstacle or boundary detected in the current sweep direction, or goal outside map
            self.get_logger().info(f"Obstacle/boundary at sweep step. Goal_map_coords:({goal_mx},{goal_my}). Rotating. Old direction idx: {self.current_direction}")
            self.current_direction = (self.current_direction + self.turn_direction) % 4 # Rotate
            self.get_logger().info(f"New sweep direction idx: {self.current_direction}")
            # After rotating, the next timer call will attempt to find a goal in the new direction.
            return None

def main(args=None):
    rclpy.init(args=args)
    frontier_explorer_node = FrontierExplorer()
    try:
        rclpy.spin(frontier_explorer_node)
    except KeyboardInterrupt:
        frontier_explorer_node.get_logger().info("Keyboard interrupt, shutting down FrontierExplorer.")
    finally:
        frontier_explorer_node.destroy_node()
        if rclpy.ok(): # Check if shutdown wasn't already called
            rclpy.shutdown()

if __name__ == '__main__':
    main()
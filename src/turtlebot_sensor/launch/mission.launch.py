# Example: mission_coordinator_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value=EnvironmentVariable('ROS_DOMAIN_ID', default_value='0'), # Or your preferred default
        description='Robot ID from ROS_DOMAIN_ID environment variable'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )

    robot_id = LaunchConfiguration('robot_id')
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # Construct the robot namespace for the node itself, if desired
    # If MissionCoordinator shouldn't be in a namespace itself, but just needs
    # to *access* namespaced topics, the node's namespace can be global ('/').
    # However, its TF listener needs to remap.
    # For simplicity, let's assume MissionCoordinator itself doesn't need to be namespaced,
    # only its topic subscriptions.
    
    mission_coordinator_node = Node(
        package='turtlebot_sensor', # Replace with your_package_name
        executable='mission_node',  # Replace with your executable name for MissionCoordinator
        name='mission_coordinator',
        # namespace= '', # Or ['/T', robot_id] if you want the node itself namespaced
        parameters=[{
            'use_sim_time': use_sim_time,
            # Add any other parameters MissionCoordinator needs
            # 'target_map_frame': 'odom', # This is already a param in your node
            # 'home_arrival_threshold': 0.3,
        }],
        remappings=[
            # TF remappings are crucial
            ('/tf', ['/T', robot_id, '/tf']),
            ('/tf_static', ['/T', robot_id, '/tf_static']),
            
            # Remap other topics if MissionCoordinator publishes/subscribes to them under its own namespace
            # but they should actually be under the robot's namespace.
            # Example, if MissionCoordinator itself is not namespaced but its topics should be:
            # ('mission_state', ['/T', robot_id, '/mission_state']),
            # ('exploration_enable', ['/T', robot_id, '/exploration_enable']),
            # ('goal_pose', ['/T', robot_id, '/goal_pose']),
            # ('cmd_vel', ['/T', robot_id, '/cmd_vel']),
            # ('mission_markers', ['/T', robot_id, '/mission_markers']),
            
            # Your current MissionCoordinator code *already* constructs topics like:
            # f"{self.robot_namespace}/odom" which becomes "/T<ID>/odom"
            # f"{self.robot_namespace}/mission_state" which becomes "/T<ID>/mission_state"
            # So, if the MissionCoordinator node *itself* is NOT namespaced in the launch file (namespace=''),
            # then these topic constructions are fine.
            # The remappings for /tf and /tf_static are for the *internal* subscriptions
            # made by the tf2_ros.TransformListener.
        ],
        output='screen',
        # Ensure ROS_DOMAIN_ID is set correctly in the environment where this launch file is run,
        # or pass it to the node via parameters if preferred.
        # The LaunchConfiguration('robot_id') will pick up ROS_DOMAIN_ID.
    )

    return LaunchDescription([
        robot_id_arg,
        use_sim_time_arg,
        mission_coordinator_node
    ])
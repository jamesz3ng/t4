#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():
    # Declare launch arguments
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value=EnvironmentVariable('ROS_DOMAIN_ID', default_value='20'),
        description='Robot ID from ROS_DOMAIN_ID environment variable'
    )
    
    launch_planning_arg = DeclareLaunchArgument(
        'launch_planning',
        default_value='true',
        description='Whether to launch the planning node'
    )
    
    launch_explorer_arg = DeclareLaunchArgument(
        'launch_explorer', 
        default_value='true',
        description='Whether to launch the frontier explorer node'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )

    # Get launch configuration values
    robot_id = LaunchConfiguration('robot_id')
    launch_planning = LaunchConfiguration('launch_planning')
    launch_explorer = LaunchConfiguration('launch_explorer')
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # Robot namespace
    robot_namespace = ['/T', robot_id]
    
    # Planning Node
    planning_node = Node(
        package='turtlebot_sensor',
        executable='plan_node',
        name='plan_node',
        namespace=robot_namespace,
        parameters=[{
            'use_sim_time': use_sim_time,
            'map_topic_base': 'map',
            'goal_topic_base': 'goal_pose', 
            'cmd_vel_topic_base': 'cmd_vel',
            'waypoints_topic_base': 'waypoints'
        }],
        remappings=[
            # Critical TF remappings for namespaced robots
            ('/tf', ['/T', robot_id, '/tf']),
            ('/tf_static', ['/T', robot_id, '/tf_static']),
        ],
        output='screen',
        condition=IfCondition(launch_planning)
    )
    
    # Frontier Explorer Node  
    frontier_explorer_node = Node(
        package='turtlebot_sensor',
        executable='explorer',
        name='explorer',
        namespace=robot_namespace,
        parameters=[{
            'use_sim_time': use_sim_time,
            'map_topic': 'map',
            'odom_topic': 'odom',
            'goal_topic': 'goal_pose',
            'cmd_vel_topic': 'cmd_vel',
            'waypoints_topic': 'waypoints',
            'cube_pose_topic': 'cube_pose',
            'qos_depth': 10
        }],
        remappings=[
            # TF remappings for frontier explorer if it uses TF
            ('/tf', ['/T', robot_id, '/tf']),
            ('/tf_static', ['/T', robot_id, '/tf_static']),
        ],
        output='screen',
        condition=IfCondition(launch_explorer)
    )

    # Log info about what's being launched
    log_info = LogInfo(
        msg=['Launching TurtleBot nodes for robot T', robot_id, 
             ' - Planning: ', launch_planning, 
             ' - Explorer: ', launch_explorer]
    )

    return LaunchDescription([
        # Launch arguments
        robot_id_arg,
        launch_planning_arg,
        launch_explorer_arg,
        use_sim_time_arg,
        
        # Log message
        log_info,
        
        # Nodes
        planning_node,
        frontier_explorer_node,
    ])
#!/usr/bin/env python3

"""
Launch file for cube detection system
Includes all static transforms and the cube detection node
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments for easy configuration
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='26',
        description='Robot ID for namespaced topics (e.g., 26 for T26)'
    )
    
    cube_size_arg = DeclareLaunchArgument(
        'cube_physical_width_m',
        default_value='0.25',
        description='Physical width of the cube in meters'
    )
    
    # Get launch configuration values
    robot_id = LaunchConfiguration('robot_id')
    cube_size = LaunchConfiguration('cube_physical_width_m')
    launch_rviz = LaunchConfiguration('launch_rviz')

    return LaunchDescription([
        robot_id_arg,
        cube_size_arg,
        
        # Static transform publishers for camera chain
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_shell_transform',
            arguments=['0.0', '0.0', '0.12', '0.0', '0.0', '0.0', 'base_link', 'shell_link'],
            output='screen'
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='shell_to_camera_bracket_transform',
            arguments=['-0.118', '0.0', '0.05257', '0.0', '0.0', '0.0', 'shell_link', 'oakd_camera_bracket'],
            output='screen'
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher', 
            name='bracket_to_oakd_transform',
            arguments=['0.0584', '0.0', '0.09676', '0.0', '0.0', '0.0', 'oakd_camera_bracket', 'oakd_link'],
            output='screen'
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='oakd_to_rgb_frame_transform', 
            arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'oakd_link', 'oakd_rgb_camera_frame'],
            output='screen'
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='rgb_frame_to_optical_transform',
            arguments=['0.0', '0.0', '0.0', '0.5', '-0.5', '0.5', '-0.5', 'oakd_rgb_camera_frame', 'oakd_rgb_camera_optical_frame'],
            output='screen'
        ),
        
        # Cube detection node with parameters
        Node(
            package='turtlebot_sensor',
            executable='sync_node',
            name='cube_detection_node',
            parameters=[{
                'cube_physical_width_m': cube_size,
                'camera_optical_frame_id': 'oakd_rgb_camera_optical_frame',
                'target_map_frame_id': 'base_link',
                'publish_rviz_marker': True,
                'publish_debug_image': True,
                'use_cv_imshow_debug': True,
                
                # HSV parameters (adjust for your cube color)
                'hue_min': 15,
                'hue_max': 39,
                'sat_min': 90,
                'sat_max': 240,
                'val_min': 123,
                'val_max': 202,
                
                # Detection parameters
                'min_contour_area': 500,
                'max_contour_area': 30000,
                'epsilon_factor': 0.02,
                'confidence_threshold': 30.0,
                
                # Temporal smoothing
                'temporal_buffer_size': 4,
                'min_consistent_detections': 2,
            }],
            output='screen',
            emulate_tty=True
        ),
        
    ])
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    params = {
        "image_sub_topic": "/T7/oakd/rgb/image_raw/compressed",
        "hue_min": "16",  # Default values for gold color detection
        "hue_max": "40",
        "sat_min": "82",
        "sat_max": "255",
        "val_min": "100",
        "val_max": "255",
        # Stability parameters
        "confidence_threshold": "0.5",
        "smoothing_window": "5",
        "min_detection_area": "500",
    }
    
    # Declare all parameters in a loop
    declare_params = [
        DeclareLaunchArgument(
            name, default_value=value, description=f"Parameter: {name}"
        )
        for name, value in params.items()
    ]
    
    # Create LaunchConfiguration automatically
    param_configs = {name: LaunchConfiguration(name) for name in params.keys()}
    
    return LaunchDescription(
        declare_params
        + [
            Node(
                package="turtlebot_sensor",
                executable="sync_node",
                name="sync_node",
                output="screen",
                parameters=[param_configs],
            )
        ]
    )
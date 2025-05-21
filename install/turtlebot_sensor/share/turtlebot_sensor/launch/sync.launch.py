from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    params = {
        "namespace": "",
        "image_sub_topic": "/T7/oakd/rgb/image_raw/compressed",  # Updated the topic path to match your node
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
                package="turtlebot_sensor",  # Your package name
                executable="sync_node",       # Node executable name
                name="sync_node",            # Node name
                output="screen",
                parameters=[param_configs],
            )
        ]
    )
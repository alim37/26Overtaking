from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    return LaunchDescription([
        Node(
            package='autodrive_f1tenth',
            executable='pure_pursuit',
            name='pure_pursuit',
            emulate_tty=True,
            output='screen',
        ),
        Node(
            package='autodrive_f1tenth',
            executable='thesis',
            name='thesis',
            emulate_tty=True,
            output='screen',
        ),
    ])
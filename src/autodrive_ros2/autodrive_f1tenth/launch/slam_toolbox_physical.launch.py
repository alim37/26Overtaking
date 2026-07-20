#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    slam_params_file = LaunchConfiguration("slam_params_file")
    scan_topic = LaunchConfiguration("scan_topic")
    odom_frame = LaunchConfiguration("odom_frame")
    base_frame = LaunchConfiguration("base_frame")
    map_frame = LaunchConfiguration("map_frame")
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "slam_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("autodrive_f1tenth"), "config", "slam_toolbox_f1tenth.yaml"]
                ),
            ),
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("map_frame", default_value="map"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[
                    slam_params_file,
                    {
                        "odom_frame": odom_frame,
                        "base_frame": base_frame,
                        "map_frame": map_frame,
                        "scan_topic": scan_topic,
                        "use_sim_time": use_sim_time,
                    },
                ],
            ),
        ]
    )

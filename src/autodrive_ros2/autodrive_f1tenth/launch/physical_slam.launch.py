#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pose_topic = LaunchConfiguration("pose_topic")
    imu_topic = LaunchConfiguration("imu_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    output_path = LaunchConfiguration("output_path")
    accumulated_map_output_path = LaunchConfiguration("accumulated_map_output_path")
    current_scan_topic = LaunchConfiguration("current_scan_topic")
    accumulated_map_topic = LaunchConfiguration("accumulated_map_topic")
    world_frame = LaunchConfiguration("world_frame")
    use_tf_projection = LaunchConfiguration("use_tf_projection")
    require_tf_projection = LaunchConfiguration("require_tf_projection")
    stop_after_one_lap = LaunchConfiguration("stop_after_one_lap")
    lap_start_radius_m = LaunchConfiguration("lap_start_radius_m")
    lap_min_distance_m = LaunchConfiguration("lap_min_distance_m")

    package_share = FindPackageShare("autodrive_f1tenth")
    default_output = PathJoinSubstitution([package_share, "..", "..", "output", "slam_runs", "physical_empty_track_baseline.csv"])
    default_map_output = PathJoinSubstitution(
        [package_share, "..", "..", "output", "slam_runs", "physical_empty_track_baseline_map_points.csv"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("pose_topic", default_value="/ips"),
            DeclareLaunchArgument("imu_topic", default_value="/imu"),
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument("output_path", default_value=default_output),
            DeclareLaunchArgument("accumulated_map_output_path", default_value=default_map_output),
            DeclareLaunchArgument("current_scan_topic", default_value="/physical/slam/current_scan"),
            DeclareLaunchArgument("accumulated_map_topic", default_value="/physical/slam/accumulated_map"),
            DeclareLaunchArgument("world_frame", default_value="map"),
            DeclareLaunchArgument("use_tf_projection", default_value="true"),
            DeclareLaunchArgument("require_tf_projection", default_value="false"),
            DeclareLaunchArgument("stop_after_one_lap", default_value="false"),
            DeclareLaunchArgument("lap_start_radius_m", default_value="1.5"),
            DeclareLaunchArgument("lap_min_distance_m", default_value="35.0"),
            Node(
                package="autodrive_f1tenth",
                executable="slam",
                name="physical_slam_recorder",
                output="screen",
                parameters=[
                    {
                        "pose_topic": pose_topic,
                        "imu_topic": imu_topic,
                        "scan_topic": scan_topic,
                        "output_path": output_path,
                        "accumulated_map_output_path": accumulated_map_output_path,
                        "current_scan_topic": current_scan_topic,
                        "accumulated_map_topic": accumulated_map_topic,
                        "world_frame": world_frame,
                        "use_tf_projection": use_tf_projection,
                        "require_tf_projection": require_tf_projection,
                        "stop_after_one_lap": stop_after_one_lap,
                        "lap_start_radius_m": lap_start_radius_m,
                        "lap_min_distance_m": lap_min_distance_m,
                    }
                ],
            ),
        ]
    )

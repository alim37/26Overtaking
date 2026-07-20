#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("autodrive_f1tenth")
    default_baseline = PathJoinSubstitution(
        [package_share, "..", "..", "output", "slam_runs", "physical_empty_track_baseline.csv"]
    )
    default_wall_mask = PathJoinSubstitution(
        [package_share, "..", "..", "output", "slam_runs", "slam_toolbox_boundary_wall_mask.csv"]
    )
    default_confidence_log = PathJoinSubstitution(
        [package_share, "..", "..", "output", "confidence_runs", "physical_confidence_run.csv"]
    )

    pose_topic = LaunchConfiguration("pose_topic")
    imu_topic = LaunchConfiguration("imu_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    target_pose_topic = LaunchConfiguration("target_pose_topic")
    baseline_csv_path = LaunchConfiguration("baseline_csv_path")
    wall_mask_csv_path = LaunchConfiguration("wall_mask_csv_path")
    confidence_output_path = LaunchConfiguration("confidence_output_path")
    run_confidence_logger = LaunchConfiguration("run_confidence_logger")
    run_overtake_decision = LaunchConfiguration("run_overtake_decision")

    return LaunchDescription(
        [
            DeclareLaunchArgument("pose_topic", default_value="/ips"),
            DeclareLaunchArgument("imu_topic", default_value="/imu"),
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument("target_pose_topic", default_value="/target_pose"),
            DeclareLaunchArgument("baseline_csv_path", default_value=default_baseline),
            DeclareLaunchArgument("wall_mask_csv_path", default_value=default_wall_mask),
            DeclareLaunchArgument("confidence_output_path", default_value=default_confidence_log),
            DeclareLaunchArgument("run_confidence_logger", default_value="true"),
            DeclareLaunchArgument("run_overtake_decision", default_value="false"),
            Node(
                package="autodrive_f1tenth",
                executable="target_vehicle_tracker",
                name="physical_target_vehicle_tracker",
                output="screen",
                parameters=[
                    {
                        "pose_topic": pose_topic,
                        "imu_topic": imu_topic,
                        "scan_topic": scan_topic,
                        "target_pose_topic": target_pose_topic,
                        "baseline_csv_path": baseline_csv_path,
                        "wall_mask_csv_path": wall_mask_csv_path,
                    }
                ],
            ),
            Node(
                package="autodrive_f1tenth",
                executable="confidence_log",
                name="physical_confidence_logger",
                output="screen",
                condition=IfCondition(run_confidence_logger),
                parameters=[
                    {
                        "pose_topic": pose_topic,
                        "confidence_topic": "/autodrive/f1tenth_1/target_tracker/tracking_confidence",
                        "visible_topic": "/autodrive/f1tenth_1/target_tracker/target_visible",
                        "throttle_command_topic": "/autodrive/f1tenth_1/throttle_command",
                        "steering_command_topic": "/autodrive/f1tenth_1/steering_command",
                        "output_path": confidence_output_path,
                        "stop_after_one_lap": "false",
                    }
                ],
            ),
            Node(
                package="autodrive_f1tenth",
                executable="overtake_decision",
                name="physical_overtake_decision",
                output="screen",
                condition=IfCondition(run_overtake_decision),
                parameters=[
                    {
                        "pose_topic": pose_topic,
                        "imu_topic": imu_topic,
                        "wall_mask_csv_path": wall_mask_csv_path,
                    }
                ],
            ),
        ]
    )

#!/usr/bin/env python3

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_root = Path(__file__).resolve().parents[1]
    output_root = package_root / "output"
    slam_output = output_root / "slam_runs"
    tracker_mode = LaunchConfiguration("tracker_mode")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "tracker_mode",
                default_value="forza",
                description="Opponent tracker to run: 'forza' or 'baseline'",
            ),
            Node(
                package="autodrive_f1tenth",
                executable="pure_pursuit",
                name="pure_pursuit_f1tenth",
                output="screen",
                parameters=[{"stop_after_laps": 2}],
            ),
            Node(
                package="autodrive_f1tenth",
                executable="target_vehicle_tracker",
                name="target_vehicle_tracker",
                output="screen",
                condition=IfCondition(
                    PythonExpression(["'", tracker_mode, "' == 'baseline'"])
                ),
                parameters=[
                    {
                        "baseline_csv_path": str(slam_output / "empty_track_baseline.csv"),
                        "wall_mask_csv_path": str(slam_output / "slam_toolbox_boundary_wall_mask.csv"),
                        "debug_log_path": str(slam_output / "target_tracker_debug.csv"),
                    }
                ],
            ),
            Node(
                package="autodrive_f1tenth",
                executable="forza_opponent_tracker",
                name="forza_opponent_tracker",
                output="screen",
                condition=IfCondition(
                    PythonExpression(["'", tracker_mode, "' == 'forza'"])
                ),
                parameters=[
                    str(package_root / "config" / "forza_opponent_tracker.yaml"),
                    {
                        "wall_mask_csv_path": str(
                            slam_output / "slam_toolbox_boundary_wall_mask.csv"
                        ),
                    },
                ],
            ),
            Node(
                package="autodrive_f1tenth",
                executable="predictive_spliner_decision",
                name="predictive_spliner_decision",
                output="screen",
                parameters=[
                    {
                        "wall_mask_csv_path": str(slam_output / "slam_toolbox_boundary_wall_mask.csv"),
                        "figure_output_path": str(
                            output_root / "predictive_spliner" / "predictive_spliner_two_lap.png"
                        ),
                        "figure_after_laps": 2,
                    }
                ],
            ),
            Node(
                package="autodrive_f1tenth",
                executable="confidence_log",
                name="confidence_logger",
                output="screen",
                parameters=[
                    {
                        "output_path": str(output_root / "confidence_runs" / "confidence_run.csv"),
                    }
                ],
            ),
        ]
    )

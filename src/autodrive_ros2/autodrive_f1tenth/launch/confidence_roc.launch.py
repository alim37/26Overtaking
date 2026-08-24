#!/usr/bin/env python3

from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_root = Path(__file__).resolve().parents[1]
    output_root = package_root / "output"
    slam_output = output_root / "slam_runs"

    return LaunchDescription(
        [
            Node(
                package="autodrive_f1tenth",
                executable="pure_pursuit",
                name="pure_pursuit_f1tenth",
                output="screen",
                parameters=[
                    {
                        "stop_after_laps": 1,
                    }
                ],
            ),
            Node(
                package="autodrive_f1tenth",
                executable="forza_opponent_tracker",
                name="forza_opponent_tracker",
                output="screen",
                parameters=[
                    str(package_root / "config" / "forza_opponent_tracker.yaml"),
                    {"wall_mask_csv_path": str(slam_output / "slam_toolbox_boundary_wall_mask.csv")},
                ],
            ),
            Node(
                package="autodrive_f1tenth",
                executable="confidence_roc_decision",
                name="confidence_roc_decision",
                output="screen",
                parameters=[
                    {
                        "wall_mask_csv_path": str(slam_output / "slam_toolbox_boundary_wall_mask.csv"),
                        "figure_output_path": str(output_root / "confidence_roc" / "confidence_roc_one_lap.png"),
                        "prediction_horizon_sec": 6.0,
                        "confidence_threshold": 0.55,
                        "confidence_exit_threshold": 0.42,
                        "current_confidence_weight": 0.75,
                        "curvature_penalty_gain": 3.0,
                        "current_curvature_weight": 0.90,
                        "future_curvature_percentile": 75.0,
                        "curvature_smoothing_m": 1.50,
                        "speed_penalty_per_meter": 0.010,
                        "max_region_gap_m": 1.50,
                        "min_samples_per_bin": 1,
                        "wrap_prediction_horizon": False,
                        "figure_after_laps": 1,
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
                        "output_path": str(output_root / "confidence_runs" / "confidence_roc_run.csv"),
                        "stop_after_one_lap": True,
                    }
                ],
            ),
        ]
    )

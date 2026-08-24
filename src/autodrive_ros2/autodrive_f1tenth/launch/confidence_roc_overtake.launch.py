#!/usr/bin/env python3

from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


# -----------------------------------------------------------------------------
# Confidence-RoC overtake test knobs
# 1.00 = same nominal speed as f1tenth_2; 1.30 = f1tenth_1 is 30% faster.
BASE_TARGET_SPEED = 0.13
F1TENTH_1_SPEED_MULTIPLIER = 1.15
OVERTAKE_SPEED_DELAY_SEC = 0.25
# -----------------------------------------------------------------------------


def generate_launch_description() -> LaunchDescription:
    package_root = Path(__file__).resolve().parents[1]
    output_root = package_root / "output"
    slam_output = output_root / "slam_runs"
    ez_path = (
        output_root
        / "engagement_zones_dynamic"
        / "engagement_zone_dynamic_one_lap_path.csv"
    )

    return LaunchDescription(
        [
            # Car 1 owns its command topics through this node for the whole lap.
            # Internally it switches from the normal spline to the EZ path at c_start.
            Node(
                package="autodrive_f1tenth",
                executable="engagement_zone_controller",
                name="engagement_zone_controller",
                output="screen",
                parameters=[
                    {
                        "path_csv": str(ez_path),
                        "base_target_speed": BASE_TARGET_SPEED,
                        "speed_multiplier": F1TENTH_1_SPEED_MULTIPLIER,
                        "engagement_speed_delay_sec": OVERTAKE_SPEED_DELAY_SEC,
                        "stop_after_laps": 1,
                        "wait_for_startup_gate": True,
                    }
                ],
            ),
            # Car 2 remains the constant-speed evader on the normal spline.
            Node(
                package="autodrive_f1tenth",
                executable="pure_pursuit",
                name="pure_pursuit_f1tenth_2",
                output="screen",
                parameters=[
                    {
                        "active_car_ids": [2],
                        "target_speed": BASE_TARGET_SPEED,
                        "stop_after_laps": 1,
                        "startup_gate_required_car_ids": "1,2",
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
                    {
                        "wall_mask_csv_path": str(
                            slam_output / "slam_toolbox_boundary_wall_mask.csv"
                        )
                    },
                ],
            ),
            Node(
                package="autodrive_f1tenth",
                executable="confidence_roc_decision",
                name="confidence_roc_decision",
                output="screen",
                parameters=[
                    {
                        "wall_mask_csv_path": str(
                            slam_output / "slam_toolbox_boundary_wall_mask.csv"
                        ),
                        "figure_output_path": str(
                            output_root
                            / "confidence_roc"
                            / "confidence_roc_ez_test_one_lap.png"
                        ),
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
                        "output_path": str(
                            output_root
                            / "confidence_runs"
                            / "confidence_roc_ez_test.csv"
                        ),
                        "stop_after_one_lap": True,
                    }
                ],
            ),
        ]
    )

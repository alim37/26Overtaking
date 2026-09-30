#!/usr/bin/env python3

from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


# Standalone DynSSM + NMPC test knobs.
NOMINAL_VELOCITY_MPS = 3.25
NMPC_HORIZON = 8
OVERTAKE_SPEED_MULTIPLIER = 1.30
OVERTAKE_SPEED_DELAY_SEC = 0.25
PWM_TO_SIM_SCALE = 0.27
PURE_PURSUIT_THROTTLE_LIMIT = 0.13


def generate_launch_description():
    package_root = Path(__file__).resolve().parents[1]
    workspace_src = package_root.parents[1]
    model_dir = workspace_src / "DynSSM/output/DynSSM_ORCA/ssm_gru_tune_01"
    overtake_path = (
        package_root
        / "output/engagement_zones_dynamic_frenet"
        / "engagement_zone_dynamic_frenet_one_lap_path.csv"
    )
    return LaunchDescription(
        [
            Node(
                package="autodrive_f1tenth",
                executable="dynssm_nmpc_controller",
                name="dynssm_nmpc_controller",
                output="screen",
                parameters=[{
                    "path_csv": str(overtake_path),
                    "car_id": 1,
                    "peer_car_id": 2,
                    "enable_overtake_path": True,
                    "wait_for_peer_ips": True,
                    "nominal_velocity_mps": NOMINAL_VELOCITY_MPS,
                    "nmpc_horizon": NMPC_HORIZON,
                    "engagement_speed_multiplier": OVERTAKE_SPEED_MULTIPLIER,
                    "engagement_speed_delay_sec": OVERTAKE_SPEED_DELAY_SEC,
                    "pwm_to_sim_scale": PWM_TO_SIM_SCALE,
                    "max_sim_throttle": PURE_PURSUIT_THROTTLE_LIMIT,
                    "use_dynssm_adaptation": True,
                    "require_dynssm_active": True,
                    "dynssm_checkpoint": str(model_dir / "best_model_val_rmse.pth"),
                    "dynssm_config": str(model_dir / "config.json"),
                    "dynssm_scaler": str(model_dir / "scaler.pkl"),
                    "wait_for_startup_gate": False,
                    "stop_after_laps": 1,
                }],
            ),
            Node(
                package="autodrive_f1tenth",
                executable="dynssm_nmpc_controller",
                name="dynssm_nmpc_controller_f1tenth_2",
                output="screen",
                parameters=[{
                    "path_csv": str(overtake_path),
                    "car_id": 2,
                    "peer_car_id": 1,
                    "enable_overtake_path": False,
                    "wait_for_peer_ips": True,
                    "nominal_velocity_mps": NOMINAL_VELOCITY_MPS,
                    "nmpc_horizon": NMPC_HORIZON,
                    "engagement_speed_multiplier": 1.0,
                    "pwm_to_sim_scale": PWM_TO_SIM_SCALE,
                    "max_sim_throttle": PURE_PURSUIT_THROTTLE_LIMIT,
                    "use_dynssm_adaptation": True,
                    "require_dynssm_active": True,
                    "dynssm_checkpoint": str(model_dir / "best_model_val_rmse.pth"),
                    "dynssm_config": str(model_dir / "config.json"),
                    "dynssm_scaler": str(model_dir / "scaler.pkl"),
                    "wait_for_startup_gate": False,
                    "stop_after_laps": 1,
                }],
            ),
        ]
    )

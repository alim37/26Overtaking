#!/usr/bin/env python3
"""Launch the two-lap CAPE controller/model overtake benchmark."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Defaults can be edited here or overridden on the ros2 launch command line.
CONTROLLER = "nmpc"  # pure_pursuit, nmpc, adaptive, robust
MODEL = "epm"  # epm, bicycle
NOMINAL_VELOCITY_MPS = 3.25
OVERTAKE_SPEED_MULTIPLIER = 1.30
OVERTAKE_SPEED_DELAY_SEC = 0.25
NMPC_HORIZON = 12


def generate_launch_description():
    package_root = Path(__file__).resolve().parents[1]
    output_root = package_root / "output"
    model_dir = (
        package_root.parents[1]
        / "DynSSM/output/DynSSM_ORCA/ssm_gru_tune_01"
    )
    path_csv = (
        output_root
        / "engagement_zones_dynamic_frenet"
        / "engagement_zone_dynamic_frenet_one_lap_path.csv"
    )
    return LaunchDescription([
        DeclareLaunchArgument("controller", default_value=CONTROLLER),
        DeclareLaunchArgument("model", default_value=MODEL),
        Node(
            package="autodrive_f1tenth",
            executable="0_cape_overtake_controller",
            name="cape_overtake_controller",
            output="screen",
            parameters=[{
                "controller": LaunchConfiguration("controller"),
                "model": LaunchConfiguration("model"),
                "path_csv": str(path_csv),
                "car_id": 1,
                "peer_car_id": 2,
                "enable_overtake_path": True,
                "overtake_start_lap": 1,
                "stop_after_laps": 2,
                "wait_for_peer_ips": True,
                "wait_for_startup_gate": False,
                "nominal_velocity_mps": NOMINAL_VELOCITY_MPS,
                "nmpc_horizon": NMPC_HORIZON,
                "engagement_speed_multiplier": OVERTAKE_SPEED_MULTIPLIER,
                "engagement_speed_delay_sec": OVERTAKE_SPEED_DELAY_SEC,
                "max_sim_throttle": 0.13,
                "state_correction_gain": 0.80,
                "use_dynssm_adaptation": True,
                "require_dynssm_active": True,
                "dynssm_checkpoint": str(model_dir / "best_model_val_rmse.pth"),
                "dynssm_config": str(model_dir / "config.json"),
                "dynssm_scaler": str(model_dir / "scaler.pkl"),
                "benchmark_output_dir": str(output_root / "cape_overtake"),
            }],
        ),
        Node(
            package="autodrive_f1tenth",
            executable="0_cape_overtake_controller",
            name="cape_target_controller",
            output="screen",
            parameters=[{
                "controller": LaunchConfiguration("controller"),
                "model": LaunchConfiguration("model"),
                "path_csv": str(path_csv),
                "car_id": 2,
                "peer_car_id": 1,
                "enable_overtake_path": False,
                "overtake_start_lap": 1,
                "stop_after_laps": 2,
                "wait_for_peer_ips": True,
                "wait_for_startup_gate": False,
                "nominal_velocity_mps": NOMINAL_VELOCITY_MPS,
                "nmpc_horizon": NMPC_HORIZON,
                "engagement_speed_multiplier": 1.0,
                "max_sim_throttle": 0.13,
                "state_correction_gain": 0.80,
                "use_dynssm_adaptation": True,
                "require_dynssm_active": True,
                "dynssm_checkpoint": str(model_dir / "best_model_val_rmse.pth"),
                "dynssm_config": str(model_dir / "config.json"),
                "dynssm_scaler": str(model_dir / "scaler.pkl"),
                "benchmark_output_dir": str(output_root / "cape_overtake"),
            }],
        ),
    ])

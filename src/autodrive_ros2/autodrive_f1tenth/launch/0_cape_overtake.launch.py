#!/usr/bin/env python3
"""Launch the one-lap CAPE controller/model overtake benchmark."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# Defaults can be edited here or overridden on the ros2 launch command line.
CONTROLLER = "nmpc"  # pure_pursuit, nmpc, adaptive, robust
MODEL = "epm"  # epm, bicycle
NOMINAL_VELOCITY_MPS = 3.25
OVERTAKE_SPEED_MULTIPLIER = 1.30
OVERTAKE_SPEED_DELAY_SEC = 0.25
NMPC_HORIZON = 12
DYNSSM_EXPERIMENT = "unity_ethz_epm_v7_substep_nmpc"
MINIMUM_VEHICLE_SEPARATION_M = 0.65
THROTTLE_ACTUATOR_TAU_SEC = 0.14
STEERING_ACTUATOR_TAU_SEC = 0.13
STEERING_COMMAND_DELAY_STEPS = 1
STEERING_RATE_WEIGHT = 24.0
STEERING_ACCELERATION_WEIGHT = 12.0
STEERING_ACTUATOR_GAP_WEIGHT = 40.0
NMPC_HEADING_WEIGHT = 8.0
NMPC_YAW_RATE_WEIGHT = 0.75
STEERING_COMMAND_GAIN = 0.5234
MAX_STEERING_RATE_RAD_S = 1.30
MAX_STEERING_RAD = 0.45
REFERENCE_CENTERING_MAX_M = 0.15
TRACK_BASE_MARGIN_M = 0.03
ADAPTIVE_PARAMETER_TRUST_FRACTION = 0.10
PURE_PURSUIT_STEERING_GAIN = 4.5
PURE_PURSUIT_MAX_STEERING_RAD = 0.50


def generate_launch_description():
    package_root = Path(__file__).resolve().parents[1]
    output_root = package_root / "output"
    model_dir = (
        package_root.parents[1]
        / "DynSSM/output/DynSSM_ORCA"
        / DYNSSM_EXPERIMENT
    )
    path_csv = (
        output_root
        / "engagement_zones_dynamic_frenet"
        / "engagement_zone_dynamic_frenet_one_lap_path.csv"
    )
    bicycle_parameter_file = package_root / "config" / "bicycle_model_ethz.json"
    return LaunchDescription([
        DeclareLaunchArgument("controller", default_value=CONTROLLER),
        DeclareLaunchArgument("model", default_value=MODEL),
        DeclareLaunchArgument(
            "overtake_speed_multiplier",
            default_value=str(OVERTAKE_SPEED_MULTIPLIER),
        ),
        Node(
            package="autodrive_f1tenth",
            executable="0_cape_overtake_controller",
            name="cape_overtake_controller",
            output="screen",
            parameters=[{
                "controller": LaunchConfiguration("controller"),
                "model": LaunchConfiguration("model"),
                "bicycle_parameter_file": str(bicycle_parameter_file),
                "path_csv": str(path_csv),
                "car_id": 1,
                "peer_car_id": 2,
                "enable_overtake_path": True,
                "overtake_start_lap": 0,
                "stop_after_laps": 0,
                "wait_for_peer_ips": True,
                "wait_for_startup_gate": False,
                "nominal_velocity_mps": NOMINAL_VELOCITY_MPS,
                "nmpc_horizon": NMPC_HORIZON,
                "engagement_speed_multiplier": ParameterValue(
                    LaunchConfiguration("overtake_speed_multiplier"),
                    value_type=float,
                ),
                "experiment_overtake_speed_multiplier": ParameterValue(
                    LaunchConfiguration("overtake_speed_multiplier"),
                    value_type=float,
                ),
                "engagement_speed_delay_sec": OVERTAKE_SPEED_DELAY_SEC,
                "max_sim_throttle": 0.13,
                "minimum_vehicle_separation_m": MINIMUM_VEHICLE_SEPARATION_M,
                "throttle_actuator_tau_sec": THROTTLE_ACTUATOR_TAU_SEC,
                "steering_actuator_tau_sec": STEERING_ACTUATOR_TAU_SEC,
                "steering_command_delay_steps": STEERING_COMMAND_DELAY_STEPS,
                "steering_rate_weight": STEERING_RATE_WEIGHT,
                "steering_acceleration_weight": STEERING_ACCELERATION_WEIGHT,
                "steering_actuator_gap_weight": STEERING_ACTUATOR_GAP_WEIGHT,
                "nmpc_heading_weight": NMPC_HEADING_WEIGHT,
                "nmpc_yaw_rate_weight": NMPC_YAW_RATE_WEIGHT,
                "steering_command_gain": STEERING_COMMAND_GAIN,
                "max_steering_rate_rad_s": MAX_STEERING_RATE_RAD_S,
                "max_steering_rad": MAX_STEERING_RAD,
                "reference_centering_max_m": REFERENCE_CENTERING_MAX_M,
                "robust_nominal_margin_m": TRACK_BASE_MARGIN_M,
                "state_correction_gain": 0.80,
                "adaptive_parameter_trust_fraction": ADAPTIVE_PARAMETER_TRUST_FRACTION,
                "pure_pursuit_steering_gain": PURE_PURSUIT_STEERING_GAIN,
                "pure_pursuit_max_steering_rad": PURE_PURSUIT_MAX_STEERING_RAD,
                "use_dynssm_adaptation": True,
                "use_fixed_longitudinal_control_parameters": False,
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
                "bicycle_parameter_file": str(bicycle_parameter_file),
                "path_csv": str(path_csv),
                "car_id": 2,
                "peer_car_id": 1,
                "enable_overtake_path": False,
                "overtake_start_lap": 0,
                "stop_after_laps": 0,
                "wait_for_peer_ips": True,
                "wait_for_startup_gate": False,
                "nominal_velocity_mps": NOMINAL_VELOCITY_MPS,
                "nmpc_horizon": NMPC_HORIZON,
                "engagement_speed_multiplier": 1.0,
                "experiment_overtake_speed_multiplier": ParameterValue(
                    LaunchConfiguration("overtake_speed_multiplier"),
                    value_type=float,
                ),
                "max_sim_throttle": 0.13,
                "minimum_vehicle_separation_m": MINIMUM_VEHICLE_SEPARATION_M,
                "throttle_actuator_tau_sec": THROTTLE_ACTUATOR_TAU_SEC,
                "steering_actuator_tau_sec": STEERING_ACTUATOR_TAU_SEC,
                "steering_command_delay_steps": STEERING_COMMAND_DELAY_STEPS,
                "steering_rate_weight": STEERING_RATE_WEIGHT,
                "steering_acceleration_weight": STEERING_ACCELERATION_WEIGHT,
                "steering_actuator_gap_weight": STEERING_ACTUATOR_GAP_WEIGHT,
                "nmpc_heading_weight": NMPC_HEADING_WEIGHT,
                "nmpc_yaw_rate_weight": NMPC_YAW_RATE_WEIGHT,
                "steering_command_gain": STEERING_COMMAND_GAIN,
                "max_steering_rate_rad_s": MAX_STEERING_RATE_RAD_S,
                "max_steering_rad": MAX_STEERING_RAD,
                "reference_centering_max_m": REFERENCE_CENTERING_MAX_M,
                "robust_nominal_margin_m": TRACK_BASE_MARGIN_M,
                "state_correction_gain": 0.80,
                "adaptive_parameter_trust_fraction": ADAPTIVE_PARAMETER_TRUST_FRACTION,
                "pure_pursuit_steering_gain": PURE_PURSUIT_STEERING_GAIN,
                "pure_pursuit_max_steering_rad": PURE_PURSUIT_MAX_STEERING_RAD,
                "use_dynssm_adaptation": True,
                "use_fixed_longitudinal_control_parameters": False,
                "require_dynssm_active": True,
                "dynssm_checkpoint": str(model_dir / "best_model_val_rmse.pth"),
                "dynssm_config": str(model_dir / "config.json"),
                "dynssm_scaler": str(model_dir / "scaler.pkl"),
                "benchmark_output_dir": str(output_root / "cape_overtake"),
            }],
        ),
    ])

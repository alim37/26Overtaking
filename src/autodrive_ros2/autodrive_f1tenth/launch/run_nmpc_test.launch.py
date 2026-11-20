#!/usr/bin/env python3
"""Run one IEEE AIxVR nominal-NMPC model-comparison lap."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Paper defaults. Command-line launch arguments override these values.
MODEL = "epm"  # epm, dpm, ddm, or single_track
TRACK = "ethz"
TARGET_SPEED_MPS = 2.50
NMPC_SAFETY_MIN_SPEED_MPS = 1.80
NMPC_HORIZON = 12
CONTROL_PERIOD_SEC = 0.10
DYNSSM_EXPERIMENT = "unity_ethz_epm_v7_substep_nmpc"
DDM_OUTPUT_ROOT = "output_retrained_20261117"
DPM_OUTPUT_ROOT = "output_retrained_h10_20261120"
# Shared AIxVR comparison tuning; all model variants receive identical settings.
MINIMUM_VEHICLE_SEPARATION_M = 0.65
THROTTLE_ACTUATOR_TAU_SEC = 0.06
THROTTLE_COMMAND_DELAY_STEPS = 3
STEERING_ACTUATOR_TAU_SEC = 0.055
STEERING_COMMAND_DELAY_STEPS = 3
THROTTLE_RATE_WEIGHT = 8.0
THROTTLE_REFERENCE_WEIGHT = 40.0
THROTTLE_ACCELERATION_WEIGHT = 30.0
# Identified on training laps: v[k+1] = 0.34029 v[k] + 4.42983 u[k].
LONGITUDINAL_SPEED_GAIN_MPS_PER_MODEL_PWM = 6.7147846483
LONGITUDINAL_SPEED_TAU_SEC = 0.0843739947
MAX_THROTTLE_RATE_MODEL_S = 1.20
STEERING_RATE_WEIGHT = 24.0
STEERING_ACCELERATION_WEIGHT = 12.0
STEERING_ACTUATOR_GAP_WEIGHT = 40.0
NMPC_HEADING_WEIGHT = 8.0
NMPC_SPEED_WEIGHT = 12.0
NMPC_MINIMUM_SPEED_WEIGHT = 250.0
NMPC_YAW_RATE_WEIGHT = 0.75
STEERING_COMMAND_GAIN = 0.5234
MAX_STEERING_RATE_RAD_S = 1.30
# Keep NMPC inside the steering-command support of the eight-lap dataset.
MAX_STEERING_RAD = 0.46
REFERENCE_CENTERING_MAX_M = 0.25
REFERENCE_MINIMUM_WALL_CLEARANCE_M = 0.40
TRACK_BASE_MARGIN_M = 0.03
TRACK_BOUNDARY_MARGIN_M = 0.03
TRACK_SLACK_LINEAR_WEIGHT = 1.0e5
VEHICLE_LENGTH_M = 0.55
VEHICLE_WIDTH_M = 0.30
PURE_PURSUIT_STEERING_GAIN = 4.5
PURE_PURSUIT_MAX_STEERING_RAD = 0.50
# The dynamics and combined profiles expect this separate Unity build. These
# values are metadata only; the controller's internal model remains unchanged.
EXTERNAL_DYNAMICS_CONFIGURATION_ID = "mass4p164_front_tire0p8"
EXTERNAL_VEHICLE_MASS_KG = 4.164
EXTERNAL_FRONT_TIRE_STIFFNESS = 0.8
EXTERNAL_REAR_TIRE_STIFFNESS = 1.0


def _launch_test(context):
    model = LaunchConfiguration("model").perform(context).strip().lower()
    track = LaunchConfiguration("track").perform(context).strip().lower()
    speed = float(LaunchConfiguration("target_speed_mps").perform(context))
    car_id = int(LaunchConfiguration("car_id").perform(context))
    uncertainty_experiment = LaunchConfiguration(
        "uncertainty_experiment"
    ).perform(context).strip().lower()
    uncertainty_severity = float(
        LaunchConfiguration("uncertainty_severity").perform(context)
    )
    uncertainty_seed = int(
        LaunchConfiguration("uncertainty_seed").perform(context)
    )

    if model not in {"epm", "single_track", "dpm", "ddm"}:
        raise RuntimeError("model must be epm, dpm, ddm, or single_track")
    valid_experiments = {
        "off", "nominal", "perception", "actuation", "dynamics", "combined"
    }
    if uncertainty_experiment not in valid_experiments:
        raise RuntimeError(
            "uncertainty_experiment must be off, nominal, perception, "
            "actuation, dynamics, or combined"
        )
    if uncertainty_severity < 0.0:
        raise RuntimeError("uncertainty_severity must be nonnegative")

    package_root = Path(__file__).resolve().parents[1]
    output_root = package_root / "output" / "aixvr_nmpc"
    dynssm_root = package_root.parents[1] / "DynSSM"
    if model in {"ddm", "dpm"}:
        learned_output_root = (
            DPM_OUTPUT_ROOT if model == "dpm" else DDM_OUTPUT_ROOT
        )
        model_dir = (
            dynssm_root
            / learned_output_root
            / f"AIxVR_{model.upper()}"
            / "unity_ethz_8lap"
        )
        checkpoint = model_dir / "best_model.pth"
        normalization = model_dir / "normalization.npz"
    else:
        model_dir = dynssm_root / "output" / "DynSSM_ORCA" / DYNSSM_EXPERIMENT
        checkpoint = model_dir / "best_model_val_rmse.pth"
        normalization = model_dir / "scaler.pkl"
    path_csv = (
        package_root
        / "output"
        / "engagement_zones_dynamic_frenet"
        / "engagement_zone_dynamic_frenet_one_lap_path.csv"
    )
    bicycle_file = package_root / "config" / "bicycle_model_ethz.json"
    runtime_model = "bicycle" if model == "single_track" else model
    learned_model = model in {"epm", "ddm", "dpm"}

    use_bridge = uncertainty_experiment != "off"
    severity = uncertainty_severity
    perception_on = uncertainty_experiment in {"perception", "combined"}
    actuation_on = uncertainty_experiment in {"actuation", "combined"}
    external_dynamics_on = uncertainty_experiment in {"dynamics", "combined"}

    # These follow the CAPE uncertainty structure. Severity 1.0 corresponds
    # to its base 1% sensing and command-noise scale.
    measurement_noise = 0.01 * severity if perception_on else 0.0
    position_noise_std = 0.01 * severity if perception_on else 0.0
    position_bias_walk = 0.002 * severity if perception_on else 0.0
    yaw_rate_bias_walk = 0.0005 * severity if perception_on else 0.0
    command_noise = 0.01 * severity if actuation_on else 0.0
    command_delay = CONTROL_PERIOD_SEC * severity if actuation_on else 0.0
    # Dynamics mismatch is applied directly in the dedicated Unity build.
    # Keep the ROS command path neutral to avoid perturbing the plant twice.
    throttle_effectiveness = 1.0
    steering_effectiveness = 1.0
    response_tau = 0.0

    relay = f"/autodrive_uncertainty/f1tenth_{car_id}"
    base = f"/autodrive/f1tenth_{car_id}"
    remappings = []
    if use_bridge:
        remappings = [
            (f"{base}/ips", f"{relay}/ips"),
            (f"{base}/odom", f"{relay}/odom"),
            (f"{base}/imu", f"{relay}/imu"),
            (f"{base}/throttle_command", f"{relay}/throttle_command_nominal"),
            (f"{base}/steering_command", f"{relay}/steering_command_nominal"),
        ]

    severity_label = f"{severity:.2f}".replace(".", "p")
    experiment_stem = (
        f"{uncertainty_experiment}_s{severity_label}_seed{uncertainty_seed}_"
        f"nmpc_{model}_{track}"
        if use_bridge else f"nominal_nmpc_{model}_{track}"
    )

    actions = [Node(
        package="autodrive_f1tenth",
        executable="0_cape_overtake_controller",
        name=f"aixvr_nmpc_{model}",
        output="screen",
        parameters=[{
            "controller": "nmpc",
            "model": runtime_model,
            "track_name": track,
            "car_id": car_id,
            "peer_car_id": 2 if car_id == 1 else 1,
            "path_csv": str(path_csv),
            "bicycle_parameter_file": str(bicycle_file),
            "enable_overtake_path": False,
            "single_lap_benchmark": True,
            "stop_after_laps": 1,
            "wait_for_peer_ips": False,
            "wait_for_startup_gate": False,
            "nominal_velocity_mps": speed,
            "nmpc_minimum_speed_mps": NMPC_SAFETY_MIN_SPEED_MPS,
            "nmpc_minimum_speed_weight": NMPC_MINIMUM_SPEED_WEIGHT,
            "engagement_speed_multiplier": 1.0,
            "experiment_overtake_speed_multiplier": 1.0,
            "control_period": CONTROL_PERIOD_SEC,
            "nmpc_horizon": NMPC_HORIZON,
            "max_sim_throttle": 0.13,
            "minimum_vehicle_separation_m": MINIMUM_VEHICLE_SEPARATION_M,
            "throttle_actuator_tau_sec": THROTTLE_ACTUATOR_TAU_SEC,
            "throttle_command_delay_steps": THROTTLE_COMMAND_DELAY_STEPS,
            "steering_actuator_tau_sec": STEERING_ACTUATOR_TAU_SEC,
            "steering_command_delay_steps": STEERING_COMMAND_DELAY_STEPS,
            "throttle_rate_weight": THROTTLE_RATE_WEIGHT,
            "throttle_reference_weight": THROTTLE_REFERENCE_WEIGHT,
            "throttle_acceleration_weight": THROTTLE_ACCELERATION_WEIGHT,
            "longitudinal_speed_gain_mps_per_model_pwm": (
                LONGITUDINAL_SPEED_GAIN_MPS_PER_MODEL_PWM
            ),
            "longitudinal_speed_tau_sec": LONGITUDINAL_SPEED_TAU_SEC,
            "max_throttle_rate_model_s": MAX_THROTTLE_RATE_MODEL_S,
            "steering_rate_weight": STEERING_RATE_WEIGHT,
            "steering_acceleration_weight": STEERING_ACCELERATION_WEIGHT,
            "steering_actuator_gap_weight": STEERING_ACTUATOR_GAP_WEIGHT,
            "nmpc_heading_weight": NMPC_HEADING_WEIGHT,
            "nmpc_speed_weight": NMPC_SPEED_WEIGHT,
            "nmpc_yaw_rate_weight": NMPC_YAW_RATE_WEIGHT,
            "steering_command_gain": STEERING_COMMAND_GAIN,
            "max_steering_rate_rad_s": MAX_STEERING_RATE_RAD_S,
            "max_steering_rad": MAX_STEERING_RAD,
            "reference_centering_max_m": REFERENCE_CENTERING_MAX_M,
            "reference_minimum_wall_clearance_m": (
                REFERENCE_MINIMUM_WALL_CLEARANCE_M
            ),
            "robust_nominal_margin_m": TRACK_BASE_MARGIN_M,
            "enforce_track_boundaries": True,
            "track_boundary_margin_m": TRACK_BOUNDARY_MARGIN_M,
            "track_slack_linear_weight": TRACK_SLACK_LINEAR_WEIGHT,
            "vehicle_length_m": VEHICLE_LENGTH_M,
            "vehicle_width_m": VEHICLE_WIDTH_M,
            "state_correction_gain": 0.80,
            "pure_pursuit_steering_gain": PURE_PURSUIT_STEERING_GAIN,
            "pure_pursuit_max_steering_rad": PURE_PURSUIT_MAX_STEERING_RAD,
            "use_dynssm_adaptation": learned_model,
            "use_fixed_longitudinal_control_parameters": False,
            "require_dynssm_active": learned_model,
            "dynssm_checkpoint": str(checkpoint),
            "dynssm_config": str(model_dir / "config.json"),
            "dynssm_scaler": str(normalization),
            "benchmark_output_dir": str(output_root),
            "benchmark_file_stem": experiment_stem,
            "benchmark_model_label": model,
        }],
        remappings=remappings,
    )]
    if use_bridge:
        actions.insert(0, Node(
            package="autodrive_f1tenth",
            executable="aixvr_uncertainty_bridge",
            name=f"aixvr_uncertainty_{model}_{car_id}",
            output="screen",
            parameters=[{
                "car_id": car_id,
                "experiment": uncertainty_experiment,
                "severity": severity,
                "seed": uncertainty_seed,
                "control_period_sec": CONTROL_PERIOD_SEC,
                "measurement_noise_fraction": measurement_noise,
                "position_noise_std_m": position_noise_std,
                "position_bias_walk_m_sqrt_s": position_bias_walk,
                "yaw_rate_bias_walk_rad_s_sqrt_s": yaw_rate_bias_walk,
                "throttle_delay_sec": command_delay,
                "steering_delay_sec": command_delay,
                "actuation_noise_fraction": command_noise,
                "throttle_response_tau_sec": response_tau,
                "steering_response_tau_sec": response_tau,
                "throttle_effectiveness": throttle_effectiveness,
                "steering_effectiveness": steering_effectiveness,
                "external_dynamics_active": external_dynamics_on,
                "external_dynamics_configuration_id": (
                    EXTERNAL_DYNAMICS_CONFIGURATION_ID
                    if external_dynamics_on else "nominal_unity"
                ),
                "external_vehicle_mass_kg": (
                    EXTERNAL_VEHICLE_MASS_KG if external_dynamics_on else 0.0
                ),
                "external_front_tire_stiffness": (
                    EXTERNAL_FRONT_TIRE_STIFFNESS if external_dynamics_on else 0.0
                ),
                "external_rear_tire_stiffness": (
                    EXTERNAL_REAR_TIRE_STIFFNESS if external_dynamics_on else 0.0
                ),
                "max_abs_throttle": 0.13,
                "max_abs_steering_rad": MAX_STEERING_RAD,
                "vehicle_width_m": VEHICLE_WIDTH_M,
                "boundary_margin_m": TRACK_BOUNDARY_MARGIN_M,
                "wall_mask_csv": str(
                    package_root
                    / "output/slam_runs/slam_toolbox_boundary_wall_mask.csv"
                ),
                "output_dir": str(output_root / "uncertainty"),
                "output_stem": experiment_stem,
                "clean_pure_pursuit_warmup": True,
            }],
        ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "model",
            default_value=MODEL,
            description="Internal model: epm, dpm, ddm, or single_track.",
        ),
        DeclareLaunchArgument("track", default_value=TRACK),
        DeclareLaunchArgument(
            "target_speed_mps", default_value=str(TARGET_SPEED_MPS)
        ),
        DeclareLaunchArgument("car_id", default_value="1"),
        DeclareLaunchArgument(
            "uncertainty_experiment",
            default_value="off",
            description=(
                "off, nominal, perception, actuation, dynamics, or combined"
            ),
        ),
        DeclareLaunchArgument(
            "uncertainty_severity",
            default_value="0.0",
            description="Nonnegative CAPE-relative perturbation severity.",
        ),
        DeclareLaunchArgument("uncertainty_seed", default_value="47"),
        OpaqueFunction(function=_launch_test),
    ])

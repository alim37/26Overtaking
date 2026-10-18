#!/usr/bin/env python3

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, LogInfo, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# One profile is selected per lap. Lap 1 is an unexcited baseline; later laps
# vary throttle setpoint, steering gain, and reproducible bounded excitation.
TARGET_THROTTLE_COMMANDS = [0.10, 0.115, 0.13, 0.11, 0.125, 0.105, 0.135, 0.12]
STEERING_GAINS = [1.00, 0.99, 1.01, 1.00, 0.98, 1.02, 0.99, 1.01]
STEERING_EXCITATION_RAD = [0.0, 0.008, 0.010, 0.012, 0.008, 0.014, 0.010, 0.012]
STEERING_EXCITATION_HZ = [0.23, 0.31, 0.41, 0.29, 0.37, 0.19, 0.43, 0.27]
THROTTLE_EXCITATION = [0.0, 0.004, 0.006, 0.005, 0.007, 0.005, 0.006, 0.008]
THROTTLE_EXCITATION_HZ = [0.13, 0.17, 0.19, 0.23, 0.11, 0.29, 0.21, 0.15]
VALIDATION_LAPS = [3, 6]
TEST_LAPS = [8]

NUM_LAPS = len(TARGET_THROTTLE_COMMANDS)
PURE_PURSUIT_BASE_THROTTLE = 0.13


def generate_launch_description() -> LaunchDescription:
    track = LaunchConfiguration("track")
    package_root = Path(__file__).resolve().parents[1]

    pure_pursuit = Node(
        package="autodrive_f1tenth",
        executable="pure_pursuit",
        name="epm_sysid_pure_pursuit",
        output="screen",
        parameters=[
            {
                "active_car_ids": [1],
                "track_name": track,
                "target_speed": PURE_PURSUIT_BASE_THROTTLE,
                "lookahead_distance": 2.5,
                "num_path_points": 800,
                "use_learned_model": False,
                "enable_logging": False,
                "stop_after_laps": NUM_LAPS,
                "lap_start_radius_m": 1.5,
                "lap_min_distance_m": 35.0,
                "startup_gate_required_car_ids": "1",
            }
        ],
        remappings=[
            (
                "/autodrive/f1tenth_1/steering_command",
                "/autodrive/f1tenth_1/sysid/steering_nominal",
            ),
            (
                "/autodrive/f1tenth_1/throttle_command",
                "/autodrive/f1tenth_1/sysid/throttle_nominal",
            ),
        ],
    )

    collector = Node(
        package="autodrive_f1tenth",
        executable="epm_sysid_collector",
        name="epm_sysid_collector",
        output="screen",
        parameters=[
            {
                "output_dir": str(package_root / "output" / "epm_sysid"),
                "track_name": track,
                "direction": "forward",
                "num_laps": NUM_LAPS,
                "nominal_base_throttle": PURE_PURSUIT_BASE_THROTTLE,
                "target_throttle_commands": TARGET_THROTTLE_COMMANDS,
                "steering_gains": STEERING_GAINS,
                "steering_excitation_rad": STEERING_EXCITATION_RAD,
                "steering_excitation_hz": STEERING_EXCITATION_HZ,
                "throttle_excitation": THROTTLE_EXCITATION,
                "throttle_excitation_hz": THROTTLE_EXCITATION_HZ,
                "validation_laps": VALIDATION_LAPS,
                "test_laps": TEST_LAPS,
                "max_abs_steering_rad": 0.54,
                "max_throttle": 0.16,
                "corner_protection_steering_rad": 0.15,
                "sync_tolerance_sec": 0.03,
            }
        ],
    )

    shutdown = RegisterEventHandler(
        OnProcessExit(
            target_action=collector,
            on_exit=[
                LogInfo(msg="EPM system-identification run complete."),
                EmitEvent(event=Shutdown(reason="EPM system-ID collection complete")),
            ],
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "track",
                default_value="ethz",
                choices=["ethz", "ethzmobil"],
                description="Reference waypoint set used by pure pursuit.",
            ),
            pure_pursuit,
            collector,
            shutdown,
        ]
    )

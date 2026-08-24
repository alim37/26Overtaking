#!/usr/bin/env python3

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, LogInfo, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Change this value between experiments. Pure pursuit still applies its existing
# reduced-speed scale automatically when the steering angle exceeds 10 degrees.
TARGET_SPEED = 0.13


def generate_launch_description() -> LaunchDescription:
    track_name = LaunchConfiguration("track")
    package_root = Path(__file__).resolve().parents[1]
    output_path = (
        package_root
        / "output"
        / "ethz_all_data"
        / "ethz_all_data_in_progress.csv"
    )

    pure_pursuit = Node(
        package="autodrive_f1tenth",
        executable="pure_pursuit",
        name="pure_pursuit_f1tenth",
        output="screen",
        parameters=[
            {
                "active_car_ids": [1],
                "track_name": track_name,
                "target_speed": TARGET_SPEED,
                "num_path_points": 800,
                "use_learned_model": False,
                "enable_logging": False,
                "stop_after_one_lap": True,
                "stop_after_laps": 1,
                "lap_start_radius_m": 1.5,
                "lap_min_distance_m": 35.0,
            }
        ],
    )

    data_logger = Node(
        package="autodrive_f1tenth",
        executable="ethz_all_data_logger",
        name="ethz_all_data_logger",
        output="screen",
        parameters=[
            {
                "output_path": str(output_path),
                "track_name": track_name,
                "sample_period_sec": 0.05,
                "lap_start_radius_m": 1.5,
                "lap_min_distance_m": 35.0,
            }
        ],
    )

    stop_after_log = RegisterEventHandler(
        OnProcessExit(
            target_action=data_logger,
            on_exit=[
                LogInfo(msg="One-lap ETHZ data collection finished; stopping the experiment stack."),
                EmitEvent(event=Shutdown(reason="ETHZ one-lap data collection complete")),
            ],
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "track",
                default_value="ethz",
                choices=["ethz", "ethzmobil"],
                description="Waypoint set used by pure pursuit.",
            ),
            pure_pursuit,
            data_logger,
            stop_after_log,
        ]
    )

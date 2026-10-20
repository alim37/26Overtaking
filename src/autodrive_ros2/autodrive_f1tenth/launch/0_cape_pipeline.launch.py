#!/usr/bin/env python3
"""Run a controlled CAPE controller/model/overtake-speed experiment."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package_share = Path(get_package_share_directory("autodrive_f1tenth"))
    benchmark_launch = package_share / "launch" / "0_cape_overtake.launch.py"

    controller = LaunchConfiguration("controller")
    model = LaunchConfiguration("model")
    speed_multiplier = LaunchConfiguration("overtake_speed_multiplier")

    return LaunchDescription([
        DeclareLaunchArgument(
            "controller",
            default_value="pure_pursuit",
            choices=["pure_pursuit", "nmpc", "robust", "adaptive"],
            description="Controller used by both cars.",
        ),
        DeclareLaunchArgument(
            "model",
            default_value="epm",
            choices=["epm", "bicycle"],
            description="Vehicle model used by both cars.",
        ),
        DeclareLaunchArgument(
            "overtake_speed_multiplier",
            default_value="1.00",
            description=(
                "Car-1 engagement speed multiplier; car 2 remains at 1.00. "
                "Examples: 1.00, 1.10, 1.30."
            ),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(benchmark_launch)),
            launch_arguments={
                "controller": controller,
                "model": model,
                "overtake_speed_multiplier": speed_multiplier,
            }.items(),
        ),
    ])

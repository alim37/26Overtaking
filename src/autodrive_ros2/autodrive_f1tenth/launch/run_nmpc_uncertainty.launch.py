#!/usr/bin/env python3
"""Run a seeded AIxVR NMPC uncertainty experiment."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    benchmark = Path(__file__).resolve().parent / "run_nmpc_test.launch.py"
    return LaunchDescription([
        DeclareLaunchArgument("model", default_value="epm"),
        DeclareLaunchArgument("track", default_value="ethz"),
        DeclareLaunchArgument("target_speed_mps", default_value="2.5"),
        DeclareLaunchArgument("car_id", default_value="1"),
        DeclareLaunchArgument(
            "experiment",
            default_value="perception",
            description=(
                "nominal, perception, actuation, dynamics, or combined. Dynamics "
                "expects the mass=4.164kg/front-tire-stiffness=0.8 Unity build; "
                "combined uses that build plus perception and actuation injection."
            ),
        ),
        DeclareLaunchArgument(
            "severity",
            default_value="1.0",
            description=(
                "CAPE-relative sensing/actuation severity. It does not scale the "
                "external Unity dynamics configuration."
            ),
        ),
        DeclareLaunchArgument("seed", default_value="47"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(benchmark)),
            launch_arguments={
                "model": LaunchConfiguration("model"),
                "track": LaunchConfiguration("track"),
                "target_speed_mps": LaunchConfiguration("target_speed_mps"),
                "car_id": LaunchConfiguration("car_id"),
                "uncertainty_experiment": LaunchConfiguration("experiment"),
                "uncertainty_severity": LaunchConfiguration("severity"),
                "uncertainty_seed": LaunchConfiguration("seed"),
            }.items(),
        ),
    ])

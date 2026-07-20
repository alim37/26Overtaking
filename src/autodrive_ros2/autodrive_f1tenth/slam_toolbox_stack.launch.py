#!/usr/bin/env python3

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_root = Path(__file__).resolve().parents[1]
    pure_pursuit_script = str(package_root / "autodrive_f1tenth" / "pure_pursuit.py")
    run_pure_pursuit = LaunchConfiguration("run_pure_pursuit")
    run_slam_recorder = LaunchConfiguration("run_slam_recorder")
    run_tf_logger = LaunchConfiguration("run_tf_logger")
    run_map_logger = LaunchConfiguration("run_map_logger")
    slam_params_file = LaunchConfiguration("slam_params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "run_pure_pursuit",
                default_value="true",
                description="Whether to also run the existing pure pursuit controller while mapping.",
            ),
            DeclareLaunchArgument(
                "slam_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("autodrive_f1tenth"), "config", "slam_toolbox_f1tenth.yaml"]
                ),
                description="slam_toolbox parameter file.",
            ),
            DeclareLaunchArgument(
                "run_slam_recorder",
                default_value="true",
                description="Whether to also record the old IPS+LiDAR baseline CSV with slam.py.",
            ),
            DeclareLaunchArgument(
                "run_tf_logger",
                default_value="true",
                description="Whether to log odom_1->f1tenth_1 and map->odom_1 during mapping.",
            ),
            DeclareLaunchArgument(
                "run_map_logger",
                default_value="true",
                description="Whether to save slam_toolbox all/outer/inner boundaries after lap 1.",
            ),
            ExecuteProcess(
                cmd=[
                    "python3",
                    pure_pursuit_script,
                    "--ros-args",
                    "-p",
                    "active_car_ids:=[1]",
                    "-p",
                    "stop_after_one_lap:=true",
                    "-p",
                    "lap_start_radius_m:=1.5",
                    "-p",
                    "lap_min_distance_m:=35.0",
                    "-r",
                    "__node:=pure_pursuit_f1tenth",
                ],
                output="screen",
                condition=IfCondition(run_pure_pursuit),
            ),
            Node(
                package="autodrive_f1tenth",
                executable="slam",
                name="slam_recorder",
                output="screen",
                condition=IfCondition(run_slam_recorder),
            ),
            Node(
                package="autodrive_f1tenth",
                executable="slam_tf_logger",
                name="slam_tf_logger",
                output="screen",
                condition=IfCondition(run_tf_logger),
            ),
            Node(
                package="autodrive_f1tenth",
                executable="slam_toolbox_map_logger",
                name="slam_toolbox_map_logger",
                output="screen",
                condition=IfCondition(run_map_logger),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([FindPackageShare("slam_toolbox"), "launch", "online_async_launch.py"])
                ),
                launch_arguments={
                    "use_sim_time": "false",
                    "slam_params_file": slam_params_file,
                }.items(),
            ),
        ]
    )

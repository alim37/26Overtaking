#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="autodrive_f1tenth",
                executable="pure_pursuit",
                name="pure_pursuit_f1tenth",
                output="screen",
            ),
            Node(
                package="autodrive_f1tenth",
                executable="slam",
                name="slam_recorder",
                output="screen",
            ),
        ]
    )

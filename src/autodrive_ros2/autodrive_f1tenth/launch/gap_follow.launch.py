#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="autodrive_f1tenth",
                executable="gap_follow",
                name="gap_follow_f1tenth",
                output="screen",
            ),
        ]
    )

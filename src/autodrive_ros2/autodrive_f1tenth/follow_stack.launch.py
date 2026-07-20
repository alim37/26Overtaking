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
                executable="target_vehicle_tracker",
                name="target_vehicle_tracker",
                output="screen",
            ),
            Node(
                package="autodrive_f1tenth",
                executable="confidence_log",
                name="confidence_logger",
                output="screen",
            ),
            Node(
                package="autodrive_f1tenth",
                executable="overtake_decision",
                name="overtake_decision",
                output="screen",
            ),
            # Node(
            #     package="autodrive_f1tenth",
            #     executable="pure_pursuit_target",
            #     name="pure_pursuit_target",
            #     output="screen",
            # ),
            # Node(
            #     package="autodrive_f1tenth",
            #     executable="safety",
            #     name="safety_override",
            #     output="screen",
            # ),
        ]
    )

# Copyright (c) 2023, Tinker Twins
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

################################################################################

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    use_v1_state_bundle_for_slam = LaunchConfiguration('use_v1_state_bundle_for_slam')
    publish_v1_map_to_odom_identity = LaunchConfiguration('publish_v1_map_to_odom_identity')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_v1_state_bundle_for_slam',
            default_value='false',
            description='Use the synchronized V1StateBundle event for vehicle 1 SLAM-focused ROS publishing.',
        ),
        DeclareLaunchArgument(
            'publish_v1_map_to_odom_identity',
            default_value='false',
            description='Optionally publish an identity map->odom_1 transform for bundle-mode RViz use without slam_toolbox.',
        ),
        Node(
            package='autodrive_f1tenth',
            executable='autodrive_incoming_bridge_2',
            name='autodrive_incoming_bridge_2',
            emulate_tty=True,
            output='screen',
            parameters=[
                {
                    'use_v1_state_bundle_for_slam': use_v1_state_bundle_for_slam,
                    'publish_v1_map_to_odom_identity': publish_v1_map_to_odom_identity,
                }
            ],
        ),
        Node(
            package='autodrive_f1tenth',
            executable='autodrive_outgoing_bridge_2',
            name='autodrive_outgoing_bridge_2',
            emulate_tty=True,
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            arguments=['-d', [FindPackageShare("autodrive_f1tenth"), '/rviz', '/simulator_2.rviz',]]
        ),
    ])

#!/usr/bin/env python3

from __future__ import annotations

import rclpy
from rclpy.executors import MultiThreadedExecutor

from autodrive_f1tenth.pure_pursuit import PurePursuitF1Tenth
from autodrive_f1tenth.pure_pursuit_target import PurePursuitTargetFollower
from autodrive_f1tenth.safety import SafetyOverride
from autodrive_f1tenth.target_vehicle_tracker_foreground import TargetVehicleTrackerForeground


def main(args=None) -> None:
    rclpy.init(args=args)
    pure_pursuit = PurePursuitF1Tenth()
    target_tracker = TargetVehicleTrackerForeground()
    target_follower = PurePursuitTargetFollower()
    safety_override = SafetyOverride()
    executor = MultiThreadedExecutor()
    executor.add_node(pure_pursuit)
    executor.add_node(target_tracker)
    executor.add_node(target_follower)
    executor.add_node(safety_override)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        safety_override.destroy_node()
        target_follower.destroy_node()
        target_tracker.destroy_node()
        pure_pursuit.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

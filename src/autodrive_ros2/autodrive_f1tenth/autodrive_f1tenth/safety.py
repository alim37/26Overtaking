#!/usr/bin/env python3

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32


class SafetyOverride(Node):
    def __init__(self) -> None:
        super().__init__("safety_override")

        self.declare_parameter("scan_topic", "/autodrive/f1tenth_1/lidar")
        self.declare_parameter("pose_topic", "/autodrive/f1tenth_1/ips")
        self.declare_parameter("steering_feedback_topic", "/autodrive/f1tenth_1/steering")
        self.declare_parameter("throttle_feedback_topic", "/autodrive/f1tenth_1/throttle")
        self.declare_parameter("steering_command_topic", "/autodrive/f1tenth_1/steering_command")
        self.declare_parameter("throttle_command_topic", "/autodrive/f1tenth_1/throttle_command")
        self.declare_parameter("safety_active_topic", "/autodrive/f1tenth_1/safety/active")
        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("min_speed_mps", 0.25)
        self.declare_parameter("base_trigger_distance_m", 0.38)
        self.declare_parameter("speed_distance_gain", 0.35)
        self.declare_parameter("release_distance_margin_m", 0.08)
        self.declare_parameter("frontal_half_angle_deg", 28.0)
        self.declare_parameter("side_window_min_deg", 8.0)
        self.declare_parameter("side_window_max_deg", 55.0)
        self.declare_parameter("danger_close_distance_m", 0.28)
        self.declare_parameter("correction_gain", 0.14)
        self.declare_parameter("max_correction", 0.12)
        self.declare_parameter("safe_throttle", 0.06)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.steering_feedback_topic = str(self.get_parameter("steering_feedback_topic").value)
        self.throttle_feedback_topic = str(self.get_parameter("throttle_feedback_topic").value)
        self.steering_command_topic = str(self.get_parameter("steering_command_topic").value)
        self.throttle_command_topic = str(self.get_parameter("throttle_command_topic").value)
        self.safety_active_topic = str(self.get_parameter("safety_active_topic").value)
        self.control_period = float(self.get_parameter("control_period").value)
        self.min_speed_mps = float(self.get_parameter("min_speed_mps").value)
        self.base_trigger_distance_m = float(self.get_parameter("base_trigger_distance_m").value)
        self.speed_distance_gain = float(self.get_parameter("speed_distance_gain").value)
        self.release_distance_margin_m = float(self.get_parameter("release_distance_margin_m").value)
        self.frontal_half_angle_deg = float(self.get_parameter("frontal_half_angle_deg").value)
        self.side_window_min_deg = float(self.get_parameter("side_window_min_deg").value)
        self.side_window_max_deg = float(self.get_parameter("side_window_max_deg").value)
        self.danger_close_distance_m = float(self.get_parameter("danger_close_distance_m").value)
        self.correction_gain = float(self.get_parameter("correction_gain").value)
        self.max_correction = float(self.get_parameter("max_correction").value)
        self.safe_throttle = float(self.get_parameter("safe_throttle").value)

        self.latest_scan: LaserScan | None = None
        self.current_speed_mps = 0.0
        self.last_pose: tuple[float, float] | None = None
        self.last_pose_time_sec: float | None = None
        self.current_steering_fb = 0.0
        self.current_throttle_fb = 0.0
        self.safety_active = False

        self.steer_pub = self.create_publisher(Float32, self.steering_command_topic, 10)
        self.throttle_pub = self.create_publisher(Float32, self.throttle_command_topic, 10)
        self.active_pub = self.create_publisher(Bool, self.safety_active_topic, 10)

        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.create_subscription(Point, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(Float32, self.steering_feedback_topic, self.steer_fb_cb, 10)
        self.create_subscription(Float32, self.throttle_feedback_topic, self.throttle_fb_cb, 10)

        self.create_timer(self.control_period, self.control_loop)
        self.get_logger().info(
            "Safety override ready. "
            f"scan={self.scan_topic}, active_topic={self.safety_active_topic}"
        )

    def scan_cb(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def pose_cb(self, msg: Point) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        current = (float(msg.x), float(msg.y))
        if self.last_pose is not None and self.last_pose_time_sec is not None:
            dt = max(now_sec - self.last_pose_time_sec, 1e-6)
            dx = current[0] - self.last_pose[0]
            dy = current[1] - self.last_pose[1]
            self.current_speed_mps = math.hypot(dx, dy) / dt
        self.last_pose = current
        self.last_pose_time_sec = now_sec

    def steer_fb_cb(self, msg: Float32) -> None:
        self.current_steering_fb = float(msg.data)

    def throttle_fb_cb(self, msg: Float32) -> None:
        self.current_throttle_fb = float(msg.data)

    def min_range_in_window(self, scan: LaserScan, min_angle_deg: float, max_angle_deg: float) -> float:
        angle = scan.angle_min
        best = math.inf
        for distance in scan.ranges:
            angle_deg = math.degrees(angle)
            if min_angle_deg <= angle_deg <= max_angle_deg:
                if math.isfinite(distance) and scan.range_min <= distance <= scan.range_max:
                    best = min(best, float(distance))
            angle += scan.angle_increment
        return best

    def compute_correction(self, scan: LaserScan) -> tuple[bool, float, float]:
        speed = self.current_speed_mps
        trigger_distance = self.base_trigger_distance_m + self.speed_distance_gain * max(0.0, speed - self.min_speed_mps)
        if self.safety_active:
            trigger_distance -= self.release_distance_margin_m
        if speed < self.min_speed_mps:
            return False, 0.0, trigger_distance

        front_min = self.min_range_in_window(scan, -self.frontal_half_angle_deg, self.frontal_half_angle_deg)
        left_min = self.min_range_in_window(scan, self.side_window_min_deg, self.side_window_max_deg)
        right_min = self.min_range_in_window(scan, -self.side_window_max_deg, -self.side_window_min_deg)

        if not math.isfinite(front_min):
            return False, 0.0, trigger_distance

        turning_left = self.current_steering_fb > 0.03
        turning_right = self.current_steering_fb < -0.03
        wall_bias = 0.0
        if math.isfinite(left_min) and math.isfinite(right_min):
            wall_bias = right_min - left_min
        elif math.isfinite(left_min):
            wall_bias = -left_min
        elif math.isfinite(right_min):
            wall_bias = right_min

        imminent = front_min < trigger_distance
        turn_risk = (turning_left and math.isfinite(left_min) and left_min < trigger_distance) or (
            turning_right and math.isfinite(right_min) and right_min < trigger_distance
        )
        centered_danger = front_min < self.danger_close_distance_m
        active = imminent and (turn_risk or centered_danger or abs(wall_bias) > 0.05)
        if not active:
            return False, 0.0, trigger_distance

        steer_direction = -1.0 if wall_bias < 0.0 else 1.0
        if abs(wall_bias) < 1e-3:
            steer_direction = -1.0 if self.current_steering_fb > 0.0 else 1.0
        severity = max(0.0, trigger_distance - front_min) / max(trigger_distance, 1e-6)
        correction = steer_direction * min(self.max_correction, self.correction_gain * severity)
        return True, correction, trigger_distance

    def control_loop(self) -> None:
        if self.latest_scan is None:
            self.active_pub.publish(Bool(data=False))
            self.safety_active = False
            return

        active, correction, _ = self.compute_correction(self.latest_scan)
        self.safety_active = active
        self.active_pub.publish(Bool(data=active))

        if not active:
            return

        steering_cmd = max(-1.0, min(1.0, self.current_steering_fb + correction))
        throttle_cmd = min(self.current_throttle_fb, self.safe_throttle)
        self.steer_pub.publish(Float32(data=float(steering_cmd)))
        self.throttle_pub.publish(Float32(data=float(throttle_cmd)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyOverride()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.active_pub.publish(Bool(data=False))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

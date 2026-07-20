#!/usr/bin/env python3

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


class PurePursuitTargetFollower(Node):
    def __init__(self) -> None:
        super().__init__("pure_pursuit_target")

        self.declare_parameter("target_point_topic", "/autodrive/f1tenth_1/target_tracker/target_point")
        self.declare_parameter("target_visible_topic", "/autodrive/f1tenth_1/target_tracker/target_visible")
        self.declare_parameter("tracking_confidence_topic", "/autodrive/f1tenth_1/target_tracker/tracking_confidence")
        self.declare_parameter("follow_active_topic", "/autodrive/f1tenth_1/target_tracker/follow_active")
        self.declare_parameter("safety_active_topic", "/autodrive/f1tenth_1/safety/active")
        self.declare_parameter("steering_command_topic", "/autodrive/f1tenth_1/steering_command")
        self.declare_parameter("throttle_command_topic", "/autodrive/f1tenth_1/throttle_command")
        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("wheelbase", 0.30)
        self.declare_parameter("lookahead_distance", 1.35)
        self.declare_parameter("target_speed", 0.135)
        self.declare_parameter("max_steer_deg", 90.0)
        self.declare_parameter("confidence_threshold", 0.80)
        self.declare_parameter("max_target_age_sec", 0.35)
        self.declare_parameter("min_follow_distance", 0.50)
        self.declare_parameter("max_follow_distance", 6.0)

        self.target_point_topic = str(self.get_parameter("target_point_topic").value)
        self.target_visible_topic = str(self.get_parameter("target_visible_topic").value)
        self.tracking_confidence_topic = str(self.get_parameter("tracking_confidence_topic").value)
        self.follow_active_topic = str(self.get_parameter("follow_active_topic").value)
        self.safety_active_topic = str(self.get_parameter("safety_active_topic").value)
        self.steering_command_topic = str(self.get_parameter("steering_command_topic").value)
        self.throttle_command_topic = str(self.get_parameter("throttle_command_topic").value)
        self.control_period = float(self.get_parameter("control_period").value)
        self.wheelbase = float(self.get_parameter("wheelbase").value)
        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        self.target_speed = float(self.get_parameter("target_speed").value)
        self.max_steer = math.radians(float(self.get_parameter("max_steer_deg").value))
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.max_target_age_sec = float(self.get_parameter("max_target_age_sec").value)
        self.min_follow_distance = float(self.get_parameter("min_follow_distance").value)
        self.max_follow_distance = float(self.get_parameter("max_follow_distance").value)

        self.latest_target: tuple[float, float] | None = None
        self.target_visible = False
        self.tracking_confidence = 0.0
        self.target_stamp_sec = 0.0
        self.follow_active = False
        self.safety_active = False

        self.steer_pub = self.create_publisher(Float32, self.steering_command_topic, 10)
        self.throttle_pub = self.create_publisher(Float32, self.throttle_command_topic, 10)
        self.follow_active_pub = self.create_publisher(Bool, self.follow_active_topic, 10)

        self.create_subscription(PointStamped, self.target_point_topic, self.target_point_cb, 10)
        self.create_subscription(Bool, self.target_visible_topic, self.target_visible_cb, 10)
        self.create_subscription(Float32, self.tracking_confidence_topic, self.tracking_confidence_cb, 10)
        self.create_subscription(Bool, self.safety_active_topic, self.safety_active_cb, 10)

        self.create_timer(self.control_period, self.control_loop)
        self.get_logger().info(
            "Pure pursuit target follower ready. "
            f"target={self.target_point_topic}, confidence={self.tracking_confidence_topic}, active={self.follow_active_topic}"
        )

    def target_point_cb(self, msg: PointStamped) -> None:
        self.latest_target = (float(msg.point.x), float(msg.point.y))
        self.target_stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def target_visible_cb(self, msg: Bool) -> None:
        self.target_visible = bool(msg.data)

    def tracking_confidence_cb(self, msg: Float32) -> None:
        self.tracking_confidence = float(msg.data)

    def safety_active_cb(self, msg: Bool) -> None:
        self.safety_active = bool(msg.data)

    def should_follow_target(self, now_sec: float) -> bool:
        if not self.target_visible or self.latest_target is None:
            return False
        if self.tracking_confidence < self.confidence_threshold:
            return False
        if now_sec - self.target_stamp_sec > self.max_target_age_sec:
            return False

        tx, ty = self.latest_target
        distance = math.hypot(tx, ty)
        if tx <= 0.0:
            return False
        if distance < self.min_follow_distance or distance > self.max_follow_distance:
            return False
        return True

    def control_loop(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.safety_active:
            self.follow_active = False
            self.follow_active_pub.publish(Bool(data=False))
            return
        active = self.should_follow_target(now_sec)
        self.follow_active = active
        self.follow_active_pub.publish(Bool(data=active))

        if not active or self.latest_target is None:
            return

        tx, ty = self.latest_target
        lookahead = max(math.hypot(tx, ty), self.lookahead_distance)
        alpha = math.atan2(ty, tx)
        steering = math.atan2(4.0 * self.wheelbase * math.sin(alpha), lookahead)
        steering = max(-self.max_steer, min(self.max_steer, steering))

        throttle = self.target_speed
        if tx < 1.5:
            throttle *= 0.7
        if abs(ty) > 1.0:
            throttle *= 0.8

        self.steer_pub.publish(Float32(data=float(steering)))
        self.throttle_pub.publish(Float32(data=float(throttle)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PurePursuitTargetFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.follow_active_pub.publish(Bool(data=False))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

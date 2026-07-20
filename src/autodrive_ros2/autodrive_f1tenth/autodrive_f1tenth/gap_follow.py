#!/usr/bin/env python3

import math
from dataclasses import dataclass, field

import numpy as np

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker


@dataclass
class GapFollowerState:
    car_id: int
    latest_scan: LaserScan | None = None
    target_angle: float = 0.0
    best_distance: float = 0.0
    current_speed: float = 0.0
    current_steering: float = 0.0
    gap_found: bool = False
    last_error: float = 0.0
    error_integral: float = 0.0
    last_control_time: float | None = None
    debug_counter: int = 0
    steer_pub: any = None
    throttle_pub: any = None
    marker_pub: any = None
    gap_points: list[tuple[float, float]] = field(default_factory=list)


class DualGapFollower(Node):
    def __init__(self) -> None:
        super().__init__("gap_follow_f1tenth")

        self.declare_parameter("active_car_ids", [1, 2])
        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("base_speed", 0.13)
        self.declare_parameter("min_speed", 0.08)
        self.declare_parameter("max_range", 8.0)
        self.declare_parameter("min_range", 0.06)
        self.declare_parameter("gap_threshold", 1.2)
        self.declare_parameter("bubble_radius_m", 0.55)
        self.declare_parameter("forward_fov_deg", 100.0)
        self.declare_parameter("smoothing_window", 5)
        self.declare_parameter("steering_kp", 0.9)
        self.declare_parameter("steering_ki", 0.0)
        self.declare_parameter("steering_kd", 0.12)
        self.declare_parameter("max_steer_deg", 90.0)
        self.declare_parameter("slow_steer_deg", 20.0)
        self.declare_parameter("high_steer_speed_scale", 0.65)
        self.declare_parameter("publish_markers", True)

        self.control_period = float(self.get_parameter("control_period").value)
        self.base_speed = float(self.get_parameter("base_speed").value)
        self.min_speed = float(self.get_parameter("min_speed").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.min_range = float(self.get_parameter("min_range").value)
        self.gap_threshold = float(self.get_parameter("gap_threshold").value)
        self.bubble_radius_m = float(self.get_parameter("bubble_radius_m").value)
        self.forward_fov = math.radians(float(self.get_parameter("forward_fov_deg").value))
        self.smoothing_window = max(1, int(self.get_parameter("smoothing_window").value))
        self.steering_kp = float(self.get_parameter("steering_kp").value)
        self.steering_ki = float(self.get_parameter("steering_ki").value)
        self.steering_kd = float(self.get_parameter("steering_kd").value)
        self.max_steer = math.radians(float(self.get_parameter("max_steer_deg").value))
        self.slow_steer_deg = float(self.get_parameter("slow_steer_deg").value)
        self.high_steer_speed_scale = float(self.get_parameter("high_steer_speed_scale").value)
        self.publish_markers = bool(self.get_parameter("publish_markers").value)
        self.active_car_ids = self._parse_active_car_ids(self.get_parameter("active_car_ids").value)

        self.car_states: dict[int, GapFollowerState] = {}
        for car_id in self.active_car_ids:
            state = GapFollowerState(car_id=car_id)
            state.steer_pub = self.create_publisher(
                Float32,
                f"/autodrive/f1tenth_{car_id}/steering_command",
                10,
            )
            state.throttle_pub = self.create_publisher(
                Float32,
                f"/autodrive/f1tenth_{car_id}/throttle_command",
                10,
            )
            if self.publish_markers:
                state.marker_pub = self.create_publisher(
                    Marker,
                    f"/autodrive/f1tenth_{car_id}/gap_follow/marker",
                    10,
                )

            self.create_subscription(
                LaserScan,
                f"/autodrive/f1tenth_{car_id}/lidar",
                lambda msg, current_car_id=car_id: self.scan_cb(current_car_id, msg),
                10,
            )
            self.car_states[car_id] = state

        self.create_timer(self.control_period, self.control_loop)
        self.get_logger().info(
            "Gap follow ready. "
            f"active_cars={self.active_car_ids}, base_speed={self.base_speed:.2f}, "
            f"forward_fov_deg={math.degrees(self.forward_fov):.1f}"
        )

    def _parse_active_car_ids(self, raw_value: object) -> tuple[int, ...]:
        car_ids: list[int] = []
        if isinstance(raw_value, int):
            tokens = [raw_value]
        elif isinstance(raw_value, str):
            tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
        elif isinstance(raw_value, (list, tuple)):
            tokens = list(raw_value)
        else:
            raise ValueError(f"Unsupported active_car_ids value: {raw_value!r}")

        for token in tokens:
            car_id = int(token)
            if car_id not in (1, 2):
                raise ValueError(f"Unsupported car_id '{car_id}' in active_car_ids")
            if car_id not in car_ids:
                car_ids.append(car_id)
        if not car_ids:
            raise ValueError("active_car_ids must contain at least one car id")
        return tuple(car_ids)

    def scan_cb(self, car_id: int, msg: LaserScan) -> None:
        state = self.car_states[car_id]
        state.latest_scan = msg

        angles = msg.angle_min + np.arange(len(msg.ranges), dtype=float) * msg.angle_increment
        ranges = np.asarray(msg.ranges, dtype=float)
        valid = np.isfinite(ranges) & (ranges >= self.min_range) & (ranges <= self.max_range)
        forward = np.abs(angles) <= self.forward_fov
        processed = np.where(valid & forward, ranges, 0.0)

        if self.smoothing_window > 1:
            kernel = np.ones(self.smoothing_window, dtype=float) / float(self.smoothing_window)
            processed = np.convolve(processed, kernel, mode="same")
            processed = np.where(forward, processed, 0.0)

        if not np.any(processed > 0.0):
            state.gap_found = False
            state.target_angle = 0.0
            state.best_distance = 0.0
            state.gap_points.clear()
            return

        closest_idx = int(np.argmax(np.where(processed > 0.0, 1.0 / np.maximum(processed, 1e-6), 0.0)))
        bubble_half_width = max(
            1,
            int(self.bubble_radius_m / max(msg.angle_increment * max(processed[closest_idx], 0.1), 1e-6)),
        )
        masked = processed.copy()
        masked[max(0, closest_idx - bubble_half_width) : min(len(masked), closest_idx + bubble_half_width + 1)] = 0.0

        gap_segments = self._find_gap_segments(masked)
        if not gap_segments:
            fallback = self._fallback_target(masked, angles)
            if fallback is None:
                state.gap_found = False
                state.target_angle = 0.0
                state.best_distance = 0.0
                state.gap_points.clear()
                return

            chosen_idx, start_idx, end_idx = fallback
            state.target_angle = float(angles[chosen_idx])
            state.best_distance = float(masked[chosen_idx])
            state.gap_found = True
            state.gap_points = [
                (
                    float(masked[start_idx] * math.cos(angles[start_idx])),
                    float(masked[start_idx] * math.sin(angles[start_idx])),
                ),
                (
                    float(masked[end_idx] * math.cos(angles[end_idx])),
                    float(masked[end_idx] * math.sin(angles[end_idx])),
                ),
            ]
            return

        best_segment = max(gap_segments, key=lambda seg: self._score_segment(seg, masked, angles))
        start_idx, end_idx = best_segment
        segment_ranges = masked[start_idx : end_idx + 1]
        segment_angles = angles[start_idx : end_idx + 1]
        best_idx_local = int(np.argmax(segment_ranges))
        best_idx = start_idx + best_idx_local
        center_idx = (start_idx + end_idx) // 2
        chosen_idx = int(round(0.7 * best_idx + 0.3 * center_idx))
        chosen_idx = max(start_idx, min(end_idx, chosen_idx))

        state.target_angle = float(angles[chosen_idx])
        state.best_distance = float(masked[chosen_idx])
        state.gap_found = True
        state.gap_points = [
            (
                float(masked[start_idx] * math.cos(angles[start_idx])),
                float(masked[start_idx] * math.sin(angles[start_idx])),
            ),
            (
                float(masked[end_idx] * math.cos(angles[end_idx])),
                float(masked[end_idx] * math.sin(angles[end_idx])),
            ),
        ]

    def _find_gap_segments(self, masked_ranges: np.ndarray) -> list[tuple[int, int]]:
        segments: list[tuple[int, int]] = []
        start_idx: int | None = None
        for idx, value in enumerate(masked_ranges):
            if value >= self.gap_threshold:
                if start_idx is None:
                    start_idx = idx
            elif start_idx is not None:
                if idx - start_idx >= 3:
                    segments.append((start_idx, idx - 1))
                start_idx = None
        if start_idx is not None and len(masked_ranges) - start_idx >= 3:
            segments.append((start_idx, len(masked_ranges) - 1))
        return segments

    def _fallback_target(self, masked_ranges: np.ndarray, angles: np.ndarray) -> tuple[int, int, int] | None:
        valid_idxs = np.flatnonzero(masked_ranges > 0.0)
        if valid_idxs.size == 0:
            return None

        runs: list[tuple[int, int]] = []
        run_start = int(valid_idxs[0])
        prev_idx = int(valid_idxs[0])
        for idx in valid_idxs[1:]:
            idx = int(idx)
            if idx == prev_idx + 1:
                prev_idx = idx
                continue
            runs.append((run_start, prev_idx))
            run_start = idx
            prev_idx = idx
        runs.append((run_start, prev_idx))

        best_run = max(
            runs,
            key=lambda run: 0.7 * float(np.max(masked_ranges[run[0] : run[1] + 1]))
            + 0.3 * float(run[1] - run[0] + 1)
            - 0.2 * abs(float(angles[(run[0] + run[1]) // 2])),
        )
        start_idx, end_idx = best_run
        chosen_idx = int(np.argmax(masked_ranges[start_idx : end_idx + 1])) + start_idx
        return chosen_idx, start_idx, end_idx

    def _score_segment(self, segment: tuple[int, int], masked_ranges: np.ndarray, angles: np.ndarray) -> float:
        start_idx, end_idx = segment
        segment_ranges = masked_ranges[start_idx : end_idx + 1]
        segment_angles = angles[start_idx : end_idx + 1]
        width_score = float(end_idx - start_idx + 1)
        distance_score = float(np.max(segment_ranges))
        center_angle = float(segment_angles[len(segment_angles) // 2])
        center_penalty = abs(center_angle)
        return 1.5 * distance_score + 0.05 * width_score - 0.8 * center_penalty

    def control_loop(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        for car_id, state in self.car_states.items():
            steering_cmd = 0.0
            throttle_cmd = 0.0

            if state.gap_found:
                if state.last_control_time is None:
                    dt = self.control_period
                else:
                    dt = max(1e-3, now - state.last_control_time)

                error = state.target_angle
                state.error_integral += error * dt
                state.error_integral = max(-1.0, min(1.0, state.error_integral))
                error_derivative = (error - state.last_error) / dt
                steering_cmd = (
                    self.steering_kp * error
                    + self.steering_ki * state.error_integral
                    + self.steering_kd * error_derivative
                )
                steering_cmd = max(-self.max_steer, min(self.max_steer, steering_cmd))

                steer_abs_deg = abs(math.degrees(steering_cmd))
                if steer_abs_deg > self.slow_steer_deg:
                    throttle_cmd = self.base_speed * self.high_steer_speed_scale
                else:
                    throttle_cmd = self.base_speed

                if state.best_distance > 0.0:
                    closeness_scale = max(0.4, min(1.0, state.best_distance / 2.0))
                    throttle_cmd *= closeness_scale
                throttle_cmd = max(self.min_speed, throttle_cmd)

                state.last_error = error
                state.last_control_time = now
            else:
                state.error_integral = 0.0
                state.last_error = 0.0
                state.last_control_time = now

            state.current_steering = float(steering_cmd)
            state.current_speed = float(throttle_cmd)

            state.steer_pub.publish(Float32(data=state.current_steering))
            state.throttle_pub.publish(Float32(data=state.current_speed))

            if self.publish_markers and state.marker_pub is not None:
                self._publish_marker(state)

            state.debug_counter += 1
            if state.debug_counter % 20 == 0:
                self.get_logger().info(
                    f"car={car_id} gap_found={state.gap_found} "
                    f"angle={math.degrees(state.target_angle):.1f}deg "
                    f"dist={state.best_distance:.2f} "
                    f"cmd=({state.current_speed:.2f},{state.current_steering:.3f})"
                )

    def _publish_marker(self, state: GapFollowerState) -> None:
        marker = Marker()
        marker.header.frame_id = f"lidar_{state.car_id}"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "gap_follow"
        marker.id = state.car_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD if state.gap_found and len(state.gap_points) == 2 else Marker.DELETE
        marker.scale.x = 0.08
        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 0.2
        marker.color.a = 1.0
        if marker.action == Marker.ADD:
            marker.points = []
            for x, y in state.gap_points:
                point = Point()
                point.x = x
                point.y = y
                point.z = 0.0
                marker.points.append(point)
        state.marker_pub.publish(marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DualGapFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

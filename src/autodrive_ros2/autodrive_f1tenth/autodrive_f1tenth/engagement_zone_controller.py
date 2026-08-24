#!/usr/bin/env python3

import csv
import math
from pathlib import Path

import numpy as np

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String
from visualization_msgs.msg import Marker, MarkerArray

from autodrive_f1tenth.pure_pursuit import load_manual_reference_line


class EngagementZoneController(Node):
    """Pure pursuit for car 1 with a one-time switch onto the planned EZ path."""

    NORMAL = "NORMAL"
    ENGAGEMENT = "ENGAGEMENT"
    COMPLETE = "COMPLETE"

    def __init__(self) -> None:
        super().__init__("engagement_zone_controller")

        package_root = Path(__file__).resolve().parents[1]
        default_path = (
            package_root
            / "output"
            / "engagement_zones_dynamic"
            / "engagement_zone_dynamic_one_lap_path.csv"
        )

        self.declare_parameter("path_csv", str(default_path))
        self.declare_parameter("track_name", "ethz")
        self.declare_parameter("num_path_points", 800)
        self.declare_parameter("base_target_speed", 0.13)
        self.declare_parameter("speed_multiplier", 1.0)
        self.declare_parameter("engagement_speed_delay_sec", 1.0)
        self.declare_parameter("lookahead_distance", 2.5)
        self.declare_parameter("engagement_lookahead_distance", 2.5)
        self.declare_parameter("wheelbase", 0.30)
        self.declare_parameter("max_steer_deg", 90.0)
        self.declare_parameter("slow_steer_deg", 10.0)
        self.declare_parameter("high_steer_speed_scale", 0.65)
        self.declare_parameter("control_period", 0.10)
        self.declare_parameter("c_start_radius_m", 1.0)
        self.declare_parameter("c_end_radius_m", 0.8)
        self.declare_parameter("stop_after_laps", 1)
        self.declare_parameter("lap_start_radius_m", 1.5)
        self.declare_parameter("lap_min_distance_m", 35.0)
        self.declare_parameter("wait_for_startup_gate", False)

        self.path_csv = Path(str(self.get_parameter("path_csv").value)).expanduser()
        self.normal_path = load_manual_reference_line(
            int(self.get_parameter("num_path_points").value),
            track_name=str(self.get_parameter("track_name").value),
        )
        self.engagement_path = self._load_engagement_path(self.path_csv)
        self.c_start = self.engagement_path[0]
        self.c_end = self.engagement_path[-1]

        self.base_target_speed = float(self.get_parameter("base_target_speed").value)
        self.speed_multiplier = float(self.get_parameter("speed_multiplier").value)
        if self.speed_multiplier <= 0.0:
            raise ValueError("speed_multiplier must be positive")
        self.engagement_target_speed = self.base_target_speed * self.speed_multiplier
        self.engagement_speed_delay_sec = max(
            0.0, float(self.get_parameter("engagement_speed_delay_sec").value)
        )
        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        self.engagement_lookahead_distance = float(
            self.get_parameter("engagement_lookahead_distance").value
        )
        self.wheelbase = float(self.get_parameter("wheelbase").value)
        self.max_steer = math.radians(float(self.get_parameter("max_steer_deg").value))
        self.slow_steer_deg = float(self.get_parameter("slow_steer_deg").value)
        self.high_steer_speed_scale = float(
            self.get_parameter("high_steer_speed_scale").value
        )
        self.control_period = float(self.get_parameter("control_period").value)
        self.c_start_radius_m = float(self.get_parameter("c_start_radius_m").value)
        self.c_end_radius_m = float(self.get_parameter("c_end_radius_m").value)
        self.stop_after_laps = int(self.get_parameter("stop_after_laps").value)
        self.lap_start_radius_m = float(self.get_parameter("lap_start_radius_m").value)
        self.lap_min_distance_m = float(self.get_parameter("lap_min_distance_m").value)
        self.wait_for_startup_gate = bool(
            self.get_parameter("wait_for_startup_gate").value
        )

        self.position: np.ndarray | None = None
        self.previous_position: np.ndarray | None = None
        self.heading: float | None = None
        self.normal_idx = 0
        self.engagement_idx = 0
        self.mode = self.NORMAL
        self.engagement_completed = False
        self.engagement_started_time: float | None = None
        self.start_position: np.ndarray | None = None
        self.total_distance_m = 0.0
        self.left_start_zone = False
        self.lap_count = 0
        self.stopped = False
        self.startup_gate_open = not self.wait_for_startup_gate
        self.last_log_time = -math.inf

        self.steer_pub = self.create_publisher(
            Float32, "/autodrive/f1tenth_1/steering_command", 10
        )
        self.throttle_pub = self.create_publisher(
            Float32, "/autodrive/f1tenth_1/throttle_command", 10
        )
        self.lap_pub = self.create_publisher(
            Int32, "/autodrive/f1tenth_1/pure_pursuit/lap_count", 10
        )
        self.mode_pub = self.create_publisher(
            String, "/autodrive/f1tenth_1/engagement_zone/mode", 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, "/autodrive/f1tenth_1/engagement_zone/path_markers", 10
        )
        self.create_subscription(Point, "/autodrive/f1tenth_1/ips", self.ips_cb, 10)
        self.create_subscription(
            Bool, "/autodrive/startup_gate/open", self.startup_gate_cb, 10
        )
        self.create_timer(self.control_period, self.control_loop)
        self.create_timer(0.5, self.publish_markers)

        self.get_logger().info(
            "EZ controller ready. "
            f"path={self.path_csv}, points={len(self.engagement_path)}, "
            f"normal_speed={self.base_target_speed:.3f}, "
            f"engagement_speed={self.engagement_target_speed:.3f} "
            f"({self.speed_multiplier:.2f}x), "
            f"speed_delay={self.engagement_speed_delay_sec:.2f}s, "
            f"c_start=({self.c_start[0]:.2f},{self.c_start[1]:.2f}), "
            f"c_end=({self.c_end[0]:.2f},{self.c_end[1]:.2f})"
        )

    @staticmethod
    def _load_engagement_path(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Engagement-zone path CSV not found: {path}")
        points: list[tuple[float, float]] = []
        with path.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                try:
                    point = (float(row["ego_x_m"]), float(row["ego_y_m"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if not points or math.hypot(point[0] - points[-1][0], point[1] - points[-1][1]) > 1e-4:
                    points.append(point)
        if len(points) < 2:
            raise ValueError(f"Engagement-zone path has fewer than two usable points: {path}")
        return np.asarray(points, dtype=float)

    def ips_cb(self, msg: Point) -> None:
        current = np.array([float(msg.x), float(msg.y)], dtype=float)
        if self.position is None:
            self.position = current
            self.previous_position = current.copy()
            self.start_position = current.copy()
            self.normal_idx = int(np.argmin(np.linalg.norm(self.normal_path - current, axis=1)))
            next_idx = (self.normal_idx + 1) % len(self.normal_path)
            tangent = self.normal_path[next_idx] - self.normal_path[self.normal_idx]
            self.heading = math.atan2(float(tangent[1]), float(tangent[0]))
            return

        displacement = current - self.position
        distance = float(np.linalg.norm(displacement))
        if distance > 1e-5:
            self.heading = math.atan2(float(displacement[1]), float(displacement[0]))
            self.total_distance_m += distance
        self.previous_position = self.position
        self.position = current
        self._update_lap_state()

    def startup_gate_cb(self, msg: Bool) -> None:
        self.startup_gate_open = bool(msg.data)

    def _update_lap_state(self) -> None:
        if self.position is None or self.start_position is None or self.stopped:
            return
        distance_to_start = float(np.linalg.norm(self.position - self.start_position))
        if distance_to_start > self.lap_start_radius_m:
            self.left_start_zone = True
        if (
            self.left_start_zone
            and self.total_distance_m >= self.lap_min_distance_m
            and distance_to_start <= self.lap_start_radius_m
        ):
            self.lap_count += 1
            self.left_start_zone = False
            self.lap_pub.publish(Int32(data=self.lap_count))
            self.get_logger().info(f"Car 1 completed lap {self.lap_count}")
            if self.stop_after_laps > 0 and self.lap_count >= self.stop_after_laps:
                self.stopped = True

    def _update_mode(self) -> None:
        if self.position is None:
            return
        if self.mode == self.NORMAL and not self.engagement_completed:
            if float(np.linalg.norm(self.position - self.c_start)) <= self.c_start_radius_m:
                self.mode = self.ENGAGEMENT
                self.engagement_started_time = (
                    self.get_clock().now().nanoseconds * 1e-9
                )
                self.engagement_idx = int(
                    np.argmin(np.linalg.norm(self.engagement_path - self.position, axis=1))
                )
                self.get_logger().info(
                    f"Reached c_start: switching car 1 to EZ path at index {self.engagement_idx}"
                )
        elif self.mode == self.ENGAGEMENT:
            at_end = self.engagement_idx >= len(self.engagement_path) - 2
            near_end = float(np.linalg.norm(self.position - self.c_end)) <= self.c_end_radius_m
            if at_end and near_end:
                self.mode = self.COMPLETE
                self.engagement_completed = True
                self.normal_idx = int(
                    np.argmin(np.linalg.norm(self.normal_path - self.position, axis=1))
                )
                self.get_logger().info("Reached c_end: returning car 1 to the normal spline")

    def control_loop(self) -> None:
        if not self.startup_gate_open:
            self.steer_pub.publish(Float32(data=0.0))
            self.throttle_pub.publish(Float32(data=0.0))
            return
        if self.position is None or self.heading is None:
            return
        if self.stopped:
            self.steer_pub.publish(Float32(data=0.0))
            self.throttle_pub.publish(Float32(data=0.0))
            return

        self._update_mode()
        if self.mode == self.ENGAGEMENT:
            path = self.engagement_path
            lookahead = self.engagement_lookahead_distance
            steering, self.engagement_idx = self._pursue_open(
                path, self.engagement_idx, lookahead
            )
        else:
            path = self.normal_path
            lookahead = self.lookahead_distance
            steering, self.normal_idx = self._pursue_closed(
                path, self.normal_idx, lookahead
            )

        now = self.get_clock().now().nanoseconds * 1e-9
        speed_delay_elapsed = (
            self.mode == self.ENGAGEMENT
            and self.engagement_started_time is not None
            and now - self.engagement_started_time >= self.engagement_speed_delay_sec
        )
        if speed_delay_elapsed:
            throttle = self.engagement_target_speed
        else:
            throttle = self.base_target_speed
        if abs(math.degrees(steering)) > self.slow_steer_deg:
            throttle *= self.high_steer_speed_scale
        self.steer_pub.publish(Float32(data=float(steering)))
        self.throttle_pub.publish(Float32(data=float(throttle)))
        self.mode_pub.publish(String(data=self.mode))

        if now - self.last_log_time >= 1.0:
            self.last_log_time = now
            self.get_logger().info(
                f"mode={self.mode} pos=({self.position[0]:.2f},{self.position[1]:.2f}) "
                f"cmd=({throttle:.3f},{steering:.3f})"
            )

    def _pursue_closed(
        self, path: np.ndarray, idx: int, lookahead: float
    ) -> tuple[float, int]:
        for _ in range(len(path)):
            target = path[idx]
            if float(np.linalg.norm(target - self.position)) > lookahead:
                break
            idx = (idx + 1) % len(path)
        return self._steering_to(target, lookahead), idx

    def _pursue_open(
        self, path: np.ndarray, idx: int, lookahead: float
    ) -> tuple[float, int]:
        while idx < len(path) - 1:
            target = path[idx]
            if float(np.linalg.norm(target - self.position)) > lookahead:
                break
            idx += 1
        target = path[min(idx, len(path) - 1)]
        return self._steering_to(target, lookahead), idx

    def _steering_to(self, target: np.ndarray, lookahead: float) -> float:
        delta = target - self.position
        cos_yaw = math.cos(-self.heading)
        sin_yaw = math.sin(-self.heading)
        vehicle_x = float(delta[0]) * cos_yaw - float(delta[1]) * sin_yaw
        vehicle_y = float(delta[0]) * sin_yaw + float(delta[1]) * cos_yaw
        alpha = math.atan2(vehicle_y, vehicle_x)
        steering = math.atan2(4.0 * self.wheelbase * math.sin(alpha), lookahead)
        return max(-self.max_steer, min(self.max_steer, steering))

    def publish_markers(self) -> None:
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        line = Marker()
        line.header.frame_id = "map"
        line.header.stamp = stamp
        line.ns = "engagement_zone_path"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.07
        line.color.r = 0.05
        line.color.g = 0.35
        line.color.b = 1.0
        line.color.a = 0.95
        for x_value, y_value in self.engagement_path:
            line.points.append(Point(x=float(x_value), y=float(y_value), z=0.08))
        markers.markers.append(line)

        for marker_id, (label, point, red, green) in enumerate(
            (("c_start", self.c_start, 0.65, 0.0), ("c_end", self.c_end, 0.0, 0.65)),
            start=1,
        ):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "engagement_zone_endpoints"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = Point(x=float(point[0]), y=float(point[1]), z=0.10)
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.30
            marker.color.r = red
            marker.color.g = green
            marker.color.b = 0.1
            marker.color.a = 1.0
            markers.markers.append(marker)

        self.marker_pub.publish(markers)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EngagementZoneController()
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

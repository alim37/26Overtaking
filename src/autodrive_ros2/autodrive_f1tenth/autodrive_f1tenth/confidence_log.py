#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


def get_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "tracks" / "src").exists():
            return parent
    return Path.cwd()


class ConfidenceLogger(Node):
    def __init__(self) -> None:
        super().__init__("confidence_logger")

        repo_root = get_repo_root()
        default_output = repo_root / "output" / "confidence_runs" / "confidence_run.csv"

        self.declare_parameter("pose_topic", "/autodrive/f1tenth_1/ips")
        self.declare_parameter("confidence_topic", "/autodrive/f1tenth_1/target_tracker/tracking_confidence")
        self.declare_parameter("visible_topic", "/autodrive/f1tenth_1/target_tracker/target_visible")
        self.declare_parameter("throttle_command_topic", "/autodrive/f1tenth_1/throttle_command")
        self.declare_parameter("steering_command_topic", "/autodrive/f1tenth_1/steering_command")
        self.declare_parameter("output_path", str(default_output))
        self.declare_parameter("sample_period", 0.05)
        self.declare_parameter("stop_after_one_lap", True)
        self.declare_parameter("lap_start_radius_m", 1.5)
        self.declare_parameter("lap_min_distance_m", 35.0)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.confidence_topic = str(self.get_parameter("confidence_topic").value)
        self.visible_topic = str(self.get_parameter("visible_topic").value)
        self.throttle_command_topic = str(self.get_parameter("throttle_command_topic").value)
        self.steering_command_topic = str(self.get_parameter("steering_command_topic").value)
        output_path_str = str(self.get_parameter("output_path").value)
        self.output_path = Path(output_path_str).expanduser()
        self._uses_default_output_name = self.output_path == default_output
        self.sample_period = float(self.get_parameter("sample_period").value)
        self.stop_after_one_lap = bool(self.get_parameter("stop_after_one_lap").value)
        self.lap_start_radius_m = float(self.get_parameter("lap_start_radius_m").value)
        self.lap_min_distance_m = float(self.get_parameter("lap_min_distance_m").value)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.output_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "time_sec",
                "elapsed_sec",
                "confidence",
                "visible",
                "avg_speed_mps",
                "ips_x_m",
                "ips_y_m",
                "speed_mps",
                "throttle_command",
                "steering_command",
            ]
        )

        self.start_time_sec: float | None = None
        self.latest_pose: tuple[float, float] | None = None
        self.latest_speed_mps = 0.0
        self.last_pose_time_sec: float | None = None
        self.latest_confidence = 0.0
        self.latest_visible = False
        self.latest_throttle_cmd = 0.0
        self.latest_steering_cmd = 0.0

        self.start_pose_xy: tuple[float, float] | None = None
        self.prev_pose_xy: tuple[float, float] | None = None
        self.total_distance_m = 0.0
        self.left_start_zone = False
        self.completed_lap = False
        self.rows_written = 0
        self.saved_once = False

        self.create_subscription(Point, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(Float32, self.confidence_topic, self.confidence_cb, 10)
        self.create_subscription(Bool, self.visible_topic, self.visible_cb, 10)
        self.create_subscription(Float32, self.throttle_command_topic, self.throttle_cb, 10)
        self.create_subscription(Float32, self.steering_command_topic, self.steering_cb, 10)
        self.create_timer(self.sample_period, self.log_sample)

        self.get_logger().info(
            f"Confidence logger ready. pose={self.pose_topic}, confidence={self.confidence_topic}, output={self.output_path}"
        )

    def pose_cb(self, msg: Point) -> None:
        current_xy = (float(msg.x), float(msg.y))
        self.latest_pose = current_xy

        if self.start_pose_xy is None:
            self.start_pose_xy = current_xy
            self.prev_pose_xy = current_xy
            self.last_pose_time_sec = self.get_clock().now().nanoseconds * 1e-9
            return

        if self.prev_pose_xy is not None:
            segment_distance = math.hypot(current_xy[0] - self.prev_pose_xy[0], current_xy[1] - self.prev_pose_xy[1])
            self.total_distance_m += segment_distance
            if self.last_pose_time_sec is not None:
                now_sec = self.get_clock().now().nanoseconds * 1e-9
                dt = max(now_sec - self.last_pose_time_sec, 1e-6)
                self.latest_speed_mps = segment_distance / dt
                self.last_pose_time_sec = now_sec
            else:
                self.last_pose_time_sec = self.get_clock().now().nanoseconds * 1e-9
        self.prev_pose_xy = current_xy

        distance_to_start = math.hypot(current_xy[0] - self.start_pose_xy[0], current_xy[1] - self.start_pose_xy[1])
        if distance_to_start > self.lap_start_radius_m:
            self.left_start_zone = True

        if (
            self.stop_after_one_lap
            and not self.completed_lap
            and self.left_start_zone
            and self.total_distance_m >= self.lap_min_distance_m
            and distance_to_start <= self.lap_start_radius_m
        ):
            self.completed_lap = True
            self.save_csv()
            self.get_logger().info(
                f"Completed one lap and saved {self.rows_written} confidence samples to {self.output_path}"
            )

    def confidence_cb(self, msg: Float32) -> None:
        self.latest_confidence = float(msg.data)

    def visible_cb(self, msg: Bool) -> None:
        self.latest_visible = bool(msg.data)

    def throttle_cb(self, msg: Float32) -> None:
        self.latest_throttle_cmd = float(msg.data)

    def steering_cb(self, msg: Float32) -> None:
        self.latest_steering_cmd = float(msg.data)

    def log_sample(self) -> None:
        if self.completed_lap or self.latest_pose is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.start_time_sec is None:
            self.start_time_sec = now_sec
        elapsed_sec = now_sec - self.start_time_sec

        self.csv_writer.writerow(
            [
                f"{now_sec:.9f}",
                f"{elapsed_sec:.6f}",
                f"{self.latest_confidence:.6f}",
                int(self.latest_visible),
                f"{self.average_speed_mps():.6f}",
                f"{self.latest_pose[0]:.6f}",
                f"{self.latest_pose[1]:.6f}",
                f"{self.latest_speed_mps:.6f}",
                f"{self.latest_throttle_cmd:.6f}",
                f"{self.latest_steering_cmd:.6f}",
            ]
        )
        self.rows_written += 1
        if self.rows_written % 20 == 0:
            self.csv_file.flush()

    def save_csv(self) -> None:
        if self.saved_once:
            return
        self.csv_file.flush()
        self.csv_file.close()
        if self._uses_default_output_name:
            avg_speed = self.average_speed_mps()
            speed_tag = f"{avg_speed:.3f}".rstrip("0").rstrip(".").replace(".", "p")
            renamed_path = self.output_path.with_name(f"confidence_run_{speed_tag}.csv")
            self.output_path.rename(renamed_path)
            self.output_path = renamed_path
        self.saved_once = True

    def average_speed_mps(self) -> float:
        if self.start_time_sec is None:
            return 0.0
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        elapsed = max(now_sec - self.start_time_sec, 1e-6)
        return self.total_distance_m / elapsed

    def destroy_node(self) -> bool:
        try:
            if not self.saved_once:
                self.csv_file.flush()
                self.csv_file.close()
                self.get_logger().info(
                    f"Saved {self.rows_written} confidence samples to {self.output_path}"
                )
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConfidenceLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

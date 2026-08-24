#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32


class EthzAllDataLogger(Node):
    def __init__(self) -> None:
        super().__init__("ethz_all_data_logger")

        package_root = Path(__file__).resolve().parents[1]
        default_output = package_root / "output" / "ethz_all_data" / "ethz_all_data.csv"
        self.declare_parameter("output_path", str(default_output))
        self.declare_parameter("track_name", "ethz")
        self.declare_parameter("sample_period_sec", 0.05)
        self.declare_parameter("lap_start_radius_m", 1.5)
        self.declare_parameter("lap_min_distance_m", 35.0)

        self.output_path = Path(str(self.get_parameter("output_path").value)).expanduser()
        self.track_name = str(self.get_parameter("track_name").value).strip().lower()
        self.sample_period_sec = float(self.get_parameter("sample_period_sec").value)
        self.lap_start_radius_m = float(self.get_parameter("lap_start_radius_m").value)
        self.lap_min_distance_m = float(self.get_parameter("lap_min_distance_m").value)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.output_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(
            [
                "logger_timestamp_nanoseconds",
                "x_m",
                "y_m",
                "velocity_mps",
                "acceleration_mps2",
                "steering_command",
                "throttle_command",
                "steering_feedback",
                "throttle_feedback",
            ]
        )

        self.latest_pose: tuple[float, float, float] | None = None
        self.previous_pose: tuple[float, float] | None = None
        self.velocity_mps: float | None = None
        self.max_velocity_mps = 0.0
        self.acceleration_mps2: float | None = None
        self.steering_feedback = 0.0
        self.throttle_feedback = 0.0
        self.steering_command = 0.0
        self.throttle_command = 0.0

        self.start_pose: tuple[float, float] | None = None
        self.total_distance_m = 0.0
        self.left_start_zone = False
        self.completed_lap = False
        self.rows_written = 0
        self.file_closed = False
        self.shutdown_timer = None

        prefix = "/autodrive/f1tenth_1"
        self.create_subscription(Point, f"{prefix}/ips", self.pose_cb, 10)
        self.create_subscription(Odometry, f"{prefix}/odom", self.odom_cb, 10)
        self.create_subscription(Imu, f"{prefix}/imu", self.imu_cb, 10)
        self.create_subscription(Float32, f"{prefix}/steering", self.steering_feedback_cb, 10)
        self.create_subscription(Float32, f"{prefix}/throttle", self.throttle_feedback_cb, 10)
        self.create_subscription(Float32, f"{prefix}/steering_command", self.steering_command_cb, 10)
        self.create_subscription(Float32, f"{prefix}/throttle_command", self.throttle_command_cb, 10)
        self.create_timer(self.sample_period_sec, self.log_sample)

        self.get_logger().info(f"ETHZ all-data logger ready. output={self.output_path}")

    def pose_cb(self, msg: Point) -> None:
        current_pose = (float(msg.x), float(msg.y), float(msg.z))
        current_xy = current_pose[:2]
        self.latest_pose = current_pose

        if self.start_pose is None:
            self.start_pose = current_xy
            self.previous_pose = current_xy
            return

        if self.previous_pose is not None:
            dx = current_xy[0] - self.previous_pose[0]
            dy = current_xy[1] - self.previous_pose[1]
            segment_distance = math.hypot(dx, dy)
            self.total_distance_m += segment_distance

        self.previous_pose = current_xy
        distance_to_start = math.hypot(
            current_xy[0] - self.start_pose[0],
            current_xy[1] - self.start_pose[1],
        )
        if distance_to_start > self.lap_start_radius_m:
            self.left_start_zone = True

        if (
            not self.completed_lap
            and self.left_start_zone
            and self.total_distance_m >= self.lap_min_distance_m
            and distance_to_start <= self.lap_start_radius_m
        ):
            self.completed_lap = True
            self.log_sample(force=True)
            self.close_file()
            self.get_logger().info(
                f"Completed one lap. Saved {self.rows_written} rows to {self.output_path}"
            )
            self.shutdown_timer = self.create_timer(0.25, self.request_shutdown)

    def odom_cb(self, msg: Odometry) -> None:
        self.velocity_mps = abs(float(msg.twist.twist.linear.z))
        self.max_velocity_mps = max(self.max_velocity_mps, self.velocity_mps)

    def imu_cb(self, msg: Imu) -> None:
        acceleration = msg.linear_acceleration
        self.acceleration_mps2 = math.sqrt(
            float(acceleration.x) ** 2
            + float(acceleration.y) ** 2
            + float(acceleration.z) ** 2
        )

    def steering_feedback_cb(self, msg: Float32) -> None:
        self.steering_feedback = float(msg.data)

    def throttle_feedback_cb(self, msg: Float32) -> None:
        self.throttle_feedback = float(msg.data)

    def steering_command_cb(self, msg: Float32) -> None:
        self.steering_command = float(msg.data)

    def throttle_command_cb(self, msg: Float32) -> None:
        self.throttle_command = float(msg.data)

    def log_sample(self, force: bool = False) -> None:
        if self.file_closed or self.latest_pose is None:
            return
        if self.completed_lap and not force:
            return

        logger_timestamp_nanoseconds = self.get_clock().now().nanoseconds
        self.writer.writerow(
            [
                logger_timestamp_nanoseconds,
                f"{self.latest_pose[0]:.6f}",
                f"{self.latest_pose[1]:.6f}",
                "" if self.velocity_mps is None else f"{self.velocity_mps:.6f}",
                "" if self.acceleration_mps2 is None else f"{self.acceleration_mps2:.6f}",
                f"{self.steering_command:.6f}",
                f"{self.throttle_command:.6f}",
                f"{self.steering_feedback:.6f}",
                f"{self.throttle_feedback:.6f}",
            ]
        )
        self.rows_written += 1
        if self.rows_written % 20 == 0:
            self.csv_file.flush()

    def close_file(self) -> None:
        if self.file_closed:
            return
        self.csv_file.flush()
        self.csv_file.close()
        self.file_closed = True
        speed_tag = f"{self.max_velocity_mps:.2f}".rstrip("0").rstrip(".")
        if self.track_name == "ethzmobil":
            final_dir = self.output_path.parent.parent / "ethz_mobil_all"
            final_name = f"ethz_mobil_all_speed_{speed_tag}.csv"
        else:
            final_dir = self.output_path.parent
            final_name = f"ethz_all_data_speed_{speed_tag}.csv"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / final_name
        if final_path != self.output_path:
            self.output_path.replace(final_path)
            self.output_path = final_path

    def request_shutdown(self) -> None:
        if self.shutdown_timer is not None:
            self.shutdown_timer.cancel()
        if rclpy.ok():
            rclpy.shutdown()

    def destroy_node(self) -> bool:
        self.close_file()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EthzAllDataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

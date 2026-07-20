#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros


def get_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "tracks" / "src").exists():
            return parent
    return Path.cwd()


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class SlamTfLogger(Node):
    def __init__(self) -> None:
        super().__init__("slam_tf_logger")

        repo_root = get_repo_root()
        default_output = repo_root / "output" / "slam_runs" / "slam_tf_log.csv"

        self.declare_parameter("output_path", str(default_output))
        self.declare_parameter("odom_frame", "odom_1")
        self.declare_parameter("base_frame", "f1tenth_1")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("pose_topic", "/autodrive/f1tenth_1/ips")
        self.declare_parameter("sample_period", 0.05)
        self.declare_parameter("lap_start_radius_m", 1.5)
        self.declare_parameter("lap_min_distance_m", 35.0)

        self.output_path = Path(str(self.get_parameter("output_path").value)).expanduser()
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.sample_period = float(self.get_parameter("sample_period").value)
        self.lap_start_radius_m = float(self.get_parameter("lap_start_radius_m").value)
        self.lap_min_distance_m = float(self.get_parameter("lap_min_distance_m").value)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.output_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "time_sec",
                "odom_to_base_x",
                "odom_to_base_y",
                "odom_to_base_yaw_deg",
                "map_to_odom_x",
                "map_to_odom_y",
                "map_to_odom_yaw_deg",
                "ips_x",
                "ips_y",
                "ips_z",
                "distance_to_start_m",
                "total_distance_m",
                "near_start",
                "left_start_zone",
                "lap_candidate",
            ]
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_pose: tuple[float, float, float] | None = None
        self.start_pose_xy: tuple[float, float] | None = None
        self.prev_pose_xy: tuple[float, float] | None = None
        self.total_distance_m = 0.0
        self.left_start_zone = False
        self.rows_written = 0

        self.create_subscription(Point, self.pose_topic, self.pose_cb, 10)
        self.create_timer(self.sample_period, self.sample_cb)

        self.get_logger().info(
            f"SLAM TF logger ready. output={self.output_path}, "
            f"odom={self.odom_frame}, base={self.base_frame}, map={self.map_frame}"
        )

    def pose_cb(self, msg: Point) -> None:
        self.latest_pose = (float(msg.x), float(msg.y), float(msg.z))
        current_xy = (float(msg.x), float(msg.y))

        if self.start_pose_xy is None:
            self.start_pose_xy = current_xy
            self.prev_pose_xy = current_xy
            self.get_logger().info(f"TF log start pose locked at ({current_xy[0]:.2f}, {current_xy[1]:.2f})")
            return

        if self.prev_pose_xy is not None:
            self.total_distance_m += math.hypot(current_xy[0] - self.prev_pose_xy[0], current_xy[1] - self.prev_pose_xy[1])
        self.prev_pose_xy = current_xy

        distance_to_start = math.hypot(current_xy[0] - self.start_pose_xy[0], current_xy[1] - self.start_pose_xy[1])
        if distance_to_start > self.lap_start_radius_m:
            self.left_start_zone = True

    def sample_cb(self) -> None:
        if self.latest_pose is None:
            return

        try:
            odom_to_base = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame, Time())
            map_to_odom = self.tf_buffer.lookup_transform(self.map_frame, self.odom_frame, Time())
        except Exception:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        ips_x, ips_y, ips_z = self.latest_pose

        if self.start_pose_xy is None:
            distance_to_start = 0.0
            near_start = True
        else:
            distance_to_start = math.hypot(ips_x - self.start_pose_xy[0], ips_y - self.start_pose_xy[1])
            near_start = distance_to_start <= self.lap_start_radius_m

        lap_candidate = self.left_start_zone and self.total_distance_m >= self.lap_min_distance_m and near_start

        ob_t = odom_to_base.transform.translation
        ob_r = odom_to_base.transform.rotation
        mo_t = map_to_odom.transform.translation
        mo_r = map_to_odom.transform.rotation

        self.csv_writer.writerow(
            [
                f"{now_sec:.6f}",
                f"{ob_t.x:.6f}",
                f"{ob_t.y:.6f}",
                f"{math.degrees(quaternion_to_yaw(ob_r.x, ob_r.y, ob_r.z, ob_r.w)):.6f}",
                f"{mo_t.x:.6f}",
                f"{mo_t.y:.6f}",
                f"{math.degrees(quaternion_to_yaw(mo_r.x, mo_r.y, mo_r.z, mo_r.w)):.6f}",
                f"{ips_x:.6f}",
                f"{ips_y:.6f}",
                f"{ips_z:.6f}",
                f"{distance_to_start:.6f}",
                f"{self.total_distance_m:.6f}",
                int(near_start),
                int(self.left_start_zone),
                int(lap_candidate),
            ]
        )
        self.rows_written += 1

        if self.rows_written % 50 == 0:
            self.csv_file.flush()

    def destroy_node(self) -> bool:
        try:
            self.csv_file.flush()
            self.csv_file.close()
            self.get_logger().info(f"Saved {self.rows_written} TF samples to {self.output_path}")
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlamTfLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

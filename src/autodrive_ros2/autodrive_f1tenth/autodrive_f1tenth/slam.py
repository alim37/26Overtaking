#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import struct
from collections import deque
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu, LaserScan, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener


def get_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "package.xml").exists():
            return parent
        if (parent / ".git").exists() or (parent / "tracks" / "src").exists():
            return parent
    return Path(__file__).resolve().parents[1]


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def make_pointcloud2(points: list[tuple[float, float, float]], frame_id: str, stamp) -> PointCloud2:
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    data = bytearray()
    for x, y, z in points:
        data.extend(struct.pack("fff", float(x), float(y), float(z)))
    return PointCloud2(
        header=Header(frame_id=frame_id, stamp=stamp),
        height=1,
        width=len(points),
        fields=fields,
        is_bigendian=False,
        point_step=12,
        row_step=12 * len(points),
        data=bytes(data),
        is_dense=True,
    )


class SlamRecorder(Node):
    def __init__(self) -> None:
        super().__init__("slam_recorder")

        repo_root = get_repo_root()
        default_output = repo_root / "output" / "slam_runs" / "empty_track_baseline.csv"
        default_map_output = repo_root / "output" / "slam_runs" / "empty_track_baseline_map_points.csv"

        self.declare_parameter("pose_topic", "/autodrive/f1tenth_1/ips")
        self.declare_parameter("imu_topic", "/autodrive/f1tenth_1/imu")
        self.declare_parameter("scan_topic", "/autodrive/f1tenth_1/lidar")
        self.declare_parameter("output_path", str(default_output))
        self.declare_parameter("accumulated_map_output_path", str(default_map_output))
        self.declare_parameter("flush_every_scans", 5)
        self.declare_parameter("stop_after_one_lap", True)
        self.declare_parameter("lap_start_radius_m", 1.5)
        self.declare_parameter("lap_min_distance_m", 35.0)
        self.declare_parameter("pose_history_size", 400)
        self.declare_parameter("imu_history_size", 800)
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("use_tf_projection", True)
        self.declare_parameter("require_tf_projection", True)
        self.declare_parameter("current_scan_topic", "/autodrive/f1tenth_1/slam/current_scan")
        self.declare_parameter("accumulated_map_topic", "/autodrive/f1tenth_1/slam/accumulated_map")
        self.declare_parameter("map_cell_size_m", 0.04)
        self.declare_parameter("map_min_hits", 1)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.output_path = Path(str(self.get_parameter("output_path").value)).expanduser()
        self.accumulated_map_output_path = Path(str(self.get_parameter("accumulated_map_output_path").value)).expanduser()
        self.flush_every_scans = int(self.get_parameter("flush_every_scans").value)
        self.stop_after_one_lap = bool(self.get_parameter("stop_after_one_lap").value)
        self.lap_start_radius_m = float(self.get_parameter("lap_start_radius_m").value)
        self.lap_min_distance_m = float(self.get_parameter("lap_min_distance_m").value)
        self.pose_history_size = int(self.get_parameter("pose_history_size").value)
        self.imu_history_size = int(self.get_parameter("imu_history_size").value)
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.use_tf_projection = bool(self.get_parameter("use_tf_projection").value)
        self.require_tf_projection = bool(self.get_parameter("require_tf_projection").value)
        self.current_scan_topic = str(self.get_parameter("current_scan_topic").value)
        self.accumulated_map_topic = str(self.get_parameter("accumulated_map_topic").value)
        self.map_cell_size_m = float(self.get_parameter("map_cell_size_m").value)
        self.map_min_hits = int(self.get_parameter("map_min_hits").value)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.accumulated_map_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.output_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "scan_id",
                "stamp_sec",
                "beam_idx",
                "angle_deg",
                "range_m",
                "local_x_m",
                "local_y_m",
                "yaw_deg",
                "world_x_m",
                "world_y_m",
                "ips_projected_x_m",
                "ips_projected_y_m",
                "ips_x_m",
                "ips_y_m",
                "ips_z_m",
                "frame_id",
                "projection_mode",
            ]
        )

        self.latest_pose: tuple[float, float, float] | None = None
        self.latest_yaw: float | None = None
        self.pose_history: deque[tuple[float, float, float, float]] = deque(maxlen=self.pose_history_size)
        self.yaw_history: deque[tuple[float, float]] = deque(maxlen=self.imu_history_size)
        self.start_pose_xy: tuple[float, float] | None = None
        self.prev_pose_xy: tuple[float, float] | None = None
        self.total_distance_m = 0.0
        self.left_start_zone = False
        self.completed_lap = False
        self.scan_id = 0
        self.rows_written = 0
        self.skipped_scan_count = 0
        self.map_cells: dict[tuple[int, int], tuple[float, float, int]] = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.current_scan_pub = self.create_publisher(PointCloud2, self.current_scan_topic, 10)
        self.accumulated_map_pub = self.create_publisher(PointCloud2, self.accumulated_map_topic, 10)

        self.create_subscription(Point, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(Imu, self.imu_topic, self.imu_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)

        self.get_logger().info(
            f"SLAM recorder ready. pose={self.pose_topic}, imu={self.imu_topic}, scan={self.scan_topic}, "
            f"output={self.output_path}, map_output={self.accumulated_map_output_path}, "
            f"current_scan={self.current_scan_topic}, accumulated_map={self.accumulated_map_topic}, "
            f"use_tf_projection={self.use_tf_projection}, require_tf_projection={self.require_tf_projection}"
        )

    def lookup_scan_transform(self, msg: LaserScan):
        if not self.use_tf_projection:
            return None
        try:
            scan_time = Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec)
            return self.tf_buffer.lookup_transform(self.world_frame, msg.header.frame_id, scan_time)
        except TransformException:
            return None

    def transform_local_point(self, transform, local_x: float, local_y: float) -> tuple[float, float] | None:
        if transform is None:
            return None
        tx = float(transform.transform.translation.x)
        ty = float(transform.transform.translation.y)
        q = transform.transform.rotation
        yaw = quaternion_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        world_x = tx + local_x * cos_yaw - local_y * sin_yaw
        world_y = ty + local_x * sin_yaw + local_y * cos_yaw
        return world_x, world_y

    def pose_cb(self, msg: Point) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        self.latest_pose = (float(msg.x), float(msg.y), float(msg.z))
        self.pose_history.append((now, float(msg.x), float(msg.y), float(msg.z)))
        current_xy = (float(msg.x), float(msg.y))

        if self.start_pose_xy is None:
            self.start_pose_xy = current_xy
            self.prev_pose_xy = current_xy
            self.get_logger().info(f"Lap start pose locked at ({current_xy[0]:.2f}, {current_xy[1]:.2f})")
            return

        if self.prev_pose_xy is not None:
            self.total_distance_m += math.hypot(current_xy[0] - self.prev_pose_xy[0], current_xy[1] - self.prev_pose_xy[1])
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
            self.get_logger().info(
                f"Completed one lap: traveled {self.total_distance_m:.2f} m and returned within "
                f"{distance_to_start:.2f} m of the start. Stopping recorder."
            )
            self.csv_file.flush()
            raise SystemExit

    def imu_cb(self, msg: Imu) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        self.latest_yaw = quaternion_to_yaw(
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        )
        self.yaw_history.append((now, self.latest_yaw))

    def interpolate_pose(self, query_time_sec: float) -> tuple[float, float, float] | None:
        if not self.pose_history:
            return self.latest_pose
        if len(self.pose_history) == 1:
            _, x, y, z = self.pose_history[0]
            return (x, y, z)

        history = list(self.pose_history)
        if query_time_sec <= history[0][0]:
            _, x, y, z = history[0]
            return (x, y, z)
        if query_time_sec >= history[-1][0]:
            _, x, y, z = history[-1]
            return (x, y, z)

        for idx in range(1, len(history)):
            t1, x1, y1, z1 = history[idx]
            t0, x0, y0, z0 = history[idx - 1]
            if query_time_sec <= t1:
                alpha = (query_time_sec - t0) / max(t1 - t0, 1e-9)
                return (
                    (1.0 - alpha) * x0 + alpha * x1,
                    (1.0 - alpha) * y0 + alpha * y1,
                    (1.0 - alpha) * z0 + alpha * z1,
                )
        return self.latest_pose

    def interpolate_yaw(self, query_time_sec: float) -> float | None:
        if not self.yaw_history:
            return self.latest_yaw
        if len(self.yaw_history) == 1:
            return self.yaw_history[0][1]

        history = list(self.yaw_history)
        if query_time_sec <= history[0][0]:
            return history[0][1]
        if query_time_sec >= history[-1][0]:
            return history[-1][1]

        for idx in range(1, len(history)):
            t1, yaw1 = history[idx]
            t0, yaw0 = history[idx - 1]
            if query_time_sec <= t1:
                alpha = (query_time_sec - t0) / max(t1 - t0, 1e-9)
                dyaw = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))
                return yaw0 + alpha * dyaw
        return self.latest_yaw

    def scan_cb(self, msg: LaserScan) -> None:
        receipt_sec = self.get_clock().now().nanoseconds * 1e-9
        pose = self.interpolate_pose(receipt_sec)
        yaw = self.interpolate_yaw(receipt_sec)

        if pose is None or yaw is None:
            return

        ips_x, ips_y, ips_z = pose
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        scan_transform = self.lookup_scan_transform(msg)
        projection_mode = "tf_map" if scan_transform is not None else "ips_imu"
        if self.require_tf_projection and scan_transform is None:
            self.skipped_scan_count += 1
            if self.skipped_scan_count % 20 == 0:
                self.get_logger().warn(
                    f"Skipped {self.skipped_scan_count} scans waiting for TF projection into {self.world_frame}"
                )
            return
        angle = msg.angle_min
        current_scan_points: list[tuple[float, float, float]] = []

        for beam_idx, distance in enumerate(msg.ranges):
            if math.isfinite(distance) and msg.range_min <= distance <= msg.range_max:
                local_x = distance * math.cos(angle)
                local_y = distance * math.sin(angle)
                ips_world_x = ips_x + local_x * cos_yaw - local_y * sin_yaw
                ips_world_y = ips_y + local_x * sin_yaw + local_y * cos_yaw
                tf_world_xy = self.transform_local_point(scan_transform, local_x, local_y)
                if tf_world_xy is not None:
                    world_x, world_y = tf_world_xy
                else:
                    world_x, world_y = ips_world_x, ips_world_y
                current_scan_points.append((world_x, world_y, 0.0))
                self.update_map_cell(world_x, world_y)
                self.csv_writer.writerow(
                    [
                        self.scan_id,
                        f"{stamp_sec:.9f}",
                        beam_idx,
                        f"{math.degrees(angle):.6f}",
                        f"{distance:.6f}",
                        f"{local_x:.6f}",
                        f"{local_y:.6f}",
                        f"{math.degrees(yaw):.6f}",
                        f"{world_x:.6f}",
                        f"{world_y:.6f}",
                        f"{ips_world_x:.6f}",
                        f"{ips_world_y:.6f}",
                        f"{ips_x:.6f}",
                        f"{ips_y:.6f}",
                        f"{ips_z:.6f}",
                        self.world_frame,
                        projection_mode,
                    ]
                )
                self.rows_written += 1
            angle += msg.angle_increment

        self.current_scan_pub.publish(make_pointcloud2(current_scan_points, self.world_frame, msg.header.stamp))
        self.accumulated_map_pub.publish(
            make_pointcloud2(self.accumulated_map_points(), self.world_frame, msg.header.stamp)
        )

        self.scan_id += 1
        if self.flush_every_scans > 0 and self.scan_id % self.flush_every_scans == 0:
            self.csv_file.flush()

        if self.scan_id % 20 == 0:
            self.get_logger().info(
                f"Recorded {self.scan_id} scans and {self.rows_written} valid LiDAR points to {self.output_path}"
            )

    def update_map_cell(self, world_x: float, world_y: float) -> None:
        cell = (
            int(math.floor(world_x / self.map_cell_size_m)),
            int(math.floor(world_y / self.map_cell_size_m)),
        )
        if cell not in self.map_cells:
            self.map_cells[cell] = (world_x, world_y, 1)
            return

        prev_x, prev_y, hits = self.map_cells[cell]
        next_hits = hits + 1
        avg_x = prev_x + (world_x - prev_x) / next_hits
        avg_y = prev_y + (world_y - prev_y) / next_hits
        self.map_cells[cell] = (avg_x, avg_y, next_hits)

    def accumulated_map_points(self) -> list[tuple[float, float, float]]:
        return [
            (x, y, 0.0)
            for x, y, hits in self.map_cells.values()
            if hits >= self.map_min_hits
        ]

    def save_accumulated_map_csv(self) -> None:
        with self.accumulated_map_output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["cell_x", "cell_y", "world_x_m", "world_y_m", "hits"])
            for cell_x, cell_y in sorted(self.map_cells.keys()):
                world_x, world_y, hits = self.map_cells[(cell_x, cell_y)]
                if hits < self.map_min_hits:
                    continue
                writer.writerow(
                    [
                        cell_x,
                        cell_y,
                        f"{world_x:.6f}",
                        f"{world_y:.6f}",
                        hits,
                    ]
                )

    def destroy_node(self) -> bool:
        try:
            self.csv_file.flush()
            self.csv_file.close()
            self.save_accumulated_map_csv()
            self.get_logger().info(
                f"Saved SLAM baseline with {self.scan_id} scans and {self.rows_written} points to {self.output_path}; "
                f"saved accumulated map points to {self.accumulated_map_output_path}; "
                f"skipped {self.skipped_scan_count} scans without TF projection"
            )
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlamRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

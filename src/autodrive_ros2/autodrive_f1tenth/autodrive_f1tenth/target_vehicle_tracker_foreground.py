#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point, PointStamped, Vector3Stamped
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan, PointCloud2, PointField
from std_msgs.msg import Bool, Float32, Header
from visualization_msgs.msg import Marker


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


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "tracks" / "src").exists():
            return parent
    return Path.cwd()


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class ForegroundCandidate:
    points: list[tuple[float, float, float, float, float, int]]
    centroid_x: float
    centroid_y: float
    span_x: float
    span_y: float
    radial_distance: float
    beam_count: int


@dataclass
class TrackedTarget:
    stamp_sec: float
    centroid_x: float
    centroid_y: float


class TargetVehicleTrackerForeground(Node):
    def __init__(self) -> None:
        super().__init__("target_vehicle_tracker_foreground")

        repo_root = find_repo_root()
        self.declare_parameter("scan_topic", "/autodrive/f1tenth_1/lidar")
        self.declare_parameter("pose_topic", "/autodrive/f1tenth_1/ips")
        self.declare_parameter("imu_topic", "/autodrive/f1tenth_1/imu")
        self.declare_parameter("target_pose_topic", "/autodrive/f1tenth_2/ips")
        self.declare_parameter(
            "wall_map_csv_path",
            str(repo_root / "output" / "slam_runs" / "slam_toolbox_boundary.csv"),
        )
        self.declare_parameter("map_cell_size_m", 0.05)
        self.declare_parameter("ray_step_m", 0.05)
        self.declare_parameter("foreground_margin_m", 0.22)
        self.declare_parameter("front_only", True)
        self.declare_parameter("min_range", 0.10)
        self.declare_parameter("max_range", 12.0)
        self.declare_parameter("min_target_distance", 0.4)
        self.declare_parameter("max_target_distance", 8.0)
        self.declare_parameter("min_target_width", 0.05)
        self.declare_parameter("max_target_width", 1.20)
        self.declare_parameter("min_target_length", 0.05)
        self.declare_parameter("max_target_length", 1.50)
        self.declare_parameter("min_target_beam_count", 2)
        self.declare_parameter("max_target_beam_count", 40)
        self.declare_parameter("cluster_distance_threshold", 0.30)
        self.declare_parameter("cluster_distance_scale", 2.0)
        self.declare_parameter("target_match_distance", 1.2)
        self.declare_parameter("target_timeout_sec", 0.8)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("dynamic_cloud_topic", "/autodrive/f1tenth_1/target_tracker_fg/dynamic_cloud")
        self.declare_parameter("wall_reference_cloud_topic", "/autodrive/f1tenth_1/target_tracker_fg/wall_cloud")
        self.declare_parameter("raw_cluster_centroids_topic", "/autodrive/f1tenth_1/target_tracker_fg/raw_cluster_centroids")
        self.declare_parameter("tracking_arrow_topic", "/autodrive/f1tenth_1/target_tracker_fg/tracking_arrow")
        self.declare_parameter("tracking_vector_topic", "/autodrive/f1tenth_1/target_tracker_fg/tracking_vector")
        self.declare_parameter("target_point_topic", "/autodrive/f1tenth_1/target_tracker_fg/target_point")
        self.declare_parameter("tracking_visible_topic", "/autodrive/f1tenth_1/target_tracker_fg/target_visible")
        self.declare_parameter("tracking_confidence_topic", "/autodrive/f1tenth_1/target_tracker_fg/tracking_confidence")
        self.declare_parameter(
            "debug_log_path",
            str(repo_root / "output" / "slam_runs" / "target_tracker_foreground_debug.csv"),
        )

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self.wall_map_csv_path = Path(str(self.get_parameter("wall_map_csv_path").value)).expanduser()
        self.map_cell_size_m = float(self.get_parameter("map_cell_size_m").value)
        self.ray_step_m = float(self.get_parameter("ray_step_m").value)
        self.foreground_margin_m = float(self.get_parameter("foreground_margin_m").value)
        self.front_only = bool(self.get_parameter("front_only").value)
        self.min_range = float(self.get_parameter("min_range").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.min_target_distance = float(self.get_parameter("min_target_distance").value)
        self.max_target_distance = float(self.get_parameter("max_target_distance").value)
        self.min_target_width = float(self.get_parameter("min_target_width").value)
        self.max_target_width = float(self.get_parameter("max_target_width").value)
        self.min_target_length = float(self.get_parameter("min_target_length").value)
        self.max_target_length = float(self.get_parameter("max_target_length").value)
        self.min_target_beam_count = int(self.get_parameter("min_target_beam_count").value)
        self.max_target_beam_count = int(self.get_parameter("max_target_beam_count").value)
        self.cluster_distance_threshold = float(self.get_parameter("cluster_distance_threshold").value)
        self.cluster_distance_scale = float(self.get_parameter("cluster_distance_scale").value)
        self.target_match_distance = float(self.get_parameter("target_match_distance").value)
        self.target_timeout_sec = float(self.get_parameter("target_timeout_sec").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.dynamic_cloud_topic = str(self.get_parameter("dynamic_cloud_topic").value)
        self.wall_reference_cloud_topic = str(self.get_parameter("wall_reference_cloud_topic").value)
        self.raw_cluster_centroids_topic = str(self.get_parameter("raw_cluster_centroids_topic").value)
        self.tracking_arrow_topic = str(self.get_parameter("tracking_arrow_topic").value)
        self.tracking_vector_topic = str(self.get_parameter("tracking_vector_topic").value)
        self.target_point_topic = str(self.get_parameter("target_point_topic").value)
        self.tracking_visible_topic = str(self.get_parameter("tracking_visible_topic").value)
        self.tracking_confidence_topic = str(self.get_parameter("tracking_confidence_topic").value)
        self.debug_log_path = Path(str(self.get_parameter("debug_log_path").value)).expanduser()

        self.wall_cells = self.load_wall_cells()
        self.ego_position: tuple[float, float] | None = None
        self.target_truth_position: tuple[float, float] | None = None
        self.ego_yaw: float | None = None
        self.scan_frame_id = "lidar_1"
        self._last_angle_increment = 0.0
        self.current_target: TrackedTarget | None = None
        self.current_confidence = 0.0
        self.latest_dynamic_points: list[tuple[float, float, float]] = []
        self.latest_wall_points: list[tuple[float, float, float]] = []
        self.latest_cluster_centroids: list[tuple[float, float, float]] = []
        self.last_foreground_beam_count = 0
        self.last_cluster_count = 0

        self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.debug_file = self.debug_log_path.open("w", newline="", encoding="utf-8")
        self.debug_writer = csv.writer(self.debug_file)
        self.debug_writer.writerow(
            [
                "stamp_sec",
                "ego_x",
                "ego_y",
                "ego_yaw_deg",
                "target_truth_x",
                "target_truth_y",
                "foreground_beam_count",
                "cluster_count",
                "target_detected",
                "detected_local_x",
                "detected_local_y",
                "detected_world_x",
                "detected_world_y",
                "confidence",
            ]
        )

        self.dynamic_pub = self.create_publisher(PointCloud2, self.dynamic_cloud_topic, 10)
        self.wall_pub = self.create_publisher(PointCloud2, self.wall_reference_cloud_topic, 10)
        self.raw_cluster_pub = self.create_publisher(PointCloud2, self.raw_cluster_centroids_topic, 10)
        self.arrow_pub = self.create_publisher(Marker, self.tracking_arrow_topic, 10)
        self.vector_pub = self.create_publisher(Vector3Stamped, self.tracking_vector_topic, 10)
        self.target_point_pub = self.create_publisher(PointStamped, self.target_point_topic, 10)
        self.visible_pub = self.create_publisher(Bool, self.tracking_visible_topic, 10)
        self.confidence_pub = self.create_publisher(Float32, self.tracking_confidence_topic, 10)

        self.create_subscription(Point, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(Point, self.target_pose_topic, self.target_pose_cb, 10)
        self.create_subscription(Imu, self.imu_topic, self.imu_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.create_timer(1.0 / max(self.publish_rate_hz, 1e-3), self.publish_outputs)

        self.get_logger().info(
            f"Foreground tracker ready. scan={self.scan_topic}, pose={self.pose_topic}, imu={self.imu_topic}, "
            f"wall_map={self.wall_map_csv_path}"
        )

    def load_wall_cells(self) -> set[tuple[int, int]]:
        csv_path = self.wall_map_csv_path
        fallback = find_repo_root() / "ros2" / "install" / "autodrive_f1tenth" / "lib" / "output" / "slam_runs" / csv_path.name
        if not csv_path.exists() and fallback.exists():
            csv_path = fallback
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find wall map CSV at {self.wall_map_csv_path}")

        cells: set[tuple[int, int]] = set()
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                world_x = float(row["world_x_m"])
                world_y = float(row["world_y_m"])
                cells.add(self.world_to_cell(world_x, world_y))
        self.get_logger().info(f"Loaded wall map cells={len(cells)} from {csv_path}")
        return cells

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(math.floor(x / self.map_cell_size_m)),
            int(math.floor(y / self.map_cell_size_m)),
        )

    def pose_cb(self, msg: Point) -> None:
        self.ego_position = (float(msg.x), float(msg.y))

    def target_pose_cb(self, msg: Point) -> None:
        self.target_truth_position = (float(msg.x), float(msg.y))

    def imu_cb(self, msg: Imu) -> None:
        self.ego_yaw = quaternion_to_yaw(
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        )

    def local_to_world(self, local_x: float, local_y: float) -> tuple[float, float] | None:
        if self.ego_position is None or self.ego_yaw is None:
            return None
        cos_yaw = math.cos(self.ego_yaw)
        sin_yaw = math.sin(self.ego_yaw)
        return (
            self.ego_position[0] + local_x * cos_yaw - local_y * sin_yaw,
            self.ego_position[1] + local_x * sin_yaw + local_y * cos_yaw,
        )

    def scan_cb(self, msg: LaserScan) -> None:
        self.scan_frame_id = msg.header.frame_id or "lidar_1"
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.ego_position is None or self.ego_yaw is None:
            return

        self._last_angle_increment = float(msg.angle_increment)
        angle = float(msg.angle_min)
        foreground_points: list[tuple[float, float, float, float, float, int]] = []
        wall_points: list[tuple[float, float, float]] = []
        all_foreground_groups: list[list[tuple[float, float, float, float, float, int]]] = []
        current_group: list[tuple[float, float, float, float, float, int]] = []

        for beam_idx, measured_range in enumerate(msg.ranges):
            if not math.isfinite(measured_range) or measured_range < self.min_range or measured_range > self.max_range:
                angle += msg.angle_increment
                if current_group:
                    all_foreground_groups.append(current_group)
                    current_group = []
                continue

            local_x = measured_range * math.cos(angle)
            local_y = measured_range * math.sin(angle)
            if self.front_only and local_x <= 0.0:
                angle += msg.angle_increment
                if current_group:
                    all_foreground_groups.append(current_group)
                    current_group = []
                continue

            expected_range = self.expected_wall_range(angle, msg.range_max)
            if math.isfinite(expected_range) and measured_range < expected_range - self.foreground_margin_m:
                point = (local_x, local_y, 0.0, math.degrees(angle), measured_range, beam_idx)
                foreground_points.append(point)
                if current_group and beam_idx == current_group[-1][5] + 1:
                    current_group.append(point)
                else:
                    if current_group:
                        all_foreground_groups.append(current_group)
                    current_group = [point]
            else:
                wall_points.append((local_x, local_y, 0.0))
                if current_group:
                    all_foreground_groups.append(current_group)
                    current_group = []

            angle += msg.angle_increment

        if current_group:
            all_foreground_groups.append(current_group)

        candidates = [candidate for candidate in (self.to_candidate(group) for group in all_foreground_groups) if candidate is not None]
        self.latest_dynamic_points = [(p[0], p[1], p[2]) for p in foreground_points]
        self.latest_wall_points = wall_points
        self.latest_cluster_centroids = [(c.centroid_x, c.centroid_y, 0.0) for c in candidates]
        self.last_foreground_beam_count = len(foreground_points)
        self.last_cluster_count = len(candidates)

        chosen = self.select_target(candidates, stamp_sec)
        if chosen is None:
            self.current_confidence = 0.0
            self.append_debug_row(stamp_sec, None)
            return

        self.current_target = TrackedTarget(stamp_sec=stamp_sec, centroid_x=chosen.centroid_x, centroid_y=chosen.centroid_y)
        self.current_confidence = self.compute_confidence(chosen)
        self.append_debug_row(stamp_sec, chosen)

    def expected_wall_range(self, local_angle: float, max_range: float) -> float:
        if self.ego_position is None or self.ego_yaw is None:
            return float("inf")

        world_angle = self.ego_yaw + local_angle
        cos_angle = math.cos(world_angle)
        sin_angle = math.sin(world_angle)
        step = max(self.ray_step_m, self.map_cell_size_m * 0.5)
        distance = self.min_range
        while distance <= max_range:
            world_x = self.ego_position[0] + distance * cos_angle
            world_y = self.ego_position[1] + distance * sin_angle
            if self.world_to_cell(world_x, world_y) in self.wall_cells:
                return distance
            distance += step
        return float("inf")

    def to_candidate(
        self,
        group: list[tuple[float, float, float, float, float, int]],
    ) -> ForegroundCandidate | None:
        if len(group) < self.min_target_beam_count or len(group) > self.max_target_beam_count:
            return None
        xs = [p[0] for p in group]
        ys = [p[1] for p in group]
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)
        if centroid_x <= 0.0:
            return None
        radial_distance = math.hypot(centroid_x, centroid_y)
        if radial_distance < self.min_target_distance or radial_distance > self.max_target_distance:
            return None
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        width = min(span_x, span_y)
        length = max(span_x, span_y)
        if width < self.min_target_width or width > self.max_target_width:
            return None
        if length < self.min_target_length or length > self.max_target_length:
            return None
        return ForegroundCandidate(
            points=group,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            span_x=span_x,
            span_y=span_y,
            radial_distance=radial_distance,
            beam_count=len(group),
        )

    def select_target(self, candidates: list[ForegroundCandidate], stamp_sec: float) -> ForegroundCandidate | None:
        if not candidates:
            if self.current_target is not None and stamp_sec - self.current_target.stamp_sec <= self.target_timeout_sec:
                return ForegroundCandidate(
                    points=[],
                    centroid_x=self.current_target.centroid_x,
                    centroid_y=self.current_target.centroid_y,
                    span_x=0.0,
                    span_y=0.0,
                    radial_distance=math.hypot(self.current_target.centroid_x, self.current_target.centroid_y),
                    beam_count=0,
                )
            self.current_target = None
            return None

        if self.current_target is None or stamp_sec - self.current_target.stamp_sec > self.target_timeout_sec:
            return min(candidates, key=lambda c: (abs(c.centroid_y), c.radial_distance))

        best_candidate = None
        best_distance = float("inf")
        for candidate in candidates:
            jump = math.hypot(
                candidate.centroid_x - self.current_target.centroid_x,
                candidate.centroid_y - self.current_target.centroid_y,
            )
            if jump < best_distance:
                best_distance = jump
                best_candidate = candidate
        if best_candidate is None or best_distance > self.target_match_distance:
            return ForegroundCandidate(
                points=[],
                centroid_x=self.current_target.centroid_x,
                centroid_y=self.current_target.centroid_y,
                span_x=0.0,
                span_y=0.0,
                radial_distance=math.hypot(self.current_target.centroid_x, self.current_target.centroid_y),
                beam_count=0,
            )
        return best_candidate

    def compute_confidence(self, candidate: ForegroundCandidate) -> float:
        beam_score = max(0.0, min(1.0, candidate.beam_count / max(self.max_target_beam_count * 0.4, 1.0)))
        center_score = max(0.0, 1.0 - abs(candidate.centroid_y) / 2.5)
        distance_score = max(0.0, 1.0 - candidate.radial_distance / max(self.max_target_distance, 1e-6))
        return min(0.99, max(0.0, 0.4 * beam_score + 0.35 * center_score + 0.25 * distance_score))

    def append_debug_row(self, stamp_sec: float, target: ForegroundCandidate | None) -> None:
        ego_x = self.ego_position[0] if self.ego_position is not None else float("nan")
        ego_y = self.ego_position[1] if self.ego_position is not None else float("nan")
        ego_yaw_deg = math.degrees(self.ego_yaw) if self.ego_yaw is not None else float("nan")
        truth_x = self.target_truth_position[0] if self.target_truth_position is not None else float("nan")
        truth_y = self.target_truth_position[1] if self.target_truth_position is not None else float("nan")
        detected = 1 if target is not None else 0
        local_x = target.centroid_x if target is not None else float("nan")
        local_y = target.centroid_y if target is not None else float("nan")
        world_x = float("nan")
        world_y = float("nan")
        if target is not None:
            world_xy = self.local_to_world(target.centroid_x, target.centroid_y)
            if world_xy is not None:
                world_x, world_y = world_xy
        self.debug_writer.writerow(
            [
                f"{stamp_sec:.6f}",
                f"{ego_x:.6f}",
                f"{ego_y:.6f}",
                f"{ego_yaw_deg:.6f}",
                f"{truth_x:.6f}",
                f"{truth_y:.6f}",
                self.last_foreground_beam_count,
                self.last_cluster_count,
                detected,
                f"{local_x:.6f}",
                f"{local_y:.6f}",
                f"{world_x:.6f}",
                f"{world_y:.6f}",
                f"{self.current_confidence:.6f}",
            ]
        )
        self.debug_file.flush()

    def publish_outputs(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.dynamic_pub.publish(make_pointcloud2(self.latest_dynamic_points, self.scan_frame_id, stamp))
        self.wall_pub.publish(make_pointcloud2(self.latest_wall_points, self.scan_frame_id, stamp))
        self.raw_cluster_pub.publish(make_pointcloud2(self.latest_cluster_centroids, self.scan_frame_id, stamp))
        self.confidence_pub.publish(Float32(data=float(self.current_confidence)))

        visible = Bool(data=self.current_target is not None)
        self.visible_pub.publish(visible)

        arrow = Marker()
        arrow.header.frame_id = self.scan_frame_id
        arrow.header.stamp = stamp
        arrow.ns = "target_tracker_fg"
        arrow.id = 1
        arrow.type = Marker.ARROW
        arrow.scale.x = 0.08
        arrow.scale.y = 0.16
        arrow.scale.z = 0.20
        arrow.color.a = 1.0

        label = Marker()
        label.header.frame_id = self.scan_frame_id
        label.header.stamp = stamp
        label.ns = "target_tracker_fg"
        label.id = 2
        label.type = Marker.TEXT_VIEW_FACING
        label.scale.z = 0.35
        label.color.a = 1.0

        if self.current_target is None:
            arrow.action = Marker.DELETE
            label.action = Marker.DELETE
            self.arrow_pub.publish(arrow)
            self.arrow_pub.publish(label)
            return

        arrow.action = Marker.ADD
        arrow.color.r = 0.1
        arrow.color.g = 1.0
        arrow.color.b = 0.1
        origin = Point(x=0.0, y=0.0, z=0.0)
        target = Point(x=self.current_target.centroid_x, y=self.current_target.centroid_y, z=0.0)
        arrow.points = [origin, target]
        self.arrow_pub.publish(arrow)

        label.action = Marker.ADD
        label.color.r = arrow.color.r
        label.color.g = arrow.color.g
        label.color.b = arrow.color.b
        label.pose.position.x = target.x + 0.20
        label.pose.position.y = target.y + 0.20
        label.pose.position.z = 0.20
        label.pose.orientation.w = 1.0
        label.text = f"{self.current_confidence:.2f}"
        self.arrow_pub.publish(label)

        point_msg = PointStamped()
        point_msg.header.frame_id = self.scan_frame_id
        point_msg.header.stamp = stamp
        point_msg.point = target
        self.target_point_pub.publish(point_msg)

        vector_msg = Vector3Stamped()
        vector_msg.header.frame_id = self.scan_frame_id
        vector_msg.header.stamp = stamp
        norm = max(math.hypot(target.x, target.y), 1e-6)
        vector_msg.vector.x = target.x / norm
        vector_msg.vector.y = target.y / norm
        vector_msg.vector.z = 0.0
        self.vector_pub.publish(vector_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TargetVehicleTrackerForeground()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.debug_file.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

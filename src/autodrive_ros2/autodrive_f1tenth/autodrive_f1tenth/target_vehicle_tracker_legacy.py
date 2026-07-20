#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped, Vector3Stamped
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan, PointCloud2, PointField
from std_msgs.msg import Bool, Float32
from std_msgs.msg import Header
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


def make_cloud(points: list[tuple[float, float, float]], frame_id: str, stamp) -> PointCloud2:
    return make_pointcloud2(points, frame_id, stamp)


@dataclass
class ClusterCandidate:
    points: list[tuple[float, float, float, float, float, int]]
    centroid_x: float
    centroid_y: float
    radial_distance: float
    span_x: float
    span_y: float
    range_span: float
    beam_count: int
    foreground_strength: float


@dataclass
class TargetState:
    stamp_sec: float
    centroid_x: float
    centroid_y: float
    velocity_x: float
    velocity_y: float
    lock_frames: int


@dataclass
class BaselineScan:
    scan_id: int
    pose_x: float
    pose_y: float
    yaw_deg: float
    ranges: np.ndarray


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "package.xml").exists():
            return parent
        if (parent / ".git").exists() or (parent / "tracks" / "src").exists():
            return parent
    return Path(__file__).resolve().parents[1]


class TargetVehicleTracker(Node):
    """
    Baseline-scan target tracker.

    - use the empty-track baseline CSV recorded by `slam.py`
    - find the nearest baseline scan for the current ego IPS pose + IMU heading
    - compare live LiDAR beams directly against that empty-track reference
    - if a live beam is significantly closer than baseline, treat it as foreground
    - group consecutive foreground beams and track the most plausible car cluster
    """

    def __init__(self) -> None:
        super().__init__("target_vehicle_tracker")

        self.declare_parameter("scan_topic", "/autodrive/f1tenth_1/lidar")
        self.declare_parameter("pose_topic", "/autodrive/f1tenth_1/ips")
        self.declare_parameter("target_pose_topic", "/autodrive/f1tenth_2/ips")
        self.declare_parameter("imu_topic", "/autodrive/f1tenth_1/imu")
        self.declare_parameter("front_only", True)
        self.declare_parameter("forward_angle_window_deg", 5.0)
        self.declare_parameter("min_range", 0.10)
        self.declare_parameter("max_range", 12.0)
        repo_root = find_repo_root()
        self.declare_parameter("baseline_csv_path", str(repo_root / "output" / "slam_runs" / "empty_track_baseline.csv"))
        self.declare_parameter("baseline_position_match_m", 1.5)
        self.declare_parameter("baseline_yaw_match_deg", 25.0)
        self.declare_parameter("baseline_range_margin_m", 0.25)
        self.declare_parameter("baseline_fit_negative_residual_weight", 0.20)
        self.declare_parameter("baseline_fit_positive_residual_weight", 1.00)
        self.declare_parameter("baseline_fit_pose_weight", 0.15)
        self.declare_parameter(
            "track_wall_csv_path",
            str(repo_root / "output" / "slam_runs" / "slam_toolbox_boundary.csv"),
        )
        self.declare_parameter(
            "boundary_labeled_csv_path",
            str(repo_root / "output" / "slam_runs" / "slam_toolbox_boundary_labeled.csv"),
        )
        self.declare_parameter(
            "inner_boundary_csv_path",
            str(repo_root / "output" / "slam_runs" / "slam_toolbox_boundary_inner.csv"),
        )
        self.declare_parameter(
            "outer_boundary_csv_path",
            str(repo_root / "output" / "slam_runs" / "slam_toolbox_boundary_outer.csv"),
        )
        self.declare_parameter("boundary_cell_size_m", 0.05)
        self.declare_parameter("boundary_match_radius_m", 0.12)
        self.declare_parameter("min_hits_threshold", 0)
        self.declare_parameter("wall_reference_cloud_topic", "/autodrive/f1tenth_1/target_tracker/wall_reference_cloud")
        self.declare_parameter("cluster_distance_threshold", 0.30)
        self.declare_parameter("cluster_distance_scale", 2.0)
        self.declare_parameter("min_cluster_points", 2)
        self.declare_parameter("min_target_distance", 0.4)
        self.declare_parameter("max_target_distance", 8.0)
        self.declare_parameter("min_target_width", 0.10)
        self.declare_parameter("max_target_width", 1.20)
        self.declare_parameter("min_target_beam_count", 2)
        self.declare_parameter("max_target_beam_count", 40)
        self.declare_parameter("preferred_target_beam_count", 20)
        self.declare_parameter("max_target_range_span_m", 0.60)
        self.declare_parameter("target_match_distance", 1.0)
        self.declare_parameter("target_timeout_sec", 0.8)
        self.declare_parameter("target_lock_distance", 0.9)
        self.declare_parameter("target_search_switch_distance", 0.6)
        self.declare_parameter("target_lock_required_frames", 3)
        self.declare_parameter("target_hold_confidence_floor", 0.55)
        self.declare_parameter("target_hold_decay", 0.92)
        self.declare_parameter("dynamic_cloud_topic", "/autodrive/f1tenth_1/target_tracker/dynamic_cloud")
        self.declare_parameter("wall_filtered_cloud_topic", "/autodrive/f1tenth_1/target_tracker/wall_filtered_cloud")
        self.declare_parameter("candidate_cloud_topic", "/autodrive/f1tenth_1/target_tracker/candidate_cloud")
        self.declare_parameter("raw_cluster_centroids_topic", "/autodrive/f1tenth_1/target_tracker/raw_cluster_centroids")
        self.declare_parameter("target_point_topic", "/autodrive/f1tenth_1/target_tracker/target_point")
        self.declare_parameter("tracking_vector_topic", "/autodrive/f1tenth_1/target_tracker/tracking_vector")
        self.declare_parameter("tracking_visible_topic", "/autodrive/f1tenth_1/target_tracker/target_visible")
        self.declare_parameter("tracking_arrow_topic", "/autodrive/f1tenth_1/target_tracker/tracking_arrow")
        self.declare_parameter("tracking_confidence_topic", "/autodrive/f1tenth_1/target_tracker/tracking_confidence")
        self.declare_parameter("follow_active_topic", "/autodrive/f1tenth_1/target_tracker/follow_active")
        self.declare_parameter("confidence_green_threshold", 0.80)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter(
            "debug_log_path",
            str(repo_root / "output" / "slam_runs" / "target_tracker_debug.csv"),
        )

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.front_only = bool(self.get_parameter("front_only").value)
        self.forward_angle_window_deg = float(self.get_parameter("forward_angle_window_deg").value)
        self.min_range = float(self.get_parameter("min_range").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.baseline_csv_path = Path(str(self.get_parameter("baseline_csv_path").value)).expanduser()
        self.baseline_position_match_m = float(self.get_parameter("baseline_position_match_m").value)
        self.baseline_yaw_match_deg = float(self.get_parameter("baseline_yaw_match_deg").value)
        self.baseline_range_margin_m = float(self.get_parameter("baseline_range_margin_m").value)
        self.baseline_fit_negative_residual_weight = float(
            self.get_parameter("baseline_fit_negative_residual_weight").value
        )
        self.baseline_fit_positive_residual_weight = float(
            self.get_parameter("baseline_fit_positive_residual_weight").value
        )
        self.baseline_fit_pose_weight = float(self.get_parameter("baseline_fit_pose_weight").value)
        self.track_wall_csv_path = Path(str(self.get_parameter("track_wall_csv_path").value)).expanduser()
        self.boundary_labeled_csv_path = Path(str(self.get_parameter("boundary_labeled_csv_path").value)).expanduser()
        self.inner_boundary_csv_path = Path(str(self.get_parameter("inner_boundary_csv_path").value)).expanduser()
        self.outer_boundary_csv_path = Path(str(self.get_parameter("outer_boundary_csv_path").value)).expanduser()
        self.boundary_cell_size_m = float(self.get_parameter("boundary_cell_size_m").value)
        self.boundary_match_radius_m = float(self.get_parameter("boundary_match_radius_m").value)
        self.min_hits_threshold = int(self.get_parameter("min_hits_threshold").value)
        self.wall_reference_cloud_topic = str(self.get_parameter("wall_reference_cloud_topic").value)
        self.cluster_distance_threshold = float(self.get_parameter("cluster_distance_threshold").value)
        self.cluster_distance_scale = float(self.get_parameter("cluster_distance_scale").value)
        self.min_cluster_points = int(self.get_parameter("min_cluster_points").value)
        self.min_target_distance = float(self.get_parameter("min_target_distance").value)
        self.max_target_distance = float(self.get_parameter("max_target_distance").value)
        self.min_target_width = float(self.get_parameter("min_target_width").value)
        self.max_target_width = float(self.get_parameter("max_target_width").value)
        self.min_target_beam_count = int(self.get_parameter("min_target_beam_count").value)
        self.max_target_beam_count = int(self.get_parameter("max_target_beam_count").value)
        self.preferred_target_beam_count = int(self.get_parameter("preferred_target_beam_count").value)
        self.max_target_range_span_m = float(self.get_parameter("max_target_range_span_m").value)
        self.target_match_distance = float(self.get_parameter("target_match_distance").value)
        self.target_timeout_sec = float(self.get_parameter("target_timeout_sec").value)
        self.target_lock_distance = float(self.get_parameter("target_lock_distance").value)
        self.target_search_switch_distance = float(self.get_parameter("target_search_switch_distance").value)
        self.target_lock_required_frames = int(self.get_parameter("target_lock_required_frames").value)
        self.target_hold_confidence_floor = float(self.get_parameter("target_hold_confidence_floor").value)
        self.target_hold_decay = float(self.get_parameter("target_hold_decay").value)
        self.dynamic_cloud_topic = str(self.get_parameter("dynamic_cloud_topic").value)
        self.wall_filtered_cloud_topic = str(self.get_parameter("wall_filtered_cloud_topic").value)
        self.candidate_cloud_topic = str(self.get_parameter("candidate_cloud_topic").value)
        self.raw_cluster_centroids_topic = str(self.get_parameter("raw_cluster_centroids_topic").value)
        self.target_point_topic = str(self.get_parameter("target_point_topic").value)
        self.tracking_vector_topic = str(self.get_parameter("tracking_vector_topic").value)
        self.tracking_visible_topic = str(self.get_parameter("tracking_visible_topic").value)
        self.tracking_arrow_topic = str(self.get_parameter("tracking_arrow_topic").value)
        self.tracking_confidence_topic = str(self.get_parameter("tracking_confidence_topic").value)
        self.follow_active_topic = str(self.get_parameter("follow_active_topic").value)
        self.confidence_green_threshold = float(self.get_parameter("confidence_green_threshold").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.debug_log_path = Path(str(self.get_parameter("debug_log_path").value)).expanduser()

        self.ego_position: tuple[float, float] | None = None
        self.target_truth_position: tuple[float, float] | None = None
        self.ego_yaw: float | None = None
        self.ego_yaw_rate = 0.0
        self._last_angle_increment = 0.0
        self.scan_frame_id = "laser"
        self.latest_dynamic_points: list[tuple[float, float, float]] = []
        self.latest_wall_reference_points: list[tuple[float, float, float]] = []
        self.latest_wall_filtered_points: list[tuple[float, float, float]] = []
        self.latest_candidate_points: list[tuple[float, float, float]] = []
        self.latest_raw_cluster_centroids: list[tuple[float, float, float]] = []
        self.current_target: TargetState | None = None
        self.search_candidate: TargetState | None = None
        self.current_confidence = 0.0
        self.follow_active = False
        self.last_wall_beam_count = 0
        self.last_dynamic_beam_count = 0
        self.last_cluster_count = 0
        self.baseline_scans, self.baseline_beam_count = self._load_baseline_scans()
        self.last_baseline_scan_id: int | None = None
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
                "baseline_scan_id",
                "wall_beam_count",
                "dynamic_beam_count",
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
        self.wall_reference_pub = self.create_publisher(PointCloud2, self.wall_reference_cloud_topic, 10)
        self.wall_filtered_pub = self.create_publisher(PointCloud2, self.wall_filtered_cloud_topic, 10)
        self.candidate_pub = self.create_publisher(PointCloud2, self.candidate_cloud_topic, 10)
        self.raw_cluster_centroids_pub = self.create_publisher(PointCloud2, self.raw_cluster_centroids_topic, 10)
        self.target_point_pub = self.create_publisher(PointStamped, self.target_point_topic, 10)
        self.vector_pub = self.create_publisher(Vector3Stamped, self.tracking_vector_topic, 10)
        self.visible_pub = self.create_publisher(Bool, self.tracking_visible_topic, 10)
        self.arrow_pub = self.create_publisher(Marker, self.tracking_arrow_topic, 10)
        self.confidence_pub = self.create_publisher(Float32, self.tracking_confidence_topic, 10)

        self.create_subscription(Point, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(Point, self.target_pose_topic, self.target_pose_cb, 10)
        self.create_subscription(Imu, self.imu_topic, self.imu_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.create_subscription(Bool, self.follow_active_topic, self.follow_active_cb, 10)
        self.create_timer(1.0 / max(self.publish_rate_hz, 1e-3), self.publish_outputs)

        self.get_logger().info(
            "Target tracker ready. "
            f"scan={self.scan_topic}, pose={self.pose_topic}, imu={self.imu_topic}, baseline={self.baseline_csv_path}"
        )

    def _load_baseline_scans(self) -> tuple[list[BaselineScan], int]:
        if not self.baseline_csv_path.exists():
            fallback = find_repo_root() / "ros2" / "install" / "autodrive_f1tenth" / "lib" / "output" / "slam_runs" / "empty_track_baseline.csv"
            if fallback.exists():
                self.baseline_csv_path = fallback
        if not self.baseline_csv_path.exists():
            raise FileNotFoundError(f"Could not find SLAM baseline CSV at {self.baseline_csv_path}")

        grouped: dict[int, dict[str, object]] = {}
        max_beam_idx = -1
        with self.baseline_csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                scan_id = int(row["scan_id"])
                beam_idx = int(row["beam_idx"])
                max_beam_idx = max(max_beam_idx, beam_idx)
                group = grouped.setdefault(
                    scan_id,
                    {
                        "pose_x": float(row["ips_x_m"]),
                        "pose_y": float(row["ips_y_m"]),
                        "yaw_deg": float(row.get("yaw_deg", "0.0") or 0.0),
                        "ranges": {},
                    },
                )
                group["ranges"][beam_idx] = float(row["range_m"])

        beam_count = max_beam_idx + 1
        baseline_scans: list[BaselineScan] = []
        for scan_id in sorted(grouped):
            group = grouped[scan_id]
            ranges = np.full(beam_count, np.inf, dtype=float)
            for beam_idx, rng in group["ranges"].items():
                ranges[int(beam_idx)] = float(rng)
            baseline_scans.append(
                BaselineScan(
                    scan_id=scan_id,
                    pose_x=float(group["pose_x"]),
                    pose_y=float(group["pose_y"]),
                    yaw_deg=float(group["yaw_deg"]),
                    ranges=ranges,
                )
            )

        self.get_logger().info(f"Loaded {len(baseline_scans)} baseline scans from {self.baseline_csv_path}")
        return baseline_scans, beam_count

    def pose_cb(self, msg: Point) -> None:
        self.ego_position = (float(msg.x), float(msg.y))

    def target_pose_cb(self, msg: Point) -> None:
        self.target_truth_position = (float(msg.x), float(msg.y))

    def imu_cb(self, msg: Imu) -> None:
        self.ego_yaw_rate = float(msg.angular_velocity.z)
        self.ego_yaw = quaternion_to_yaw(
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        )

    def follow_active_cb(self, msg: Bool) -> None:
        self.follow_active = bool(msg.data)

    def scan_cb(self, msg: LaserScan) -> None:
        self.scan_frame_id = msg.header.frame_id or "map"
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        points = self.scan_to_points(msg)
        expected_ranges = self.expected_baseline_ranges(msg, points)
        dynamic_points = self.extract_anomalous_points(points, expected_ranges)
        clusters = self.cluster_points(dynamic_points)
        self.last_dynamic_beam_count = len(dynamic_points)
        self.last_cluster_count = len(clusters)
        self.latest_raw_cluster_centroids = [
            (
                sum(point[0] for point in cluster) / len(cluster),
                sum(point[1] for point in cluster) / len(cluster),
                0.0,
            )
            for cluster in clusters
            if cluster
        ]
        candidates = [candidate for candidate in (self.to_candidate(cluster) for cluster in clusters) if candidate is not None]
        if not candidates:
            candidates = [candidate for candidate in (self.to_candidate_fallback(cluster) for cluster in clusters) if candidate is not None]
        self.latest_wall_filtered_points = [(point[0], point[1], point[2]) for point in dynamic_points]
        self.latest_candidate_points = [
            (point[0], point[1], point[2])
            for candidate in candidates
            for point in candidate.points
        ]
        target = self.select_target(candidates, stamp_sec)

        if target is None:
            if self.current_target is not None and stamp_sec - self.current_target.stamp_sec <= self.target_timeout_sec:
                self.current_confidence = max(
                    self.target_hold_confidence_floor,
                    self.current_confidence * self.target_hold_decay,
                )
                self.append_debug_row(stamp_sec, None)
                return
            if self.current_target is not None and stamp_sec - self.current_target.stamp_sec > self.target_timeout_sec:
                self.current_target = None
            self.current_confidence = 0.0
            self.latest_dynamic_points = []
            self.append_debug_row(stamp_sec, None)
            return

        self.current_confidence = self.compute_confidence(target, stamp_sec)
        self.current_target = self.update_locked_state(target, stamp_sec)
        self.latest_dynamic_points = [(point[0], point[1], point[2]) for point in target.points]
        self.append_debug_row(stamp_sec, target)

    def local_to_world(self, local_x: float, local_y: float) -> tuple[float, float] | None:
        if self.ego_position is None or self.ego_yaw is None:
            return None
        cos_yaw = math.cos(self.ego_yaw)
        sin_yaw = math.sin(self.ego_yaw)
        world_x = self.ego_position[0] + local_x * cos_yaw - local_y * sin_yaw
        world_y = self.ego_position[1] + local_x * sin_yaw + local_y * cos_yaw
        return world_x, world_y

    def scan_to_points(self, scan: LaserScan) -> list[tuple[float, float, float, float, float, int]]:
        points: list[tuple[float, float, float, float, float, int]] = []
        self._last_angle_increment = float(scan.angle_increment)
        angle = scan.angle_min
        for beam_idx, distance in enumerate(scan.ranges):
            if math.isfinite(distance) and self.min_range <= distance <= self.max_range:
                angle_deg = math.degrees(angle)
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                in_forward_window = abs(angle_deg) <= self.forward_angle_window_deg
                if (not self.front_only or x > 0.0) and in_forward_window:
                    points.append((x, y, 0.0, angle_deg, distance, beam_idx))
            angle += scan.angle_increment
        return points

    def expected_baseline_ranges(
        self,
        scan: LaserScan,
        points: list[tuple[float, float, float, float, float, int]] | None = None,
    ) -> np.ndarray:
        beam_count = len(scan.ranges)
        expected_ranges = np.full(beam_count, np.inf, dtype=float)
        self.latest_wall_reference_points = []
        if self.ego_position is None or self.ego_yaw is None or not self.baseline_scans:
            return expected_ranges

        current_yaw_deg = math.degrees(self.ego_yaw)
        candidate_scans: list[tuple[float, BaselineScan]] = []
        for baseline_scan in self.baseline_scans:
            pos_distance = math.hypot(
                baseline_scan.pose_x - self.ego_position[0],
                baseline_scan.pose_y - self.ego_position[1],
            )
            yaw_error = abs((baseline_scan.yaw_deg - current_yaw_deg + 180.0) % 360.0 - 180.0)
            if pos_distance > self.baseline_position_match_m or yaw_error > self.baseline_yaw_match_deg:
                continue
            pose_score = pos_distance + 0.05 * yaw_error
            candidate_scans.append((pose_score, baseline_scan))

        best_scan: BaselineScan | None = None
        best_score = float("inf")
        beam_points = points if points is not None else self.scan_to_points(scan)
        for pose_score, baseline_scan in candidate_scans:
            fit_score = self.baseline_fit_score(beam_points, baseline_scan)
            score = fit_score + self.baseline_fit_pose_weight * pose_score
            if score < best_score:
                best_score = score
                best_scan = baseline_scan

        if best_scan is None:
            return expected_ranges

        self.last_baseline_scan_id = best_scan.scan_id
        usable = min(beam_count, len(best_scan.ranges))
        expected_ranges[:usable] = best_scan.ranges[:usable]
        angle_min = float(scan.angle_min)
        angle_increment = float(scan.angle_increment)
        reference_points = []
        angle = angle_min
        for beam_idx in range(usable):
            rng = expected_ranges[beam_idx]
            if math.isfinite(rng):
                x = rng * math.cos(angle)
                y = rng * math.sin(angle)
                if (not self.front_only or x > 0.0) and abs(math.degrees(angle)) <= self.forward_angle_window_deg:
                    reference_points.append((x, y, 0.0))
            angle += angle_increment
        self.latest_wall_reference_points = reference_points
        self.last_wall_beam_count = len(reference_points)
        return expected_ranges

    def baseline_fit_score(
        self,
        points: list[tuple[float, float, float, float, float, int]],
        baseline_scan: BaselineScan,
    ) -> float:
        if not points:
            return float("inf")

        residual_sum = 0.0
        used = 0
        for point in points:
            beam_idx = point[5]
            if beam_idx >= len(baseline_scan.ranges):
                continue
            expected_range = baseline_scan.ranges[beam_idx]
            if not math.isfinite(expected_range):
                continue
            residual = point[4] - expected_range
            if residual >= 0.0:
                residual_sum += self.baseline_fit_positive_residual_weight * residual
            else:
                # A small foreground object makes ranges shorter than the empty-track wall return.
                # Penalize that lightly so the wall-dominant scan still wins.
                residual_sum += self.baseline_fit_negative_residual_weight * abs(residual)
            used += 1

        if used == 0:
            return float("inf")
        return residual_sum / used

    def extract_anomalous_points(
        self,
        points: list[tuple[float, float, float, float, float, int]],
        expected_ranges: np.ndarray,
    ) -> list[tuple[float, float, float, float, float, int, float]]:
        dynamic_points: list[tuple[float, float, float, float, float, int, float]] = []
        for point in points:
            beam_idx = point[5]
            if beam_idx >= len(expected_ranges):
                continue
            expected_range = expected_ranges[beam_idx]
            if not math.isfinite(expected_range):
                continue
            range_delta = expected_range - point[4]
            if range_delta > self.baseline_range_margin_m:
                dynamic_points.append((point[0], point[1], point[2], point[3], point[4], point[5], range_delta))
        if not dynamic_points:
            return []

        grouped: list[list[tuple[float, float, float, float, float, int, float]]] = []
        current_group = [dynamic_points[0]]
        for point in dynamic_points[1:]:
            if point[5] == current_group[-1][5] + 1:
                current_group.append(point)
            else:
                grouped.append(current_group)
                current_group = [point]
        grouped.append(current_group)

        filtered_groups = [
            group for group in grouped if self.min_target_beam_count <= len(group) <= self.max_target_beam_count
        ]
        return [point for group in filtered_groups for point in group]

    def append_debug_row(self, stamp_sec: float, target: ClusterCandidate | None) -> None:
        ego_x = self.ego_position[0] if self.ego_position is not None else float("nan")
        ego_y = self.ego_position[1] if self.ego_position is not None else float("nan")
        ego_yaw_deg = math.degrees(self.ego_yaw) if self.ego_yaw is not None else float("nan")
        truth_x = self.target_truth_position[0] if self.target_truth_position is not None else float("nan")
        truth_y = self.target_truth_position[1] if self.target_truth_position is not None else float("nan")

        detected_world_x = float("nan")
        detected_world_y = float("nan")
        detected_local_x = float("nan")
        detected_local_y = float("nan")
        detected = 0
        if target is not None:
            detected = 1
            detected_local_x = target.centroid_x
            detected_local_y = target.centroid_y
            world_xy = self.local_to_world(target.centroid_x, target.centroid_y)
            if world_xy is not None:
                detected_world_x, detected_world_y = world_xy

        self.debug_writer.writerow(
            [
                f"{stamp_sec:.6f}",
                f"{ego_x:.6f}",
                f"{ego_y:.6f}",
                f"{ego_yaw_deg:.6f}",
                f"{truth_x:.6f}",
                f"{truth_y:.6f}",
                self.last_baseline_scan_id if self.last_baseline_scan_id is not None else -1,
                self.last_wall_beam_count,
                self.last_dynamic_beam_count,
                self.last_cluster_count,
                detected,
                f"{detected_local_x:.6f}",
                f"{detected_local_y:.6f}",
                f"{detected_world_x:.6f}",
                f"{detected_world_y:.6f}",
                f"{self.current_confidence:.6f}",
            ]
        )
        self.debug_file.flush()

    def cluster_points(
        self,
        points: list[tuple[float, float, float, float, float, int, float]],
    ) -> list[list[tuple[float, float, float, float, float, int, float]]]:
        if not points:
            return []

        clusters: list[list[tuple[float, float, float, float, float, int]]] = []
        current_cluster = [points[0]]
        for point in points[1:]:
            prev = current_cluster[-1]
            prev_range = math.hypot(prev[0], prev[1])
            point_range = math.hypot(point[0], point[1])
            adaptive_threshold = max(
                self.cluster_distance_threshold,
                self.cluster_distance_scale * max(prev_range, point_range) * abs(self._last_angle_increment),
            )
            if math.hypot(point[0] - prev[0], point[1] - prev[1]) <= adaptive_threshold:
                current_cluster.append(point)
            else:
                if len(current_cluster) >= self.min_cluster_points:
                    clusters.append(current_cluster)
                current_cluster = [point]

        if len(current_cluster) >= self.min_cluster_points:
            clusters.append(current_cluster)
        return clusters

    def to_candidate(
        self,
        cluster: list[tuple[float, float, float, float, float, int, float]],
    ) -> ClusterCandidate | None:
        xs = [point[0] for point in cluster]
        ys = [point[1] for point in cluster]
        ranges = [point[4] for point in cluster]
        strengths = [point[6] for point in cluster]
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)
        radial_distance = math.hypot(centroid_x, centroid_y)
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        range_span = max(ranges) - min(ranges)
        width = min(span_x, span_y)
        beam_count = len(cluster)

        if radial_distance < self.min_target_distance or radial_distance > self.max_target_distance:
            return None
        if width > self.max_target_width:
            return None
        if range_span > self.max_target_range_span_m:
            return None
        if centroid_x <= 0.0:
            return None
        if beam_count < self.min_target_beam_count or beam_count > self.max_target_beam_count:
            return None

        return ClusterCandidate(
            points=cluster,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            radial_distance=radial_distance,
            span_x=span_x,
            span_y=span_y,
            range_span=range_span,
            beam_count=beam_count,
            foreground_strength=sum(strengths) / max(len(strengths), 1),
        )

    def to_candidate_fallback(
        self,
        cluster: list[tuple[float, float, float, float, float, int, float]],
    ) -> ClusterCandidate | None:
        if len(cluster) < self.min_cluster_points:
            return None

        xs = [point[0] for point in cluster]
        ys = [point[1] for point in cluster]
        ranges = [point[4] for point in cluster]
        strengths = [point[6] for point in cluster]
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)
        radial_distance = math.hypot(centroid_x, centroid_y)
        if centroid_x <= 0.0 or radial_distance < self.min_target_distance or radial_distance > self.max_target_distance:
            return None

        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        range_span = max(ranges) - min(ranges)
        beam_count = len(cluster)
        return ClusterCandidate(
            points=cluster,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            radial_distance=radial_distance,
            span_x=span_x,
            span_y=span_y,
            range_span=range_span,
            beam_count=beam_count,
            foreground_strength=sum(strengths) / max(len(strengths), 1),
        )

    def select_target(self, candidates: list[ClusterCandidate], stamp_sec: float) -> ClusterCandidate | None:
        if not candidates:
            return None

        if self.current_target is None:
            seed_candidate = min(
                candidates,
                key=lambda candidate: (
                    candidate.range_span,
                    abs(candidate.centroid_y),
                    candidate.radial_distance,
                    abs(candidate.beam_count - self.preferred_target_beam_count),
                    -candidate.beam_count,
                ),
            )
            return self.search_mode_target(seed_candidate, stamp_sec)

        best_candidate = None
        best_score = float("inf")
        predicted_x, predicted_y = self.predict_target_position(stamp_sec)
        for candidate in candidates:
            distance = math.hypot(
                candidate.centroid_x - predicted_x,
                candidate.centroid_y - predicted_y,
            )
            beam_penalty = 0.03 * abs(candidate.beam_count - self.preferred_target_beam_count)
            range_penalty = candidate.range_span
            score = distance + 0.15 * abs(candidate.centroid_y) + beam_penalty + range_penalty
            if score < best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            return None
        best_distance = math.hypot(best_candidate.centroid_x - predicted_x, best_candidate.centroid_y - predicted_y)
        if best_distance > self.target_match_distance and stamp_sec - self.current_target.stamp_sec <= self.target_timeout_sec:
            return None
        return best_candidate

    def search_mode_target(self, seed_candidate: ClusterCandidate, stamp_sec: float) -> ClusterCandidate | None:
        if self.search_candidate is None:
            self.search_candidate = TargetState(
                stamp_sec=stamp_sec,
                centroid_x=seed_candidate.centroid_x,
                centroid_y=seed_candidate.centroid_y,
                velocity_x=0.0,
                velocity_y=0.0,
                lock_frames=1,
            )
            return seed_candidate

        jump = math.hypot(
            seed_candidate.centroid_x - self.search_candidate.centroid_x,
            seed_candidate.centroid_y - self.search_candidate.centroid_y,
        )
        if jump <= self.target_search_switch_distance:
            dt = max(stamp_sec - self.search_candidate.stamp_sec, 1e-6)
            self.search_candidate = TargetState(
                stamp_sec=stamp_sec,
                centroid_x=seed_candidate.centroid_x,
                centroid_y=seed_candidate.centroid_y,
                velocity_x=(seed_candidate.centroid_x - self.search_candidate.centroid_x) / dt,
                velocity_y=(seed_candidate.centroid_y - self.search_candidate.centroid_y) / dt,
                lock_frames=self.search_candidate.lock_frames + 1,
            )
            if self.search_candidate.lock_frames >= self.target_lock_required_frames:
                self.current_target = self.search_candidate
            return seed_candidate

        self.search_candidate = TargetState(
            stamp_sec=stamp_sec,
            centroid_x=seed_candidate.centroid_x,
            centroid_y=seed_candidate.centroid_y,
            velocity_x=0.0,
            velocity_y=0.0,
            lock_frames=1,
        )
        return seed_candidate

    def predict_target_position(self, stamp_sec: float) -> tuple[float, float]:
        if self.current_target is None:
            return 0.0, 0.0
        dt = max(0.0, stamp_sec - self.current_target.stamp_sec)
        return (
            self.current_target.centroid_x + self.current_target.velocity_x * dt,
            self.current_target.centroid_y + self.current_target.velocity_y * dt,
        )

    def update_locked_state(self, candidate: ClusterCandidate, stamp_sec: float) -> TargetState:
        if self.current_target is None:
            existing_lock_frames = self.search_candidate.lock_frames if self.search_candidate is not None else 1
            return TargetState(
                stamp_sec=stamp_sec,
                centroid_x=candidate.centroid_x,
                centroid_y=candidate.centroid_y,
                velocity_x=0.0,
                velocity_y=0.0,
                lock_frames=existing_lock_frames,
            )

        dt = max(stamp_sec - self.current_target.stamp_sec, 1e-6)
        velocity_x = (candidate.centroid_x - self.current_target.centroid_x) / dt
        velocity_y = (candidate.centroid_y - self.current_target.centroid_y) / dt
        return TargetState(
            stamp_sec=stamp_sec,
            centroid_x=candidate.centroid_x,
            centroid_y=candidate.centroid_y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            lock_frames=self.current_target.lock_frames + 1,
        )

    def compute_confidence(self, candidate: ClusterCandidate, stamp_sec: float) -> float:
        foreground_strength_score = max(0.0, min(1.0, candidate.foreground_strength))
        range_consistency_score = max(0.0, 1.0 - candidate.range_span / max(self.max_target_range_span_m, 1e-6))
        center_score = max(0.0, 1.0 - abs(candidate.centroid_y) / 1.0)
        beam_score = max(
            0.0,
            1.0 - abs(candidate.beam_count - self.preferred_target_beam_count) / max(self.preferred_target_beam_count, 1.0),
        )
        continuity_score = 0.5
        if self.current_target is not None and stamp_sec - self.current_target.stamp_sec <= self.target_timeout_sec:
            track_error = math.hypot(
                candidate.centroid_x - self.current_target.centroid_x,
                candidate.centroid_y - self.current_target.centroid_y,
            )
            continuity_score = max(0.0, 1.0 - track_error / max(self.target_match_distance, 1e-6))

        confidence = (
            0.40 * foreground_strength_score
            + 0.20 * range_consistency_score
            + 0.15 * center_score
            + 0.15 * beam_score
            + 0.10 * continuity_score
        )
        return min(0.99, max(0.0, confidence))

    def publish_outputs(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.dynamic_pub.publish(make_cloud(self.latest_dynamic_points, self.scan_frame_id, stamp))
        self.wall_reference_pub.publish(make_cloud(self.latest_wall_reference_points, self.scan_frame_id, stamp))
        self.wall_filtered_pub.publish(make_cloud(self.latest_wall_filtered_points, self.scan_frame_id, stamp))
        self.candidate_pub.publish(make_cloud(self.latest_candidate_points, self.scan_frame_id, stamp))
        self.raw_cluster_centroids_pub.publish(make_cloud(self.latest_raw_cluster_centroids, self.scan_frame_id, stamp))
        self.confidence_pub.publish(Float32(data=float(self.current_confidence)))

        visible = Bool()
        visible.data = self.current_target is not None
        self.visible_pub.publish(visible)

        arrow = Marker()
        arrow.header.frame_id = self.scan_frame_id
        arrow.header.stamp = stamp
        arrow.ns = "target_tracker"
        arrow.id = 1
        arrow.type = Marker.ARROW
        arrow.scale.x = 0.08
        arrow.scale.y = 0.16
        arrow.scale.z = 0.20
        arrow.color.a = 1.0

        label = Marker()
        label.header.frame_id = self.scan_frame_id
        label.header.stamp = stamp
        label.ns = "target_tracker"
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

        is_confident = self.current_confidence >= self.confidence_green_threshold
        if self.follow_active:
            arrow.color.r = 0.1
            arrow.color.g = 0.3
            arrow.color.b = 1.0
        else:
            arrow.color.r = 0.1 if is_confident else 1.0
            arrow.color.g = 1.0 if is_confident else 0.1
            arrow.color.b = 0.1
        label.color.r = arrow.color.r
        label.color.g = arrow.color.g
        label.color.b = arrow.color.b

        origin = Point()
        origin.x = 0.0
        origin.y = 0.0
        origin.z = 0.0
        target = Point()
        target.x = self.current_target.centroid_x
        target.y = self.current_target.centroid_y
        target.z = 0.0
        arrow.action = Marker.ADD
        arrow.points = [origin, target]
        self.arrow_pub.publish(arrow)

        label.action = Marker.ADD
        label.pose.position.x = target.x + 0.20
        label.pose.position.y = target.y + 0.20
        label.pose.position.z = 0.20
        label.pose.orientation.w = 1.0
        label.text = f"{self.current_confidence:.2f}"
        self.arrow_pub.publish(label)

        target_point = PointStamped()
        target_point.header.frame_id = self.scan_frame_id
        target_point.header.stamp = stamp
        target_point.point = target
        self.target_point_pub.publish(target_point)

        vector = Vector3Stamped()
        vector.header.frame_id = self.scan_frame_id
        vector.header.stamp = stamp
        dx = self.current_target.centroid_x
        dy = self.current_target.centroid_y
        norm = max(math.hypot(dx, dy), 1e-6)
        vector.vector.x = dx / norm
        vector.vector.y = dy / norm
        vector.vector.z = 0.0
        self.vector_pub.publish(vector)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TargetVehicleTracker()
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

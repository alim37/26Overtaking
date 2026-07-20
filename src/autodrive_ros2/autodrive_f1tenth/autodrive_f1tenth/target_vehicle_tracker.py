#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from scipy.spatial import cKDTree
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


def make_cloud(points: list[tuple[float, float, float]], frame_id: str, stamp) -> PointCloud2:
    return make_pointcloud2(points, frame_id, stamp)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "tracks" / "src").exists():
            return parent
    return Path.cwd()


@dataclass
class BeamSample:
    local_x: float
    local_y: float
    angle_rad: float
    angle_deg: float
    measured_range: float
    beam_idx: int


@dataclass
class ForegroundBeam:
    local_x: float
    local_y: float
    angle_rad: float
    angle_deg: float
    measured_range: float
    beam_idx: int
    residual_m: float
    wall_distance_m: float


@dataclass
class BaselineScan:
    scan_id: int
    pose_x: float
    pose_y: float
    yaw_deg: float
    ranges: np.ndarray


@dataclass
class ClusterCandidate:
    points: list[ForegroundBeam]
    centroid_x: float
    centroid_y: float
    radial_distance: float
    beam_count: int
    width_m: float
    range_span_m: float
    mean_residual_m: float
    mean_angle_deg: float
    mean_wall_distance_m: float


@dataclass
class TargetState:
    stamp_sec: float
    centroid_x: float
    centroid_y: float
    velocity_x: float
    velocity_y: float
    lock_frames: int


class TargetVehicleTracker(Node):
    """
    Residual-based target tracker.

    1. Match the current ego IPS pose + IMU yaw against the saved empty-track baseline scans.
    2. Compare each live beam to the matched empty-track beam at the same beam index.
    3. Mark beams as foreground when the live beam is meaningfully closer than the baseline.
    4. Group consecutive foreground beams, pick plausible car clusters, then track one target over time.
    """

    def __init__(self) -> None:
        super().__init__("target_vehicle_tracker")

        repo_root = find_repo_root()
        self.declare_parameter("scan_topic", "/autodrive/f1tenth_1/lidar")
        self.declare_parameter("pose_topic", "/autodrive/f1tenth_1/ips")
        self.declare_parameter("target_pose_topic", "/autodrive/f1tenth_2/ips")
        self.declare_parameter("imu_topic", "/autodrive/f1tenth_1/imu")
        self.declare_parameter("front_only", True)
        self.declare_parameter("forward_angle_window_deg", 10.0)
        self.declare_parameter("min_range", 0.10)
        self.declare_parameter("max_range", 12.0)
        self.declare_parameter(
            "baseline_csv_path",
            str(repo_root / "output" / "slam_runs" / "empty_track_baseline.csv"),
        )
        self.declare_parameter(
            "wall_mask_csv_path",
            str(repo_root / "output" / "slam_runs" / "slam_toolbox_boundary_wall_mask.csv"),
        )
        self.declare_parameter("baseline_position_match_m", 1.5)
        self.declare_parameter("baseline_yaw_match_deg", 25.0)
        self.declare_parameter("baseline_fit_positive_residual_weight", 1.00)
        self.declare_parameter("baseline_fit_negative_residual_weight", 0.20)
        self.declare_parameter("baseline_fit_pose_weight", 0.15)
        self.declare_parameter("baseline_range_margin_m", 0.25)
        self.declare_parameter("min_distance_from_wall_m", 0.05)
        self.declare_parameter("wall_mask_cell_size_m", 0.05)
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
        self.declare_parameter("target_lock_distance", 0.8)
        self.declare_parameter("target_search_switch_distance", 0.5)
        self.declare_parameter("target_lock_required_frames", 3)
        self.declare_parameter("target_hold_confidence_floor", 0.55)
        self.declare_parameter("target_hold_decay", 0.92)
        self.declare_parameter("dynamic_cloud_topic", "/autodrive/f1tenth_1/target_tracker/dynamic_cloud")
        self.declare_parameter("wall_reference_cloud_topic", "/autodrive/f1tenth_1/target_tracker/wall_reference_cloud")
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
        self.wall_mask_csv_path = Path(str(self.get_parameter("wall_mask_csv_path").value)).expanduser()
        self.baseline_position_match_m = float(self.get_parameter("baseline_position_match_m").value)
        self.baseline_yaw_match_deg = float(self.get_parameter("baseline_yaw_match_deg").value)
        self.baseline_fit_positive_residual_weight = float(
            self.get_parameter("baseline_fit_positive_residual_weight").value
        )
        self.baseline_fit_negative_residual_weight = float(
            self.get_parameter("baseline_fit_negative_residual_weight").value
        )
        self.baseline_fit_pose_weight = float(self.get_parameter("baseline_fit_pose_weight").value)
        self.baseline_range_margin_m = float(self.get_parameter("baseline_range_margin_m").value)
        self.min_distance_from_wall_m = float(self.get_parameter("min_distance_from_wall_m").value)
        self.wall_mask_cell_size_m = float(self.get_parameter("wall_mask_cell_size_m").value)
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
        self.wall_reference_cloud_topic = str(self.get_parameter("wall_reference_cloud_topic").value)
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
        self.scan_frame_id = "lidar_1"
        self._last_angle_increment = 0.0
        self.follow_active = False

        self.baseline_scans, self.baseline_beam_count = self._load_baseline_scans()
        self.wall_mask_points, self.wall_mask_tree = self._load_wall_mask_points()
        self.last_baseline_scan_id: int | None = None

        self.current_target: TargetState | None = None
        self.search_target: TargetState | None = None
        self.current_confidence = 0.0

        self.latest_dynamic_points: list[tuple[float, float, float]] = []
        self.latest_wall_reference_points: list[tuple[float, float, float]] = []
        self.latest_wall_filtered_points: list[tuple[float, float, float]] = []
        self.latest_candidate_points: list[tuple[float, float, float]] = []
        self.latest_raw_cluster_centroids: list[tuple[float, float, float]] = []
        self.last_wall_beam_count = 0
        self.last_dynamic_beam_count = 0
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
                "lock_frames",
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
            "Residual target tracker ready. "
            f"scan={self.scan_topic}, pose={self.pose_topic}, imu={self.imu_topic}, baseline={self.baseline_csv_path}, "
            f"wall_mask={self.wall_mask_csv_path}, min_wall_distance={self.min_distance_from_wall_m:.3f}m"
        )

    def _load_baseline_scans(self) -> tuple[list[BaselineScan], int]:
        if not self.baseline_csv_path.exists():
            fallback = (
                find_repo_root()
                / "ros2"
                / "install"
                / "autodrive_f1tenth"
                / "lib"
                / "output"
                / "slam_runs"
                / "empty_track_baseline.csv"
            )
            if fallback.exists():
                self.baseline_csv_path = fallback
        if not self.baseline_csv_path.exists():
            raise FileNotFoundError(f"Could not find SLAM baseline CSV at {self.baseline_csv_path}")

        grouped: dict[int, dict[str, object]] = {}
        max_beam_idx = -1
        with self.baseline_csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if "scan_id" not in (reader.fieldnames or []):
                raise KeyError(
                    f"{self.baseline_csv_path} does not contain scan-based baseline columns. "
                    "Expected at least: scan_id, beam_idx, range_m, ips_x_m, ips_y_m, yaw_deg"
                )
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

    def _load_wall_mask_points(self) -> tuple[np.ndarray, cKDTree | None]:
        csv_path = self.wall_mask_csv_path
        if not csv_path.exists():
            fallback = (
                find_repo_root()
                / "ros2"
                / "install"
                / "autodrive_f1tenth"
                / "lib"
                / "output"
                / "slam_runs"
                / csv_path.name
            )
            if fallback.exists():
                csv_path = fallback
                self.wall_mask_csv_path = fallback
        if not csv_path.exists():
            self.get_logger().warn(f"Wall mask CSV not found at {self.wall_mask_csv_path}; continuing without wall mask")
            return np.empty((0, 2), dtype=float), None

        points: list[tuple[float, float]] = []
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if "world_x_m" in row and "world_y_m" in row:
                    x = float(row["world_x_m"])
                    y = float(row["world_y_m"])
                    points.append((x, y))
        point_array = np.asarray(points, dtype=float)
        tree = cKDTree(point_array) if len(point_array) > 0 else None
        self.get_logger().info(f"Loaded wall mask points={len(point_array)} from {csv_path}")
        return point_array, tree

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

    def follow_active_cb(self, msg: Bool) -> None:
        self.follow_active = bool(msg.data)

    def local_to_world(self, local_x: float, local_y: float) -> tuple[float, float] | None:
        if self.ego_position is None or self.ego_yaw is None:
            return None
        cos_yaw = math.cos(self.ego_yaw)
        sin_yaw = math.sin(self.ego_yaw)
        return (
            self.ego_position[0] + local_x * cos_yaw - local_y * sin_yaw,
            self.ego_position[1] + local_x * sin_yaw + local_y * cos_yaw,
        )

    def distance_to_wall(self, world_x: float, world_y: float) -> float:
        if self.wall_mask_tree is None:
            return float("inf")
        distance, _ = self.wall_mask_tree.query([world_x, world_y], k=1)
        return float(distance)

    def scan_to_beams(self, scan: LaserScan) -> list[BeamSample]:
        beams: list[BeamSample] = []
        self._last_angle_increment = float(scan.angle_increment)
        angle = float(scan.angle_min)
        for beam_idx, distance in enumerate(scan.ranges):
            if math.isfinite(distance) and self.min_range <= distance <= self.max_range:
                local_x = distance * math.cos(angle)
                local_y = distance * math.sin(angle)
                beams.append(
                    BeamSample(
                        local_x=local_x,
                        local_y=local_y,
                        angle_rad=angle,
                        angle_deg=math.degrees(angle),
                        measured_range=float(distance),
                        beam_idx=beam_idx,
                    )
                )
            angle += scan.angle_increment
        return beams

    def baseline_fit_score(self, beams: list[BeamSample], baseline_scan: BaselineScan) -> float:
        residual_sum = 0.0
        used = 0
        for beam in beams:
            if beam.beam_idx >= len(baseline_scan.ranges):
                continue
            expected_range = baseline_scan.ranges[beam.beam_idx]
            if not math.isfinite(expected_range):
                continue
            residual = beam.measured_range - expected_range
            if residual >= 0.0:
                residual_sum += self.baseline_fit_positive_residual_weight * residual
            else:
                residual_sum += self.baseline_fit_negative_residual_weight * abs(residual)
            used += 1
        if used == 0:
            return float("inf")
        return residual_sum / used

    def match_baseline_scan(self, beams: list[BeamSample]) -> BaselineScan | None:
        if self.ego_position is None or self.ego_yaw is None or not self.baseline_scans:
            return None

        current_yaw_deg = math.degrees(self.ego_yaw)
        best_scan: BaselineScan | None = None
        best_score = float("inf")
        for baseline_scan in self.baseline_scans:
            pos_distance = math.hypot(
                baseline_scan.pose_x - self.ego_position[0],
                baseline_scan.pose_y - self.ego_position[1],
            )
            yaw_error = abs((baseline_scan.yaw_deg - current_yaw_deg + 180.0) % 360.0 - 180.0)
            if pos_distance > self.baseline_position_match_m or yaw_error > self.baseline_yaw_match_deg:
                continue
            pose_score = pos_distance + 0.05 * yaw_error
            fit_score = self.baseline_fit_score(beams, baseline_scan)
            total_score = fit_score + self.baseline_fit_pose_weight * pose_score
            if total_score < best_score:
                best_score = total_score
                best_scan = baseline_scan

        self.last_baseline_scan_id = best_scan.scan_id if best_scan is not None else None
        return best_scan

    def classify_foreground(
        self,
        beams: list[BeamSample],
        baseline_scan: BaselineScan | None,
    ) -> list[ForegroundBeam]:
        self.latest_wall_reference_points = []
        self.latest_wall_filtered_points = []
        if baseline_scan is None:
            self.last_wall_beam_count = 0
            return []

        foreground_beams: list[ForegroundBeam] = []
        wall_reference_points: list[tuple[float, float, float]] = []
        wall_filtered_points: list[tuple[float, float, float]] = []

        for beam in beams:
            if beam.beam_idx >= len(baseline_scan.ranges):
                continue
            expected_range = baseline_scan.ranges[beam.beam_idx]
            if not math.isfinite(expected_range):
                continue

            expected_x = expected_range * math.cos(beam.angle_rad)
            expected_y = expected_range * math.sin(beam.angle_rad)
            wall_reference_points.append((expected_x, expected_y, 0.0))

            residual = expected_range - beam.measured_range
            if residual > self.baseline_range_margin_m:
                world_xy = self.local_to_world(beam.local_x, beam.local_y)
                wall_distance_m = float("inf")
                if world_xy is not None:
                    wall_distance_m = self.distance_to_wall(world_xy[0], world_xy[1])
                if wall_distance_m <= self.min_distance_from_wall_m:
                    wall_filtered_points.append((beam.local_x, beam.local_y, 0.0))
                else:
                    foreground_beams.append(
                        ForegroundBeam(
                            local_x=beam.local_x,
                            local_y=beam.local_y,
                            angle_rad=beam.angle_rad,
                            angle_deg=beam.angle_deg,
                            measured_range=beam.measured_range,
                            beam_idx=beam.beam_idx,
                            residual_m=residual,
                            wall_distance_m=wall_distance_m,
                        )
                    )
            else:
                wall_filtered_points.append((beam.local_x, beam.local_y, 0.0))

        self.latest_wall_reference_points = wall_reference_points
        self.latest_wall_filtered_points = wall_filtered_points
        self.last_wall_beam_count = len(wall_reference_points)
        return foreground_beams

    def cluster_foreground(self, beams: list[ForegroundBeam]) -> list[list[ForegroundBeam]]:
        if not beams:
            return []

        grouped: list[list[ForegroundBeam]] = []
        current_group = [beams[0]]
        for beam in beams[1:]:
            prev = current_group[-1]
            prev_range = prev.measured_range
            curr_range = beam.measured_range
            adaptive_threshold = max(
                self.cluster_distance_threshold,
                self.cluster_distance_scale * max(prev_range, curr_range) * abs(self._last_angle_increment),
            )
            consecutive = beam.beam_idx == prev.beam_idx + 1
            close_enough = math.hypot(beam.local_x - prev.local_x, beam.local_y - prev.local_y) <= adaptive_threshold
            if consecutive and close_enough:
                current_group.append(beam)
            else:
                if len(current_group) >= self.min_cluster_points:
                    grouped.append(current_group)
                current_group = [beam]

        if len(current_group) >= self.min_cluster_points:
            grouped.append(current_group)
        return grouped

    def to_candidate(self, cluster: list[ForegroundBeam]) -> ClusterCandidate | None:
        beam_count = len(cluster)
        if beam_count < self.min_target_beam_count or beam_count > self.max_target_beam_count:
            return None

        xs = [beam.local_x for beam in cluster]
        ys = [beam.local_y for beam in cluster]
        ranges = [beam.measured_range for beam in cluster]
        residuals = [beam.residual_m for beam in cluster]
        angles = [beam.angle_deg for beam in cluster]
        wall_distances = [beam.wall_distance_m for beam in cluster]

        centroid_x = sum(xs) / beam_count
        centroid_y = sum(ys) / beam_count
        radial_distance = math.hypot(centroid_x, centroid_y)
        if radial_distance < self.min_target_distance or radial_distance > self.max_target_distance:
            return None
        if self.front_only and centroid_x <= 0.0:
            return None

        first = cluster[0]
        last = cluster[-1]
        width_m = math.hypot(last.local_x - first.local_x, last.local_y - first.local_y)
        if width_m < self.min_target_width or width_m > self.max_target_width:
            return None

        range_span_m = max(ranges) - min(ranges)
        if range_span_m > self.max_target_range_span_m:
            return None

        return ClusterCandidate(
            points=cluster,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            radial_distance=radial_distance,
            beam_count=beam_count,
            width_m=width_m,
            range_span_m=range_span_m,
            mean_residual_m=sum(residuals) / beam_count,
            mean_angle_deg=sum(angles) / beam_count,
            mean_wall_distance_m=sum(wall_distances) / beam_count,
        )

    def seed_candidate(self, candidates: list[ClusterCandidate]) -> ClusterCandidate | None:
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda candidate: (
                abs(candidate.beam_count - self.preferred_target_beam_count),
                candidate.range_span_m,
                -candidate.mean_residual_m,
                abs(candidate.mean_angle_deg),
                candidate.radial_distance,
            ),
        )

    def predict_target_position(self, stamp_sec: float) -> tuple[float, float]:
        if self.current_target is None:
            return 0.0, 0.0
        dt = max(stamp_sec - self.current_target.stamp_sec, 0.0)
        return (
            self.current_target.centroid_x + self.current_target.velocity_x * dt,
            self.current_target.centroid_y + self.current_target.velocity_y * dt,
        )

    def update_locked_state(self, candidate: ClusterCandidate, stamp_sec: float) -> TargetState:
        if self.current_target is None:
            return TargetState(
                stamp_sec=stamp_sec,
                centroid_x=candidate.centroid_x,
                centroid_y=candidate.centroid_y,
                velocity_x=0.0,
                velocity_y=0.0,
                lock_frames=1,
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
            lock_frames=min(self.current_target.lock_frames + 1, self.target_lock_required_frames + 4),
        )

    def search_mode_target(self, candidate: ClusterCandidate, stamp_sec: float) -> ClusterCandidate | None:
        if self.search_target is None:
            self.search_target = TargetState(
                stamp_sec=stamp_sec,
                centroid_x=candidate.centroid_x,
                centroid_y=candidate.centroid_y,
                velocity_x=0.0,
                velocity_y=0.0,
                lock_frames=1,
            )
            return candidate

        jump = math.hypot(
            candidate.centroid_x - self.search_target.centroid_x,
            candidate.centroid_y - self.search_target.centroid_y,
        )
        if jump <= self.target_search_switch_distance:
            dt = max(stamp_sec - self.search_target.stamp_sec, 1e-6)
            velocity_x = (candidate.centroid_x - self.search_target.centroid_x) / dt
            velocity_y = (candidate.centroid_y - self.search_target.centroid_y) / dt
            self.search_target = TargetState(
                stamp_sec=stamp_sec,
                centroid_x=candidate.centroid_x,
                centroid_y=candidate.centroid_y,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                lock_frames=self.search_target.lock_frames + 1,
            )
        else:
            self.search_target = TargetState(
                stamp_sec=stamp_sec,
                centroid_x=candidate.centroid_x,
                centroid_y=candidate.centroid_y,
                velocity_x=0.0,
                velocity_y=0.0,
                lock_frames=1,
            )

        if self.search_target.lock_frames >= self.target_lock_required_frames:
            self.current_target = self.search_target
        return candidate

    def select_target(self, candidates: list[ClusterCandidate], stamp_sec: float) -> ClusterCandidate | None:
        if not candidates:
            self.search_target = None
            return None

        if self.current_target is None:
            seed = self.seed_candidate(candidates)
            if seed is None:
                return None
            return self.search_mode_target(seed, stamp_sec)

        predicted_x, predicted_y = self.predict_target_position(stamp_sec)
        best_candidate = None
        best_score = float("inf")
        for candidate in candidates:
            jump = math.hypot(candidate.centroid_x - predicted_x, candidate.centroid_y - predicted_y)
            beam_penalty = abs(candidate.beam_count - self.preferred_target_beam_count) / max(
                self.preferred_target_beam_count, 1
            )
            center_penalty = abs(candidate.mean_angle_deg) / 30.0
            score = (
                jump
                + 0.25 * beam_penalty
                + 0.20 * candidate.range_span_m
                + 0.10 * center_penalty
                - 0.20 * min(candidate.mean_residual_m, 1.0)
            )
            if score < best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            return None

        jump = math.hypot(best_candidate.centroid_x - predicted_x, best_candidate.centroid_y - predicted_y)
        if jump > self.target_match_distance:
            return None
        return best_candidate

    def compute_confidence(self, candidate: ClusterCandidate, stamp_sec: float) -> float:
        lock_score = 0.0
        if self.current_target is not None:
            lock_score = min(self.current_target.lock_frames / max(self.target_lock_required_frames, 1), 1.0)

        residual_score = max(0.0, min(1.0, (candidate.mean_residual_m - self.baseline_range_margin_m) / 0.50))
        beam_score = max(
            0.0,
            1.0 - abs(candidate.beam_count - self.preferred_target_beam_count) / max(self.preferred_target_beam_count, 1),
        )
        span_score = max(0.0, 1.0 - candidate.range_span_m / max(self.max_target_range_span_m, 1e-6))
        wall_separation_score = max(
            0.0,
            min(1.0, (candidate.mean_wall_distance_m - self.min_distance_from_wall_m) / 0.25),
        )

        stability_score = 0.5
        if self.current_target is not None:
            predicted_x, predicted_y = self.predict_target_position(stamp_sec)
            jump = math.hypot(candidate.centroid_x - predicted_x, candidate.centroid_y - predicted_y)
            stability_score = max(0.0, 1.0 - jump / max(self.target_match_distance, 1e-6))

        confidence = (
            0.40 * lock_score
            + 0.30 * stability_score
            + 0.10 * residual_score
            + 0.10 * wall_separation_score
            + 0.05 * beam_score
            + 0.05 * span_score
        )
        return max(0.0, min(0.99, confidence))

    def append_debug_row(self, stamp_sec: float, target: ClusterCandidate | None) -> None:
        ego_x = self.ego_position[0] if self.ego_position is not None else float("nan")
        ego_y = self.ego_position[1] if self.ego_position is not None else float("nan")
        ego_yaw_deg = math.degrees(self.ego_yaw) if self.ego_yaw is not None else float("nan")
        truth_x = self.target_truth_position[0] if self.target_truth_position is not None else float("nan")
        truth_y = self.target_truth_position[1] if self.target_truth_position is not None else float("nan")

        detected = 0
        local_x = float("nan")
        local_y = float("nan")
        world_x = float("nan")
        world_y = float("nan")
        if target is not None:
            detected = 1
            local_x = target.centroid_x
            local_y = target.centroid_y
            world_xy = self.local_to_world(local_x, local_y)
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
                self.last_baseline_scan_id if self.last_baseline_scan_id is not None else -1,
                self.last_wall_beam_count,
                self.last_dynamic_beam_count,
                self.last_cluster_count,
                detected,
                f"{local_x:.6f}",
                f"{local_y:.6f}",
                f"{world_x:.6f}",
                f"{world_y:.6f}",
                f"{self.current_confidence:.6f}",
                self.current_target.lock_frames if self.current_target is not None else 0,
            ]
        )
        self.debug_file.flush()

    def scan_cb(self, msg: LaserScan) -> None:
        self.scan_frame_id = msg.header.frame_id or "lidar_1"
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.ego_position is None or self.ego_yaw is None:
            return

        beams = self.scan_to_beams(msg)
        baseline_scan = self.match_baseline_scan(beams)
        foreground_beams = self.classify_foreground(beams, baseline_scan)
        clusters = self.cluster_foreground(foreground_beams)
        candidates = [candidate for candidate in (self.to_candidate(cluster) for cluster in clusters) if candidate is not None]

        self.last_dynamic_beam_count = len(foreground_beams)
        self.last_cluster_count = len(candidates)
        self.latest_dynamic_points = [(beam.local_x, beam.local_y, 0.0) for beam in foreground_beams]
        self.latest_candidate_points = [
            (beam.local_x, beam.local_y, 0.0)
            for candidate in candidates
            for beam in candidate.points
        ]
        self.latest_raw_cluster_centroids = [(c.centroid_x, c.centroid_y, 0.0) for c in candidates]

        target = self.select_target(candidates, stamp_sec)
        if target is None:
            if self.current_target is not None and stamp_sec - self.current_target.stamp_sec <= self.target_timeout_sec:
                self.current_confidence = max(
                    self.target_hold_confidence_floor,
                    self.current_confidence * self.target_hold_decay,
                )
                self.append_debug_row(stamp_sec, None)
                return
            self.current_target = None
            self.current_confidence = 0.0
            self.latest_dynamic_points = []
            self.append_debug_row(stamp_sec, None)
            return

        self.current_target = self.update_locked_state(target, stamp_sec)
        self.current_confidence = self.compute_confidence(target, stamp_sec)
        self.latest_dynamic_points = [(beam.local_x, beam.local_y, 0.0) for beam in target.points]
        self.append_debug_row(stamp_sec, target)

    def publish_outputs(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.dynamic_pub.publish(make_cloud(self.latest_dynamic_points, self.scan_frame_id, stamp))
        self.wall_reference_pub.publish(make_cloud(self.latest_wall_reference_points, self.scan_frame_id, stamp))
        self.wall_filtered_pub.publish(make_cloud(self.latest_wall_filtered_points, self.scan_frame_id, stamp))
        self.candidate_pub.publish(make_cloud(self.latest_candidate_points, self.scan_frame_id, stamp))
        self.raw_cluster_centroids_pub.publish(make_cloud(self.latest_raw_cluster_centroids, self.scan_frame_id, stamp))
        self.confidence_pub.publish(Float32(data=float(self.current_confidence)))

        visible = Bool(data=self.current_target is not None)
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

        origin = Point(x=0.0, y=0.0, z=0.0)
        target = Point(x=self.current_target.centroid_x, y=self.current_target.centroid_y, z=0.0)
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
        norm = max(math.hypot(target.x, target.y), 1e-6)
        vector.vector.x = target.x / norm
        vector.vector.y = target.y / norm
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

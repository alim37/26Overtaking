#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped, Vector3Stamped
from rclpy.node import Node
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Float32
from visualization_msgs.msg import Marker, MarkerArray

from autodrive_f1tenth.pure_pursuit import load_manual_reference_line


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def circular_delta(a: float, b: float, period: float) -> float:
    return (a - b + 0.5 * period) % period - 0.5 * period


@dataclass
class Detection:
    local_x: float
    local_y: float
    world_x: float
    world_y: float
    s: float
    d: float
    size_m: float
    beam_count: int
    range_m: float


@dataclass
class TrackedObstacle:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    last_stamp: float
    size_m: float
    beam_count: int
    history: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=30))
    hits: int = 1
    misses: int = 0
    static_votes: int = 0
    dynamic_votes: int = 0
    classification: str = "unknown"
    updated_this_scan: bool = True


class ForzaOpponentTracker(Node):
    """ROS 2 port of the ForzaETH adaptive-breakpoint detector and temporal tracker."""

    def __init__(self) -> None:
        super().__init__("forza_opponent_tracker")

        self.declare_parameter("scan_topic", "/autodrive/f1tenth_1/lidar")
        self.declare_parameter("pose_topic", "/autodrive/f1tenth_1/ips")
        self.declare_parameter("imu_topic", "/autodrive/f1tenth_1/imu")
        self.declare_parameter("wall_mask_csv_path", "")
        self.declare_parameter("num_path_points", 800)
        self.declare_parameter("lambda_angle_deg", 10.0)
        self.declare_parameter("lidar_sigma_m", 0.03)
        self.declare_parameter("min_two_point_distance_m", 0.01)
        self.declare_parameter("min_cluster_beams", 5)
        self.declare_parameter("max_cluster_size_m", 0.70)
        self.declare_parameter("opponent_length_m", 0.55)
        self.declare_parameter("opponent_width_m", 0.30)
        self.declare_parameter("max_viewing_distance_m", 9.0)
        self.declare_parameter("boundary_inflation_m", 0.08)
        self.declare_parameter("fallback_half_track_width_m", 1.5)
        self.declare_parameter("association_distance_m", 0.45)
        self.declare_parameter("dynamic_association_multiplier", 1.5)
        self.declare_parameter("min_classification_measurements", 6)
        self.declare_parameter("static_std_threshold_m", 0.08)
        self.declare_parameter("dynamic_std_threshold_m", 0.16)
        self.declare_parameter("min_dynamic_speed_mps", 0.20)
        self.declare_parameter("track_ttl_scans", 10)
        self.declare_parameter("dynamic_hold_scans", 12)
        self.declare_parameter("max_front_angle_deg", 70.0)
        self.declare_parameter("measurement_variance", 0.01)
        self.declare_parameter("process_variance_s", 1.0)
        self.declare_parameter("process_variance_d", 2.0)
        self.declare_parameter("confidence_green_threshold", 0.90)
        self.declare_parameter("target_point_topic", "/autodrive/f1tenth_1/target_tracker/target_point")
        self.declare_parameter("tracking_vector_topic", "/autodrive/f1tenth_1/target_tracker/tracking_vector")
        self.declare_parameter("target_visible_topic", "/autodrive/f1tenth_1/target_tracker/target_visible")
        self.declare_parameter("tracking_arrow_topic", "/autodrive/f1tenth_1/target_tracker/tracking_arrow")
        self.declare_parameter("tracking_confidence_topic", "/autodrive/f1tenth_1/target_tracker/tracking_confidence")
        self.declare_parameter("follow_active_topic", "/autodrive/f1tenth_1/target_tracker/follow_active")
        self.declare_parameter("debug_markers_topic", "/autodrive/f1tenth_1/forza_tracker/debug_markers")

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.wall_mask_csv_path = Path(str(self.get_parameter("wall_mask_csv_path").value)).expanduser()
        self.lambda_angle = math.radians(float(self.get_parameter("lambda_angle_deg").value))
        self.lidar_sigma_m = float(self.get_parameter("lidar_sigma_m").value)
        self.min_two_point_distance_m = float(self.get_parameter("min_two_point_distance_m").value)
        self.min_cluster_beams = int(self.get_parameter("min_cluster_beams").value)
        self.max_cluster_size_m = float(self.get_parameter("max_cluster_size_m").value)
        self.opponent_length_m = float(self.get_parameter("opponent_length_m").value)
        self.opponent_width_m = float(self.get_parameter("opponent_width_m").value)
        self.max_viewing_distance_m = float(self.get_parameter("max_viewing_distance_m").value)
        self.boundary_inflation_m = float(self.get_parameter("boundary_inflation_m").value)
        self.fallback_half_track_width_m = float(self.get_parameter("fallback_half_track_width_m").value)
        self.association_distance_m = float(self.get_parameter("association_distance_m").value)
        self.dynamic_association_multiplier = float(self.get_parameter("dynamic_association_multiplier").value)
        self.min_classification_measurements = int(self.get_parameter("min_classification_measurements").value)
        self.static_std_threshold_m = float(self.get_parameter("static_std_threshold_m").value)
        self.dynamic_std_threshold_m = float(self.get_parameter("dynamic_std_threshold_m").value)
        self.min_dynamic_speed_mps = float(self.get_parameter("min_dynamic_speed_mps").value)
        self.track_ttl_scans = int(self.get_parameter("track_ttl_scans").value)
        self.dynamic_hold_scans = int(self.get_parameter("dynamic_hold_scans").value)
        self.max_front_angle = math.radians(float(self.get_parameter("max_front_angle_deg").value))
        self.measurement_variance = float(self.get_parameter("measurement_variance").value)
        self.process_variance_s = float(self.get_parameter("process_variance_s").value)
        self.process_variance_d = float(self.get_parameter("process_variance_d").value)
        self.confidence_green_threshold = float(self.get_parameter("confidence_green_threshold").value)

        self.track_points = np.asarray(
            load_manual_reference_line(int(self.get_parameter("num_path_points").value)), dtype=float
        )
        rolled = np.roll(self.track_points, -1, axis=0)
        self.segment_vectors = rolled - self.track_points
        self.segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)
        self.segment_lengths = np.maximum(self.segment_lengths, 1e-6)
        self.cum_s = np.concatenate([[0.0], np.cumsum(self.segment_lengths[:-1])])
        self.track_length = float(np.sum(self.segment_lengths))
        self.tangents = self.segment_vectors / self.segment_lengths[:, None]
        self.normals = np.column_stack([-self.tangents[:, 1], self.tangents[:, 0]])
        self.track_tree = cKDTree(self.track_points)
        self.left_boundaries, self.right_boundaries, wall_count = self._build_boundary_profile()

        self.ego_position: tuple[float, float] | None = None
        self.ego_yaw: float | None = None
        self.scan_frame_id = "lidar_1"
        self.follow_active = False
        self.tracks: dict[int, TrackedObstacle] = {}
        self.next_track_id = 1
        self.selected_track_id: int | None = None

        self.target_point_pub = self.create_publisher(
            PointStamped, str(self.get_parameter("target_point_topic").value), 10
        )
        self.vector_pub = self.create_publisher(
            Vector3Stamped, str(self.get_parameter("tracking_vector_topic").value), 10
        )
        self.visible_pub = self.create_publisher(Bool, str(self.get_parameter("target_visible_topic").value), 10)
        self.arrow_pub = self.create_publisher(Marker, str(self.get_parameter("tracking_arrow_topic").value), 10)
        self.confidence_pub = self.create_publisher(
            Float32, str(self.get_parameter("tracking_confidence_topic").value), 10
        )
        self.debug_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("debug_markers_topic").value), 10
        )

        self.create_subscription(Point, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(Imu, self.imu_topic, self.imu_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.create_subscription(
            Bool, str(self.get_parameter("follow_active_topic").value), self.follow_active_cb, 10
        )

        self.get_logger().info(
            "Forza-style opponent tracker ready. "
            f"scan={self.scan_topic}, wall_points={wall_count}, track_length={self.track_length:.2f}m, "
            f"boundary_inflation={self.boundary_inflation_m:.2f}m"
        )

    def _load_wall_points(self) -> np.ndarray:
        if not self.wall_mask_csv_path.exists():
            self.get_logger().warn(
                f"Wall mask not found at {self.wall_mask_csv_path}; using fallback track width"
            )
            return np.empty((0, 2), dtype=float)
        points: list[tuple[float, float]] = []
        with self.wall_mask_csv_path.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                try:
                    points.append((float(row["world_x_m"]), float(row["world_y_m"])))
                except (KeyError, TypeError, ValueError):
                    continue
        return np.asarray(points, dtype=float).reshape(-1, 2)

    @staticmethod
    def _circular_fill(values: np.ndarray, fallback: float) -> np.ndarray:
        valid = np.flatnonzero(np.isfinite(values))
        if len(valid) == 0:
            return np.full_like(values, fallback)
        n_values = len(values)
        xp = np.concatenate([valid - n_values, valid, valid + n_values])
        fp = np.concatenate([values[valid], values[valid], values[valid]])
        filled = np.interp(np.arange(n_values), xp, fp)
        return median_filter(filled, size=nine_if_possible(n_values), mode="wrap")

    def _build_boundary_profile(self) -> tuple[np.ndarray, np.ndarray, int]:
        wall_points = self._load_wall_points()
        n_points = len(self.track_points)
        if len(wall_points) == 0:
            fallback = np.full(n_points, self.fallback_half_track_width_m, dtype=float)
            return fallback.copy(), fallback.copy(), 0

        _, indices = self.track_tree.query(wall_points, k=1)
        indices = np.asarray(indices, dtype=int)
        relative = wall_points - self.track_points[indices]
        lateral = np.sum(relative * self.normals[indices], axis=1)
        left_samples: list[list[float]] = [[] for _ in range(n_points)]
        right_samples: list[list[float]] = [[] for _ in range(n_points)]
        for idx, d_value in zip(indices, lateral):
            if 0.15 < d_value < 4.0:
                left_samples[int(idx)].append(float(d_value))
            elif -4.0 < d_value < -0.15:
                right_samples[int(idx)].append(float(-d_value))

        left = np.full(n_points, np.nan, dtype=float)
        right = np.full(n_points, np.nan, dtype=float)
        for idx in range(n_points):
            if left_samples[idx]:
                left[idx] = float(np.percentile(left_samples[idx], 15.0))
            if right_samples[idx]:
                right[idx] = float(np.percentile(right_samples[idx], 15.0))
        left = self._circular_fill(left, self.fallback_half_track_width_m)
        right = self._circular_fill(right, self.fallback_half_track_width_m)
        return left, right, len(wall_points)

    def pose_cb(self, msg: Point) -> None:
        self.ego_position = (float(msg.x), float(msg.y))

    def imu_cb(self, msg: Imu) -> None:
        self.ego_yaw = quaternion_to_yaw(
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        )

    def follow_active_cb(self, msg: Bool) -> None:
        self.follow_active = bool(msg.data)

    def project_to_track(self, x: float, y: float) -> tuple[float, float, int]:
        point = np.array([x, y], dtype=float)
        _, nearest_idx = self.track_tree.query(point, k=1)
        candidates = ((int(nearest_idx) - 1) % len(self.track_points), int(nearest_idx))
        best: tuple[float, float, float, int] | None = None
        for idx in candidates:
            vector = self.segment_vectors[idx]
            length_sq = float(np.dot(vector, vector))
            fraction = float(np.clip(np.dot(point - self.track_points[idx], vector) / length_sq, 0.0, 1.0))
            projection = self.track_points[idx] + fraction * vector
            distance_sq = float(np.dot(point - projection, point - projection))
            d_value = float(np.dot(point - projection, self.normals[idx]))
            s_value = float((self.cum_s[idx] + fraction * self.segment_lengths[idx]) % self.track_length)
            candidate = (distance_sq, s_value, d_value, idx)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        return best[1], best[2], best[3]

    def track_frame_at_s(self, s_value: float) -> tuple[np.ndarray, np.ndarray]:
        s_mod = float(s_value % self.track_length)
        idx = int(np.searchsorted(self.cum_s, s_mod, side="right") - 1)
        idx = max(0, min(idx, len(self.track_points) - 1))
        fraction = float(
            np.clip((s_mod - self.cum_s[idx]) / self.segment_lengths[idx], 0.0, 1.0)
        )
        center = self.track_points[idx] + fraction * self.segment_vectors[idx]
        return center, self.normals[idx]

    def is_inside_track(self, d_value: float, idx: int) -> bool:
        left_limit = max(0.05, float(self.left_boundaries[idx]) - self.boundary_inflation_m)
        right_limit = max(0.05, float(self.right_boundaries[idx]) - self.boundary_inflation_m)
        return -right_limit < d_value < left_limit

    def local_to_world(self, x_local: float, y_local: float) -> tuple[float, float]:
        assert self.ego_position is not None and self.ego_yaw is not None
        cos_yaw = math.cos(self.ego_yaw)
        sin_yaw = math.sin(self.ego_yaw)
        return (
            self.ego_position[0] + x_local * cos_yaw - y_local * sin_yaw,
            self.ego_position[1] + x_local * sin_yaw + y_local * cos_yaw,
        )

    def world_to_local(self, x_world: float, y_world: float) -> tuple[float, float]:
        assert self.ego_position is not None and self.ego_yaw is not None
        dx = x_world - self.ego_position[0]
        dy = y_world - self.ego_position[1]
        cos_yaw = math.cos(self.ego_yaw)
        sin_yaw = math.sin(self.ego_yaw)
        return (dx * cos_yaw + dy * sin_yaw, -dx * sin_yaw + dy * cos_yaw)

    def adaptive_clusters(self, msg: LaserScan) -> list[np.ndarray]:
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + np.arange(len(ranges), dtype=float) * msg.angle_increment
        valid = np.isfinite(ranges) & (ranges >= max(msg.range_min, 0.05)) & (
            ranges <= min(msg.range_max, self.max_viewing_distance_m)
        )
        clusters: list[list[tuple[float, float, float]]] = []
        current: list[tuple[float, float, float]] = []
        previous: tuple[float, float, float] | None = None
        denominator = max(math.sin(max(self.lambda_angle - abs(msg.angle_increment), 1e-4)), 1e-4)

        for idx in range(len(ranges)):
            if not valid[idx]:
                if current:
                    clusters.append(current)
                    current = []
                previous = None
                continue
            radius = float(ranges[idx])
            point = (radius * math.cos(float(angles[idx])), radius * math.sin(float(angles[idx])), radius)
            if previous is not None:
                gap = math.hypot(point[0] - previous[0], point[1] - previous[1])
                adaptive_gap = (
                    min(radius, previous[2]) * math.sin(abs(msg.angle_increment)) / denominator
                    + 3.0 * self.lidar_sigma_m
                ) / 2.0
                adaptive_gap = max(self.min_two_point_distance_m, adaptive_gap)
                if gap > adaptive_gap:
                    if current:
                        clusters.append(current)
                    current = []
            current.append(point)
            previous = point
        if current:
            clusters.append(current)
        return [np.asarray(cluster, dtype=float) for cluster in clusters]

    def detect_obstacles(self, msg: LaserScan) -> list[Detection]:
        detections: list[Detection] = []
        for cluster in self.adaptive_clusters(msg):
            if len(cluster) < self.min_cluster_beams:
                continue
            local_xy = cluster[:, :2]
            size_m = float(math.hypot(np.ptp(local_xy[:, 0]), np.ptp(local_xy[:, 1])))
            if size_m > self.max_cluster_size_m:
                continue
            surface = np.median(local_xy, axis=0)
            range_m = float(math.hypot(float(surface[0]), float(surface[1])))
            surface_world = np.asarray(
                self.local_to_world(float(surface[0]), float(surface[1])), dtype=float
            )
            _, _, surface_track_idx = self.project_to_track(float(surface_world[0]), float(surface_world[1]))
            ego_world = np.asarray(self.ego_position, dtype=float)
            line_of_sight = surface_world - ego_world
            line_of_sight /= max(float(np.linalg.norm(line_of_sight)), 1e-6)
            support_distance = (
                0.5 * self.opponent_length_m * abs(float(np.dot(line_of_sight, self.tangents[surface_track_idx])))
                + 0.5 * self.opponent_width_m * abs(float(np.dot(line_of_sight, self.normals[surface_track_idx])))
            )
            center_world = surface_world + support_distance * line_of_sight
            world_x, world_y = float(center_world[0]), float(center_world[1])
            local_x, local_y = self.world_to_local(world_x, world_y)
            s_value, d_value, track_idx = self.project_to_track(world_x, world_y)
            if not self.is_inside_track(d_value, track_idx):
                continue
            detections.append(
                Detection(
                    local_x=local_x,
                    local_y=local_y,
                    world_x=world_x,
                    world_y=world_y,
                    s=s_value,
                    d=d_value,
                    size_m=size_m,
                    beam_count=len(cluster),
                    range_m=range_m,
                )
            )
        return detections

    def predict_track(self, track: TrackedObstacle, stamp_sec: float) -> None:
        dt = min(max(stamp_sec - track.last_stamp, 1e-3), 0.25)
        transition = np.array(
            [[1.0, dt, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, dt], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        process_noise = np.diag(
            [0.25 * dt**4 * self.process_variance_s, dt**2 * self.process_variance_s,
             0.25 * dt**4 * self.process_variance_d, dt**2 * self.process_variance_d]
        )
        track.state = transition @ track.state
        track.covariance = transition @ track.covariance @ transition.T + process_noise
        track.last_stamp = stamp_sec
        track.updated_this_scan = False

    def update_track(self, track: TrackedObstacle, detection: Detection, stamp_sec: float) -> None:
        previous_history = track.history[-1] if track.history else None
        measured_s = track.state[0] + circular_delta(detection.s, track.state[0] % self.track_length, self.track_length)
        measurement = np.array([measured_s, detection.d], dtype=float)
        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=float)
        measurement_noise = np.eye(2, dtype=float) * self.measurement_variance
        innovation = measurement - observation @ track.state
        innovation_covariance = observation @ track.covariance @ observation.T + measurement_noise
        kalman_gain = track.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        track.state = track.state + kalman_gain @ innovation
        track.covariance = (np.eye(4) - kalman_gain @ observation) @ track.covariance

        if previous_history is not None and track.hits == 1:
            previous_t, previous_s, previous_d = previous_history
            dt = max(stamp_sec - previous_t, 1e-3)
            track.state[1] = circular_delta(detection.s, previous_s, self.track_length) / dt
            track.state[3] = (detection.d - previous_d) / dt

        track.history.append((stamp_sec, detection.s, detection.d))
        track.hits += 1
        track.misses = 0
        track.size_m = detection.size_m
        track.beam_count = detection.beam_count
        track.updated_this_scan = True
        self.classify_track(track)

    def classify_track(self, track: TrackedObstacle) -> None:
        if len(track.history) < self.min_classification_measurements:
            return
        history = list(track.history)[-20:]
        reference_s = history[0][1]
        s_values = np.array(
            [reference_s + circular_delta(sample[1], reference_s, self.track_length) for sample in history], dtype=float
        )
        d_values = np.array([sample[2] for sample in history], dtype=float)
        position_std = max(float(np.std(s_values)), float(np.std(d_values)))
        if position_std < self.static_std_threshold_m:
            track.static_votes += 1
        elif position_std > self.dynamic_std_threshold_m:
            track.dynamic_votes += 1

        if track.dynamic_votes >= 2 and abs(float(track.state[1])) >= self.min_dynamic_speed_mps:
            track.classification = "dynamic"
        elif track.static_votes >= 2 and track.static_votes >= track.dynamic_votes:
            track.classification = "static"

    def new_track(self, detection: Detection, stamp_sec: float) -> None:
        track = TrackedObstacle(
            track_id=self.next_track_id,
            state=np.array([detection.s, 0.0, detection.d, 0.0], dtype=float),
            covariance=np.diag([0.04, 1.0, 0.04, 1.0]),
            last_stamp=stamp_sec,
            size_m=detection.size_m,
            beam_count=detection.beam_count,
        )
        track.history.append((stamp_sec, detection.s, detection.d))
        self.tracks[track.track_id] = track
        self.next_track_id += 1

    def associate(self, detections: list[Detection], stamp_sec: float) -> None:
        for track in self.tracks.values():
            self.predict_track(track, stamp_sec)

        pairs: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            multiplier = self.dynamic_association_multiplier if track.classification == "dynamic" else 1.0
            max_distance = self.association_distance_m * multiplier
            for detection_idx, detection in enumerate(detections):
                ds = circular_delta(detection.s, track.state[0] % self.track_length, self.track_length)
                dd = detection.d - track.state[2]
                distance = math.hypot(ds, dd)
                if distance <= max_distance:
                    pairs.append((distance, track_id, detection_idx))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_id, detection_idx in sorted(pairs):
            if track_id in used_tracks or detection_idx in used_detections:
                continue
            self.update_track(self.tracks[track_id], detections[detection_idx], stamp_sec)
            used_tracks.add(track_id)
            used_detections.add(detection_idx)

        for track_id, track in list(self.tracks.items()):
            if track_id not in used_tracks:
                track.misses += 1
            ttl = self.dynamic_hold_scans if track.classification == "dynamic" else self.track_ttl_scans
            if track.misses > ttl:
                del self.tracks[track_id]
                if self.selected_track_id == track_id:
                    self.selected_track_id = None

        for detection_idx, detection in enumerate(detections):
            if detection_idx not in used_detections:
                self.new_track(detection, stamp_sec)

    def track_world_position(self, track: TrackedObstacle) -> tuple[float, float]:
        center, normal = self.track_frame_at_s(float(track.state[0]))
        point = center + float(track.state[2]) * normal
        return float(point[0]), float(point[1])

    def select_target(self) -> TrackedObstacle | None:
        candidates: list[tuple[float, TrackedObstacle]] = []
        for track in self.tracks.values():
            if track.classification != "dynamic":
                continue
            world_x, world_y = self.track_world_position(track)
            local_x, local_y = self.world_to_local(world_x, world_y)
            angle = abs(math.atan2(local_y, local_x))
            distance = math.hypot(local_x, local_y)
            if local_x <= 0.0 or angle > self.max_front_angle or distance > self.max_viewing_distance_m:
                continue
            continuity_bonus = 4.0 if track.track_id == self.selected_track_id else 0.0
            score = continuity_bonus + min(track.hits, 20) / 10.0 + min(abs(float(track.state[1])), 5.0) / 5.0
            score -= 0.05 * distance + 0.2 * track.misses
            candidates.append((score, track))
        if not candidates:
            return None
        selected = max(candidates, key=lambda item: item[0])[1]
        self.selected_track_id = selected.track_id
        return selected

    def confidence_for(self, track: TrackedObstacle) -> float:
        history_score = min(len(track.history) / max(self.min_classification_measurements * 2, 1), 1.0)
        speed_score = float(np.clip((abs(track.state[1]) - self.min_dynamic_speed_mps) / 0.8, 0.0, 1.0))
        variance_score = float(np.clip(1.0 - math.sqrt(max(track.covariance[0, 0], 0.0)) / 0.8, 0.0, 1.0))
        visibility_score = max(0.0, 1.0 - track.misses / max(self.dynamic_hold_scans, 1))
        return float(np.clip(0.30 * history_score + 0.30 * speed_score + 0.25 * variance_score + 0.15 * visibility_score, 0.0, 0.99))

    def publish_debug(self, detections: list[Detection], stamp) -> None:
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)
        for marker_id, detection in enumerate(detections, start=1):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "forza_detections"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = detection.world_x
            marker.pose.position.y = detection.world_y
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.16
            marker.color.a = 0.8
            marker.color.r = 0.1
            marker.color.g = 0.8
            marker.color.b = 1.0
            markers.markers.append(marker)
        self.debug_pub.publish(markers)

    def publish_target(self, target: TrackedObstacle | None, stamp) -> None:
        confidence = self.confidence_for(target) if target is not None else 0.0
        measurement_fresh = target is not None and target.updated_this_scan
        self.confidence_pub.publish(Float32(data=confidence))
        self.visible_pub.publish(Bool(data=measurement_fresh))

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
        label.header = arrow.header
        label.ns = arrow.ns
        label.id = 2
        label.type = Marker.TEXT_VIEW_FACING
        label.scale.z = 0.35
        label.color.a = 1.0

        if target is None:
            arrow.action = Marker.DELETE
            label.action = Marker.DELETE
            self.arrow_pub.publish(arrow)
            self.arrow_pub.publish(label)
            return

        world_x, world_y = self.track_world_position(target)
        local_x, local_y = self.world_to_local(world_x, world_y)
        if self.follow_active:
            arrow.color.r, arrow.color.g, arrow.color.b = 0.1, 0.3, 1.0
        elif confidence > self.confidence_green_threshold:
            arrow.color.r, arrow.color.g, arrow.color.b = 0.1, 1.0, 0.1
        else:
            arrow.color.r, arrow.color.g, arrow.color.b = 1.0, 0.1, 0.1
        label.color.r = arrow.color.r
        label.color.g = arrow.color.g
        label.color.b = arrow.color.b

        origin = Point(x=0.0, y=0.0, z=0.0)
        endpoint = Point(x=local_x, y=local_y, z=0.0)
        arrow.action = Marker.ADD
        arrow.points = [origin, endpoint]
        self.arrow_pub.publish(arrow)
        label.action = Marker.ADD
        label.pose.position.x = local_x + 0.20
        label.pose.position.y = local_y + 0.20
        label.pose.position.z = 0.20
        label.pose.orientation.w = 1.0
        label.text = f"{confidence:.2f}"
        self.arrow_pub.publish(label)

        # Keep the held RViz estimate visible, but never train the GP on a prediction.
        if not measurement_fresh:
            return

        target_point = PointStamped()
        target_point.header = arrow.header
        target_point.point = endpoint
        self.target_point_pub.publish(target_point)
        vector = Vector3Stamped()
        vector.header = arrow.header
        norm = max(math.hypot(local_x, local_y), 1e-6)
        vector.vector.x = local_x / norm
        vector.vector.y = local_y / norm
        self.vector_pub.publish(vector)

    def scan_cb(self, msg: LaserScan) -> None:
        if self.ego_position is None or self.ego_yaw is None:
            return
        self.scan_frame_id = msg.header.frame_id or self.scan_frame_id
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp_sec <= 0.0:
            stamp_sec = self.get_clock().now().nanoseconds * 1e-9
        detections = self.detect_obstacles(msg)
        self.associate(detections, stamp_sec)
        target = self.select_target()
        stamp = msg.header.stamp if msg.header.stamp.sec > 0 else self.get_clock().now().to_msg()
        self.publish_debug(detections, stamp)
        self.publish_target(target, stamp)


def nine_if_possible(length: int) -> int:
    if length >= 9:
        return 9
    return max(1, length if length % 2 == 1 else length - 1)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ForzaOpponentTracker()
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

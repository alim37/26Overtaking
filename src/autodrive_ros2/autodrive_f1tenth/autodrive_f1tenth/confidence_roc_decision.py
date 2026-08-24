#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, Int32
from visualization_msgs.msg import Marker, MarkerArray

from autodrive_f1tenth.pure_pursuit import load_manual_reference_line


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class ConfidenceRocDecision(Node):
    """Select track regions whose six-second target-tracking confidence stays high."""

    def __init__(self) -> None:
        super().__init__("confidence_roc_decision")

        package_root = Path(__file__).resolve().parents[1]
        output_root = package_root / "output"
        self.declare_parameter("pose_topic", "/autodrive/f1tenth_1/ips")
        self.declare_parameter("imu_topic", "/autodrive/f1tenth_1/imu")
        self.declare_parameter("target_point_topic", "/autodrive/f1tenth_1/target_tracker/target_point")
        self.declare_parameter("target_visible_topic", "/autodrive/f1tenth_1/target_tracker/target_visible")
        self.declare_parameter("tracking_confidence_topic", "/autodrive/f1tenth_1/target_tracker/tracking_confidence")
        self.declare_parameter("lap_count_topic", "/autodrive/f1tenth_1/pure_pursuit/lap_count")
        self.declare_parameter(
            "wall_mask_csv_path",
            str(output_root / "slam_runs" / "slam_toolbox_boundary_wall_mask.csv"),
        )
        self.declare_parameter(
            "figure_output_path",
            str(output_root / "confidence_roc" / "confidence_roc_two_lap.png"),
        )
        self.declare_parameter("num_path_points", 800)
        self.declare_parameter("bin_size_m", 0.25)
        self.declare_parameter("prediction_horizon_sec", 6.0)
        self.declare_parameter("prediction_dt_sec", 0.25)
        self.declare_parameter("confidence_threshold", 0.55)
        self.declare_parameter("confidence_exit_threshold", 0.42)
        self.declare_parameter("speed_penalty_per_meter", 0.010)
        self.declare_parameter("mean_confidence_weight", 0.70)
        self.declare_parameter("low_confidence_weight", 0.30)
        self.declare_parameter("current_confidence_weight", 0.75)
        self.declare_parameter("curvature_penalty_gain", 3.0)
        self.declare_parameter("current_curvature_weight", 0.90)
        self.declare_parameter("future_curvature_percentile", 75.0)
        self.declare_parameter("curvature_smoothing_m", 1.50)
        self.declare_parameter("low_confidence_percentile", 20.0)
        self.declare_parameter("min_samples_per_bin", 1)
        self.declare_parameter("min_profile_coverage_ratio", 0.25)
        self.declare_parameter("min_region_length_m", 1.0)
        self.declare_parameter("max_region_gap_m", 1.50)
        self.declare_parameter("wrap_prediction_horizon", False)
        self.declare_parameter("wall_inset_m", 0.05)
        self.declare_parameter("fallback_half_track_width_m", 0.70)
        self.declare_parameter("figure_after_laps", 2)
        self.declare_parameter("score_topic", "/autodrive/f1tenth_1/confidence_roc/score")
        self.declare_parameter("overtake_allowed_topic", "/autodrive/f1tenth_1/confidence_roc/overtake_allowed")
        self.declare_parameter("region_markers_topic", "/autodrive/f1tenth_1/confidence_roc/regions")
        self.declare_parameter("roc_start_topic", "/autodrive/f1tenth_1/confidence_roc/c_start_s")
        self.declare_parameter("roc_end_topic", "/autodrive/f1tenth_1/confidence_roc/c_end_s")

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.target_point_topic = str(self.get_parameter("target_point_topic").value)
        self.target_visible_topic = str(self.get_parameter("target_visible_topic").value)
        self.tracking_confidence_topic = str(self.get_parameter("tracking_confidence_topic").value)
        self.lap_count_topic = str(self.get_parameter("lap_count_topic").value)
        self.wall_mask_csv_path = Path(str(self.get_parameter("wall_mask_csv_path").value)).expanduser()
        self.figure_output_path = Path(str(self.get_parameter("figure_output_path").value)).expanduser()
        self.bin_size_m = float(self.get_parameter("bin_size_m").value)
        self.prediction_horizon_sec = float(self.get_parameter("prediction_horizon_sec").value)
        self.prediction_dt_sec = float(self.get_parameter("prediction_dt_sec").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.confidence_exit_threshold = float(self.get_parameter("confidence_exit_threshold").value)
        self.speed_penalty_per_meter = float(self.get_parameter("speed_penalty_per_meter").value)
        self.mean_confidence_weight = float(self.get_parameter("mean_confidence_weight").value)
        self.low_confidence_weight = float(self.get_parameter("low_confidence_weight").value)
        self.current_confidence_weight = float(self.get_parameter("current_confidence_weight").value)
        self.curvature_penalty_gain = float(self.get_parameter("curvature_penalty_gain").value)
        self.current_curvature_weight = float(self.get_parameter("current_curvature_weight").value)
        self.future_curvature_percentile = float(self.get_parameter("future_curvature_percentile").value)
        self.curvature_smoothing_m = float(self.get_parameter("curvature_smoothing_m").value)
        self.low_confidence_percentile = float(self.get_parameter("low_confidence_percentile").value)
        self.min_samples_per_bin = int(self.get_parameter("min_samples_per_bin").value)
        self.min_profile_coverage_ratio = float(self.get_parameter("min_profile_coverage_ratio").value)
        self.min_region_length_m = float(self.get_parameter("min_region_length_m").value)
        self.max_region_gap_m = float(self.get_parameter("max_region_gap_m").value)
        self.wrap_prediction_horizon = bool(self.get_parameter("wrap_prediction_horizon").value)
        self.wall_inset_m = float(self.get_parameter("wall_inset_m").value)
        self.fallback_half_track_width_m = float(self.get_parameter("fallback_half_track_width_m").value)
        self.figure_after_laps = int(self.get_parameter("figure_after_laps").value)

        self.track_points = np.asarray(
            load_manual_reference_line(int(self.get_parameter("num_path_points").value)), dtype=float
        )
        rolled = np.roll(self.track_points, -1, axis=0)
        self.segment_vectors = rolled - self.track_points
        self.segment_lengths = np.maximum(np.linalg.norm(self.segment_vectors, axis=1), 1e-6)
        self.cum_s = np.concatenate([[0.0], np.cumsum(self.segment_lengths[:-1])])
        self.track_length = float(np.sum(self.segment_lengths))
        self.tangents = self.segment_vectors / self.segment_lengths[:, None]
        self.normals = np.column_stack([-self.tangents[:, 1], self.tangents[:, 0]])
        self.track_tree = cKDTree(self.track_points)
        self.num_bins = max(8, int(math.ceil(self.track_length / max(self.bin_size_m, 1e-3))))

        self.wall_points = self.load_wall_points()
        self.left_widths, self.right_widths = self.build_track_width_profile()
        self.profile_curvature = self.build_curvature_profile()
        self.confidence_samples: dict[int, list[float]] = defaultdict(list)
        self.visibility_samples: dict[int, list[float]] = defaultdict(list)
        self.speed_samples: dict[int, list[float]] = defaultdict(list)
        self.opponent_observations: deque[tuple[float, float]] = deque(maxlen=6000)
        self.ego_trajectory: deque[tuple[float, float]] = deque(maxlen=6000)
        self.trajectory_samples: deque[tuple[float, ...]] = deque(maxlen=6000)
        self.speed_history: deque[float] = deque(maxlen=15)

        self.ego_position: tuple[float, float] | None = None
        self.ego_yaw: float | None = None
        self.previous_pose: tuple[float, float] | None = None
        self.previous_pose_time: float | None = None
        self.latest_confidence = 0.0
        self.target_visible = False
        self.lap_count = 0
        self.figure_saved = False
        self.profile_ready = False
        self.profile_coverage = 0.0
        self.profile_confidence = np.zeros(self.num_bins, dtype=float)
        self.profile_speed = np.zeros(self.num_bins, dtype=float)
        self.profile_score = np.zeros(self.num_bins, dtype=float)
        self.regions: list[tuple[float, float]] = []
        self.c_start_s: float | None = None
        self.c_end_s: float | None = None

        self.score_pub = self.create_publisher(Float32, str(self.get_parameter("score_topic").value), 10)
        self.allowed_pub = self.create_publisher(Bool, str(self.get_parameter("overtake_allowed_topic").value), 10)
        self.roc_start_pub = self.create_publisher(Float32, str(self.get_parameter("roc_start_topic").value), 10)
        self.roc_end_pub = self.create_publisher(Float32, str(self.get_parameter("roc_end_topic").value), 10)
        self.region_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("region_markers_topic").value), 10
        )
        self.create_subscription(Point, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(Imu, self.imu_topic, self.imu_cb, 10)
        self.create_subscription(PointStamped, self.target_point_topic, self.target_point_cb, 10)
        self.create_subscription(Bool, self.target_visible_topic, self.visible_cb, 10)
        self.create_subscription(Float32, self.tracking_confidence_topic, self.confidence_cb, 10)
        self.create_subscription(Int32, self.lap_count_topic, self.lap_count_cb, 10)
        self.create_timer(0.5, self.update_decision)

        self.get_logger().info(
            "Confidence RoC ready. "
            f"horizon={self.prediction_horizon_sec:.1f}s, threshold={self.confidence_threshold:.2f}, "
            f"bins={self.num_bins}, figure={self.figure_output_path}"
        )

    def load_wall_points(self) -> np.ndarray:
        if not self.wall_mask_csv_path.exists():
            self.get_logger().warn(f"Wall mask not found at {self.wall_mask_csv_path}")
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
    def fill_circular(values: np.ndarray, fallback: float) -> np.ndarray:
        valid = np.flatnonzero(np.isfinite(values))
        if len(valid) == 0:
            return np.full_like(values, fallback)
        count = len(values)
        xp = np.concatenate([valid - count, valid, valid + count])
        fp = np.concatenate([values[valid], values[valid], values[valid]])
        return np.interp(np.arange(count), xp, fp)

    def build_track_width_profile(self) -> tuple[np.ndarray, np.ndarray]:
        fallback = np.full(len(self.track_points), self.fallback_half_track_width_m, dtype=float)
        if len(self.wall_points) == 0:
            return fallback.copy(), fallback.copy()
        _, indices = self.track_tree.query(self.wall_points, k=1)
        indices = np.asarray(indices, dtype=int)
        relative = self.wall_points - self.track_points[indices]
        lateral = np.sum(relative * self.normals[indices], axis=1)
        left_samples: list[list[float]] = [[] for _ in self.track_points]
        right_samples: list[list[float]] = [[] for _ in self.track_points]
        for idx, d_value in zip(indices, lateral):
            if 0.15 < d_value < 4.0:
                left_samples[int(idx)].append(float(d_value))
            elif -4.0 < d_value < -0.15:
                right_samples[int(idx)].append(float(-d_value))
        left = np.full(len(self.track_points), np.nan, dtype=float)
        right = np.full(len(self.track_points), np.nan, dtype=float)
        for idx in range(len(self.track_points)):
            if left_samples[idx]:
                left[idx] = float(np.percentile(left_samples[idx], 10.0))
            if right_samples[idx]:
                right[idx] = float(np.percentile(right_samples[idx], 10.0))
        left = median_filter(self.fill_circular(left, self.fallback_half_track_width_m), 21, mode="wrap")
        right = median_filter(self.fill_circular(right, self.fallback_half_track_width_m), 21, mode="wrap")
        return np.clip(left, 0.20, 3.0), np.clip(right, 0.20, 3.0)

    def build_curvature_profile(self) -> np.ndarray:
        """Return absolute centerline curvature in 1/m at each confidence bin."""
        heading = np.unwrap(np.arctan2(self.tangents[:, 1], self.tangents[:, 0]))
        curvature = np.abs(np.gradient(heading, self.cum_s, edge_order=1))
        curvature = np.clip(curvature, 0.0, 2.0)
        sample_spacing = float(np.median(self.segment_lengths))
        window = max(3, int(round(self.curvature_smoothing_m / max(sample_spacing, 1e-3))))
        if window % 2 == 0:
            window += 1
        curvature = median_filter(curvature, size=window, mode="nearest")
        centers = (np.arange(self.num_bins, dtype=float) + 0.5) * self.bin_size_m
        return np.interp(
            centers,
            self.cum_s,
            curvature,
            left=float(curvature[0]),
            right=float(curvature[-1]),
        )

    def project_to_track(self, x: float, y: float) -> tuple[float, int]:
        point = np.array([x, y], dtype=float)
        _, nearest = self.track_tree.query(point, k=1)
        best: tuple[float, float, int] | None = None
        for idx in ((int(nearest) - 1) % len(self.track_points), int(nearest)):
            vector = self.segment_vectors[idx]
            fraction = float(
                np.clip(
                    np.dot(point - self.track_points[idx], vector) / np.dot(vector, vector),
                    0.0,
                    1.0,
                )
            )
            projected = self.track_points[idx] + fraction * vector
            distance_sq = float(np.dot(point - projected, point - projected))
            s_value = float((self.cum_s[idx] + fraction * self.segment_lengths[idx]) % self.track_length)
            candidate = (distance_sq, s_value, idx)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        return best[1], best[2]

    def pose_cb(self, msg: Point) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        position = (float(msg.x), float(msg.y))
        if self.previous_pose is not None and self.previous_pose_time is not None:
            dt = now - self.previous_pose_time
            if dt > 1e-3:
                speed = math.hypot(position[0] - self.previous_pose[0], position[1] - self.previous_pose[1]) / dt
                if math.isfinite(speed) and speed < 15.0:
                    self.speed_history.append(speed)
        self.previous_pose = position
        self.previous_pose_time = now
        self.ego_position = position
        if not self.ego_trajectory or math.dist(position, self.ego_trajectory[-1]) >= 0.02:
            self.ego_trajectory.append(position)
        s_value, _ = self.project_to_track(*position)
        bin_idx = min(self.num_bins - 1, int(s_value / self.bin_size_m))
        speed = float(np.median(self.speed_history)) if self.speed_history else 0.0
        self.confidence_samples[bin_idx].append(float(np.clip(self.latest_confidence, 0.0, 0.99)))
        self.visibility_samples[bin_idx].append(float(self.target_visible))
        self.speed_samples[bin_idx].append(speed)

    def imu_cb(self, msg: Imu) -> None:
        self.ego_yaw = quaternion_to_yaw(
            float(msg.orientation.x), float(msg.orientation.y),
            float(msg.orientation.z), float(msg.orientation.w),
        )

    def confidence_cb(self, msg: Float32) -> None:
        self.latest_confidence = float(msg.data)

    def visible_cb(self, msg: Bool) -> None:
        self.target_visible = bool(msg.data)

    def target_point_cb(self, msg: PointStamped) -> None:
        if not self.target_visible or self.ego_position is None or self.ego_yaw is None:
            return
        cos_yaw = math.cos(self.ego_yaw)
        sin_yaw = math.sin(self.ego_yaw)
        x_local = float(msg.point.x)
        y_local = float(msg.point.y)
        opponent_position = (
            self.ego_position[0] + x_local * cos_yaw - y_local * sin_yaw,
            self.ego_position[1] + x_local * sin_yaw + y_local * cos_yaw,
        )
        self.opponent_observations.append(opponent_position)
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if stamp_sec <= 0.0:
            stamp_sec = self.get_clock().now().nanoseconds * 1e-9
        ego_speed = float(np.median(self.speed_history)) if self.speed_history else 0.0
        ego_s, _ = self.project_to_track(*self.ego_position)
        self.trajectory_samples.append(
            (
                stamp_sec,
                self.ego_position[0],
                self.ego_position[1],
                self.ego_yaw,
                ego_speed,
                opponent_position[0],
                opponent_position[1],
                float(np.clip(self.latest_confidence, 0.0, 0.99)),
                ego_s,
            )
        )

    def lap_count_cb(self, msg: Int32) -> None:
        self.lap_count = max(self.lap_count, int(msg.data))

    def periodic_interp(self, profile: np.ndarray, s_values: np.ndarray) -> np.ndarray:
        centers = (np.arange(self.num_bins, dtype=float) + 0.5) * self.bin_size_m
        centers %= self.track_length
        xp = np.concatenate([centers - self.track_length, centers, centers + self.track_length])
        fp = np.concatenate([profile, profile, profile])
        return np.interp(np.mod(s_values, self.track_length), xp, fp)

    def build_profiles(self, allow_partial: bool = False) -> bool:
        confidence = np.full(self.num_bins, np.nan, dtype=float)
        visibility = np.full(self.num_bins, np.nan, dtype=float)
        speed = np.full(self.num_bins, np.nan, dtype=float)
        valid_count = 0
        for idx in range(self.num_bins):
            if len(self.confidence_samples[idx]) < self.min_samples_per_bin:
                continue
            confidence[idx] = float(np.median(self.confidence_samples[idx]))
            visibility[idx] = float(np.mean(self.visibility_samples[idx]))
            speed[idx] = float(np.median(self.speed_samples[idx]))
            valid_count += 1
        coverage = valid_count / max(self.num_bins, 1)
        self.profile_coverage = coverage
        if valid_count == 0 or (not allow_partial and coverage < self.min_profile_coverage_ratio):
            return False
        fallback_speed = float(np.nanmedian(speed)) if np.any(np.isfinite(speed)) else 0.0
        confidence = self.fill_circular(confidence, 0.0)
        visibility = self.fill_circular(visibility, 0.0)
        speed = self.fill_circular(speed, fallback_speed)
        # Confidence already includes tracker persistence and visibility. Apply only a
        # light visibility term here so isolated missed frames do not split the RoC.
        self.profile_confidence = median_filter(confidence * (0.85 + 0.15 * visibility), 9, mode="nearest")
        self.profile_speed = median_filter(speed, 9, mode="nearest")
        return True

    def calculate_horizon_scores(self) -> np.ndarray:
        scores = np.zeros(self.num_bins, dtype=float)
        times = np.arange(0.0, self.prediction_horizon_sec + self.prediction_dt_sec, self.prediction_dt_sec)
        centers = (np.arange(self.num_bins, dtype=float) + 0.5) * self.bin_size_m
        for idx in range(self.num_bins):
            start_s = (idx + 0.5) * self.bin_size_m
            ego_speed = max(0.0, float(self.profile_speed[idx]))
            future_s = start_s + ego_speed * times
            if self.wrap_prediction_horizon:
                future_confidence = self.periodic_interp(self.profile_confidence, future_s)
            else:
                # The experiment ends after one lap. Do not let low confidence near
                # the beginning of the next lap suppress the final overtake corridor.
                future_s = future_s[future_s <= self.track_length]
                if len(future_s) == 0:
                    future_s = np.array([start_s], dtype=float)
                future_confidence = np.interp(
                    future_s,
                    centers,
                    self.profile_confidence,
                    left=float(self.profile_confidence[0]),
                    right=float(self.profile_confidence[-1]),
                )
            robust_confidence = (
                self.mean_confidence_weight * float(np.mean(future_confidence))
                + self.low_confidence_weight
                * float(np.percentile(future_confidence, self.low_confidence_percentile))
            )
            current_confidence = float(self.profile_confidence[idx])
            confidence_score = (
                self.current_confidence_weight * current_confidence
                + (1.0 - self.current_confidence_weight) * robust_confidence
            )
            future_curvature = np.interp(
                future_s,
                centers,
                self.profile_curvature,
                left=float(self.profile_curvature[0]),
                right=float(self.profile_curvature[-1]),
            )
            robust_curvature = float(
                np.percentile(future_curvature, self.future_curvature_percentile)
            )
            current_curvature_factor = math.exp(
                -self.curvature_penalty_gain * float(self.profile_curvature[idx])
            )
            future_curvature_factor = math.exp(
                -self.curvature_penalty_gain * robust_curvature
            )
            curvature_factor = (
                self.current_curvature_weight * current_curvature_factor
                + (1.0 - self.current_curvature_weight) * future_curvature_factor
            )
            distance_travelled = ego_speed * self.prediction_horizon_sec
            speed_factor = math.exp(-self.speed_penalty_per_meter * distance_travelled)
            scores[idx] = confidence_score * speed_factor * curvature_factor
        return median_filter(scores, 9, mode="nearest")

    def extract_regions(self, scores: np.ndarray) -> list[tuple[float, float]]:
        high_mask = scores >= self.confidence_threshold
        low_mask = scores >= min(self.confidence_exit_threshold, self.confidence_threshold)
        gap_bins = max(1, int(round(self.max_region_gap_m / self.bin_size_m)))
        min_bins = max(1, int(math.ceil(self.min_region_length_m / self.bin_size_m)))
        mask = low_mask.copy()
        false_indices = np.flatnonzero(~mask)
        if len(false_indices):
            run_start = 0
            while run_start < len(false_indices):
                run_end = run_start + 1
                while (
                    run_end < len(false_indices)
                    and false_indices[run_end] == false_indices[run_end - 1] + 1
                ):
                    run_end += 1
                gap = false_indices[run_start:run_end]
                bounded = gap[0] > 0 and gap[-1] < self.num_bins - 1
                if bounded and len(gap) <= gap_bins:
                    mask[gap] = True
                run_start = run_end

        candidates: list[tuple[float, int, int]] = []
        idx = 0
        while idx < self.num_bins:
            if not mask[idx]:
                idx += 1
                continue
            start = idx
            while idx < self.num_bins and mask[idx]:
                idx += 1
            end = idx
            if end - start < min_bins or not np.any(high_mask[start:end]):
                continue
            length_m = min(self.track_length, (end - start) * self.bin_size_m)
            mean_score = float(np.mean(scores[start:end]))
            peak_score = float(np.max(scores[start:end]))
            candidates.append((length_m * (0.8 * mean_score + 0.2 * peak_score), start, end))

        if not candidates:
            return []

        # Planning needs one unambiguous corridor. The longest consistently strong
        # component wins; c_start/c_end are its measured threshold crossings.
        _, start, end = max(candidates, key=lambda item: item[0])
        return [(start * self.bin_size_m, min(end * self.bin_size_m, self.track_length))]

    def region_indices(self, start_s: float, end_s: float, count: int = 120) -> np.ndarray:
        length = (end_s - start_s) % self.track_length
        if math.isclose(length, 0.0) and (
            math.isclose(start_s, 0.0) and math.isclose(end_s, self.track_length)
        ):
            length = self.track_length
        samples = (start_s + np.linspace(0.0, length, count)) % self.track_length
        return np.array([int(np.argmin(np.abs(self.cum_s - sample))) for sample in samples], dtype=int)

    def publish_regions(self) -> None:
        marker_array = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        marker_array.markers.append(delete)
        stamp = self.get_clock().now().to_msg()
        for marker_id, (start_s, end_s) in enumerate(self.regions, start=1):
            indices = self.region_indices(start_s, end_s, count=80)
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "confidence_roc"
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.18
            marker.color.a = 1.0
            marker.color.r = 0.65
            marker.color.g = 0.15
            marker.color.b = 0.85
            for idx in indices:
                marker.points.append(Point(x=float(self.track_points[idx, 0]), y=float(self.track_points[idx, 1]), z=0.06))
            marker_array.markers.append(marker)
        self.region_pub.publish(marker_array)

    def update_decision(self) -> None:
        self.profile_ready = self.build_profiles()
        if not self.profile_ready:
            if self.lap_count < self.figure_after_laps:
                return
            if not self.build_profiles(allow_partial=True):
                self.get_logger().error("Cannot save confidence RoC figure: no populated confidence bins")
                return
            self.get_logger().warn(
                "Saving confidence RoC from partial profile: "
                f"coverage={self.profile_coverage:.1%}, required={self.min_profile_coverage_ratio:.1%}"
            )
        self.profile_score = self.calculate_horizon_scores()
        self.regions = self.extract_regions(self.profile_score)
        if self.regions:
            self.c_start_s, self.c_end_s = self.regions[0]
            self.roc_start_pub.publish(Float32(data=float(self.c_start_s)))
            self.roc_end_pub.publish(Float32(data=float(self.c_end_s)))
        else:
            self.c_start_s = None
            self.c_end_s = None
            self.roc_start_pub.publish(Float32(data=-1.0))
            self.roc_end_pub.publish(Float32(data=-1.0))
        if self.ego_position is not None:
            s_value, _ = self.project_to_track(*self.ego_position)
            bin_idx = min(self.num_bins - 1, int(s_value / self.bin_size_m))
            current_score = float(self.profile_score[bin_idx])
            self.score_pub.publish(Float32(data=current_score))
            self.allowed_pub.publish(Bool(data=current_score >= self.confidence_threshold))
        self.publish_regions()
        if self.lap_count >= self.figure_after_laps and not self.figure_saved:
            self.save_figure()

    def save_profile(self) -> None:
        profile_path = self.figure_output_path.with_name(
            f"{self.figure_output_path.stem}_profile.csv"
        )
        selected = np.zeros(self.num_bins, dtype=bool)
        if self.regions:
            start_s, end_s = self.regions[0]
            centers = (np.arange(self.num_bins, dtype=float) + 0.5) * self.bin_size_m
            selected = (centers >= start_s) & (centers <= end_s)
        with profile_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "s_m",
                    "confidence",
                    "speed_mps",
                    "curvature_1pm",
                    "horizon_score",
                    "selected_roc",
                ]
            )
            for idx in range(self.num_bins):
                writer.writerow(
                    [
                        (idx + 0.5) * self.bin_size_m,
                        float(self.profile_confidence[idx]),
                        float(self.profile_speed[idx]),
                        float(self.profile_curvature[idx]),
                        float(self.profile_score[idx]),
                        int(selected[idx]),
                    ]
                )

    def save_trajectory(self) -> None:
        trajectory_path = self.figure_output_path.with_name(
            f"{self.figure_output_path.stem}_trajectory.csv"
        )
        with trajectory_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "stamp_sec",
                    "ego_x_m",
                    "ego_y_m",
                    "ego_yaw_rad",
                    "ego_speed_mps",
                    "opponent_x_m",
                    "opponent_y_m",
                    "tracking_confidence",
                    "ego_s_m",
                ]
            )
            writer.writerows(self.trajectory_samples)

    def save_figure(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.figure_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(11, 7))
        if len(self.wall_points):
            ax.scatter(
                self.wall_points[:, 0], self.wall_points[:, 1], s=1.0,
                color="0.55", alpha=0.45, label="SLAM track boundary", rasterized=True,
            )
        if self.ego_trajectory:
            ego = np.asarray(self.ego_trajectory, dtype=float)
            ax.plot(
                ego[:, 0], ego[:, 1], color="#9a6700",
                linewidth=1.6, linestyle="--", label="Ego trajectory", zorder=3,
            )
        if self.opponent_observations:
            observations = np.asarray(self.opponent_observations, dtype=float)
            ax.scatter(
                observations[:, 0], observations[:, 1], s=8, color="#f2c94c",
                alpha=0.55, label="Observed opponent", zorder=4,
            )
        for region_index, (start_s, end_s) in enumerate(self.regions):
            indices = self.region_indices(start_s, end_s)
            centers = self.track_points[indices]
            normals = self.normals[indices]
            left = centers + np.maximum(self.left_widths[indices] - self.wall_inset_m, 0.05)[:, None] * normals
            right = centers - np.maximum(self.right_widths[indices] - self.wall_inset_m, 0.05)[:, None] * normals
            polygon = np.vstack([left, right[::-1]])
            ax.fill(
                polygon[:, 0], polygon[:, 1], color="#c77dff", alpha=0.30,
                label="Confidence-derived Region of Collision" if region_index == 0 else None,
                zorder=2,
            )
        if self.c_start_s is not None and self.c_end_s is not None:
            start_idx = int(np.argmin(np.abs(self.cum_s - self.c_start_s)))
            end_idx = int(np.argmin(np.abs(self.cum_s - min(self.c_end_s, self.cum_s[-1]))))
            start_point = self.track_points[start_idx]
            end_point = self.track_points[end_idx]
            ax.scatter(*start_point, s=90, color="#6f2dbd", zorder=6)
            ax.scatter(*end_point, s=90, color="#d0006f", zorder=6)
            ax.annotate(
                r"$c_{start}$", start_point, xytext=(8, 8), textcoords="offset points",
                color="#6f2dbd", fontsize=11,
            )
            ax.annotate(
                r"$c_{end}$", end_point, xytext=(8, 8), textcoords="offset points",
                color="#d0006f", fontsize=11,
            )
        if not self.regions:
            ax.text(0.5, 0.5, "No confidence-qualified region", transform=ax.transAxes, ha="center")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Confidence-and-Curvature Six-Second Overtake Region")
        ax.grid(alpha=0.20)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.figure_output_path, dpi=220, bbox_inches="tight")
        fig.savefig(self.figure_output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        self.save_profile()
        self.save_trajectory()
        self.figure_saved = True
        self.get_logger().info(f"Saved confidence RoC figure to {self.figure_output_path}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConfidenceRocDecision()
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

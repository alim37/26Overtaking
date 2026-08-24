#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from scipy.ndimage import median_filter
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, Int32
from visualization_msgs.msg import Marker

from autodrive_f1tenth.pure_pursuit import load_manual_reference_line


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "tracks" / "src").exists():
            return parent
    return Path.cwd()


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def circular_delta(a: float, b: float, length: float) -> float:
    delta = (a - b + 0.5 * length) % length - 0.5 * length
    return delta


def circular_interp(start_s: float, end_s: float, alpha: float, length: float) -> float:
    return (start_s + alpha * circular_delta(end_s, start_s, length)) % length


@dataclass
class TrackProjection:
    s: float
    d: float
    idx: int
    x: float
    y: float


@dataclass
class OpponentObservation:
    stamp_sec: float
    s: float
    d: float
    v_s: float
    x: float
    y: float


class OneDimensionalGaussianProcess:
    def __init__(self, kernel: str, length_scale: float, sigma_f: float, noise: float, period: float) -> None:
        self.kernel = kernel
        self.length_scale = float(max(length_scale, 1e-3))
        self.sigma_f = float(max(sigma_f, 1e-6))
        self.noise = float(max(noise, 1e-8))
        self.period = float(period)
        self.x_train: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.cho: tuple[np.ndarray, bool] | None = None

    def _kernel_fn(self, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        xa = xa.reshape(-1, 1)
        xb = xb.reshape(1, -1)
        dx = np.abs(xa - xb)
        dx = np.minimum(dx, self.period - dx)
        if self.kernel == "rbf":
            return (self.sigma_f**2) * np.exp(-0.5 * (dx / self.length_scale) ** 2)
        # Matern 3/2
        sqrt3 = math.sqrt(3.0)
        z = sqrt3 * dx / self.length_scale
        return (self.sigma_f**2) * (1.0 + z) * np.exp(-z)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) == 0:
            self.x_train = None
            self.alpha = None
            self.cho = None
            return
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        K = self._kernel_fn(x, x)
        K[np.diag_indices_from(K)] += self.noise
        cho = cho_factor(K, lower=True, check_finite=False)
        alpha = cho_solve(cho, y, check_finite=False)
        self.x_train = x
        self.alpha = alpha
        self.cho = cho

    def predict(self, x_star: np.ndarray, return_std: bool = False) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        x_star = np.asarray(x_star, dtype=float).reshape(-1)
        if self.x_train is None or self.alpha is None or self.cho is None:
            mean = np.zeros_like(x_star)
            std = np.full_like(x_star, 1e3)
            return (mean, std) if return_std else mean
        K_star = self._kernel_fn(x_star, self.x_train)
        mean = K_star @ self.alpha
        if not return_std:
            return mean
        v = cho_solve(self.cho, K_star.T, check_finite=False)
        K_ss = np.diag(self._kernel_fn(x_star, x_star))
        var = np.maximum(K_ss - np.sum(K_star * v.T, axis=1), 1e-9)
        return mean, np.sqrt(var)


class PredictiveSplinerDecision(Node):
    """
    Predictive Spliner-inspired overtake/collision region estimator.

    Lap 1:
      Learn opponent behavior along the centerline in Frenet-like (s, d) coordinates.
    Lap 2+:
      Predict the future Region of Collision (RoC) and preferred overtake side.
    """

    def __init__(self) -> None:
        super().__init__("predictive_spliner_decision")

        repo_root = find_repo_root()
        self.declare_parameter("pose_topic", "/autodrive/f1tenth_1/ips")
        self.declare_parameter("imu_topic", "/autodrive/f1tenth_1/imu")
        self.declare_parameter("target_point_topic", "/autodrive/f1tenth_1/target_tracker/target_point")
        self.declare_parameter("target_visible_topic", "/autodrive/f1tenth_1/target_tracker/target_visible")
        self.declare_parameter("tracking_confidence_topic", "/autodrive/f1tenth_1/target_tracker/tracking_confidence")
        self.declare_parameter("lap_count_topic", "/autodrive/f1tenth_1/pure_pursuit/lap_count")
        self.declare_parameter(
            "wall_mask_csv_path",
            str(repo_root / "output" / "slam_runs" / "slam_toolbox_boundary_wall_mask.csv"),
        )
        self.declare_parameter("num_path_points", 800)
        self.declare_parameter("bin_size_m", 0.25)
        self.declare_parameter("model_update_period_sec", 1.0)
        self.declare_parameter("min_tracking_confidence", 0.55)
        self.declare_parameter("min_observations_per_bin", 1)
        self.declare_parameter("min_bin_coverage_ratio", 0.25)
        self.declare_parameter("min_model_observations", 80)
        self.declare_parameter("require_opponent_lap_for_model", False)
        self.declare_parameter("lap_wrap_high_ratio", 0.80)
        self.declare_parameter("lap_wrap_low_ratio", 0.20)
        self.declare_parameter("min_lap_distance_m", 25.0)
        self.declare_parameter("prediction_horizon_sec", 6.0)
        self.declare_parameter("prediction_dt_sec", 0.05)
        self.declare_parameter("ego_accel_mps2", 0.0)
        self.declare_parameter("ego_nominal_d_m", 0.0)
        self.declare_parameter("car_length_m", 0.55)
        self.declare_parameter("car_width_m", 0.30)
        self.declare_parameter("safety_margin_long_m", 0.12)
        self.declare_parameter("safety_margin_lat_m", 0.10)
        self.declare_parameter("roc_lateral_release_distance_m", 1.0)
        self.declare_parameter("pass_offset_m", 0.45)
        self.declare_parameter("wall_clearance_threshold_m", 0.18)
        self.declare_parameter("gp_d_length_scale_m", 1.2)
        self.declare_parameter("gp_vs_length_scale_m", 1.8)
        self.declare_parameter("gp_d_sigma_f", 0.35)
        self.declare_parameter("gp_vs_sigma_f", 0.80)
        self.declare_parameter("gp_d_noise", 0.02)
        self.declare_parameter("gp_vs_noise", 0.05)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("roc_start_topic", "/autodrive/f1tenth_1/predictive_spliner/roc_start_s")
        self.declare_parameter("roc_end_topic", "/autodrive/f1tenth_1/predictive_spliner/roc_end_s")
        self.declare_parameter("model_ready_topic", "/autodrive/f1tenth_1/predictive_spliner/model_ready")
        self.declare_parameter("lap_phase_topic", "/autodrive/f1tenth_1/predictive_spliner/lap_phase")
        self.declare_parameter("overtake_allowed_topic", "/autodrive/f1tenth_1/predictive_spliner/overtake_allowed")
        self.declare_parameter("preferred_side_topic", "/autodrive/f1tenth_1/predictive_spliner/preferred_side")
        self.declare_parameter("roc_marker_topic", "/autodrive/f1tenth_1/predictive_spliner/roc_marker")
        self.declare_parameter("status_marker_topic", "/autodrive/f1tenth_1/predictive_spliner/status_marker")
        self.declare_parameter("figure_output_path", "")
        self.declare_parameter("figure_after_laps", 2)
        self.declare_parameter("roc_plot_half_width_m", 0.70)
        self.declare_parameter("roc_wall_inset_m", 0.05)
        self.declare_parameter("roc_persistence_ratio", 0.40)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.target_point_topic = str(self.get_parameter("target_point_topic").value)
        self.target_visible_topic = str(self.get_parameter("target_visible_topic").value)
        self.tracking_confidence_topic = str(self.get_parameter("tracking_confidence_topic").value)
        self.lap_count_topic = str(self.get_parameter("lap_count_topic").value)
        self.wall_mask_csv_path = Path(str(self.get_parameter("wall_mask_csv_path").value)).expanduser()
        figure_output_raw = str(self.get_parameter("figure_output_path").value).strip()
        default_figure_path = (
            self.wall_mask_csv_path.parent.parent
            / "predictive_spliner"
            / "predictive_spliner_two_lap.png"
        )
        self.figure_output_path = Path(figure_output_raw).expanduser() if figure_output_raw else default_figure_path
        self.figure_after_laps = int(self.get_parameter("figure_after_laps").value)
        self.roc_plot_half_width_m = float(self.get_parameter("roc_plot_half_width_m").value)
        self.roc_wall_inset_m = float(self.get_parameter("roc_wall_inset_m").value)
        self.roc_persistence_ratio = float(self.get_parameter("roc_persistence_ratio").value)
        self.bin_size_m = float(self.get_parameter("bin_size_m").value)
        self.model_update_period_sec = float(self.get_parameter("model_update_period_sec").value)
        self.min_tracking_confidence = float(self.get_parameter("min_tracking_confidence").value)
        self.min_observations_per_bin = int(self.get_parameter("min_observations_per_bin").value)
        self.min_bin_coverage_ratio = float(self.get_parameter("min_bin_coverage_ratio").value)
        self.min_model_observations = int(self.get_parameter("min_model_observations").value)
        self.require_opponent_lap_for_model = bool(
            self.get_parameter("require_opponent_lap_for_model").value
        )
        self.lap_wrap_high_ratio = float(self.get_parameter("lap_wrap_high_ratio").value)
        self.lap_wrap_low_ratio = float(self.get_parameter("lap_wrap_low_ratio").value)
        self.min_lap_distance_m = float(self.get_parameter("min_lap_distance_m").value)
        self.prediction_horizon_sec = float(self.get_parameter("prediction_horizon_sec").value)
        self.prediction_dt_sec = float(self.get_parameter("prediction_dt_sec").value)
        self.ego_accel_mps2 = float(self.get_parameter("ego_accel_mps2").value)
        self.ego_nominal_d_m = float(self.get_parameter("ego_nominal_d_m").value)
        self.car_length_m = float(self.get_parameter("car_length_m").value)
        self.car_width_m = float(self.get_parameter("car_width_m").value)
        self.safety_margin_long_m = float(self.get_parameter("safety_margin_long_m").value)
        self.safety_margin_lat_m = float(self.get_parameter("safety_margin_lat_m").value)
        self.roc_lateral_release_distance_m = float(
            self.get_parameter("roc_lateral_release_distance_m").value
        )
        self.pass_offset_m = float(self.get_parameter("pass_offset_m").value)
        self.wall_clearance_threshold_m = float(self.get_parameter("wall_clearance_threshold_m").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        path = load_manual_reference_line(int(self.get_parameter("num_path_points").value))
        self.track_points = np.asarray(path, dtype=float)
        rolled = np.roll(self.track_points, -1, axis=0)
        segment_vecs = rolled - self.track_points
        segment_lengths = np.linalg.norm(segment_vecs, axis=1)
        segment_lengths = np.where(segment_lengths < 1e-6, 1e-6, segment_lengths)
        self.cum_s = np.concatenate([[0.0], np.cumsum(segment_lengths[:-1])])
        self.track_length = float(np.sum(segment_lengths))
        tangents = segment_vecs / segment_lengths[:, None]
        self.normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        self.track_tree = cKDTree(self.track_points)
        self.num_bins = max(8, int(math.ceil(self.track_length / max(self.bin_size_m, 1e-3))))

        self.gp_d = OneDimensionalGaussianProcess(
            kernel="matern32",
            length_scale=float(self.get_parameter("gp_d_length_scale_m").value),
            sigma_f=float(self.get_parameter("gp_d_sigma_f").value),
            noise=float(self.get_parameter("gp_d_noise").value),
            period=self.track_length,
        )
        self.gp_vs = OneDimensionalGaussianProcess(
            kernel="rbf",
            length_scale=float(self.get_parameter("gp_vs_length_scale_m").value),
            sigma_f=float(self.get_parameter("gp_vs_sigma_f").value),
            noise=float(self.get_parameter("gp_vs_noise").value),
            period=self.track_length,
        )

        self.wall_points, self.wall_tree = self._load_wall_mask_points()
        self.left_track_widths, self.right_track_widths = self._build_track_width_profile()

        self.ego_position: tuple[float, float] | None = None
        self.ego_yaw: float | None = None
        self.target_visible = False
        self.tracking_confidence = 0.0
        self.latest_target_local: tuple[float, float] | None = None
        self.last_target_obs: OpponentObservation | None = None
        self.ego_progress_prev: float | None = None
        self.ego_pose_history: deque[tuple[float, float]] = deque(maxlen=100)
        self.ego_lap_distance = 0.0
        self.ego_lap_count = 0
        self.ego_last_lap_distance = 0.0
        self.opponent_wrap_count = 0
        self.opponent_progress_prev: float | None = None
        self.last_model_update_sec = 0.0
        self.last_readiness_log_sec = 0.0
        self.model_ready = False

        self.bin_d_values: dict[int, list[float]] = defaultdict(list)
        self.bin_vs_values: dict[int, list[float]] = defaultdict(list)
        self.observation_history: deque[OpponentObservation] = deque(maxlen=4000)

        self.latest_roc_start_s: float | None = None
        self.latest_roc_end_s: float | None = None
        self.latest_preferred_side = 0
        self.latest_overtake_allowed = False
        self.latest_gp_uncertainty = 0.0
        self.roc_history: deque[tuple[float, float, float, int]] = deque(maxlen=10000)
        self.figure_save_attempted = False
        self.figure_saved = False

        self.roc_start_pub = self.create_publisher(Float32, str(self.get_parameter("roc_start_topic").value), 10)
        self.roc_end_pub = self.create_publisher(Float32, str(self.get_parameter("roc_end_topic").value), 10)
        self.model_ready_pub = self.create_publisher(Bool, str(self.get_parameter("model_ready_topic").value), 10)
        self.lap_phase_pub = self.create_publisher(Int32, str(self.get_parameter("lap_phase_topic").value), 10)
        self.overtake_allowed_pub = self.create_publisher(Bool, str(self.get_parameter("overtake_allowed_topic").value), 10)
        self.preferred_side_pub = self.create_publisher(Int32, str(self.get_parameter("preferred_side_topic").value), 10)
        self.roc_marker_pub = self.create_publisher(Marker, str(self.get_parameter("roc_marker_topic").value), 10)
        self.status_marker_pub = self.create_publisher(Marker, str(self.get_parameter("status_marker_topic").value), 10)

        self.create_subscription(Point, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(Imu, self.imu_topic, self.imu_cb, 10)
        self.create_subscription(PointStamped, self.target_point_topic, self.target_point_cb, 10)
        self.create_subscription(Bool, self.target_visible_topic, self.target_visible_cb, 10)
        self.create_subscription(Float32, self.tracking_confidence_topic, self.tracking_confidence_cb, 10)
        self.create_subscription(Int32, self.lap_count_topic, self.lap_count_cb, 10)

        self.create_timer(1.0 / max(self.publish_rate_hz, 1e-3), self.timer_cb)

        self.get_logger().info(
            "Predictive Spliner decision ready. "
            f"pose={self.pose_topic}, imu={self.imu_topic}, target={self.target_point_topic}, "
            f"track_length={self.track_length:.2f}m, bins={self.num_bins}, wall_mask={self.wall_mask_csv_path}, "
            f"figure={self.figure_output_path}"
        )

    def _load_wall_mask_points(self) -> tuple[np.ndarray, cKDTree | None]:
        if not self.wall_mask_csv_path.exists():
            self.get_logger().warn(f"Wall mask CSV not found at {self.wall_mask_csv_path}; side scoring will be geometric only.")
            return np.empty((0, 2), dtype=float), None
        points: list[tuple[float, float]] = []
        with self.wall_mask_csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                points.append((float(row["world_x_m"]), float(row["world_y_m"])))
        point_array = np.asarray(points, dtype=float)
        tree = cKDTree(point_array) if len(point_array) > 0 else None
        self.get_logger().info(f"Loaded wall mask points={len(point_array)} from {self.wall_mask_csv_path}")
        return point_array, tree

    @staticmethod
    def _fill_circular_profile(values: np.ndarray, fallback: float) -> np.ndarray:
        valid = np.flatnonzero(np.isfinite(values))
        if len(valid) == 0:
            return np.full_like(values, fallback)
        count = len(values)
        sample_indices = np.concatenate([valid - count, valid, valid + count])
        sample_values = np.concatenate([values[valid], values[valid], values[valid]])
        filled = np.interp(np.arange(count), sample_indices, sample_values)
        return median_filter(filled, size=21, mode="wrap")

    def _build_track_width_profile(self) -> tuple[np.ndarray, np.ndarray]:
        fallback = np.full(len(self.track_points), self.roc_plot_half_width_m, dtype=float)
        if len(self.wall_points) == 0:
            return fallback.copy(), fallback.copy()

        _, nearest_indices = self.track_tree.query(self.wall_points, k=1)
        nearest_indices = np.asarray(nearest_indices, dtype=int)
        relative = self.wall_points - self.track_points[nearest_indices]
        lateral = np.sum(relative * self.normals[nearest_indices], axis=1)
        left_samples: list[list[float]] = [[] for _ in range(len(self.track_points))]
        right_samples: list[list[float]] = [[] for _ in range(len(self.track_points))]
        for idx, lateral_distance in zip(nearest_indices, lateral):
            if 0.15 < lateral_distance < 4.0:
                left_samples[int(idx)].append(float(lateral_distance))
            elif -4.0 < lateral_distance < -0.15:
                right_samples[int(idx)].append(float(-lateral_distance))

        left = np.full(len(self.track_points), np.nan, dtype=float)
        right = np.full(len(self.track_points), np.nan, dtype=float)
        for idx in range(len(self.track_points)):
            if left_samples[idx]:
                left[idx] = float(np.percentile(left_samples[idx], 10.0))
            if right_samples[idx]:
                right[idx] = float(np.percentile(right_samples[idx], 10.0))

        left = self._fill_circular_profile(left, self.roc_plot_half_width_m)
        right = self._fill_circular_profile(right, self.roc_plot_half_width_m)
        return np.clip(left, 0.20, 3.0), np.clip(right, 0.20, 3.0)

    def pose_cb(self, msg: Point) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self.ego_position = (float(msg.x), float(msg.y))
        projection = self.project_to_track(self.ego_position[0], self.ego_position[1])
        self.ego_pose_history.append((now_sec, projection.s))
        if self.ego_progress_prev is not None:
            self.ego_lap_distance += abs(circular_delta(projection.s, self.ego_progress_prev, self.track_length))
            crossed_start = (
                self.ego_progress_prev > self.track_length * self.lap_wrap_high_ratio
                and projection.s < self.track_length * self.lap_wrap_low_ratio
            )
            lap_distance = self.ego_lap_distance - self.ego_last_lap_distance
            if crossed_start and lap_distance >= self.min_lap_distance_m:
                self.ego_lap_count += 1
                self.ego_last_lap_distance = self.ego_lap_distance
                self.get_logger().info(f"Predictive Spliner ego completed lap {self.ego_lap_count}")
        self.ego_progress_prev = projection.s

    def imu_cb(self, msg: Imu) -> None:
        self.ego_yaw = quaternion_to_yaw(
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        )

    def target_visible_cb(self, msg: Bool) -> None:
        self.target_visible = bool(msg.data)

    def tracking_confidence_cb(self, msg: Float32) -> None:
        self.tracking_confidence = float(msg.data)

    def lap_count_cb(self, msg: Int32) -> None:
        authoritative_lap_count = max(0, int(msg.data))
        if authoritative_lap_count > self.ego_lap_count:
            self.ego_lap_count = authoritative_lap_count
            self.get_logger().info(f"Received pure-pursuit lap count {self.ego_lap_count}")

    def local_to_world(self, local_x: float, local_y: float) -> tuple[float, float] | None:
        if self.ego_position is None or self.ego_yaw is None:
            return None
        cos_yaw = math.cos(self.ego_yaw)
        sin_yaw = math.sin(self.ego_yaw)
        return (
            self.ego_position[0] + local_x * cos_yaw - local_y * sin_yaw,
            self.ego_position[1] + local_x * sin_yaw + local_y * cos_yaw,
        )

    def target_point_cb(self, msg: PointStamped) -> None:
        self.latest_target_local = (float(msg.point.x), float(msg.point.y))
        if not self.target_visible or self.tracking_confidence < self.min_tracking_confidence:
            return
        world_xy = self.local_to_world(self.latest_target_local[0], self.latest_target_local[1])
        if world_xy is None:
            return
        projection = self.project_to_track(world_xy[0], world_xy[1])
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        v_s = 0.0
        if self.last_target_obs is not None:
            dt = max(stamp_sec - self.last_target_obs.stamp_sec, 1e-3)
            v_s = circular_delta(projection.s, self.last_target_obs.s, self.track_length) / dt
            if (
                self.last_target_obs.s > self.track_length * self.lap_wrap_high_ratio
                and projection.s < self.track_length * self.lap_wrap_low_ratio
                and self.ego_lap_distance >= self.min_lap_distance_m
            ):
                self.opponent_wrap_count += 1

        obs = OpponentObservation(
            stamp_sec=stamp_sec,
            s=projection.s,
            d=projection.d,
            v_s=v_s,
            x=world_xy[0],
            y=world_xy[1],
        )
        self.last_target_obs = obs
        self.opponent_progress_prev = projection.s
        self.observation_history.append(obs)

        bin_idx = min(self.num_bins - 1, max(0, int(projection.s / self.bin_size_m)))
        self.bin_d_values[bin_idx].append(projection.d)
        if abs(v_s) > 1e-3:
            self.bin_vs_values[bin_idx].append(v_s)

    def project_to_track(self, x: float, y: float) -> TrackProjection:
        _, idx = self.track_tree.query([x, y], k=1)
        idx = int(idx)
        ref = self.track_points[idx]
        normal = self.normals[idx]
        d = float(np.dot(np.array([x, y]) - ref, normal))
        s = float(self.cum_s[idx])
        return TrackProjection(s=s, d=d, idx=idx, x=float(ref[0]), y=float(ref[1]))

    def maybe_fit_models(self, now_sec: float) -> None:
        if now_sec - self.last_model_update_sec < self.model_update_period_sec:
            return
        self.last_model_update_sec = now_sec

        valid_bins = [idx for idx, values in self.bin_d_values.items() if len(values) >= self.min_observations_per_bin]
        coverage_ratio = len(valid_bins) / max(self.num_bins, 1)
        observation_count = sum(len(self.bin_d_values[idx]) for idx in valid_bins)
        lap_requirement_met = not self.require_opponent_lap_for_model or self.opponent_wrap_count >= 1
        was_ready = self.model_ready
        self.model_ready = (
            lap_requirement_met
            and coverage_ratio >= self.min_bin_coverage_ratio
            and observation_count >= self.min_model_observations
        )
        if not self.model_ready:
            if now_sec - self.last_readiness_log_sec >= 5.0:
                self.last_readiness_log_sec = now_sec
                self.get_logger().info(
                    "Opponent GP waiting: "
                    f"observations={observation_count}/{self.min_model_observations}, "
                    f"coverage={coverage_ratio:.1%}/{self.min_bin_coverage_ratio:.1%}, "
                    f"wraps={self.opponent_wrap_count}, lap_required={int(self.require_opponent_lap_for_model)}"
                )
            return

        if not was_ready:
            self.get_logger().info(
                "Opponent GP ready: "
                f"observations={observation_count}, coverage={coverage_ratio:.1%}, "
                f"wraps={self.opponent_wrap_count}"
            )

        x_train: list[float] = []
        d_train: list[float] = []
        vs_train: list[float] = []
        for bin_idx in sorted(valid_bins):
            s_center = (bin_idx + 0.5) * self.bin_size_m
            x_train.append(s_center % self.track_length)
            d_train.append(float(np.median(self.bin_d_values[bin_idx])))
            vs_values = self.bin_vs_values.get(bin_idx, [])
            if vs_values:
                vs_train.append(float(np.median(vs_values)))
            else:
                vs_train.append(0.0)

        x_arr = np.asarray(x_train, dtype=float)
        self.gp_d.fit(x_arr, np.asarray(d_train, dtype=float))
        self.gp_vs.fit(x_arr, np.asarray(vs_train, dtype=float))

    def predict_spatial_collision_corridor(
        self,
        start_s: float,
        lateral_threshold: float,
    ) -> tuple[float, list[float], list[float], list[float]]:
        step_m = max(0.05, self.bin_size_m)
        lap_end_s = float(self.cum_s[-1])
        if start_s >= lap_end_s:
            return lap_end_s, [start_s], [0.0], [0.0]

        corridor_s = np.arange(start_s, lap_end_s + step_m, step_m)
        corridor_s = np.minimum(corridor_s, lap_end_s)
        predicted_d, predicted_std = self.gp_d.predict(corridor_s, return_std=True)
        lateral_overlap = np.abs(predicted_d - self.ego_nominal_d_m) <= lateral_threshold
        release_samples = max(1, int(math.ceil(self.roc_lateral_release_distance_m / step_m)))

        end_index = len(corridor_s) - 1
        separated_count = 0
        for idx, overlaps in enumerate(lateral_overlap):
            separated_count = 0 if overlaps else separated_count + 1
            if separated_count >= release_samples:
                end_index = max(0, idx - separated_count)
                break

        selected_s = corridor_s[: end_index + 1]
        selected_d = predicted_d[: end_index + 1]
        selected_std = predicted_std[: end_index + 1]
        return (
            float(corridor_s[end_index]),
            selected_s.astype(float).tolist(),
            selected_d.astype(float).tolist(),
            selected_std.astype(float).tolist(),
        )

    def predict_region_of_collision(self) -> None:
        self.latest_roc_start_s = None
        self.latest_roc_end_s = None
        self.latest_preferred_side = 0
        self.latest_overtake_allowed = False
        self.latest_gp_uncertainty = 0.0

        if not self.model_ready or self.ego_position is None or self.ego_yaw is None or self.last_target_obs is None:
            return

        ego_proj = self.project_to_track(self.ego_position[0], self.ego_position[1])
        ego_v_s = self.estimate_ego_longitudinal_speed()
        if ego_v_s <= 0.05:
            return

        s_ego = ego_proj.s
        s_opp = self.last_target_obs.s
        roc_start = None
        roc_end = None

        long_threshold = self.car_length_m + self.safety_margin_long_m
        lat_threshold = self.car_width_m + self.safety_margin_lat_m

        time_steps = np.arange(0.0, self.prediction_horizon_sec + self.prediction_dt_sec, self.prediction_dt_sec)
        roc_samples: list[float] = []
        roc_d_values: list[float] = []
        std_values: list[float] = []

        for t in time_steps:
            d_opp, d_std = self.gp_d.predict(np.array([s_opp]), return_std=True)
            v_opp, _ = self.gp_vs.predict(np.array([s_opp]), return_std=True)
            d_opp_val = float(d_opp[0])
            d_std_val = float(d_std[0])
            v_opp_val = max(0.01, float(v_opp[0]))

            forward_gap = (s_opp - s_ego) % self.track_length
            opponent_ahead = forward_gap <= 0.5 * self.track_length
            lateral_gap = abs(d_opp_val - self.ego_nominal_d_m)
            physical_overlap = (
                opponent_ahead
                and forward_gap <= long_threshold
                and lateral_gap <= lat_threshold
            )

            if physical_overlap:
                roc_start = s_ego
                roc_end, roc_samples, roc_d_values, std_values = self.predict_spatial_collision_corridor(
                    roc_start,
                    lat_threshold,
                )
                break

            s_ego = (s_ego + ego_v_s * self.prediction_dt_sec + 0.5 * self.ego_accel_mps2 * self.prediction_dt_sec**2) % self.track_length
            s_opp = (s_opp + v_opp_val * self.prediction_dt_sec) % self.track_length

        if roc_start is None:
            return
        assert roc_end is not None

        self.latest_roc_start_s = roc_start % self.track_length
        self.latest_roc_end_s = roc_end % self.track_length
        self.latest_gp_uncertainty = float(np.mean(std_values)) if std_values else 0.0
        self.latest_preferred_side = self.score_overtake_side(roc_samples, roc_d_values)
        self.latest_overtake_allowed = (
            self.latest_preferred_side != 0 and self.latest_gp_uncertainty < 0.35 and len(roc_samples) >= 2
        )
        self.roc_history.append(
            (
                self.latest_roc_start_s,
                self.latest_roc_end_s,
                self.latest_gp_uncertainty,
                self.latest_preferred_side,
            )
        )

    def estimate_ego_longitudinal_speed(self) -> float:
        if len(self.ego_pose_history) < 2:
            return 0.0
        samples = list(self.ego_pose_history)
        t0, s0 = samples[0]
        t1, s1 = samples[-1]
        dt = max(t1 - t0, 1e-3)
        ds = abs(circular_delta(s1, s0, self.track_length))
        return ds / dt

    def distance_to_wall(self, x: float, y: float) -> float:
        if self.wall_tree is None:
            return float("inf")
        distance, _ = self.wall_tree.query([x, y], k=1)
        return float(distance)

    def score_overtake_side(self, roc_samples: list[float], roc_d_values: list[float]) -> int:
        if not roc_samples:
            return 0
        mean_d = float(np.mean(roc_d_values)) if roc_d_values else 0.0
        preferred_by_opponent = -1 if mean_d >= 0.0 else 1  # pass opposite the opponent side relative to centerline

        left_score = self.score_side(roc_samples, sign=1)
        right_score = self.score_side(roc_samples, sign=-1)
        if preferred_by_opponent > 0:
            left_score += 0.08
        else:
            right_score += 0.08

        best_score = max(left_score, right_score)
        if best_score < self.wall_clearance_threshold_m:
            return 0
        return 1 if left_score >= right_score else -1

    def score_side(self, roc_samples: list[float], sign: int) -> float:
        distances: list[float] = []
        for s in roc_samples[:: max(1, len(roc_samples) // 20 or 1)]:
            idx = int(np.argmin(np.abs(self.cum_s - s)))
            center = self.track_points[idx]
            normal = self.normals[idx]
            sample = center + sign * self.pass_offset_m * normal
            distances.append(self.distance_to_wall(float(sample[0]), float(sample[1])))
        if not distances:
            return 0.0
        return float(min(distances))

    def aggregate_roc_history(self) -> tuple[float, float, float, int] | None:
        if not self.roc_history:
            return None

        occupancy = np.zeros(len(self.cum_s), dtype=int)
        for roc_start, roc_end, _, _ in self.roc_history:
            forward_length = (roc_end - roc_start) % self.track_length
            distance_from_start = (self.cum_s - roc_start) % self.track_length
            occupancy[distance_from_start <= forward_length] += 1

        persistence_threshold = max(
            2,
            int(math.ceil(float(np.max(occupancy)) * self.roc_persistence_ratio)),
        )
        mask = occupancy >= persistence_threshold
        if not np.any(mask):
            return self.roc_history[-1]

        doubled = np.concatenate([mask, mask])
        best_start = 0
        best_length = 0
        current_start = 0
        current_length = 0
        for idx, occupied in enumerate(doubled):
            if occupied:
                if current_length == 0:
                    current_start = idx
                current_length = min(current_length + 1, len(mask))
                if current_start < len(mask) and current_length > best_length:
                    best_start = current_start
                    best_length = current_length
            else:
                current_length = 0

        start_idx = best_start % len(mask)
        end_idx = (best_start + max(best_length - 1, 0)) % len(mask)
        uncertainties = [entry[2] for entry in self.roc_history]
        sides = [entry[3] for entry in self.roc_history if entry[3] != 0]
        side = max(set(sides), key=sides.count) if sides else 0
        return (
            float(self.cum_s[start_idx]),
            float(self.cum_s[end_idx]),
            float(np.mean(uncertainties)),
            int(side),
        )

    def save_two_lap_figure(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.figure_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(11, 7))

        if len(self.wall_points) > 0:
            ax.scatter(
                self.wall_points[:, 0],
                self.wall_points[:, 1],
                s=1.0,
                color="0.55",
                alpha=0.45,
                label="SLAM track boundary",
                rasterized=True,
            )

        ax.plot(
            self.track_points[:, 0],
            self.track_points[:, 1],
            color="#9a6700",
            linewidth=1.4,
            linestyle="--",
            label="Ego reference trajectory",
        )

        observations = list(self.observation_history)
        if observations:
            ax.scatter(
                [obs.x for obs in observations],
                [obs.y for obs in observations],
                s=8,
                color="#f2c94c",
                alpha=0.55,
                label="Observed opponent",
                zorder=4,
            )

        if self.model_ready:
            predicted_d = np.asarray(self.gp_d.predict(self.cum_s), dtype=float)
            predicted_xy = self.track_points + predicted_d[:, None] * self.normals
            ax.plot(
                predicted_xy[:, 0],
                predicted_xy[:, 1],
                color="#f0b400",
                linewidth=2.2,
                label="GP opponent trajectory",
                zorder=5,
            )

        plotted_roc = self.aggregate_roc_history()
        if plotted_roc is not None:
            roc_start, roc_end, uncertainty, side = plotted_roc
            forward_length = (roc_end - roc_start) % self.track_length
            roc_s = (roc_start + np.linspace(0.0, forward_length, 100)) % self.track_length
            roc_indices = np.array([int(np.argmin(np.abs(self.cum_s - s))) for s in roc_s], dtype=int)
            centers = self.track_points[roc_indices]
            normals = self.normals[roc_indices]
            left_widths = np.maximum(
                self.left_track_widths[roc_indices] - self.roc_wall_inset_m,
                0.05,
            )
            right_widths = np.maximum(
                self.right_track_widths[roc_indices] - self.roc_wall_inset_m,
                0.05,
            )
            left_edge = centers + left_widths[:, None] * normals
            right_edge = centers - right_widths[:, None] * normals
            polygon = np.vstack([left_edge, right_edge[::-1]])
            ax.fill(
                polygon[:, 0],
                polygon[:, 1],
                color="#c77dff",
                alpha=0.30,
                label="Predicted Region of Collision",
                zorder=2,
            )
            start_point = centers[0]
            end_point = centers[-1]
            ax.scatter(*start_point, s=65, color="#7b2cbf", zorder=7)
            ax.scatter(*end_point, s=65, color="#d0006f", zorder=7)
            ax.annotate(r"$c_{start}$", start_point, xytext=(7, 7), textcoords="offset points", color="#5a189a")
            ax.annotate(r"$c_{end}$", end_point, xytext=(7, 7), textcoords="offset points", color="#9d174d")
            side_name = "left" if side > 0 else "right" if side < 0 else "none"
            ax.text(
                0.02,
                0.02,
                f"RoC uncertainty: {uncertainty:.3f} | preferred side: {side_name}",
                transform=ax.transAxes,
                fontsize=9,
            )
        else:
            ax.text(0.5, 0.5, "No Region of Collision predicted", transform=ax.transAxes, ha="center")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Predictive Spliner: Two-Lap Collision Prediction")
        ax.grid(alpha=0.20)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.figure_output_path, dpi=220, bbox_inches="tight")
        pdf_path = self.figure_output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        self.figure_saved = True
        self.get_logger().info(f"Saved Predictive Spliner figures to {self.figure_output_path} and {pdf_path}")

    def publish_outputs(self) -> None:
        self.roc_start_pub.publish(Float32(data=float(self.latest_roc_start_s if self.latest_roc_start_s is not None else 0.0)))
        self.roc_end_pub.publish(Float32(data=float(self.latest_roc_end_s if self.latest_roc_end_s is not None else 0.0)))
        self.model_ready_pub.publish(Bool(data=bool(self.model_ready)))
        lap_phase = 2 if self.model_ready else 1
        self.lap_phase_pub.publish(Int32(data=lap_phase))
        self.overtake_allowed_pub.publish(Bool(data=bool(self.latest_overtake_allowed)))
        self.preferred_side_pub.publish(Int32(data=int(self.latest_preferred_side)))

        stamp = self.get_clock().now().to_msg()

        roc_marker = Marker()
        roc_marker.header.frame_id = "map"
        roc_marker.header.stamp = stamp
        roc_marker.ns = "predictive_spliner"
        roc_marker.id = 1
        roc_marker.type = Marker.LINE_STRIP
        roc_marker.action = Marker.ADD if self.latest_roc_start_s is not None and self.latest_roc_end_s is not None else Marker.DELETE
        roc_marker.scale.x = 0.15
        roc_marker.color.a = 1.0
        roc_marker.color.r = 1.0
        roc_marker.color.g = 0.85
        roc_marker.color.b = 0.1
        if roc_marker.action == Marker.ADD:
            roc_marker.points = []
            num_points = 40
            for i in range(num_points + 1):
                alpha = i / max(num_points, 1)
                s = circular_interp(self.latest_roc_start_s, self.latest_roc_end_s, alpha, self.track_length)
                idx = int(np.argmin(np.abs(self.cum_s - s)))
                point = Point()
                point.x = float(self.track_points[idx, 0])
                point.y = float(self.track_points[idx, 1])
                point.z = 0.05
                roc_marker.points.append(point)
        self.roc_marker_pub.publish(roc_marker)

        status_marker = Marker()
        status_marker.header.frame_id = "map"
        status_marker.header.stamp = stamp
        status_marker.ns = "predictive_spliner"
        status_marker.id = 2
        status_marker.type = Marker.TEXT_VIEW_FACING
        status_marker.action = Marker.ADD
        status_marker.scale.z = 0.5
        status_marker.color.a = 1.0
        if self.latest_overtake_allowed:
            status_marker.color.r = 0.1
            status_marker.color.g = 1.0
            status_marker.color.b = 0.1
        else:
            status_marker.color.r = 1.0
            status_marker.color.g = 0.2
            status_marker.color.b = 0.1
        anchor = self.track_points[0]
        status_marker.pose.position.x = float(anchor[0])
        status_marker.pose.position.y = float(anchor[1])
        status_marker.pose.position.z = 1.0
        status_marker.pose.orientation.w = 1.0
        side_text = "LEFT" if self.latest_preferred_side > 0 else "RIGHT" if self.latest_preferred_side < 0 else "NONE"
        status_marker.text = (
            f"PS lap={2 if self.model_ready else 1} "
            f"ready={int(self.model_ready)} "
            f"OT={int(self.latest_overtake_allowed)} "
            f"side={side_text} "
            f"unc={self.latest_gp_uncertainty:.2f}"
        )
        self.status_marker_pub.publish(status_marker)

    def timer_cb(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self.maybe_fit_models(now_sec)
        self.predict_region_of_collision()
        self.publish_outputs()
        if self.ego_lap_count >= self.figure_after_laps and not self.figure_save_attempted:
            self.figure_save_attempted = True
            try:
                self.save_two_lap_figure()
            except Exception as exc:
                self.get_logger().error(f"Failed to save Predictive Spliner figure: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PredictiveSplinerDecision()
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

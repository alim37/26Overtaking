#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

from autodrive_f1tenth.pure_pursuit import load_manual_reference_line


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"No samples found in {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in rows[0]
    }


class TrackGeometry:
    def __init__(self, wall_points: np.ndarray, num_points: int = 800) -> None:
        self.points = np.asarray(load_manual_reference_line(num_points), dtype=float)
        self.vectors = np.roll(self.points, -1, axis=0) - self.points
        self.lengths = np.maximum(np.linalg.norm(self.vectors, axis=1), 1e-6)
        self.cum_s = np.concatenate([[0.0], np.cumsum(self.lengths[:-1])])
        self.track_length = float(np.sum(self.lengths))
        self.tangents = self.vectors / self.lengths[:, None]
        self.normals = np.column_stack([-self.tangents[:, 1], self.tangents[:, 0]])
        self.tree = cKDTree(self.points)
        self.wall_tree = cKDTree(wall_points)
        self.left_widths, self.right_widths = self._wall_widths(wall_points)

    @staticmethod
    def _fill(values: np.ndarray, fallback: float) -> np.ndarray:
        valid = np.flatnonzero(np.isfinite(values))
        if not len(valid):
            return np.full_like(values, fallback)
        count = len(values)
        xp = np.concatenate([valid - count, valid, valid + count])
        fp = np.concatenate([values[valid], values[valid], values[valid]])
        return np.interp(np.arange(count), xp, fp)

    def _wall_widths(self, wall_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, indices = self.tree.query(wall_points, k=1)
        relative = wall_points - self.points[indices]
        lateral = np.sum(relative * self.normals[indices], axis=1)
        left_samples: list[list[float]] = [[] for _ in self.points]
        right_samples: list[list[float]] = [[] for _ in self.points]
        for index, distance in zip(indices, lateral):
            if 0.15 < distance < 4.0:
                left_samples[int(index)].append(float(distance))
            elif -4.0 < distance < -0.15:
                right_samples[int(index)].append(float(-distance))
        left = np.full(len(self.points), np.nan, dtype=float)
        right = np.full(len(self.points), np.nan, dtype=float)
        for index in range(len(self.points)):
            if left_samples[index]:
                left[index] = float(np.percentile(left_samples[index], 10.0))
            if right_samples[index]:
                right[index] = float(np.percentile(right_samples[index], 10.0))
        left = median_filter(self._fill(left, 0.70), 21, mode="wrap")
        right = median_filter(self._fill(right, 0.70), 21, mode="wrap")
        return np.clip(left, 0.20, 3.0), np.clip(right, 0.20, 3.0)

    def project(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, indices = self.tree.query(xy, k=1)
        relative = xy - self.points[indices]
        lateral = np.sum(relative * self.normals[indices], axis=1)
        return self.cum_s[indices], lateral, np.asarray(indices, dtype=int)

    def region_polygon(
        self,
        start_s: float,
        end_s: float,
        inset_m: float = 0.0,
        count: int = 180,
    ) -> np.ndarray:
        length = (end_s - start_s) % self.track_length
        samples = (start_s + np.linspace(0.0, length, count)) % self.track_length
        indices = np.array([int(np.argmin(np.abs(self.cum_s - value))) for value in samples])
        left_widths = np.maximum(self.left_widths[indices] - inset_m, 0.05)
        right_widths = np.maximum(self.right_widths[indices] - inset_m, 0.05)
        left = self.points[indices] + left_widths[:, None] * self.normals[indices]
        right = self.points[indices] - right_widths[:, None] * self.normals[indices]
        return np.vstack([left, right[::-1]])


def load_wall_points(path: Path) -> np.ndarray:
    points: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            points.append((float(row["world_x_m"]), float(row["world_y_m"])))
    return np.asarray(points, dtype=float)


def selected_region(profile: dict[str, np.ndarray]) -> tuple[float, float]:
    selected = profile["selected_roc"] > 0.5
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise ValueError("The confidence profile does not contain a selected RoC")
    s_values = profile["s_m"]
    spacing = float(np.median(np.diff(s_values))) if len(s_values) > 1 else 0.25
    return max(0.0, float(s_values[indices[0]] - 0.5 * spacing)), float(
        s_values[indices[-1]] + 0.5 * spacing
    )


def estimate_speed(stamps: np.ndarray, xy: np.ndarray) -> np.ndarray:
    dt = np.diff(stamps, prepend=stamps[0])
    distance = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[[0]]), axis=1)
    speed = np.divide(distance, dt, out=np.full_like(distance, np.nan), where=dt > 1e-3)
    finite = np.isfinite(speed) & (speed >= 0.0) & (speed < 12.0)
    fallback = float(np.median(speed[finite])) if np.any(finite) else 0.0
    speed[~finite] = fallback
    return median_filter(speed, size=11, mode="nearest")


def in_region(s_values: np.ndarray, start_s: float, end_s: float, track_length: float) -> np.ndarray:
    if start_s <= end_s:
        return (s_values >= start_s) & (s_values <= end_s)
    return (s_values >= start_s) | (s_values <= end_s % track_length)


def save_boundary(path: Path, segments: list[np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["segment_id", "world_x_m", "world_y_m"])
        for segment_id, segment in enumerate(segments):
            for x_value, y_value in segment:
                writer.writerow([segment_id, float(x_value), float(y_value)])


def build_overtake_path(
    geometry: TrackGeometry,
    ego_xy: np.ndarray,
    opponent_xy: np.ndarray,
    opponent_mask: np.ndarray,
    c_start: float,
    c_end: float,
    overtake_clearance_m: float,
    corridor_inset_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    ego_all_s, ego_all_d, _ = geometry.project(ego_xy)
    ego_region_indices = np.flatnonzero(
        in_region(ego_all_s, c_start, c_end, geometry.track_length)
    )
    if not len(ego_region_indices):
        raise ValueError("No ego trajectory samples fall inside the selected RoC")
    start_ego_index = ego_region_indices[
        np.argmin(np.abs(ego_all_s[ego_region_indices] - c_start))
    ]
    end_ego_index = ego_region_indices[
        np.argmin(np.abs(ego_all_s[ego_region_indices] - c_end))
    ]
    path_s = np.linspace(ego_all_s[start_ego_index], ego_all_s[end_ego_index], 240)
    path_indices = np.array(
        [int(np.argmin(np.abs(geometry.cum_s - value))) for value in path_s],
        dtype=int,
    )
    opponent_s, opponent_d, _ = geometry.project(opponent_xy[opponent_mask])
    order = np.argsort(opponent_s)
    opponent_s = opponent_s[order]
    opponent_d = opponent_d[order]
    unique_s, inverse = np.unique(np.round(opponent_s, 2), return_inverse=True)
    unique_d = np.array(
        [float(np.median(opponent_d[inverse == index])) for index in range(len(unique_s))],
        dtype=float,
    )
    target_d = np.interp(path_s, unique_s, unique_d, left=unique_d[0], right=unique_d[-1])
    target_d = median_filter(target_d, size=11, mode="nearest")

    ego_region = in_region(ego_all_s, c_start, c_end, geometry.track_length)
    ego_s = ego_all_s[ego_region]
    ego_d = ego_all_d[ego_region]
    ego_order = np.argsort(ego_s)
    ego_s = ego_s[ego_order]
    ego_d = ego_d[ego_order]
    ego_unique_s, ego_inverse = np.unique(np.round(ego_s, 2), return_inverse=True)
    ego_unique_d = np.array(
        [float(np.median(ego_d[ego_inverse == index])) for index in range(len(ego_unique_s))],
        dtype=float,
    )
    ego_reference_d = np.interp(
        path_s,
        ego_unique_s,
        ego_unique_d,
        left=ego_unique_d[0],
        right=ego_unique_d[-1],
    )
    ego_reference_d = median_filter(ego_reference_d, size=9, mode="nearest")

    left_room = geometry.left_widths[path_indices] - target_d - corridor_inset_m
    right_room = geometry.right_widths[path_indices] + target_d - corridor_inset_m
    left_score = float(np.percentile(left_room, 10.0))
    right_score = float(np.percentile(right_room, 10.0))
    if left_score >= right_score:
        side_sign = 1.0
        side_name = "left"
    else:
        side_sign = -1.0
        side_name = "right"

    pass_d = target_d + side_sign * overtake_clearance_m
    pass_d = np.clip(
        pass_d,
        -geometry.right_widths[path_indices] + corridor_inset_m,
        geometry.left_widths[path_indices] - corridor_inset_m,
    )
    progress = np.linspace(0.0, 1.0, len(path_s))
    transition_fraction = 0.18
    enter = np.clip(progress / transition_fraction, 0.0, 1.0)
    leave = np.clip((1.0 - progress) / transition_fraction, 0.0, 1.0)
    enter = enter * enter * (3.0 - 2.0 * enter)
    leave = leave * leave * (3.0 - 2.0 * leave)
    blend = enter * leave
    path_d = median_filter(
        ego_reference_d + blend * (pass_d - ego_reference_d),
        size=9,
        mode="nearest",
    )
    path_d = np.clip(
        path_d,
        -geometry.right_widths[path_indices] + corridor_inset_m,
        geometry.left_widths[path_indices] - corridor_inset_m,
    )

    # Enforce clearance against the actual occupied wall-mask points, not only
    # the centerline-derived left/right width approximation.
    for offset_index, (track_index, desired_d) in enumerate(zip(path_indices, path_d)):
        lower = -geometry.right_widths[track_index] + corridor_inset_m
        upper = geometry.left_widths[track_index] - corridor_inset_m
        candidates_d = np.linspace(lower, upper, 161)
        candidates_xy = (
            geometry.points[track_index]
            + candidates_d[:, None] * geometry.normals[track_index]
        )
        wall_distances, _ = geometry.wall_tree.query(candidates_xy, k=1)
        safe = wall_distances >= corridor_inset_m
        if np.any(safe):
            safe_d = candidates_d[safe]
            path_d[offset_index] = safe_d[np.argmin(np.abs(safe_d - desired_d))]

    path_xy = geometry.points[path_indices] + path_d[:, None] * geometry.normals[path_indices]
    # The plotted c_start/c_end are observed ego poses, not centerline approximations.
    path_xy[0] = ego_xy[start_ego_index]
    path_xy[-1] = ego_xy[end_ego_index]
    path_d[0] = ego_all_d[start_ego_index]
    path_d[-1] = ego_all_d[end_ego_index]
    return path_s, path_d, path_xy, side_name


def save_overtake_path(
    path: Path,
    path_s: np.ndarray,
    path_d: np.ndarray,
    path_xy: np.ndarray,
    side_name: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["s_m", "lateral_offset_m", "world_x_m", "world_y_m", "overtake_side"])
        for s_value, d_value, xy in zip(path_s, path_d, path_xy):
            writer.writerow([float(s_value), float(d_value), float(xy[0]), float(xy[1]), side_name])


def generate_engagement_zone(
    trajectory_path: Path,
    profile_path: Path,
    wall_path: Path,
    output_path: Path,
    engagement_horizon_sec: float,
    capture_radius_m: float,
    target_speed_mps: float,
    ego_width_m: float,
    target_width_m: float,
    vehicle_gap_m: float,
    wall_safety_margin_m: float,
    min_confidence: float,
    grid_resolution_m: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.path import Path as MatplotlibPath

    trajectory = read_numeric_csv(trajectory_path)
    profile = read_numeric_csv(profile_path)
    walls = load_wall_points(wall_path)
    geometry = TrackGeometry(walls)
    c_start, c_end = selected_region(profile)
    overtake_clearance_m = 0.5 * (ego_width_m + target_width_m) + vehicle_gap_m
    corridor_inset_m = 0.5 * ego_width_m + wall_safety_margin_m

    ego_xy = np.column_stack([trajectory["ego_x_m"], trajectory["ego_y_m"]])
    opponent_xy = np.column_stack([trajectory["opponent_x_m"], trajectory["opponent_y_m"]])
    ego_s, _, _ = geometry.project(ego_xy)
    _, _, opponent_track_indices = geometry.project(opponent_xy)
    measured_opponent_speed = estimate_speed(trajectory["stamp_sec"], opponent_xy)
    if target_speed_mps > 0.0:
        constant_target_speed = target_speed_mps
    else:
        reliable = trajectory["tracking_confidence"] >= min_confidence
        constant_target_speed = float(np.median(measured_opponent_speed[reliable]))
    ego_speed = median_filter(trajectory["ego_speed_mps"], size=11, mode="nearest")
    sample_mask = in_region(ego_s, c_start, c_end, geometry.track_length)
    sample_mask &= trajectory["tracking_confidence"] >= min_confidence
    sample_indices = np.flatnonzero(sample_mask)
    if not len(sample_indices):
        raise ValueError("No synchronized high-confidence opponent samples fall inside the selected RoC")
    path_s, path_d, planned_path, side_name = build_overtake_path(
        geometry,
        ego_xy,
        opponent_xy,
        sample_mask,
        c_start,
        c_end,
        overtake_clearance_m,
        corridor_inset_m,
    )

    min_xy = np.min(walls, axis=0) - 0.5
    max_xy = np.max(walls, axis=0) + 0.5
    x_grid = np.arange(min_xy[0], max_xy[0] + grid_resolution_m, grid_resolution_m)
    y_grid = np.arange(min_xy[1], max_xy[1] + grid_resolution_m, grid_resolution_m)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
    grid_s, grid_d, grid_indices = geometry.project(grid_xy)
    # The displayed RoC and engagement zone cover the full SLAM track envelope.
    # Vehicle-width clearance remains enforced separately on the planned path.
    track_mask = (
        (grid_d <= geometry.left_widths[grid_indices])
        & (grid_d >= -geometry.right_widths[grid_indices])
    )
    roc_polygon = geometry.region_polygon(c_start, c_end)
    roc_mask = MatplotlibPath(roc_polygon).contains_points(grid_xy)
    candidate_indices = np.flatnonzero(track_mask & roc_mask)
    candidates = grid_xy[candidate_indices]
    engaged = np.zeros(len(candidates), dtype=bool)

    stride = max(1, len(sample_indices) // 250)
    for index in sample_indices[::stride]:
        # Ego is the pursuer. The pure-pursuit target is the constant-speed evader.
        evader_direction = geometry.tangents[opponent_track_indices[index]]
        evader_travel = constant_target_speed * engagement_horizon_sec
        pursuer_range = max(0.0, float(ego_speed[index])) * engagement_horizon_sec
        zone_center = ego_xy[index] - evader_travel * evader_direction
        zone_radius = pursuer_range + capture_radius_m
        engaged |= np.sum((candidates - zone_center) ** 2, axis=1) <= zone_radius**2

    engagement_mask = np.zeros(len(grid_xy), dtype=float)
    engagement_mask[candidate_indices] = engaged.astype(float)
    engagement_mask = engagement_mask.reshape(xx.shape)
    if not np.any(engagement_mask):
        raise ValueError("The configured engagement horizon produced no zone inside the RoC")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(walls[:, 0], walls[:, 1], s=1.0, color="0.65", alpha=0.35, rasterized=True)
    safe_roc_mask = (track_mask & roc_mask).reshape(xx.shape).astype(float)
    ax.contourf(
        xx, yy, safe_roc_mask, levels=[0.5, 1.5],
        colors=["#c77dff"], alpha=0.18, zorder=1,
    )
    ax.plot(
        ego_xy[:, 0], ego_xy[:, 1], "--", color="#9a6700",
        linewidth=1.5, label="Ego trajectory (pursuer)",
    )
    ax.scatter(
        opponent_xy[:, 0], opponent_xy[:, 1], s=8, color="#f2c94c",
        alpha=0.65, label="Observed target (evader)", zorder=4,
    )
    contour = ax.contour(xx, yy, engagement_mask, levels=[0.5], colors=["#159447"], linewidths=2.2)
    segments = [segment for segment in contour.allsegs[0] if len(segment) >= 2]
    save_boundary(output_path.with_suffix(".csv"), segments)
    ax.plot(
        planned_path[:, 0], planned_path[:, 1], color="#0066cc",
        linewidth=2.6, label="Planned ego engagement path", zorder=6,
    )
    ax.scatter(
        planned_path[[0, -1], 0], planned_path[[0, -1], 1],
        s=30, color=["#9c27b0", "#d35400"], zorder=7,
    )
    ax.annotate(r"$c_{start}$", planned_path[0], xytext=(5, 5), textcoords="offset points")
    ax.annotate(r"$c_{end}$", planned_path[-1], xytext=(5, 5), textcoords="offset points")
    path_output = output_path.with_name(f"{output_path.stem}_path.csv")
    save_overtake_path(path_output, path_s, path_d, planned_path, side_name)

    ax.legend(
        handles=[
            Line2D([], [], color="#9a6700", linestyle="--", label="Ego trajectory (pursuer)"),
            Line2D([], [], color="#f2c94c", marker="o", linestyle="", label="Observed target (evader)"),
            Line2D([], [], color="#159447", linewidth=2.2, label="Engagement zone boundary"),
            Line2D([], [], color="#0066cc", linewidth=2.6, label="Planned ego engagement path"),
            Line2D([], [], color="#c77dff", linewidth=7, alpha=0.35, label="Region of Collision"),
        ],
        loc="best",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Engagement Zone Inside Confidence-Derived RoC")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved engagement-zone figure to {output_path}")
    print(f"Saved engagement-zone boundary to {output_path.with_suffix('.csv')}")
    print(f"Saved planned ego path to {path_output} (selected side: {side_name})")
    print(f"Target speed used as constant evader speed: {constant_target_speed:.3f} m/s")
    print(
        f"Vehicle-aware clearance: center separation={overtake_clearance_m:.3f} m, "
        f"track inset={corridor_inset_m:.3f} m"
    )


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    output_root = package_root / "output"
    parser = argparse.ArgumentParser(description="Build a racing engagement-zone envelope inside the saved RoC")
    parser.add_argument(
        "--trajectory-csv", type=Path,
        default=output_root / "confidence_roc" / "confidence_roc_one_lap_trajectory.csv",
    )
    parser.add_argument(
        "--profile-csv", type=Path,
        default=output_root / "confidence_roc" / "confidence_roc_one_lap_profile.csv",
    )
    parser.add_argument(
        "--wall-mask-csv", type=Path,
        default=output_root / "slam_runs" / "slam_toolbox_boundary_wall_mask.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=output_root / "engagement_zones" / "engagement_zone_one_lap.png",
    )
    parser.add_argument("--engagement-horizon-sec", type=float, default=1.0)
    parser.add_argument("--capture-radius-m", type=float, default=0.45)
    parser.add_argument(
        "--target-speed-mps", type=float, default=0.0,
        help="Constant evader speed; <=0 estimates one robust value from the observed target trajectory",
    )
    parser.add_argument("--ego-width-m", type=float, default=0.30)
    parser.add_argument("--target-width-m", type=float, default=0.30)
    parser.add_argument("--vehicle-gap-m", type=float, default=0.30)
    parser.add_argument("--wall-safety-margin-m", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--grid-resolution-m", type=float, default=0.08)
    args = parser.parse_args()
    if not args.trajectory_csv.exists():
        raise SystemExit(
            f"Missing {args.trajectory_csv}. Run confidence_roc.launch.py once with the updated logger first."
        )
    generate_engagement_zone(
        args.trajectory_csv,
        args.profile_csv,
        args.wall_mask_csv,
        args.output,
        args.engagement_horizon_sec,
        args.capture_radius_m,
        args.target_speed_mps,
        args.ego_width_m,
        args.target_width_m,
        args.vehicle_gap_m,
        args.wall_safety_margin_m,
        args.min_confidence,
        args.grid_resolution_m,
    )


if __name__ == "__main__":
    main()

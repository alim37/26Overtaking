#!/usr/bin/env python3

"""Deterministic Frenet polynomial-lattice planner for the dynamic engagement zone."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from autodrive_f1tenth import engagement_zones_dynamic as dynamic
from autodrive_f1tenth.engagement_zones import TrackGeometry, in_region
from autodrive_f1tenth.engagement_zones_dynamic import (
    Candidate,
    connected_free_corridor,
    interpolate_track_indices,
    make_candidate,
    observed_lateral_profile,
    smooth_observations,
)


def quintic_smoothstep(value: np.ndarray) -> np.ndarray:
    """Minimum-jerk 0-to-1 transition with zero endpoint slope and acceleration."""
    value = np.clip(value, 0.0, 1.0)
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def plan_frenet_lattice(
    geometry: TrackGeometry,
    trajectory: dict[str, np.ndarray],
    c_start: float,
    c_end: float,
    samples: int,
    seed: int,
    engagement_model: str,
    maximum_zone_range_m: float,
    ego_length_m: float,
    ego_width_m: float,
    target_length_m: float,
    target_width_m: float,
    vehicle_gap_m: float,
    box_buffer_m: float,
    wall_margin_m: float,
    curvature_limit_1pm: float,
    min_confidence: float,
    planning_speed_mps: float,
    pass_margin_m: float,
    minimum_progress_gain_m: float,
    entry_blend_distance_m: float,
    exit_blend_distance_m: float,
    maximum_start_heading_error: float,
) -> tuple[Candidate, np.ndarray, int, int, int]:
    del seed  # The lattice is deterministic by design.
    ego_xy = np.column_stack([trajectory["ego_x_m"], trajectory["ego_y_m"]])
    ego_s, ego_d, _ = geometry.project(ego_xy)
    region_indices = np.flatnonzero(in_region(ego_s, c_start, c_end, geometry.track_length))
    if not len(region_indices):
        raise ValueError("No ego samples fall inside the selected confidence RoC")

    start_index = int(region_indices[np.argmin(np.abs(ego_s[region_indices] - c_start))])
    end_index = int(region_indices[np.argmin(np.abs(ego_s[region_indices] - c_end))])
    path_s = np.linspace(c_start, ego_s[end_index], 260)
    path_indices = interpolate_track_indices(geometry, path_s)
    progress = np.linspace(0.0, 1.0, len(path_s))
    control_progress = np.linspace(0.0, 1.0, 16)
    control_indices = np.linspace(0, len(path_s) - 1, len(control_progress)).astype(int)

    confidence = trajectory["tracking_confidence"]
    opponent_xy = np.column_stack(
        [
            smooth_observations(trajectory["opponent_x_m"], confidence, min_confidence),
            smooth_observations(trajectory["opponent_y_m"], confidence, min_confidence),
        ]
    )
    observation_time = trajectory["stamp_sec"]
    reliable_speed = trajectory["ego_speed_mps"][region_indices]
    valid_speed = reliable_speed[reliable_speed > 0.1]
    measured_speed = float(np.median(valid_speed)) if len(valid_speed) else 0.1
    ego_speed = planning_speed_mps if planning_speed_mps > 0.0 else measured_speed

    ego_radius = 0.5 * math.hypot(ego_length_m, ego_width_m)
    target_radius = 0.5 * math.hypot(target_length_m, target_width_m)
    vehicle_radius = ego_radius + target_radius + vehicle_gap_m
    wall_clearance = ego_radius + wall_margin_m
    lower, upper = connected_free_corridor(geometry, path_indices, wall_clearance)
    observed_reference_d = observed_lateral_profile(
        path_s, ego_s[region_indices], ego_d[region_indices]
    )
    required_start_heading = float(trajectory["ego_yaw_rad"][start_index])

    baseline = np.interp(
        control_progress,
        progress,
        np.interp(path_s, ego_s[region_indices], ego_d[region_indices]),
    )
    baseline[0] = ego_d[start_index]
    baseline[-1] = ego_d[end_index]

    # A bounded lattice replaces 8,000 random splines. Each profile uses
    # minimum-jerk quintic entry/exit transitions toward one track side.
    candidate_controls: list[np.ndarray] = [baseline.copy()]
    route_length = max(float(path_s[-1] - path_s[0]), 1.0)
    entry_lengths = (2.5, 3.5, 4.5, 5.5)
    exit_lengths = (2.0, 3.0, 4.0)
    lateral_fractions = (0.58, 0.72, 0.86, 0.96)
    shifts = (0.0, 0.08)

    for entry_length in entry_lengths:
        for exit_length in exit_lengths:
            for fraction in lateral_fractions:
                for shift in shifts:
                    # Interleave passing sides so a small online time budget
                    # always evaluates both sides of the opponent.
                    for side_profile in (lower, upper):
                        entry_u = (path_s - path_s[0] - shift * route_length) / entry_length
                        exit_u = (path_s[-1] - path_s) / exit_length
                        window = quintic_smoothstep(entry_u) * quintic_smoothstep(exit_u)
                        desired = observed_reference_d + fraction * window * (
                            side_profile - observed_reference_d
                        )
                        controls = np.interp(control_progress, progress, desired)
                        controls[0] = baseline[0]
                        controls[-1] = baseline[-1]
                        candidate_controls.append(controls)

    if samples > 0:
        candidate_controls = candidate_controls[:samples]

    best: Candidate | None = None
    feasible_count = 0
    start_time = float(observation_time[start_index])
    for controls in candidate_controls:
        candidate = make_candidate(
            controls, control_progress, progress, path_s, path_indices, geometry,
            ego_speed, observation_time, opponent_xy, start_time,
            engagement_model, maximum_zone_range_m, vehicle_radius,
            ego_length_m, ego_width_m, target_length_m, target_width_m,
            box_buffer_m, wall_margin_m, curvature_limit_1pm, lower, upper,
            pass_margin_m, minimum_progress_gain_m, observed_reference_d,
            entry_blend_distance_m, exit_blend_distance_m,
            required_start_heading, maximum_start_heading_error,
        )
        feasible_count += int(candidate.feasible)
        if best is None or (candidate.feasible, -candidate.cost) > (
            best.feasible, -best.cost
        ):
            best = candidate

    if best is None:
        raise RuntimeError("Frenet lattice did not generate any candidates")
    return best, path_s, start_index, end_index, feasible_count


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    output_root = package_root / "output"
    parser = argparse.ArgumentParser(
        description="Lightweight deterministic Frenet-lattice engagement-zone planner"
    )
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
        default=(output_root / "engagement_zones_dynamic_frenet"
                 / "engagement_zone_dynamic_frenet_one_lap.png"),
    )
    parser.add_argument(
        "--samples", type=int, default=65,
        help="Bounded lattice size; 65 is intended for online replanning",
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--engagement-model", choices=("box", "cardioid"), default="box")
    parser.add_argument("--maximum-zone-range-m", type=float, default=0.0)
    parser.add_argument("--ego-length-m", type=float, default=0.50)
    parser.add_argument("--ego-width-m", type=float, default=0.30)
    parser.add_argument("--target-length-m", type=float, default=0.50)
    parser.add_argument("--target-width-m", type=float, default=0.30)
    parser.add_argument("--engagement-box-buffer-m", type=float, default=0.08)
    parser.add_argument("--engagement-zone-width-scale", type=float, default=1.00)
    parser.add_argument("--roc-start-buffer-sec", type=float, default=2.00)
    parser.add_argument("--vehicle-gap-m", type=float, default=0.12)
    parser.add_argument("--wall-margin-m", type=float, default=0.05)
    parser.add_argument("--curvature-limit-1pm", type=float, default=3.5)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--pass-margin-m", type=float, default=1.00)
    parser.add_argument("--minimum-progress-gain-m", type=float, default=2.00)
    parser.add_argument("--entry-blend-distance-m", type=float, default=5.00)
    parser.add_argument("--exit-blend-distance-m", type=float, default=3.00)
    parser.add_argument("--maximum-start-heading-error-deg", type=float, default=8.0)
    parser.add_argument("--planning-speed-mps", type=float, default=3.90)
    args = parser.parse_args()
    for path in (args.trajectory_csv, args.profile_csv, args.wall_mask_csv):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    args.path_label = "Frenet polynomial-lattice path"
    args.planner_title = "Frenet Polynomial Planning Around a Dynamic Evader"
    args.planner_name = "Frenet-lattice"
    original_planner = dynamic.sample_dynamic_path
    dynamic.sample_dynamic_path = plan_frenet_lattice
    started = time.perf_counter()
    try:
        dynamic.generate(args)
    finally:
        dynamic.sample_dynamic_path = original_planner
    print(f"Frenet-lattice total generation time: {time.perf_counter() - started:.3f} s")


if __name__ == "__main__":
    main()

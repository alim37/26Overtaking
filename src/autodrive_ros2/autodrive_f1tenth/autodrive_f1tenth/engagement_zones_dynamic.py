#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, median_filter

from autodrive_f1tenth.engagement_zones import (
    TrackGeometry,
    in_region,
    load_wall_points,
    read_numeric_csv,
    selected_region,
)


@dataclass
class Candidate:
    d: np.ndarray
    xy: np.ndarray
    heading: np.ndarray
    time: np.ndarray
    opponent_xy: np.ndarray
    opponent_heading: np.ndarray
    zone_radius: np.ndarray
    cost: float
    feasible: bool
    min_clearance: float
    min_wall_clearance: float
    max_curvature: float
    initial_progress_advantage: float
    final_progress_advantage: float
    progress_gain: float
    pass_completed: bool
    start_heading_error: float


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def smooth_observations(values: np.ndarray, confidence: np.ndarray, minimum: float) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    reliable = np.flatnonzero(confidence >= minimum)
    if not len(reliable):
        raise ValueError("No reliable opponent observations are available")
    missing = np.flatnonzero(confidence < minimum)
    if len(missing):
        result[missing] = np.interp(missing, reliable, result[reliable])
    return median_filter(result, size=9, mode="nearest")


def path_heading(xy: np.ndarray) -> np.ndarray:
    delta = np.gradient(xy, axis=0)
    return np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))


def path_time(xy: np.ndarray, speed_mps: float) -> np.ndarray:
    ds = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[[0]]), axis=1)
    return np.cumsum(ds) / max(speed_mps, 0.1)


def rectangle_corners(
    centers: np.ndarray,
    headings: np.ndarray,
    length_m: float,
    width_m: float,
) -> np.ndarray:
    local = np.array(
        [
            [0.5 * length_m, 0.5 * width_m],
            [0.5 * length_m, -0.5 * width_m],
            [-0.5 * length_m, -0.5 * width_m],
            [-0.5 * length_m, 0.5 * width_m],
        ],
        dtype=float,
    )
    cosine = np.cos(headings)
    sine = np.sin(headings)
    rotation = np.stack(
        [
            np.stack([cosine, -sine], axis=1),
            np.stack([sine, cosine], axis=1),
        ],
        axis=1,
    )
    return centers[:, None, :] + np.einsum("nij,kj->nki", rotation, local)


def signed_track_delta(ego_s: np.ndarray, opponent_s: np.ndarray, track_length: float) -> np.ndarray:
    return (ego_s - opponent_s + 0.5 * track_length) % track_length - 0.5 * track_length


def observed_lateral_profile(
    path_s: np.ndarray,
    observed_s: np.ndarray,
    observed_d: np.ndarray,
) -> np.ndarray:
    order = np.argsort(observed_s)
    sorted_s = observed_s[order]
    sorted_d = observed_d[order]
    unique_s, inverse = np.unique(np.round(sorted_s, 3), return_inverse=True)
    unique_d = np.array(
        [float(np.median(sorted_d[inverse == index])) for index in range(len(unique_s))]
    )
    return np.interp(path_s, unique_s, unique_d, left=unique_d[0], right=unique_d[-1])


def dynamic_evader(
    plan_time: np.ndarray,
    observation_time: np.ndarray,
    opponent_xy: np.ndarray,
    start_time: float,
) -> np.ndarray:
    query = start_time + plan_time
    predicted = np.column_stack(
        [
            np.interp(query, observation_time, opponent_xy[:, axis])
            for axis in range(2)
        ]
    )
    # Continue the measured terminal motion if the candidate extends past the log.
    tail_dt = max(observation_time[-1] - observation_time[-8], 1e-3)
    tail_velocity = (opponent_xy[-1] - opponent_xy[-8]) / tail_dt
    beyond = query > observation_time[-1]
    if np.any(beyond):
        predicted[beyond] = opponent_xy[-1] + (
            query[beyond] - observation_time[-1]
        )[:, None] * tail_velocity
    return predicted


def cardioid_radius(
    ego_xy: np.ndarray,
    ego_heading: np.ndarray,
    opponent_xy: np.ndarray,
    maximum_range_m: float,
    vehicle_radius_m: float,
) -> np.ndarray:
    """Equation (8) from Wolek et al., evaluated against a moving evader."""
    line_of_sight = np.arctan2(
        ego_xy[:, 1] - opponent_xy[:, 1],
        ego_xy[:, 0] - opponent_xy[:, 0],
    )
    relative_bearing = wrap_angle(ego_heading - line_of_sight - np.pi)
    return vehicle_radius_m + 0.5 * maximum_range_m * (np.cos(relative_bearing) + 1.0)


def oriented_box_clearance(
    ego_xy: np.ndarray,
    ego_heading: np.ndarray,
    opponent_xy: np.ndarray,
    opponent_heading: np.ndarray,
    ego_length_m: float,
    ego_width_m: float,
    target_length_m: float,
    target_width_m: float,
    box_buffer_m: float,
) -> np.ndarray:
    """Signed SAT clearance between ego and buffered target rectangles."""
    ego_long = np.column_stack([np.cos(ego_heading), np.sin(ego_heading)])
    ego_lat = np.column_stack([-np.sin(ego_heading), np.cos(ego_heading)])
    target_long = np.column_stack(
        [np.cos(opponent_heading), np.sin(opponent_heading)]
    )
    target_lat = np.column_stack(
        [-np.sin(opponent_heading), np.cos(opponent_heading)]
    )
    axes = np.stack([ego_long, ego_lat, target_long, target_lat], axis=1)
    center_delta = ego_xy - opponent_xy
    center_projection = np.abs(np.einsum("ni,nki->nk", center_delta, axes))

    ego_projection = (
        0.5 * ego_length_m * np.abs(np.einsum("ni,nki->nk", ego_long, axes))
        + 0.5 * ego_width_m * np.abs(np.einsum("ni,nki->nk", ego_lat, axes))
    )
    target_half_length = 0.5 * target_length_m + box_buffer_m
    target_half_width = 0.5 * target_width_m + box_buffer_m
    target_projection = (
        target_half_length
        * np.abs(np.einsum("ni,nki->nk", target_long, axes))
        + target_half_width
        * np.abs(np.einsum("ni,nki->nk", target_lat, axes))
    )
    axis_separation = center_projection - ego_projection - target_projection
    return np.max(axis_separation, axis=1)


def interpolate_track_indices(geometry: TrackGeometry, path_s: np.ndarray) -> np.ndarray:
    return np.array(
        [int(np.argmin(np.abs(geometry.cum_s - value))) for value in path_s],
        dtype=int,
    )


def connected_free_corridor(
    geometry: TrackGeometry,
    path_indices: np.ndarray,
    wall_clearance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the wall-free lateral interval connected to the reference line."""
    lower = np.empty(len(path_indices), dtype=float)
    upper = np.empty(len(path_indices), dtype=float)
    cache: dict[int, tuple[float, float]] = {}
    for output_index, track_index in enumerate(path_indices):
        key = int(track_index)
        if key not in cache:
            samples = np.linspace(
                -geometry.right_widths[key], geometry.left_widths[key], 241
            )
            points = geometry.points[key] + samples[:, None] * geometry.normals[key]
            distances, _ = geometry.wall_tree.query(points, k=1)
            safe = distances >= wall_clearance_m
            anchor = int(np.argmin(np.abs(samples)))
            if not safe[anchor]:
                safe_indices = np.flatnonzero(safe)
                if not len(safe_indices):
                    cache[key] = (-0.02, 0.02)
                    lower[output_index], upper[output_index] = cache[key]
                    continue
                anchor = int(safe_indices[np.argmin(np.abs(samples[safe_indices]))])
            first = anchor
            last = anchor
            while first > 0 and safe[first - 1]:
                first -= 1
            while last + 1 < len(safe) and safe[last + 1]:
                last += 1
            cache[key] = (float(samples[first]), float(samples[last]))
        lower[output_index], upper[output_index] = cache[key]
    return lower, upper


def make_candidate(
    controls: np.ndarray,
    control_progress: np.ndarray,
    path_progress: np.ndarray,
    path_s: np.ndarray,
    path_indices: np.ndarray,
    geometry: TrackGeometry,
    ego_speed_mps: float,
    observation_time: np.ndarray,
    opponent_xy: np.ndarray,
    start_time: float,
    engagement_model: str,
    maximum_zone_range_m: float,
    vehicle_radius_m: float,
    ego_length_m: float,
    ego_width_m: float,
    target_length_m: float,
    target_width_m: float,
    box_buffer_m: float,
    wall_margin_m: float,
    curvature_limit_1pm: float,
    safe_lower: np.ndarray,
    safe_upper: np.ndarray,
    pass_margin_m: float,
    minimum_progress_gain_m: float,
    observed_reference_d: np.ndarray,
    entry_blend_distance_m: float,
    exit_blend_distance_m: float,
    required_start_heading: float,
    maximum_start_heading_error: float,
) -> Candidate:
    lateral = CubicSpline(control_progress, controls, bc_type="clamped")(path_progress)
    lateral = np.clip(lateral, safe_lower, safe_upper)
    # Projection onto the corridor can introduce corners. Alternate smoothing
    # and projection to retain wall feasibility without leaving a jagged path.
    for _ in range(4):
        lateral = gaussian_filter1d(lateral, sigma=2.0, mode="nearest")
        lateral = np.clip(lateral, safe_lower, safe_upper)

    entry_fraction = np.clip(
        (path_s - path_s[0]) / max(entry_blend_distance_m, 1e-3), 0.0, 1.0
    )
    entry_weight = entry_fraction**2 * (3.0 - 2.0 * entry_fraction)
    lateral = (1.0 - entry_weight) * observed_reference_d + entry_weight * lateral

    exit_fraction = np.clip(
        (path_s[-1] - path_s) / max(exit_blend_distance_m, 1e-3), 0.0, 1.0
    )
    exit_weight = exit_fraction**2 * (3.0 - 2.0 * exit_fraction)
    lateral = exit_weight * lateral + (1.0 - exit_weight) * observed_reference_d
    for _ in range(3):
        lateral = gaussian_filter1d(lateral, sigma=4.0, mode="nearest")
        lateral = np.clip(lateral, safe_lower, safe_upper)

    xy = geometry.points[path_indices] + lateral[:, None] * geometry.normals[path_indices]
    endpoints = xy[[0, -1]].copy()
    xy = gaussian_filter1d(xy, sigma=2.5, axis=0, mode="nearest")
    endpoint_correction = np.linspace(endpoints[0] - xy[0], endpoints[1] - xy[-1], len(xy))
    xy += endpoint_correction
    heading = path_heading(xy)
    times = path_time(xy, ego_speed_mps)
    moving_opponent = dynamic_evader(times, observation_time, opponent_xy, start_time)
    opponent_heading = path_heading(moving_opponent)
    if engagement_model == "box":
        dynamic_clearance = oriented_box_clearance(
            xy,
            heading,
            moving_opponent,
            opponent_heading,
            ego_length_m,
            ego_width_m,
            target_length_m,
            target_width_m,
            box_buffer_m,
        )
        buffered_length = target_length_m + 2.0 * box_buffer_m
        buffered_width = target_width_m + 2.0 * box_buffer_m
        zone_radius = np.full(
            len(xy), 0.5 * math.hypot(buffered_length, buffered_width), dtype=float
        )
    else:
        zone_radius = cardioid_radius(
            xy, heading, moving_opponent, maximum_zone_range_m, vehicle_radius_m
        )
        separation = np.linalg.norm(xy - moving_opponent, axis=1)
        dynamic_clearance = separation - zone_radius
    footprint = rectangle_corners(xy, heading, ego_length_m, ego_width_m)
    wall_distance, _ = geometry.wall_tree.query(footprint.reshape(-1, 2), k=1)
    wall_distance = wall_distance.reshape(len(xy), 4)
    minimum_wall_distance = np.min(wall_distance, axis=1)
    ds = np.maximum(np.gradient(path_s), 1e-3)
    curvature = np.abs(np.gradient(np.unwrap(heading)) / ds)
    opponent_s, _, _ = geometry.project(moving_opponent)
    progress_advantage = signed_track_delta(path_s, opponent_s, geometry.track_length)
    progress_gain = float(progress_advantage[-1] - progress_advantage[0])
    start_heading_error = abs(
        float(wrap_angle(np.array([heading[0] - required_start_heading]))[0])
    )
    pass_completed = bool(
        progress_advantage[-1] >= pass_margin_m
        and progress_gain >= minimum_progress_gain_m
    )
    feasible = bool(
        np.all(dynamic_clearance >= 0.0)
        and np.all(minimum_wall_distance >= wall_margin_m)
        and np.max(curvature[2:-2]) <= curvature_limit_1pm
        and pass_completed
    )

    path_length = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    curvature_cost = float(np.mean(curvature**2))
    penetration = float(np.mean(np.maximum(-dynamic_clearance, 0.0) ** 2))
    proximity = float(np.mean(np.exp(-np.maximum(dynamic_clearance, 0.0) / 0.35)))
    wall_penalty = float(
        np.mean(np.maximum(wall_margin_m - minimum_wall_distance, 0.0) ** 2)
    )
    pass_shortfall = max(pass_margin_m - float(progress_advantage[-1]), 0.0)
    cost = (
        path_length
        + 1.8 * curvature_cost
        + 400.0 * penetration
        + 0.7 * proximity
        + 400.0 * wall_penalty
        + 40.0 * pass_shortfall**2
        - 0.20 * float(progress_advantage[-1])
    )
    return Candidate(
        d=lateral,
        xy=xy,
        heading=heading,
        time=times,
        opponent_xy=moving_opponent,
        opponent_heading=opponent_heading,
        zone_radius=zone_radius,
        cost=cost,
        feasible=feasible,
        min_clearance=float(np.min(dynamic_clearance)),
        min_wall_clearance=float(np.min(minimum_wall_distance)),
        max_curvature=float(np.max(curvature[2:-2])),
        initial_progress_advantage=float(progress_advantage[0]),
        final_progress_advantage=float(progress_advantage[-1]),
        progress_gain=progress_gain,
        pass_completed=pass_completed,
        start_heading_error=start_heading_error,
    )


def sample_dynamic_path(
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

    confidence = trajectory["tracking_confidence"]
    opponent_xy = np.column_stack(
        [
            smooth_observations(trajectory["opponent_x_m"], confidence, min_confidence),
            smooth_observations(trajectory["opponent_y_m"], confidence, min_confidence),
        ]
    )
    observation_time = trajectory["stamp_sec"]
    reliable_speed = trajectory["ego_speed_mps"][region_indices]
    measured_ego_speed = float(np.median(reliable_speed[reliable_speed > 0.1]))
    ego_speed = planning_speed_mps if planning_speed_mps > 0.0 else measured_ego_speed
    ego_radius = 0.5 * math.hypot(ego_length_m, ego_width_m)
    target_radius = 0.5 * math.hypot(target_length_m, target_width_m)
    vehicle_radius = ego_radius + target_radius + vehicle_gap_m
    wall_clearance = ego_radius + wall_margin_m
    lower, upper = connected_free_corridor(geometry, path_indices, wall_clearance)
    observed_reference_d = observed_lateral_profile(
        path_s, ego_s[region_indices], ego_d[region_indices]
    )
    required_start_heading = float(trajectory["ego_yaw_rad"][start_index])

    control_indices = np.linspace(0, len(path_s) - 1, len(control_progress)).astype(int)
    baseline = np.interp(
        control_progress,
        progress,
        np.interp(path_s, ego_s[region_indices], ego_d[region_indices]),
    )
    baseline[0] = ego_d[start_index]
    baseline[-1] = ego_d[end_index]
    rng = np.random.default_rng(seed)
    best: Candidate | None = None
    feasible_count = 0

    # Include the observed lane and both deliberate pass sides before random sampling.
    control_sets = [baseline.copy()]
    for fraction in (0.35, 0.60, 0.82):
        left = baseline.copy()
        right = baseline.copy()
        left[1:-1] = (1.0 - fraction) * baseline[1:-1] + fraction * upper[control_indices[1:-1]]
        right[1:-1] = (1.0 - fraction) * baseline[1:-1] + fraction * lower[control_indices[1:-1]]
        control_sets.extend([left, right])

    baseline_candidate = make_candidate(
        baseline,
        control_progress,
        progress,
        path_s,
        path_indices,
        geometry,
        ego_speed,
        observation_time,
        opponent_xy,
        float(observation_time[start_index]),
        engagement_model,
        maximum_zone_range_m,
        vehicle_radius,
        ego_length_m,
        ego_width_m,
        target_length_m,
        target_width_m,
        box_buffer_m,
        wall_margin_m,
        curvature_limit_1pm,
        lower,
        upper,
        pass_margin_m,
        minimum_progress_gain_m,
        observed_reference_d,
        entry_blend_distance_m,
        exit_blend_distance_m,
        required_start_heading,
        maximum_start_heading_error,
    )
    opponent_s, opponent_d, _ = geometry.project(baseline_candidate.opponent_xy)
    longitudinal_delta = np.abs(opponent_s - path_s)
    longitudinal_delta = np.minimum(longitudinal_delta, geometry.track_length - longitudinal_delta)
    for aggressiveness in (0.65, 0.82, 0.95):
        informed = baseline.copy()
        for control_index, sample_index in enumerate(control_indices[1:-1], start=1):
            nearby = longitudinal_delta[sample_index] < 3.0
            if not nearby:
                continue
            left_separation = abs(upper[sample_index] - opponent_d[sample_index])
            right_separation = abs(lower[sample_index] - opponent_d[sample_index])
            edge = upper[sample_index] if left_separation >= right_separation else lower[sample_index]
            informed[control_index] = (
                (1.0 - aggressiveness) * baseline[control_index]
                + aggressiveness * edge
            )
        informed[1:-1] = median_filter(informed, size=3, mode="nearest")[1:-1]
        control_sets.append(informed)
    for _ in range(max(0, samples - len(control_sets))):
        controls = rng.uniform(lower[control_indices], upper[control_indices])
        controls[0] = baseline[0]
        controls[-1] = baseline[-1]
        # Correlated controls produce driveable paths more often than white-noise offsets.
        controls[1:-1] = median_filter(controls, size=3, mode="nearest")[1:-1]
        control_sets.append(controls)

    for controls in control_sets:
        candidate = make_candidate(
            controls,
            control_progress,
            progress,
            path_s,
            path_indices,
            geometry,
            ego_speed,
            observation_time,
            opponent_xy,
            float(observation_time[start_index]),
            engagement_model,
            maximum_zone_range_m,
            vehicle_radius,
            ego_length_m,
            ego_width_m,
            target_length_m,
            target_width_m,
            box_buffer_m,
            wall_margin_m,
            curvature_limit_1pm,
            lower,
            upper,
            pass_margin_m,
            minimum_progress_gain_m,
            observed_reference_d,
            entry_blend_distance_m,
            exit_blend_distance_m,
            required_start_heading,
            maximum_start_heading_error,
        )
        feasible_count += int(candidate.feasible)
        if best is None or (candidate.feasible, -candidate.cost) > (best.feasible, -best.cost):
            best = candidate
    assert best is not None
    return best, path_s, start_index, end_index, feasible_count


def save_path(path: Path, path_s: np.ndarray, candidate: Candidate) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "time_sec", "s_m", "lateral_offset_m", "ego_x_m", "ego_y_m",
                "ego_heading_rad", "evader_x_m", "evader_y_m", "engagement_radius_m",
            ]
        )
        for index in range(len(path_s)):
            writer.writerow(
                [
                    candidate.time[index], path_s[index], candidate.d[index],
                    candidate.xy[index, 0], candidate.xy[index, 1], candidate.heading[index],
                    candidate.opponent_xy[index, 0], candidate.opponent_xy[index, 1],
                    candidate.zone_radius[index],
                ]
            )


def save_validation(
    path: Path,
    candidate: Candidate,
    feasible_count: int,
    sample_count: int,
    raw_c_start_m: float,
    buffered_c_start_m: float,
    c_start_buffer_sec: float,
    c_start_buffer_m: float,
    engagement_model: str,
    engagement_zone_range_m: float,
    engagement_box_buffer_m: float,
    engagement_box_length_m: float,
    engagement_box_width_m: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerows(
            [
                ["selected_feasible", int(candidate.feasible)],
                ["pass_completed", int(candidate.pass_completed)],
                ["raw_c_start_m", raw_c_start_m],
                ["buffered_c_start_m", buffered_c_start_m],
                ["c_start_buffer_sec", c_start_buffer_sec],
                ["c_start_buffer_m", c_start_buffer_m],
                ["engagement_model", engagement_model],
                ["engagement_zone_range_m", engagement_zone_range_m],
                ["engagement_box_buffer_m", engagement_box_buffer_m],
                ["engagement_box_length_m", engagement_box_length_m],
                ["engagement_box_width_m", engagement_box_width_m],
                ["sample_count", sample_count],
                ["feasible_sample_count", feasible_count],
                ["minimum_dynamic_clearance_m", candidate.min_clearance],
                ["minimum_footprint_wall_clearance_m", candidate.min_wall_clearance],
                ["maximum_curvature_1pm", candidate.max_curvature],
                ["initial_progress_advantage_m", candidate.initial_progress_advantage],
                ["final_progress_advantage_m", candidate.final_progress_advantage],
                ["progress_gain_m", candidate.progress_gain],
                ["start_heading_error_deg", math.degrees(candidate.start_heading_error)],
                ["cost", candidate.cost],
            ]
        )


def generate(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    trajectory = read_numeric_csv(args.trajectory_csv)
    profile = read_numeric_csv(args.profile_csv)
    walls = load_wall_points(args.wall_mask_csv)
    geometry = TrackGeometry(walls)
    raw_c_start, c_end = selected_region(profile)
    ego_xy = np.column_stack([trajectory["ego_x_m"], trajectory["ego_y_m"]])
    ego_s, _, _ = geometry.project(ego_xy)
    start_delta = np.abs(
        (ego_s - raw_c_start + 0.5 * geometry.track_length)
        % geometry.track_length
        - 0.5 * geometry.track_length
    )
    raw_start_index = int(np.argmin(start_delta))
    speed_slice = trajectory["ego_speed_mps"][
        max(0, raw_start_index - 5) : raw_start_index + 6
    ]
    valid_speeds = speed_slice[np.isfinite(speed_slice) & (speed_slice > 0.1)]
    fallback_speed = max(float(args.planning_speed_mps), 0.1)
    c_start_speed_mps = (
        float(np.median(valid_speeds)) if len(valid_speeds) else fallback_speed
    )
    c_start_buffer_sec = max(float(args.roc_start_buffer_sec), 0.0)
    c_start_buffer_m = c_start_speed_mps * c_start_buffer_sec
    c_start = (raw_c_start + c_start_buffer_m) % geometry.track_length
    engagement_zone_range_m = (
        float(args.maximum_zone_range_m)
        if args.maximum_zone_range_m > 0.0
        else args.engagement_zone_width_scale
        * max(args.ego_width_m, args.target_width_m)
    )
    best, path_s, start_index, end_index, feasible_count = sample_dynamic_path(
        geometry,
        trajectory,
        c_start,
        c_end,
        args.samples,
        args.seed,
        args.engagement_model,
        engagement_zone_range_m,
        args.ego_length_m,
        args.ego_width_m,
        args.target_length_m,
        args.target_width_m,
        args.vehicle_gap_m,
        args.engagement_box_buffer_m,
        args.wall_margin_m,
        args.curvature_limit_1pm,
        args.min_confidence,
        args.planning_speed_mps,
        args.pass_margin_m,
        args.minimum_progress_gain_m,
        args.entry_blend_distance_m,
        args.exit_blend_distance_m,
        math.radians(args.maximum_start_heading_error_deg),
    )
    if not best.feasible:
        raise RuntimeError(
            "No validated overtake was found: "
            f"dynamic_clearance={best.min_clearance:.3f} m, "
            f"wall_clearance={best.min_wall_clearance:.3f} m, "
            f"maximum_curvature={best.max_curvature:.3f} 1/m, "
            f"final_advantage={best.final_progress_advantage:.3f} m, "
            f"start_heading_error={math.degrees(best.start_heading_error):.1f} deg"
        )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    path_output = output.with_name(f"{output.stem}_path.csv")
    validation_output = output.with_name(f"{output.stem}_validation.csv")
    save_path(path_output, path_s, best)
    save_validation(
        validation_output,
        best,
        feasible_count,
        args.samples,
        raw_c_start,
        c_start,
        c_start_buffer_sec,
        c_start_buffer_m,
        args.engagement_model,
        engagement_zone_range_m,
        args.engagement_box_buffer_m,
        args.target_length_m + 2.0 * args.engagement_box_buffer_m,
        args.target_width_m + 2.0 * args.engagement_box_buffer_m,
    )

    opponent_xy = np.column_stack([trajectory["opponent_x_m"], trajectory["opponent_y_m"]])
    roc_polygon = geometry.region_polygon(c_start, c_end)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(walls[:, 0], walls[:, 1], s=1.0, color="0.62", alpha=0.38, rasterized=True)
    ax.add_patch(Polygon(roc_polygon, closed=True, color="#c77dff", alpha=0.16, zorder=1))
    ax.plot(ego_xy[:, 0], ego_xy[:, 1], "--", color="#9a6700", linewidth=1.4, label="Observed ego")
    ax.scatter(opponent_xy[:, 0], opponent_xy[:, 1], s=7, color="#f2c94c", alpha=0.65, label="Observed dynamic evader")

    segments = np.stack([best.opponent_xy[:-1], best.opponent_xy[1:]], axis=1)
    zone_colors = plt.get_cmap("YlOrRd")(np.linspace(0.25, 0.95, len(segments)))
    ax.add_collection(LineCollection(segments, colors=zone_colors, linewidths=3.0, alpha=0.8, zorder=5))
    stride = max(1, len(best.time) // 16)
    if args.engagement_model == "box":
        box_length = args.target_length_m + 2.0 * args.engagement_box_buffer_m
        box_width = args.target_width_m + 2.0 * args.engagement_box_buffer_m
        boxes = rectangle_corners(
            best.opponent_xy, best.opponent_heading, box_length, box_width
        )
        for index in range(0, len(best.time), stride):
            ax.add_patch(
                Polygon(
                    boxes[index],
                    closed=True,
                    facecolor="#f6a04d",
                    edgecolor="#d95f02",
                    linewidth=0.8,
                    alpha=0.24,
                    zorder=4,
                )
            )
    else:
        theta = np.linspace(0.0, 2.0 * np.pi, 80)
        ego_radius = 0.5 * math.hypot(args.ego_length_m, args.ego_width_m)
        target_radius = 0.5 * math.hypot(args.target_length_m, args.target_width_m)
        vehicle_radius = ego_radius + target_radius + args.vehicle_gap_m
        for index in range(0, len(best.time), stride):
            radius = vehicle_radius + 0.5 * engagement_zone_range_m * (
                1.0 - np.cos(best.heading[index] - theta)
            )
            cardioid = best.opponent_xy[index] + radius[:, None] * np.column_stack(
                [np.cos(theta), np.sin(theta)]
            )
            ax.plot(
                cardioid[:, 0], cardioid[:, 1],
                color="#d95f02", linewidth=0.65, alpha=0.24,
            )

    path_color = "#0066cc" if best.feasible else "#c62828"
    path_normal = np.column_stack([-np.sin(best.heading), np.cos(best.heading)])
    path_left = best.xy + 0.5 * args.ego_width_m * path_normal
    path_right = best.xy - 0.5 * args.ego_width_m * path_normal
    swept_polygon = np.vstack([path_left, path_right[::-1]])
    ax.add_patch(
        Polygon(
            swept_polygon,
            closed=True,
            facecolor="#4da3ff",
            edgecolor="none",
            alpha=0.22,
            zorder=6,
        )
    )
    ax.plot(best.xy[:, 0], best.xy[:, 1], color=path_color, linewidth=2.8, label="Sampled dynamic-EZ path", zorder=7)
    ax.scatter(best.xy[[0, -1], 0], best.xy[[0, -1], 1], s=34, color=["#9c27b0", "#d35400"], zorder=8)
    ax.annotate(r"$c_{start}$", best.xy[0], xytext=(5, 5), textcoords="offset points")
    ax.annotate(r"$c_{end}$", best.xy[-1], xytext=(5, 5), textcoords="offset points")
    ax.text(
        0.015,
        0.015,
        "VALIDATED OVERTAKE\n"
        f"progress: {best.initial_progress_advantage:+.2f} -> {best.final_progress_advantage:+.2f} m\n"
        f"dynamic clearance: {best.min_clearance:.2f} m\n"
        f"footprint-wall clearance: {best.min_wall_clearance:.2f} m",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#159447"},
    )
    ax.legend(
        handles=[
            Line2D([], [], color="#9a6700", linestyle="--", label="Observed ego"),
            Line2D([], [], color="#f2c94c", marker="o", linestyle="", label="Observed dynamic evader"),
            Line2D([], [], color="#d95f02", linewidth=2.0, label="Buffered target engagement box"),
            Line2D([], [], color=path_color, linewidth=2.8, label="Sampled dynamic-EZ path"),
            Line2D([], [], color="#c77dff", linewidth=7, alpha=0.35, label="Confidence Region of Collision"),
        ],
        loc="best",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Sampling-Based Planning Around a Dynamic Evader")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dynamic engagement-zone figure to {output}")
    print(f"Saved dynamic planned path to {path_output}")
    print(f"Saved feasibility report to {validation_output}")
    print(
        f"selected RoC=({c_start:.3f}, {c_end:.3f}) m "
        f"(raw_start={raw_c_start:.3f}, buffer={c_start_buffer_sec:.3f} s/"
        f"{c_start_buffer_m:.3f} m), "
        f"engagement_model={args.engagement_model}, "
        f"box_buffer={args.engagement_box_buffer_m:.3f} m, "
        f"samples={args.samples}, feasible={feasible_count}, selected_feasible={best.feasible}, "
        f"minimum_dynamic_clearance={best.min_clearance:.3f} m, "
        f"minimum_wall_clearance={best.min_wall_clearance:.3f} m, "
        f"progress={best.initial_progress_advantage:+.3f}->{best.final_progress_advantage:+.3f} m, "
        f"start_heading_error={math.degrees(best.start_heading_error):.2f} deg, "
        f"cost={best.cost:.3f}"
    )


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    output_root = package_root / "output"
    parser = argparse.ArgumentParser(
        description="Offline sampling-based path planning around a time-varying evader"
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
        default=output_root / "engagement_zones_dynamic" / "engagement_zone_dynamic_one_lap.png",
    )
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--engagement-model",
        choices=("box", "cardioid"),
        default="box",
        help="Opponent exclusion geometry used during candidate collision checks",
    )
    parser.add_argument(
        "--maximum-zone-range-m",
        type=float,
        default=0.0,
        help="Explicit engagement-zone range; <=0 derives it from car width",
    )
    parser.add_argument("--ego-length-m", type=float, default=0.50)
    parser.add_argument("--ego-width-m", type=float, default=0.30)
    parser.add_argument("--target-length-m", type=float, default=0.50)
    parser.add_argument("--target-width-m", type=float, default=0.30)
    parser.add_argument("--engagement-box-buffer-m", type=float, default=0.08)
    # Keep the directional engagement zone equal to one target-vehicle width.
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
    parser.add_argument(
        "--planning-speed-mps", type=float, default=3.90,
        help="Ego speed used to time candidate paths; <=0 uses the recorded median speed",
    )
    args = parser.parse_args()
    for path in (args.trajectory_csv, args.profile_csv, args.wall_mask_csv):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")
    generate(args)


if __name__ == "__main__":
    main()

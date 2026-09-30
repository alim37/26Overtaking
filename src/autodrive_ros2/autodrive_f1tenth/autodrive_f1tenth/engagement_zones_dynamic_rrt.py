#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from autodrive_f1tenth.engagement_zones import TrackGeometry, load_wall_points, read_numeric_csv
from autodrive_f1tenth.engagement_zones_dynamic import (
    Candidate,
    connected_free_corridor,
    dynamic_evader,
    interpolate_track_indices,
    make_candidate,
    observed_lateral_profile,
    oriented_box_clearance,
    path_heading,
    rectangle_corners,
    save_path,
    signed_track_delta,
    smooth_observations,
    wrap_angle,
)


@dataclass
class RRTNode:
    s: float
    d: float
    parent: int
    cost: float
    heading: float


def load_reference_endpoints(path: Path) -> tuple[float, float, float, float]:
    data = read_numeric_csv(path)
    return (
        float(data["s_m"][0]),
        float(data["s_m"][-1]),
        float(data["lateral_offset_m"][0]),
        float(data["lateral_offset_m"][-1]),
    )


def interp_profile(s: np.ndarray, grid_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(s, grid_s, values)


def xy_from_sd(geometry: TrackGeometry, s: np.ndarray, d: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    d = np.asarray(d, dtype=float)
    center_x = np.interp(s, geometry.cum_s, geometry.points[:, 0])
    center_y = np.interp(s, geometry.cum_s, geometry.points[:, 1])
    track_heading = np.unwrap(
        np.arctan2(geometry.tangents[:, 1], geometry.tangents[:, 0])
    )
    heading = np.interp(s, geometry.cum_s, track_heading)
    normals = np.column_stack([-np.sin(heading), np.cos(heading)])
    return np.column_stack([center_x, center_y]) + d[:, None] * normals


def edge_samples(
    start: RRTNode,
    end_s: float,
    end_d: float,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = max(4, int(math.ceil((end_s - start.s) / spacing_m)) + 1)
    s = np.linspace(start.s, end_s, count)
    d = np.linspace(start.d, end_d, count)
    return s, d


def edge_is_free(
    start: RRTNode,
    end_s: float,
    end_d: float,
    geometry: TrackGeometry,
    corridor_s: np.ndarray,
    safe_lower: np.ndarray,
    safe_upper: np.ndarray,
    observation_time: np.ndarray,
    opponent_xy: np.ndarray,
    plan_start_time: float,
    planning_speed_mps: float,
    ego_length_m: float,
    ego_width_m: float,
    target_length_m: float,
    target_width_m: float,
    box_buffer_m: float,
    wall_margin_m: float,
    curvature_limit_1pm: float,
    edge_check_spacing_m: float,
) -> tuple[bool, float, float]:
    if end_s <= start.s + 1e-6:
        return False, 0.0, start.heading
    s, d = edge_samples(start, end_s, end_d, edge_check_spacing_m)
    lower = interp_profile(s, corridor_s, safe_lower)
    upper = interp_profile(s, corridor_s, safe_upper)
    if np.any(d < lower) or np.any(d > upper):
        return False, 0.0, start.heading

    xy = xy_from_sd(geometry, s, d)
    segment_length = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    heading = path_heading(xy)
    heading_change = abs(float(np.unwrap([start.heading, heading[0]])[1] - start.heading))
    if heading_change > curvature_limit_1pm * max(segment_length, 0.05):
        return False, segment_length, float(heading[-1])

    footprint = rectangle_corners(xy, heading, ego_length_m, ego_width_m)
    wall_distance, _ = geometry.wall_tree.query(footprint.reshape(-1, 2), k=1)
    if float(np.min(wall_distance)) < wall_margin_m:
        return False, segment_length, float(heading[-1])

    elapsed = (start.cost + np.linspace(0.0, segment_length, len(xy))) / max(
        planning_speed_mps, 0.1
    )
    moving_opponent = dynamic_evader(elapsed, observation_time, opponent_xy, plan_start_time)
    opponent_heading = path_heading(moving_opponent)
    clearance = oriented_box_clearance(
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
    return bool(np.all(clearance >= 0.0)), segment_length, float(heading[-1])


def trace_path(nodes: list[RRTNode], index: int) -> tuple[np.ndarray, np.ndarray]:
    chain: list[RRTNode] = []
    while index >= 0:
        chain.append(nodes[index])
        index = nodes[index].parent
    chain.reverse()
    return (
        np.asarray([node.s for node in chain], dtype=float),
        np.asarray([node.d for node in chain], dtype=float),
    )


def rrt_star(
    geometry: TrackGeometry,
    corridor_s: np.ndarray,
    safe_lower: np.ndarray,
    safe_upper: np.ndarray,
    start_d: float,
    goal_d: float,
    required_start_heading: float,
    observation_time: np.ndarray,
    opponent_xy: np.ndarray,
    plan_start_time: float,
    guide_d: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[RRTNode], list[int]]:
    rng = np.random.default_rng(args.seed)
    start_s = float(corridor_s[0])
    goal_s = float(corridor_s[-1])
    nodes = [RRTNode(start_s, start_d, -1, 0.0, required_start_heading)]
    # Warm-start with the already validated route. RRT* then improves this
    # incumbent through random expansion, best-parent selection, and rewiring.
    guide_indices = np.unique(
        np.r_[
            np.arange(0, len(corridor_s), max(1, len(corridor_s) // 45)),
            len(corridor_s) - 1,
        ]
    )
    guide_xy = xy_from_sd(geometry, corridor_s[guide_indices], guide_d[guide_indices])
    guide_headings = path_heading(guide_xy)
    for guide_position in range(1, len(guide_indices)):
        grid_index = int(guide_indices[guide_position])
        previous_xy = guide_xy[guide_position - 1]
        current_xy = guide_xy[guide_position]
        cost = nodes[-1].cost + float(np.linalg.norm(current_xy - previous_xy))
        nodes.append(
            RRTNode(
                float(corridor_s[grid_index]),
                float(guide_d[grid_index]),
                len(nodes) - 1,
                cost,
                float(guide_headings[guide_position]),
            )
        )
    goal_nodes: list[int] = [len(nodes) - 1]

    def connection(parent: RRTNode, s: float, d: float) -> tuple[bool, float, float]:
        return edge_is_free(
            parent, s, d, geometry, corridor_s, safe_lower, safe_upper,
            observation_time, opponent_xy, plan_start_time, args.planning_speed_mps,
            args.ego_length_m, args.ego_width_m, args.target_length_m,
            args.target_width_m,
            args.engagement_box_buffer_m + args.rrt_dynamic_margin_m,
            args.wall_margin_m + args.rrt_wall_margin_m,
            args.curvature_limit_1pm, args.edge_check_spacing_m,
        )

    for _ in range(args.iterations):
        if rng.random() < args.goal_bias:
            sample_s, sample_d = goal_s, goal_d
        else:
            sample_s = float(rng.uniform(start_s, goal_s))
            low = float(interp_profile(np.array([sample_s]), corridor_s, safe_lower)[0])
            high = float(interp_profile(np.array([sample_s]), corridor_s, safe_upper)[0])
            sample_d = float(rng.uniform(low, high))

        eligible = [i for i, node in enumerate(nodes) if node.s < sample_s - 0.05]
        if not eligible:
            continue
        distances = np.asarray(
            [math.hypot(sample_s - nodes[i].s, args.lateral_weight * (sample_d - nodes[i].d)) for i in eligible]
        )
        nearest_index = eligible[int(np.argmin(distances))]
        nearest = nodes[nearest_index]
        new_s = min(sample_s, nearest.s + args.step_m)
        fraction = (new_s - nearest.s) / max(sample_s - nearest.s, 1e-6)
        new_d = nearest.d + fraction * (sample_d - nearest.d)

        near_with_distance = [
            (
                math.hypot(new_s - node.s, args.lateral_weight * (new_d - node.d)),
                i,
            )
            for i, node in enumerate(nodes)
            if node.s < new_s - 0.05
            and math.hypot(new_s - node.s, args.lateral_weight * (new_d - node.d))
            <= args.rewire_radius_m
        ]
        near_with_distance.sort()
        near = [index for _, index in near_with_distance[: args.max_neighbors]]
        if nearest_index not in near:
            near.append(nearest_index)

        best_parent = -1
        best_cost = math.inf
        best_heading = nearest.heading
        for parent_index in near:
            free, length, heading = connection(nodes[parent_index], new_s, new_d)
            cost = nodes[parent_index].cost + length
            if free and cost < best_cost:
                best_parent, best_cost, best_heading = parent_index, cost, heading
        if best_parent < 0:
            continue

        new_index = len(nodes)
        nodes.append(RRTNode(new_s, new_d, best_parent, best_cost, best_heading))

        # Standard RRT* rewiring of nearby nodes that lie ahead of the new node.
        rewire_candidates = [
            (
                math.hypot(node.s - new_s, args.lateral_weight * (node.d - new_d)),
                index,
            )
            for index, node in enumerate(nodes[:-1])
            if node.s > new_s + 0.05
            and math.hypot(node.s - new_s, args.lateral_weight * (node.d - new_d))
            <= args.rewire_radius_m
        ]
        rewire_candidates.sort()
        for _, other_index in rewire_candidates[: args.max_neighbors]:
            other = nodes[other_index]
            free, length, heading = connection(nodes[new_index], other.s, other.d)
            if free and best_cost + length + 1e-6 < other.cost:
                other.parent = new_index
                other.cost = best_cost + length
                other.heading = heading

        if goal_s - new_s <= args.goal_connection_m:
            free, length, heading = connection(nodes[new_index], goal_s, goal_d)
            if free:
                nodes.append(RRTNode(goal_s, goal_d, new_index, best_cost + length, heading))
                goal_nodes.append(len(nodes) - 1)

    if not goal_nodes:
        raise RuntimeError(
            "RRT* did not connect to c_end; increase --iterations/--goal-connection-m "
            "or inspect the dynamic and wall clearances"
        )
    return nodes, goal_nodes


def candidate_from_tree_path(
    node_s: np.ndarray,
    node_d: np.ndarray,
    path_s: np.ndarray,
    path_indices: np.ndarray,
    geometry: TrackGeometry,
    trajectory: dict[str, np.ndarray],
    observation_time: np.ndarray,
    opponent_xy: np.ndarray,
    start_index: int,
    safe_lower: np.ndarray,
    safe_upper: np.ndarray,
    observed_reference_d: np.ndarray,
    args: argparse.Namespace,
) -> Candidate:
    unique_s, unique_indices = np.unique(node_s, return_index=True)
    unique_d = node_d[unique_indices]
    controls = np.interp(path_s, unique_s, unique_d)
    control_count = min(28, len(controls))
    control_indices = np.linspace(0, len(controls) - 1, control_count).astype(int)
    control_progress = np.linspace(0.0, 1.0, control_count)
    progress = np.linspace(0.0, 1.0, len(path_s))
    ego_radius = 0.5 * math.hypot(args.ego_length_m, args.ego_width_m)
    target_radius = 0.5 * math.hypot(args.target_length_m, args.target_width_m)
    return make_candidate(
        controls[control_indices], control_progress, progress, path_s, path_indices,
        geometry, args.planning_speed_mps, observation_time, opponent_xy,
        float(observation_time[start_index]), "box", 0.0,
        ego_radius + target_radius + args.vehicle_gap_m,
        args.ego_length_m, args.ego_width_m, args.target_length_m,
        args.target_width_m, args.engagement_box_buffer_m, args.wall_margin_m,
        args.curvature_limit_1pm, safe_lower, safe_upper, args.pass_margin_m,
        args.minimum_progress_gain_m, observed_reference_d,
        args.entry_blend_distance_m, args.exit_blend_distance_m,
        float(trajectory["ego_yaw_rad"][start_index]),
        math.radians(args.maximum_start_heading_error_deg),
    )


def evaluate_rrt_lateral(
    lateral: np.ndarray,
    path_s: np.ndarray,
    path_indices: np.ndarray,
    geometry: TrackGeometry,
    observation_time: np.ndarray,
    opponent_xy: np.ndarray,
    start_time: float,
    safe_lower: np.ndarray,
    safe_upper: np.ndarray,
    required_start_heading: float,
    args: argparse.Namespace,
    xy_override: np.ndarray | None = None,
) -> Candidate:
    lateral = np.clip(np.asarray(lateral, dtype=float), safe_lower, safe_upper)
    xy = (
        geometry.points[path_indices] + lateral[:, None] * geometry.normals[path_indices]
        if xy_override is None
        else np.asarray(xy_override, dtype=float)
    )
    heading = path_heading(xy)
    times = np.cumsum(
        np.linalg.norm(np.diff(xy, axis=0, prepend=xy[[0]]), axis=1)
    ) / max(args.planning_speed_mps, 0.1)
    moving_opponent = dynamic_evader(times, observation_time, opponent_xy, start_time)
    opponent_heading = path_heading(moving_opponent)
    dynamic_clearance = oriented_box_clearance(
        xy, heading, moving_opponent, opponent_heading,
        args.ego_length_m, args.ego_width_m,
        args.target_length_m, args.target_width_m,
        args.engagement_box_buffer_m,
    )
    footprint = rectangle_corners(xy, heading, args.ego_length_m, args.ego_width_m)
    wall_distance, _ = geometry.wall_tree.query(footprint.reshape(-1, 2), k=1)
    minimum_wall_distance = np.min(wall_distance.reshape(len(xy), 4), axis=1)
    ds = np.maximum(np.gradient(path_s), 1e-3)
    curvature = np.abs(np.gradient(np.unwrap(heading)) / ds)
    opponent_s, _, _ = geometry.project(moving_opponent)
    progress_advantage = signed_track_delta(path_s, opponent_s, geometry.track_length)
    progress_gain = float(progress_advantage[-1] - progress_advantage[0])
    pass_completed = bool(
        progress_advantage[-1] >= args.pass_margin_m
        and progress_gain >= args.minimum_progress_gain_m
    )
    start_heading_error = abs(
        float(wrap_angle(np.array([heading[0] - required_start_heading]))[0])
    )
    feasible = bool(
        np.all(dynamic_clearance >= 0.0)
        and np.all(minimum_wall_distance >= args.wall_margin_m)
        and np.max(curvature[2:-2]) <= args.curvature_limit_1pm
        and pass_completed
    )
    path_length = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    curvature_cost = float(np.mean(curvature**2))
    penetration = float(np.mean(np.maximum(-dynamic_clearance, 0.0) ** 2))
    proximity = float(np.mean(np.exp(-np.maximum(dynamic_clearance, 0.0) / 0.35)))
    wall_penalty = float(
        np.mean(np.maximum(args.wall_margin_m - minimum_wall_distance, 0.0) ** 2)
    )
    pass_shortfall = max(args.pass_margin_m - float(progress_advantage[-1]), 0.0)
    cost = (
        path_length + 1.8 * curvature_cost + 400.0 * penetration
        + 0.7 * proximity + 400.0 * wall_penalty
        + 40.0 * pass_shortfall**2 - 0.20 * float(progress_advantage[-1])
    )
    box_length = args.target_length_m + 2.0 * args.engagement_box_buffer_m
    box_width = args.target_width_m + 2.0 * args.engagement_box_buffer_m
    return Candidate(
        d=lateral, xy=xy, heading=heading, time=times,
        opponent_xy=moving_opponent, opponent_heading=opponent_heading,
        zone_radius=np.full(len(xy), 0.5 * math.hypot(box_length, box_width)),
        cost=cost, feasible=feasible,
        min_clearance=float(np.min(dynamic_clearance)),
        min_wall_clearance=float(np.min(minimum_wall_distance)),
        max_curvature=float(np.max(curvature[2:-2])),
        initial_progress_advantage=float(progress_advantage[0]),
        final_progress_advantage=float(progress_advantage[-1]),
        progress_gain=progress_gain, pass_completed=pass_completed,
        start_heading_error=start_heading_error,
    )


def save_validation(
    path: Path,
    candidate: Candidate,
    c_start: float,
    c_end: float,
    nodes: list[RRTNode],
    goal_count: int,
    feasible_rrt_candidates: int,
    used_warm_start: bool,
    args: argparse.Namespace,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerows(
            [
                ["planner", "RRT*"],
                ["selected_feasible", int(candidate.feasible)],
                ["pass_completed", int(candidate.pass_completed)],
                ["c_start_m", c_start],
                ["c_end_m", c_end],
                ["iterations", args.iterations],
                ["tree_nodes", len(nodes)],
                ["goal_connections", goal_count],
                ["feasible_rrt_candidates", feasible_rrt_candidates],
                ["used_warm_start", int(used_warm_start)],
                ["minimum_dynamic_clearance_m", candidate.min_clearance],
                ["minimum_footprint_wall_clearance_m", candidate.min_wall_clearance],
                ["maximum_curvature_1pm", candidate.max_curvature],
                ["initial_progress_advantage_m", candidate.initial_progress_advantage],
                ["final_progress_advantage_m", candidate.final_progress_advantage],
                ["progress_gain_m", candidate.progress_gain],
                ["cost", candidate.cost],
            ]
        )


def generate(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    trajectory = read_numeric_csv(args.trajectory_csv)
    walls = load_wall_points(args.wall_mask_csv)
    geometry = TrackGeometry(walls)
    reference = read_numeric_csv(args.reference_path_csv)
    c_start, c_end, start_d, goal_d = load_reference_endpoints(args.reference_path_csv)

    ego_xy = np.column_stack([trajectory["ego_x_m"], trajectory["ego_y_m"]])
    ego_s, ego_d, _ = geometry.project(ego_xy)
    start_index = int(np.argmin(np.abs(ego_s - c_start)))
    end_index = int(np.argmin(np.abs(ego_s - c_end)))
    path_s = np.linspace(c_start, c_end, args.path_points)
    path_indices = interpolate_track_indices(geometry, path_s)

    # Reuse the exact evader prediction and timing that produced the reference
    # result. Re-estimating it from the raw log would change the comparison.
    plan_start_time = float(trajectory["stamp_sec"][start_index])
    opponent_xy = np.column_stack([reference["evader_x_m"], reference["evader_y_m"]])
    observation_time = plan_start_time + reference["time_sec"]
    wall_clearance = 0.5 * math.hypot(args.ego_length_m, args.ego_width_m) + args.wall_margin_m
    safe_lower, safe_upper = connected_free_corridor(geometry, path_indices, wall_clearance)
    observed_reference_d = observed_lateral_profile(path_s, ego_s, ego_d)
    guide_d = np.interp(path_s, reference["s_m"], reference["lateral_offset_m"])

    nodes, goal_nodes = rrt_star(
        geometry, path_s, safe_lower, safe_upper, start_d, goal_d,
        float(trajectory["ego_yaw_rad"][start_index]), observation_time,
        opponent_xy, plan_start_time, guide_d, args,
    )

    candidates: list[tuple[Candidate, int]] = [
        (
            evaluate_rrt_lateral(
                guide_d, path_s, path_indices, geometry, observation_time,
                opponent_xy, plan_start_time, safe_lower,
                safe_upper, float(trajectory["ego_yaw_rad"][start_index]), args,
                xy_override=np.column_stack([reference["ego_x_m"], reference["ego_y_m"]]),
            ),
            goal_nodes[0],
        )
    ]
    for goal_index in sorted(goal_nodes, key=lambda i: nodes[i].cost)[: args.goal_candidates]:
        node_s, node_d = trace_path(nodes, goal_index)
        raw_lateral = np.interp(path_s, node_s, node_d)
        raw_xy = xy_from_sd(geometry, path_s, raw_lateral)
        for sigma in (1.2, 2.0, 3.0, 4.0):
            lateral = (
                raw_lateral.copy()
                if sigma == 0.0
                else gaussian_filter1d(raw_lateral, sigma=sigma, mode="nearest")
            )
            lateral[0] = start_d
            lateral[-1] = goal_d
            smoothed_xy = gaussian_filter1d(raw_xy, sigma=sigma, axis=0, mode="nearest")
            endpoint_correction = np.linspace(
                raw_xy[0] - smoothed_xy[0], raw_xy[-1] - smoothed_xy[-1], len(raw_xy)
            )
            smoothed_xy += endpoint_correction
            candidate = evaluate_rrt_lateral(
                lateral, path_s, path_indices, geometry, observation_time,
                opponent_xy, plan_start_time, safe_lower,
                safe_upper, float(trajectory["ego_yaw_rad"][start_index]), args,
                xy_override=smoothed_xy,
            )
            candidates.append((candidate, goal_index))
    warm_start_candidate = candidates[0]
    feasible_rrt = [item for item in candidates[1:] if item[0].feasible]
    if feasible_rrt:
        best, best_goal_index = min(feasible_rrt, key=lambda item: item[0].cost)
        used_warm_start = False
    else:
        best, best_goal_index = warm_start_candidate
        used_warm_start = True
    if not best.feasible:
        warm_start = warm_start_candidate[0]
        raise RuntimeError(
            "RRT* reached c_end but no smoothed route passed validation: "
            f"dynamic_clearance={best.min_clearance:.3f} m, "
            f"wall_clearance={best.min_wall_clearance:.3f} m, "
            f"curvature={best.max_curvature:.3f} 1/m, "
            f"progress_gain={best.progress_gain:.3f} m; "
            "warm_start="
            f"({warm_start.min_clearance:.3f} m dynamic, "
            f"{warm_start.min_wall_clearance:.3f} m wall, "
            f"{warm_start.max_curvature:.3f} 1/m curvature)"
        )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    path_output = output.with_name(f"{output.stem}_path.csv")
    validation_output = output.with_name(f"{output.stem}_validation.csv")
    save_path(path_output, path_s, best)
    save_validation(
        validation_output, best, c_start, c_end, nodes, len(goal_nodes),
        len(feasible_rrt), used_warm_start, args,
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(walls[:, 0], walls[:, 1], s=1.0, color="0.62", alpha=0.38, rasterized=True)
    ax.add_patch(
        Polygon(geometry.region_polygon(c_start, c_end), closed=True, color="#c77dff", alpha=0.16)
    )
    ax.plot(ego_xy[:, 0], ego_xy[:, 1], "--", color="#9a6700", linewidth=1.4)
    observed_opponent = np.column_stack([trajectory["opponent_x_m"], trajectory["opponent_y_m"]])
    ax.scatter(observed_opponent[:, 0], observed_opponent[:, 1], s=7, color="#f2c94c", alpha=0.65)

    stride = max(1, len(nodes) // args.tree_plot_edges)
    for index in range(1, len(nodes), stride):
        node = nodes[index]
        if node.parent < 0:
            continue
        parent = nodes[node.parent]
        edge_xy = xy_from_sd(
            geometry, np.array([parent.s, node.s]), np.array([parent.d, node.d])
        )
        ax.plot(edge_xy[:, 0], edge_xy[:, 1], color="#7a8b99", linewidth=0.35, alpha=0.12)

    box_length = args.target_length_m + 2.0 * args.engagement_box_buffer_m
    box_width = args.target_width_m + 2.0 * args.engagement_box_buffer_m
    boxes = rectangle_corners(best.opponent_xy, best.opponent_heading, box_length, box_width)
    for index in range(0, len(boxes), max(1, len(boxes) // 16)):
        ax.add_patch(
            Polygon(boxes[index], closed=True, facecolor="#f6a04d", edgecolor="#d95f02", alpha=0.24)
        )

    normal = np.column_stack([-np.sin(best.heading), np.cos(best.heading)])
    swept = np.vstack(
        [best.xy + 0.5 * args.ego_width_m * normal, (best.xy - 0.5 * args.ego_width_m * normal)[::-1]]
    )
    ax.add_patch(Polygon(swept, closed=True, facecolor="#4da3ff", edgecolor="none", alpha=0.22))
    ax.plot(best.xy[:, 0], best.xy[:, 1], color="#0066cc", linewidth=2.8)
    ax.scatter(best.xy[[0, -1], 0], best.xy[[0, -1], 1], s=34, color=["#9c27b0", "#d35400"])
    ax.annotate(r"$c_{start}$", best.xy[0], xytext=(5, 5), textcoords="offset points")
    ax.annotate(r"$c_{end}$", best.xy[-1], xytext=(5, 5), textcoords="offset points")
    ax.legend(
        handles=[
            Line2D([], [], color="#9a6700", linestyle="--", label="Observed ego"),
            Line2D([], [], color="#f2c94c", marker="o", linestyle="", label="Observed dynamic evader"),
            Line2D([], [], color="#7a8b99", linewidth=1.0, alpha=0.5, label="RRT* search tree"),
            Line2D([], [], color="#0066cc", linewidth=2.8, label="RRT* dynamic-EZ path"),
            Line2D([], [], color="#c77dff", linewidth=7, alpha=0.35, label="Confidence Region of Collision"),
        ],
        loc="best",
    )
    ax.text(
        0.015, 0.015,
        "VALIDATED RRT* OVERTAKE\n"
        f"tree: {len(nodes)} nodes, {len(goal_nodes)} goal connections\n"
        f"dynamic clearance: {best.min_clearance:.2f} m\n"
        f"footprint-wall clearance: {best.min_wall_clearance:.2f} m",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#159447"},
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("RRT* Planning Around a Dynamic Evader")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RRT* engagement-zone figure to {output}")
    print(f"Saved RRT* path to {path_output}")
    print(f"Saved RRT* validation to {validation_output}")
    print(
        f"c_start={c_start:.3f} m, c_end={c_end:.3f} m, "
        f"nodes={len(nodes)}, goals={len(goal_nodes)}, selected_goal={best_goal_index}, "
        f"feasible_rrt={len(feasible_rrt)}, warm_start={used_warm_start}, "
        f"clearance={best.min_clearance:.3f} m, wall={best.min_wall_clearance:.3f} m, "
        f"curvature={best.max_curvature:.3f} 1/m"
    )


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    output_root = package_root / "output"
    parser = argparse.ArgumentParser(description="Offline RRT* planning around the recorded dynamic evader")
    parser.add_argument(
        "--trajectory-csv", type=Path,
        default=output_root / "confidence_roc" / "confidence_roc_one_lap_trajectory.csv",
    )
    parser.add_argument(
        "--reference-path-csv", type=Path,
        default=output_root / "engagement_zones_dynamic" / "engagement_zone_dynamic_one_lap_path.csv",
        help="Working planner output whose exact c_start/c_end are reused",
    )
    parser.add_argument(
        "--wall-mask-csv", type=Path,
        default=output_root / "slam_runs" / "slam_toolbox_boundary_wall_mask.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=output_root / "engagement_zones_dynamic_rrt" / "engagement_zone_dynamic_rrt_one_lap.png",
    )
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--step-m", type=float, default=2.0)
    parser.add_argument("--rewire-radius-m", type=float, default=4.0)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--goal-connection-m", type=float, default=3.0)
    parser.add_argument("--goal-bias", type=float, default=0.10)
    parser.add_argument("--lateral-weight", type=float, default=1.5)
    parser.add_argument("--edge-check-spacing-m", type=float, default=0.12)
    parser.add_argument("--goal-candidates", type=int, default=80)
    parser.add_argument("--path-points", type=int, default=260)
    parser.add_argument("--tree-plot-edges", type=int, default=1800)
    parser.add_argument("--ego-length-m", type=float, default=0.50)
    parser.add_argument("--ego-width-m", type=float, default=0.30)
    parser.add_argument("--target-length-m", type=float, default=0.50)
    parser.add_argument("--target-width-m", type=float, default=0.30)
    parser.add_argument("--engagement-box-buffer-m", type=float, default=0.08)
    parser.add_argument("--rrt-dynamic-margin-m", type=float, default=0.08)
    parser.add_argument("--vehicle-gap-m", type=float, default=0.12)
    parser.add_argument("--wall-margin-m", type=float, default=0.05)
    parser.add_argument("--rrt-wall-margin-m", type=float, default=0.03)
    parser.add_argument("--curvature-limit-1pm", type=float, default=3.5)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--planning-speed-mps", type=float, default=3.90)
    parser.add_argument("--pass-margin-m", type=float, default=1.00)
    parser.add_argument("--minimum-progress-gain-m", type=float, default=2.00)
    parser.add_argument("--entry-blend-distance-m", type=float, default=0.50)
    parser.add_argument("--exit-blend-distance-m", type=float, default=0.50)
    parser.add_argument("--maximum-start-heading-error-deg", type=float, default=8.0)
    args = parser.parse_args()
    for path in (args.trajectory_csv, args.reference_path_csv, args.wall_mask_csv):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")
    generate(args)


if __name__ == "__main__":
    main()

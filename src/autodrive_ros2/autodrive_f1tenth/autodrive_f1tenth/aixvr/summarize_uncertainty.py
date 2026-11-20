#!/usr/bin/env python3
"""Summarize time-aligned AIxVR uncertainty experiments."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def _float(row, key, default=math.nan):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _nearest_indices(reference_times, query_times):
    indices = np.searchsorted(reference_times, query_times)
    indices = np.clip(indices, 0, len(reference_times) - 1)
    previous = np.maximum(indices - 1, 0)
    use_previous = (
        np.abs(query_times - reference_times[previous])
        < np.abs(query_times - reference_times[indices])
    )
    return np.where(use_previous, previous, indices)


def _rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values * values))) if len(values) else math.nan


def _max_abs(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.max(np.abs(values))) if len(values) else math.nan


def _variation(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sum(np.abs(np.diff(values)))) if len(values) > 1 else 0.0


def _events(flags):
    flags = np.asarray(flags, dtype=bool)
    return int(np.sum(flags & ~np.r_[False, flags[:-1]]))


def summarize(controller_path, bridge_path):
    controller = _read(controller_path)
    bridge = _read(bridge_path)
    if not controller or not bridge:
        raise ValueError("empty controller or uncertainty CSV")

    control_time = np.asarray([_float(row, "time_sec") for row in controller])
    truth_time = np.asarray([_float(row, "time_sec") for row in bridge])
    valid_control = np.isfinite(control_time)
    valid_truth = np.isfinite(truth_time)
    control_time = control_time[valid_control]
    controller = [row for row, valid in zip(controller, valid_control) if valid]
    truth_time = truth_time[valid_truth]
    bridge = [row for row, valid in zip(bridge, valid_truth) if valid]
    if not controller or not bridge:
        raise ValueError("no finite timestamps")
    order = np.argsort(truth_time)
    truth_time = truth_time[order]
    bridge = [bridge[index] for index in order]
    matched = _nearest_indices(truth_time, control_time)

    true_lateral = []
    true_longitudinal = []
    true_position = []
    true_velocity_error = []
    for row, truth_index in zip(controller, matched):
        truth = bridge[int(truth_index)]
        dx = _float(truth, "true_x_m") - _float(row, "reference_x_m")
        dy = _float(truth, "true_y_m") - _float(row, "reference_y_m")
        heading = _float(row, "reference_heading_rad")
        if all(math.isfinite(value) for value in (dx, dy, heading)):
            tangent = np.array([math.cos(heading), math.sin(heading)])
            normal = np.array([-math.sin(heading), math.cos(heading)])
            delta = np.array([dx, dy])
            true_longitudinal.append(float(delta @ tangent))
            true_lateral.append(float(delta @ normal))
            true_position.append(float(np.linalg.norm(delta)))
        true_speed = _float(truth, "true_speed_mps")
        target_speed = _float(row, "target_reference_speed_mps")
        true_velocity_error.append(true_speed - target_speed)

    boundary_flags = [int(_float(row, "boundary_violation", 0.0) >= 0.5) for row in bridge]
    crashed = any(_float(row, "wall_contact", 0.0) >= 0.5 for row in bridge)
    solve_flags = np.asarray([
        _float(row, "solver_success", 0.0) >= 0.5 for row in controller
    ])
    completed = (
        any(_float(row, "completed", 0.0) >= 0.5 for row in controller)
        and not crashed
    )
    completion_values = [
        _float(row, "completion_time_s") for row in controller
        if math.isfinite(_float(row, "completion_time_s"))
    ]
    first = bridge[0]
    model = controller[0].get("model", "unknown")
    return {
        "model": model,
        "experiment": first.get("experiment", "unknown"),
        "severity": _float(first, "severity"),
        "seed": int(_float(first, "seed", -1)),
        "completed": int(completed),
        "crashed": int(crashed),
        "completion_time_s": (
            float(completion_values[-1]) if completion_values else math.nan
        ),
        "true_lateral_rmse_m": _rms(true_lateral),
        "true_lateral_max_m": _max_abs(true_lateral),
        "true_longitudinal_rmse_m": _rms(true_longitudinal),
        "true_position_rmse_m": _rms(true_position),
        "true_position_max_m": _max_abs(true_position),
        "true_velocity_rmse_mps": _rms(true_velocity_error),
        "boundary_violation_samples": int(np.sum(boundary_flags)),
        "boundary_violation_events": _events(boundary_flags),
        "solver_success_rate": float(np.mean(solve_flags)) if len(solve_flags) else math.nan,
        "throttle_total_variation": _variation([
            _float(row, "applied_throttle") for row in bridge
        ]),
        "steering_total_variation_rad": _variation([
            _float(row, "applied_steering_rad") for row in bridge
        ]),
        "mean_position_measurement_error_m": float(np.nanmean([
            _float(row, "position_measurement_error_m") for row in bridge
        ])),
        "controller_csv": str(controller_path),
        "uncertainty_csv": str(bridge_path),
    }


def main(args=None):
    package_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=package_root / "output/aixvr_nmpc"
    )
    parser.add_argument("--output", type=Path, default=None)
    parsed = parser.parse_args(args)
    root = parsed.root.expanduser().resolve()
    bridge_root = root / "uncertainty"
    summaries = []
    for bridge_path in sorted(bridge_root.glob("*.csv")):
        controller_path = root / bridge_path.name
        if not controller_path.exists():
            print(f"Skipping {bridge_path.name}: controller CSV not found")
            continue
        try:
            summaries.append(summarize(controller_path, bridge_path))
        except ValueError as exc:
            print(f"Skipping {bridge_path.name}: {exc}")
    if not summaries:
        raise SystemExit(f"No complete uncertainty/controller CSV pairs under {root}")
    output = parsed.output or root / "aixvr_uncertainty_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {len(summaries)} uncertainty summaries to {output}")


if __name__ == "__main__":
    main()

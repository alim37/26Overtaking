#!/usr/bin/env python3
"""Overlay the four AIxVR NMPC model-comparison runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


MODEL_STYLES = {
    "epm": ("EPM", "#0072B2"),
    "dpm": ("DPM", "#D55E00"),
    "ddm": ("DDM", "#009E73"),
    "single_track": ("Single-track", "#CC79A7"),
}


def numeric_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    if name not in rows[0]:
        raise ValueError(f"CSV does not contain required column '{name}'")
    values = []
    for row in rows:
        try:
            values.append(float(row[name]))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def load_run(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    columns = {
        "time": "elapsed_time_s",
        "x": "x_m",
        "y": "y_m",
        "reference_x": "reference_x_m",
        "reference_y": "reference_y_m",
        "velocity": "speed_mps",
        "raw_velocity": "raw_speed_mps",
        "acceleration": "longitudinal_acceleration_mps2",
        "ex": "longitudinal_error_m",
        "ey": "lateral_error_m",
        "position_error": "position_error_m",
        "throttle": "throttle_command",
        "steering": "steering_command_rad",
        "solve_time": "solve_time_ms",
        "solver_success": "solver_success",
        "completed": "completed",
    }
    return {key: numeric_column(rows, name) for key, name in columns.items()}


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def rmse(values: np.ndarray) -> float:
    values = finite(values)
    return float(np.sqrt(np.mean(values * values))) if values.size else np.nan


def percentile(values: np.ndarray, quantile: float) -> float:
    values = finite(values)
    return float(np.percentile(values, quantile)) if values.size else np.nan


def write_summary(runs: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    fields = (
        "model", "completion_time_s", "mean_velocity_mps",
        "velocity_rmse_from_2p5_mps", "acceleration_rms_mps2",
        "longitudinal_error_rmse_m", "lateral_error_rmse_m",
        "position_error_rmse_m", "position_error_max_m",
        "mean_throttle_command", "steering_rms_rad",
        "median_solve_time_ms", "p95_solve_time_ms",
        "solver_success_percent",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model, data in runs.items():
            label = MODEL_STYLES[model][0]
            velocity = finite(data["velocity"])
            success = finite(data["solver_success"])
            writer.writerow({
                "model": label,
                "completion_time_s": float(finite(data["time"])[-1]),
                "mean_velocity_mps": float(np.mean(velocity)),
                "velocity_rmse_from_2p5_mps": rmse(velocity - 2.5),
                "acceleration_rms_mps2": rmse(data["acceleration"]),
                "longitudinal_error_rmse_m": rmse(data["ex"]),
                "lateral_error_rmse_m": rmse(data["ey"]),
                "position_error_rmse_m": rmse(data["position_error"]),
                "position_error_max_m": float(np.nanmax(data["position_error"])),
                "mean_throttle_command": float(np.nanmean(data["throttle"])),
                "steering_rms_rad": rmse(data["steering"]),
                "median_solve_time_ms": percentile(data["solve_time"], 50.0),
                "p95_solve_time_ms": percentile(data["solve_time"], 95.0),
                "solver_success_percent": 100.0 * float(np.mean(success)),
            })


def plot_comparison(
    runs: dict[str, dict[str, np.ndarray]], output_dir: Path
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "aixvr_nmpc_trajectory_comparison.png"
    errors_path = output_dir / "aixvr_nmpc_tracking_errors_comparison.png"
    velocity_path = output_dir / "aixvr_nmpc_velocity_comparison.png"
    acceleration_path = output_dir / "aixvr_nmpc_acceleration_comparison.png"
    controls_path = output_dir / "aixvr_nmpc_controls_comparison.png"
    summary_path = output_dir / "aixvr_nmpc_summary.csv"

    fig, ax = plt.subplots(figsize=(9, 8))
    first = next(iter(runs.values()))
    ax.plot(
        first["reference_x"], first["reference_y"], color="#555555",
        linewidth=1.4, linestyle="--", label="Reference",
    )
    for model, data in runs.items():
        label, color = MODEL_STYLES[model]
        ax.plot(data["x"], data["y"], color=color, linewidth=1.7, label=label)
    ax.scatter(first["x"][0], first["y"][0], color="#111111", s=38, zorder=5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("NMPC Model Comparison: Position")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=True)
    fig.tight_layout()
    fig.savefig(trajectory_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax, key, ylabel in zip(
        axes,
        ("ex", "ey"),
        ("Longitudinal error $e_x$ [m]", "Lateral error $e_y$ [m]"),
    ):
        for model, data in runs.items():
            label, color = MODEL_STYLES[model]
            ax.plot(
                data["time"], data[key], color=color, linewidth=1.15,
                alpha=0.92, label=label,
            )
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=4, frameon=True)
    axes[-1].set_xlabel("Elapsed time [s]")
    fig.suptitle("NMPC Tracking Errors")
    fig.tight_layout()
    fig.savefig(errors_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for model, data in runs.items():
        label, color = MODEL_STYLES[model]
        ax.plot(
            data["time"], data["velocity"], color=color,
            linewidth=1.35, alpha=0.95, label=label,
        )
    ax.axhline(2.5, color="#555555", linestyle="--", linewidth=1.1,
               label="Target")
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("NMPC Velocity Tracking")
    ax.grid(alpha=0.25)
    ax.legend(ncol=5, frameon=True)
    fig.tight_layout()
    fig.savefig(velocity_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for model, data in runs.items():
        label, color = MODEL_STYLES[model]
        ax.plot(
            data["time"], data["acceleration"], color=color,
            linewidth=1.15, alpha=0.92, label=label,
        )
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Longitudinal acceleration [m/s²]")
    ax.set_title("NMPC Longitudinal Acceleration")
    ax.grid(alpha=0.25)
    ax.legend(ncol=4, frameon=True)
    fig.tight_layout()
    fig.savefig(acceleration_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax, key, ylabel in zip(
        axes,
        ("throttle", "steering"),
        ("Throttle command", "Steering command [rad]"),
    ):
        for model, data in runs.items():
            label, color = MODEL_STYLES[model]
            # Completion publishes an intentional zero command after the lap.
            # Exclude that shutdown sample without trimming valid in-lap data.
            mask = np.isfinite(data["time"]) & (data["completed"] < 0.5)
            ax.plot(
                data["time"][mask], data[key][mask], color=color,
                linewidth=1.15, alpha=0.92, label=label,
            )
        if key == "steering":
            ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=4, frameon=True)
    axes[-1].set_xlabel("Elapsed time [s]")
    fig.suptitle("NMPC Control Commands")
    fig.tight_layout()
    fig.savefig(controls_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    write_summary(runs, summary_path)
    return (
        trajectory_path, errors_path, velocity_path, acceleration_path,
        controls_path, summary_path,
    )


def main() -> None:
    package_root = Path(__file__).resolve().parents[2]
    default_dir = package_root / "output" / "aixvr_nmpc"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_dir)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None else input_dir
    )
    runs = {}
    for model in MODEL_STYLES:
        path = input_dir / f"nominal_nmpc_{model}_ethz.csv"
        if not path.is_file():
            raise SystemExit(f"Missing required comparison run: {path}")
        runs[model] = load_run(path)

    outputs = plot_comparison(runs, output_dir)
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

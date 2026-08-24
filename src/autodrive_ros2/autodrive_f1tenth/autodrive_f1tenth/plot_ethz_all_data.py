#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def latest_run(default_dir: Path) -> Path:
    candidates = list(default_dir.glob("ethz_all_data_speed_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No ETHZ all-data CSV files found in {default_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def numeric_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    if name not in rows[0]:
        raise ValueError(f"CSV does not contain required column '{name}'")
    return np.asarray(
        [float(row[name]) if row[name].strip() else np.nan for row in rows],
        dtype=float,
    )


def load_run(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")

    timestamps_ns = numeric_column(rows, "logger_timestamp_nanoseconds")
    return {
        "time": (timestamps_ns - timestamps_ns[0]) * 1e-9,
        "x": numeric_column(rows, "x_m"),
        "y": numeric_column(rows, "y_m"),
        "velocity": numeric_column(rows, "velocity_mps"),
        "acceleration": numeric_column(rows, "acceleration_mps2"),
        "steering": numeric_column(rows, "steering_command"),
        "throttle": numeric_column(rows, "throttle_command"),
    }


def plot_run(csv_path: Path, output_dir: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_run(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_output = output_dir / f"{csv_path.stem}_path.png"
    telemetry_output = output_dir / f"{csv_path.stem}_telemetry.png"

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(data["x"], data["y"], color="#1f2933", linewidth=1.8, label="Ego path")
    ax.scatter(data["x"][0], data["y"][0], color="#2a9d3f", s=55, zorder=3, label="Start")
    ax.scatter(data["x"][-1], data["y"][-1], color="#d62828", s=55, zorder=3, label="End")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    signals = (
        ("velocity", "Velocity [m/s]", "#0077b6"),
        ("acceleration", "Acceleration [m/s^2]", "#e76f00"),
        ("steering", "Steering command", "#2a9d3f"),
        ("throttle", "Throttle command", "#d62828"),
    )
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for ax, (key, ylabel, color) in zip(axes, signals):
        ax.plot(data["time"], data[key], color=color, linewidth=1.25)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(telemetry_output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"{csv_path.name}: max velocity={np.nanmax(data['velocity']):.3f} m/s")
    print(f"Saved path plot to {path_output}")
    print(f"Saved telemetry plot to {telemetry_output}")
    return path_output, telemetry_output


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    default_dir = package_root / "output" / "ethz_all_data"
    parser = argparse.ArgumentParser(description="Plot ETHZ path and telemetry data")
    parser.add_argument(
        "csv",
        nargs="*",
        type=Path,
        help="One or more run CSVs; defaults to the newest run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults beside each input CSV",
    )
    args = parser.parse_args()

    csv_paths = args.csv or [latest_run(default_dir)]
    for raw_path in csv_paths:
        csv_path = raw_path.expanduser().resolve()
        if not csv_path.exists():
            raise SystemExit(f"CSV does not exist: {csv_path}")
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else csv_path.parent
        )
        plot_run(csv_path, output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit control-model data support and compare it with a live NMPC run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


RUNTIME_COLUMNS = {
    "VX": "speed_mps",
    "VY": "lateral_speed_mps",
    "YAW_RATE": "yaw_rate_rad_s",
    "THROTTLE_FB": "throttle_feedback",
    "STEERING_FB": "steering_feedback_rad",
    "THROTTLE_CMD": "throttle_command",
    "STEERING_CMD": "steering_command_rad",
}


def quantiles(values: np.ndarray) -> dict[str, float]:
    points = np.quantile(values, [0.0, 0.01, 0.5, 0.99, 1.0])
    return dict(zip(("min", "p01", "median", "p99", "max"), map(float, points)))


def load_runtime(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for feature, column in RUNTIME_COLUMNS.items():
        values = []
        for row in rows:
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        result[feature] = np.asarray(values, dtype=np.float64)
    # Runtime logs store Unity throttle commands; model features use PWM units.
    result["THROTTLE_CMD"] = result["THROTTLE_CMD"] / 0.27
    return result


def main() -> None:
    package_root = Path(__file__).resolve().parents[2]
    workspace_src = package_root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=workspace_src / "DynSSM/dataset/unity_ethz_epm_control_dt100_h5_r12.npz",
    )
    parser.add_argument("--runtime-csv", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    data = np.load(args.dataset)
    features = np.asarray(data["features"][:, :, :7], dtype=np.float64)
    names = [str(value) for value in data["feature_names"]]
    splits = data["dataset_splits"].astype(str)
    laps = data["lap_ids"].astype(int)
    train = features[splits == "train"]
    if not len(train):
        raise RuntimeError("Dataset has no training split")

    report: dict[str, object] = {
        "dataset": str(args.dataset),
        "windows": int(len(features)),
        "sample_time_sec": float(np.asarray(data["sample_time_sec"]).reshape(-1)[0]),
        "command_feature_semantics": str(
            np.asarray(data.get("command_feature_semantics", "legacy_zero")).reshape(-1)[0]
        ),
        "split_laps": {
            split: sorted(np.unique(laps[splits == split]).astype(int).tolist())
            for split in sorted(set(splits.tolist()))
        },
        "features": {},
        "warnings": [],
    }
    runtime = load_runtime(args.runtime_csv) if args.runtime_csv else {}

    for index, name in enumerate(names):
        train_values = train[:, :, index].reshape(-1)
        item: dict[str, object] = {
            "train": quantiles(train_values),
            "train_std": float(train_values.std()),
        }
        if name in runtime and len(runtime[name]):
            live = runtime[name]
            lower, upper = np.quantile(train_values, [0.001, 0.999])
            item["runtime"] = quantiles(live)
            outside_percent = float(
                100.0 * np.mean((live < lower) | (live > upper))
            )
            item["runtime_outside_train_p001_p999_percent"] = outside_percent
            if outside_percent > 5.0:
                report["warnings"].append(
                    f"{name}: {outside_percent:.1f}% of runtime samples are outside "
                    "the central training support."
                )
        report["features"][name] = item

    command_stds = [
        report["features"][name]["train_std"]
        for name in ("THROTTLE_CMD", "STEERING_CMD")
    ]
    if min(command_stds) < 1e-5:
        report["warnings"].append(
            "Command-history channels are constant; rebuild the NPZ with the current parser."
        )
    steering_support = report["features"]["STEERING_FB"]["train"]
    if max(abs(steering_support["min"]), abs(steering_support["max"])) < 0.30:
        report["warnings"].append(
            "Training steering feedback does not cover 0.30 rad; collect stronger safe corner excitation."
        )
    yaw_support = report["features"]["YAW_RATE"]["train"]
    if max(abs(yaw_support["min"]), abs(yaw_support["max"])) < 1.75:
        report["warnings"].append(
            "Training yaw-rate support is below 1.75 rad/s; high-curvature prediction is extrapolative."
        )

    text = json.dumps(report, indent=2)
    print(text)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

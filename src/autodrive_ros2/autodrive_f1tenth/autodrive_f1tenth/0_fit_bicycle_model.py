#!/usr/bin/env python3
"""Identify a fixed bicycle model from the held-out EPM system-ID dataset."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


MASS_KG = 3.47
REAR_AXLE_TO_CG_M = 0.17145
PWM_TO_SIM_SCALE = 0.27
NOMINAL_CM2 = 4.612560976
NOMINAL_CR0 = 4.38404878
NOMINAL_CR2 = 0.02962195122


def load_columns(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No telemetry rows found in {path}")
    numeric_names = (
        "elapsed_sec", "lap", "model_vx_mps", "model_yaw_rate_radps",
        "imu_accel_x_mps2", "applied_steering_command",
        "steering_feedback", "throttle_feedback",
    )
    columns = {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in numeric_names
    }
    columns["split"] = np.asarray([row["dataset_split"] for row in rows])
    return columns


def fit_geometry(data):
    train = (
        (data["split"] == "train")
        & (data["model_vx_mps"] > 0.5)
        & (np.abs(data["steering_feedback"]) > 0.015)
    )

    def residual(wheelbase, mask=train):
        length = float(np.atleast_1d(wheelbase)[0])
        steering = data["steering_feedback"][mask]
        speed = data["model_vx_mps"][mask]
        beta = np.arctan(
            (REAR_AXLE_TO_CG_M / length) * np.tan(steering)
        )
        predicted = speed * np.cos(beta) * np.tan(steering) / length
        return predicted - data["model_yaw_rate_radps"][mask]

    result = least_squares(
        residual,
        x0=[0.33],
        bounds=([0.20], [0.50]),
        loss="soft_l1",
        f_scale=0.10,
    )
    wheelbase = float(result.x[0])
    metrics = {}
    for split in ("train", "validation", "test"):
        mask = (
            (data["split"] == split)
            & (data["model_vx_mps"] > 0.5)
            & (np.abs(data["steering_feedback"]) > 0.015)
        )
        error = residual([wheelbase], mask)
        metrics[split] = error_metrics(error)
    return wheelbase, metrics


def actuator_transitions(data, delay_steps):
    transitions = []
    laps = data["lap"].astype(int)
    for lap in np.unique(laps):
        indices = np.flatnonzero(laps == lap)
        for offset in range(1 + delay_steps, len(indices)):
            current = indices[offset]
            previous = indices[offset - 1]
            command = indices[offset - 1] if delay_steps else current
            dt = data["elapsed_sec"][current] - data["elapsed_sec"][previous]
            if 0.02 < dt < 0.20:
                transitions.append((
                    data["split"][current],
                    dt,
                    data["steering_feedback"][previous],
                    data["applied_steering_command"][command],
                    data["steering_feedback"][current],
                ))
    return transitions


def fit_steering_actuator(data):
    candidates = []
    for delay_steps in (0, 1):
        transitions = actuator_transitions(data, delay_steps)

        def residual(parameters, split):
            gain, tau = parameters
            selected = [row for row in transitions if row[0] == split]
            return np.asarray([
                feedback
                + (1.0 - math.exp(-dt / tau))
                * (gain * command - feedback)
                - next_feedback
                for _, dt, feedback, command, next_feedback in selected
            ])

        result = least_squares(
            lambda parameters: residual(parameters, "train"),
            x0=[0.5234, 0.13],
            bounds=([0.10, 0.005], [1.0, 1.0]),
            loss="soft_l1",
            f_scale=0.005,
        )
        metrics = {
            split: error_metrics(residual(result.x, split))
            for split in ("train", "validation", "test")
        }
        candidates.append((metrics["validation"]["rmse"], delay_steps, result.x, metrics))
    _, delay_steps, parameters, metrics = min(candidates, key=lambda item: item[0])
    return float(parameters[0]), float(parameters[1]), int(delay_steps), metrics


def fit_motor_gain(data):
    mask = (
        (data["split"] == "train")
        & (data["model_vx_mps"] > 0.5)
        & (data["throttle_feedback"] > 0.027)
        & (np.abs(data["imu_accel_x_mps2"]) < 0.35)
    )
    speed = data["model_vx_mps"][mask]
    pwm = data["throttle_feedback"][mask] / PWM_TO_SIM_SCALE
    acceleration = data["imu_accel_x_mps2"][mask]
    samples = (
        MASS_KG * acceleration
        + NOMINAL_CM2 * speed * pwm
        + NOMINAL_CR0
        + NOMINAL_CR2 * speed * speed
    ) / pwm
    cm1 = float(np.median(samples))

    metrics = {}
    for split in ("train", "validation", "test"):
        selected = (
            (data["split"] == split)
            & (data["model_vx_mps"] > 0.5)
            & (np.abs(data["imu_accel_x_mps2"]) < 3.0)
        )
        speed = data["model_vx_mps"][selected]
        pwm = data["throttle_feedback"][selected] / PWM_TO_SIM_SCALE
        predicted = (
            (cm1 - NOMINAL_CM2 * speed) * pwm
            - NOMINAL_CR0
            - NOMINAL_CR2 * speed * speed
        ) / MASS_KG
        metrics[split] = error_metrics(
            predicted - data["imu_accel_x_mps2"][selected]
        )
    return cm1, metrics


def error_metrics(error):
    error = np.asarray(error, dtype=float)
    return {
        "samples": int(error.size),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
    }


def main():
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            package_root / "output/epm_sysid"
            / "epm_sysid_ethz_forward_20261012_122738.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=package_root / "config/bicycle_model_ethz.json",
    )
    args = parser.parse_args()

    data = load_columns(args.input)
    wheelbase, yaw_metrics = fit_geometry(data)
    steering_gain, steering_tau, steering_delay, steering_metrics = (
        fit_steering_actuator(data)
    )
    cm1, acceleration_metrics = fit_motor_gain(data)
    result = {
        "schema_version": 1,
        "model": "fixed_kinematic_bicycle",
        "source_csv": str(args.input.resolve()),
        "fit_policy": {
            "fit_split": "train",
            "selection_split": "validation",
            "final_report_split": "test",
            "learned_model": False,
        },
        "geometry": {
            "wheelbase_m": wheelbase,
            "rear_axle_to_cg_m": REAR_AXLE_TO_CG_M,
            "mass_kg": MASS_KG,
        },
        "steering_actuator": {
            "command_gain": steering_gain,
            "time_constant_sec": steering_tau,
            "command_delay_steps": steering_delay,
        },
        "longitudinal": {
            "Cm1": cm1,
            "Cm2": NOMINAL_CM2,
            "Cr0": NOMINAL_CR0,
            "Cr2": NOMINAL_CR2,
            "pwm_to_sim_scale": PWM_TO_SIM_SCALE,
            "note": "Cm1 identified; correlated drag terms retained at nominal values.",
        },
        "metrics": {
            "yaw_rate_radps": yaw_metrics,
            "steering_feedback_rad": steering_metrics,
            "longitudinal_acceleration_mps2": acceleration_metrics,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved fixed bicycle parameters to {args.output}")
    print(json.dumps(result["geometry"] | result["steering_actuator"], indent=2))
    for metric, values in result["metrics"].items():
        print(f"{metric}: validation RMSE={values['validation']['rmse']:.6f}, "
              f"test RMSE={values['test']['rmse']:.6f}")


if __name__ == "__main__":
    main()

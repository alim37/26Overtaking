#!/usr/bin/env python3
"""Train fair DPM and DDM baselines on the eight-lap AutoDRIVE dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VEHICLE = {"lf": 0.15875, "lr": 0.17145, "mass": 3.47, "Iz": 0.04712}
STATE_NAMES = ["VX", "VY", "YAW_RATE"]
DDM_FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 6]
DPM_FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 6]
DDM_FEATURE_NAMES = [
    "VX", "VY", "YAW_RATE", "THROTTLE_FB", "STEERING_FB",
    "THROTTLE_CMD", "STEERING_CMD",
]
DPM_FEATURE_NAMES = DDM_FEATURE_NAMES

DDM_PARAMETERS = [
    ("Bf", 5.579, 5.0, 30.0),
    ("Cf", 1.2, 0.5, 2.0),
    ("Df", 16.2497561, 8.463414634, 76.17073171),
    ("Ef", -0.083, -2.0, 0.0),
    ("Br", 5.3852, 5.0, 30.0),
    ("Cr", 1.2691, 0.5, 2.0),
    ("Dr", 14.70095122, 8.463414634, 76.17073171),
    ("Er", -0.019, -2.0, 0.0),
    ("Cm1", 24.29, 12.145, 48.58),
    ("Cm2", 4.612560976, 2.306280488, 6.5),
    ("Cr0", 4.38404878, 2.19202439, 7.0),
    ("Cr2", 0.02962195122, 0.01481097561, 0.05),
    ("Iz", 0.04712, 0.02356, 0.09424),
    ("Shf", -0.0013, -0.02, 0.02),
    ("Svf", 0.03639268293, -0.253902439, 0.253902439),
    ("Shr", -0.00376, -0.02, 0.02),
    ("Svr", 0.07701707317, -0.253902439, 0.253902439),
]
DPM_PARAMETERS = [
    ("Bf", 5.579, 5.0, 30.0),
    ("Cf", 1.2, 0.5, 2.0),
    ("Df", 16.2497561, 8.463414634, 76.17073171),
    ("Ef", -0.083, -2.0, 0.0),
    ("Br", 5.3852, 5.0, 30.0),
    ("Cr", 1.2691, 0.5, 2.0),
    ("Dr", 14.70095122, 8.463414634, 76.17073171),
    ("Er", -0.019, -2.0, 0.0),
    ("Frx", 0.0, -20.0, 20.0),
]


class Standardizer:
    """Minimal train-only standardizer with a pickle-stable transform API."""

    def fit(self, values):
        self.mean_ = np.asarray(values, dtype=np.float64).mean(axis=0)
        self.scale_ = np.asarray(values, dtype=np.float64).std(axis=0)
        self.scale_ = np.where(self.scale_ < 1e-12, 1.0, self.scale_)
        return self

    def transform(self, values):
        return (np.asarray(values, dtype=np.float64) - self.mean_) / self.scale_


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LapDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, rollout_labels, rollout_controls, indices, scaler):
        selected = features[:, :, indices]
        shape = selected.shape
        normalized = scaler.transform(selected.reshape(-1, shape[-1])).reshape(shape)
        self.features = torch.tensor(selected, dtype=torch.float32)
        self.normalized = torch.tensor(normalized, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.rollout_labels = torch.tensor(rollout_labels, dtype=torch.float32)
        self.rollout_controls = torch.tensor(rollout_controls, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.features[index], self.normalized[index], self.labels[index],
            self.rollout_labels[index], self.rollout_controls[index],
        )


class PhysicsParameterModel(nn.Module):
    def __init__(
        self, model_name: str, input_dim: int, parameters, sample_time: float,
        scaler_mean, scaler_scale,
    ):
        super().__init__()
        self.model_name = model_name
        self.sample_time = sample_time
        self.parameter_specs = parameters
        self.param_names = [item[0] for item in parameters]
        self.gru = nn.GRU(input_dim, 64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64 + input_dim, 128), nn.GELU(), nn.LayerNorm(128),
            nn.Dropout(0.1), nn.Linear(128, 96), nn.GELU(), nn.Linear(96, len(parameters)),
        )
        defaults = torch.tensor([item[1] for item in parameters], dtype=torch.float32)
        mins = torch.tensor([item[2] for item in parameters], dtype=torch.float32)
        maxes = torch.tensor([item[3] for item in parameters], dtype=torch.float32)
        self.register_buffer("defaults", defaults)
        self.register_buffer("mins", mins)
        self.register_buffer("maxes", maxes)
        self.register_buffer(
            "scaler_mean", torch.tensor(scaler_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "scaler_scale", torch.tensor(scaler_scale, dtype=torch.float32)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def infer_parameters(self, normalized):
        encoded, _ = self.gru(normalized)
        raw = self.head(torch.cat([encoded[:, -1], normalized[:, -1]], dim=1))
        # Start at published/default values and constrain adaptation to physical bounds.
        values = self.defaults + 0.5 * torch.tanh(raw) * (self.maxes - self.mins)
        values = torch.maximum(torch.minimum(values, self.maxes), self.mins)
        return {name: values[:, i] for i, name in enumerate(self.param_names)}, values

    @staticmethod
    def pacejka(alpha, b, c, d, e):
        ba = b * alpha
        return d * torch.sin(c * torch.atan(ba - e * (ba - torch.atan(ba))))

    def physics_step(self, state, controls, params, dt=None):
        dt = self.sample_time if dt is None else dt
        substeps = max(1, int(np.ceil(dt / 0.02)))
        step_dt = dt / substeps
        throttle_fb, steering_fb = controls[:, 0], controls[:, 1]
        # Desired commands are useful encoder context, but tire and motor
        # forces must use the measured actuator state that reached the car.
        steering = steering_fb
        throttle = throttle_fb

        for _ in range(substeps):
            vx, vy, yaw_rate = state[:, 0], state[:, 1], state[:, 2]
            safe_vx = torch.clamp(torch.abs(vx), min=0.05)
            shf = params.get("Shf", torch.zeros_like(vx))
            shr = params.get("Shr", torch.zeros_like(vx))
            svf = params.get("Svf", torch.zeros_like(vx))
            svr = params.get("Svr", torch.zeros_like(vx))
            alpha_f = steering - torch.atan2(VEHICLE["lf"] * yaw_rate + vy, safe_vx) + shf
            alpha_r = torch.atan2(VEHICLE["lr"] * yaw_rate - vy, safe_vx) + shr
            f_fy = svf + self.pacejka(alpha_f, params["Bf"], params["Cf"], params["Df"], params["Ef"])
            f_ry = svr + self.pacejka(alpha_r, params["Br"], params["Cr"], params["Dr"], params["Er"])
            if self.model_name == "ddm":
                f_rx = ((params["Cm1"] - params["Cm2"] * vx) * throttle
                        - params["Cr0"] - params["Cr2"] * vx.square())
                iz = params["Iz"]
            else:
                f_rx = params["Frx"]
                iz = torch.full_like(vx, VEHICLE["Iz"])
            derivative = torch.stack([
                (f_rx - f_fy * torch.sin(steering)) / VEHICLE["mass"] + vy * yaw_rate,
                (f_ry + f_fy * torch.cos(steering)) / VEHICLE["mass"] - vx * yaw_rate,
                (f_fy * VEHICLE["lf"] * torch.cos(steering) - f_ry * VEHICLE["lr"]) / iz,
            ], dim=1)
            state = state + step_dt * derivative
        return state

    def forward(self, raw, normalized):
        params, values = self.infer_parameters(normalized)
        state = raw[:, -1, :3]
        controls = raw[:, -1, 3:7]
        return self.physics_step(state, controls, params), params, values

    def rollout(self, raw, future_feedback, params):
        state = raw[:, -1, :3]
        predictions = []
        zeros = torch.zeros_like(future_feedback)
        controls = torch.cat([future_feedback, zeros], dim=2)
        for step in range(controls.shape[1]):
            if self.model_name == "dpm":
                normalized = (raw - self.scaler_mean) / torch.clamp(
                    self.scaler_scale, min=1e-8
                )
                params, _ = self.infer_parameters(normalized)
            state = self.physics_step(state, controls[:, step], params)
            predictions.append(state)
            if self.model_name == "dpm":
                next_features = torch.cat(
                    [state, future_feedback[:, step], zeros[:, step]], dim=1
                )
                raw = torch.cat([raw[:, 1:], next_features.unsqueeze(1)], dim=1)
        return torch.stack(predictions, dim=1)


def component_metrics(prediction, target):
    error = prediction - target
    return {
        "rmse": float(torch.sqrt(error.square().mean()).item()),
        "mae": float(error.abs().mean().item()),
        "rmse_vx": float(torch.sqrt(error[:, 0].square().mean()).item()),
        "rmse_vy": float(torch.sqrt(error[:, 1].square().mean()).item()),
        "rmse_yaw_rate": float(torch.sqrt(error[:, 2].square().mean()).item()),
    }


def evaluate(model, loader, delta_scales):
    model.eval()
    one_step_predictions, one_step_targets = [], []
    rollout_predictions, rollout_targets = [], []
    with torch.no_grad():
        for raw, normalized, labels, rollout_labels, controls in loader:
            raw, normalized = raw.to(DEVICE), normalized.to(DEVICE)
            labels, rollout_labels, controls = labels.to(DEVICE), rollout_labels.to(DEVICE), controls.to(DEVICE)
            prediction, params, _ = model(raw, normalized)
            rollout = model.rollout(raw, controls, params)
            one_step_predictions.append(prediction.cpu())
            one_step_targets.append(labels.cpu())
            rollout_predictions.append(rollout.reshape(-1, 3).cpu())
            rollout_targets.append(rollout_labels.reshape(-1, 3).cpu())
    one = component_metrics(torch.cat(one_step_predictions), torch.cat(one_step_targets))
    multi = component_metrics(torch.cat(rollout_predictions), torch.cat(rollout_targets))
    one["normalized_mse"] = float((((torch.cat(one_step_predictions) - torch.cat(one_step_targets)) / delta_scales.cpu()).square().mean()).item())
    multi["normalized_mse"] = float((((torch.cat(rollout_predictions) - torch.cat(rollout_targets)) / delta_scales.cpu()).square().mean()).item())
    return {"one_step": one, "rollout_12_step": multi}


def train_one(model_name, data, args, output_root):
    feature_indices = DDM_FEATURE_INDICES if model_name == "ddm" else DPM_FEATURE_INDICES
    feature_names = DDM_FEATURE_NAMES if model_name == "ddm" else DPM_FEATURE_NAMES
    parameters = DDM_PARAMETERS if model_name == "ddm" else DPM_PARAMETERS
    splits = data["dataset_splits"].astype(str)
    masks = {name: splits == name for name in ("train", "validation", "test")}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("Dataset must have non-empty train, validation, and test splits")
    train_features = data["features"][masks["train"]][:, :, feature_indices]
    scaler = Standardizer().fit(train_features.reshape(-1, len(feature_indices)))
    datasets = {
        name: LapDataset(
            data["features"][mask], data["labels"][mask], data["rollout_labels"][mask],
            data["rollout_controls"][mask], feature_indices, scaler,
        ) for name, mask in masks.items()
    }
    loaders = {
        name: torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=name == "train")
        for name, ds in datasets.items()
    }
    train_delta = data["labels"][masks["train"]] - data["features"][masks["train"], -1, :3]
    delta_scales = torch.tensor(np.maximum(train_delta.std(axis=0), 1e-4), dtype=torch.float32, device=DEVICE)
    model = PhysicsParameterModel(
        model_name, len(feature_indices), parameters, args.sample_time,
        scaler.mean_, scaler.scale_,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, epochs=args.epochs,
        steps_per_epoch=len(loaders["train"]), pct_start=0.12,
        div_factor=10.0, final_div_factor=1000.0,
    )
    output_dir = output_root / f"AIxVR_{model_name.upper()}" / "unity_ethz_8lap"
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    config = {
        "model": model_name.upper(), "dataset": str(args.dataset), "sample_time_sec": args.sample_time,
        "history": int(data["features"].shape[1]), "rollout_steps": int(data["rollout_labels"].shape[1]),
        "feature_names": feature_names, "state_names": STATE_NAMES, "vehicle": VEHICLE,
        "parameters": [{"name": n, "default": d, "min": lo, "max": hi} for n, d, lo, hi in parameters],
        "split_laps": {
            name: sorted(np.unique(data["lap_ids"][mask]).astype(int).tolist()) for name, mask in masks.items()
        },
        "split_windows": {name: int(mask.sum()) for name, mask in masks.items()},
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "rollout_loss_weight": args.rollout_weight, "seed": args.seed,
        "command_feature_semantics": str(
            np.asarray(data.get("command_feature_semantics", "legacy_zero")).reshape(-1)[0]
        ),
        "dpm_note": (
            "No VREF was recorded; DPM infers direct Frx from measured state, "
            "actuator feedback, and recorded command history."
            if model_name == "dpm" else None
        ),
    }
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    np.savez(
        output_dir / "normalization.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        feature_names=np.asarray(feature_names),
    )

    best_epoch, best_val, stale = 0, float("inf"), 0
    history = []
    print(f"\nTraining {model_name.upper()} on {DEVICE}: train={len(datasets['train'])}, val={len(datasets['validation'])}, test={len(datasets['test'])}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total, batches = 0.0, 0
        for raw, normalized, labels, rollout_labels, controls in loaders["train"]:
            raw, normalized = raw.to(DEVICE), normalized.to(DEVICE)
            labels, rollout_labels, controls = labels.to(DEVICE), rollout_labels.to(DEVICE), controls.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            prediction, params, values = model(raw, normalized)
            rollout = model.rollout(raw, controls, params)
            prediction_loss = ((prediction - labels) / delta_scales).square().mean()
            rollout_loss = ((rollout - rollout_labels) / delta_scales).square().mean()
            regularization = ((values - model.defaults) / (model.maxes - model.mins)).square().mean()
            loss = prediction_loss + args.rollout_weight * rollout_loss + 0.01 * regularization
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {model_name} loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_total += loss.item()
            batches += 1

        val_metrics = evaluate(model, loaders["validation"], delta_scales)
        val_score = val_metrics["one_step"]["normalized_mse"] + args.rollout_weight * val_metrics["rollout_12_step"]["normalized_mse"]
        history.append({"epoch": epoch, "train_loss": train_total / batches, "validation_score": val_score, **val_metrics})
        if val_score < best_val - 1e-7:
            best_val, best_epoch, stale = val_score, epoch, 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{model_name.upper()} epoch {epoch:03d}: train={train_total / batches:.5f} val_score={val_score:.5f} val_rmse={val_metrics['one_step']['rmse']:.5f} rollout_rmse={val_metrics['rollout_12_step']['rmse']:.5f}", flush=True)
        if stale >= args.patience:
            print(f"{model_name.upper()} early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
            break

    model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=DEVICE, weights_only=True))
    metrics = {
        "best_epoch": best_epoch,
        "validation": evaluate(model, loaders["validation"], delta_scales),
        "test": evaluate(model, loaders["test"], delta_scales),
    }
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "history.json").open("w") as handle:
        json.dump(history, handle, indent=2)
    print(f"{model_name.upper()} complete: best_epoch={best_epoch}, test_rmse={metrics['test']['one_step']['rmse']:.6f}, test_rollout_rmse={metrics['test']['rollout_12_step']['rmse']:.6f}", flush=True)
    return metrics


def parse_args():
    default_root = Path(__file__).resolve().parents[4] / "DynSSM"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_root / "dataset/unity_ethz_epm_control_dt100_h5_r12.npz")
    parser.add_argument("--output-root", type=Path, default=default_root / "output")
    parser.add_argument("--models", nargs="+", choices=("dpm", "ddm"), default=("dpm", "ddm"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--rollout-weight", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--sample-time", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    data = np.load(args.dataset)
    semantics = str(
        np.asarray(data.get("command_feature_semantics", "legacy_zero")).reshape(-1)[0]
    )
    command_std = np.asarray(data["features"][:, :, 5:7]).std(axis=(0, 1))
    if semantics != "desired_actuator_commands" or np.any(command_std < 1e-5):
        raise ValueError(
            "Dataset does not contain usable desired-command history. Rebuild it "
            "with the current csv_parser_segmented.py before training; "
            f"semantics={semantics}, command_std={command_std.tolist()}"
        )
    dataset_sample_time = float(np.asarray(data["sample_time_sec"]).reshape(-1)[0])
    if not np.isclose(dataset_sample_time, args.sample_time):
        raise ValueError(f"Dataset timestep {dataset_sample_time} != requested {args.sample_time}")
    results = {}
    for model_name in args.models:
        seed_everything(args.seed)
        results[model_name] = train_one(model_name, data, args, args.output_root)
    print("\nFinal held-out lap results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

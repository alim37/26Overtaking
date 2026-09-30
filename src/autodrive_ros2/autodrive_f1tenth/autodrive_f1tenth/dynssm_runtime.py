#!/usr/bin/env python3

"""Minimal runtime-only loader for the trained DynSSM ORCA parameter model."""

import json
import pickle
from pathlib import Path

import numpy as np


class _ScalerState:
    """Receives the saved StandardScaler state without requiring scikit-learn."""


class _ScalerUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "sklearn.preprocessing._data" and name == "StandardScaler":
            return _ScalerState
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


class DynSSMRuntime:
    PARAMETER_NAMES = (
        "Bf", "Cf", "Df", "Ef", "Br", "Cr", "Dr", "Er", "Cm1",
        "Cm2", "Cr0", "Cr2", "Iz", "Shf", "Svf", "Shr", "Svr",
    )

    def __init__(self, checkpoint, config_path, scaler_path):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise RuntimeError(
                "DynSSM requires PyTorch in the ROS Python environment. Install the "
                "CPU wheel with: python3 -m pip install --user torch"
            ) from exc

        self.torch = torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with Path(config_path).open(encoding="utf-8") as stream:
            self.config = json.load(stream)
        with Path(scaler_path).open("rb") as stream:
            scaler = _ScalerUnpickler(stream).load()
        self.scaler_mean = np.asarray(scaler.mean_, dtype=np.float32)
        self.scaler_scale = np.asarray(scaler.scale_, dtype=np.float32)
        self.horizon = int(self.config["MODEL"]["HORIZON"])

        config = self.config

        class DiagonalSSMEncoder(nn.Module):
            def __init__(self, input_dim, ssm_dim):
                super().__init__()
                self.in_proj = nn.Linear(input_dim, ssm_dim)
                self.gate_proj = nn.Linear(input_dim, ssm_dim)
                self.delta_proj = nn.Linear(input_dim, ssm_dim)
                self.skip_proj = nn.Linear(input_dim, ssm_dim)
                self.log_decay = nn.Parameter(torch.linspace(-3.0, -0.2, ssm_dim))
                self.norm = nn.LayerNorm(ssm_dim)

            def forward(self, values):
                state = values.new_zeros(values.shape[0], self.log_decay.numel())
                decay = torch.exp(-torch.nn.functional.softplus(self.log_decay)).view(1, -1)
                outputs = []
                for step in range(values.shape[1]):
                    current = values[:, step, :]
                    delta = torch.zeros_like(current) if step == 0 else current - values[:, step - 1, :]
                    drive = torch.tanh(self.in_proj(current))
                    gate = torch.sigmoid(self.gate_proj(current))
                    state = decay * state + gate * drive + 0.1 * torch.tanh(self.delta_proj(delta))
                    outputs.append(self.norm(state + self.skip_proj(current)))
                encoded = torch.stack(outputs, dim=1)
                return torch.cat(
                    [encoded[:, -1], encoded.mean(dim=1), encoded.std(dim=1, unbiased=False)],
                    dim=1,
                )

        class ParameterNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                model = config["MODEL"]
                input_dim = len(config["STATE"]) + len(config["ACTIONS"])
                ssm_dim = int(model["SSM_DIM"])
                gru_hidden = int(model["GRU_HIDDEN"])
                gru_layers = int(model["GRU_LAYERS"])
                latent_dim = int(model["LATENT_DIM"])
                hidden_dim = int(model["HIDDEN_DIM"])
                dropout = float(model.get("DROPOUT", 0.0))
                self.ssm_encoder = DiagonalSSMEncoder(input_dim, ssm_dim)
                self.gru = nn.GRU(
                    input_dim,
                    gru_hidden,
                    num_layers=gru_layers,
                    batch_first=True,
                    dropout=dropout if gru_layers > 1 else 0.0,
                )
                self.fusion = nn.Sequential(
                    nn.Linear(ssm_dim * 3 + gru_hidden + input_dim, hidden_dim),
                    nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, latent_dim), nn.GELU(), nn.LayerNorm(latent_dim),
                )
                self.physics_conditioner = nn.Sequential(
                    nn.Linear(latent_dim + input_dim, hidden_dim),
                    nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                )
                self.param_head = nn.Linear(hidden_dim, len(config["PARAMETERS"]))
                defaults = [item[next(iter(item))] for item in config["PARAMETERS"]]
                minimums = [item["Min"] for item in config["PARAMETERS"]]
                maximums = [item["Max"] for item in config["PARAMETERS"]]
                self.register_buffer("param_defaults", torch.tensor(defaults, dtype=torch.float32))
                self.register_buffer("param_mins", torch.tensor(minimums, dtype=torch.float32))
                self.register_buffer("param_maxes", torch.tensor(maximums, dtype=torch.float32))
                self.register_buffer("param_ranges", self.param_maxes - self.param_mins)

            def forward(self, normalized):
                ssm = self.ssm_encoder(normalized)
                gru, _ = self.gru(normalized)
                last = normalized[:, -1]
                fused = self.fusion(torch.cat([ssm, gru[:, -1], last], dim=1))
                conditioned = self.physics_conditioner(torch.cat([fused, last], dim=1))
                scale = float(config["MODEL"]["OPTIMIZATION"].get("PARAM_ADAPT_SCALE", 0.2))
                values = self.param_defaults + scale * torch.tanh(self.param_head(conditioned)) * self.param_ranges
                return torch.maximum(torch.minimum(values, self.param_maxes), self.param_mins)

        self.model = ParameterNetwork().to(self.device)
        try:
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=self.device)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        state = {key.removeprefix("module."): value for key, value in state.items()}
        missing, _ = self.model.load_state_dict(state, strict=False)
        critical = [key for key in missing if key.startswith(("ssm_encoder", "gru", "fusion", "physics_conditioner", "param_head"))]
        if critical:
            raise RuntimeError(f"DynSSM checkpoint is incompatible; missing keys: {critical[:5]}")
        self.model.eval()

    def infer_parameters(self, history):
        history = np.asarray(history, dtype=np.float32)
        if history.shape != (self.horizon, 7):
            raise ValueError(f"Expected DynSSM history {(self.horizon, 7)}, got {history.shape}")
        normalized = (history - self.scaler_mean) / np.maximum(self.scaler_scale, 1e-8)
        tensor = self.torch.from_numpy(normalized[None]).to(self.device)
        with self.torch.no_grad():
            values = self.model(tensor)[0].detach().cpu().numpy()
        return dict(zip(self.PARAMETER_NAMES, values.astype(float)))

#!/usr/bin/env python3

import math
import pickle
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import CubicSpline

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32


MANUAL_WAYPOINTS = np.array(
    # [
    #     [-18.36, 6.65],
    #     [-15.58, 9.96],
    #     [-13.48, 13.52],
    #     [-14.84, 14.23],
    #     [-17.55, 12.61],
    #     [-20.30, 11.84],
    #     [-21.01, 13.26],
    #     [-20.05, 16.53],
    #     [-16.70, 18.70],
    #     [-12.79, 18.10],
    #     [-10.33, 14.85],
    #     [-10.63, 11.81],
    #     [-13.47, 7.60],
    #     [-15.05, 5.10],
    #     [-14.27, 3.42],
    #     [-13.75, 3.84],
    #     [-9.68, 3.97],
    #     [-7.65, 5.41],
    #     [-6.55, 5.81],
    #     [-5.21, 5.33],
    #     [-3.53, 3.67],
    #     [-2.38, 4.10],
    #     [-1.34, 8.18],
    #     [-3.47, 12.89],
    #     [-5.78, 18.30],
    #     [-10.82, 22.31],
    #     [-15.83, 22.63],
    #     [-22.29, 19.07],
    #     [-25.37, 10.40],
    #     [-24.24, 3.83],
    # ],
    [
        [-18.36, 6.65],
        [-16.99, 8.09],
        [-13.64, 13.70],
        [-14.82, 14.36],
        [-16.87, 12.91],
        [-19.51, 11.94],
        [-20.67, 12.61],
        [-20.26, 16.45],
        [-15.74, 19.00],
        [-11.82, 17.19],
        [-10.64, 11.93],
        [-14.88, 5.38],
        [-14.98, 4.08],
        [-10.00, 3.90],
        [-6.70, 6.18],
        [-5.21, 4.96],
        [-3.55, 3.76],
        [-2.96, 3.65],
        [-2.36, 3.29],
        [-2.02, 4.54],
        [-1.32, 8.37],
        [-2.73, 11.49],
        [-3.47, 12.89],
        [-5.78, 18.30],
        [-10.82, 22.31],
        [-15.83, 22.63],
        [-22.29, 19.07],
        [-25.37, 10.40],
        [-24.24, 3.83],
    ],
    dtype=float,
)

# Edit these values for quick direct runs with:
# python3 pure_pursuit_ethz_scaled.py
# ROS parameter overrides still work and will take precedence when provided.
LOCAL_RUN_CONFIG = {
    "model_name": "lepavd",
    "use_learned_model": False,
    "use_predicted_pose": False,
    "enable_logging": False,
    "log_dir": "output/ros2_pure_pursuit_runs",
    "log_label": "",
    #"num_path_points": 400,
    "num_path_points": 800,
    "lookahead_distance": 2.5,
    "wheelbase": 0.30,
    #"target_speed": 0.135,
    #"target_speed": 0.13,
    "target_speed": 0.13,
    "control_period": 0.10,
    "max_steer_deg": 90.0,
}


def catmull_rom_chain(points: np.ndarray, num_points: int = 400) -> np.ndarray:
    padded = np.vstack([points[0], points, points[-1]])
    t = np.linspace(0.0, 1.0, len(padded))
    cs_x = CubicSpline(t, padded[:, 0], bc_type="clamped")
    cs_y = CubicSpline(t, padded[:, 1], bc_type="clamped")
    ts = np.linspace(0.0, 1.0, num_points)
    return np.vstack([cs_x(ts), cs_y(ts)]).T


def get_benchmark_root(default_root: str | None = None) -> Path:
    if default_root is not None:
        return Path(default_root).expanduser()
    return Path(__file__).resolve().parents[3]


def load_manual_reference_line(num_points: int = 400) -> np.ndarray:
    return catmull_rom_chain(MANUAL_WAYPOINTS, num_points=num_points)


class RunLogger:
    def __init__(self, repo_root: Path, model_name: str, target_speed: float, out_dir: str, label: str = "") -> None:
        self.repo_root = Path(repo_root)
        self.model_name = model_name
        self.target_speed = float(target_speed)
        self.out_dir = self.repo_root / out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.label = label.strip() if label else f"{self.model_name}_{self.target_speed:.3f}".rstrip("0").rstrip(".")

        self.time: list[float] = []
        self.inputs: list[list[float]] = []
        self.states: list[np.ndarray] = []
        self.meas_states: list[np.ndarray] = []
        self.pred_states: list[np.ndarray] = []

    def append(
        self,
        t: float,
        control_state: np.ndarray,
        measured_state: np.ndarray,
        predicted_state: np.ndarray,
        throttle_cmd: float,
        steering_cmd: float,
    ) -> None:
        self.time.append(float(t))
        self.inputs.append([float(throttle_cmd), float(steering_cmd)])
        self.states.append(np.asarray(control_state, dtype=float).copy())
        self.meas_states.append(np.asarray(measured_state, dtype=float).copy())
        self.pred_states.append(np.asarray(predicted_state, dtype=float).copy())

    def save(self, path_samples: np.ndarray) -> Path | None:
        if not self.time:
            return None

        save_path = self.out_dir / f"{self.label}.npz"
        np.savez(
            save_path,
            model_name=self.model_name,
            target_speed=self.target_speed,
            time=np.asarray(self.time, dtype=float),
            inputs=np.asarray(self.inputs, dtype=float).T,
            states=np.stack(self.states, axis=1),
            meas_states=np.stack(self.meas_states, axis=1),
            pred_states=np.stack(self.pred_states, axis=1),
            path=path_samples.T,
            waypoints=MANUAL_WAYPOINTS.T,
        )
        return save_path


@dataclass
class CarRuntimeState:
    car_id: int
    idx: int = 0
    idx_initialized: bool = False
    pos: tuple[float, float] | None = None
    prev_pos: tuple[float, float] | None = None
    prev_time: float | None = None
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    steering_fb: float = 0.0
    throttle_fb: float = 0.0
    measured_yaw: float | None = None
    model_predictor: "RosLearnedStatePredictor | None" = None
    model_state: np.ndarray | None = None
    predictor_error: str | None = None
    run_logger: RunLogger | None = None
    start_pose_xy: tuple[float, float] | None = None
    total_distance_m: float = 0.0
    left_start_zone: bool = False
    completed_lap: bool = False


class RosLearnedStatePredictor:
    """Checkpoint-backed CAPE dynamics predictor for [x, y, psi, vx, vy, yaw_rate]."""

    def __init__(self, model_name: str, repo_root: Path, device_name: str = "auto") -> None:
        import torch

        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from model.models_ddm import string_to_model as ddm_string_to_model
        from model.models_lePAVD import string_to_lePAVD as lep_string_to_model

        self.torch = torch
        self.model_name = model_name.lower()
        self.repo_root = Path(repo_root)
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)

        if self.model_name == "ddm":
            cfg = self.repo_root / "cfgs" / "model" / "deep_dynamics.yaml"
            ckpt = self.repo_root / "output" / "deep_dynamics" / "ETHZ_5" / "epoch_359.pth"
            scaler_path = self.repo_root / "output" / "deep_dynamics" / "ETHZ_5" / "scaler.pkl"
            with open(cfg, "rb") as f:
                param_dict = yaml.load(f, Loader=yaml.SafeLoader)
            net = ddm_string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval=True)
        elif self.model_name == "lepavd":
            cfg = self.repo_root / "cfgs" / "model" / "lePAVD.yaml"
            ckpt = self.repo_root / "output" / "lePAVD" / "ETHZ_5" / "epoch_2919.pth"
            scaler_path = self.repo_root / "output" / "lePAVD" / "ETHZ_5" / "scaler.pkl"
            with open(cfg, "rb") as f:
                param_dict = yaml.load(f, Loader=yaml.SafeLoader)
            net = lep_string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval_mode=True)
        elif self.model_name == "ina":
            cfg = self.repo_root / "cfgs" / "model" / "ina.yaml"
            ckpt = self.repo_root / "output" / "ina" / "ina_ethz_5" / "epoch_391.pth"
            scaler_path = self.repo_root / "output" / "ina" / "ina_ethz_5" / "scaler.pkl"
            with open(cfg, "rb") as f:
                param_dict = yaml.load(f, Loader=yaml.SafeLoader)
            net = lep_string_to_model[param_dict["MODEL"]["NAME"]](param_dict, eval_mode=True)
        else:
            raise ValueError("model_name must be one of: ddm, lepavd, ina")

        net.load_state_dict(torch.load(ckpt, map_location=self.device))
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.net = net.to(self.device).eval()
        self.horizon = int(net.horizon)
        self._hist: deque[np.ndarray] = deque(maxlen=self.horizon)
        self._hidden = self.net.init_hidden(1).to(self.device) if getattr(self.net, "is_rnn", False) else None

    def reset(self, state: np.ndarray, throttle_fb: float, steering_fb: float) -> None:
        row = np.array(
            [state[3], state[4], state[5], throttle_fb, steering_fb, 0.0, 0.0],
            dtype=float,
        )
        self._hist.clear()
        for _ in range(self.horizon):
            self._hist.append(row.copy())
        if getattr(self.net, "is_rnn", False):
            self._hidden = self.net.init_hidden(1).to(self.device)

    def predict_next(
        self,
        state: np.ndarray,
        throttle_fb: float,
        steering_fb: float,
        throttle_cmd: float,
        steering_cmd: float,
        dt: float,
    ) -> np.ndarray:
        feature = np.array(
            [
                state[3],
                state[4],
                state[5],
                throttle_fb,
                steering_fb,
                throttle_cmd - throttle_fb,
                steering_cmd - steering_fb,
            ],
            dtype=float,
        )
        self._hist.append(feature)

        hist = np.asarray(self._hist, dtype=float)
        x_raw = hist.reshape(1, self.horizon, -1)
        x_norm = self.scaler.transform(hist).reshape(1, self.horizon, -1)

        x_raw_t = self.torch.from_numpy(x_raw).float().to(self.device)
        x_norm_t = self.torch.from_numpy(x_norm).float().to(self.device)

        with self.torch.no_grad():
            if getattr(self.net, "is_rnn", False):
                out, h_next, _ = self.net(x_raw_t, x_norm_t, self._hidden)
                self._hidden = h_next.detach()
            else:
                out, _, _ = self.net(x_raw_t, x_norm_t)

        vx_next, vy_next, yaw_rate_next = out[0].detach().cpu().numpy().reshape(-1)

        next_state = np.asarray(state, dtype=float).copy()
        next_state[3] = float(vx_next)
        next_state[4] = float(vy_next)
        next_state[5] = float(yaw_rate_next)

        psi_mid = state[2] + 0.5 * dt * next_state[5]
        next_state[2] = state[2] + dt * next_state[5]
        next_state[0] = state[0] + dt * (
            next_state[3] * math.cos(psi_mid) - next_state[4] * math.sin(psi_mid)
        )
        next_state[1] = state[1] + dt * (
            next_state[3] * math.sin(psi_mid) + next_state[4] * math.cos(psi_mid)
        )
        return next_state


class PurePursuitF1Tenth(Node):
    def __init__(self) -> None:
        super().__init__("pure_pursuit_f1tenth")

        repo_root = get_benchmark_root()

        self.declare_parameter("benchmark_root", str(repo_root))
        self.declare_parameter("model_name", LOCAL_RUN_CONFIG["model_name"])
        self.declare_parameter("use_learned_model", LOCAL_RUN_CONFIG["use_learned_model"])
        self.declare_parameter("use_predicted_pose", LOCAL_RUN_CONFIG["use_predicted_pose"])
        self.declare_parameter("enable_logging", LOCAL_RUN_CONFIG["enable_logging"])
        self.declare_parameter("log_dir", LOCAL_RUN_CONFIG["log_dir"])
        self.declare_parameter("log_label", LOCAL_RUN_CONFIG["log_label"])
        self.declare_parameter("num_path_points", LOCAL_RUN_CONFIG["num_path_points"])
        self.declare_parameter("lookahead_distance", LOCAL_RUN_CONFIG["lookahead_distance"])
        self.declare_parameter("wheelbase", LOCAL_RUN_CONFIG["wheelbase"])
        self.declare_parameter("target_speed", LOCAL_RUN_CONFIG["target_speed"])
        self.declare_parameter("control_period", LOCAL_RUN_CONFIG["control_period"])
        self.declare_parameter("max_steer_deg", LOCAL_RUN_CONFIG["max_steer_deg"])
        self.declare_parameter("slow_steer_deg", 10.0)
        self.declare_parameter("high_steer_speed_scale", 0.65)
        self.declare_parameter("active_car_ids", [1, 2])
        self.declare_parameter("stop_after_one_lap", False)
        self.declare_parameter("lap_start_radius_m", 1.5)
        self.declare_parameter("lap_min_distance_m", 35.0)

        self.repo_root = Path(str(self.get_parameter("benchmark_root").value)).expanduser()
        self.model_name = str(self.get_parameter("model_name").value)
        self.use_learned_model = bool(self.get_parameter("use_learned_model").value)
        self.use_predicted_pose = bool(self.get_parameter("use_predicted_pose").value)
        self.enable_logging = bool(self.get_parameter("enable_logging").value)
        self.log_dir = str(self.get_parameter("log_dir").value)
        self.log_label = str(self.get_parameter("log_label").value)
        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        self.wheelbase = float(self.get_parameter("wheelbase").value)
        self.target_speed = float(self.get_parameter("target_speed").value)
        self.control_period = float(self.get_parameter("control_period").value)
        self.max_steer = math.radians(float(self.get_parameter("max_steer_deg").value))
        self.slow_steer_deg = float(self.get_parameter("slow_steer_deg").value)
        self.high_steer_speed_scale = float(self.get_parameter("high_steer_speed_scale").value)
        active_car_ids_raw = self.get_parameter("active_car_ids").value
        self.stop_after_one_lap = bool(self.get_parameter("stop_after_one_lap").value)
        self.lap_start_radius_m = float(self.get_parameter("lap_start_radius_m").value)
        self.lap_min_distance_m = float(self.get_parameter("lap_min_distance_m").value)
        self.path = load_manual_reference_line(int(self.get_parameter("num_path_points").value))
        self.follow_active_car1 = False
        self.safety_active_car1 = False
        self.active_car_ids = self._parse_active_car_ids(active_car_ids_raw)
        self.car_states: dict[int, CarRuntimeState] = {
            car_id: CarRuntimeState(car_id=car_id) for car_id in self.active_car_ids
        }
        self.steer_pubs: dict[int, any] = {}
        self.throttle_pubs: dict[int, any] = {}

        for car_id, state in self.car_states.items():
            label = f"{self.log_label}_car{car_id}" if self.log_label else f"{self.model_name}_car{car_id}_{self.target_speed:.3f}".rstrip("0").rstrip(".")
            if self.enable_logging:
                state.run_logger = RunLogger(
                    repo_root=self.repo_root,
                    model_name=self.model_name,
                    target_speed=self.target_speed,
                    out_dir=self.log_dir,
                    label=label,
                )

            if self.use_learned_model:
                try:
                    state.model_predictor = RosLearnedStatePredictor(self.model_name, self.repo_root)
                    self.get_logger().info(
                        f"Loaded learned model '{self.model_name}' for car {car_id} with horizon {state.model_predictor.horizon}"
                    )
                except Exception as exc:
                    state.predictor_error = str(exc)
                    self.get_logger().error(
                        f"Failed to load learned model '{self.model_name}' for car {car_id}. Falling back to measured-state pure pursuit. Error: {exc}"
                    )

            self.steer_pubs[car_id] = self.create_publisher(
                Float32,
                f"/autodrive/f1tenth_{car_id}/steering_command",
                10,
            )
            self.throttle_pubs[car_id] = self.create_publisher(
                Float32,
                f"/autodrive/f1tenth_{car_id}/throttle_command",
                10,
            )

            self.create_subscription(
                Point,
                f"/autodrive/f1tenth_{car_id}/ips",
                lambda msg, current_car_id=car_id: self.ips_cb(current_car_id, msg),
                10,
            )
            self.create_subscription(
                Imu,
                f"/autodrive/f1tenth_{car_id}/imu",
                lambda msg, current_car_id=car_id: self.imu_cb(current_car_id, msg),
                10,
            )
            self.create_subscription(
                Float32,
                f"/autodrive/f1tenth_{car_id}/steering",
                lambda msg, current_car_id=car_id: self.steer_fb_cb(current_car_id, msg),
                10,
            )
            self.create_subscription(
                Float32,
                f"/autodrive/f1tenth_{car_id}/throttle",
                lambda msg, current_car_id=car_id: self.throttle_fb_cb(current_car_id, msg),
                10,
            )

        self.create_subscription(
            Bool,
            "/autodrive/f1tenth_1/target_tracker/follow_active",
            self.follow_active_cb,
            10,
        )
        self.create_subscription(
            Bool,
            "/autodrive/f1tenth_1/safety/active",
            self.safety_active_cb,
            10,
        )

        self.create_timer(self.control_period, self.control_loop)
        self.get_logger().info(
            f"Pure pursuit ready with {len(self.path)} spline samples, lookahead={self.lookahead_distance:.2f}, "
            f"target_speed={self.target_speed:.2f}, learned_model={self.use_learned_model}, "
            f"active_cars={self.active_car_ids}, stop_after_one_lap={self.stop_after_one_lap}"
        )
        if self.enable_logging:
            self.get_logger().info(f"Logging enabled -> {(self.repo_root / self.log_dir).resolve()}")

    def _parse_active_car_ids(self, raw_value: object) -> tuple[int, ...]:
        car_ids: list[int] = []
        if isinstance(raw_value, int):
            tokens = [raw_value]
        elif isinstance(raw_value, str):
            tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
        elif isinstance(raw_value, (list, tuple)):
            tokens = list(raw_value)
        else:
            raise ValueError(f"Unsupported active_car_ids value: {raw_value!r}")

        for token in tokens:
            car_id = int(token)
            if car_id not in (1, 2):
                raise ValueError(f"Unsupported car_id '{car_id}' in active_car_ids")
            if car_id not in car_ids:
                car_ids.append(car_id)
        if not car_ids:
            raise ValueError("active_car_ids must contain at least one car id")
        return tuple(car_ids)

    def ips_cb(self, car_id: int, msg: Point) -> None:
        state = self.car_states[car_id]
        now = self.get_clock().now().nanoseconds * 1e-9
        current_xy = (float(msg.x), float(msg.y))

        if state.pos is None:
            state.pos = current_xy
            state.prev_pos = state.pos
            state.prev_time = now
            state.start_pose_xy = current_xy

            dists = np.linalg.norm(self.path - np.array(state.pos), axis=1)
            state.idx = int(np.argmin(dists))
            state.idx_initialized = True
            self.get_logger().info(f"Initialized car {car_id} waypoint index to {state.idx} from IPS start")
            return

        if state.prev_time is not None:
            dt = now - state.prev_time
            if dt > 1e-6:
                dx = float(msg.x) - state.pos[0]
                dy = float(msg.y) - state.pos[1]
                state.vx = dx / dt
                state.vy = dy / dt
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    state.measured_yaw = math.atan2(dy, dx)
            state.prev_pos = state.pos

        state.pos = current_xy
        state.prev_time = now

        if state.start_pose_xy is None:
            state.start_pose_xy = current_xy
            return

        if state.prev_pos is not None:
            state.total_distance_m += math.hypot(current_xy[0] - state.prev_pos[0], current_xy[1] - state.prev_pos[1])

        distance_to_start = math.hypot(
            current_xy[0] - state.start_pose_xy[0],
            current_xy[1] - state.start_pose_xy[1],
        )
        if distance_to_start > self.lap_start_radius_m:
            state.left_start_zone = True

        if (
            self.stop_after_one_lap
            and not state.completed_lap
            and state.left_start_zone
            and state.total_distance_m >= self.lap_min_distance_m
            and distance_to_start <= self.lap_start_radius_m
        ):
            state.completed_lap = True
            self.get_logger().info(
                f"Car {car_id} completed one lap at distance {state.total_distance_m:.2f} m. "
                f"Stopping pure pursuit commands for this car."
            )

    def imu_cb(self, car_id: int, msg: Imu) -> None:
        self.car_states[car_id].yaw_rate = float(msg.angular_velocity.z)

    def steer_fb_cb(self, car_id: int, msg: Float32) -> None:
        self.car_states[car_id].steering_fb = float(msg.data)

    def throttle_fb_cb(self, car_id: int, msg: Float32) -> None:
        self.car_states[car_id].throttle_fb = float(msg.data)

    def follow_active_cb(self, msg: Bool) -> None:
        self.follow_active_car1 = bool(msg.data)

    def safety_active_cb(self, msg: Bool) -> None:
        self.safety_active_car1 = bool(msg.data)

    def _measured_state(self, state: CarRuntimeState) -> np.ndarray | None:
        if state.pos is None or state.prev_pos is None:
            return None

        dx = state.pos[0] - state.prev_pos[0]
        dy = state.pos[1] - state.prev_pos[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            tx, ty = self.path[state.idx]
            yaw = math.atan2(ty - state.pos[1], tx - state.pos[0])
        else:
            yaw = math.atan2(dy, dx)

        state.measured_yaw = yaw
        return np.array([state.pos[0], state.pos[1], yaw, state.vx, state.vy, state.yaw_rate], dtype=float)

    def _controller_state(self, state: CarRuntimeState, measured_state: np.ndarray) -> np.ndarray:
        if state.model_state is None:
            return measured_state

        if self.use_predicted_pose:
            return state.model_state.copy()

        controller_state = measured_state.copy()
        controller_state[3:6] = state.model_state[3:6]
        return controller_state

    def control_loop(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        for car_id, state in self.car_states.items():
            if state.completed_lap:
                self.steer_pubs[car_id].publish(Float32(data=0.0))
                self.throttle_pubs[car_id].publish(Float32(data=0.0))
                continue
            if car_id == 1 and self.safety_active_car1:
                continue
            if car_id == 1 and self.follow_active_car1:
                continue
            if state.pos is None or state.prev_pos is None or not state.idx_initialized:
                continue

            measured_state = self._measured_state(state)
            if measured_state is None:
                continue

            if state.model_predictor is not None and state.model_state is None:
                state.model_predictor.reset(measured_state, state.throttle_fb, state.steering_fb)
                state.model_state = measured_state.copy()

            controller_state = self._controller_state(state, measured_state)
            steering_cmd, new_idx = self.pursue(controller_state, self.path, state.idx)
            state.idx = new_idx
            steer_abs_deg = abs(math.degrees(steering_cmd))
            if steer_abs_deg > self.slow_steer_deg:
                throttle_cmd = float(self.target_speed * self.high_steer_speed_scale)
            else:
                throttle_cmd = float(self.target_speed)

            self.steer_pubs[car_id].publish(Float32(data=steering_cmd))
            self.throttle_pubs[car_id].publish(Float32(data=throttle_cmd))

            predicted_before_update = state.model_state.copy() if state.model_state is not None else measured_state.copy()
            if state.model_predictor is not None:
                state.model_state = state.model_predictor.predict_next(
                    controller_state,
                    state.throttle_fb,
                    state.steering_fb,
                    throttle_cmd,
                    steering_cmd,
                    self.control_period,
                )

            if state.run_logger is not None:
                state.run_logger.append(
                    t=now,
                    control_state=controller_state,
                    measured_state=measured_state,
                    predicted_state=predicted_before_update,
                    throttle_cmd=throttle_cmd,
                    steering_cmd=steering_cmd,
                )

            if int(now * 2) % 2 == 0:
                self.get_logger().info(
                    f"car={car_id} pos=({measured_state[0]:.2f},{measured_state[1]:.2f}) "
                    f"cmd=({throttle_cmd:.2f},{steering_cmd:.3f}) idx={state.idx}"
                )

    def pursue(self, state: np.ndarray, path: np.ndarray, idx: int) -> tuple[float, int]:
        x, y, yaw = float(state[0]), float(state[1]), float(state[2])
        n_points = len(path)

        while True:
            tx, ty = path[idx % n_points]
            if math.hypot(tx - x, ty - y) > self.lookahead_distance:
                break
            idx += 1
            if idx >= n_points:
                idx = 0

        dx = tx - x
        dy = ty - y
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        xv = dx * cos_yaw - dy * sin_yaw
        yv = dx * sin_yaw + dy * cos_yaw

        alpha = math.atan2(yv, xv)
        delta = math.atan2(4.0 * self.wheelbase * math.sin(alpha), self.lookahead_distance)
        delta = max(-self.max_steer, min(self.max_steer, delta))
        return delta, idx

    def destroy_node(self) -> bool:
        for car_id, state in self.car_states.items():
            if state.run_logger is not None:
                save_path = state.run_logger.save(self.path)
                if save_path is not None:
                    self.get_logger().info(f"Saved car {car_id} run log to {save_path}")
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PurePursuitF1Tenth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

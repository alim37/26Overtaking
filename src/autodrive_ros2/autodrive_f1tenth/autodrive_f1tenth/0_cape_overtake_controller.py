#!/usr/bin/env python3
"""One-lap CAPE controller and model comparison node."""

import csv
import json
import math
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import rclpy
from std_msgs.msg import Bool, Float32, String

from autodrive_f1tenth.dynssm_nmpc import LePAVDParameters, NominalLePAVDNMPC
from autodrive_f1tenth.dynssm_nmpc_controller import DynSSMNMPCController


class CAPEOvertakeController(DynSSMNMPCController):
    """Pure pursuit/NMPC/adaptive/robust control with EPM or bicycle state propagation."""

    COMPLETE = "COMPLETED_LAP"

    def __init__(self):
        super().__init__()
        defaults = {
            "controller": "nmpc",
            "model": "epm",
            "overtake_start_lap": 0,
            "adaptive_update_interval": 5,
            "adaptive_parameter_trust_fraction": 0.10,
            "robust_nominal_margin_m": 0.03,
            "robust_dynamic_error_to_margin": 0.12,
            "robust_error_warmup_samples": 30,
            "robust_error_history_samples": 200,
            "robust_max_margin_m": 0.30,
            "robust_track_half_width_m": 0.50,
            "robust_track_slack_weight": 1.0e6,
            "state_correction_gain": 0.80,
            "minimum_vehicle_separation_m": 0.65,
            "pure_pursuit_speed_kp": 0.025,
            "pure_pursuit_steering_gain": 4.5,
            "pure_pursuit_max_steering_rad": 0.50,
            "experiment_overtake_speed_multiplier": -1.0,
            "bicycle_parameter_file": "",
            "benchmark_output_dir": "",
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        get = lambda name: self.get_parameter(name).value
        self.controller = str(get("controller")).lower()
        self.model_name = str(get("model")).lower()
        if self.controller not in {"pure_pursuit", "nmpc", "adaptive", "robust"}:
            raise ValueError("controller must be pure_pursuit, nmpc, adaptive, or robust")
        if self.model_name not in {"epm", "bicycle"}:
            raise ValueError("model must be epm or bicycle")
        if self.controller == "adaptive" and self.model_name != "epm":
            raise ValueError(
                "adaptive control requires model:=epm; no validated online "
                "bicycle-parameter estimator is implemented"
            )
        if self.speed_multiplier <= 0.0:
            raise ValueError("engagement_speed_multiplier must be positive")
        if self.model_name == "bicycle":
            # The fixed bicycle baseline must not consume learned EPM parameters.
            self.dynssm = None
            self.dynssm_active = True
            self.bicycle_parameters = self._load_bicycle_parameters(
                str(get("bicycle_parameter_file"))
            )
        else:
            self.bicycle_parameters = None

        self.overtake_start_lap = int(get("overtake_start_lap"))
        self.adaptive_interval = max(1, int(get("adaptive_update_interval")))
        self.adaptive_parameter_trust_fraction = max(
            0.0, float(get("adaptive_parameter_trust_fraction"))
        )
        self.robust_m0 = float(get("robust_nominal_margin_m"))
        self.robust_dynamic_error_to_margin = float(
            get("robust_dynamic_error_to_margin")
        )
        self.robust_error_warmup = int(get("robust_error_warmup_samples"))
        self.robust_max_margin = float(get("robust_max_margin_m"))
        self.robust_track_half_width = float(get("robust_track_half_width_m"))
        self.robust_track_slack_weight = float(get("robust_track_slack_weight"))
        self.correction_gain = float(get("state_correction_gain"))
        self.minimum_vehicle_separation = float(
            get("minimum_vehicle_separation_m")
        )
        self.pure_pursuit_speed_kp = float(get("pure_pursuit_speed_kp"))
        self.pure_pursuit_steering_gain = float(
            get("pure_pursuit_steering_gain")
        )
        self.pure_pursuit_max_steering = float(
            get("pure_pursuit_max_steering_rad")
        )
        configured_experiment_speed = float(
            get("experiment_overtake_speed_multiplier")
        )
        self.experiment_speed_multiplier = (
            configured_experiment_speed
            if configured_experiment_speed > 0.0 else self.speed_multiplier
        )
        self.robust_margin = self.robust_m0
        self.model_mismatch = 0.0
        self.adaptation_step = 0
        self.adaptive_update_count = 0
        self.adaptive_anchor_parameters = None
        self.robust_dynamic_error = 0.0
        self.robust_error_history = deque(
            maxlen=int(get("robust_error_history_samples"))
        )
        self.estimated_position = None
        self.estimated_heading = None
        self.estimated_state = None
        self.estimator_time = None
        self.command_throttle = 0.0
        self.command_steering = 0.0
        self.objective = math.nan
        self.solve_ms = math.nan
        self.collision_slack_m2 = 0.0
        self.track_slack_m = 0.0
        self.predicted_min_separation_m = math.nan
        self.control_dt = math.nan
        self.last_control_time = None
        self.epm_warmup_applied = False
        self.peer_position = None
        self.peer_velocity = np.zeros(2, dtype=float)
        self.peer_sample_time = None
        self.peer_normal_index = None
        self.target_reference_speed = self.nominal_velocity
        self.experiment_start_time = None
        self.completion_time = math.nan
        self.external_completion_requested = False
        self.bicycle_warmup_complete = self.model_name != "bicycle"
        self._rebuild_nmpc(self.nmpc.model.p)

        output_dir = str(get("benchmark_output_dir")).strip()
        root = Path(output_dir) if output_dir else Path(__file__).resolve().parents[1] / "output/cape_overtake"
        root.mkdir(parents=True, exist_ok=True)
        speed_label = f"{self.experiment_speed_multiplier:.2f}"
        suffix = "" if self.enable_overtake_path else "_target"
        self.output_csv = root / (
            f"{self.controller}_overtake_{self.model_name}_{speed_label}{suffix}.csv"
        )
        self.csv_file = self.output_csv.open("w", newline="", encoding="utf-8")
        fields = [
            "time_sec", "elapsed_time_s", "completion_time_s", "completed",
            "car_id", "controller", "model", "overtake_speed_multiplier",
            "lap", "mode", "x_m", "y_m",
            "estimated_x_m", "estimated_y_m", "heading_rad", "speed_mps",
            "raw_speed_mps",
            "lateral_speed_mps", "yaw_rate_rad_s", "throttle_command",
            "steering_command_rad", "throttle_feedback", "steering_feedback_rad",
            "objective", "solve_time_ms", "model_mismatch_m", "robust_margin_m",
            "peer_x_m", "peer_y_m", "vehicle_separation_m", "cross_track_error_m",
            "peer_velocity_x_mps", "peer_velocity_y_mps", "peer_speed_mps",
            "reference_x_m", "reference_y_m", "reference_heading_rad",
            "predicted_x_m", "predicted_y_m", "predicted_heading_rad",
            "predicted_speed_mps", "predicted_lateral_speed_mps",
            "predicted_yaw_rate_rad_s", "solver_success",
            "predicted_min_separation_m", "collision_slack_m2", "track_slack_m",
            "adaptive_update_count", "robust_dynamic_error", "solver_status",
            "epm_Bf", "epm_Df", "epm_Br", "epm_Dr", "epm_Cm1", "epm_Cm2",
            "epm_Cr0", "epm_Cr2", "epm_Iz",
            "raw_epm_Bf", "raw_epm_Df", "raw_epm_Br", "raw_epm_Dr",
            "raw_epm_Cm1", "raw_epm_Cm2", "raw_epm_Cr0", "raw_epm_Cr2",
            "raw_epm_Iz",
            "epm_drive_gain", "target_reference_speed_mps",
            "control_dt_sec", "odom_age_sec",
        ]
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fields)
        self.writer.writeheader()
        self.csv_file.flush()
        self.variant_pub = self.create_publisher(
            String, f"/autodrive/f1tenth_{self.car_id}/cape/variant", 10
        )
        self.completion_pub = self.create_publisher(
            Bool, "/autodrive/cape/experiment_complete", 10
        )
        if not self.enable_overtake_path:
            self.create_subscription(
                Bool,
                "/autodrive/cape/experiment_complete",
                self._completion_cb,
                10,
            )
        self.get_logger().info(
            f"CAPE ready: controller={self.controller}, model={self.model_name}, "
            f"overtake_lap={self.overtake_start_lap + 1}, output={self.output_csv}"
        )

    def _load_bicycle_parameters(self, configured_path):
        path_text = configured_path.strip()
        if not path_text:
            self.get_logger().warning(
                "No bicycle parameter file configured; using nominal fixed parameters"
            )
            return LePAVDParameters()
        path = Path(path_text).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Bicycle parameter file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model") != "fixed_kinematic_bicycle":
            raise ValueError(f"Unsupported bicycle parameter schema in {path}")
        geometry = payload["geometry"]
        actuator = payload["steering_actuator"]
        longitudinal = payload["longitudinal"]
        self.model_wheelbase = float(geometry["wheelbase_m"])
        self.steering_command_gain = float(actuator["command_gain"])
        self.steering_actuator_tau = float(actuator["time_constant_sec"])
        self.steering_command_delay_steps = int(
            actuator["command_delay_steps"]
        )
        parameters = replace(
            LePAVDParameters(),
            lr=float(geometry["rear_axle_to_cg_m"]),
            mass=float(geometry["mass_kg"]),
            Cm1=float(longitudinal["Cm1"]),
            Cm2=float(longitudinal["Cm2"]),
            Cr0=float(longitudinal["Cr0"]),
            Cr2=float(longitudinal["Cr2"]),
        )
        self.get_logger().info(
            "Loaded identified fixed bicycle model: "
            f"file={path}, wheelbase={self.model_wheelbase:.5f}m, "
            f"steering_gain={self.steering_command_gain:.5f}, "
            f"steering_tau={self.steering_actuator_tau:.5f}s, "
            f"delay={self.steering_command_delay_steps}, "
            f"Cm1={parameters.Cm1:.5f}"
        )
        return parameters

    def _rebuild_nmpc(self, parameters):
        # Keep the model comparison scientifically clean: EPM uses its full
        # lateral/longitudinal dynamics; the bicycle baseline uses kinematics.
        wheelbase = self.model_wheelbase if self.model_name == "bicycle" else None
        selected_parameters = (
            self.bicycle_parameters
            if self.model_name == "bicycle"
            and self.bicycle_parameters is not None
            else parameters
        )
        self.nmpc = NominalLePAVDNMPC(
            horizon=self.horizon,
            sample_time=self.control_period,
            pwm_bounds=(
                0.0,
                self.max_sim_throttle
                * max(1.0, self.speed_multiplier)
                / max(self.pwm_to_sim_scale, 1.0e-6),
            ),
            steering_bounds=(-self.max_steering, self.max_steering),
            max_steering_rate=self.max_steering_rate,
            max_throttle_rate=self.max_throttle_rate,
            throttle_rate_weight=self.throttle_rate_weight,
            steering_rate_weight=self.steering_rate_weight,
            steering_acceleration_weight=self.steering_acceleration_weight,
            steering_actuator_gap_weight=self.steering_actuator_gap_weight,
            heading_weight=self.nmpc_heading_weight,
            yaw_rate_weight=self.nmpc_yaw_rate_weight,
            throttle_actuator_tau=self.throttle_actuator_tau,
            steering_actuator_tau=self.steering_actuator_tau,
            steering_command_delay_steps=self.steering_command_delay_steps,
            steering_command_gain=self.steering_command_gain,
            collision_distance=self.minimum_vehicle_separation,
            collision_slack_weight=self.collision_slack_weight,
            enable_collision=self.enable_overtake_path,
            model_parameters=selected_parameters,
            kinematic_wheelbase=wheelbase,
            track_half_width=(
                self.robust_track_half_width
                if self.controller == "robust"
                else None
            ),
            track_slack_weight=self.robust_track_slack_weight,
        )

    def _update_mode(self, now):
        if not self.enable_overtake_path:
            self.mode = self.NORMAL
            return
        if self.mode == self.NORMAL and not self.engagement_complete:
            if (
                self.lap_count >= self.overtake_start_lap
                and np.linalg.norm(self.position - self.c_start) <= self.c_start_radius
            ):
                self.mode = self.ENGAGEMENT
                self.engagement_start_time = now
                self.engagement_index = int(np.argmin(
                    np.linalg.norm(self.engagement_path.points - self.position, axis=1)
                ))
                self.get_logger().info(
                    "Reached c_start on lap 1; beginning overtake"
                )
        elif self.mode == self.ENGAGEMENT:
            if (
                self.engagement_index >= len(self.engagement_path.points) - 1
                and np.linalg.norm(
                    self.position - self.engagement_path.points[-1]
                ) <= self.c_end_radius
            ):
                self._complete_experiment(now)

    def _complete_experiment(self, now):
        if self.engagement_complete:
            return
        self.mode = self.COMPLETE
        self.engagement_complete = True
        self.completion_time = (
            now - self.experiment_start_time
            if self.experiment_start_time is not None else math.nan
        )
        message = (
            "COMPLETED LAP at final overtake-reference sample"
            if self.enable_overtake_path
            else "Ego completed lap; stopping target"
        )
        self.get_logger().info(
            f"{message}; completion_time={self.completion_time:.3f}s"
        )
        if self.enable_overtake_path:
            self.completion_pub.publish(Bool(data=True))

    def _completion_cb(self, msg):
        if msg.data:
            self.external_completion_requested = True

    def _estimate_pose(self, now):
        measured = np.asarray(self.position, dtype=float)
        measured_state = np.array([
            measured[0], measured[1], self.heading, self.speed,
            self.lateral_speed, self.yaw_rate,
        ])
        if self.estimated_state is None:
            self.estimated_state = measured_state.copy()
            self.estimator_time = now
        else:
            dt = float(np.clip(now - self.estimator_time, 0.0, 0.25))
            self.estimator_time = now
            model_control = np.array([
                self.throttle_feedback,
                self.steering_feedback,
            ])
            derivative = np.asarray(
                self.nmpc._dynamics_function(
                    self.estimated_state,
                    model_control,
                    self.nmpc.model.parameter_vector(
                        self.current_model_parameters
                    ),
                )
            ).reshape(6)
            self.estimated_state += dt * derivative
            gain = self.correction_gain
            self.estimated_state[:2] += gain * (
                measured_state[:2] - self.estimated_state[:2]
            )
            self.estimated_state[3:] += gain * (
                measured_state[3:] - self.estimated_state[3:]
            )
        self.estimated_position = self.estimated_state[:2]
        self.estimated_heading = float(self.estimated_state[2])
        error = math.atan2(
            math.sin(self.heading - self.estimated_heading),
            math.cos(self.heading - self.estimated_heading),
        )
        self.estimated_state[2] += self.correction_gain * error
        self.estimated_heading = float(self.estimated_state[2])

    def _pure_pursuit(self, path, index, engagement, desired_speed):
        targets, _ = path.sample(float(path.arc[index]), self.startup_pp_lookahead, 2)
        delta = targets[1] - self.estimated_position
        c, s = math.cos(-self.estimated_heading), math.sin(-self.estimated_heading)
        local_x = delta[0] * c - delta[1] * s
        local_y = delta[0] * s + delta[1] * c
        alpha = math.atan2(local_y, local_x)
        steering = math.atan2(
            self.pure_pursuit_steering_gain
            * self.model_wheelbase
            * math.sin(alpha),
            self.startup_pp_lookahead,
        )
        steering = float(np.clip(
            steering,
            -self.pure_pursuit_max_steering,
            self.pure_pursuit_max_steering,
        ))
        phase_limit = self.max_sim_throttle * (
            self.speed_multiplier if engagement else 1.0
        )
        feedforward = self.max_sim_throttle * (
            desired_speed / max(self.nominal_velocity, 1.0e-6)
        )
        throttle = feedforward + self.pure_pursuit_speed_kp * (
            desired_speed - self.speed
        )
        throttle = float(np.clip(throttle, 0.0, phase_limit))
        if abs(steering) >= self.corner_steering:
            throttle *= self.corner_throttle_scale
        return float(throttle), steering

    def _update_adaptive_parameters(self):
        """Update EPM parameters inside a local, persistently excited region."""
        if self.speed < self.dynssm_min_adaptation_speed:
            return
        self.adaptation_step += 1
        if self.adaptation_step % self.adaptive_interval:
            return
        if self.current_model_parameters is None:
            return
        if self.adaptive_anchor_parameters is None:
            self.adaptive_anchor_parameters = self.current_model_parameters.copy()
        previous_stamp = self.last_model_inference_stamp
        if self._update_dynssm_parameters():
            if self.last_model_inference_stamp != previous_stamp:
                for name, anchor in self.adaptive_anchor_parameters.items():
                    radius = self.adaptive_parameter_trust_fraction * max(
                        abs(anchor), 1.0e-6
                    )
                    self.current_model_parameters[name] = float(np.clip(
                        self.current_model_parameters[name],
                        anchor - radius,
                        anchor + radius,
                    ))
                self.adaptive_update_count += 1

    def _update_robust_margin(self):
        if len(self.robust_error_history) < self.robust_error_warmup:
            self.robust_margin = self.robust_m0
            return
        errors = np.asarray(self.robust_error_history, dtype=float)
        dynamic_bound = float(np.mean(errors) + 2.0 * np.std(errors))
        self.robust_margin = min(
            self.robust_max_margin,
            self.robust_m0
            + self.robust_dynamic_error_to_margin * dynamic_bound,
        )

    def peer_ips_cb(self, msg):
        self.peer_ips_ready = True
        now = self.get_clock().now().nanoseconds * 1.0e-9
        position = np.array([float(msg.x), float(msg.y)])
        if self.peer_position is not None and self.peer_sample_time is not None:
            dt = now - self.peer_sample_time
            if 0.01 <= dt <= 0.5:
                measured_velocity = (position - self.peer_position) / dt
                if np.linalg.norm(measured_velocity) <= 12.0:
                    self.peer_velocity = (
                        0.7 * self.peer_velocity + 0.3 * measured_velocity
                    )
        self.peer_position = position
        self.peer_sample_time = now
        if self.peer_normal_index is None:
            self.peer_normal_index = int(np.argmin(
                np.linalg.norm(self.normal_path.points - position, axis=1)
            ))
        else:
            offsets = np.arange(-4, 45, dtype=int)
            candidates = (self.peer_normal_index + offsets) % len(
                self.normal_path.points
            )
            distances = np.linalg.norm(
                self.normal_path.points[candidates] - position, axis=1
            )
            self.peer_normal_index = int(candidates[int(np.argmin(distances))])

    def _publish(self, throttle, steering):
        self.command_throttle = float(throttle)
        self.command_steering = float(steering)
        self.throttle_pub.publish(Float32(data=self.command_throttle))
        self.steer_pub.publish(Float32(data=self.command_steering))

    def _log(self, now):
        epm = self.current_model_parameters or {}
        raw_epm = self.raw_model_parameters or {}
        separation = (
            float(np.linalg.norm(self.position - self.peer_position))
            if self.peer_position is not None else math.nan
        )
        active_path = (
            self.engagement_path
            if self.enable_overtake_path
            and self.mode in {self.ENGAGEMENT, self.COMPLETE}
            else self.normal_path
        )
        path_distances = np.linalg.norm(active_path.points - self.position, axis=1)
        reference_index = int(np.argmin(path_distances))
        reference_point = active_path.points[reference_index]
        if active_path.closed:
            previous_point = active_path.points[
                (reference_index - 1) % len(active_path.points)
            ]
            next_point = active_path.points[
                (reference_index + 1) % len(active_path.points)
            ]
        else:
            previous_point = active_path.points[max(0, reference_index - 1)]
            next_point = active_path.points[
                min(len(active_path.points) - 1, reference_index + 1)
            ]
        reference_heading = math.atan2(
            next_point[1] - previous_point[1],
            next_point[0] - previous_point[0],
        )
        cross_track_error = float(path_distances[reference_index])
        elapsed = (
            now - self.experiment_start_time
            if self.experiment_start_time is not None else 0.0
        )
        predicted_state = (
            self.last_prediction[:, 1]
            if self.last_prediction is not None
            and self.last_prediction.ndim == 2
            and self.last_prediction.shape[1] > 1
            else np.full(6, math.nan)
        )
        solver_status = str(self.nmpc.last_status)
        solver_success = int(solver_status in {
            "Solve_Succeeded", "Solved_To_Acceptable_Level"
        })
        self.writer.writerow({
            "time_sec": f"{now:.6f}", "elapsed_time_s": elapsed,
            "completion_time_s": self.completion_time,
            "completed": int(self.engagement_complete),
            "car_id": self.car_id, "controller": self.controller,
            "model": self.model_name, "lap": self.lap_count + 1, "mode": self.mode,
            "overtake_speed_multiplier": self.experiment_speed_multiplier,
            "x_m": self.position[0], "y_m": self.position[1],
            "estimated_x_m": self.estimated_position[0],
            "estimated_y_m": self.estimated_position[1],
            "heading_rad": self.estimated_heading, "speed_mps": self.speed,
            "raw_speed_mps": self.raw_odom_speed,
            "lateral_speed_mps": self.lateral_speed, "yaw_rate_rad_s": self.yaw_rate,
            "throttle_command": self.command_throttle,
            "steering_command_rad": self.command_steering,
            "throttle_feedback": self.throttle_feedback,
            "steering_feedback_rad": self.steering_feedback,
            "objective": self.objective, "solve_time_ms": self.solve_ms,
            "model_mismatch_m": self.model_mismatch,
            "robust_margin_m": self.robust_margin,
            "peer_x_m": self.peer_position[0] if self.peer_position is not None else math.nan,
            "peer_y_m": self.peer_position[1] if self.peer_position is not None else math.nan,
            "vehicle_separation_m": separation,
            "cross_track_error_m": cross_track_error,
            "peer_velocity_x_mps": self.peer_velocity[0],
            "peer_velocity_y_mps": self.peer_velocity[1],
            "peer_speed_mps": float(np.linalg.norm(self.peer_velocity)),
            "reference_x_m": reference_point[0],
            "reference_y_m": reference_point[1],
            "reference_heading_rad": reference_heading,
            "predicted_x_m": predicted_state[0],
            "predicted_y_m": predicted_state[1],
            "predicted_heading_rad": predicted_state[2],
            "predicted_speed_mps": predicted_state[3],
            "predicted_lateral_speed_mps": predicted_state[4],
            "predicted_yaw_rate_rad_s": predicted_state[5],
            "solver_success": solver_success,
            "predicted_min_separation_m": self.predicted_min_separation_m,
            "collision_slack_m2": self.collision_slack_m2,
            "track_slack_m": self.track_slack_m,
            "adaptive_update_count": self.adaptive_update_count,
            "robust_dynamic_error": self.robust_dynamic_error,
            "solver_status": solver_status,
            "epm_Bf": epm.get("Bf", math.nan),
            "epm_Df": epm.get("Df", math.nan),
            "epm_Br": epm.get("Br", math.nan),
            "epm_Dr": epm.get("Dr", math.nan),
            "epm_Cm1": epm.get("Cm1", math.nan),
            "epm_Cm2": epm.get("Cm2", math.nan),
            "epm_Cr0": epm.get("Cr0", math.nan),
            "epm_Cr2": epm.get("Cr2", math.nan),
            "epm_Iz": epm.get("Iz", math.nan),
            "raw_epm_Bf": raw_epm.get("Bf", math.nan),
            "raw_epm_Df": raw_epm.get("Df", math.nan),
            "raw_epm_Br": raw_epm.get("Br", math.nan),
            "raw_epm_Dr": raw_epm.get("Dr", math.nan),
            "raw_epm_Cm1": raw_epm.get("Cm1", math.nan),
            "raw_epm_Cm2": raw_epm.get("Cm2", math.nan),
            "raw_epm_Cr0": raw_epm.get("Cr0", math.nan),
            "raw_epm_Cr2": raw_epm.get("Cr2", math.nan),
            "raw_epm_Iz": raw_epm.get("Iz", math.nan),
            "epm_drive_gain": (
                epm.get("Cm1", math.nan)
                - epm.get("Cm2", math.nan) * self.speed
            ),
            "target_reference_speed_mps": self.target_reference_speed,
            "control_dt_sec": self.control_dt,
            "odom_age_sec": (
                now - self.last_odom_stamp
                if self.last_odom_stamp is not None else math.nan
            ),
        })
        self.csv_file.flush()

    def control_loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_control_time is not None:
            self.control_dt = now - self.last_control_time
        self.last_control_time = now
        self._update_startup_gate(now)
        if not self.startup_gate_open or self.position is None or self.heading is None:
            self._publish_stop()
            return
        if self.experiment_start_time is None:
            self.experiment_start_time = now
        if self.external_completion_requested and not self.engagement_complete:
            self._complete_experiment(now)
        if self.stopped:
            self._publish_stop()
            return
        if self.mode == self.COMPLETE:
            self.previous_control[:] = 0.0
            self._publish(0.0, 0.0)
            self._estimate_pose(now)
            self._log(now)
            self.stopped = True
            return
        if not self.bicycle_warmup_complete:
            throttle, steering = self._startup_pure_pursuit_command()
            self.previous_control[:] = [
                throttle / max(self.pwm_to_sim_scale, 1.0e-6), steering
            ]
            self.startup_sample_count += 1
            self._publish(throttle, steering)
            self.mode_pub.publish(String(data=(
                "PURE_PURSUIT_WARMUP_"
                f"{self.startup_sample_count}/{self.startup_pp_samples}"
            )))
            if self.startup_sample_count >= self.startup_pp_samples:
                self.bicycle_warmup_complete = True
                self.get_logger().info(
                    f"Completed {self.startup_pp_samples}-sample "
                    "pure-pursuit warm-up; bicycle NMPC taking control"
                )
            return
        if self.dynssm is not None and not self.dynssm_active:
            self._run_startup_pure_pursuit()
            self.model_status_pub.publish(Bool(data=self.dynssm_active))
            return
        if self.dynssm_active and not self.epm_warmup_applied:
            self._rebuild_nmpc(self.nmpc.model.p)
            self.epm_warmup_applied = True
        if self.controller == "adaptive":
            self._update_adaptive_parameters()

        self._estimate_pose(now)
        self._update_mode(now)
        if self.mode == self.COMPLETE:
            self.previous_control[:] = 0.0
            self._publish(0.0, 0.0)
            self._log(now)
            self.stopped = True
            return
        engagement = self.mode == self.ENGAGEMENT
        if engagement:
            self.engagement_index = self._advance_path_index(
                self.engagement_path, self.engagement_index, False
            )
            if (
                self.engagement_index >= len(self.engagement_path.points) - 1
                and np.linalg.norm(
                    self.position - self.engagement_path.points[-1]
                ) <= self.c_end_radius
            ):
                self._complete_experiment(now)
                self.previous_control[:] = 0.0
                self._publish(0.0, 0.0)
                self._log(now)
                self.stopped = True
                return
            path, index = self.engagement_path, self.engagement_index
        else:
            self.normal_index = self._advance_path_index(
                self.normal_path, self.normal_index, True
            )
            path, index = self.normal_path, self.normal_index

        speed_up = (
            engagement and self.engagement_start_time is not None
            and now - self.engagement_start_time >= self.speed_delay
        )
        desired_speed = self.nominal_velocity * (
            self.speed_multiplier if speed_up else 1.0
        )
        self.target_reference_speed = desired_speed
        if self.controller == "pure_pursuit":
            throttle, steering = self._pure_pursuit(
                path, index, engagement, desired_speed
            )
            self.previous_control[:] = [throttle / self.pwm_to_sim_scale, steering]
            self._publish(throttle, steering)
            self.mode_pub.publish(String(data=f"CAPE_{self.controller}_{self.model_name}_{self.mode}"))
            self._log(now)
            return

        # This NMPC tracks velocity as well as position. Keep the spatial and
        # velocity references consistent so low speed cannot collapse the path
        # horizon and make acceleration appear to overshoot the reference.
        reference_spacing = desired_speed * self.control_period
        points, headings = path.sample(
            float(path.arc[index]), reference_spacing, self.horizon + 1
        )
        track_left = track_right = None
        if self.controller == "robust":
            track_left, track_right = path.sample_clearances(
                float(path.arc[index]), reference_spacing, self.horizon + 1
            )
        turn = round((self.estimated_heading - headings[0]) / (2.0 * math.pi)) * 2.0 * math.pi
        headings += turn
        reference = np.zeros((6, self.horizon + 1))
        reference[:2], reference[2], reference[3] = points.T, headings, desired_speed
        reference[5, :-1] = np.diff(headings) / self.control_period
        reference[5, -1] = reference[5, -2]
        if self.controller == "robust":
            self._update_robust_margin()
        state = np.array([
            *self.estimated_position, self.estimated_heading, self.speed,
            self.lateral_speed, self.yaw_rate,
        ])
        peer_reference = None
        if (
            self.enable_overtake_path
            and engagement
            and self.peer_position is not None
            and self.peer_normal_index is not None
        ):
            peer_arc = float(self.normal_path.arc[self.peer_normal_index])
            peer_speed = float(np.linalg.norm(self.peer_velocity))
            peer_points, _ = self.normal_path.sample(
                peer_arc,
                peer_speed * self.control_period,
                self.horizon + 1,
            )
            peer_reference = peer_points.T
        started = time.perf_counter()
        try:
            phase_throttle_limit = self.max_sim_throttle * (
                self.speed_multiplier if speed_up else 1.0
            )
            control, prediction, self.objective = self.nmpc.solve(
                state, reference, self.previous_control,
                actuator_state=[self.throttle_feedback, self.steering_feedback],
                model_parameters=self.current_model_parameters,
                peer_reference=peer_reference,
                track_margin=(
                    self.robust_margin if self.controller == "robust" else 0.0
                ),
                track_left_clearance=(
                    track_left[1:] if track_left is not None else None
                ),
                track_right_clearance=(
                    track_right[1:] if track_right is not None else None
                ),
                pwm_upper_bound=(
                    phase_throttle_limit
                    / max(self.pwm_to_sim_scale, 1.0e-6)
                ),
            )
            self.solve_ms = (time.perf_counter() - started) * 1000.0
            if self.last_prediction is not None:
                self.model_mismatch = float(
                    np.linalg.norm(state[:2] - self.last_prediction[:2, 1])
                )
                self.robust_dynamic_error = float(np.linalg.norm(
                    state[3:6] - self.last_prediction[3:6, 1]
                ))
                if self.controller == "robust":
                    self.robust_error_history.append(self.robust_dynamic_error)
            self.last_prediction = prediction
            self.collision_slack_m2 = float(
                np.max(self.nmpc.last_collision_slack)
            )
            self.track_slack_m = float(np.max(self.nmpc.last_track_slack))
            self.predicted_min_separation_m = (
                float(np.min(np.linalg.norm(prediction[:2] - peer_reference, axis=0)))
                if peer_reference is not None else math.nan
            )
        except Exception as exc:
            self.solve_ms = (time.perf_counter() - started) * 1000.0
            if not rclpy.ok():
                return
            self.get_logger().warning(
                f"CAPE solve missed; holding previous feasible NMPC command: {exc}"
            )
            self._publish(
                self.previous_control[0] * self.pwm_to_sim_scale,
                self.previous_control[1],
            )
            self.mode_pub.publish(String(data=f"CAPE_NMPC_HOLD_{self.mode}"))
            self._log(now)
            return
        published_throttle = float(control[0] * self.pwm_to_sim_scale)
        published_steering = float(control[1])
        self.previous_control[:] = [
            published_throttle / max(self.pwm_to_sim_scale, 1.0e-6),
            published_steering,
        ]
        self._publish(published_throttle, published_steering)
        self.mode_pub.publish(String(data=f"CAPE_{self.controller}_{self.model_name}_{self.mode}"))
        self.variant_pub.publish(String(data=f"{self.controller}:{self.model_name}"))
        self._log(now)

    def destroy_node(self):
        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CAPEOvertakeController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

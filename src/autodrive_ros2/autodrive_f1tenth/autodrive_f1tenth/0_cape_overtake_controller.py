#!/usr/bin/env python3
"""Two-lap CAPE controller and model comparison node."""

import csv
import math
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from std_msgs.msg import Bool, Float32, String

from autodrive_f1tenth.dynssm_nmpc import LePAVDParameters, NominalLePAVDNMPC
from autodrive_f1tenth.dynssm_nmpc_controller import DynSSMNMPCController


class CAPEOvertakeController(DynSSMNMPCController):
    """Pure pursuit/NMPC/adaptive/robust control with EPM or bicycle state propagation."""

    def __init__(self):
        super().__init__()
        defaults = {
            "controller": "nmpc",
            "model": "epm",
            "overtake_start_lap": 1,
            "adaptive_update_interval": 5,
            "adaptive_learning_rate": 0.08,
            "robust_nominal_margin_m": 0.05,
            "robust_mismatch_gain": 0.05,
            "robust_max_margin_m": 0.15,
            "robust_track_half_width_m": 0.50,
            "state_correction_gain": 0.80,
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
        if self.model_name == "bicycle":
            # The fixed bicycle baseline must not consume learned EPM parameters.
            self.dynssm = None
            self.dynssm_active = True

        self.overtake_start_lap = int(get("overtake_start_lap"))
        self.adaptive_interval = max(1, int(get("adaptive_update_interval")))
        self.adaptive_lr = float(get("adaptive_learning_rate"))
        self.robust_m0 = float(get("robust_nominal_margin_m"))
        self.robust_gamma = float(get("robust_mismatch_gain"))
        self.robust_max_margin = float(get("robust_max_margin_m"))
        self.robust_track_half_width = float(get("robust_track_half_width_m"))
        self.correction_gain = float(get("state_correction_gain"))
        self.robust_margin = self.robust_m0
        self.model_mismatch = 0.0
        self.adaptation_step = 0
        self.previous_adaptation = None
        self.estimated_position = None
        self.estimated_heading = None
        self.estimated_state = None
        self.estimator_time = None
        self.command_throttle = 0.0
        self.command_steering = 0.0
        self.objective = math.nan
        self.solve_ms = math.nan
        self.epm_warmup_applied = False
        self.peer_position = None
        self._rebuild_nmpc(self.nmpc.model.p)

        output_dir = str(get("benchmark_output_dir")).strip()
        root = Path(output_dir) if output_dir else Path(__file__).resolve().parents[1] / "output/cape_overtake"
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_csv = root / (
            f"0_cape_{self.controller}_{self.model_name}_car{self.car_id}_{timestamp}.csv"
        )
        self.csv_file = self.output_csv.open("w", newline="", encoding="utf-8")
        fields = [
            "time_sec", "controller", "model", "lap", "mode", "x_m", "y_m",
            "estimated_x_m", "estimated_y_m", "heading_rad", "speed_mps",
            "lateral_speed_mps", "yaw_rate_rad_s", "throttle_command",
            "steering_command_rad", "throttle_feedback", "steering_feedback_rad",
            "objective", "solve_time_ms", "model_mismatch_m", "robust_margin_m",
            "peer_x_m", "peer_y_m", "vehicle_separation_m", "cross_track_error_m",
        ]
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fields)
        self.writer.writeheader()
        self.csv_file.flush()
        self.variant_pub = self.create_publisher(
            String, f"/autodrive/f1tenth_{self.car_id}/cape/variant", 10
        )
        self.get_logger().info(
            f"CAPE ready: controller={self.controller}, model={self.model_name}, "
            f"overtake_lap={self.overtake_start_lap + 1}, output={self.output_csv}"
        )

    def _rebuild_nmpc(self, parameters):
        # Keep the model comparison scientifically clean: EPM uses its full
        # lateral/longitudinal dynamics; the bicycle baseline uses kinematics.
        wheelbase = self.model_wheelbase if self.model_name == "bicycle" else None
        self.nmpc = NominalLePAVDNMPC(
            horizon=self.horizon,
            sample_time=self.control_period,
            pwm_bounds=(0.0, 1.0),
            steering_bounds=(-self.max_steering, self.max_steering),
            max_steering_rate=self.max_steering_rate,
            model_parameters=parameters,
            kinematic_wheelbase=wheelbase,
            track_half_width=(
                self.robust_track_half_width
                if self.controller == "robust"
                else None
            ),
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
                self.get_logger().info("Reached c_start; beginning lap-2 overtake")
        elif self.mode == self.ENGAGEMENT:
            remaining = self.engagement_path.length - float(
                self.engagement_path.arc[self.engagement_index]
            )
            if (
                remaining <= self.c_end_handoff_distance
                and np.linalg.norm(self.position - self.c_end) <= self.c_end_radius
            ):
                self.mode = self.COMPLETE
                self.engagement_complete = True
                self.get_logger().info("Reached c_end; returning to nominal raceline")

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
                self.command_throttle / max(self.pwm_to_sim_scale, 1e-6),
                self.command_steering,
            ])
            derivative = np.asarray(
                self.nmpc._dynamics_function(self.estimated_state, model_control)
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

    def _pure_pursuit(self, path, index, engagement):
        targets, _ = path.sample(float(path.arc[index]), self.startup_pp_lookahead, 2)
        delta = targets[1] - self.estimated_position
        c, s = math.cos(-self.estimated_heading), math.sin(-self.estimated_heading)
        local_x = delta[0] * c - delta[1] * s
        local_y = delta[0] * s + delta[1] * c
        alpha = math.atan2(local_y, local_x)
        steering = math.atan2(
            2.0 * self.model_wheelbase * math.sin(alpha), self.startup_pp_lookahead
        )
        steering = float(np.clip(steering, -self.max_steering, self.max_steering))
        throttle = self.max_sim_throttle * (self.speed_multiplier if engagement else 1.0)
        if abs(steering) >= self.corner_steering:
            throttle *= self.corner_throttle_scale
        return float(throttle), steering

    def _adapt(self, state, control):
        current = (float(state[3]), np.asarray(control, dtype=float).copy())
        if self.previous_adaptation is None:
            self.previous_adaptation = current
            return
        previous = self.previous_adaptation
        self.previous_adaptation = current
        self.adaptation_step += 1
        if self.adaptation_step % self.adaptive_interval:
            return
        old_speed, old_control = previous
        p = self.nmpc.model.p
        pwm = float(old_control[0])
        predicted = old_speed + self.control_period * (
            (p.Cm1 - p.Cm2 * old_speed) * pwm - p.Cr0 - p.Cr2 * old_speed**2
        ) / p.mass
        residual = float(np.clip(state[3] - predicted, -1.0, 1.0))
        gradient = self.control_period * np.array([pwm, -1.0, -old_speed**2]) / p.mass
        update = self.adaptive_lr * residual * gradient / max(np.dot(gradient, gradient), 1.0)
        nominal = LePAVDParameters()
        values = np.array([p.Cm1, p.Cr0, p.Cr2]) + update
        base = np.array([nominal.Cm1, nominal.Cr0, nominal.Cr2])
        values = np.clip(values, 0.5 * base, 1.5 * base)
        self._rebuild_nmpc(replace(p, Cm1=values[0], Cr0=values[1], Cr2=values[2]))

    def peer_ips_cb(self, msg):
        self.peer_ips_ready = True
        self.peer_position = np.array([float(msg.x), float(msg.y)])

    def _publish(self, throttle, steering):
        self.command_throttle = float(throttle)
        self.command_steering = float(steering)
        self.throttle_pub.publish(Float32(data=self.command_throttle))
        self.steer_pub.publish(Float32(data=self.command_steering))

    def _log(self, now):
        separation = (
            float(np.linalg.norm(self.position - self.peer_position))
            if self.peer_position is not None else math.nan
        )
        active_path = self.engagement_path if self.mode == self.ENGAGEMENT else self.normal_path
        cross_track_error = float(
            np.min(np.linalg.norm(active_path.points - self.position, axis=1))
        )
        self.writer.writerow({
            "time_sec": f"{now:.6f}", "controller": self.controller,
            "model": self.model_name, "lap": self.lap_count + 1, "mode": self.mode,
            "x_m": self.position[0], "y_m": self.position[1],
            "estimated_x_m": self.estimated_position[0],
            "estimated_y_m": self.estimated_position[1],
            "heading_rad": self.estimated_heading, "speed_mps": self.speed,
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
        })
        self.csv_file.flush()

    def control_loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        self._update_startup_gate(now)
        if not self.startup_gate_open or self.position is None or self.heading is None:
            self._publish_stop()
            return
        if self.stopped:
            self._publish_stop()
            return
        if self.dynssm is not None and not self.dynssm_active:
            self._run_startup_pure_pursuit()
            self.model_status_pub.publish(Bool(data=self.dynssm_active))
            return
        if self.dynssm_active and not self.epm_warmup_applied:
            self._rebuild_nmpc(self.nmpc.model.p)
            self.epm_warmup_applied = True

        self._estimate_pose(now)
        self._update_mode(now)
        engagement = self.mode == self.ENGAGEMENT
        if engagement:
            self.engagement_index = self._advance_path_index(
                self.engagement_path, self.engagement_index, False
            )
            path, index = self.engagement_path, self.engagement_index
        else:
            self.normal_index = self._advance_path_index(
                self.normal_path, self.normal_index, True
            )
            path, index = self.normal_path, self.normal_index

        if self.controller == "pure_pursuit" or self.mode == self.COMPLETE:
            throttle, steering = self._pure_pursuit(path, index, engagement)
            self.previous_control[:] = [throttle / self.pwm_to_sim_scale, steering]
            self._publish(throttle, steering)
            self.mode_pub.publish(String(data=f"CAPE_{self.controller}_{self.model_name}_{self.mode}"))
            self._log(now)
            return

        speed_up = (
            engagement and self.engagement_start_time is not None
            and now - self.engagement_start_time >= self.speed_delay
        )
        desired_speed = self.nominal_velocity * (self.speed_multiplier if speed_up else 1.0)
        points, headings = path.sample(
            float(path.arc[index]), desired_speed * self.control_period, self.horizon + 1
        )
        turn = round((self.estimated_heading - headings[0]) / (2.0 * math.pi)) * 2.0 * math.pi
        headings += turn
        reference = np.zeros((6, self.horizon + 1))
        reference[:2], reference[2], reference[3] = points.T, headings, desired_speed
        reference[5, :-1] = np.diff(headings) / self.control_period
        reference[5, -1] = reference[5, -2]
        if self.controller == "robust":
            self.robust_margin = min(
                self.robust_max_margin,
                self.robust_m0 + self.robust_gamma * self.model_mismatch,
            )
        state = np.array([
            *self.estimated_position, self.estimated_heading, self.speed,
            self.lateral_speed, self.yaw_rate,
        ])
        started = time.perf_counter()
        try:
            control, prediction, self.objective = self.nmpc.solve(
                state, reference, self.previous_control,
                track_margin=(self.robust_margin if self.controller == "robust" else 0.0),
            )
            self.solve_ms = (time.perf_counter() - started) * 1000.0
            if self.last_prediction is not None:
                self.model_mismatch = float(
                    np.linalg.norm(state[:2] - self.last_prediction[:2, 1])
                )
            self.last_prediction = prediction
        except Exception as exc:
            self.solve_ms = (time.perf_counter() - started) * 1000.0
            if not rclpy.ok():
                return
            self.get_logger().warning(
                f"CAPE solve missed; using forward pure-pursuit fallback: {exc}"
            )
            throttle, steering = self._pure_pursuit(path, index, engagement)
            self.previous_control[:] = [
                throttle / max(self.pwm_to_sim_scale, 1e-6), steering
            ]
            self._publish(throttle, steering)
            self.mode_pub.publish(String(data=f"CAPE_FALLBACK_{self.mode}"))
            self._log(now)
            return
        if self.controller == "adaptive":
            self._adapt(state, control)
        self.previous_control = np.asarray(control)
        limit = self.max_sim_throttle * (self.speed_multiplier if speed_up else 1.0)
        if abs(control[1]) >= self.corner_steering:
            limit *= self.corner_throttle_scale
        self._publish(
            float(np.clip(control[0] * self.pwm_to_sim_scale, -0.1, limit)),
            float(control[1]),
        )
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

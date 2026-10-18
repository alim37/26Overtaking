#!/usr/bin/env python3
"""Two-lap CAPE controller and model comparison node."""

import csv
import math
import time
from collections import deque
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
            "robust_nominal_margin_m": 0.03,
            "robust_dynamic_error_to_margin": 0.12,
            "robust_error_warmup_samples": 30,
            "robust_error_history_samples": 200,
            "robust_max_margin_m": 0.30,
            "robust_track_half_width_m": 0.50,
            "robust_track_slack_weight": 1.0e6,
            "state_correction_gain": 0.80,
            "minimum_vehicle_separation_m": 0.65,
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
        self.robust_margin = self.robust_m0
        self.model_mismatch = 0.0
        self.adaptation_step = 0
        self.adaptive_update_count = 0
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
            "raw_speed_mps",
            "lateral_speed_mps", "yaw_rate_rad_s", "throttle_command",
            "steering_command_rad", "throttle_feedback", "steering_feedback_rad",
            "objective", "solve_time_ms", "model_mismatch_m", "robust_margin_m",
            "peer_x_m", "peer_y_m", "vehicle_separation_m", "cross_track_error_m",
            "predicted_min_separation_m", "collision_slack_m2", "track_slack_m",
            "adaptive_update_count", "robust_dynamic_error", "solver_status",
            "epm_Bf", "epm_Df", "epm_Br", "epm_Dr", "epm_Cm1", "epm_Cm2",
            "epm_Cr0", "epm_Cr2", "epm_Iz",
            "raw_epm_Bf", "raw_epm_Df", "raw_epm_Br", "raw_epm_Dr",
            "raw_epm_Cm1", "raw_epm_Cm2", "raw_epm_Cr0", "raw_epm_Cr2",
            "raw_epm_Iz",
            "epm_drive_gain", "control_dt_sec", "odom_age_sec",
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
            model_parameters=parameters,
            kinematic_wheelbase=wheelbase,
            # DynSSM evaluates every NMPC variant against the same physical
            # track constraints. Robust NMPC differs only by extra tightening.
            track_half_width=self.robust_track_half_width,
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

    def _update_adaptive_parameters(self):
        """Refresh the complete EPM vector using the trained DynSSM predictor."""
        self.adaptation_step += 1
        if self.adaptation_step % self.adaptive_interval:
            return
        previous_stamp = self.last_model_inference_stamp
        if self._update_dynssm_parameters():
            if self.last_model_inference_stamp != previous_stamp:
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
            "predicted_min_separation_m": self.predicted_min_separation_m,
            "collision_slack_m2": self.collision_slack_m2,
            "track_slack_m": self.track_slack_m,
            "adaptive_update_count": self.adaptive_update_count,
            "robust_dynamic_error": self.robust_dynamic_error,
            "solver_status": self.nmpc.last_status,
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
        if self.controller == "adaptive":
            self._update_adaptive_parameters()

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
        # Match DynSSM's ConstantSpeed planner: spatial horizon progression is
        # based on measured speed; desired speed remains the velocity target.
        reference_spacing = max(self.speed, 0.0) * self.control_period
        points, headings = path.sample(
            float(path.arc[index]), reference_spacing, self.horizon + 1
        )
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
                    self.robust_margin
                    if self.controller == "robust"
                    else self.robust_m0
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

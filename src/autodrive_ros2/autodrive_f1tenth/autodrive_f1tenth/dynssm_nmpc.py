#!/usr/bin/env python3

"""Nominal lePAVD NMPC core adapted from DaRC-ARMs-lab/DynSSM.

State: [x, y, yaw, vx, vy, yaw_rate]. Control: [pwm, steering].
"""

from dataclasses import dataclass

import casadi as ca
import numpy as np


@dataclass(frozen=True)
class LePAVDParameters:
    lf: float = 0.15875
    lr: float = 0.17145
    mass: float = 3.47
    h_cg: float = 0.074
    gravity: float = 9.81
    Bf: float = 5.579
    Cf: float = 1.2
    Df: float = 16.2497561
    Ef: float = -0.083
    Br: float = 5.3852
    Cr: float = 1.2691
    Dr: float = 14.70095122
    Er: float = -0.019
    Cm1: float = 24.29
    Cm2: float = 4.612560976
    Cr0: float = 4.38404878
    Cr2: float = 0.02962195122
    Iz: float = 0.04712
    Shf: float = -0.0013
    Svf: float = 0.03639268293
    Shr: float = -0.00376
    Svr: float = 0.07701707317


class LePAVDDynamics:
    """CasADi form of the DynSSM nominal dynamic bicycle model."""

    PARAMETER_NAMES = (
        "Bf", "Cf", "Df", "Ef", "Br", "Cr", "Dr", "Er", "Cm1",
        "Cm2", "Cr0", "Cr2", "Iz", "Shf", "Svf", "Shr", "Svr",
    )

    def __init__(self, parameters=None, kinematic_wheelbase=None):
        self.p = parameters or LePAVDParameters()
        self.kinematic_wheelbase = kinematic_wheelbase

    def parameter_vector(self, parameters=None):
        if parameters is None:
            parameters = self.p
        if isinstance(parameters, dict):
            return np.asarray(
                [parameters[name] for name in self.PARAMETER_NAMES], dtype=float
            )
        if isinstance(parameters, LePAVDParameters):
            return np.asarray(
                [getattr(parameters, name) for name in self.PARAMETER_NAMES], dtype=float
            )
        return np.asarray(parameters, dtype=float).reshape(len(self.PARAMETER_NAMES))

    def derivative(self, state, control, model_parameters=None):
        p = self.p
        if model_parameters is None:
            model_parameters = self.parameter_vector()
        (
            Bf, Cf, Df, Ef, Br, Cr, Dr, Er, Cm1, Cm2, Cr0, Cr2, Iz,
            Shf, Svf, Shr, Svr,
        ) = [model_parameters[index] for index in range(len(self.PARAMETER_NAMES))]
        yaw, vx_raw, vy_raw, yaw_rate_raw = state[2], state[3], state[4], state[5]
        pwm, steering = control[0], control[1]
        vx = ca.fmin(ca.fmax(vx_raw, 0.05), 10.0)
        vy = ca.if_else(vx_raw < 0.05, 0.0, vy_raw)
        yaw_rate = ca.if_else(vx_raw < 0.05, 0.0, yaw_rate_raw)
        force_x = (Cm1 - Cm2 * vx) * pwm - Cr0 - Cr2 * vx * vx

        if self.kinematic_wheelbase is not None:
            desired_yaw_rate = vx * ca.tan(steering) / self.kinematic_wheelbase
            return ca.vertcat(
                vx * ca.cos(yaw),
                vx * ca.sin(yaw),
                yaw_rate,
                force_x / p.mass,
                -5.0 * vy,
                5.0 * (desired_yaw_rate - yaw_rate),
            )

        alpha_f = steering - ca.atan2(p.lf * yaw_rate + vy, vx) + Shf
        alpha_r = ca.atan2(p.lr * yaw_rate - vy, vx) + Shr
        bf_alpha = Bf * alpha_f
        br_alpha = Br * alpha_r
        theta_f = ca.atan(bf_alpha - Ef * (bf_alpha - ca.atan(bf_alpha)))
        theta_r = ca.atan(br_alpha - Er * (br_alpha - ca.atan(br_alpha)))
        force_f = Svf + Df * ca.sin(Cf * theta_f)
        force_r = Svr + Dr * ca.sin(Cr * theta_r)

        accel_x = force_x / p.mass
        axle_length = p.lf + p.lr
        load_f = ca.fmax(
            1e-3,
            (p.mass * p.gravity * p.lr - p.h_cg * p.mass * accel_x) / axle_length,
        )
        load_r = ca.fmax(
            1e-3,
            (p.mass * p.gravity * p.lf + p.h_cg * p.mass * accel_x) / axle_length,
        )
        nominal_load = p.mass * p.gravity / 4.0
        force_f *= load_f / nominal_load
        force_r *= load_r / nominal_load

        return ca.vertcat(
            vx * ca.cos(yaw) - vy * ca.sin(yaw),
            vx * ca.sin(yaw) + vy * ca.cos(yaw),
            yaw_rate,
            (force_x - force_f * ca.sin(steering)) / p.mass + vy * yaw_rate,
            (force_r + force_f * ca.cos(steering)) / p.mass - vx * yaw_rate,
            (force_f * p.lf * ca.cos(steering) - force_r * p.lr) / Iz,
        )


class NominalLePAVDNMPC:
    """Warm-started nonlinear MPC using the nominal lePAVD model."""

    VEHICLE_STATE_SIZE = 6
    STATE_SIZE = 6
    INPUT_SIZE = 2

    def __init__(
        self,
        horizon=12,
        sample_time=0.10,
        pwm_bounds=(-0.10, 1.0),
        steering_bounds=(-0.35, 0.35),
        max_steering_rate=2.5,
        max_throttle_rate=2.0,
        throttle_rate_weight=0.25,
        steering_rate_weight=8.0,
        steering_acceleration_weight=0.0,
        steering_actuator_gap_weight=0.0,
        heading_weight=2.5,
        yaw_rate_weight=0.25,
        model_parameters=None,
        kinematic_wheelbase=None,
        track_half_width=None,
        track_slack_weight=1.0e6,
        throttle_actuator_tau=0.14,
        steering_actuator_tau=0.14,
        steering_command_delay_steps=0,
        steering_command_gain=0.5234,
        collision_distance=0.65,
        collision_slack_weight=4000.0,
        enable_collision=False,
        integration_substeps=5,
    ):
        self.horizon = int(horizon)
        self.sample_time = float(sample_time)
        self.pwm_bounds = pwm_bounds
        self.steering_bounds = steering_bounds
        self.max_steering_rate = float(max_steering_rate)
        self.max_throttle_rate = float(max_throttle_rate)
        self.throttle_rate_weight = float(throttle_rate_weight)
        self.steering_rate_weight = float(steering_rate_weight)
        self.steering_acceleration_weight = float(
            steering_acceleration_weight
        )
        self.steering_actuator_gap_weight = float(steering_actuator_gap_weight)
        self.heading_weight = float(heading_weight)
        self.yaw_rate_weight = float(yaw_rate_weight)
        self.track_half_width = (
            None if track_half_width is None else float(track_half_width)
        )
        self.track_slack_weight = max(float(track_slack_weight), 0.0)
        self.throttle_actuator_tau = max(float(throttle_actuator_tau), 1.0e-3)
        self.steering_actuator_tau = max(float(steering_actuator_tau), 1.0e-3)
        self.steering_command_delay_steps = int(steering_command_delay_steps)
        if self.steering_command_delay_steps not in (0, 1):
            raise ValueError("steering_command_delay_steps must be 0 or 1")
        self.steering_command_gain = float(steering_command_gain)
        self.collision_distance = max(float(collision_distance), 0.0)
        self.collision_slack_weight = max(float(collision_slack_weight), 0.0)
        self.enable_collision = bool(enable_collision)
        self.integration_substeps = max(1, int(integration_substeps))
        self.model = LePAVDDynamics(model_parameters, kinematic_wheelbase)
        self._last_solution = None
        self.last_collision_slack = np.zeros(self.horizon, dtype=float)
        self.last_track_slack = np.zeros(self.horizon, dtype=float)
        self.last_status = "not_solved"
        self._build_solver()

    def _build_solver(self):
        n, m, horizon = self.STATE_SIZE, self.INPUT_SIZE, self.horizon
        rollout_state = ca.SX.sym("rollout_state", self.VEHICLE_STATE_SIZE)
        rollout_control = ca.SX.sym("rollout_control", m)
        rollout_parameters = ca.SX.sym(
            "rollout_parameters", len(self.model.PARAMETER_NAMES)
        )
        self._dynamics_function = ca.Function(
            "lepavd_rollout",
            [rollout_state, rollout_control, rollout_parameters],
            [self.model.derivative(
                rollout_state, rollout_control, rollout_parameters
            )],
        )
        states = ca.SX.sym("states", n, horizon + 1)
        controls = ca.SX.sym("controls", m, horizon)
        collision_slack = ca.SX.sym("collision_slack", horizon)
        track_slack = ca.SX.sym("track_slack", horizon)
        reference_size = n * (horizon + 1)
        peer_size = 2 * (horizon + 1) if self.enable_collision else 0
        model_parameter_size = len(self.model.PARAMETER_NAMES)
        parameters = ca.SX.sym(
            "parameters",
            n + reference_size + m + m + m + model_parameter_size
            + peer_size + int(self.enable_collision),
        )
        initial_state = parameters[:n]
        reference_end = n + reference_size
        reference = ca.reshape(
            parameters[n:reference_end], n, horizon + 1
        )
        previous_control = parameters[reference_end : reference_end + m]
        previous_previous_control = parameters[
            reference_end + m : reference_end + 2 * m
        ]
        actuator_start = reference_end + 2 * m
        applied_control = parameters[actuator_start : actuator_start + m]
        model_start = actuator_start + m
        dynamic_parameters = parameters[
            model_start : model_start + model_parameter_size
        ]
        if self.enable_collision:
            peer_start = model_start + model_parameter_size
            peer_reference = ca.reshape(
                parameters[peer_start : peer_start + peer_size], 2, horizon + 1
            )
            collision_active = parameters[-1]
        q = ca.diag(ca.DM([
            18.0, 18.0, self.heading_weight, 2.0, 0.15,
            self.yaw_rate_weight,
        ]))
        terminal_q = 5.0 * q
        rate_r = ca.diag(
            ca.DM([self.throttle_rate_weight, self.steering_rate_weight])
        )
        input_r = ca.diag(ca.DM([0.02, 0.10]))
        objective = 0
        equality = [states[:, 0] - initial_state]
        steering_inequality = []
        throttle_inequality = []
        collision_inequality = []
        track_inequality = []
        previous_control_delta = previous_control - previous_previous_control

        for index in range(horizon):
            error = states[:, index] - reference[:, index]
            control_delta = controls[:, index] - (
                previous_control if index == 0 else controls[:, index - 1]
            )
            control_acceleration = control_delta - previous_control_delta
            objective += (
                ca.mtimes([error.T, q, error])
                + ca.mtimes([control_delta.T, rate_r, control_delta])
                + ca.mtimes([controls[:, index].T, input_r, controls[:, index]])
                + self.steering_acceleration_weight
                * control_acceleration[1] ** 2
            )
            previous_control_delta = control_delta
            steering_command = controls[1, index]
            if self.steering_command_delay_steps == 1:
                steering_command = (
                    previous_control[1]
                    if index == 0 else controls[1, index - 1]
                )
            target_control = ca.vertcat(
                controls[0, index], self.steering_command_gain * steering_command
            )
            response = ca.DM([
                1.0 - np.exp(-self.sample_time / self.throttle_actuator_tau),
                1.0 - np.exp(-self.sample_time / self.steering_actuator_tau),
            ])
            next_applied_control = applied_control + response * (
                target_control - applied_control
            )
            objective += self.steering_actuator_gap_weight * (
                target_control[1] - next_applied_control[1]
            ) ** 2
            mean_applied_control = 0.5 * (applied_control + next_applied_control)
            propagated_state = states[:, index]
            integration_dt = self.sample_time / self.integration_substeps
            for _ in range(self.integration_substeps):
                propagated_state = propagated_state + integration_dt * self.model.derivative(
                    propagated_state, mean_applied_control, dynamic_parameters
                )
            equality.append(states[:, index + 1] - propagated_state)
            applied_control = next_applied_control
            steering_inequality.append(control_delta[1])
            throttle_inequality.append(control_delta[0])
            if self.enable_collision:
                peer_dx = states[0, index + 1] - peer_reference[0, index + 1]
                peer_dy = states[1, index + 1] - peer_reference[1, index + 1]
                collision_inequality.append(
                    peer_dx * peer_dx
                    + peer_dy * peer_dy
                    + collision_slack[index]
                    + (1.0 - collision_active) * 1.0e4
                )
                objective += self.collision_slack_weight * collision_slack[index] ** 2
            if self.track_half_width is not None:
                # Constrain the predicted next state. Constraining state zero
                # can make the NLP infeasible because that state is fixed.
                normal_x = -ca.sin(reference[2, index + 1])
                normal_y = ca.cos(reference[2, index + 1])
                lateral_error = (
                    normal_x * (states[0, index + 1] - reference[0, index + 1])
                    + normal_y * (states[1, index + 1] - reference[1, index + 1])
                )
                track_inequality.extend([
                    lateral_error - track_slack[index],
                    -lateral_error - track_slack[index],
                ])
                objective += self.track_slack_weight * track_slack[index] ** 2

        terminal_error = states[:, -1] - reference[:, -1]
        objective += ca.mtimes([terminal_error.T, terminal_q, terminal_error])
        decisions = [ca.reshape(states, -1, 1), ca.reshape(controls, -1, 1)]
        if self.enable_collision:
            decisions.append(collision_slack)
        if self.track_half_width is not None:
            decisions.append(track_slack)
        decision = ca.vertcat(*decisions)
        problem = {
            "x": decision,
            "p": parameters,
            "f": objective,
            "g": ca.vertcat(*(
                equality
                + steering_inequality
                + throttle_inequality
                + collision_inequality
                + track_inequality
            )),
        }
        self._solver = ca.nlpsol(
            "lepavd_nmpc",
            "ipopt",
            problem,
            {
                "expand": True,
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.max_iter": 60,
                "ipopt.acceptable_tol": 1e-4,
                "ipopt.acceptable_iter": 5,
                "ipopt.sb": "yes",
            },
        )

        state_variables = n * (horizon + 1)
        total_variables = (
            state_variables + m * horizon + (horizon if self.enable_collision else 0)
            + (horizon if self.track_half_width is not None else 0)
        )
        self._lower_x = np.full(total_variables, -np.inf)
        self._upper_x = np.full(total_variables, np.inf)
        for index in range(horizon):
            offset = state_variables + index * m
            self._lower_x[offset : offset + m] = [self.pwm_bounds[0], self.steering_bounds[0]]
            self._upper_x[offset : offset + m] = [self.pwm_bounds[1], self.steering_bounds[1]]
        slack_offset = state_variables + m * horizon
        if self.enable_collision:
            self._lower_x[slack_offset : slack_offset + horizon] = 0.0
            slack_offset += horizon
        if self.track_half_width is not None:
            self._lower_x[slack_offset : slack_offset + horizon] = 0.0
        equality_size = n * (horizon + 1)
        steering_step = self.max_steering_rate * self.sample_time
        throttle_step = self.max_throttle_rate * self.sample_time
        lower_parts = [
            np.zeros(equality_size),
            np.full(horizon, -steering_step),
            np.full(horizon, -throttle_step),
        ]
        upper_parts = [
            np.zeros(equality_size),
            np.full(horizon, steering_step),
            np.full(horizon, throttle_step),
        ]
        if self.enable_collision:
            lower_parts.append(np.full(horizon, self.collision_distance**2))
            upper_parts.append(np.full(horizon, np.inf))
        self._lower_g = np.concatenate(lower_parts)
        self._upper_g = np.concatenate(upper_parts)
        if self.track_half_width is not None:
            self._lower_g = np.concatenate([
                self._lower_g, np.full(2 * horizon, -np.inf)
            ])
            self._upper_g = np.concatenate([
                self._upper_g, np.full(2 * horizon, self.track_half_width)
            ])

    def solve(
        self,
        state,
        reference,
        previous_control,
        actuator_state=None,
        model_parameters=None,
        peer_reference=None,
        track_margin=0.0,
        track_left_clearance=None,
        track_right_clearance=None,
        pwm_upper_bound=None,
    ):
        vehicle_state = np.asarray(state, dtype=float).reshape(self.VEHICLE_STATE_SIZE)
        if actuator_state is None:
            actuator_state = previous_control
        state = vehicle_state
        reference = np.asarray(reference, dtype=float).reshape(
            self.VEHICLE_STATE_SIZE, self.horizon + 1
        )
        previous_control = np.asarray(previous_control, dtype=float).reshape(self.INPUT_SIZE)
        previous_previous_control = getattr(
            self, "_previous_control_argument", previous_control
        )
        actuator_state = np.asarray(actuator_state, dtype=float).reshape(self.INPUT_SIZE)
        model_parameters = self.model.parameter_vector(model_parameters)
        parameter_parts = [
            state,
            reference.reshape(-1, order="F"),
            previous_control,
            previous_previous_control,
            actuator_state,
            model_parameters,
        ]
        collision_active = self.enable_collision and peer_reference is not None
        if self.enable_collision:
            if peer_reference is None:
                peer_reference = np.zeros((2, self.horizon + 1), dtype=float)
            peer_reference = np.asarray(peer_reference, dtype=float).reshape(
                2, self.horizon + 1
            )
            parameter_parts.extend([
                peer_reference.reshape(-1, order="F"),
                [float(collision_active)],
            ])
        parameters = np.concatenate(parameter_parts)
        if self._last_solution is None:
            state_guess = np.empty_like(reference)
            control_seed = np.array([0.65, previous_control[1]], dtype=float)
            state_guess[:, 0] = state
            for index in range(self.horizon):
                target_control = np.array([
                    control_seed[0], self.steering_command_gain * control_seed[1]
                ])
                response = np.array([
                    1.0 - np.exp(-self.sample_time / self.throttle_actuator_tau),
                    1.0 - np.exp(-self.sample_time / self.steering_actuator_tau),
                ])
                next_actuator = actuator_state + response * (
                    target_control - actuator_state
                )
                propagated_state = state_guess[:, index].copy()
                mean_actuator = 0.5 * (actuator_state + next_actuator)
                integration_dt = self.sample_time / self.integration_substeps
                for _ in range(self.integration_substeps):
                    derivative = np.asarray(
                        self._dynamics_function(
                            propagated_state,
                            mean_actuator,
                            model_parameters,
                        )
                    ).reshape(self.STATE_SIZE)
                    propagated_state += integration_dt * derivative
                actuator_state = next_actuator
                state_guess[:, index + 1] = propagated_state
            guesses = [
                state_guess.reshape(-1, order="F"),
                np.tile(control_seed, self.horizon),
            ]
            if self.enable_collision:
                guesses.append(np.zeros(self.horizon))
            if self.track_half_width is not None:
                guesses.append(np.zeros(self.horizon))
            initial_guess = np.concatenate(guesses)
        else:
            initial_guess = self._last_solution
        lower_g = self._lower_g
        upper_g = self._upper_g
        upper_x = self._upper_x
        if pwm_upper_bound is not None:
            upper_x = self._upper_x.copy()
            state_count = self.STATE_SIZE * (self.horizon + 1)
            bounded_pwm = min(float(pwm_upper_bound), self.pwm_bounds[1])
            for index in range(self.horizon):
                upper_x[state_count + index * self.INPUT_SIZE] = bounded_pwm
        if self.track_half_width is not None:
            upper_g = self._upper_g.copy()
            if track_left_clearance is None or track_right_clearance is None:
                left = np.full(self.horizon, self.track_half_width)
                right = np.full(self.horizon, self.track_half_width)
            else:
                left = np.asarray(track_left_clearance, dtype=float).reshape(
                    self.horizon
                )
                right = np.asarray(track_right_clearance, dtype=float).reshape(
                    self.horizon
                )
            margin = float(track_margin)
            clearances = np.column_stack([
                np.maximum(0.02, left - margin),
                np.maximum(0.02, right - margin),
            ]).reshape(-1)
            upper_g[-2 * self.horizon :] = clearances
        result = self._solver(
            x0=initial_guess,
            p=parameters,
            lbx=self._lower_x,
            ubx=upper_x,
            lbg=lower_g,
            ubg=upper_g,
        )
        solution = np.asarray(result["x"]).reshape(-1)
        status = str(self._solver.stats().get("return_status", "NMPC failed"))
        self.last_status = status
        if (
            not bool(self._solver.stats().get("success", False))
        ) or not np.all(np.isfinite(solution)):
            raise RuntimeError(status)
        state_count = self.STATE_SIZE * (self.horizon + 1)
        prediction = solution[:state_count].reshape(
            self.STATE_SIZE, self.horizon + 1, order="F"
        )
        controls = solution[
            state_count : state_count + self.INPUT_SIZE * self.horizon
        ].reshape(
            self.INPUT_SIZE, self.horizon, order="F"
        )
        slack_offset = state_count + self.INPUT_SIZE * self.horizon
        if self.enable_collision:
            self.last_collision_slack = solution[
                slack_offset : slack_offset + self.horizon
            ].copy()
            slack_offset += self.horizon
        else:
            self.last_collision_slack.fill(0.0)
        if self.track_half_width is not None:
            self.last_track_slack = solution[
                slack_offset : slack_offset + self.horizon
            ].copy()
        else:
            self.last_track_slack.fill(0.0)
        shifted_states = np.column_stack([prediction[:, 1:], prediction[:, -1]])
        shifted_controls = np.column_stack([controls[:, 1:], controls[:, -1]])
        warm_parts = [
            shifted_states.reshape(-1, order="F"),
            shifted_controls.reshape(-1, order="F"),
        ]
        if self.enable_collision:
            warm_parts.append(np.concatenate([
                self.last_collision_slack[1:], self.last_collision_slack[-1:]
            ]))
        if self.track_half_width is not None:
            warm_parts.append(np.concatenate([
                self.last_track_slack[1:], self.last_track_slack[-1:]
            ]))
        self._last_solution = np.concatenate(warm_parts)
        self._previous_control_argument = previous_control.copy()
        return controls[:, 0], prediction, float(result["f"])

#!/usr/bin/env python3

"""Nominal lePAVD NMPC core adapted from DaRC-ARMs-lab/DynSSM.

State: [x, y, yaw, vx, vy, yaw_rate]. Control: [pwm, steering].
"""

from dataclasses import dataclass

import casadi as ca
import numpy as np


@dataclass(frozen=True)
class LePAVDParameters:
    lf: float = 0.029
    lr: float = 0.033
    mass: float = 0.041
    h_cg: float = 0.035
    gravity: float = 9.81
    Bf: float = 5.579
    Cf: float = 1.2
    Df: float = 0.192
    Ef: float = -0.083
    Br: float = 5.3852
    Cr: float = 1.2691
    Dr: float = 0.1737
    Er: float = -0.019
    Cm1: float = 0.287
    Cm2: float = 0.0545
    Cr0: float = 0.0518
    Cr2: float = 0.00035
    Iz: float = 2.78e-5
    Shf: float = -0.0013
    Svf: float = 0.00043
    Shr: float = -0.00376
    Svr: float = 0.00091


class LePAVDDynamics:
    """CasADi form of the DynSSM nominal dynamic bicycle model."""

    def __init__(self, parameters=None, kinematic_wheelbase=None):
        self.p = parameters or LePAVDParameters()
        self.kinematic_wheelbase = kinematic_wheelbase

    def derivative(self, state, control):
        p = self.p
        yaw, vx_raw, vy_raw, yaw_rate_raw = state[2], state[3], state[4], state[5]
        pwm, steering = control[0], control[1]
        vx = ca.fmin(ca.fmax(vx_raw, 0.05), 10.0)
        vy = ca.if_else(vx_raw < 0.05, 0.0, vy_raw)
        yaw_rate = ca.if_else(vx_raw < 0.05, 0.0, yaw_rate_raw)
        force_x = (p.Cm1 - p.Cm2 * vx) * pwm - p.Cr0 - p.Cr2 * vx * vx

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

        alpha_f = steering - ca.atan2(p.lf * yaw_rate + vy, vx) + p.Shf
        alpha_r = ca.atan2(p.lr * yaw_rate - vy, vx) + p.Shr
        bf_alpha = p.Bf * alpha_f
        br_alpha = p.Br * alpha_r
        theta_f = ca.atan(bf_alpha - p.Ef * (bf_alpha - ca.atan(bf_alpha)))
        theta_r = ca.atan(br_alpha - p.Er * (br_alpha - ca.atan(br_alpha)))
        force_f = p.Svf + p.Df * ca.sin(p.Cf * theta_f)
        force_r = p.Svr + p.Dr * ca.sin(p.Cr * theta_r)

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
            (force_f * p.lf * ca.cos(steering) - force_r * p.lr) / p.Iz,
        )


class NominalLePAVDNMPC:
    """Warm-started nonlinear MPC using the nominal lePAVD model."""

    STATE_SIZE = 6
    INPUT_SIZE = 2

    def __init__(
        self,
        horizon=12,
        sample_time=0.10,
        pwm_bounds=(-0.10, 1.0),
        steering_bounds=(-0.35, 0.35),
        max_steering_rate=2.5,
        model_parameters=None,
        kinematic_wheelbase=None,
    ):
        self.horizon = int(horizon)
        self.sample_time = float(sample_time)
        self.pwm_bounds = pwm_bounds
        self.steering_bounds = steering_bounds
        self.max_steering_rate = float(max_steering_rate)
        self.model = LePAVDDynamics(model_parameters, kinematic_wheelbase)
        self._last_solution = None
        self._build_solver()

    def _build_solver(self):
        n, m, horizon = self.STATE_SIZE, self.INPUT_SIZE, self.horizon
        rollout_state = ca.SX.sym("rollout_state", n)
        rollout_control = ca.SX.sym("rollout_control", m)
        self._dynamics_function = ca.Function(
            "lepavd_rollout",
            [rollout_state, rollout_control],
            [self.model.derivative(rollout_state, rollout_control)],
        )
        states = ca.SX.sym("states", n, horizon + 1)
        controls = ca.SX.sym("controls", m, horizon)
        parameters = ca.SX.sym("parameters", n + n * (horizon + 1) + m)
        initial_state = parameters[:n]
        reference = ca.reshape(parameters[n : n + n * (horizon + 1)], n, horizon + 1)
        previous_control = parameters[-m:]
        q = ca.diag(ca.DM([18.0, 18.0, 2.5, 2.0, 0.15, 0.25]))
        terminal_q = 5.0 * q
        rate_r = ca.diag(ca.DM([0.25, 8.0]))
        input_r = ca.diag(ca.DM([0.02, 0.10]))
        objective = 0
        equality = [states[:, 0] - initial_state]
        inequality = []

        for index in range(horizon):
            error = states[:, index] - reference[:, index]
            control_delta = controls[:, index] - (
                previous_control if index == 0 else controls[:, index - 1]
            )
            objective += (
                ca.mtimes([error.T, q, error])
                + ca.mtimes([control_delta.T, rate_r, control_delta])
                + ca.mtimes([controls[:, index].T, input_r, controls[:, index]])
            )
            derivative = self.model.derivative(states[:, index], controls[:, index])
            equality.append(
                states[:, index + 1]
                - states[:, index]
                - self.sample_time * derivative
            )
            inequality.append(control_delta[1])

        terminal_error = states[:, -1] - reference[:, -1]
        objective += ca.mtimes([terminal_error.T, terminal_q, terminal_error])
        decision = ca.vertcat(ca.reshape(states, -1, 1), ca.reshape(controls, -1, 1))
        problem = {
            "x": decision,
            "p": parameters,
            "f": objective,
            "g": ca.vertcat(*(equality + inequality)),
        }
        self._solver = ca.nlpsol(
            "lepavd_nmpc",
            "ipopt",
            problem,
            {
                "expand": True,
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.max_iter": 120,
                "ipopt.acceptable_tol": 1e-4,
                "ipopt.sb": "yes",
            },
        )

        state_variables = n * (horizon + 1)
        total_variables = state_variables + m * horizon
        self._lower_x = np.full(total_variables, -np.inf)
        self._upper_x = np.full(total_variables, np.inf)
        for index in range(horizon):
            offset = state_variables + index * m
            self._lower_x[offset : offset + m] = [self.pwm_bounds[0], self.steering_bounds[0]]
            self._upper_x[offset : offset + m] = [self.pwm_bounds[1], self.steering_bounds[1]]
        equality_size = n * (horizon + 1)
        steering_step = self.max_steering_rate * self.sample_time
        self._lower_g = np.concatenate(
            [np.zeros(equality_size), np.full(horizon, -steering_step)]
        )
        self._upper_g = np.concatenate(
            [np.zeros(equality_size), np.full(horizon, steering_step)]
        )

    def solve(self, state, reference, previous_control):
        state = np.asarray(state, dtype=float).reshape(self.STATE_SIZE)
        reference = np.asarray(reference, dtype=float).reshape(
            self.STATE_SIZE, self.horizon + 1
        )
        previous_control = np.asarray(previous_control, dtype=float).reshape(self.INPUT_SIZE)
        parameters = np.concatenate(
            [state, reference.reshape(-1, order="F"), previous_control]
        )
        if self._last_solution is None:
            state_guess = np.empty_like(reference)
            control_seed = np.array([0.65, previous_control[1]], dtype=float)
            state_guess[:, 0] = state
            for index in range(self.horizon):
                derivative = np.asarray(
                    self._dynamics_function(state_guess[:, index], control_seed)
                ).reshape(self.STATE_SIZE)
                state_guess[:, index + 1] = (
                    state_guess[:, index] + self.sample_time * derivative
                )
            initial_guess = np.concatenate(
                [state_guess.reshape(-1, order="F"), np.tile(control_seed, self.horizon)]
            )
        else:
            initial_guess = self._last_solution
        result = self._solver(
            x0=initial_guess,
            p=parameters,
            lbx=self._lower_x,
            ubx=self._upper_x,
            lbg=self._lower_g,
            ubg=self._upper_g,
        )
        solution = np.asarray(result["x"]).reshape(-1)
        status = str(self._solver.stats().get("return_status", "NMPC failed"))
        usable_iteration_limit = status == "Maximum_Iterations_Exceeded"
        if (
            not bool(self._solver.stats().get("success", False))
            and not usable_iteration_limit
        ) or not np.all(np.isfinite(solution)):
            raise RuntimeError(status)
        self._last_solution = solution
        state_count = self.STATE_SIZE * (self.horizon + 1)
        prediction = solution[:state_count].reshape(
            self.STATE_SIZE, self.horizon + 1, order="F"
        )
        controls = solution[state_count:].reshape(
            self.INPUT_SIZE, self.horizon, order="F"
        )
        return controls[:, 0], prediction, float(result["f"])

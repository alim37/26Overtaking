#!/usr/bin/env python3
"""Reproducible sensing and plant-interface uncertainty for AIxVR trials."""

import csv
import json
import math
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial import cKDTree
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, String


def _finite(value, fallback=0.0):
    value = float(value)
    return value if math.isfinite(value) else float(fallback)


class UncertaintyBridge(Node):
    """Perturb controller measurements and commands without changing NMPC code."""

    def __init__(self):
        super().__init__("aixvr_uncertainty_bridge")
        package_root = Path(__file__).resolve().parents[2]
        defaults = {
            "car_id": 1,
            "experiment": "nominal",
            "severity": 0.0,
            "seed": 47,
            "control_period_sec": 0.10,
            "measurement_noise_fraction": 0.0,
            "position_noise_std_m": 0.0,
            "speed_noise_floor_mps": 0.02,
            "yaw_rate_noise_floor_rad_s": 0.002,
            "position_bias_walk_m_sqrt_s": 0.0,
            "yaw_rate_bias_walk_rad_s_sqrt_s": 0.0,
            "throttle_delay_sec": 0.0,
            "steering_delay_sec": 0.0,
            "actuation_noise_fraction": 0.0,
            "throttle_response_tau_sec": 0.0,
            "steering_response_tau_sec": 0.0,
            "throttle_effectiveness": 1.0,
            "steering_effectiveness": 1.0,
            "external_dynamics_active": False,
            "external_dynamics_configuration_id": "nominal_unity",
            "external_vehicle_mass_kg": 0.0,
            "external_front_tire_stiffness": 0.0,
            "external_rear_tire_stiffness": 0.0,
            "max_abs_throttle": 0.13,
            "max_abs_steering_rad": 0.46,
            "vehicle_width_m": 0.30,
            "boundary_margin_m": 0.03,
            "wall_mask_csv": str(
                package_root / "output/slam_runs/slam_toolbox_boundary_wall_mask.csv"
            ),
            "output_dir": str(package_root / "output/aixvr_uncertainty"),
            "output_stem": "uncertainty_trial",
            "clean_pure_pursuit_warmup": True,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        get = lambda name: self.get_parameter(name).value

        self.car_id = int(get("car_id"))
        self.experiment = str(get("experiment")).strip().lower()
        self.severity = float(get("severity"))
        self.seed = int(get("seed"))
        self.clean_warmup = bool(get("clean_pure_pursuit_warmup"))
        self.uncertainty_active = not self.clean_warmup
        self.period = max(1.0e-3, float(get("control_period_sec")))
        self.measurement_noise = max(0.0, float(get("measurement_noise_fraction")))
        self.position_noise_std = max(0.0, float(get("position_noise_std_m")))
        self.speed_noise_floor = max(0.0, float(get("speed_noise_floor_mps")))
        self.yaw_rate_noise_floor = max(
            0.0, float(get("yaw_rate_noise_floor_rad_s"))
        )
        self.position_bias_walk = max(
            0.0, float(get("position_bias_walk_m_sqrt_s"))
        )
        self.yaw_rate_bias_walk = max(
            0.0, float(get("yaw_rate_bias_walk_rad_s_sqrt_s"))
        )
        self.throttle_delay = max(0.0, float(get("throttle_delay_sec")))
        self.steering_delay = max(0.0, float(get("steering_delay_sec")))
        self.actuation_noise = max(0.0, float(get("actuation_noise_fraction")))
        self.throttle_tau = max(0.0, float(get("throttle_response_tau_sec")))
        self.steering_tau = max(0.0, float(get("steering_response_tau_sec")))
        self.throttle_effectiveness = max(0.0, float(get("throttle_effectiveness")))
        self.steering_effectiveness = max(0.0, float(get("steering_effectiveness")))
        self.external_dynamics_active = bool(get("external_dynamics_active"))
        self.external_dynamics_configuration_id = str(
            get("external_dynamics_configuration_id")
        )
        self.external_vehicle_mass = float(get("external_vehicle_mass_kg"))
        self.external_front_tire_stiffness = float(
            get("external_front_tire_stiffness")
        )
        self.external_rear_tire_stiffness = float(
            get("external_rear_tire_stiffness")
        )
        self.max_throttle = abs(float(get("max_abs_throttle")))
        self.max_steering = abs(float(get("max_abs_steering_rad")))
        self.vehicle_width = max(0.0, float(get("vehicle_width_m")))
        self.boundary_margin = max(0.0, float(get("boundary_margin_m")))
        self.rng = np.random.default_rng(self.seed)
        self.position_bias = np.zeros(2, dtype=float)
        self.yaw_rate_bias = 0.0
        self.last_bias_time = None
        self.last_throttle_time = None
        self.last_steering_time = None
        self.filtered_throttle = 0.0
        self.filtered_steering = 0.0
        self.nominal_throttle = 0.0
        self.nominal_steering = 0.0
        self.applied_throttle = 0.0
        self.applied_steering = 0.0
        self.true_position = None
        self.measured_position = None
        self.true_speed = math.nan
        self.measured_speed = math.nan
        self.true_yaw_rate = math.nan
        self.measured_yaw_rate = math.nan
        self.start_time = None
        self.completed = False

        throttle_steps = int(round(self.throttle_delay / self.period))
        steering_steps = int(round(self.steering_delay / self.period))
        self.throttle_buffer = deque(
            [0.0] * throttle_steps, maxlen=throttle_steps or None
        )
        self.steering_buffer = deque(
            [0.0] * steering_steps, maxlen=steering_steps or None
        )

        self.wall_tree = self._load_wall_tree(Path(str(get("wall_mask_csv"))))
        output_dir = Path(str(get("output_dir"))).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in str(get("output_stem"))
        )
        self.output_csv = output_dir / f"{stem}.csv"
        self.output_json = output_dir / f"{stem}.json"
        self.csv_file = self.output_csv.open("w", newline="", encoding="utf-8")
        self.fields = [
            "time_sec", "elapsed_time_s", "experiment", "severity", "seed",
            "true_x_m", "true_y_m", "measured_x_m", "measured_y_m",
            "position_measurement_error_m", "true_speed_mps", "measured_speed_mps",
            "true_yaw_rate_rad_s", "measured_yaw_rate_rad_s",
            "nominal_throttle", "applied_throttle", "nominal_steering_rad",
            "applied_steering_rad", "distance_to_wall_m", "boundary_violation",
            "completed",
            "uncertainty_active",
        ]
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fields)
        self.writer.writeheader()
        self.csv_file.flush()
        self._write_metadata()

        base = f"/autodrive/f1tenth_{self.car_id}"
        relay = f"/autodrive_uncertainty/f1tenth_{self.car_id}"
        self.ips_pub = self.create_publisher(Point, f"{relay}/ips", 10)
        self.truth_ips_pub = self.create_publisher(Point, f"{relay}/ground_truth_ips", 10)
        self.odom_pub = self.create_publisher(Odometry, f"{relay}/odom", 10)
        self.imu_pub = self.create_publisher(Imu, f"{relay}/imu", 10)
        self.throttle_pub = self.create_publisher(Float32, f"{base}/throttle_command", 10)
        self.steering_pub = self.create_publisher(Float32, f"{base}/steering_command", 10)
        self.create_subscription(Point, f"{base}/ips", self._ips_cb, 10)
        self.create_subscription(Odometry, f"{base}/odom", self._odom_cb, 10)
        self.create_subscription(Imu, f"{base}/imu", self._imu_cb, 10)
        self.create_subscription(
            Float32, f"{relay}/throttle_command_nominal", self._throttle_cb, 10
        )
        self.create_subscription(
            Float32, f"{relay}/steering_command_nominal", self._steering_cb, 10
        )
        self.create_subscription(
            Bool, "/autodrive/cape/experiment_complete", self._completion_cb, 10
        )
        self.create_subscription(
            String, f"{base}/dynssm_nmpc/mode", self._mode_cb, 10
        )
        self.create_timer(self.period, self._log_sample)

        self.get_logger().info(
            "AIxVR uncertainty bridge ready: "
            f"experiment={self.experiment}, severity={self.severity:.2f}, seed={self.seed}, "
            f"position_noise={self.position_noise_std:.3f}m, "
            f"dynamic_state_noise={100.0 * self.measurement_noise:.3f}%, "
            f"delays=({self.throttle_delay:.3f},{self.steering_delay:.3f})s, "
            f"effectiveness=({self.throttle_effectiveness:.3f},"
            f"{self.steering_effectiveness:.3f}), "
            f"external_dynamics={self.external_dynamics_configuration_id}, "
            f"output={self.output_csv}"
        )

    def _load_wall_tree(self, path):
        if not path.exists():
            self.get_logger().warning(f"Wall mask not found: {path}")
            return None
        points = []
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    points.append((float(row["world_x_m"]), float(row["world_y_m"])))
                except (KeyError, TypeError, ValueError):
                    continue
        return cKDTree(np.asarray(points, dtype=float)) if points else None

    def _write_metadata(self):
        payload = {
            "experiment": self.experiment,
            "severity": self.severity,
            "seed": self.seed,
            "measurement_noise_fraction": self.measurement_noise,
            "position_noise_std_m": self.position_noise_std,
            "position_bias_walk_m_sqrt_s": self.position_bias_walk,
            "yaw_rate_bias_walk_rad_s_sqrt_s": self.yaw_rate_bias_walk,
            "throttle_delay_sec": self.throttle_delay,
            "steering_delay_sec": self.steering_delay,
            "actuation_noise_fraction": self.actuation_noise,
            "throttle_response_tau_sec": self.throttle_tau,
            "steering_response_tau_sec": self.steering_tau,
            "throttle_effectiveness": self.throttle_effectiveness,
            "steering_effectiveness": self.steering_effectiveness,
            "external_dynamics_active": self.external_dynamics_active,
            "external_dynamics_configuration_id": (
                self.external_dynamics_configuration_id
            ),
            "external_vehicle_mass_kg": self.external_vehicle_mass,
            "external_front_tire_stiffness": self.external_front_tire_stiffness,
            "external_rear_tire_stiffness": self.external_rear_tire_stiffness,
            "external_dynamics_note": (
                "Expected Unity configuration recorded by the launch file; ROS cannot "
                "verify Unity Inspector values. No command-side dynamics proxy is applied."
            ),
            "clean_pure_pursuit_warmup": self.clean_warmup,
            "vehicle_width_m": self.vehicle_width,
        }
        self.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _advance_bias(self, now):
        if self.last_bias_time is None:
            self.last_bias_time = now
            return
        dt = float(np.clip(now - self.last_bias_time, 0.0, 0.5))
        self.last_bias_time = now
        root_dt = math.sqrt(dt)
        self.position_bias += self.rng.normal(
            0.0, self.position_bias_walk * root_dt, size=2
        )
        self.yaw_rate_bias += float(self.rng.normal(
            0.0, self.yaw_rate_bias_walk * root_dt
        ))

    def _noisy(self, value, floor):
        sigma = self.measurement_noise * max(abs(float(value)), float(floor))
        return float(value) + float(self.rng.normal(0.0, sigma))

    def _ips_cb(self, msg):
        now = self.get_clock().now().nanoseconds * 1.0e-9
        self.true_position = np.array([float(msg.x), float(msg.y)], dtype=float)
        if not self.uncertainty_active:
            self.measured_position = self.true_position.copy()
            self.ips_pub.publish(deepcopy(msg))
            self.truth_ips_pub.publish(deepcopy(msg))
            return
        self._advance_bias(now)
        measured = self.true_position + self.position_bias
        # Position uncertainty must not depend on the arbitrary world-frame
        # origin. Use additive metric noise rather than multiplying global x/y.
        measured += self.rng.normal(0.0, self.position_noise_std, size=2)
        self.measured_position = measured
        out = deepcopy(msg)
        out.x, out.y = float(measured[0]), float(measured[1])
        self.ips_pub.publish(out)
        self.truth_ips_pub.publish(deepcopy(msg))

    def _odom_cb(self, msg):
        out = deepcopy(msg)
        self.true_speed = abs(float(msg.twist.twist.linear.z))
        if not self.uncertainty_active:
            self.measured_speed = self.true_speed
            self.odom_pub.publish(out)
            return
        self.measured_speed = self._noisy(self.true_speed, self.speed_noise_floor)
        sign = -1.0 if float(msg.twist.twist.linear.z) < 0.0 else 1.0
        out.twist.twist.linear.z = sign * self.measured_speed
        out.twist.twist.linear.x = self._noisy(
            float(msg.twist.twist.linear.x), self.speed_noise_floor
        )
        self.odom_pub.publish(out)

    def _imu_cb(self, msg):
        out = deepcopy(msg)
        self.true_yaw_rate = float(msg.angular_velocity.z)
        if not self.uncertainty_active:
            self.measured_yaw_rate = self.true_yaw_rate
            self.imu_pub.publish(out)
            return
        self.measured_yaw_rate = self._noisy(
            self.true_yaw_rate + self.yaw_rate_bias,
            self.yaw_rate_noise_floor,
        )
        out.angular_velocity.z = self.measured_yaw_rate
        self.imu_pub.publish(out)

    def _delay(self, value, buffer):
        if buffer.maxlen is None:
            return float(value)
        delayed = float(buffer[0])
        buffer.append(float(value))
        return delayed

    def _command_noise(self, value):
        sigma = self.actuation_noise * max(abs(float(value)), 1.0e-3)
        return float(value) + float(self.rng.normal(0.0, sigma))

    def _response(self, current, target, tau, dt):
        if tau <= 0.0:
            return float(target)
        alpha = float(np.clip(dt / (tau + dt), 0.0, 1.0))
        return float(current + alpha * (target - current))

    def _command_dt(self, channel):
        now = self.get_clock().now().nanoseconds * 1.0e-9
        attribute = f"last_{channel}_time"
        previous = getattr(self, attribute)
        dt = self.period if previous is None else now - previous
        setattr(self, attribute, now)
        return float(np.clip(dt, 1.0e-3, 0.5))

    def _throttle_cb(self, msg):
        self.nominal_throttle = float(msg.data)
        if not self.uncertainty_active:
            self.filtered_throttle = self.nominal_throttle
            self.applied_throttle = float(np.clip(
                self.nominal_throttle, -self.max_throttle, self.max_throttle
            ))
            self.throttle_pub.publish(Float32(data=self.applied_throttle))
            return
        target = self._delay(self.nominal_throttle, self.throttle_buffer)
        target = self._command_noise(target) * self.throttle_effectiveness
        self.filtered_throttle = self._response(
            self.filtered_throttle,
            target,
            self.throttle_tau,
            self._command_dt("throttle"),
        )
        self.applied_throttle = float(np.clip(
            self.filtered_throttle, -self.max_throttle, self.max_throttle
        ))
        self.throttle_pub.publish(Float32(data=self.applied_throttle))

    def _steering_cb(self, msg):
        self.nominal_steering = float(msg.data)
        if not self.uncertainty_active:
            self.filtered_steering = self.nominal_steering
            self.applied_steering = float(np.clip(
                self.nominal_steering, -self.max_steering, self.max_steering
            ))
            self.steering_pub.publish(Float32(data=self.applied_steering))
            return
        target = self._delay(self.nominal_steering, self.steering_buffer)
        target = self._command_noise(target) * self.steering_effectiveness
        self.filtered_steering = self._response(
            self.filtered_steering,
            target,
            self.steering_tau,
            self._command_dt("steering"),
        )
        self.applied_steering = float(np.clip(
            self.filtered_steering, -self.max_steering, self.max_steering
        ))
        self.steering_pub.publish(Float32(data=self.applied_steering))

    def _completion_cb(self, msg):
        self.completed = bool(msg.data)

    def _mode_cb(self, msg):
        if not self.clean_warmup or self.uncertainty_active:
            return
        mode = str(msg.data)
        if not mode or mode.startswith("PURE_PURSUIT_WARMUP"):
            return
        self.uncertainty_active = True
        self.position_bias[:] = 0.0
        self.yaw_rate_bias = 0.0
        self.last_bias_time = None
        self.last_throttle_time = None
        self.last_steering_time = None
        self.filtered_throttle = self.nominal_throttle
        self.filtered_steering = self.nominal_steering
        if self.throttle_buffer.maxlen is not None:
            self.throttle_buffer.clear()
            self.throttle_buffer.extend(
                [self.nominal_throttle] * self.throttle_buffer.maxlen
            )
        if self.steering_buffer.maxlen is not None:
            self.steering_buffer.clear()
            self.steering_buffer.extend(
                [self.nominal_steering] * self.steering_buffer.maxlen
            )
        self.get_logger().info(
            f"Pure-pursuit history complete; uncertainty activated in mode={mode}"
        )

    def _log_sample(self):
        if self.true_position is None or self.measured_position is None:
            return
        now = self.get_clock().now().nanoseconds * 1.0e-9
        if self.start_time is None:
            self.start_time = now
        wall_distance = (
            float(self.wall_tree.query(self.true_position)[0])
            if self.wall_tree is not None else math.nan
        )
        violation_threshold = 0.5 * self.vehicle_width + self.boundary_margin
        self.writer.writerow({
            "time_sec": f"{now:.6f}",
            "elapsed_time_s": f"{now - self.start_time:.6f}",
            "experiment": self.experiment,
            "severity": self.severity,
            "seed": self.seed,
            "true_x_m": self.true_position[0],
            "true_y_m": self.true_position[1],
            "measured_x_m": self.measured_position[0],
            "measured_y_m": self.measured_position[1],
            "position_measurement_error_m": float(np.linalg.norm(
                self.measured_position - self.true_position
            )),
            "true_speed_mps": self.true_speed,
            "measured_speed_mps": self.measured_speed,
            "true_yaw_rate_rad_s": self.true_yaw_rate,
            "measured_yaw_rate_rad_s": self.measured_yaw_rate,
            "nominal_throttle": self.nominal_throttle,
            "applied_throttle": self.applied_throttle,
            "nominal_steering_rad": self.nominal_steering,
            "applied_steering_rad": self.applied_steering,
            "distance_to_wall_m": wall_distance,
            "boundary_violation": int(
                math.isfinite(wall_distance) and wall_distance < violation_threshold
            ),
            "completed": int(self.completed),
            "uncertainty_active": int(self.uncertainty_active),
        })
        self.csv_file.flush()

    def destroy_node(self):
        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UncertaintyBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

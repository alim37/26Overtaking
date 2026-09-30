#!/usr/bin/env python3

"""Run nominal lePAVD NMPC and switch its reference to the Frenet overtake."""

import csv
import math
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, Int32, String
from visualization_msgs.msg import Marker, MarkerArray

from autodrive_f1tenth.dynssm_nmpc import LePAVDParameters, NominalLePAVDNMPC
from autodrive_f1tenth.dynssm_runtime import DynSSMRuntime
from autodrive_f1tenth.pure_pursuit import load_manual_reference_line


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class PathReference:
    def __init__(self, points, closed):
        self.points = np.asarray(points, dtype=float)
        self.closed = bool(closed)
        self.interpolation_points = (
            np.vstack([self.points, self.points[0]]) if closed else self.points
        )
        segment_lengths = np.linalg.norm(
            np.diff(self.interpolation_points, axis=0), axis=1
        )
        self.arc = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        self.length = float(self.arc[-1])

    def nearest_arc(self, position):
        index = int(np.argmin(np.linalg.norm(self.points - position, axis=1)))
        return float(self.arc[index])

    def sample(self, start_arc, spacing, count):
        query = start_arc + np.arange(count, dtype=float) * max(spacing, 0.03)
        query = query % self.length if self.closed else np.clip(query, 0.0, self.length)
        x = np.interp(query, self.arc, self.interpolation_points[:, 0])
        y = np.interp(query, self.arc, self.interpolation_points[:, 1])
        points = np.column_stack([x, y])
        delta = np.gradient(points, axis=0)
        headings = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
        return points, headings


class DynSSMNMPCController(Node):
    NORMAL = "NORMAL_NMPC"
    ENGAGEMENT = "FRENET_OVERTAKE_NMPC"
    COMPLETE = "COMPLETE_NMPC"

    def __init__(self):
        super().__init__("dynssm_nmpc_controller")
        package_root = Path(__file__).resolve().parents[1]
        default_path = (
            package_root
            / "output/engagement_zones_dynamic_frenet"
            / "engagement_zone_dynamic_frenet_one_lap_path.csv"
        )
        defaults = {
            "path_csv": str(default_path),
            "car_id": 1,
            "peer_car_id": 2,
            "enable_overtake_path": True,
            "wait_for_peer_ips": False,
            "startup_hold_sec": 1.0,
            "track_name": "ethz",
            "num_path_points": 800,
            "control_period": 0.10,
            "nmpc_horizon": 12,
            "nominal_velocity_mps": 3.25,
            "engagement_speed_multiplier": 1.30,
            "engagement_speed_delay_sec": 0.25,
            "pwm_to_sim_scale": 0.27,
            "max_sim_throttle": 0.22,
            "dynssm_min_adaptation_speed_mps": 0.50,
            "model_wheelbase_m": 0.30,
            "max_steering_rad": 0.75,
            "max_steering_rate_rad_s": 5.0,
            "corner_steering_rad": 0.175,
            "corner_throttle_scale": 0.65,
            "c_start_radius_m": 1.0,
            "c_end_radius_m": 0.8,
            "stop_after_laps": 1,
            "lap_start_radius_m": 1.5,
            "lap_min_distance_m": 35.0,
            "wait_for_startup_gate": True,
            "use_dynssm_adaptation": False,
            "require_dynssm_active": False,
            "dynssm_checkpoint": str(
                package_root.parents[1]
                / "DynSSM/output/DynSSM_ORCA/ssm_gru_tune_01/best_model_val_rmse.pth"
            ),
            "dynssm_config": str(
                package_root.parents[1]
                / "DynSSM/output/DynSSM_ORCA/ssm_gru_tune_01/config.json"
            ),
            "dynssm_scaler": str(
                package_root.parents[1]
                / "DynSSM/output/DynSSM_ORCA/ssm_gru_tune_01/scaler.pkl"
            ),
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        value = lambda name: self.get_parameter(name).value
        self.control_period = float(value("control_period"))
        self.car_id = int(value("car_id"))
        self.peer_car_id = int(value("peer_car_id"))
        self.enable_overtake_path = bool(value("enable_overtake_path"))
        self.wait_for_peer_ips = bool(value("wait_for_peer_ips"))
        self.startup_hold_sec = float(value("startup_hold_sec"))
        self.horizon = int(value("nmpc_horizon"))
        self.nominal_velocity = float(value("nominal_velocity_mps"))
        self.speed_multiplier = float(value("engagement_speed_multiplier"))
        self.speed_delay = float(value("engagement_speed_delay_sec"))
        self.pwm_to_sim_scale = float(value("pwm_to_sim_scale"))
        self.max_sim_throttle = float(value("max_sim_throttle"))
        self.dynssm_min_adaptation_speed = float(
            value("dynssm_min_adaptation_speed_mps")
        )
        self.model_wheelbase = float(value("model_wheelbase_m"))
        self.max_steering = float(value("max_steering_rad"))
        self.max_steering_rate = float(value("max_steering_rate_rad_s"))
        self.corner_steering = float(value("corner_steering_rad"))
        self.corner_throttle_scale = float(value("corner_throttle_scale"))
        self.c_start_radius = float(value("c_start_radius_m"))
        self.c_end_radius = float(value("c_end_radius_m"))
        self.stop_after_laps = int(value("stop_after_laps"))
        self.lap_start_radius = float(value("lap_start_radius_m"))
        self.lap_min_distance = float(value("lap_min_distance_m"))
        self.startup_gate_open = not bool(value("wait_for_startup_gate"))
        if self.wait_for_peer_ips:
            self.startup_gate_open = False

        normal_points = load_manual_reference_line(
            int(value("num_path_points")), track_name=str(value("track_name"))
        )
        self.path_csv = Path(str(value("path_csv"))).expanduser()
        engagement_points = self._load_path(self.path_csv)
        self.normal_path = PathReference(normal_points, closed=True)
        self.engagement_path = PathReference(engagement_points, closed=False)
        self.c_start, self.c_end = engagement_points[0], engagement_points[-1]
        self.nmpc = NominalLePAVDNMPC(
            horizon=self.horizon,
            sample_time=self.control_period,
            steering_bounds=(-self.max_steering, self.max_steering),
            max_steering_rate=self.max_steering_rate,
            model_parameters=LePAVDParameters(),
            kinematic_wheelbase=self.model_wheelbase,
        )
        self.use_dynssm = bool(value("use_dynssm_adaptation"))
        self.require_dynssm = bool(value("require_dynssm_active"))
        self.dynssm = None
        self.dynssm_active = False
        self.model_history = deque(maxlen=5)
        if self.use_dynssm:
            self.dynssm = DynSSMRuntime(
                Path(str(value("dynssm_checkpoint"))).expanduser(),
                Path(str(value("dynssm_config"))).expanduser(),
                Path(str(value("dynssm_scaler"))).expanduser(),
            )
            self.model_history = deque(maxlen=self.dynssm.horizon)

        self.position = None
        self.heading = None
        self.speed = 0.0
        self.odom_speed = 0.0
        self.ips_speed = 0.0
        self.lateral_speed = 0.0
        self.yaw_rate = 0.0
        self.previous_control = np.zeros(2, dtype=float)
        self.mode = self.NORMAL
        self.engagement_start_time = None
        self.engagement_complete = False
        self.start_position = None
        self.total_distance = 0.0
        self.left_start_zone = False
        self.lap_count = 0
        self.stopped = False
        self.last_log_time = -math.inf
        self.last_ips_time = None
        self.speed_sample_position = None
        self.speed_sample_time = None
        self.last_prediction = None
        self.peer_ips_ready = False
        self.both_ready_since = None
        self.throttle_feedback = 0.0
        self.steering_feedback = 0.0

        prefix = f"/autodrive/f1tenth_{self.car_id}"
        self.steer_pub = self.create_publisher(Float32, f"{prefix}/steering_command", 10)
        self.throttle_pub = self.create_publisher(Float32, f"{prefix}/throttle_command", 10)
        self.mode_pub = self.create_publisher(String, f"{prefix}/dynssm_nmpc/mode", 10)
        self.model_status_pub = self.create_publisher(
            Bool, f"{prefix}/dynssm_nmpc/model_active", 10
        )
        self.lap_pub = self.create_publisher(Int32, f"{prefix}/nmpc/lap_count", 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, f"{prefix}/dynssm_nmpc/markers", 10
        )
        self.create_subscription(Point, f"{prefix}/ips", self.ips_cb, 10)
        self.create_subscription(Odometry, f"{prefix}/odom", self.odom_cb, 10)
        self.create_subscription(Imu, f"{prefix}/imu", self.imu_cb, 10)
        self.create_subscription(
            Float32, f"{prefix}/throttle", self.throttle_feedback_cb, 10
        )
        self.create_subscription(
            Float32, f"{prefix}/steering", self.steering_feedback_cb, 10
        )
        self.create_subscription(Bool, "/autodrive/startup_gate/open", self.gate_cb, 10)
        if self.wait_for_peer_ips:
            self.create_subscription(
                Point,
                f"/autodrive/f1tenth_{self.peer_car_id}/ips",
                self.peer_ips_cb,
                10,
            )
        self.create_timer(self.control_period, self.control_loop)
        self.create_timer(0.2, self.publish_markers)

        self.get_logger().info(
            f"Car {self.car_id} DynSSM + F1TENTH NMPC ready. "
            f"path={self.path_csv}, N={self.horizon}, dt={self.control_period:.2f}s, "
            f"speed={self.nominal_velocity:.2f}m/s, overtake={self.speed_multiplier:.2f}x, "
            f"pwm_to_sim_scale={self.pwm_to_sim_scale:.3f}, "
            f"dynssm_loaded={self.dynssm is not None}"
        )

    @staticmethod
    def _load_path(path):
        if not path.exists():
            raise FileNotFoundError(f"Frenet overtake path not found: {path}")
        points = []
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    point = (float(row["ego_x_m"]), float(row["ego_y_m"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if not points or np.linalg.norm(np.asarray(point) - points[-1]) > 1e-4:
                    points.append(point)
        if len(points) < 3:
            raise ValueError(f"Frenet path has fewer than three valid points: {path}")
        return np.asarray(points, dtype=float)

    def ips_cb(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        current = np.array([float(msg.x), float(msg.y)], dtype=float)
        if self.position is None:
            self.position = current
            self.start_position = current.copy()
            self.speed_sample_position = current.copy()
            self.speed_sample_time = now
            arc = self.normal_path.nearest_arc(current)
            _, headings = self.normal_path.sample(arc, 0.1, 2)
            self.heading = float(headings[0])
        else:
            displacement = current - self.position
            distance = float(np.linalg.norm(displacement))
            dt = now - self.last_ips_time if self.last_ips_time is not None else 0.0
            if distance > 0.005:
                measured_heading = math.atan2(displacement[1], displacement[0])
                delta_heading = wrap_angle(measured_heading - self.heading)
                self.heading = wrap_angle(self.heading + 0.45 * delta_heading)
                if dt >= 0.03:
                    measured_rate = delta_heading / dt
                    if abs(measured_rate) <= 15.0:
                        self.yaw_rate = 0.7 * self.yaw_rate + 0.3 * measured_rate
                self.total_distance += distance
            speed_dt = now - self.speed_sample_time
            if speed_dt >= 0.08:
                measured_speed = float(
                    np.linalg.norm(current - self.speed_sample_position) / speed_dt
                )
                if 0.0 <= measured_speed <= 12.0:
                    self.ips_speed = 0.7 * self.ips_speed + 0.3 * measured_speed
                self.speed_sample_position = current.copy()
                self.speed_sample_time = now
            self.speed = self.odom_speed if self.odom_speed > 0.05 else self.ips_speed
            self.position = current
            self._update_lap()
        self.last_ips_time = now

    def odom_cb(self, msg):
        # The AutoDRIVE bridge uses body Z as forward velocity.
        self.odom_speed = abs(float(msg.twist.twist.linear.z))
        # Unity body X points right, opposite lePAVD's positive-left lateral axis.
        self.lateral_speed = -float(msg.twist.twist.linear.x)
        self.speed = self.odom_speed if self.odom_speed > 0.05 else self.ips_speed

    def imu_cb(self, msg):
        measured = float(msg.angular_velocity.z)
        if math.isfinite(measured) and abs(measured) > 1e-4:
            self.yaw_rate = measured

    def throttle_feedback_cb(self, msg):
        self.throttle_feedback = float(msg.data) / max(self.pwm_to_sim_scale, 1e-6)

    def steering_feedback_cb(self, msg):
        self.steering_feedback = float(msg.data)

    def gate_cb(self, msg):
        if not self.wait_for_peer_ips:
            self.startup_gate_open = bool(msg.data)

    def peer_ips_cb(self, _msg):
        self.peer_ips_ready = True

    def _update_startup_gate(self, now):
        if not self.wait_for_peer_ips:
            return
        if self.position is None or not self.peer_ips_ready:
            self.both_ready_since = None
            self.startup_gate_open = False
            return
        if self.both_ready_since is None:
            self.both_ready_since = now
            self.get_logger().info(
                f"Car {self.car_id}: both IPS streams ready; holding for "
                f"{self.startup_hold_sec:.2f}s"
            )
        self.startup_gate_open = now - self.both_ready_since >= self.startup_hold_sec

    def _update_lap(self):
        if self.position is None or self.start_position is None or self.stopped:
            return
        distance_to_start = float(np.linalg.norm(self.position - self.start_position))
        if distance_to_start > self.lap_start_radius:
            self.left_start_zone = True
        if (
            self.left_start_zone
            and self.total_distance >= self.lap_min_distance
            and distance_to_start <= self.lap_start_radius
        ):
            self.lap_count += 1
            self.left_start_zone = False
            self.lap_pub.publish(Int32(data=self.lap_count))
            if self.stop_after_laps > 0 and self.lap_count >= self.stop_after_laps:
                self.stopped = True
                self.get_logger().info(f"Completed lap {self.lap_count}; stopping car 1")

    def _update_mode(self, now):
        if not self.enable_overtake_path:
            self.mode = self.NORMAL
            return
        if self.mode == self.NORMAL and not self.engagement_complete:
            if np.linalg.norm(self.position - self.c_start) <= self.c_start_radius:
                self.mode = self.ENGAGEMENT
                self.engagement_start_time = now
                self.get_logger().info("Reached c_start; NMPC switched to Frenet reference")
        elif self.mode == self.ENGAGEMENT:
            arc = self.engagement_path.nearest_arc(self.position)
            if (
                np.linalg.norm(self.position - self.c_end) <= self.c_end_radius
                and arc >= 0.92 * self.engagement_path.length
            ):
                self.mode = self.COMPLETE
                self.engagement_complete = True
                self.get_logger().info("Reached c_end; NMPC returned to centerline")

    def _publish_stop(self):
        self.steer_pub.publish(Float32(data=0.0))
        self.throttle_pub.publish(Float32(data=0.0))
        self.previous_control[:] = 0.0

    def control_loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        self._update_startup_gate(now)
        if not self.startup_gate_open or self.position is None or self.heading is None:
            self._publish_stop()
            return
        if self.stopped:
            self._publish_stop()
            return
        if (
            self.dynssm is not None
            and not self.dynssm_active
            and self.speed >= self.dynssm_min_adaptation_speed
        ):
            self.model_history.append(
                [
                    self.speed,
                    self.lateral_speed,
                    self.yaw_rate,
                    self.throttle_feedback,
                    self.steering_feedback,
                    self.previous_control[0],
                    self.previous_control[1],
                ]
            )
            if len(self.model_history) == self.dynssm.horizon:
                learned = self.dynssm.infer_parameters(np.asarray(self.model_history))
                self.nmpc = NominalLePAVDNMPC(
                    horizon=self.horizon,
                    sample_time=self.control_period,
                    steering_bounds=(-self.max_steering, self.max_steering),
                    max_steering_rate=self.max_steering_rate,
                    model_parameters=LePAVDParameters(**learned),
                    kinematic_wheelbase=self.model_wheelbase,
                )
                self.previous_control[:] = 0.0
                self.dynssm_active = True
                summary = ", ".join(
                    f"{name}={learned[name]:.5g}"
                    for name in ("Bf", "Df", "Br", "Dr", "Cm1", "Iz")
                )
                self.get_logger().info(f"DynSSM model ACTIVE; NMPC rebuilt: {summary}")
        self.model_status_pub.publish(Bool(data=self.dynssm_active))
        self._update_mode(now)
        path = self.engagement_path if self.mode == self.ENGAGEMENT else self.normal_path
        overtake_speed_active = (
            self.mode == self.ENGAGEMENT
            and self.engagement_start_time is not None
            and now - self.engagement_start_time >= self.speed_delay
        )
        desired_speed = self.nominal_velocity * (
            self.speed_multiplier if overtake_speed_active else 1.0
        )
        points, headings = path.sample(
            path.nearest_arc(self.position),
            desired_speed * self.control_period,
            self.horizon + 1,
        )
        headings += round((self.heading - headings[0]) / (2.0 * math.pi)) * 2.0 * math.pi
        reference = np.zeros((6, self.horizon + 1), dtype=float)
        reference[0:2, :] = points.T
        reference[2, :] = headings
        reference[3, :] = desired_speed
        reference[5, :-1] = np.diff(headings) / self.control_period
        reference[5, -1] = reference[5, -2]
        state = np.array(
            [*self.position, self.heading, self.speed, self.lateral_speed, self.yaw_rate],
            dtype=float,
        )
        try:
            control, self.last_prediction, objective = self.nmpc.solve(
                state, reference, self.previous_control
            )
        except Exception as exc:
            self.get_logger().warning(
                f"NMPC solve missed; holding previous command for this cycle: {exc}"
            )
            throttle_limit = self.max_sim_throttle * (
                self.speed_multiplier if overtake_speed_active else 1.0
            )
            held_throttle = float(
                np.clip(
                    self.previous_control[0] * self.pwm_to_sim_scale,
                    -0.1,
                    throttle_limit,
                )
            )
            self.throttle_pub.publish(Float32(data=held_throttle))
            self.steer_pub.publish(Float32(data=float(self.previous_control[1])))
            return
        self.previous_control = control
        throttle_limit = self.max_sim_throttle * (
            self.speed_multiplier if overtake_speed_active else 1.0
        )
        if abs(float(control[1])) >= self.corner_steering:
            throttle_limit *= self.corner_throttle_scale
        throttle = float(
            np.clip(control[0] * self.pwm_to_sim_scale, -0.1, throttle_limit)
        )
        steering = float(control[1])
        self.throttle_pub.publish(Float32(data=throttle))
        self.steer_pub.publish(Float32(data=steering))
        self.mode_pub.publish(String(data=self.mode))
        if now - self.last_log_time >= 1.0:
            self.last_log_time = now
            self.get_logger().info(
                f"mode={self.mode} speed={self.speed:.2f}/{desired_speed:.2f}m/s "
                f"model_u=({control[0]:.3f},{steering:.3f}) "
                f"sim_u=({throttle:.3f},{steering:.3f}) J={objective:.2f}"
            )

    def publish_markers(self):
        if self.last_prediction is None:
            return
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "dynssm_nmpc_prediction"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.08
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.95, 0.25, 0.05, 1.0
        for index in range(self.last_prediction.shape[1]):
            marker.points.append(
                Point(
                    x=float(self.last_prediction[0, index]),
                    y=float(self.last_prediction[1, index]),
                    z=0.12,
                )
            )
        self.marker_pub.publish(MarkerArray(markers=[marker]))


def main(args=None):
    rclpy.init(args=args)
    node = DynSSMNMPCController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._publish_stop()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

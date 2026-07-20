#!/usr/bin/env python3
import os
import sys
import math
import yaml
import pickle
import numpy as np
import csv

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from sensor_msgs.msg import Imu

import torch
from scipy.interpolate import CubicSpline

# ==================== LE-PAVD PATH ====================
LEPAVD_ROOT = os.path.expanduser("~/LE-PAVD-main")
if LEPAVD_ROOT not in sys.path:
    sys.path.append(LEPAVD_ROOT)


def catmull_rom_chain(pts, num_points=200):
    p = np.vstack([pts[0], pts, pts[-1]])
    t = np.linspace(0, 1, len(p))
    cs_x = CubicSpline(t, p[:, 0], bc_type="clamped")
    cs_y = CubicSpline(t, p[:, 1], bc_type="clamped")
    ts = np.linspace(0, 1, num_points)
    return np.vstack([cs_x(ts), cs_y(ts)]).T


def angle_wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class LEPAVDClosedLoop(Node):
    """
    IPS/IMU anchored closed-loop, but uses LE-PAVD to hallucinate N steps ahead.
    Crucial fix: during hallucination we ADVANCE waypoint index so predictions
    actually lead to each waypoint through corners.
    """

    def __init__(self):
        super().__init__("lepavd_predicted_pose_pure_pursuit")
        
        # ---------------- CSV Logging (LE-PAVD format) ----------------
        log_path = os.path.expanduser("~/lepavd_closed_loop.csv")
        self.csv_file = open(log_path, "w", newline="")
        self.logger = csv.writer(self.csv_file)

        # EXACT same header as training data
        self.logger.writerow([
            "vx", "vy", "yaw_rate",
            "throttle_fb", "steering_fb",
            "throttle_cmd", "steering_cmd"
        ])

        self.get_logger().info(f"Logging closed-loop data to {log_path}")


        # ---------------- Device ----------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------- Paths ----------------
        self.cfg_path = os.path.expanduser("~/LE-PAVD-main/cfgs/model/lePAVD_iac.yaml")
        self.ckpt_path = os.path.expanduser(
            "~/LE-PAVD-main/output/lePAVD_iac/lepavd_2.0ms_12/epoch_2993.pth"
        )
        self.scaler_path = os.path.join(os.path.dirname(self.ckpt_path), "scaler.pkl")

        if not os.path.exists(self.cfg_path):
            raise FileNotFoundError(self.cfg_path)
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(self.ckpt_path)
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Missing scaler.pkl at {self.scaler_path}")

        # ---------------- Load scaler ----------------
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # ---------------- Load model ----------------
        from model.models_lePAVD import string_to_lePAVD

        with open(self.cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.H = int(cfg["MODEL"]["HORIZON"])  # should be 12
        self.feature_dim = 7

        self.model = string_to_lePAVD["lePAVD_iac"](cfg, csv_path=None)
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.h = self.model.init_hidden(1).to(self.device)

        self.get_logger().info(
            f"LE-PAVD loaded on {self.device} | H={self.H} | scaler={self.scaler_path}"
        )

        # ---------------- Buffers (raw + normalized) ----------------
        self.raw_buffer = torch.zeros((1, self.H, self.feature_dim), device=self.device)
        self.norm_buffer = torch.zeros((1, self.H, self.feature_dim), device=self.device)
        self.warm_started = False

        # ---------------- Path / waypoints ----------------
        raw_wpts = np.array(
            [
                [0.25, -5.25],
                [-0.61, -6.57],
                [-1.92, -6.81],
                [-2.79, -5.67],
                [-2.17, -3.95],
                [-1.48, -2.55],
                [-1.10, -2.03],
                [-2.25, 3.66],
                [-1.59, 5.02],
                [-0.15, 5.08],
                [0.55, 3.94],
                [0.75, 4.13],
                [0.98, 2.54],
            ],
            dtype=np.float32,
        )
        self.path = catmull_rom_chain(raw_wpts, num_points=200)
        self.idx = 0
        self.idx_initialized = False

        # ---------------- Pure Pursuit params ----------------
        self.lookahead = 1.0
        self.wheelbase = 0.30
        self.max_steer = math.radians(90)

        # ---------------- Control timing ----------------
        self.dt = 0.10
        self.target_speed = 0.40

        # ---------------- State estimates from sensors ----------------
        self.x_hat = None
        self.y_hat = None
        self.yaw_hat = 0.0

        self.vx_est = 0.0
        self.vy_est = 0.0
        self.yaw_rate_est = 0.0

        self.throttle_fb = 0.0
        self.steering_fb = 0.0

        self.prev_ips = None
        self.prev_t = None

        # ---------------- Publishers ----------------
        self.steer_pub = self.create_publisher(
            Float32, "/autodrive/f1tenth_1/steering_command", 10
        )
        self.throttle_pub = self.create_publisher(
            Float32, "/autodrive/f1tenth_1/throttle_command", 10
        )

        # ---------------- Subscribers ----------------
        self.create_subscription(Point, "/autodrive/f1tenth_1/ips", self.ips_cb, 10)
        self.create_subscription(Imu, "/autodrive/f1tenth_1/imu", self.imu_cb, 10)
        self.create_subscription(Float32, "/autodrive/f1tenth_1/steering", self.steer_fb_cb, 10)
        self.create_subscription(Float32, "/autodrive/f1tenth_1/throttle", self.throttle_fb_cb, 10)

        # ---------------- Timer ----------------
        self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(
            "Running: N-step LE-PAVD hallucination with waypoint-advancing Pure Pursuit (IPS anchored)"
        )

    # ==================== Callbacks ====================

    def ips_cb(self, msg: Point):
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.x_hat is None:
            self.x_hat = float(msg.x)
            self.y_hat = float(msg.y)
            self.prev_ips = (self.x_hat, self.y_hat)
            self.prev_t = now

            dists = np.linalg.norm(self.path - np.array([self.x_hat, self.y_hat]), axis=1)
            self.idx = int(np.argmin(dists))
            self.idx_initialized = True

            self.get_logger().info(
                f"IPS init: x={self.x_hat:.3f}, y={self.y_hat:.3f}, idx={self.idx}"
            )
            return

        # anchor measurement
        self.x_hat = float(msg.x)
        self.y_hat = float(msg.y)

        if self.prev_t is not None:
            dt = now - self.prev_t
            if dt > 1e-6:
                dx = self.x_hat - self.prev_ips[0]
                dy = self.y_hat - self.prev_ips[1]
                self.vx_est = dx / dt
                self.vy_est = dy / dt

                speed = math.hypot(self.vx_est, self.vy_est)
                if speed > 0.05:
                    self.yaw_hat = math.atan2(self.vy_est, self.vx_est)

        self.prev_ips = (self.x_hat, self.y_hat)
        self.prev_t = now

    def imu_cb(self, msg: Imu):
        self.yaw_rate_est = float(msg.angular_velocity.z)

    def steer_fb_cb(self, msg: Float32):
        self.steering_fb = float(msg.data)

    def throttle_fb_cb(self, msg: Float32):
        self.throttle_fb = float(msg.data)

    # ==================== Pure Pursuit ====================

    def pursue(self, pos_xy, yaw, path, idx):
        """
        Compute steering using a provided pose (pos_xy, yaw).
        Returns: (delta, new_idx, target_xy)
        """
        x, y = pos_xy
        N = len(path)

        # find lookahead target by walking forward from idx
        # (bounded loop to avoid pathological infinite loops)
        for _ in range(N):
            tx, ty = path[idx % N]
            if math.hypot(tx - x, ty - y) > self.lookahead:
                break
            idx = (idx + 1) % N

        tx, ty = path[idx % N]
        dx, dy = tx - x, ty - y

        # transform to vehicle frame using yaw
        cos_y = math.cos(-yaw)
        sin_y = math.sin(-yaw)
        xv = dx * cos_y - dy * sin_y
        yv = dx * sin_y + dy * cos_y

        alpha = math.atan2(yv, xv)
        delta = math.atan2(4.0 * self.wheelbase * math.sin(alpha), self.lookahead)
        delta = float(np.clip(delta, -self.max_steer, self.max_steer))
        return delta, idx, (tx, ty)

    # ==================== Normalization ====================

    def normalize_feat(self, feat_np: np.ndarray) -> np.ndarray:
        """
        feat_np shape (7,) -> (7,)
        uses training scaler
        """
        return self.scaler.transform(feat_np.reshape(1, -1)).reshape(-1).astype(np.float32)

    # ==================== N-step hallucination (waypoint-aware) ====================

    def rollout_hallucinated_trajectory(
        self,
        N,
        x0,
        y0,
        yaw0,
        vx0,
        vy0,
        yawrate0,
        throttle_cmd,
    ):
        """
        Roll out N steps using the model recursively.
        Returns list of tuples: [(x, y, yaw, idx_sim), ...] length N.

        Important:
        - Uses COPIES of buffers + hidden state so self.h is not advanced.
        - During rollout we advance a simulated waypoint index idx_sim so the rollout
          actually leads through corners (this fixes "go straight into wall").
        - Assumes vx,vy are in GLOBAL frame (consistent with IPS finite-diff).
        """
        pts = []  # ALWAYS return a list, never None

        #N = int(max(1, min(int(N), int(self.H))))
        N = 6
        raw_sim = self.raw_buffer.clone()
        norm_sim = self.norm_buffer.clone()
        h_sim = self.h.clone()

        xk, yk, yawk = float(x0), float(y0), float(yaw0)
        vxk, vyk, wrk = float(vx0), float(vy0), float(yawrate0)

        # waypoint index evolves during rollout
        idx_sim = int(self.idx)

        # assume future feedback ~ commanded / held (simple but stable)
        throttle_fb_sim = float(self.throttle_fb)
        steering_fb_sim = float(self.steering_fb)

        for _ in range(N):
            # compute steering for this simulated pose AND advance idx_sim
            steering_k, idx_sim, _ = self.pursue((xk, yk), yawk, self.path, idx_sim)

            feat_sim = np.array(
                [
                    vxk,
                    vyk,
                    wrk,
                    throttle_fb_sim,
                    steering_fb_sim,
                    float(throttle_cmd),
                    float(steering_k),
                ],
                dtype=np.float32,
            )
            feat_sim_norm = self.normalize_feat(feat_sim)

            feat_t = torch.from_numpy(feat_sim).to(self.device)
            feat_norm_t = torch.from_numpy(feat_sim_norm).to(self.device)

            raw_sim[:, :-1, :] = raw_sim[:, 1:, :].clone()
            raw_sim[:, -1, :] = feat_t
            norm_sim[:, :-1, :] = norm_sim[:, 1:, :].clone()
            norm_sim[:, -1, :] = feat_norm_t

            with torch.no_grad():
                out, h_sim, _ = self.model(raw_sim, norm_sim, h_sim)

            vx_next, vy_next, yawrate_next = out.squeeze().tolist()

            xk = xk + float(vx_next) * self.dt
            yk = yk + float(vy_next) * self.dt
            yawk = angle_wrap(yawk + float(yawrate_next) * self.dt)

            vxk, vyk, wrk = float(vx_next), float(vy_next), float(yawrate_next)

            # update assumed feedback
            steering_fb_sim = float(steering_k)
            throttle_fb_sim = float(throttle_cmd)

            pts.append((xk, yk, yawk, int(idx_sim)))

        return pts

    # ==================== Control Loop ====================

    def control_loop(self):
        if self.x_hat is None or not self.idx_initialized:
            return

        throttle_cmd = float(self.target_speed)

        # 1) Base steering from CURRENT measured pose (for feature construction)
        steering_base, idx_tmp, _ = self.pursue(
            (self.x_hat, self.y_hat), self.yaw_hat, self.path, self.idx
        )

        # 2) Build feature vector (MUST match training order)
        feat = np.array(
            [
                self.vx_est,
                self.vy_est,
                self.yaw_rate_est,
                self.throttle_fb,
                self.steering_fb,
                throttle_cmd,
                steering_base,
            ],
            dtype=np.float32,
        )

        feat_norm = self.normalize_feat(feat)

        feat_t = torch.from_numpy(feat).to(self.device)
        feat_norm_t = torch.from_numpy(feat_norm).to(self.device)

        # 3) Warm start buffers so GRU/BN see realistic data
        if not self.warm_started:
            self.raw_buffer[:] = feat_t.view(1, 1, -1).repeat(1, self.H, 1)
            self.norm_buffer[:] = feat_norm_t.view(1, 1, -1).repeat(1, self.H, 1)
            self.warm_started = True

            # take the base idx update once so we start "facing" the path correctly
            self.idx = idx_tmp

            self.steer_pub.publish(Float32(data=float(steering_base)))
            self.throttle_pub.publish(Float32(data=float(throttle_cmd)))
            return

        # 4) Roll real buffers
        self.raw_buffer[:, :-1, :] = self.raw_buffer[:, 1:, :].clone()
        self.raw_buffer[:, -1, :] = feat_t

        self.norm_buffer[:, :-1, :] = self.norm_buffer[:, 1:, :].clone()
        self.norm_buffer[:, -1, :] = feat_norm_t

        # 5) Update hidden state ONCE using real history
        with torch.no_grad():
            out_now, self.h, _ = self.model(self.raw_buffer, self.norm_buffer, self.h)

        vx_pred, vy_pred, yaw_rate_pred = out_now.squeeze().tolist()

        # 6) Waypoint-aware N-step hallucination
        N = min(12, self.H)
        hallucinated = self.rollout_hallucinated_trajectory(
            N=N,
            x0=self.x_hat,
            y0=self.y_hat,
            yaw0=self.yaw_hat,
            vx0=vx_pred,
            vy0=vy_pred,
            yawrate0=yaw_rate_pred,
            throttle_cmd=throttle_cmd,
        )

        # safety fallback (shouldn't trigger now, but keeps you running no matter what)
        if hallucinated is None or len(hallucinated) == 0:
            x1 = self.x_hat + float(vx_pred) * self.dt
            y1 = self.y_hat + float(vy_pred) * self.dt
            yaw1 = angle_wrap(self.yaw_hat + float(yaw_rate_pred) * self.dt)
            hallucinated = [(x1, y1, yaw1, int(self.idx))]

        # 7) Choose a point in the hallucinated horizon
        #    (0.66 works fine; you can tune this)
        #k_target = max(0, min(len(hallucinated) - 1, int(0.66 * len(hallucinated))))
        k_target = int(0.30 * len(hallucinated))
        x_t, y_t, yaw_t, idx_t = hallucinated[k_target]

        # 8) Use the HALLUCINATED waypoint index to compute steering from CURRENT measured pose.
        #    This is the key “lead to each waypoint” fix.
        steering_cmd, idx_cmd, _ = self.pursue(
            (x_t, y_t), yaw_t, self.path, int(idx_t)
        )

        self.idx = idx_cmd

        # 9) Publish
        self.steer_pub.publish(Float32(data=float(steering_cmd)))
        self.throttle_pub.publish(Float32(data=float(throttle_cmd)))
        
        # ---------------- CSV LOGGING (EXACT FEATURE MATCH) ----------------
        self.logger.writerow([
            self.vx_est,
            self.vy_est,
            self.yaw_rate_est,
            self.throttle_fb,
            self.steering_fb,
            throttle_cmd,
            steering_cmd
        ])
    
    def destroy_node(self):
        try:
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()



def main(args=None):
    rclpy.init(args=args)
    node = LEPAVDClosedLoop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from sensor_msgs.msg import Imu
import numpy as np
import math
import csv
import os
from scipy.interpolate import CubicSpline

def catmull_rom_chain(pts, num_points=200):
    p = np.vstack([pts[0], pts, pts[-1]])
    t = np.linspace(0, 1, len(p))
    cs_x = CubicSpline(t, p[:, 0], bc_type='clamped')
    cs_y = CubicSpline(t, p[:, 1], bc_type='clamped')
    ts = np.linspace(0, 1, num_points)
    return np.vstack([cs_x(ts), cs_y(ts)]).T


class PurePursuitF1Tenth(Node):
    def __init__(self):
        super().__init__('pure_pursuit_f1tenth_1')

        #log_path = os.path.expanduser("~/lepavd_training_data.csv")
        log_path = os.path.expanduser("~/aaron_10laps.csv")
        self.csv_file = open(log_path, "w", newline='')
        self.logger = csv.writer(self.csv_file)

        self.logger.writerow([
            "t",
            "x","y",
            "vx", "vy", "yaw_rate",
            "throttle_fb", "steering_fb",
            "throttle_cmd", "steering_cmd"
        ])

        raw = np.array([
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
        ])
        self.path = catmull_rom_chain(raw, num_points=200)
        self.idx = 0
        self.idx_initialized = False  

        self.lookahead_distance = 2.0
        self.wheelbase = 0.30
        self.max_steer = math.radians(90)
        self.target_speed = 0.40

        self.steer_pub = self.create_publisher(Float32, "/autodrive/f1tenth_1/steering_command", 10)
        self.throttle_pub = self.create_publisher(Float32, "/autodrive/f1tenth_1/throttle_command", 10)
        self.create_subscription(Point, "/autodrive/f1tenth_1/ips", self.ips_cb, 10)
        self.create_subscription(Imu, "/autodrive/f1tenth_1/imu", self.imu_cb, 10)
        self.create_subscription(Float32, "/autodrive/f1tenth_1/steering", self.steer_fb_cb, 10)
        self.create_subscription(Float32, "/autodrive/f1tenth_1/throttle", self.throttle_fb_cb, 10)

        self.pos = None         
        self.prev_pos = None    
        self.prev_time = None   
        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0
        self.steering_fb = 0.0
        self.throttle_fb = 0.0
        
        self.create_timer(0.1, self.control_loop)


    def ips_cb(self, msg: Point):
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.pos is None:
            self.pos = (msg.x, msg.y)
            self.prev_pos = (msg.x, msg.y)
            self.prev_time = now

            dists = np.linalg.norm(self.path - np.array(self.pos), axis=1)
            self.idx = int(np.argmin(dists))
            self.idx_initialized = True
            self.get_logger().info(f"Initialized idx to {self.idx} based on IPS start")
            return

        if self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 1e-6:
                dx = msg.x - self.pos[0]
                dy = msg.y - self.pos[1]
                self.vx = dx / dt
                self.vy = dy / dt

            self.prev_pos = self.pos

        self.pos = (msg.x, msg.y)
        self.prev_time = now

    def imu_cb(self, msg: Imu):
        self.yaw_rate = msg.angular_velocity.z

    def steer_fb_cb(self, msg: Float32):
        self.steering_fb = float(msg.data)

    def throttle_fb_cb(self, msg: Float32):
        self.throttle_fb = float(msg.data)

    def control_loop(self):
        if self.pos is None or self.prev_pos is None or not self.idx_initialized:
            return

        steering_cmd, new_idx = self.pursue(self.pos, self.prev_pos, self.path, self.idx)
        self.idx = new_idx
        throttle_cmd = float(self.target_speed)

        self.steer_pub.publish(Float32(data=steering_cmd))
        self.throttle_pub.publish(Float32(data=throttle_cmd))
        
        t = self.get_clock().now().nanoseconds * 1e-9

        self.logger.writerow([
            t,
            float(self.pos[0]), float(self.pos[1]),
            self.vx, self.vy, self.yaw_rate,
            self.throttle_fb, self.steering_fb,
            throttle_cmd, steering_cmd
        ])

    def pursue(self, pos, prev, path, idx):
        x, y = pos
        xp, yp = prev

        dx = self.pos[0] - self.prev_pos[0]
        dy = self.pos[1] - self.prev_pos[1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            tx, ty = self.path[self.idx]
            yaw = math.atan2(ty - self.pos[1], tx - self.pos[0])
        else:
            yaw = math.atan2(dy, dx)

        N = len(path)
        while True:
            tx, ty = path[idx % N]
            if math.hypot(tx - x, ty - y) > self.lookahead_distance:
                break
            idx += 1
            if idx >= N:
                idx = 0

        dx, dy = tx - x, ty - y
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)

        xv = dx * cos_yaw - dy * sin_yaw
        yv = dx * sin_yaw + dy * cos_yaw

        alpha = math.atan2(yv, xv)
        delta = math.atan2(4 * self.wheelbase * math.sin(alpha),self.lookahead_distance)

        return max(-self.max_steer, min(self.max_steer, delta)), idx

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitF1Tenth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

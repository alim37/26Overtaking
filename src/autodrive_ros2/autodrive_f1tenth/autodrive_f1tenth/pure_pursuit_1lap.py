#!/usr/bin/env python3

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from scipy.interpolate import CubicSpline
from std_msgs.msg import Float32


def catmull_rom_chain(pts, num_points=100):
    p = np.vstack([pts[0], pts, pts[-1]])
    t = np.linspace(0, 1, len(p))
    cs_x = CubicSpline(t, p[:, 0], bc_type="clamped")
    cs_y = CubicSpline(t, p[:, 1], bc_type="clamped")
    ts = np.linspace(0, 1, num_points)
    return np.vstack([cs_x(ts), cs_y(ts)]).T


class DualPurePursuitOneLap(Node):
    def __init__(self):
        super().__init__("dual_pure_pursuit_1lap")

        raw1 = np.array([
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
        raw2 = np.array([
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

        self.spline_car1 = catmull_rom_chain(raw1, num_points=200)
        self.spline_car2 = catmull_rom_chain(raw2, num_points=200)
        self.idx1 = 0
        self.idx2 = 0

        self.lookahead_distance = 1.0
        self.wheelbase = 0.3
        self.max_steering_angle = math.radians(90)
        self.target_speed = 0.1
        self.start_delay_sec = 5.0
        self.car2_extra_run_sec = 2.0
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        self.prev1 = None
        self.pos1 = None
        self.prev2 = None
        self.pos2 = None

        self.car1_started = False
        self.car2_started = False
        self.car1_lap_complete = False
        self.car2_lap_complete = False
        self.car2_stop_time = None
        self.car1_stop_sent = False
        self.car2_stop_sent = False

        self.steer_pub1 = self.create_publisher(Float32, "/autodrive/f1tenth_1/steering_command", 10)
        self.throttle_pub1 = self.create_publisher(Float32, "/autodrive/f1tenth_1/throttle_command", 10)
        self.steer_pub2 = self.create_publisher(Float32, "/autodrive/f1tenth_2/steering_command", 10)
        self.throttle_pub2 = self.create_publisher(Float32, "/autodrive/f1tenth_2/throttle_command", 10)

        self.create_subscription(Point, "/autodrive/f1tenth_1/ips", self.ips_cb1, 10)
        self.create_subscription(Point, "/autodrive/f1tenth_2/ips", self.ips_cb2, 10)

        self.create_timer(0.1, self.control_loop)

    def ips_cb1(self, msg: Point):
        self.prev1 = self.pos1
        self.pos1 = (msg.x, msg.y)

    def ips_cb2(self, msg: Point):
        self.prev2 = self.pos2
        self.pos2 = (msg.x, msg.y)

    def publish_stop(self, car: int):
        getattr(self, f"steer_pub{car}").publish(Float32(data=0.0))
        getattr(self, f"throttle_pub{car}").publish(Float32(data=0.0))

    def control_loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.start_time < self.start_delay_sec:
            self.publish_stop(1)
            self.publish_stop(2)
            return

        self.control_car1()
        self.control_car2(now)

    def control_car1(self):
        if self.pos1 is None or self.prev1 is None:
            return

        if self.car1_lap_complete:
            if not self.car1_stop_sent:
                self.publish_stop(1)
                self.car1_stop_sent = True
                self.get_logger().info("Car1 first lap complete, stopping vehicle.")
            return

        delta, new_idx = self.pursue(self.pos1, self.prev1, self.spline_car1, self.idx1)
        if not self.car1_started:
            self.car1_started = True
            self.get_logger().info("Start delay complete, beginning pure pursuit for car1.")
        elif new_idx < self.idx1:
            self.car1_lap_complete = True
            self.publish_stop(1)
            self.car1_stop_sent = True
            self.get_logger().info("Car1 first lap complete, stopping vehicle.")
            return

        self.idx1 = new_idx
        self.steer_pub1.publish(Float32(data=delta))
        self.throttle_pub1.publish(Float32(data=self.target_speed))

    def control_car2(self, now: float):
        if self.pos2 is None or self.prev2 is None:
            return

        if self.car2_lap_complete:
            if now >= self.car2_stop_time:
                if not self.car2_stop_sent:
                    self.publish_stop(2)
                    self.car2_stop_sent = True
                    self.get_logger().info("Car2 extra 2 seconds complete, stopping vehicle.")
                return

            delta, new_idx = self.pursue(self.pos2, self.prev2, self.spline_car2, self.idx2)
            self.idx2 = new_idx
            self.steer_pub2.publish(Float32(data=delta))
            self.throttle_pub2.publish(Float32(data=self.target_speed))
            return

        delta, new_idx = self.pursue(self.pos2, self.prev2, self.spline_car2, self.idx2)
        if not self.car2_started:
            self.car2_started = True
            self.get_logger().info("Start delay complete, beginning pure pursuit for car2.")
        elif new_idx < self.idx2:
            self.car2_lap_complete = True
            self.car2_stop_time = now + self.car2_extra_run_sec
            self.get_logger().info("Car2 first lap complete, continuing for 2 more seconds.")

        self.idx2 = new_idx
        self.steer_pub2.publish(Float32(data=delta))
        self.throttle_pub2.publish(Float32(data=self.target_speed))

    def pursue(self, pos, prev, path, idx):
        x, y = pos
        xp, yp = prev

        yaw = math.atan2(y - yp, x - xp)
        n_points = len(path)

        while True:
            tx, ty = path[idx % n_points]
            if math.hypot(tx - x, ty - y) > self.lookahead_distance:
                break
            idx += 1
            if idx >= n_points:
                idx = 0
                break

        dx, dy = tx - x, ty - y
        xv = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        yv = dx * math.sin(-yaw) + dy * math.cos(-yaw)

        alpha = math.atan2(yv, xv)
        delta = math.atan2(2 * self.wheelbase * math.sin(alpha), self.lookahead_distance)

        return max(-self.max_steering_angle, min(self.max_steering_angle, delta)), idx


def main(args=None):
    rclpy.init(args=args)
    node = DualPurePursuitOneLap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

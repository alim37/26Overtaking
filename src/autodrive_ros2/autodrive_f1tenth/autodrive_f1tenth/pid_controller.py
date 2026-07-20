#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import math

class DualWallFollowerNode(Node):
    def __init__(self):
        super().__init__('dual_wall_follower')

        # PID gains (shared)
        self.kp, self.ki, self.kd = 3.0, 0.001, 0.10

        # State for vehicle 1
        self.prev_err1  = 0.0
        self.int_err1   = 0.0
        self._setup_vehicle(
            namespace='f1tenth_1',
            throttle_topic='/autodrive/f1tenth_1/throttle_command',
            steer_topic   ='/autodrive/f1tenth_1/steering_command',
            scan_topic    ='/autodrive/f1tenth_1/lidar',
            idx           =1
        )

        # State for vehicle 2
        self.prev_err2  = 0.0
        self.int_err2   = 0.0
        self._setup_vehicle(
            namespace='f1tenth_2',
            throttle_topic='/autodrive/f1tenth_2/throttle_command',
            steer_topic   ='/autodrive/f1tenth_2/steering_command',
            scan_topic    ='/autodrive/f1tenth_2/lidar',
            idx           =2
        )

        self.get_logger().info('DualWallFollowerNode initialized for both cars')

    def _setup_vehicle(self, namespace, throttle_topic, steer_topic, scan_topic, idx):
        # publishers
        setattr(self, f'steer_pub{idx}',
            self.create_publisher(Float32, steer_topic, 10))
        setattr(self, f'throttle_pub{idx}',
            self.create_publisher(Float32, throttle_topic, 10))

        # QoS for LaserScan (best effort)
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # subscriber
        self.create_subscription(
            LaserScan,
            scan_topic,
            lambda msg, i=idx: self.scan_callback(msg, i),
            qos
        )

    def scan_callback(self, scan: LaserScan, idx: int):
        # pick state variables by idx
        if idx == 1:
            prev_err = 'prev_err1'; int_err = 'int_err1'
        else:
            prev_err = 'prev_err2'; int_err = 'int_err2'

        err = self.calculate_error(scan)
        # integral & derivative
        setattr(self, int_err, getattr(self, int_err) + err)
        deriv = err - getattr(self, prev_err)
        setattr(self, prev_err, err)

        # PID output
        raw = -(self.kp * err + self.ki * getattr(self, int_err) + self.kd * deriv)
        control = max(-0.5, min(0.5, raw))

        # throttle logic
        speed = 1.5 if abs(control) <= 10*math.pi/180 else 0.5

        # publish
        steer_msg = Float32(data=control)
        thr_msg   = Float32(data=speed)
        getattr(self, f'steer_pub{idx}').publish(steer_msg)
        getattr(self, f'throttle_pub{idx}').publish(thr_msg)

        self.get_logger().info(
            f'[V{idx}] err={err:.3f} → steer={control:.3f}, throttle={speed:.3f}'
        )

    def calculate_error(self, scan: LaserScan) -> float:
        a = self.get_range(scan, 45)
        b = self.get_range(scan, 90)
        α = math.atan2(
            a * math.cos(math.radians(45)) - b,
            a * math.sin(math.radians(45))
        )
        D  = b * math.cos(α)
        D1 = D + math.sin(α)
        return 1.0 - D1

    def get_range(self, scan: LaserScan, deg: float) -> float:
        a = math.radians(deg)
        idx = int((a - scan.angle_min) / scan.angle_increment)
        return scan.ranges[idx] if 0 <= idx < len(scan.ranges) else float('inf')

def main():
    rclpy.init()
    node = DualWallFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()

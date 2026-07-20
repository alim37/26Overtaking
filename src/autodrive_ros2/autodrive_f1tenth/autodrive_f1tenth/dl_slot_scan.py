#!/usr/bin/env python3

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan, PointCloud2

from autodrive_f1tenth.dl_slot import make_pointcloud2, quaternion_to_yaw


class DLSLOTScanNode(Node):
    """
    Heading-only scan accumulator.

    This node ignores IPS translation completely and only uses IMU heading
    to rotate each LiDAR scan into a fixed frame centered at the origin.
    It is useful for debugging whether position updates are distorting the map.
    """

    def __init__(self) -> None:
        super().__init__("dl_slot_scan")

        self.declare_parameter("scan_topic", "/autodrive/f1tenth_1/lidar")
        self.declare_parameter("imu_topic", "/autodrive/f1tenth_1/imu")
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("static_cloud_topic", "/autodrive/f1tenth_1/dl_slot_scan/static_cloud")
        self.declare_parameter("current_cloud_topic", "/autodrive/f1tenth_1/dl_slot_scan/current_cloud")
        self.declare_parameter("min_range", 0.05)
        self.declare_parameter("max_range", 15.0)
        self.declare_parameter("voxel_size", 0.10)
        self.declare_parameter("static_promotion_hits", 1)
        self.declare_parameter("publish_rate_hz", 20.0)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.static_cloud_topic = str(self.get_parameter("static_cloud_topic").value)
        self.current_cloud_topic = str(self.get_parameter("current_cloud_topic").value)
        self.min_range = float(self.get_parameter("min_range").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.voxel_size = float(self.get_parameter("voxel_size").value)
        self.static_promotion_hits = int(self.get_parameter("static_promotion_hits").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.yaw: float | None = None
        self.static_voxels: dict[tuple[int, int], int] = {}
        self.latest_current_points: list[tuple[float, float, float]] = []

        self.static_pub = self.create_publisher(PointCloud2, self.static_cloud_topic, 10)
        self.current_pub = self.create_publisher(PointCloud2, self.current_cloud_topic, 10)

        self.create_subscription(Imu, self.imu_topic, self.imu_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.create_timer(1.0 / max(self.publish_rate_hz, 1e-3), self.publish_outputs)

        self.get_logger().info(
            "DL-SLOT scan-only node ready. "
            f"scan={self.scan_topic}, imu={self.imu_topic}, ips=disabled"
        )

    def imu_cb(self, msg: Imu) -> None:
        self.yaw = quaternion_to_yaw(
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        )

    def scan_cb(self, msg: LaserScan) -> None:
        yaw = 0.0 if self.yaw is None else self.yaw
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        angle = msg.angle_min
        current_points: list[tuple[float, float, float]] = []
        for distance in msg.ranges:
            if math.isfinite(distance) and self.min_range <= distance <= self.max_range:
                local_x = distance * math.cos(angle)
                local_y = distance * math.sin(angle)

                # Rotate by heading only; keep the origin fixed with no IPS translation.
                world_x = local_x * cos_yaw - local_y * sin_yaw
                world_y = local_x * sin_yaw + local_y * cos_yaw
                current_points.append((world_x, world_y, 0.0))

                key = (
                    int(math.floor(world_x / self.voxel_size)),
                    int(math.floor(world_y / self.voxel_size)),
                )
                self.static_voxels[key] = self.static_voxels.get(key, 0) + 1

            angle += msg.angle_increment

        self.latest_current_points = current_points

    def publish_outputs(self) -> None:
        stamp = self.get_clock().now().to_msg()
        static_points = [
            ((ix + 0.5) * self.voxel_size, (iy + 0.5) * self.voxel_size, 0.0)
            for (ix, iy), hits in self.static_voxels.items()
            if hits >= self.static_promotion_hits
        ]

        self.static_pub.publish(make_pointcloud2(static_points, self.world_frame, stamp))
        self.current_pub.publish(make_pointcloud2(self.latest_current_points, self.world_frame, stamp))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DLSLOTScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

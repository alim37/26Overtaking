#!/usr/bin/env python3

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Point, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


class SlamToolboxBridge(Node):
    def __init__(self) -> None:
        super().__init__("slam_toolbox_bridge")

        self.declare_parameter("pose_topic", "/autodrive/f1tenth_1/ips")
        self.declare_parameter("imu_topic", "/autodrive/f1tenth_1/imu")
        self.declare_parameter("lidar_topic", "/autodrive/f1tenth_1/lidar")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/autodrive/f1tenth_1/odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom_1")
        self.declare_parameter("base_frame", "base_link_1")
        self.declare_parameter("lidar_frame", "lidar_1")
        self.declare_parameter("publish_static_lidar_tf", True)
        self.declare_parameter("publish_rate_hz", 100.0)
        self.declare_parameter("scan_stamp_lag_sec", 0.02)
        self.declare_parameter("lidar_x", 0.0)
        self.declare_parameter("lidar_y", 0.0)
        self.declare_parameter("lidar_z", 0.0)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.lidar_topic = str(self.get_parameter("lidar_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.lidar_frame = str(self.get_parameter("lidar_frame").value)
        self.publish_static_lidar_tf = bool(self.get_parameter("publish_static_lidar_tf").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.scan_stamp_lag_sec = float(self.get_parameter("scan_stamp_lag_sec").value)
        self.lidar_x = float(self.get_parameter("lidar_x").value)
        self.lidar_y = float(self.get_parameter("lidar_y").value)
        self.lidar_z = float(self.get_parameter("lidar_z").value)

        self.latest_position: tuple[float, float, float] | None = None
        self.latest_orientation = (0.0, 0.0, 0.0, 1.0)
        self.latest_yaw_rate = 0.0
        self.latest_linear_speed = 0.0
        self.prev_position: tuple[float, float, float] | None = None
        self.prev_pose_time_sec: float | None = None
        self.latest_tf_stamp = None

        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.scan_pub = self.create_publisher(LaserScan, self.scan_topic, qos_profile_sensor_data)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.create_subscription(Point, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(Imu, self.imu_topic, self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.lidar_topic, self.lidar_cb, qos_profile_sensor_data)
        self.create_timer(1.0 / max(self.publish_rate_hz, 1.0), self.publish_odometry_and_tf)

        if self.publish_static_lidar_tf:
            self.publish_static_transforms()

        self.get_logger().info(
            f"Slam toolbox bridge ready. pose={self.pose_topic}, imu={self.imu_topic}, "
            f"lidar={self.lidar_topic}, scan={self.scan_topic}, odom={self.odom_topic}, "
            f"frames: {self.odom_frame}->{self.base_frame}->{self.lidar_frame}"
        )

    def publish_static_transforms(self) -> None:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.child_frame_id = self.lidar_frame
        msg.transform.translation.x = self.lidar_x
        msg.transform.translation.y = self.lidar_y
        msg.transform.translation.z = self.lidar_z
        msg.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(msg)

    def pose_cb(self, msg: Point) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        position = (float(msg.x), float(msg.y), float(msg.z))
        if self.prev_position is not None and self.prev_pose_time_sec is not None:
            dt = max(now_sec - self.prev_pose_time_sec, 1e-6)
            dx = position[0] - self.prev_position[0]
            dy = position[1] - self.prev_position[1]
            self.latest_linear_speed = math.hypot(dx, dy) / dt
        self.latest_position = position
        self.prev_position = position
        self.prev_pose_time_sec = now_sec

    def imu_cb(self, msg: Imu) -> None:
        self.latest_orientation = (
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        )
        self.latest_yaw_rate = float(msg.angular_velocity.z)

    def publish_odometry_and_tf(self) -> None:
        if self.latest_position is None:
            return

        stamp = self.get_clock().now().to_msg()
        self.latest_tf_stamp = stamp

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.latest_position[0]
        odom.pose.pose.position.y = self.latest_position[1]
        odom.pose.pose.position.z = self.latest_position[2]
        odom.pose.pose.orientation.x = self.latest_orientation[0]
        odom.pose.pose.orientation.y = self.latest_orientation[1]
        odom.pose.pose.orientation.z = self.latest_orientation[2]
        odom.pose.pose.orientation.w = self.latest_orientation[3]
        odom.twist.twist.linear.x = self.latest_linear_speed
        odom.twist.twist.angular.z = self.latest_yaw_rate
        self.odom_pub.publish(odom)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.odom_frame
        tf_msg.child_frame_id = self.base_frame
        tf_msg.transform.translation.x = self.latest_position[0]
        tf_msg.transform.translation.y = self.latest_position[1]
        tf_msg.transform.translation.z = self.latest_position[2]
        tf_msg.transform.rotation.x = self.latest_orientation[0]
        tf_msg.transform.rotation.y = self.latest_orientation[1]
        tf_msg.transform.rotation.z = self.latest_orientation[2]
        tf_msg.transform.rotation.w = self.latest_orientation[3]
        self.tf_broadcaster.sendTransform(tf_msg)

    def lidar_cb(self, msg: LaserScan) -> None:
        if self.latest_position is None:
            return

        scan_msg = LaserScan()
        scan_msg.header = msg.header
        scan_msg.header.frame_id = self.lidar_frame

        if self.latest_tf_stamp is not None:
            sec = int(self.latest_tf_stamp.sec)
            nanosec = int(self.latest_tf_stamp.nanosec - self.scan_stamp_lag_sec * 1e9)
            while nanosec < 0:
                sec -= 1
                nanosec += 1_000_000_000
            scan_msg.header.stamp.sec = sec
            scan_msg.header.stamp.nanosec = nanosec
        else:
            scan_msg.header.stamp = self.get_clock().now().to_msg()

        scan_msg.angle_min = msg.angle_min
        scan_msg.angle_max = msg.angle_max
        scan_msg.angle_increment = msg.angle_increment
        scan_msg.time_increment = msg.time_increment
        scan_msg.scan_time = msg.scan_time
        scan_msg.range_min = msg.range_min
        scan_msg.range_max = msg.range_max
        scan_msg.ranges = msg.ranges
        scan_msg.intensities = msg.intensities
        self.scan_pub.publish(scan_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlamToolboxBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

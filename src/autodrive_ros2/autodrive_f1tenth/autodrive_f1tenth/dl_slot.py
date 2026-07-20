#!/usr/bin/env python3

from __future__ import annotations

import math
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan, PointCloud2, PointField
from std_msgs.msg import Header


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def make_pointcloud2(points: list[tuple[float, float, float]], frame_id: str, stamp) -> PointCloud2:
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 12
    data = bytearray()
    for x, y, z in points:
        data.extend(struct.pack("fff", float(x), float(y), float(z)))
    return PointCloud2(
        header=Header(frame_id=frame_id, stamp=stamp),
        height=1,
        width=len(points),
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * len(points),
        data=bytes(data),
        is_dense=True,
    )


@dataclass
class Track:
    track_id: int
    cls: str = "cluster"
    history: Deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=10))
    hits: int = 0
    misses: int = 0
    dynamic: bool = False

    def add_observation(self, stamp_sec: float, x: float, y: float) -> None:
        self.history.append((stamp_sec, x, y))
        self.hits += 1
        self.misses = 0

    def mark_missed(self) -> None:
        self.misses += 1

    def last_position(self) -> tuple[float, float] | None:
        if not self.history:
            return None
        _, x, y = self.history[-1]
        return (x, y)

    def speed(self) -> float:
        if len(self.history) < 2:
            return 0.0
        t0, x0, y0 = self.history[0]
        t1, x1, y1 = self.history[-1]
        dt = max(t1 - t0, 1e-6)
        return math.hypot(x1 - x0, y1 - y0) / dt

    def heading(self) -> float:
        if len(self.history) < 2:
            return 0.0
        _, x0, y0 = self.history[-2]
        _, x1, y1 = self.history[-1]
        return math.atan2(y1 - y0, x1 - x0)

    def predicted_position(self, stamp_sec: float) -> tuple[float, float] | None:
        if not self.history:
            return None
        if len(self.history) < 2:
            return self.last_position()
        t0, x0, y0 = self.history[-2]
        t1, x1, y1 = self.history[-1]
        dt = max(t1 - t0, 1e-6)
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        horizon = stamp_sec - t1
        return (x1 + vx * horizon, y1 + vy * horizon)


class DLSLOTNode(Node):
    """
    Paper-inspired dynamic LiDAR mapping and tracking node.

    This is a lightweight ROS 2 implementation tailored to the current repo:
    it separates dynamic scan clusters from persistent static structure and
    publishes both as point clouds in a world frame estimated from IPS + IMU.
    """

    def __init__(self) -> None:
        super().__init__("dl_slot")

        self.declare_parameter("scan_topic", "/autodrive/f1tenth_1/lidar")
        self.declare_parameter("pose_topic", "/autodrive/f1tenth_1/ips")
        self.declare_parameter("imu_topic", "/autodrive/f1tenth_1/imu")
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("static_cloud_topic", "/autodrive/f1tenth_1/dl_slot/static_cloud")
        self.declare_parameter("dynamic_cloud_topic", "/autodrive/f1tenth_1/dl_slot/dynamic_cloud")
        self.declare_parameter("dynamic_tracks_topic", "/autodrive/f1tenth_1/dl_slot/dynamic_tracks")
        self.declare_parameter("min_range", 0.05)
        self.declare_parameter("max_range", 15.0)
        self.declare_parameter("cluster_distance_threshold", 0.35)
        self.declare_parameter("cluster_distance_scale", 2.5)
        self.declare_parameter("min_cluster_points", 3)
        self.declare_parameter("track_match_distance", 1.0)
        self.declare_parameter("track_timeout_sec", 1.0)
        self.declare_parameter("dynamic_speed_threshold", 0.4)
        self.declare_parameter("track_init_hits", 3)
        self.declare_parameter("voxel_size", 0.10)
        self.declare_parameter("static_promotion_hits", 1)
        self.declare_parameter("static_prune_after_sec", 8.0)
        self.declare_parameter("publish_rate_hz", 200.0)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.static_cloud_topic = str(self.get_parameter("static_cloud_topic").value)
        self.dynamic_cloud_topic = str(self.get_parameter("dynamic_cloud_topic").value)
        self.dynamic_tracks_topic = str(self.get_parameter("dynamic_tracks_topic").value)
        self.min_range = float(self.get_parameter("min_range").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.cluster_distance_threshold = float(self.get_parameter("cluster_distance_threshold").value)
        self.cluster_distance_scale = float(self.get_parameter("cluster_distance_scale").value)
        self.min_cluster_points = int(self.get_parameter("min_cluster_points").value)
        self.track_match_distance = float(self.get_parameter("track_match_distance").value)
        self.track_timeout_sec = float(self.get_parameter("track_timeout_sec").value)
        self.dynamic_speed_threshold = float(self.get_parameter("dynamic_speed_threshold").value)
        self.track_init_hits = int(self.get_parameter("track_init_hits").value)
        self.voxel_size = float(self.get_parameter("voxel_size").value)
        self.static_promotion_hits = int(self.get_parameter("static_promotion_hits").value)
        self.static_prune_after_sec = float(self.get_parameter("static_prune_after_sec").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.position: tuple[float, float] | None = None
        self.last_pose_time: float | None = None
        self.yaw: float | None = None
        self.yaw_rate = 0.0
        self._last_angle_increment = 0.0

        self.tracks: dict[int, Track] = {}
        self.next_track_id = 0
        self.static_voxels: dict[tuple[int, int], dict[str, float]] = {}
        self.latest_dynamic_points: list[tuple[float, float, float]] = []
        self.latest_dynamic_track_poses: list[tuple[float, float, float]] = []

        self.static_pub = self.create_publisher(PointCloud2, self.static_cloud_topic, 10)
        self.dynamic_pub = self.create_publisher(PointCloud2, self.dynamic_cloud_topic, 10)
        self.track_pub = self.create_publisher(PoseArray, self.dynamic_tracks_topic, 10)

        self.create_subscription(Point, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(Imu, self.imu_topic, self.imu_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.create_timer(1.0 / max(self.publish_rate_hz, 1e-3), self.publish_outputs)

        self.get_logger().info(
            "DL-SLOT-inspired node ready. "
            f"scan={self.scan_topic}, pose={self.pose_topic}, imu={self.imu_topic}"
        )

    def pose_cb(self, msg: Point) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        new_position = (float(msg.x), float(msg.y))

        self.position = new_position
        self.last_pose_time = now

    def imu_cb(self, msg: Imu) -> None:
        self.yaw_rate = float(msg.angular_velocity.z)
        self.yaw = quaternion_to_yaw(
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        )

    def scan_cb(self, msg: LaserScan) -> None:
        if self.position is None:
            return

        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        yaw = self.current_yaw(stamp_sec)

        local_points = self.scan_to_local_points(msg)
        clusters = self.cluster_points(local_points)
        world_clusters = [self.transform_cluster(points, self.position, yaw) for points in clusters]

        assignments = self.associate_tracks(world_clusters, stamp_sec)
        dynamic_points: list[tuple[float, float, float]] = []
        dynamic_track_poses: list[tuple[float, float, float]] = []

        for cluster_idx, world_points in enumerate(world_clusters):
            track = assignments.get(cluster_idx)
            if track is None:
                continue

            speed = track.speed()
            track.dynamic = track.hits >= self.track_init_hits and speed >= self.dynamic_speed_threshold
            centroid_x = sum(point[0] for point in world_points) / len(world_points)
            centroid_y = sum(point[1] for point in world_points) / len(world_points)

            if track.dynamic:
                dynamic_points.extend(world_points)
                dynamic_track_poses.append((centroid_x, centroid_y, track.heading()))
            else:
                self.update_static_map(world_points, stamp_sec)

        self.latest_dynamic_points = dynamic_points
        self.latest_dynamic_track_poses = dynamic_track_poses
        self.prune_static_map(stamp_sec)
        self.prune_tracks(stamp_sec)

    def current_yaw(self, stamp_sec: float) -> float:
        if self.yaw is not None:
            return self.yaw
        return 0.0

    def scan_to_local_points(self, scan: LaserScan) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        self._last_angle_increment = float(scan.angle_increment)
        angle = scan.angle_min
        for distance in scan.ranges:
            if math.isfinite(distance) and self.min_range <= distance <= self.max_range:
                points.append((distance * math.cos(angle), distance * math.sin(angle)))
            angle += scan.angle_increment
        return points

    def cluster_points(self, points: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
        if not points:
            return []

        clusters: list[list[tuple[float, float]]] = []
        current_cluster = [points[0]]

        for point in points[1:]:
            prev = current_cluster[-1]
            prev_range = math.hypot(prev[0], prev[1])
            point_range = math.hypot(point[0], point[1])
            adaptive_threshold = max(
                self.cluster_distance_threshold,
                self.cluster_distance_scale * max(prev_range, point_range) * abs(self._last_angle_increment),
            )
            if math.hypot(point[0] - prev[0], point[1] - prev[1]) <= adaptive_threshold:
                current_cluster.append(point)
            else:
                if len(current_cluster) >= self.min_cluster_points:
                    clusters.append(current_cluster)
                current_cluster = [point]

        if len(current_cluster) >= self.min_cluster_points:
            clusters.append(current_cluster)
        return clusters

    def transform_cluster(
        self,
        cluster: list[tuple[float, float]],
        position: tuple[float, float],
        yaw: float,
    ) -> list[tuple[float, float, float]]:
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        world_points: list[tuple[float, float, float]] = []
        for local_x, local_y in cluster:
            world_x = position[0] + local_x * cos_yaw - local_y * sin_yaw
            world_y = position[1] + local_x * sin_yaw + local_y * cos_yaw
            world_points.append((world_x, world_y, 0.0))
        return world_points

    def associate_tracks(
        self,
        world_clusters: list[list[tuple[float, float, float]]],
        stamp_sec: float,
    ) -> dict[int, Track]:
        centroids = []
        for cluster in world_clusters:
            x = sum(point[0] for point in cluster) / len(cluster)
            y = sum(point[1] for point in cluster) / len(cluster)
            centroids.append((x, y))

        unmatched_tracks = set(self.tracks.keys())
        assignments: dict[int, Track] = {}

        for cluster_idx, centroid in enumerate(centroids):
            best_track_id = None
            best_distance = float("inf")
            for track_id in unmatched_tracks:
                predicted = self.tracks[track_id].predicted_position(stamp_sec)
                if predicted is None:
                    continue
                distance = math.hypot(predicted[0] - centroid[0], predicted[1] - centroid[1])
                if distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is not None and best_distance <= self.track_match_distance:
                track = self.tracks[best_track_id]
                track.add_observation(stamp_sec, centroid[0], centroid[1])
                assignments[cluster_idx] = track
                unmatched_tracks.remove(best_track_id)
            else:
                track = Track(track_id=self.next_track_id)
                self.next_track_id += 1
                track.add_observation(stamp_sec, centroid[0], centroid[1])
                self.tracks[track.track_id] = track
                assignments[cluster_idx] = track

        for track_id in unmatched_tracks:
            self.tracks[track_id].mark_missed()

        return assignments

    def update_static_map(self, world_points: list[tuple[float, float, float]], stamp_sec: float) -> None:
        for x, y, _ in world_points:
            key = (int(math.floor(x / self.voxel_size)), int(math.floor(y / self.voxel_size)))
            entry = self.static_voxels.setdefault(key, {"hits": 0.0, "last_seen": stamp_sec})
            entry["hits"] += 1.0
            entry["last_seen"] = stamp_sec

    def prune_static_map(self, stamp_sec: float) -> None:
        stale_keys = [
            key
            for key, entry in self.static_voxels.items()
            if stamp_sec - entry["last_seen"] > self.static_prune_after_sec and entry["hits"] < self.static_promotion_hits
        ]
        for key in stale_keys:
            del self.static_voxels[key]

    def prune_tracks(self, stamp_sec: float) -> None:
        stale_ids = []
        for track_id, track in self.tracks.items():
            if not track.history:
                stale_ids.append(track_id)
                continue
            last_stamp = track.history[-1][0]
            if stamp_sec - last_stamp > self.track_timeout_sec:
                stale_ids.append(track_id)
        for track_id in stale_ids:
            del self.tracks[track_id]

    def publish_outputs(self) -> None:
        stamp = self.get_clock().now().to_msg()

        static_points: list[tuple[float, float, float]] = []
        for (ix, iy), entry in self.static_voxels.items():
            if entry["hits"] >= self.static_promotion_hits:
                static_points.append(((ix + 0.5) * self.voxel_size, (iy + 0.5) * self.voxel_size, 0.0))

        static_cloud = make_pointcloud2(static_points, self.world_frame, stamp)
        dynamic_cloud = make_pointcloud2(self.latest_dynamic_points, self.world_frame, stamp)
        self.static_pub.publish(static_cloud)
        self.dynamic_pub.publish(dynamic_cloud)

        pose_array = PoseArray()
        pose_array.header.frame_id = self.world_frame
        pose_array.header.stamp = stamp
        for x, y, yaw in self.latest_dynamic_track_poses:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.0
            qx, qy, qz, qw = yaw_to_quaternion(yaw)
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            pose_array.poses.append(pose)
        self.track_pub.publish(pose_array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DLSLOTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

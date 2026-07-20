#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from scipy.spatial import cKDTree
from std_msgs.msg import Bool, Float32
from visualization_msgs.msg import Marker


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "tracks" / "src").exists():
            return parent
    return Path.cwd()


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class TargetSample:
    stamp_sec: float
    world_x: float
    world_y: float


class OvertakeDecision(Node):
    def __init__(self) -> None:
        super().__init__("overtake_decision")

        repo_root = find_repo_root()
        self.declare_parameter("pose_topic", "/autodrive/f1tenth_1/ips")
        self.declare_parameter("imu_topic", "/autodrive/f1tenth_1/imu")
        self.declare_parameter("target_point_topic", "/autodrive/f1tenth_1/target_tracker/target_point")
        self.declare_parameter("target_visible_topic", "/autodrive/f1tenth_1/target_tracker/target_visible")
        self.declare_parameter("tracking_confidence_topic", "/autodrive/f1tenth_1/target_tracker/tracking_confidence")
        self.declare_parameter(
            "wall_mask_csv_path",
            str(repo_root / "output" / "slam_runs" / "slam_toolbox_boundary_wall_mask.csv"),
        )
        self.declare_parameter("min_tracking_confidence", 0.70)
        self.declare_parameter("history_size", 8)
        self.declare_parameter("min_history_points", 4)
        self.declare_parameter("min_target_motion_m", 0.40)
        self.declare_parameter("min_wall_clearance_m", 0.35)
        self.declare_parameter("good_wall_clearance_m", 1.20)
        self.declare_parameter("good_relative_heading_deg", 12.0)
        self.declare_parameter("max_relative_heading_deg", 35.0)
        self.declare_parameter("good_curvature_rad_per_m", 0.05)
        self.declare_parameter("max_curvature_rad_per_m", 0.25)
        self.declare_parameter("weight_wall_clearance", 0.40)
        self.declare_parameter("weight_relative_heading", 0.35)
        self.declare_parameter("weight_curvature", 0.25)
        self.declare_parameter("overtake_score_threshold", 0.65)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("wall_clearance_topic", "/autodrive/f1tenth_1/target_tracker/wall_clearance_m")
        self.declare_parameter("relative_heading_topic", "/autodrive/f1tenth_1/target_tracker/relative_heading_deg")
        self.declare_parameter("target_curvature_topic", "/autodrive/f1tenth_1/target_tracker/target_curvature")
        self.declare_parameter("overtake_score_topic", "/autodrive/f1tenth_1/target_tracker/overtake_score")
        self.declare_parameter("overtake_allowed_topic", "/autodrive/f1tenth_1/target_tracker/overtake_allowed")
        self.declare_parameter("overtake_marker_topic", "/autodrive/f1tenth_1/target_tracker/overtake_marker")

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.target_point_topic = str(self.get_parameter("target_point_topic").value)
        self.target_visible_topic = str(self.get_parameter("target_visible_topic").value)
        self.tracking_confidence_topic = str(self.get_parameter("tracking_confidence_topic").value)
        self.wall_mask_csv_path = Path(str(self.get_parameter("wall_mask_csv_path").value)).expanduser()
        self.min_tracking_confidence = float(self.get_parameter("min_tracking_confidence").value)
        self.history_size = int(self.get_parameter("history_size").value)
        self.min_history_points = int(self.get_parameter("min_history_points").value)
        self.min_target_motion_m = float(self.get_parameter("min_target_motion_m").value)
        self.min_wall_clearance_m = float(self.get_parameter("min_wall_clearance_m").value)
        self.good_wall_clearance_m = float(self.get_parameter("good_wall_clearance_m").value)
        self.good_relative_heading_deg = float(self.get_parameter("good_relative_heading_deg").value)
        self.max_relative_heading_deg = float(self.get_parameter("max_relative_heading_deg").value)
        self.good_curvature_rad_per_m = float(self.get_parameter("good_curvature_rad_per_m").value)
        self.max_curvature_rad_per_m = float(self.get_parameter("max_curvature_rad_per_m").value)
        self.weight_wall_clearance = float(self.get_parameter("weight_wall_clearance").value)
        self.weight_relative_heading = float(self.get_parameter("weight_relative_heading").value)
        self.weight_curvature = float(self.get_parameter("weight_curvature").value)
        self.overtake_score_threshold = float(self.get_parameter("overtake_score_threshold").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.wall_clearance_topic = str(self.get_parameter("wall_clearance_topic").value)
        self.relative_heading_topic = str(self.get_parameter("relative_heading_topic").value)
        self.target_curvature_topic = str(self.get_parameter("target_curvature_topic").value)
        self.overtake_score_topic = str(self.get_parameter("overtake_score_topic").value)
        self.overtake_allowed_topic = str(self.get_parameter("overtake_allowed_topic").value)
        self.overtake_marker_topic = str(self.get_parameter("overtake_marker_topic").value)

        self.weight_sum = max(
            self.weight_wall_clearance + self.weight_relative_heading + self.weight_curvature,
            1e-6,
        )

        self.ego_position: tuple[float, float] | None = None
        self.ego_yaw: float | None = None
        self.target_visible = False
        self.tracking_confidence = 0.0
        self.latest_target_local: tuple[float, float] | None = None
        self.target_history: deque[TargetSample] = deque(maxlen=max(self.history_size, 3))

        self.wall_points, self.wall_tree = self._load_wall_mask_points()

        self.latest_wall_clearance = 0.0
        self.latest_relative_heading_deg = 180.0
        self.latest_target_curvature = 0.0
        self.latest_overtake_score = 0.0
        self.latest_overtake_allowed = False
        self.latest_target_world: tuple[float, float] | None = None

        self.wall_clearance_pub = self.create_publisher(Float32, self.wall_clearance_topic, 10)
        self.relative_heading_pub = self.create_publisher(Float32, self.relative_heading_topic, 10)
        self.target_curvature_pub = self.create_publisher(Float32, self.target_curvature_topic, 10)
        self.overtake_score_pub = self.create_publisher(Float32, self.overtake_score_topic, 10)
        self.overtake_allowed_pub = self.create_publisher(Bool, self.overtake_allowed_topic, 10)
        self.marker_pub = self.create_publisher(Marker, self.overtake_marker_topic, 10)

        self.create_subscription(Point, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(PointStamped, self.target_point_topic, self.target_point_cb, 10)
        self.create_subscription(Bool, self.target_visible_topic, self.target_visible_cb, 10)
        self.create_subscription(Float32, self.tracking_confidence_topic, self.tracking_confidence_cb, 10)

        from sensor_msgs.msg import Imu
        self.create_subscription(Imu, self.imu_topic, self.imu_cb, 10)

        self.create_timer(1.0 / max(self.publish_rate_hz, 1e-3), self.publish_outputs)
        self.get_logger().info(
            "Overtake decision ready. "
            f"target={self.target_point_topic}, confidence={self.tracking_confidence_topic}, wall_mask={self.wall_mask_csv_path}"
        )

    def _load_wall_mask_points(self) -> tuple[np.ndarray, cKDTree | None]:
        csv_path = self.wall_mask_csv_path
        if not csv_path.exists():
            fallback = (
                find_repo_root()
                / "ros2"
                / "install"
                / "autodrive_f1tenth"
                / "lib"
                / "output"
                / "slam_runs"
                / csv_path.name
            )
            if fallback.exists():
                csv_path = fallback
                self.wall_mask_csv_path = fallback
        if not csv_path.exists():
            self.get_logger().warn(f"Wall mask CSV not found at {self.wall_mask_csv_path}")
            return np.empty((0, 2), dtype=float), None

        points: list[tuple[float, float]] = []
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                x = float(row["world_x_m"])
                y = float(row["world_y_m"])
                points.append((x, y))
        point_array = np.asarray(points, dtype=float)
        tree = cKDTree(point_array) if len(point_array) > 0 else None
        self.get_logger().info(f"Loaded wall mask points={len(point_array)} from {csv_path}")
        return point_array, tree

    def pose_cb(self, msg: Point) -> None:
        self.ego_position = (float(msg.x), float(msg.y))

    def imu_cb(self, msg) -> None:
        x = float(msg.orientation.x)
        y = float(msg.orientation.y)
        z = float(msg.orientation.z)
        w = float(msg.orientation.w)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        self.ego_yaw = math.atan2(siny_cosp, cosy_cosp)

    def target_visible_cb(self, msg: Bool) -> None:
        self.target_visible = bool(msg.data)

    def tracking_confidence_cb(self, msg: Float32) -> None:
        self.tracking_confidence = float(msg.data)

    def local_to_world(self, local_x: float, local_y: float) -> tuple[float, float] | None:
        if self.ego_position is None or self.ego_yaw is None:
            return None
        cos_yaw = math.cos(self.ego_yaw)
        sin_yaw = math.sin(self.ego_yaw)
        return (
            self.ego_position[0] + local_x * cos_yaw - local_y * sin_yaw,
            self.ego_position[1] + local_x * sin_yaw + local_y * cos_yaw,
        )

    def target_point_cb(self, msg: PointStamped) -> None:
        local_x = float(msg.point.x)
        local_y = float(msg.point.y)
        self.latest_target_local = (local_x, local_y)
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        world_xy = self.local_to_world(local_x, local_y)
        if world_xy is None:
            return
        self.latest_target_world = world_xy
        self.target_history.append(TargetSample(stamp_sec=stamp_sec, world_x=world_xy[0], world_y=world_xy[1]))
        self.update_decision()

    def distance_to_wall(self, world_x: float, world_y: float) -> float:
        if self.wall_tree is None:
            return float("inf")
        distance, _ = self.wall_tree.query([world_x, world_y], k=1)
        return float(distance)

    def estimate_target_heading(self) -> tuple[float | None, float]:
        if len(self.target_history) < self.min_history_points:
            return None, 0.0
        first = self.target_history[0]
        last = self.target_history[-1]
        dx = last.world_x - first.world_x
        dy = last.world_y - first.world_y
        motion = math.hypot(dx, dy)
        if motion < self.min_target_motion_m:
            return None, motion
        return math.atan2(dy, dx), motion

    def estimate_target_curvature(self) -> float:
        if len(self.target_history) < 3:
            return 0.0
        headings: list[float] = []
        distances: list[float] = []
        samples = list(self.target_history)
        for prev, curr in zip(samples[:-1], samples[1:]):
            dx = curr.world_x - prev.world_x
            dy = curr.world_y - prev.world_y
            ds = math.hypot(dx, dy)
            if ds < 1e-4:
                continue
            headings.append(math.atan2(dy, dx))
            distances.append(ds)
        if len(headings) < 2 or sum(distances) < self.min_target_motion_m:
            return 0.0

        heading_change = 0.0
        for prev_h, curr_h in zip(headings[:-1], headings[1:]):
            heading_change += normalize_angle(curr_h - prev_h)
        path_length = max(sum(distances), 1e-6)
        return abs(heading_change) / path_length

    def update_decision(self) -> None:
        self.latest_overtake_allowed = False
        self.latest_overtake_score = 0.0

        if (
            not self.target_visible
            or self.tracking_confidence < self.min_tracking_confidence
            or self.ego_position is None
            or self.ego_yaw is None
            or self.latest_target_world is None
        ):
            return

        target_heading, motion = self.estimate_target_heading()
        if target_heading is None:
            return

        wall_clearance = self.distance_to_wall(self.latest_target_world[0], self.latest_target_world[1])
        relative_heading_deg = abs(math.degrees(normalize_angle(target_heading - self.ego_yaw)))
        target_curvature = self.estimate_target_curvature()

        clearance_score = max(
            0.0,
            min(
                1.0,
                (wall_clearance - self.min_wall_clearance_m)
                / max(self.good_wall_clearance_m - self.min_wall_clearance_m, 1e-6),
            ),
        )
        heading_score = max(
            0.0,
            min(
                1.0,
                (self.max_relative_heading_deg - relative_heading_deg)
                / max(self.max_relative_heading_deg - self.good_relative_heading_deg, 1e-6),
            ),
        )
        curvature_score = max(
            0.0,
            min(
                1.0,
                (self.max_curvature_rad_per_m - target_curvature)
                / max(self.max_curvature_rad_per_m - self.good_curvature_rad_per_m, 1e-6),
            ),
        )

        weighted_score = (
            self.weight_wall_clearance * clearance_score
            + self.weight_relative_heading * heading_score
            + self.weight_curvature * curvature_score
        ) / self.weight_sum
        weighted_score *= min(self.tracking_confidence / max(self.min_tracking_confidence, 1e-6), 1.0)

        self.latest_wall_clearance = wall_clearance
        self.latest_relative_heading_deg = relative_heading_deg
        self.latest_target_curvature = target_curvature
        self.latest_overtake_score = max(0.0, min(0.99, weighted_score))
        self.latest_overtake_allowed = (
            wall_clearance >= self.min_wall_clearance_m
            and motion >= self.min_target_motion_m
            and self.latest_overtake_score >= self.overtake_score_threshold
        )

    def publish_outputs(self) -> None:
        self.wall_clearance_pub.publish(Float32(data=float(self.latest_wall_clearance)))
        self.relative_heading_pub.publish(Float32(data=float(self.latest_relative_heading_deg)))
        self.target_curvature_pub.publish(Float32(data=float(self.latest_target_curvature)))
        self.overtake_score_pub.publish(Float32(data=float(self.latest_overtake_score)))
        self.overtake_allowed_pub.publish(Bool(data=bool(self.latest_overtake_allowed)))

        stamp = self.get_clock().now().to_msg()
        marker = Marker()
        marker.header.frame_id = "lidar_1"
        marker.header.stamp = stamp
        marker.ns = "overtake_decision"
        marker.id = 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.scale.z = 0.35
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = 0.6
        marker.pose.position.y = -0.6
        marker.pose.position.z = 0.5
        marker.color.a = 1.0
        if self.latest_overtake_allowed:
            marker.color.r = 0.1
            marker.color.g = 1.0
            marker.color.b = 0.1
        else:
            marker.color.r = 1.0
            marker.color.g = 0.2
            marker.color.b = 0.1
        marker.text = (
            f"OT {self.latest_overtake_score:.2f} | "
            f"C={self.latest_wall_clearance:.2f} "
            f"H={self.latest_relative_heading_deg:.1f} "
            f"K={self.latest_target_curvature:.3f}"
        )
        self.marker_pub.publish(marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OvertakeDecision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

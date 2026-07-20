# #!/usr/bin/env python3
# import math
# import numpy as np

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# from sensor_msgs.msg import LaserScan, Imu
# from geometry_msgs.msg import Quaternion, Point
# from visualization_msgs.msg import Marker

# # === Topics & Parameters ===
# LIDAR_TOPIC = '/autodrive/f1tenth_1/lidar'
# IPS_TOPIC = '/autodrive/f1tenth_1/ips'
# IMU_TOPIC = '/autodrive/f1tenth_1/imu'
# FRAME_ID = 'map'

# RANGE_MIN = 0.2
# RANGE_MAX = 3.0
# ANGLE_WINDOW_DEG = 60.0

# DEVIATION_THRESHOLD = 0.08
# STEERING_GAIN = 2.0        # Kp
# KD_GAIN = 0.2              # kept (unused for now, left in case you want PD)
# MAX_STEERING_ANGLE = 0.52

# # cluster tuning
# CLUSTER_MAX_GAP = 0.30     # meters between consecutive points to be considered same cluster
# CLUSTER_MIN_SIZE = 4       # minimum points for a cluster

# # width-based steering term
# WIDTH_GAIN = 0.4           # Kw
# W0 = 0.5                   # nominal width (meters) when car is straight

# # smoothing
# SMOOTHING_ALPHA = 0.3      # 0..1 (higher = more reactive, lower = smoother)

# DOT_SIZE = 0.06


# def quat_to_yaw(q: Quaternion) -> float:
#     """Convert quaternion to yaw angle (radians)."""
#     siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
#     cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
#     return math.atan2(siny_cosp, cosy_cosp)


# class SimpleTargetTracker(Node):
#     def __init__(self):
#         super().__init__('simple_target_tracker')

#         qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

#         self.create_subscription(LaserScan, LIDAR_TOPIC, self.on_scan, qos)
#         self.create_subscription(Point, IPS_TOPIC, self.on_point, 10)
#         self.create_subscription(Imu, IMU_TOPIC, self.on_imu, 10)

#         # publish cluster points and steering arrow separately
#         self.cluster_pub = self.create_publisher(Marker, 'target_cluster', 10)
#         self.steer_pub = self.create_publisher(Marker, 'steering_visualization', 10)

#         self.have_pose = False
#         self.have_yaw = False
#         self.x_e = self.y_e = self.yaw_e = 0.0

#         self.current_deviation = 0.0
#         self.deviation_direction = 0.0
#         self.computed_steering = 0.0
#         self.filtered_steering = 0.0

#         self.prev_deviation = 0.0
#         self.prev_time = self.get_clock().now()

#         self.get_logger().info('target tracker initialized.')

#     # === Subscriptions ===
#     def on_point(self, msg: Point):
#         self.x_e, self.y_e = msg.x, msg.y
#         self.have_pose = True

#     def on_imu(self, msg: Imu):
#         self.yaw_e = quat_to_yaw(msg.orientation)
#         self.have_yaw = True

#     # === Core Callback ===
#     def on_scan(self, scan: LaserScan):
#         if not (self.have_pose and self.have_yaw):
#             return

#         n = len(scan.ranges)
#         angles = scan.angle_min + np.arange(n) * scan.angle_increment
#         ranges = np.array(scan.ranges, dtype=float)

#         # Focus on points ahead within ±ANGLE_WINDOW_DEG
#         mask = (
#             (angles > -math.radians(ANGLE_WINDOW_DEG))
#             & (angles < math.radians(ANGLE_WINDOW_DEG))
#             & (ranges > RANGE_MIN)
#             & (ranges < RANGE_MAX)
#         )

#         if not np.any(mask):
#             # clear visualizations
#             self.publish_empty_markers(scan.header.stamp)
#             return

#         a = angles[mask]
#         r = ranges[mask]
#         xs = r * np.cos(a)
#         ys = r * np.sin(a)

#         # keep ordering by angle (useful for single-link clustering)
#         order = np.argsort(a)
#         xs = xs[order]
#         ys = ys[order]

#         points = np.column_stack([xs, ys])

#         # Simple single-link clustering in angular order (fast & online-friendly)
#         clusters = []
#         current = [0]
#         for i in range(1, len(points)):
#             gap = math.hypot(points[i, 0] - points[i-1, 0], points[i, 1] - points[i-1, 1])
#             if gap <= CLUSTER_MAX_GAP:
#                 current.append(i)
#             else:
#                 if len(current) >= CLUSTER_MIN_SIZE:
#                     clusters.append(current)
#                 current = [i]
#         if len(current) >= CLUSTER_MIN_SIZE:
#             clusters.append(current)

#         if not clusters:
#             self.publish_empty_markers(scan.header.stamp)
#             return

#         # Choose the best cluster: closest (smallest mean x) but ahead of ego (x>0)
#         best_cluster_pts = None
#         best_x_mean = float('inf')
#         for idx_list in clusters:
#             cpts = points[idx_list]
#             x_mean = np.mean(cpts[:, 0])
#             if x_mean > 0.1 and x_mean < best_x_mean:  # prefer clusters in front
#                 # simple compactness filter (avoid long walls)
#                 x_span = np.max(cpts[:, 0]) - np.min(cpts[:, 0])
#                 y_span = np.max(cpts[:, 1]) - np.min(cpts[:, 1])
#                 compactness = math.hypot(x_span, y_span)
#                 if compactness < 1.2:  # tunable
#                     best_cluster_pts = cpts
#                     best_x_mean = x_mean

#         if best_cluster_pts is None:
#             self.publish_empty_markers(scan.header.stamp)
#             return

#         # Deviation metrics
#         lateral_dev = float(np.mean(best_cluster_pts[:, 1]))   # positive = left, negative = right
#         width = float(np.max(best_cluster_pts[:, 1]) - np.min(best_cluster_pts[:, 1]))
#         num_points = int(len(best_cluster_pts))

#         self.current_deviation = abs(lateral_dev)
#         self.deviation_direction = float(np.sign(lateral_dev))

#         # Steering law: combine lateral offset and apparent width
#         kp = STEERING_GAIN
#         kw = WIDTH_GAIN
#         steering_raw = kp * lateral_dev + kw * (width - W0)  # lateral_dev includes sign

#         # clip
#         steering_raw = float(np.clip(steering_raw, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE))

#         # smooth with EMA to reduce jitter but remain reactive
#         self.filtered_steering = (1.0 - SMOOTHING_ALPHA) * self.filtered_steering + SMOOTHING_ALPHA * steering_raw
#         self.computed_steering = float(np.clip(self.filtered_steering, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE))

#         # publish visualizations: cluster points + arrow + debug text marker
#         self.publish_cluster_and_steering(scan.header.stamp, best_cluster_pts, lateral_dev, width, num_points)

#     # === Visualization ===
#     def publish_empty_markers(self, stamp):
#         # publish an empty cluster marker (delete) and zeroed arrow
#         # cluster delete: send a marker with action DELETEALL
#         m = Marker()
#         m.header.frame_id = FRAME_ID
#         m.header.stamp = stamp
#         m.ns = 'cluster_points'
#         m.id = 0
#         m.action = Marker.DELETEALL
#         self.cluster_pub.publish(m)

#         # arrow with zero steering
#         arrow = self._make_steering_arrow(stamp, 0.0)
#         self.steer_pub.publish(arrow)

#     def publish_cluster_and_steering(self, stamp, cluster_pts, lateral_dev, width, num_points):
#         # Cluster points as a POINTS marker
#         p = Marker()
#         p.header.frame_id = FRAME_ID
#         p.header.stamp = stamp
#         p.ns = 'cluster_points'
#         p.id = 1
#         p.type = Marker.POINTS
#         p.action = Marker.ADD
#         p.pose.orientation.w = 1.0
#         p.scale.x = DOT_SIZE
#         p.scale.y = DOT_SIZE
#         p.color.r = 1.0
#         p.color.g = 0.2
#         p.color.b = 0.2
#         p.color.a = 1.0

#         # Add cluster points transformed into world frame (using ego pose)
#         R = np.array([[math.cos(self.yaw_e), -math.sin(self.yaw_e)],
#                       [math.sin(self.yaw_e),  math.cos(self.yaw_e)]])
#         world_pts = (R @ cluster_pts.T).T + np.array([self.x_e, self.y_e])

#         from geometry_msgs.msg import Point as Pt
#         for wp in world_pts:
#             pt = Pt()
#             pt.x = float(wp[0])
#             pt.y = float(wp[1])
#             pt.z = 0.15
#             p.points.append(pt)

#         # Publish cluster points
#         self.cluster_pub.publish(p)

#         # Steering arrow
#         arrow = self._make_steering_arrow(stamp, self.computed_steering)
#         self.steer_pub.publish(arrow)

#         # Optional: small text marker showing debug values (useful in RViz)
#         t = Marker()
#         t.header.frame_id = FRAME_ID
#         t.header.stamp = stamp
#         t.ns = 'debug_text'
#         t.id = 2
#         t.type = Marker.TEXT_VIEW_FACING
#         t.action = Marker.ADD
#         t.pose.position.x = float(self.x_e + 0.5 * math.cos(self.yaw_e))
#         t.pose.position.y = float(self.y_e + 0.5 * math.sin(self.yaw_e))
#         t.pose.position.z = 0.6
#         t.scale.z = 0.12
#         t.color.r = t.color.g = t.color.b = t.color.a = 1.0
#         t.text = f"dev={lateral_dev:.3f}m width={width:.3f}m pts={num_points} steer={self.computed_steering:.3f}rad"
#         self.cluster_pub.publish(t)

#     def _make_steering_arrow(self, stamp, steering_angle):
#         arrow = Marker()
#         arrow.header.frame_id = FRAME_ID
#         arrow.header.stamp = stamp
#         arrow.ns = 'steering_arrow'
#         arrow.id = 0
#         arrow.type = Marker.ARROW
#         arrow.action = Marker.ADD

#         arrow_len = 1.5
#         start_x, start_y = self.x_e, self.y_e
#         end_x = start_x + arrow_len * math.cos(self.yaw_e + steering_angle)
#         end_y = start_y + arrow_len * math.sin(self.yaw_e + steering_angle)

#         from geometry_msgs.msg import Point as Pt
#         arrow.points = [Pt(x=start_x, y=start_y, z=0.3), Pt(x=end_x, y=end_y, z=0.3)]

#         arrow.scale.x = 0.15
#         arrow.scale.y = 0.25
#         arrow.scale.z = 0.3

#         mag = min(1.0, abs(steering_angle) / MAX_STEERING_ANGLE) if MAX_STEERING_ANGLE > 0 else 0.0
#         arrow.color.r = mag
#         arrow.color.g = 1.0 - mag
#         arrow.color.b = 0.0
#         arrow.color.a = 1.0

#         return arrow


# def main():
#     rclpy.init()
#     node = SimpleTargetTracker()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Quaternion, Point
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker


# === Topics & Parameters ===
LIDAR_TOPIC = '/autodrive/f1tenth_1/lidar'
IPS_TOPIC = '/autodrive/f1tenth_1/ips'
IMU_TOPIC = '/autodrive/f1tenth_1/imu'
FRAME_ID = 'map'

RANGE_MIN = 0.2
RANGE_MAX = 3.0
ANGLE_WINDOW_DEG = 60.0

DEVIATION_THRESHOLD = 0.08
STEERING_GAIN = 2.0
KD_GAIN = 0.2
MAX_STEERING_ANGLE = 0.52

CLUSTER_MAX_GAP = 0.30
CLUSTER_MIN_SIZE = 4

WIDTH_GAIN = 0.4
W0 = 0.5

SMOOTHING_ALPHA = 0.3
DOT_SIZE = 0.06

TARGET_SPEED = 0.075   # constant forward velocity


def quat_to_yaw(q: Quaternion) -> float:
    """Convert quaternion to yaw angle (radians)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class SimpleTargetTracker(Node):
    def __init__(self):
        super().__init__('simple_target_tracker')

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # === Subscriptions ===
        self.create_subscription(LaserScan, LIDAR_TOPIC, self.on_scan, qos)
        self.create_subscription(Point, IPS_TOPIC, self.on_point, 10)
        self.create_subscription(Imu, IMU_TOPIC, self.on_imu, 10)

        # === Publishers ===
        self.cluster_pub = self.create_publisher(Marker, 'target_cluster', 10)
        self.steer_viz_pub = self.create_publisher(Marker, 'steering_visualization', 10)

        # Actual vehicle control publishers
        self.steer_pub = self.create_publisher(Float32, '/autodrive/f1tenth_1/steering_command', 10)
        self.throttle_pub = self.create_publisher(Float32, '/autodrive/f1tenth_1/throttle_command', 10)

        # === State Variables ===
        self.have_pose = False
        self.have_yaw = False
        self.x_e = self.y_e = self.yaw_e = 0.0

        self.filtered_steering = 0.0
        self.computed_steering = 0.0

        self.get_logger().info('SimpleTargetTracker with driving + steering initialized.')

    # === Subscriptions ===
    def on_point(self, msg: Point):
        self.x_e, self.y_e = msg.x, msg.y
        self.have_pose = True

    def on_imu(self, msg: Imu):
        self.yaw_e = quat_to_yaw(msg.orientation)
        self.have_yaw = True

    # === Core Callback ===
    def on_scan(self, scan: LaserScan):
        if not (self.have_pose and self.have_yaw):
            return

        n = len(scan.ranges)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        ranges = np.array(scan.ranges, dtype=float)

        # Focus on points ahead within ±ANGLE_WINDOW_DEG
        mask = (
            (angles > -math.radians(ANGLE_WINDOW_DEG))
            & (angles < math.radians(ANGLE_WINDOW_DEG))
            & (ranges > RANGE_MIN)
            & (ranges < RANGE_MAX)
        )

        if not np.any(mask):
            self.publish_empty_markers(scan.header.stamp)
            return

        a = angles[mask]
        r = ranges[mask]
        xs = r * np.cos(a)
        ys = r * np.sin(a)

        order = np.argsort(a)
        xs = xs[order]
        ys = ys[order]
        points = np.column_stack([xs, ys])

        # Simple single-link clustering
        clusters = []
        current = [0]
        for i in range(1, len(points)):
            gap = math.hypot(points[i, 0] - points[i - 1, 0], points[i, 1] - points[i - 1, 1])
            if gap <= CLUSTER_MAX_GAP:
                current.append(i)
            else:
                if len(current) >= CLUSTER_MIN_SIZE:
                    clusters.append(current)
                current = [i]
        if len(current) >= CLUSTER_MIN_SIZE:
            clusters.append(current)

        if not clusters:
            self.publish_empty_markers(scan.header.stamp)
            return

        # Pick the closest forward cluster
        best_cluster_pts = None
        best_x_mean = float('inf')
        for idx_list in clusters:
            cpts = points[idx_list]
            x_mean = np.mean(cpts[:, 0])
            if x_mean > 0.1 and x_mean < best_x_mean:
                x_span = np.max(cpts[:, 0]) - np.min(cpts[:, 0])
                y_span = np.max(cpts[:, 1]) - np.min(cpts[:, 1])
                compactness = math.hypot(x_span, y_span)
                if compactness < 1.2:
                    best_cluster_pts = cpts
                    best_x_mean = x_mean

        if best_cluster_pts is None:
            self.publish_empty_markers(scan.header.stamp)
            return

        # === Compute Steering ===
        lateral_dev = float(np.mean(best_cluster_pts[:, 1]))
        width = float(np.max(best_cluster_pts[:, 1]) - np.min(best_cluster_pts[:, 1]))

        kp = STEERING_GAIN
        kw = WIDTH_GAIN
        steering_raw = kp * lateral_dev + kw * (width - W0)
        steering_raw = float(np.clip(steering_raw, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE))

        # EMA smoothing
        self.filtered_steering = (
            (1.0 - SMOOTHING_ALPHA) * self.filtered_steering
            + SMOOTHING_ALPHA * steering_raw
        )
        self.computed_steering = float(
            np.clip(self.filtered_steering, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
        )

        # === Publish steering + throttle commands ===
        self.steer_pub.publish(Float32(data=self.computed_steering))
        self.throttle_pub.publish(Float32(data=TARGET_SPEED))

        # === Visualization ===
        self.publish_cluster_and_steering(scan.header.stamp, best_cluster_pts, lateral_dev, width)

    # === Visualization Functions ===
    def publish_empty_markers(self, stamp):
        m = Marker()
        m.header.frame_id = FRAME_ID
        m.header.stamp = stamp
        m.ns = 'cluster_points'
        m.id = 0
        m.action = Marker.DELETEALL
        self.cluster_pub.publish(m)

        arrow = self._make_steering_arrow(stamp, 0.0)
        self.steer_viz_pub.publish(arrow)

    def publish_cluster_and_steering(self, stamp, cluster_pts, lateral_dev, width):
        # Cluster visualization
        p = Marker()
        p.header.frame_id = FRAME_ID
        p.header.stamp = stamp
        p.ns = 'cluster_points'
        p.id = 1
        p.type = Marker.POINTS
        p.action = Marker.ADD
        p.pose.orientation.w = 1.0
        p.scale.x = DOT_SIZE
        p.scale.y = DOT_SIZE
        p.color.r = 1.0
        p.color.g = 0.2
        p.color.b = 0.2
        p.color.a = 1.0

        R = np.array([
            [math.cos(self.yaw_e), -math.sin(self.yaw_e)],
            [math.sin(self.yaw_e), math.cos(self.yaw_e)]
        ])
        world_pts = (R @ cluster_pts.T).T + np.array([self.x_e, self.y_e])

        from geometry_msgs.msg import Point as Pt
        for wp in world_pts:
            pt = Pt()
            pt.x = float(wp[0])
            pt.y = float(wp[1])
            pt.z = 0.15
            p.points.append(pt)
        self.cluster_pub.publish(p)

        arrow = self._make_steering_arrow(stamp, self.computed_steering)
        self.steer_viz_pub.publish(arrow)

    def _make_steering_arrow(self, stamp, steering_angle):
        from geometry_msgs.msg import Point as Pt
        arrow = Marker()
        arrow.header.frame_id = FRAME_ID
        arrow.header.stamp = stamp
        arrow.ns = 'steering_arrow'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD

        arrow_len = 1.5
        start_x, start_y = self.x_e, self.y_e
        end_x = start_x + arrow_len * math.cos(self.yaw_e + steering_angle)
        end_y = start_y + arrow_len * math.sin(self.yaw_e + steering_angle)

        arrow.points = [
            Pt(x=start_x, y=start_y, z=0.3),
            Pt(x=end_x, y=end_y, z=0.3),
        ]

        arrow.scale.x = 0.15
        arrow.scale.y = 0.25
        arrow.scale.z = 0.3

        mag = min(1.0, abs(steering_angle) / MAX_STEERING_ANGLE)
        arrow.color.r = mag
        arrow.color.g = 1.0 - mag
        arrow.color.b = 0.0
        arrow.color.a = 1.0
        return arrow


def main():
    rclpy.init()
    node = SimpleTargetTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

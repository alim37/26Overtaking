#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from collections import deque
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu
import tf2_ros


def get_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "package.xml").exists():
            return parent
        if (parent / ".git").exists() or (parent / "tracks" / "src").exists():
            return parent
    return Path(__file__).resolve().parents[1]


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class SlamToolboxMapLogger(Node):
    def __init__(self) -> None:
        super().__init__("slam_toolbox_map_logger")

        repo_root = get_repo_root()
        default_output = repo_root / "output" / "slam_runs" / "slam_toolbox_boundary.csv"

        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("pose_topic", "/autodrive/f1tenth_1/ips")
        self.declare_parameter("imu_topic", "/autodrive/f1tenth_1/imu")
        self.declare_parameter("output_path", str(default_output))
        self.declare_parameter("occupied_threshold", 50)
        self.declare_parameter("lap_start_radius_m", 1.5)
        self.declare_parameter("lap_min_distance_m", 35.0)
        self.declare_parameter("save_only_after_lap", True)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "f1tenth_1")
        self.declare_parameter("export_in_ips_frame", True)
        self.declare_parameter("boundary_bubble_radius_m", 0.05)

        self.map_topic = str(self.get_parameter("map_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.output_path = Path(str(self.get_parameter("output_path").value)).expanduser()
        self.occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        self.lap_start_radius_m = float(self.get_parameter("lap_start_radius_m").value)
        self.lap_min_distance_m = float(self.get_parameter("lap_min_distance_m").value)
        self.save_only_after_lap = bool(self.get_parameter("save_only_after_lap").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.export_in_ips_frame = bool(self.get_parameter("export_in_ips_frame").value)
        self.boundary_bubble_radius_m = float(self.get_parameter("boundary_bubble_radius_m").value)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.latest_map: OccupancyGrid | None = None
        self.latest_boundaries: dict[str, list[tuple[int, int, float, float, int, int]]] | None = None

        self.start_pose_xy: tuple[float, float] | None = None
        self.prev_pose_xy: tuple[float, float] | None = None
        self.total_distance_m = 0.0
        self.left_start_zone = False
        self.completed_lap = False
        self.saved_after_lap = False
        self.latest_ips_pose_xy: tuple[float, float] | None = None
        self.latest_ips_yaw: float | None = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, 10)
        self.create_subscription(Point, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(Imu, self.imu_topic, self.imu_cb, 10)

        self.get_logger().info(
            f"SLAM toolbox map logger ready. map_topic={self.map_topic}, pose_topic={self.pose_topic}, "
            f"imu_topic={self.imu_topic}, output={self.output_path}, export_in_ips_frame={self.export_in_ips_frame}"
        )

    def pose_cb(self, msg: Point) -> None:
        current_xy = (float(msg.x), float(msg.y))
        self.latest_ips_pose_xy = current_xy

        if self.start_pose_xy is None:
            self.start_pose_xy = current_xy
            self.prev_pose_xy = current_xy
            self.get_logger().info(f"Map logger start pose locked at ({current_xy[0]:.2f}, {current_xy[1]:.2f})")
            return

        if self.prev_pose_xy is not None:
            self.total_distance_m += math.hypot(current_xy[0] - self.prev_pose_xy[0], current_xy[1] - self.prev_pose_xy[1])
        self.prev_pose_xy = current_xy

        distance_to_start = math.hypot(current_xy[0] - self.start_pose_xy[0], current_xy[1] - self.start_pose_xy[1])
        if distance_to_start > self.lap_start_radius_m:
            self.left_start_zone = True

        if (
            not self.completed_lap
            and self.left_start_zone
            and self.total_distance_m >= self.lap_min_distance_m
            and distance_to_start <= self.lap_start_radius_m
        ):
            self.completed_lap = True
            self.get_logger().info(
                f"Lap 1 complete for map logger at distance {self.total_distance_m:.2f} m. "
                "Saving boundary map on next available map update."
            )
            if self.latest_map is not None:
                self.save_from_map(self.latest_map)

    def imu_cb(self, msg: Imu) -> None:
        self.latest_ips_yaw = quaternion_to_yaw(
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        )

    def map_cb(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg
        if self.save_only_after_lap and not self.completed_lap:
            return
        if self.save_only_after_lap and self.saved_after_lap:
            return
        self.save_from_map(msg)

    def save_from_map(self, msg: OccupancyGrid) -> None:
        occupied_cells_map = self.extract_occupied_cells(msg)
        boundaries_map = self.extract_boundaries(msg, occupied_cells_map)
        bubbled_boundaries_map = self.expand_boundary_mask(msg, boundaries_map)
        transform = self.compute_map_to_ips_transform() if self.export_in_ips_frame else None

        occupied_cells = self.transform_rows_to_ips_frame(occupied_cells_map, transform)
        boundaries = {
            key: self.transform_rows_to_ips_frame(rows, transform)
            for key, rows in boundaries_map.items()
        }
        bubbled_boundaries = {
            key: self.transform_rows_to_ips_frame(rows, transform)
            for key, rows in bubbled_boundaries_map.items()
        }

        self.latest_boundaries = boundaries
        self.write_boundary_csv(self.output_path, occupied_cells)
        self.write_boundary_csv(self.output_path.with_name(f"{self.output_path.stem}_boundary.csv"), boundaries["all"])
        self.write_boundary_csv(self.output_path.with_name(f"{self.output_path.stem}_outer.csv"), boundaries["outer"])
        self.write_boundary_csv(self.output_path.with_name(f"{self.output_path.stem}_inner.csv"), boundaries["inner"])
        self.write_boundary_csv(self.output_path.with_name(f"{self.output_path.stem}_wall_mask.csv"), bubbled_boundaries["all"])
        self.write_boundary_csv(self.output_path.with_name(f"{self.output_path.stem}_outer_bubble.csv"), bubbled_boundaries["outer"])
        self.write_boundary_csv(self.output_path.with_name(f"{self.output_path.stem}_inner_bubble.csv"), bubbled_boundaries["inner"])
        self.write_labeled_csv(self.output_path.with_name(f"{self.output_path.stem}_labeled.csv"), boundaries)
        self.write_boundary_csv(self.output_path.with_name(f"{self.output_path.stem}_mapframe.csv"), occupied_cells_map)
        self.write_boundary_csv(self.output_path.with_name(f"{self.output_path.stem}_boundary_mapframe.csv"), boundaries_map["all"])
        self.write_boundary_csv(self.output_path.with_name(f"{self.output_path.stem}_outer_mapframe.csv"), boundaries_map["outer"])
        self.write_boundary_csv(self.output_path.with_name(f"{self.output_path.stem}_inner_mapframe.csv"), boundaries_map["inner"])
        self.write_boundary_csv(self.output_path.with_name(f"{self.output_path.stem}_wall_mask_mapframe.csv"), bubbled_boundaries_map["all"])
        self.write_boundary_csv(self.output_path.with_name(f"{self.output_path.stem}_outer_bubble_mapframe.csv"), bubbled_boundaries_map["outer"])
        self.write_boundary_csv(self.output_path.with_name(f"{self.output_path.stem}_inner_bubble_mapframe.csv"), bubbled_boundaries_map["inner"])
        self.write_labeled_csv(self.output_path.with_name(f"{self.output_path.stem}_labeled_mapframe.csv"), boundaries_map)
        self.saved_after_lap = True
        self.get_logger().info(
            f"Saved slam_toolbox map cells={len(occupied_cells)}, boundaries={len(boundaries['all'])}, "
            f"outer={len(boundaries['outer'])}, inner={len(boundaries['inner'])}, "
            f"wall_mask={len(bubbled_boundaries['all'])}, bubble={self.boundary_bubble_radius_m:.3f}m"
        )

    def compute_map_to_ips_transform(self) -> tuple[float, float, float] | None:
        if self.latest_ips_pose_xy is None or self.latest_ips_yaw is None:
            self.get_logger().warn("Cannot export walls in IPS frame yet: missing latest IPS pose or IMU yaw.")
            return None

        try:
            transform = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except Exception as exc:
            self.get_logger().warn(f"Cannot export walls in IPS frame yet: missing TF {self.map_frame}->{self.base_frame}: {exc}")
            return None

        map_x = float(transform.transform.translation.x)
        map_y = float(transform.transform.translation.y)
        map_yaw = quaternion_to_yaw(
            float(transform.transform.rotation.x),
            float(transform.transform.rotation.y),
            float(transform.transform.rotation.z),
            float(transform.transform.rotation.w),
        )

        yaw_delta = normalize_angle(self.latest_ips_yaw - map_yaw)
        cos_yaw = math.cos(yaw_delta)
        sin_yaw = math.sin(yaw_delta)
        tx = self.latest_ips_pose_xy[0] - (map_x * cos_yaw - map_y * sin_yaw)
        ty = self.latest_ips_pose_xy[1] - (map_x * sin_yaw + map_y * cos_yaw)
        return tx, ty, yaw_delta

    def transform_rows_to_ips_frame(
        self,
        rows: list[tuple[int, int, float, float, int, int]],
        transform: tuple[float, float, float] | None,
    ) -> list[tuple[int, int, float, float, int, int]]:
        if transform is None:
            return rows

        tx, ty, yaw = transform
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        transformed_rows: list[tuple[int, int, float, float, int, int]] = []
        for cell_x, cell_y, world_x, world_y, hits, label in rows:
            ips_x = world_x * cos_yaw - world_y * sin_yaw + tx
            ips_y = world_x * sin_yaw + world_y * cos_yaw + ty
            transformed_rows.append((cell_x, cell_y, ips_x, ips_y, hits, label))
        return transformed_rows

    def extract_occupied_cells(self, msg: OccupancyGrid) -> list[tuple[int, int, float, float, int, int]]:
        width = int(msg.info.width)
        height = int(msg.info.height)
        resolution = float(msg.info.resolution)
        origin_x = float(msg.info.origin.position.x)
        origin_y = float(msg.info.origin.position.y)
        data = list(msg.data)

        def idx(x: int, y: int) -> int:
            return y * width + x

        def is_occupied(x: int, y: int) -> bool:
            return data[idx(x, y)] >= self.occupied_threshold

        occupied_cells: list[tuple[int, int, float, float, int, int]] = []
        for gy in range(height):
            for gx in range(width):
                if not is_occupied(gx, gy):
                    continue
                world_x = origin_x + (gx + 0.5) * resolution
                world_y = origin_y + (gy + 0.5) * resolution
                hits = int(data[idx(gx, gy)])
                occupied_cells.append((gx, gy, world_x, world_y, hits, -1))
        return occupied_cells

    def extract_boundaries(
        self,
        msg: OccupancyGrid,
        occupied_cells: list[tuple[int, int, float, float, int, int]] | None = None,
    ) -> dict[str, list[tuple[int, int, float, float, int, int]]]:
        width = int(msg.info.width)
        height = int(msg.info.height)
        data = list(msg.data)

        if occupied_cells is None:
            occupied_cells = self.extract_occupied_cells(msg)

        occupied_lookup = {(gx, gy) for gx, gy, *_rest in occupied_cells}
        point_by_cell = {
            (gx, gy): (gx, gy, wx, wy, hits)
            for gx, gy, wx, wy, hits, _label in occupied_cells
        }
        boundary_cells: list[tuple[int, int, float, float, int, int]] = []
        boundary_lookup: set[tuple[int, int]] = set()
        neighbors4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for gx, gy, world_x, world_y, hits, _label in occupied_cells:
            boundary = False
            for dx, dy in neighbors4:
                nx = gx + dx
                ny = gy + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height or (nx, ny) not in occupied_lookup:
                    boundary = True
                    break
            if not boundary:
                continue
            boundary_cells.append((gx, gy, world_x, world_y, hits, -1))
            boundary_lookup.add((gx, gy))

        components = self.connected_components(boundary_lookup)
        components.sort(key=len, reverse=True)

        if len(components) >= 2:
            two = components[:2]
        elif len(components) == 1:
            two = components
        else:
            two = []

        labeled_points: list[tuple[int, int, float, float, int, int]] = []
        if two:
            centroid_x = sum(cell[2] for cell in boundary_cells) / max(len(boundary_cells), 1)
            centroid_y = sum(cell[3] for cell in boundary_cells) / max(len(boundary_cells), 1)

            component_stats: list[tuple[float, int, set[tuple[int, int]]]] = []
            for comp in two:
                mean_radius = sum(
                    math.hypot(point_by_cell[cell][2] - centroid_x, point_by_cell[cell][3] - centroid_y)
                    for cell in comp
                ) / max(len(comp), 1)
                component_stats.append((mean_radius, len(comp), comp))

            component_stats.sort(reverse=True)
            outer_cells = component_stats[0][2]
            inner_cells = component_stats[1][2] if len(component_stats) > 1 else set()

            outer_points = [
                (*point_by_cell[cell], 0)
                for cell in outer_cells
            ]
            inner_points = [
                (*point_by_cell[cell], 1)
                for cell in inner_cells
            ]
            labeled_points = outer_points + inner_points
        else:
            outer_points = []
            inner_points = []

        all_points = [(gx, gy, wx, wy, hits, -1) for gx, gy, wx, wy, hits, _ in boundary_cells]
        return {
            "all": all_points,
            "outer": outer_points,
            "inner": inner_points,
            "labeled": labeled_points,
        }

    def expand_boundary_mask(
        self,
        msg: OccupancyGrid,
        boundaries: dict[str, list[tuple[int, int, float, float, int, int]]],
    ) -> dict[str, list[tuple[int, int, float, float, int, int]]]:
        resolution = float(msg.info.resolution)
        origin_x = float(msg.info.origin.position.x)
        origin_y = float(msg.info.origin.position.y)
        width = int(msg.info.width)
        height = int(msg.info.height)

        if self.boundary_bubble_radius_m <= 0.0:
            return {
                "outer": list(boundaries["outer"]),
                "inner": list(boundaries["inner"]),
                "all": list(boundaries["outer"]) + list(boundaries["inner"]),
            }

        cell_radius = max(1, int(math.ceil(self.boundary_bubble_radius_m / max(resolution, 1e-6))))
        offsets: list[tuple[int, int]] = []
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                if math.hypot(dx * resolution, dy * resolution) <= self.boundary_bubble_radius_m + 1e-9:
                    offsets.append((dx, dy))

        def dilate_rows(rows: list[tuple[int, int, float, float, int, int]], label: int) -> list[tuple[int, int, float, float, int, int]]:
            dilated: dict[tuple[int, int], tuple[int, int, float, float, int, int]] = {}
            for gx, gy, _wx, _wy, hits, _old_label in rows:
                for dx, dy in offsets:
                    nx = gx + dx
                    ny = gy + dy
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    world_x = origin_x + (nx + 0.5) * resolution
                    world_y = origin_y + (ny + 0.5) * resolution
                    dilated[(nx, ny)] = (nx, ny, world_x, world_y, hits, label)
            return list(dilated.values())

        outer = dilate_rows(boundaries["outer"], 0)
        inner = dilate_rows(boundaries["inner"], 1)
        all_rows = {(gx, gy): (gx, gy, wx, wy, hits, -1) for gx, gy, wx, wy, hits, _label in outer + inner}
        return {
            "outer": outer,
            "inner": inner,
            "all": list(all_rows.values()),
        }

    def connected_components(self, cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
        remaining = set(cells)
        components: list[set[tuple[int, int]]] = []
        neighbors8 = (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        )

        while remaining:
            start = remaining.pop()
            comp = {start}
            queue = deque([start])
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in neighbors8:
                    nxt = (cx + dx, cy + dy)
                    if nxt in remaining:
                        remaining.remove(nxt)
                        comp.add(nxt)
                        queue.append(nxt)
            components.append(comp)
        return components

    def write_boundary_csv(self, path: Path, rows: list[tuple[int, int, float, float, int, int]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["cell_x", "cell_y", "world_x_m", "world_y_m", "hits"])
            for cell_x, cell_y, world_x, world_y, hits, _label in rows:
                writer.writerow([cell_x, cell_y, f"{world_x:.6f}", f"{world_y:.6f}", hits])

    def write_labeled_csv(self, path: Path, boundaries: dict[str, list[tuple[int, int, float, float, int, int]]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["cell_x", "cell_y", "world_x_m", "world_y_m", "hits", "boundary_label"])
            for cell_x, cell_y, world_x, world_y, hits, label in boundaries["labeled"]:
                writer.writerow(
                    [
                        cell_x,
                        cell_y,
                        f"{world_x:.6f}",
                        f"{world_y:.6f}",
                        hits,
                        "outer" if label == 0 else "inner",
                    ]
                )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlamToolboxMapLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

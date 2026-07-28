#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Range
from std_msgs.msg import UInt8
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


SENSOR_TOPIC_MAP = [
    ("tof6_1_L", 1),
    ("tof6_1_F", 2),
    ("tof6_1_R", 3),
    ("tof6_1_U", 4),
    ("tof_S", 6),
    ("tof_E", 7),
    ("tof_N", 8),
    ("tof_W", 5),
    ("tof3_1_S", 12),
    ("tof3_1_W", 9),
    ("tof3_1_N", 10),
    ("tof3_1_E", 11),
    ("tof2_N", 13),
    ("tof2_W", 14),
    ("tof2_S", 15),
    ("tof2_E", 16),
    ("tof2_1_E", 17),
    ("tof2_1_S", 18),
    ("tof2_1_W", 19),
    ("tof2_1_N", 20),
]


@dataclass(frozen=True)
class Cylinder:
    center: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]
    radius: float
    height: float


def _normalize_quaternion(
    quaternion_xyzw: Sequence[float],
) -> Tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in quaternion_xyzw))
    if norm <= 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return tuple(float(value) / norm for value in quaternion_xyzw)


def _rotate_vector(
    quaternion_xyzw: Sequence[float],
    vector: Sequence[float],
) -> Tuple[float, float, float]:
    qx, qy, qz, qw = _normalize_quaternion(quaternion_xyzw)
    x, y, z = (float(value) for value in vector)
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    )


def _inverse_rotate_vector(
    quaternion_xyzw: Sequence[float],
    vector: Sequence[float],
) -> Tuple[float, float, float]:
    qx, qy, qz, qw = _normalize_quaternion(quaternion_xyzw)
    return _rotate_vector((-qx, -qy, -qz, qw), vector)


def ray_cylinder_intersection(
    origin: Sequence[float],
    direction: Sequence[float],
    cylinder: Cylinder,
) -> Optional[float]:
    direction_norm = math.sqrt(sum(float(value) * float(value) for value in direction))
    if direction_norm <= 1e-12 or cylinder.radius <= 0.0 or cylinder.height <= 0.0:
        return None

    world_direction = tuple(float(value) / direction_norm for value in direction)
    relative_origin = tuple(
        float(origin[axis]) - cylinder.center[axis] for axis in range(3)
    )
    px, py, pz = _inverse_rotate_vector(
        cylinder.quaternion_xyzw,
        relative_origin,
    )
    dx, dy, dz = _inverse_rotate_vector(
        cylinder.quaternion_xyzw,
        world_direction,
    )

    half_height = 0.5 * cylinder.height
    candidates: List[float] = []
    radial_a = dx * dx + dy * dy
    radial_b = 2.0 * (px * dx + py * dy)
    radial_c = px * px + py * py - cylinder.radius * cylinder.radius
    if radial_a > 1e-12:
        discriminant = radial_b * radial_b - 4.0 * radial_a * radial_c
        if discriminant >= 0.0:
            root = math.sqrt(max(discriminant, 0.0))
            for distance in (
                (-radial_b - root) / (2.0 * radial_a),
                (-radial_b + root) / (2.0 * radial_a),
            ):
                if distance >= 0.0 and abs(pz + distance * dz) <= half_height + 1e-9:
                    candidates.append(distance)

    if abs(dz) > 1e-12:
        for cap_z in (-half_height, half_height):
            distance = (cap_z - pz) / dz
            if distance < 0.0:
                continue
            hit_x = px + distance * dx
            hit_y = py + distance * dy
            if hit_x * hit_x + hit_y * hit_y <= cylinder.radius * cylinder.radius + 1e-9:
                candidates.append(distance)

    return min(candidates) if candidates else None


class FixedObstacleTofRaycast(Node):
    def __init__(self) -> None:
        super().__init__("fixed_obstacle_tof_raycast")

        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("fixed_frame", "base_link")
        self.declare_parameter("obstacle_topic", "/fixed_obstacle_markers")
        self.declare_parameter("inactive_range_m", 0.90)
        self.declare_parameter("minimum_range_m", 0.001)
        self.declare_parameter("maximum_range_m", 0.90)
        self.declare_parameter("range_scale", 0.001)
        self.declare_parameter("max_raw_range", 1000.0)
        self.declare_parameter("field_of_view", 0.12)
        self.declare_parameter("proximity_topic_prefix", "/fake_proximity_distance")
        self.declare_parameter("raw_topic_prefix", "/fake_raw_distance")
        self.declare_parameter("publish_raw_topics", True)
        self.declare_parameter("publish_rmp_flag", True)
        self.declare_parameter("rmp_flag_topic", "/RMP_flag")
        self.declare_parameter("rmp_active_flag_value", 1)
        self.declare_parameter("tf_timeout_sec", 0.01)
        self.declare_parameter("fallback_position", [0.0, 0.0, 0.0])
        self.declare_parameter("fallback_radius", 0.06)
        self.declare_parameter("fallback_height", 0.8)
        self.declare_parameter("use_fallback_until_marker", True)

        self.publish_rate_hz = max(
            float(self.get_parameter("publish_rate_hz").value),
            1.0,
        )
        self.fixed_frame = str(self.get_parameter("fixed_frame").value)
        self.inactive_range_m = max(
            float(self.get_parameter("inactive_range_m").value),
            0.0,
        )
        self.minimum_range_m = max(
            float(self.get_parameter("minimum_range_m").value),
            0.0,
        )
        self.maximum_range_m = max(
            float(self.get_parameter("maximum_range_m").value),
            self.minimum_range_m,
        )
        self.range_scale = max(float(self.get_parameter("range_scale").value), 1e-9)
        self.max_raw_range = max(
            float(self.get_parameter("max_raw_range").value),
            1.0,
        )
        self.field_of_view = max(
            float(self.get_parameter("field_of_view").value),
            0.0,
        )
        self.publish_raw_topics = bool(
            self.get_parameter("publish_raw_topics").value
        )
        self.publish_rmp_flag = bool(
            self.get_parameter("publish_rmp_flag").value
        )
        self.rmp_active_flag_value = int(
            self.get_parameter("rmp_active_flag_value").value
        )
        self.tf_timeout_sec = max(
            float(self.get_parameter("tf_timeout_sec").value),
            0.0,
        )
        self.use_fallback_until_marker = bool(
            self.get_parameter("use_fallback_until_marker").value
        )

        fallback_position = tuple(
            float(value) for value in self.get_parameter("fallback_position").value
        )
        if len(fallback_position) != 3:
            raise ValueError("fallback_position must contain exactly 3 values")
        self.fallback_cylinders = [
            Cylinder(
                center=fallback_position,
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                radius=max(
                    float(self.get_parameter("fallback_radius").value),
                    0.0,
                ),
                height=max(
                    float(self.get_parameter("fallback_height").value),
                    0.0,
                ),
            )
        ]
        self.marker_cylinders: Optional[List[Cylinder]] = None

        proximity_prefix = str(
            self.get_parameter("proximity_topic_prefix").value
        ).strip()
        raw_prefix = str(self.get_parameter("raw_topic_prefix").value).strip()
        self.proximity_publishers: Dict[str, object] = {}
        self.raw_publishers: Dict[str, object] = {}
        for sensor_name, topic_index in SENSOR_TOPIC_MAP:
            self.proximity_publishers[sensor_name] = self.create_publisher(
                Range,
                f"{proximity_prefix}{topic_index}",
                10,
            )
            if self.publish_raw_topics:
                self.raw_publishers[sensor_name] = self.create_publisher(
                    Range,
                    f"{raw_prefix}{topic_index}",
                    10,
                )

        self.rmp_flag_pub = None
        if self.publish_rmp_flag:
            self.rmp_flag_pub = self.create_publisher(
                UInt8,
                str(self.get_parameter("rmp_flag_topic").value),
                10,
            )

        self.create_subscription(
            MarkerArray,
            str(self.get_parameter("obstacle_topic").value),
            self._on_obstacles,
            10,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_log_ns = 0
        self.timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self._publish,
        )
        self.get_logger().info(
            "fixed obstacle ToF raycast started: "
            f"obstacles={self.get_parameter('obstacle_topic').value}, "
            f"frame={self.fixed_frame}, sensors={len(SENSOR_TOPIC_MAP)}"
        )

    def _on_obstacles(self, message: MarkerArray) -> None:
        cylinders: List[Cylinder] = []
        for marker in message.markers:
            if marker.action != Marker.ADD or marker.type != Marker.CYLINDER:
                continue
            if marker.header.frame_id and marker.header.frame_id != self.fixed_frame:
                self.get_logger().warn(
                    "Ignoring fixed cylinder outside the configured frame: "
                    f"{marker.header.frame_id}"
                )
                continue
            radius = 0.5 * max(float(marker.scale.x), float(marker.scale.y))
            height = float(marker.scale.z)
            if radius <= 0.0 or height <= 0.0:
                continue
            cylinders.append(
                Cylinder(
                    center=(
                        float(marker.pose.position.x),
                        float(marker.pose.position.y),
                        float(marker.pose.position.z),
                    ),
                    quaternion_xyzw=(
                        float(marker.pose.orientation.x),
                        float(marker.pose.orientation.y),
                        float(marker.pose.orientation.z),
                        float(marker.pose.orientation.w),
                    ),
                    radius=radius,
                    height=height,
                )
            )
        self.marker_cylinders = cylinders

    def _current_cylinders(self) -> List[Cylinder]:
        if self.marker_cylinders is not None:
            return self.marker_cylinders
        return self.fallback_cylinders if self.use_fallback_until_marker else []

    def _sensor_ray(
        self,
        sensor_name: str,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.fixed_frame,
                sensor_name,
                Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except TransformException:
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        direction = _rotate_vector(
            (rotation.x, rotation.y, rotation.z, rotation.w),
            (1.0, 0.0, 0.0),
        )
        return (
            (translation.x, translation.y, translation.z),
            direction,
        )

    def _make_range(self, sensor_name: str, physical_range_m: float) -> Range:
        message = Range()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = sensor_name
        message.radiation_type = Range.INFRARED
        message.field_of_view = self.field_of_view
        message.min_range = 0.0
        message.max_range = self.max_raw_range
        message.range = min(
            max(physical_range_m / self.range_scale, 0.0),
            self.max_raw_range - 1.0,
        )
        return message

    def _publish(self) -> None:
        cylinders = self._current_cylinders()
        active_hits: List[Tuple[str, float]] = []
        for sensor_name, _ in SENSOR_TOPIC_MAP:
            sensor_ray = self._sensor_ray(sensor_name)
            hit_distance = None
            if sensor_ray is not None:
                origin, direction = sensor_ray
                distances = [
                    distance
                    for cylinder in cylinders
                    if (
                        distance := ray_cylinder_intersection(
                            origin,
                            direction,
                            cylinder,
                        )
                    ) is not None
                ]
                if distances:
                    hit_distance = min(distances)

            physical_range = self.inactive_range_m
            if (
                hit_distance is not None
                and self.minimum_range_m <= hit_distance <= self.maximum_range_m
            ):
                physical_range = hit_distance
                active_hits.append((sensor_name, hit_distance))

            message = self._make_range(sensor_name, physical_range)
            self.proximity_publishers[sensor_name].publish(message)
            raw_publisher = self.raw_publishers.get(sensor_name)
            if raw_publisher is not None:
                raw_publisher.publish(message)

        if self.rmp_flag_pub is not None:
            flag = UInt8()
            flag.data = self.rmp_active_flag_value
            self.rmp_flag_pub.publish(flag)

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_log_ns >= 1_000_000_000:
            summary = ", ".join(
                f"{sensor}={distance:.3f}m" for sensor, distance in active_hits
            )
            self.get_logger().info(
                f"raycast hits: {summary if summary else 'none'}"
            )
            self.last_log_ns = now_ns


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FixedObstacleTofRaycast()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

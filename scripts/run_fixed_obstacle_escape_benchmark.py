#!/usr/bin/env python3
"""Run and visualize a deterministic Escape ON/OFF overlay experiment."""

import argparse
import csv
import html
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState, Range
from std_msgs.msg import Float64MultiArray, UInt8
from visualization_msgs.msg import Marker, MarkerArray


Point3 = Tuple[float, float, float]


def vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def point_distance(lhs: Point3, rhs: Point3) -> float:
    return vector_norm([lhs[index] - rhs[index] for index in range(3)])


def pose_point(message: Pose) -> Point3:
    return (
        float(message.position.x),
        float(message.position.y),
        float(message.position.z),
    )


def finite_debug(values: Sequence[float], index: int) -> float:
    if index >= len(values):
        return 0.0
    value = float(values[index])
    return value if math.isfinite(value) else 0.0


class BenchmarkNode(Node):
    def __init__(self) -> None:
        super().__init__("fixed_obstacle_escape_benchmark")
        self.on_pose: Optional[Pose] = None
        self.off_pose: Optional[Pose] = None
        self.on_joint_velocity_norm = 0.0
        self.off_joint_velocity_norm = 0.0
        self.on_joint_positions = [0.0] * 6
        self.off_joint_positions = [0.0] * 6
        self.on_joint_state_received = False
        self.off_joint_state_received = False
        self.escape_debug: List[float] = []
        self.sensor_ranges_m: Dict[int, float] = {}
        self.sensor_update_times: Dict[int, float] = {}
        self.cylinder: Optional[Tuple[Point3, float, float]] = None

        self.goal_publisher = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.flag_publisher = self.create_publisher(UInt8, "/RMP_flag", 10)
        self.create_subscription(Pose, "/rmp_ee_pose", self._on_on_pose, 10)
        self.create_subscription(
            Pose,
            "/escape_off/rmp_ee_pose",
            self._on_off_pose,
            10,
        )
        self.create_subscription(
            JointState,
            "/escape_on/joint_states",
            self._on_on_joint_state,
            10,
        )
        self.create_subscription(
            JointState,
            "/escape_off/joint_states",
            self._on_off_joint_state,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/escape_on/tangent_escape_rmp_data",
            self._on_escape_debug,
            10,
        )
        self.create_subscription(
            MarkerArray,
            "/fixed_obstacle_markers",
            self._on_fixed_obstacles,
            10,
        )
        for index in range(1, 21):
            self.create_subscription(
                Range,
                f"/fake_proximity_distance{index}",
                lambda message, sensor_index=index: self._on_range(
                    sensor_index, message
                ),
                10,
            )

    def _on_on_pose(self, message: Pose) -> None:
        self.on_pose = message

    def _on_off_pose(self, message: Pose) -> None:
        self.off_pose = message

    def _on_on_joint_state(self, message: JointState) -> None:
        self.on_joint_state_received = True
        self.on_joint_velocity_norm = vector_norm(message.velocity)
        if len(message.position) >= 6:
            self.on_joint_positions = [float(value) for value in message.position[:6]]

    def _on_off_joint_state(self, message: JointState) -> None:
        self.off_joint_state_received = True
        self.off_joint_velocity_norm = vector_norm(message.velocity)
        if len(message.position) >= 6:
            self.off_joint_positions = [float(value) for value in message.position[:6]]

    def _on_escape_debug(self, message: Float64MultiArray) -> None:
        self.escape_debug = list(message.data)

    def _on_range(self, index: int, message: Range) -> None:
        self.sensor_ranges_m[index] = 0.001 * float(message.range)
        self.sensor_update_times[index] = time.monotonic()

    def _on_fixed_obstacles(self, message: MarkerArray) -> None:
        for marker in message.markers:
            if marker.action == Marker.ADD and marker.type == Marker.CYLINDER:
                self.cylinder = (
                    (
                        float(marker.pose.position.x),
                        float(marker.pose.position.y),
                        float(marker.pose.position.z),
                    ),
                    0.5 * max(float(marker.scale.x), float(marker.scale.y)),
                    float(marker.scale.z),
                )
                return

    def publish_goal(
        self,
        position: Point3,
        orientation: Tuple[float, float, float, float],
    ) -> None:
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "base_link"
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = position
        (
            goal.pose.orientation.x,
            goal.pose.orientation.y,
            goal.pose.orientation.z,
            goal.pose.orientation.w,
        ) = orientation
        self.goal_publisher.publish(goal)
        flag = UInt8()
        flag.data = 1
        self.flag_publisher.publish(flag)

    def ready(self) -> bool:
        return (
            self.on_pose is not None
            and self.off_pose is not None
            and self.on_joint_state_received
            and self.off_joint_state_received
            and self.goal_publisher.get_subscription_count() >= 2
        )

    def active_ranges(self, maximum_age_s: float = 0.5) -> List[float]:
        now = time.monotonic()
        return [
            value
            for index, value in self.sensor_ranges_m.items()
            if math.isfinite(value)
            and value < 0.89
            and now - self.sensor_update_times.get(index, 0.0) <= maximum_age_s
        ]

    def snapshot(self, elapsed_s: float, goal: Point3) -> Optional[Dict[str, float]]:
        if self.on_pose is None or self.off_pose is None:
            return None
        on_point = pose_point(self.on_pose)
        off_point = pose_point(self.off_pose)
        debug = self.escape_debug
        physical_ranges = [
            value for value in self.sensor_ranges_m.values()
            if math.isfinite(value)
        ]
        row = {
            "time_s": elapsed_s,
            "on_x": on_point[0],
            "on_y": on_point[1],
            "on_z": on_point[2],
            "off_x": off_point[0],
            "off_y": off_point[1],
            "off_z": off_point[2],
            "on_goal_error_m": point_distance(on_point, goal),
            "off_goal_error_m": point_distance(off_point, goal),
            "on_joint_velocity_norm_rad_s": self.on_joint_velocity_norm,
            "off_joint_velocity_norm_rad_s": self.off_joint_velocity_norm,
            "on_off_separation_m": point_distance(on_point, off_point),
            "minimum_tof_range_m": min(physical_ranges) if physical_ranges else 0.9,
            "escape_active": finite_debug(debug, 0),
            "escape_clearance_m": finite_debug(debug, 2),
            "escape_beta": finite_debug(debug, 3),
            "escape_alpha_distance": finite_debug(debug, 4),
            "escape_alpha_blocking": finite_debug(debug, 5),
            "escape_blockage": finite_debug(debug, 6),
            "escape_scalar_accel_m_s2": finite_debug(debug, 9),
            "escape_metric_scalar": finite_debug(debug, 10),
            "escape_alpha_stuck": finite_debug(debug, 44),
            "escape_stuck_accel_m_s2": finite_debug(debug, 49),
            "escape_goal_progress_m_s": finite_debug(debug, 51),
            "escape_nominal_speed_m_s": finite_debug(debug, 52),
            "escape_max_blockage": finite_debug(debug, 54),
            "escape_distance_accel_m_s2": finite_debug(debug, 63),
            "escape_approach_accel_m_s2": finite_debug(debug, 64),
            "escape_risk_accel_m_s2": finite_debug(debug, 65),
        }
        for joint in range(6):
            row[f"on_joint_{joint + 1}_rad"] = self.on_joint_positions[joint]
            row[f"off_joint_{joint + 1}_rad"] = self.off_joint_positions[joint]
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poses-file",
        default="~/ros2_ws/src/RMP_Proximity-Sensor/config/experiment_goal_poses.yaml",
    )
    parser.add_argument("--start-pose", default="escape_tuning_initial")
    parser.add_argument("--goal-pose", default="escape_tuning_goal")
    parser.add_argument("--start-timeout-sec", type=float, default=12.0)
    parser.add_argument("--start-tolerance-m", type=float, default=0.01)
    parser.add_argument("--start-settle-sec", type=float, default=0.4)
    parser.add_argument("--pre-target-hold-sec", type=float, default=1.0)
    parser.add_argument("--sensor-ready-timeout-sec", type=float, default=8.0)
    parser.add_argument("--duration-sec", type=float, default=12.0)
    parser.add_argument("--sample-rate-hz", type=float, default=50.0)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.02)
    parser.add_argument(
        "--output-directory",
        default="~/ros2_ws/log/tangent_escape_fixed_benchmark",
    )
    parser.add_argument(
        "--params-file",
        default="~/ros2_ws/src/RMP_Proximity-Sensor/config/params.yaml",
    )
    parser.add_argument("--label", default="trial")
    return parser.parse_args()


def read_pose(
    document: Dict[str, object],
    name: str,
) -> Tuple[Point3, Tuple[float, float, float, float]]:
    poses = document.get("poses", {})
    if not isinstance(poses, dict) or name not in poses:
        raise RuntimeError(f"Pose not found: {name}")
    entry = poses[name]
    if not isinstance(entry, dict):
        raise RuntimeError(f"Pose is not a mapping: {name}")
    position = tuple(float(value) for value in entry["position"])
    orientation = tuple(float(value) for value in entry["orientation"])
    if len(position) != 3 or len(orientation) != 4:
        raise RuntimeError(f"Invalid pose dimensions: {name}")
    return position, orientation


def read_fixed_cylinder(params_path: Path) -> Tuple[Point3, float, float]:
    document = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
    parameters = document.get("obstacle_manager", {}).get("ros__parameters", {})
    positions = parameters.get("default_positions", [])
    if len(positions) < 3:
        raise RuntimeError(
            "obstacle_manager.default_positions must contain one XYZ position"
        )
    return (
        tuple(float(value) for value in positions[:3]),
        float(parameters.get("default_radius", 0.06)),
        float(parameters.get("default_height", 0.8)),
    )


def run_phase(
    node: BenchmarkNode,
    position: Point3,
    orientation: Tuple[float, float, float, float],
    duration_s: float,
    rate_hz: float,
    record: bool,
    tolerance_m: float = 0.0,
    settle_s: float = 0.0,
) -> Tuple[List[Dict[str, float]], bool]:
    samples: List[Dict[str, float]] = []
    period_s = 1.0 / max(rate_hz, 1.0)
    start = time.monotonic()
    reached_since: Optional[float] = None
    while rclpy.ok() and time.monotonic() - start < duration_s:
        cycle_start = time.monotonic()
        node.publish_goal(position, orientation)
        rclpy.spin_once(node, timeout_sec=min(period_s * 0.5, 0.01))
        elapsed_s = time.monotonic() - start
        snapshot = node.snapshot(elapsed_s, position)
        if record and snapshot is not None:
            samples.append(snapshot)
        if tolerance_m > 0.0 and node.ready():
            on_error = point_distance(pose_point(node.on_pose), position)
            off_error = point_distance(pose_point(node.off_pose), position)
            settled_motion = max(
                node.on_joint_velocity_norm,
                node.off_joint_velocity_norm,
            ) <= 0.02
            if max(on_error, off_error) <= tolerance_m and settled_motion:
                if reached_since is None:
                    reached_since = time.monotonic()
                if time.monotonic() - reached_since >= settle_s:
                    return samples, True
            else:
                reached_since = None
        sleep_s = period_s - (time.monotonic() - cycle_start)
        if sleep_s > 0.0:
            time.sleep(sleep_s)
    return samples, False


def path_length(rows: Sequence[Dict[str, float]], prefix: str) -> float:
    total = 0.0
    previous: Optional[Point3] = None
    for row in rows:
        point = (row[f"{prefix}_x"], row[f"{prefix}_y"], row[f"{prefix}_z"])
        if previous is not None:
            total += point_distance(previous, point)
        previous = point
    return total


def first_reach_time(
    rows: Sequence[Dict[str, float]],
    field: str,
    tolerance_m: float,
) -> Optional[float]:
    for row in rows:
        if row[field] <= tolerance_m:
            return row["time_s"]
    return None


def local_minimum_dwell(
    rows: Sequence[Dict[str, float]],
    prefix: str,
) -> float:
    dwell = 0.0
    previous: Optional[Dict[str, float]] = None
    for row in rows:
        if previous is None:
            previous = row
            continue
        dt = max(row["time_s"] - previous["time_s"], 0.0)
        joint_speed = row[f"{prefix}_joint_velocity_norm_rad_s"]
        if row[f"{prefix}_goal_error_m"] > 0.05 and joint_speed < 0.02:
            dwell += dt
        previous = row
    return dwell


def summarize(
    rows: Sequence[Dict[str, float]],
    tolerance_m: float,
    start_reached: bool,
) -> Dict[str, object]:
    if not rows:
        raise RuntimeError("No synchronized ON/OFF samples were recorded")

    def maximum(field: str) -> float:
        return max(row[field] for row in rows)

    def mean(field: str) -> float:
        return sum(row[field] for row in rows) / len(rows)

    active_duration = 0.0
    floor_duration = 0.0
    for previous, current in zip(rows, rows[1:]):
        dt = max(current["time_s"] - previous["time_s"], 0.0)
        if current["escape_active"] >= 0.5:
            active_duration += dt
        if current["escape_stuck_accel_m_s2"] > 1e-6:
            floor_duration += dt

    summary = {
        "samples": len(rows),
        "duration_s": rows[-1]["time_s"],
        "start_pose_reached": start_reached,
        "on_final_error_m": rows[-1]["on_goal_error_m"],
        "off_final_error_m": rows[-1]["off_goal_error_m"],
        "on_minimum_error_m": min(row["on_goal_error_m"] for row in rows),
        "off_minimum_error_m": min(row["off_goal_error_m"] for row in rows),
        "on_reach_time_s": first_reach_time(
            rows, "on_goal_error_m", tolerance_m
        ),
        "off_reach_time_s": first_reach_time(
            rows, "off_goal_error_m", tolerance_m
        ),
        "on_path_length_m": path_length(rows, "on"),
        "off_path_length_m": path_length(rows, "off"),
        "on_local_minimum_dwell_s": local_minimum_dwell(rows, "on"),
        "off_local_minimum_dwell_s": local_minimum_dwell(rows, "off"),
        "maximum_on_off_separation_m": maximum("on_off_separation_m"),
        "minimum_tof_range_m": min(row["minimum_tof_range_m"] for row in rows),
        "escape_active_duration_s": active_duration,
        "stuck_floor_duration_s": floor_duration,
        "maximum_blockage": maximum("escape_max_blockage"),
        "maximum_alpha_stuck": maximum("escape_alpha_stuck"),
        "maximum_stuck_accel_m_s2": maximum("escape_stuck_accel_m_s2"),
        "maximum_risk_accel_m_s2": maximum("escape_risk_accel_m_s2"),
        "maximum_scalar_accel_m_s2": maximum("escape_scalar_accel_m_s2"),
        "maximum_metric_scalar": maximum("escape_metric_scalar"),
        "mean_metric_scalar_while_active": (
            sum(
                row["escape_metric_scalar"] for row in rows
                if row["escape_active"] >= 0.5
            ) /
            sum(1 for row in rows if row["escape_active"] >= 0.5)
            if any(row["escape_active"] >= 0.5 for row in rows)
            else 0.0
        ),
        "maximum_on_joint_velocity_norm_rad_s": maximum(
            "on_joint_velocity_norm_rad_s"
        ),
        "maximum_off_joint_velocity_norm_rad_s": maximum(
            "off_joint_velocity_norm_rad_s"
        ),
        "initial_on_joint_positions_rad": [
            rows[0][f"on_joint_{joint}_rad"] for joint in range(1, 7)
        ],
        "initial_off_joint_positions_rad": [
            rows[0][f"off_joint_{joint}_rad"] for joint in range(1, 7)
        ],
    }
    invalid_reasons: List[str] = []
    if not start_reached:
        invalid_reasons.append("start pose did not settle")
    if summary["on_path_length_m"] < 0.05:
        invalid_reasons.append("Escape ON controller did not move")
    if summary["off_path_length_m"] < 0.05:
        invalid_reasons.append("Escape OFF controller did not move")
    if summary["maximum_on_joint_velocity_norm_rad_s"] < 0.05:
        invalid_reasons.append("Escape ON joint-state stream did not show motion")
    if summary["maximum_off_joint_velocity_norm_rad_s"] < 0.05:
        invalid_reasons.append("Escape OFF joint-state stream did not show motion")
    summary["valid"] = not invalid_reasons
    summary["invalid_reasons"] = invalid_reasons
    return summary


def svg_chart(
    rows: Sequence[Dict[str, float]],
    series: Sequence[Tuple[str, str, str]],
    width: int = 760,
    height: int = 250,
) -> str:
    margin = 36
    times = [row["time_s"] for row in rows]
    values = [row[field] for row in rows for field, _, _ in series]
    x_min, x_max = min(times), max(times)
    y_min, y_max = min(values), max(values)
    if y_max - y_min < 1e-12:
        y_max = y_min + 1.0

    def sx(value: float) -> float:
        return margin + (width - 2 * margin) * (value - x_min) / max(
            x_max - x_min, 1e-9
        )

    def sy(value: float) -> float:
        return height - margin - (height - 2 * margin) * (
            value - y_min
        ) / (y_max - y_min)

    lines = []
    legend = []
    for field, label, color in series:
        points = " ".join(
            f"{sx(row['time_s']):.2f},{sy(row[field]):.2f}" for row in rows
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2"/>'
        )
        legend.append(
            f'<span><i style="background:{color}"></i>{html.escape(label)}</span>'
        )
    return (
        f'<div class="legend">{"".join(legend)}</div>'
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" '
        f'y2="{height-margin}" class="axis"/>'
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" '
        f'y2="{height-margin}" class="axis"/>'
        f'<text x="{margin}" y="20">{y_max:.4g}</text>'
        f'<text x="{margin}" y="{height-8}">{y_min:.4g}</text>'
        f'<text x="{width-margin-35}" y="{height-8}">{x_max:.2f}s</text>'
        f'{"".join(lines)}</svg>'
    )


def xy_chart(
    rows: Sequence[Dict[str, float]],
    cylinder: Optional[Tuple[Point3, float, float]],
    width: int = 620,
    height: int = 460,
) -> str:
    margin = 42
    points = [
        (row[f"{prefix}_x"], row[f"{prefix}_y"])
        for row in rows for prefix in ("on", "off")
    ]
    if cylinder is not None:
        center, radius, _ = cylinder
        points.extend(
            [
                (center[0] - radius, center[1] - radius),
                (center[0] + radius, center[1] + radius),
            ]
        )
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    padding = 0.05
    x_min, x_max = x_min - padding, x_max + padding
    y_min, y_max = y_min - padding, y_max + padding

    def sx(value: float) -> float:
        return margin + (width - 2 * margin) * (value - x_min) / max(
            x_max - x_min, 1e-9
        )

    def sy(value: float) -> float:
        return height - margin - (height - 2 * margin) * (
            value - y_min
        ) / max(y_max - y_min, 1e-9)

    paths = []
    for prefix, color in (("on", "#0f9d74"), ("off", "#7d8790")):
        points_text = " ".join(
            f"{sx(row[f'{prefix}_x']):.2f},{sy(row[f'{prefix}_y']):.2f}"
            for row in rows
        )
        paths.append(
            f'<polyline points="{points_text}" fill="none" stroke="{color}" '
            'stroke-width="3"/>'
        )
    obstacle = ""
    if cylinder is not None:
        center, radius, _ = cylinder
        pixel_radius = radius * (width - 2 * margin) / max(x_max - x_min, 1e-9)
        obstacle = (
            f'<circle cx="{sx(center[0]):.2f}" cy="{sy(center[1]):.2f}" '
            f'r="{pixel_radius:.2f}" fill="#e64b35" fill-opacity="0.35" '
            'stroke="#b5291b" stroke-width="2"/>'
        )
    return (
        '<div class="legend"><span><i style="background:#0f9d74"></i>'
        'Escape ON</span><span><i style="background:#7d8790"></i>'
        'Escape OFF</span><span><i style="background:#e64b35"></i>'
        'Cylinder</span></div>'
        f'<svg viewBox="0 0 {width} {height}" role="img">{obstacle}'
        f'{"".join(paths)}</svg>'
    )


def write_outputs(
    rows: Sequence[Dict[str, float]],
    summary: Dict[str, object],
    output_directory: Path,
    label: str,
    cylinder: Optional[Tuple[Point3, float, float]],
) -> Tuple[Path, Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{label}_{stamp}"
    csv_path = output_directory / f"{stem}.csv"
    json_path = output_directory / f"{stem}.json"
    html_path = output_directory / f"{stem}.html"

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary_rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fixed Obstacle Tangent Escape Benchmark</title>
<style>
body {{ margin: 0; font: 14px sans-serif; color: #17212b; background: #f4f6f7; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 24px; margin: 0 0 18px; }}
h2 {{ font-size: 17px; margin: 0 0 12px; }}
section {{ background: white; border: 1px solid #d5dadd; border-radius: 6px;
  padding: 18px; margin-bottom: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(420px,1fr));
  gap: 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #e5e8ea; padding: 7px; text-align: left; }}
th {{ width: 58%; font-weight: 600; }}
svg {{ display: block; width: 100%; background: #fbfcfc; }}
.axis {{ stroke: #87929b; stroke-width: 1; }}
text {{ fill: #5d6972; font-size: 11px; }}
.legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.legend i {{ width: 18px; height: 3px; display: inline-block; }}
</style>
</head>
<body><main>
<h1>Fixed Obstacle Tangent Escape Benchmark</h1>
<div class="grid">
<section><h2>XY path</h2>{xy_chart(rows, cylinder)}</section>
<section><h2>Summary</h2><table>{summary_rows}</table></section>
</div>
<section><h2>Goal error</h2>{svg_chart(rows, [
    ("on_goal_error_m", "Escape ON", "#0f9d74"),
    ("off_goal_error_m", "Escape OFF", "#7d8790"),
])}</section>
<section><h2>Escape gates</h2>{svg_chart(rows, [
    ("escape_blockage", "B", "#d35400"),
    ("escape_alpha_stuck", "alpha_stuck", "#8e44ad"),
    ("escape_stuck_accel_m_s2", "stuck floor accel", "#006d77"),
])}</section>
<section><h2>Escape acceleration</h2>{svg_chart(rows, [
    ("escape_distance_accel_m_s2", "distance", "#1f77b4"),
    ("escape_approach_accel_m_s2", "approach", "#e67e22"),
    ("escape_risk_accel_m_s2", "combined risk", "#c0392b"),
    ("escape_scalar_accel_m_s2", "after damping/limit", "#16a085"),
])}</section>
<section><h2>Effective metric</h2>{svg_chart(rows, [
    ("escape_metric_scalar", "effective metric", "#34495e"),
])}</section>
<section><h2>ON/OFF path separation</h2>{svg_chart(rows, [
    ("on_off_separation_m", "ON/OFF separation", "#0f9d74"),
])}</section>
</main></body></html>"""
    html_path.write_text(document, encoding="utf-8")
    return csv_path, json_path, html_path


def main() -> int:
    args = parse_args()
    poses_path = Path(os.path.expanduser(args.poses_file))
    document = yaml.safe_load(poses_path.read_text(encoding="utf-8")) or {}
    start_position, start_orientation = read_pose(document, args.start_pose)
    goal_position, goal_orientation = read_pose(document, args.goal_pose)
    fallback_cylinder = read_fixed_cylinder(
        Path(os.path.expanduser(args.params_file))
    )

    rclpy.init()
    node = BenchmarkNode()
    node.cylinder = fallback_cylinder
    try:
        wait_start = time.monotonic()
        while rclpy.ok() and not node.ready():
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() - wait_start > 8.0:
                raise RuntimeError(
                    "Timed out waiting for two controllers, TCP poses, and joint states "
                    f"(goal subscribers={node.goal_publisher.get_subscription_count()})"
                )

        _, start_reached = run_phase(
            node,
            start_position,
            start_orientation,
            args.start_timeout_sec,
            args.sample_rate_hz,
            record=False,
            tolerance_m=args.start_tolerance_m,
            settle_s=args.start_settle_sec,
        )
        if not start_reached:
            raise RuntimeError(
                "Both overlay controllers did not settle at the common start pose"
            )
        run_phase(
            node,
            start_position,
            start_orientation,
            args.pre_target_hold_sec,
            args.sample_rate_hz,
            record=False,
        )
        sensor_wait_start = time.monotonic()
        active_ranges = node.active_ranges()
        while (
            rclpy.ok()
            and not active_ranges
            and time.monotonic() - sensor_wait_start < args.sensor_ready_timeout_sec
        ):
            node.publish_goal(start_position, start_orientation)
            rclpy.spin_once(node, timeout_sec=0.1)
            active_ranges = node.active_ranges()
        if not active_ranges:
            raise RuntimeError(
                "A fresh fixed-cylinder raycast hit was not ready at the start pose"
            )
        rows, _ = run_phase(
            node,
            goal_position,
            goal_orientation,
            args.duration_sec,
            args.sample_rate_hz,
            record=True,
        )
        summary = summarize(rows, args.goal_tolerance_m, start_reached)
        paths = write_outputs(
            rows,
            summary,
            Path(os.path.expanduser(args.output_directory)),
            args.label,
            node.cylinder,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"CSV: {paths[0]}")
        print(f"JSON: {paths[1]}")
        print(f"HTML: {paths[2]}")
        return 0 if summary["valid"] else 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

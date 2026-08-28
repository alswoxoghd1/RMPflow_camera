#!/usr/bin/env python3
"""Quantify discontinuities in rmp_goal_logger CSV files.

The logger samples the latest goal at a fixed rate, so most adjacent rows are
duplicates.  This tool first reconstructs distinct goal updates and then
reports update rate, translational step/speed, quaternion step, and threshold
violations.  It uses only the Python standard library.
"""

import argparse
import csv
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


POSE_SUFFIXES = ("x", "y", "z", "qx", "qy", "qz", "qw")


@dataclass
class GoalUpdate:
    time_s: float
    pose: Tuple[float, float, float, float, float, float, float]
    active: bool


@dataclass
class Spike:
    source: str
    elapsed_s: float
    dt_s: float
    position_step_m: float
    implied_speed_m_s: float
    orientation_step_rad: float
    active: bool
    from_xyz: Tuple[float, float, float]
    to_xyz: Tuple[float, float, float]


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(index, len(ordered) - 1))]


def finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite value")
    return result


def row_active(row: Dict[str, str]) -> bool:
    value = row.get("rmp_active", "")
    try:
        return finite_float(value) > 0.5
    except (TypeError, ValueError):
        return False


def read_distinct_updates(
    path: str,
    pose_prefix: str,
    duplicate_epsilon: float,
) -> Tuple[int, List[GoalUpdate]]:
    updates: List[GoalUpdate] = []
    row_count = 0
    previous_pose = None
    columns = tuple(f"{pose_prefix}_{suffix}" for suffix in POSE_SUFFIXES)

    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in ("timestamp_ros_s", *columns) if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        for row in reader:
            row_count += 1
            try:
                time_s = finite_float(row["timestamp_ros_s"])
                pose = tuple(finite_float(row[name]) for name in columns)
            except (TypeError, ValueError):
                continue
            if previous_pose is not None and max(
                abs(current - previous)
                for current, previous in zip(pose, previous_pose)
            ) <= duplicate_epsilon:
                continue
            typed_pose = tuple(pose)
            updates.append(GoalUpdate(time_s, typed_pose, row_active(row)))
            previous_pose = typed_pose
    return row_count, updates


def quaternion_step_rad(first: Sequence[float], second: Sequence[float]) -> float:
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return math.nan
    # q and -q are the same rotation, hence abs(dot).
    dot = abs(sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def optional_stat(values: Sequence[float], operation) -> Optional[float]:
    return operation(values) if values else None


def analyze_file(
    path: str,
    pose_prefix: str,
    duplicate_epsilon: float,
    position_spike_m: float,
    orientation_spike_rad: float,
) -> Tuple[dict, List[Spike]]:
    row_count, updates = read_distinct_updates(path, pose_prefix, duplicate_epsilon)
    intervals: List[float] = []
    position_steps: List[float] = []
    speeds: List[float] = []
    orientation_steps: List[float] = []
    spikes: List[Spike] = []

    start_s = updates[0].time_s if updates else None
    for previous, current in zip(updates, updates[1:]):
        dt_s = current.time_s - previous.time_s
        position_step_m = math.dist(previous.pose[:3], current.pose[:3])
        orientation_step_rad = quaternion_step_rad(previous.pose[3:], current.pose[3:])
        implied_speed_m_s = position_step_m / dt_s if dt_s > 0.0 else math.inf
        if dt_s > 0.0:
            intervals.append(dt_s)
            speeds.append(implied_speed_m_s)
        position_steps.append(position_step_m)
        if math.isfinite(orientation_step_rad):
            orientation_steps.append(orientation_step_rad)

        if current.active and (
            position_step_m >= position_spike_m
            or (
                math.isfinite(orientation_step_rad)
                and orientation_step_rad >= orientation_spike_rad
            )
        ):
            spikes.append(
                Spike(
                    source=os.path.abspath(path),
                    elapsed_s=current.time_s - start_s,
                    dt_s=dt_s,
                    position_step_m=position_step_m,
                    implied_speed_m_s=implied_speed_m_s,
                    orientation_step_rad=orientation_step_rad,
                    active=current.active,
                    from_xyz=tuple(previous.pose[:3]),
                    to_xyz=tuple(current.pose[:3]),
                )
            )

    median_interval = optional_stat(intervals, statistics.median)
    summary = {
        "source": os.path.abspath(path),
        "sample_rows": row_count,
        "distinct_goal_updates": len(updates),
        "duration_s": updates[-1].time_s - updates[0].time_s if len(updates) >= 2 else 0.0,
        "median_update_rate_hz": 1.0 / median_interval if median_interval and median_interval > 0.0 else None,
        "position_step_m": {
            "p50": percentile(position_steps, 0.50),
            "p95": percentile(position_steps, 0.95),
            "max": optional_stat(position_steps, max),
        },
        "implied_speed_m_s": {
            "p95": percentile(speeds, 0.95),
            "max": optional_stat(speeds, max),
        },
        "orientation_step_rad": {
            "p95": percentile(orientation_steps, 0.95),
            "max": optional_stat(orientation_steps, max),
        },
        "active_spike_count": len(spikes),
        "thresholds": {
            "position_spike_m": position_spike_m,
            "orientation_spike_rad": orientation_spike_rad,
        },
    }
    return summary, spikes


def format_value(value: Optional[float], scale: float = 1.0, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value * scale:.{digits}f}"


def print_summary(summary: dict) -> None:
    print(os.path.basename(summary["source"]))
    print(
        "  duration={duration:.1f}s rows={rows} updates={updates} median_rate={rate}Hz".format(
            duration=summary["duration_s"],
            rows=summary["sample_rows"],
            updates=summary["distinct_goal_updates"],
            rate=format_value(summary["median_update_rate_hz"], digits=1),
        )
    )
    position = summary["position_step_m"]
    speed = summary["implied_speed_m_s"]
    orientation = summary["orientation_step_rad"]
    print(
        "  position_step_mm p50/p95/max={}/{}/{}; speed_m_s p95/max={}/{}".format(
            format_value(position["p50"], 1000.0),
            format_value(position["p95"], 1000.0),
            format_value(position["max"], 1000.0),
            format_value(speed["p95"]),
            format_value(speed["max"]),
        )
    )
    print(
        "  orientation_step_deg p95/max={}/{}; active_spikes={}".format(
            format_value(orientation["p95"], 180.0 / math.pi),
            format_value(orientation["max"], 180.0 / math.pi),
            summary["active_spike_count"],
        )
    )


def write_spikes(path: str, spikes: Iterable[Spike]) -> None:
    fields = (
        "source",
        "elapsed_s",
        "dt_s",
        "position_step_m",
        "implied_speed_m_s",
        "orientation_step_rad",
        "orientation_step_deg",
        "active",
        "from_x",
        "from_y",
        "from_z",
        "to_x",
        "to_y",
        "to_z",
    )
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for spike in spikes:
            writer.writerow(
                {
                    "source": spike.source,
                    "elapsed_s": spike.elapsed_s,
                    "dt_s": spike.dt_s,
                    "position_step_m": spike.position_step_m,
                    "implied_speed_m_s": spike.implied_speed_m_s,
                    "orientation_step_rad": spike.orientation_step_rad,
                    "orientation_step_deg": math.degrees(spike.orientation_step_rad),
                    "active": int(spike.active),
                    "from_x": spike.from_xyz[0],
                    "from_y": spike.from_xyz[1],
                    "from_z": spike.from_xyz[2],
                    "to_x": spike.to_xyz[0],
                    "to_y": spike.to_xyz[1],
                    "to_z": spike.to_xyz[2],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="rmp_goal_logger CSV file(s)")
    parser.add_argument(
        "--pose-prefix",
        default="external_goal",
        choices=("external_goal", "controller_goal"),
        help="Pose columns to analyze (default: external_goal)",
    )
    parser.add_argument(
        "--duplicate-epsilon",
        type=float,
        default=1e-10,
        help="Per-component equality tolerance used to remove logger duplicates",
    )
    parser.add_argument(
        "--position-spike-mm",
        type=float,
        default=20.0,
        help="Active-goal translation spike threshold (default: 20 mm)",
    )
    parser.add_argument(
        "--orientation-spike-deg",
        type=float,
        default=5.0,
        help="Active-goal rotation spike threshold (default: 5 deg)",
    )
    parser.add_argument("--json-output", help="Optional JSON summary path")
    parser.add_argument("--spike-csv", help="Optional detailed spike CSV path")
    parser.add_argument(
        "--show-worst",
        type=int,
        default=10,
        help="Print this many worst spikes by translation (default: 10)",
    )
    args = parser.parse_args()

    summaries = []
    all_spikes: List[Spike] = []
    for path in args.csv:
        summary, spikes = analyze_file(
            path,
            args.pose_prefix,
            max(args.duplicate_epsilon, 0.0),
            max(args.position_spike_mm, 0.0) * 1e-3,
            math.radians(max(args.orientation_spike_deg, 0.0)),
        )
        summaries.append(summary)
        all_spikes.extend(spikes)
        print_summary(summary)

    if all_spikes and args.show_worst > 0:
        print("worst active translation spikes")
        for spike in sorted(
            all_spikes,
            key=lambda item: item.position_step_m,
            reverse=True,
        )[: args.show_worst]:
            print(
                "  {} +{:.3f}s: {:.1f} mm / {:.1f} deg in {:.4f}s ({:.2f} m/s)".format(
                    os.path.basename(spike.source),
                    spike.elapsed_s,
                    spike.position_step_m * 1000.0,
                    math.degrees(spike.orientation_step_rad),
                    spike.dt_s,
                    spike.implied_speed_m_s,
                )
            )

    document = {
        "pose_prefix": args.pose_prefix,
        "files": summaries,
        "spikes": [asdict(spike) for spike in all_spikes],
    }
    if args.json_output:
        with open(args.json_output, "w") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
    if args.spike_csv:
        write_spikes(args.spike_csv, all_spikes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


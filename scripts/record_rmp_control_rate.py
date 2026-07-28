#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


SCHEMA_VERSION = 1
MIN_SCHEMA_LENGTH = 45

IDX = {
    "schema": 0,
    "target_rate": 1,
    "period": 2,
    "cycle_count": 3,
    "cycle_latest": 4,
    "cycle_total": 5,
    "cycle_max": 6,
    "solve_latest": 7,
    "solve_total": 8,
    "solve_max": 9,
    "work_latest": 10,
    "work_total": 11,
    "work_max": 12,
    "lateness_latest": 13,
    "lateness_total": 14,
    "lateness_max": 15,
    "deadline_misses": 16,
    "raw_count": 17,
    "raw_latest": 18,
    "raw_total": 19,
    "raw_max": 20,
    "raw_gap_5ms": 21,
    "raw_gap_20ms": 22,
    "raw_gap_100ms": 23,
    "measured_qd_norm": 24,
    "measured_qd_norm_node_max": 25,
    "measured_qd_abs": 26,
    "measured_qd_abs_node_max": 27,
    "qdd_norm": 28,
    "qdd_norm_node_max": 29,
    "qdd_abs": 30,
    "qdd_abs_node_max": 31,
    "command_norm": 32,
    "command_norm_node_max": 33,
    "command_abs": 34,
    "command_abs_node_max": 35,
    "cycle_window_max": 36,
    "solve_window_max": 37,
    "work_window_max": 38,
    "lateness_window_max": 39,
    "raw_window_max": 40,
    "pre_solve_total": 41,
    "pre_solve_window_max": 42,
    "post_solve_total": 43,
    "post_solve_window_max": 44,
}


SAMPLE_COLUMNS = (
    ("cycle_count", "cycle_count", 1.0),
    ("cycle_interval_ms", "cycle_latest", 1000.0),
    ("solve_ms", "solve_latest", 1000.0),
    ("work_ms", "work_latest", 1000.0),
    ("start_lateness_ms", "lateness_latest", 1000.0),
    ("deadline_miss_count", "deadline_misses", 1.0),
    ("raw_state_count", "raw_count", 1.0),
    ("raw_state_interval_ms", "raw_latest", 1000.0),
    ("raw_gap_over_5ms_count", "raw_gap_5ms", 1.0),
    ("raw_gap_over_20ms_count", "raw_gap_20ms", 1.0),
    ("raw_gap_over_100ms_count", "raw_gap_100ms", 1.0),
    ("measured_qd_norm_rad_s", "measured_qd_norm", 1.0),
    ("measured_qd_max_abs_rad_s", "measured_qd_abs", 1.0),
    ("qdd_norm_rad_s2", "qdd_norm", 1.0),
    ("qdd_max_abs_rad_s2", "qdd_abs", 1.0),
    ("command_norm", "command_norm", 1.0),
    ("command_max_abs", "command_abs", 1.0),
    ("cycle_window_max_ms", "cycle_window_max", 1000.0),
    ("solve_window_max_ms", "solve_window_max", 1000.0),
    ("work_window_max_ms", "work_window_max", 1000.0),
    ("lateness_window_max_ms", "lateness_window_max", 1000.0),
    ("raw_state_window_max_ms", "raw_window_max", 1000.0),
    ("pre_solve_window_max_ms", "pre_solve_window_max", 1000.0),
    ("post_solve_window_max_ms", "post_solve_window_max", 1000.0),
)

VECTOR_SAMPLE_COLUMNS = (
    ("measured_q", 45, "rad"),
    ("measured_qd", 51, "rad_s"),
    ("qdd", 57, "rad_s2"),
    ("command", 63, "value"),
    ("pre_guard_command", 97, "value"),
    ("command_step", 103, "value"),
)

OPTIONAL_SAMPLE_COLUMNS = (
    ("velocity_filter_update_count", 69, 1.0),
    ("velocity_filter_synced_dt_window_max_ms", 70, 1000.0),
    ("velocity_filter_input_reject_count", 95, 1.0),
    ("velocity_filter_output_reject_count", 96, 1.0),
    ("velocity_slew_clamp_count", 109, 1.0),
    ("velocity_slew_clamped", 110, 1.0),
    ("controller_packet_time_used_count", 111, 1.0),
    ("controller_packet_time_fallback_count", 112, 1.0),
    ("controller_packet_time_duplicate_count", 113, 1.0),
    ("controller_packet_time_reset_count", 114, 1.0),
    ("controller_packet_time_dt_window_max_ms", 115, 1000.0),
)

DIAGNOSTIC_VECTOR_SAMPLE_COLUMNS = (
    ("raw_position_step_abs_window_max", 71, "rad"),
    ("raw_velocity_abs_window_max", 77, "rad_s"),
    ("velocity_filter_residual_abs_window_max", 83, "rad"),
    ("filtered_velocity_abs_window_max", 89, "rad_s"),
)


class TimingRecorder(Node):
    def __init__(self, topic):
        super().__init__("rmp_control_rate_recorder")
        self.samples = []
        self.invalid_messages = 0
        self.create_subscription(Float64MultiArray, topic, self._on_timing, 10)

    def _on_timing(self, message):
        data = list(message.data)
        if (
            len(data) < MIN_SCHEMA_LENGTH
            or int(round(data[IDX["schema"]])) != SCHEMA_VERSION
            or not all(math.isfinite(value) for value in data[:MIN_SCHEMA_LENGTH])
        ):
            self.invalid_messages += 1
            return
        self.samples.append((time.monotonic(), data))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record low-overhead RMP controller timing or compare the latest "
            "50/100/200 Hz records."
        )
    )
    parser.add_argument("--topic", default="/rmp_control_timing")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--wait-timeout", type=float, default=30.0)
    parser.add_argument("--expected-rate", type=float)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--output-directory",
        default=os.path.expanduser("~/ros2_ws/log/rmpflow_rate_benchmark"),
    )
    parser.add_argument(
        "--compare-latest",
        action="store_true",
        help="Compare the newest JSON records for 50, 100, and 200 Hz.",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help=(
            "Save every received timing message beside the JSON summary so motion "
            "spikes can be correlated with controller and RB10 input timing."
        ),
    )
    return parser.parse_args()


def save_timing_samples(samples, output_path):
    first_time = samples[0][0]
    vector_fieldnames = [
        f"{prefix}_{joint + 1}_{unit}"
        for prefix, _, unit in VECTOR_SAMPLE_COLUMNS
        for joint in range(6)
    ]
    optional_fieldnames = [name for name, _, _ in OPTIONAL_SAMPLE_COLUMNS]
    diagnostic_vector_fieldnames = [
        f"{prefix}_{joint + 1}_{unit}"
        for prefix, _, unit in DIAGNOSTIC_VECTOR_SAMPLE_COLUMNS
        for joint in range(6)
    ]
    fieldnames = (
        ["elapsed_s"] +
        [name for name, _, _ in SAMPLE_COLUMNS] +
        vector_fieldnames +
        optional_fieldnames +
        diagnostic_vector_fieldnames
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for received_time, data in samples:
            row = {"elapsed_s": received_time - first_time}
            for column, key, scale in SAMPLE_COLUMNS:
                row[column] = data[IDX[key]] * scale
            for prefix, start, unit in VECTOR_SAMPLE_COLUMNS:
                for joint in range(6):
                    column = f"{prefix}_{joint + 1}_{unit}"
                    index = start + joint
                    row[column] = data[index] if index < len(data) else float("nan")
            for column, index, scale in OPTIONAL_SAMPLE_COLUMNS:
                row[column] = data[index] * scale if index < len(data) else float("nan")
            for prefix, start, unit in DIAGNOSTIC_VECTOR_SAMPLE_COLUMNS:
                for joint in range(6):
                    column = f"{prefix}_{joint + 1}_{unit}"
                    index = start + joint
                    row[column] = data[index] if index < len(data) else float("nan")
            writer.writerow(row)


def spin_until(node, predicate, timeout_sec):
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and not predicate() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    return predicate()


def process_cpu_snapshot(pid):
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = stat_text.rsplit(")", 1)[1].split()
        ticks = int(fields[11]) + int(fields[12])
        return time.monotonic(), ticks
    except (OSError, IndexError, ValueError):
        return None


def find_controller_pid():
    matches = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            executable = (proc_dir / "exe").resolve(strict=True)
            if executable.name != "rmpflow_controller":
                continue
            command = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            continue
        if "rmpflow_controller" in command and "record_rmp_control_rate" not in command:
            matches.append((int(proc_dir.name), command.strip()))
    if not matches:
        return None, []
    matches.sort(key=lambda item: item[0])
    return matches[-1][0], matches


def find_bridge_pid():
    matches = []
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            continue
        if (
            "rb10_api_bridge.py" in command
            and "record_rmp_control_rate" not in command
        ):
            matches.append((int(proc_dir.name), command.strip()))
    if not matches:
        return None, []
    matches.sort(key=lambda item: item[0])
    return matches[-1][0], matches


def delta(first, last, key):
    value = last[IDX[key]] - first[IDX[key]]
    if value < -1e-9:
        raise RuntimeError(f"Timing counter reset while recording: {key}")
    return max(value, 0.0)


def maximum(samples, key):
    values = [data[IDX[key]] for _, data in samples[1:]]
    return max(values) if values else samples[-1][1][IDX[key]]


def optional_maximum(samples, start_index, count=1, scale=1.0):
    values = [
        data[index] * scale
        for _, data in samples[1:]
        for index in range(start_index, start_index + count)
        if index < len(data) and math.isfinite(data[index])
    ]
    return max(values) if values else None


def optional_max_abs(samples, start_index, count=1, scale=1.0):
    values = [
        abs(data[index]) * scale
        for _, data in samples[1:]
        for index in range(start_index, start_index + count)
        if index < len(data) and math.isfinite(data[index])
    ]
    return max(values) if values else None


def cpu_percent_between(cpu_start, cpu_end):
    if cpu_start is None or cpu_end is None:
        return None
    cpu_elapsed = cpu_end[0] - cpu_start[0]
    cpu_ticks = cpu_end[1] - cpu_start[1]
    if cpu_elapsed <= 0.0 or cpu_ticks < 0:
        return None
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    return 100.0 * cpu_ticks / ticks_per_second / cpu_elapsed


def summarize(
    samples,
    pid,
    cpu_start,
    cpu_end,
    bridge_pid,
    bridge_cpu_start,
    bridge_cpu_end,
    label,
    topic,
):
    first_time, first = samples[0]
    last_time, last = samples[-1]
    elapsed = max(last_time - first_time, 1e-9)
    cycle_count = delta(first, last, "cycle_count")
    cycle_total = delta(first, last, "cycle_total")
    solve_total = delta(first, last, "solve_total")
    pre_solve_total = delta(first, last, "pre_solve_total")
    post_solve_total = delta(first, last, "post_solve_total")
    work_total = delta(first, last, "work_total")
    lateness_total = delta(first, last, "lateness_total")
    deadline_misses = delta(first, last, "deadline_misses")
    raw_count = delta(first, last, "raw_count")
    raw_total = delta(first, last, "raw_total")
    target_rate = last[IDX["target_rate"]]
    period = last[IDX["period"]]

    cpu_percent = cpu_percent_between(cpu_start, cpu_end)
    bridge_cpu_percent = cpu_percent_between(bridge_cpu_start, bridge_cpu_end)

    cycle_mean = cycle_total / cycle_count if cycle_count > 0.0 else 0.0
    achieved_rate = 1.0 / cycle_mean if cycle_mean > 0.0 else cycle_count / elapsed
    solve_mean = solve_total / cycle_count if cycle_count > 0.0 else 0.0
    pre_solve_mean = pre_solve_total / cycle_count if cycle_count > 0.0 else 0.0
    post_solve_mean = post_solve_total / cycle_count if cycle_count > 0.0 else 0.0
    work_mean = work_total / cycle_count if cycle_count > 0.0 else 0.0
    lateness_mean = lateness_total / cycle_count if cycle_count > 0.0 else 0.0
    miss_percent = 100.0 * deadline_misses / cycle_count if cycle_count > 0.0 else 0.0
    raw_rate = raw_count / raw_total if raw_total > 0.0 else None

    velocity_filter_update_rate = None
    velocity_filter_input_reject_count = None
    velocity_filter_output_reject_count = None
    if len(first) > 69 and len(last) > 69:
        filter_updates = max(last[69] - first[69], 0.0)
        velocity_filter_update_rate = filter_updates / elapsed
    if len(first) > 96 and len(last) > 96:
        velocity_filter_input_reject_count = int(round(max(last[95] - first[95], 0.0)))
        velocity_filter_output_reject_count = int(round(max(last[96] - first[96], 0.0)))
    velocity_slew_clamp_count = None
    if len(first) > 109 and len(last) > 109:
        velocity_slew_clamp_count = int(round(max(last[109] - first[109], 0.0)))
    controller_packet_time_used_count = None
    controller_packet_time_fallback_count = None
    controller_packet_time_duplicate_count = None
    controller_packet_time_reset_count = None
    if len(first) > 114 and len(last) > 114:
        controller_packet_time_used_count = int(round(max(last[111] - first[111], 0.0)))
        controller_packet_time_fallback_count = int(
            round(max(last[112] - first[112], 0.0))
        )
        controller_packet_time_duplicate_count = int(
            round(max(last[113] - first[113], 0.0))
        )
        controller_packet_time_reset_count = int(
            round(max(last[114] - first[114], 0.0))
        )

    rate_ratio = achieved_rate / target_rate if target_rate > 0.0 else 0.0
    status = "PASS" if rate_ratio >= 0.95 and miss_percent <= 1.0 else "WARN"

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "label": label,
        "topic": topic,
        "controller_pid": pid,
        "bridge_pid": bridge_pid,
        "sample_count": len(samples),
        "observation_duration_sec": elapsed,
        "target_rate_hz": target_rate,
        "period_budget_ms": period * 1000.0,
        "cycle_count": int(round(cycle_count)),
        "achieved_rate_hz": achieved_rate,
        "rate_ratio": rate_ratio,
        "solve_mean_ms": solve_mean * 1000.0,
        "solve_max_ms": maximum(samples, "solve_window_max") * 1000.0,
        "pre_solve_mean_ms": pre_solve_mean * 1000.0,
        "pre_solve_max_ms": maximum(samples, "pre_solve_window_max") * 1000.0,
        "post_solve_mean_ms": post_solve_mean * 1000.0,
        "post_solve_max_ms": maximum(samples, "post_solve_window_max") * 1000.0,
        "work_mean_ms": work_mean * 1000.0,
        "work_max_ms": maximum(samples, "work_window_max") * 1000.0,
        "work_budget_percent": 100.0 * work_mean / period if period > 0.0 else 0.0,
        "lateness_mean_ms": lateness_mean * 1000.0,
        "lateness_max_ms": maximum(samples, "lateness_window_max") * 1000.0,
        "deadline_miss_count": int(round(deadline_misses)),
        "deadline_miss_percent": miss_percent,
        "raw_state_rate_hz": raw_rate,
        "raw_state_max_gap_ms": maximum(samples, "raw_window_max") * 1000.0,
        "raw_gap_over_5ms_count": int(round(delta(first, last, "raw_gap_5ms"))),
        "raw_gap_over_20ms_count": int(round(delta(first, last, "raw_gap_20ms"))),
        "raw_gap_over_100ms_count": int(round(delta(first, last, "raw_gap_100ms"))),
        "velocity_filter_update_rate_hz": velocity_filter_update_rate,
        "velocity_filter_input_reject_count": velocity_filter_input_reject_count,
        "velocity_filter_output_reject_count": velocity_filter_output_reject_count,
        "velocity_filter_synced_dt_max_ms": optional_maximum(
            samples, 70, scale=1000.0
        ),
        "raw_position_step_abs_max_rad": optional_maximum(samples, 71, count=6),
        "raw_velocity_abs_max_rad_s": optional_maximum(samples, 77, count=6),
        "velocity_filter_residual_abs_max_rad": optional_maximum(
            samples, 83, count=6
        ),
        "filtered_velocity_abs_max_rad_s": optional_maximum(samples, 89, count=6),
        "pre_guard_command_abs_max": optional_max_abs(samples, 97, count=6),
        "command_step_abs_max": optional_max_abs(samples, 103, count=6),
        "velocity_slew_clamp_count": velocity_slew_clamp_count,
        "controller_packet_time_used_count": controller_packet_time_used_count,
        "controller_packet_time_fallback_count": controller_packet_time_fallback_count,
        "controller_packet_time_duplicate_count": controller_packet_time_duplicate_count,
        "controller_packet_time_reset_count": controller_packet_time_reset_count,
        "controller_packet_time_dt_max_ms": optional_maximum(
            samples, 115, scale=1000.0
        ),
        "measured_qd_norm_node_max_rad_s": last[IDX["measured_qd_norm_node_max"]],
        "measured_qd_abs_node_max_rad_s": last[IDX["measured_qd_abs_node_max"]],
        "qdd_norm_node_max_rad_s2": last[IDX["qdd_norm_node_max"]],
        "qdd_abs_node_max_rad_s2": last[IDX["qdd_abs_node_max"]],
        "command_norm_node_max": last[IDX["command_norm_node_max"]],
        "command_abs_node_max": last[IDX["command_abs_node_max"]],
        "process_cpu_percent": cpu_percent,
        "bridge_process_cpu_percent": bridge_cpu_percent,
        "status": status,
    }


def fmt(value, digits=2):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def print_summary(summary):
    print(f"Target rate: {summary['target_rate_hz']:.1f} Hz")
    print(
        "Control: "
        f"actual={summary['achieved_rate_hz']:.2f} Hz, "
        f"deadline misses={summary['deadline_miss_count']} "
        f"({summary['deadline_miss_percent']:.2f}%)"
    )
    print(
        "Compute: "
        f"pre/solve/post mean={summary['pre_solve_mean_ms']:.3f}/"
        f"{summary['solve_mean_ms']:.3f}/{summary['post_solve_mean_ms']:.3f} ms, "
        f"whole-cycle work mean/max={summary['work_mean_ms']:.3f}/"
        f"{summary['work_max_ms']:.3f} ms"
    )
    print(
        "RB10 raw state: "
        f"rate={fmt(summary['raw_state_rate_hz'])} Hz, "
        f"max gap={summary['raw_state_max_gap_ms']:.2f} ms, "
        f">20 ms={summary['raw_gap_over_20ms_count']}, "
        f">100 ms={summary['raw_gap_over_100ms_count']}"
    )
    print(
        "Peaks: "
        f"measured qd norm={summary['measured_qd_norm_node_max_rad_s']:.3f} rad/s, "
        f"qdd norm={summary['qdd_norm_node_max_rad_s2']:.3f} rad/s^2"
    )
    if summary.get("velocity_filter_update_rate_hz") is not None:
        print(
            "Velocity filter: "
            f"updates={summary['velocity_filter_update_rate_hz']:.2f} Hz, "
            f"sync dt max={fmt(summary.get('velocity_filter_synced_dt_max_ms'), 3)} ms, "
            f"raw step max={fmt(summary.get('raw_position_step_abs_max_rad'), 6)} rad, "
            f"raw qd max={fmt(summary.get('raw_velocity_abs_max_rad_s'), 3)} rad/s, "
            f"residual max={fmt(summary.get('velocity_filter_residual_abs_max_rad'), 6)} rad, "
            f"output max={fmt(summary.get('filtered_velocity_abs_max_rad_s'), 3)} rad/s, "
            f"rejects={summary.get('velocity_filter_input_reject_count', 'n/a')}/"
            f"{summary.get('velocity_filter_output_reject_count', 'n/a')} (input/output)"
        )
    if summary.get("command_step_abs_max") is not None:
        print(
            "Command guard: "
            f"pre-guard max={fmt(summary.get('pre_guard_command_abs_max'), 4)}, "
            f"sent step max={fmt(summary.get('command_step_abs_max'), 4)}, "
            f"slew clamps={summary.get('velocity_slew_clamp_count', 'n/a')}"
        )
    if summary.get("controller_packet_time_used_count") is not None:
        print(
            "Controller packet clock: "
            f"used={summary['controller_packet_time_used_count']}, "
            f"fallback={summary['controller_packet_time_fallback_count']}, "
            f"duplicates={summary['controller_packet_time_duplicate_count']}, "
            f"resets={summary['controller_packet_time_reset_count']}, "
            f"dt max={fmt(summary.get('controller_packet_time_dt_max_ms'), 3)} ms"
        )
    print(f"Controller process CPU: {fmt(summary['process_cpu_percent'], 1)}%")
    print(f"RB10 bridge process CPU: {fmt(summary.get('bridge_process_cpu_percent'), 1)}%")
    print(f"Result: {summary['status']}")


def load_latest_by_rate(directory, expected_rates):
    latest = {}
    for path in directory.glob("rmp_control_rate_*Hz_*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            rate = float(record["target_rate_hz"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        for expected in expected_rates:
            if abs(rate - expected) <= 0.5:
                previous = latest.get(expected)
                if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
                    latest[expected] = (path, record)
    return latest


def print_table(rows):
    headers = [
        "Hz",
        "Actual",
        "CPU%",
        "Work avg/max ms",
        "Miss%",
        "Late max ms",
        "Raw Hz",
        "Raw max gap ms",
        ">20ms",
        "Status",
    ]
    values = []
    for row in rows:
        values.append([
            f"{row['target_rate_hz']:.0f}",
            f"{row['achieved_rate_hz']:.1f}",
            fmt(row.get("process_cpu_percent"), 1),
            f"{row['work_mean_ms']:.2f}/{row['work_max_ms']:.2f}",
            f"{row['deadline_miss_percent']:.1f}",
            f"{row['lateness_max_ms']:.2f}",
            fmt(row.get("raw_state_rate_hz"), 1),
            f"{row['raw_state_max_gap_ms']:.1f}",
            str(row["raw_gap_over_20ms_count"]),
            row["status"],
        ])
    widths = [len(header) for header in headers]
    for row in values:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    line = "  ".join(header.ljust(width) for header, width in zip(headers, widths))
    print(line)
    print("  ".join("-" * width for width in widths))
    for row in values:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def compare_latest(output_directory):
    expected_rates = [50.0, 100.0, 200.0]
    latest = load_latest_by_rate(output_directory, expected_rates)
    missing = [rate for rate in expected_rates if rate not in latest]
    if missing:
        rates = ", ".join(f"{rate:.0f}" for rate in missing)
        print(f"Error: missing benchmark JSON for {rates} Hz in {output_directory}", file=sys.stderr)
        return 2

    rows = [latest[rate][1] for rate in expected_rates]
    print_table(rows)
    print("\nFiles:")
    for rate in expected_rates:
        print(f"  {rate:.0f} Hz: {latest[rate][0]}")

    low = rows[0]
    high = rows[-1]
    print("\nInterpretation:")
    if (
        low["work_budget_percent"] > 100.0
        and low["deadline_miss_percent"] > 5.0
    ):
        print(
            "  Per-cycle controller work already exceeds the 50 Hz budget. "
            "Controller computation/CPU throughput is strongly implicated."
        )
    elif (
        high["deadline_miss_percent"] > max(5.0, low["deadline_miss_percent"] + 3.0)
        and high["work_budget_percent"] > low["work_budget_percent"] + 15.0
    ):
        print(
            "  Control-rate/CPU load is strongly implicated: deadline misses and "
            "budget use rise at 200 Hz."
        )
    elif any(row["raw_gap_over_20ms_count"] > 0 for row in rows):
        print(
            "  RB10 input gaps are present. Compare their timing with motion spikes; "
            "CPU load alone is not established."
        )
    else:
        print(
            "  No clear rate-dependent overload signature. Inspect velocity estimation "
            "and command generation next."
        )
    return 0


def record(args):
    if args.duration <= 0.0 or args.warmup < 0.0 or args.wait_timeout <= 0.0:
        print("Error: duration/wait-timeout must be positive and warmup nonnegative.", file=sys.stderr)
        return 2

    rclpy.init()
    node = TimingRecorder(args.topic)
    try:
        print(f"Waiting for {args.topic} ...")
        if not spin_until(node, lambda: bool(node.samples), args.wait_timeout):
            print(
                f"Error: no compatible schema-{SCHEMA_VERSION} timing message received.",
                file=sys.stderr,
            )
            return 2

        if args.warmup > 0.0:
            warmup_deadline = time.monotonic() + args.warmup
            while rclpy.ok() and time.monotonic() < warmup_deadline:
                rclpy.spin_once(node, timeout_sec=0.1)

        node.samples.clear()
        if not spin_until(node, lambda: bool(node.samples), args.wait_timeout):
            print("Error: timing stream stopped after warmup.", file=sys.stderr)
            return 2

        target_rate = node.samples[0][1][IDX["target_rate"]]
        if args.expected_rate is not None and abs(target_rate - args.expected_rate) > 0.5:
            print(
                f"Error: expected {args.expected_rate:.1f} Hz but controller reports "
                f"{target_rate:.1f} Hz.",
                file=sys.stderr,
            )
            return 2

        pid, matches = find_controller_pid()
        if len(matches) > 1:
            print(
                "Warning: multiple controller processes found; CPU is measured from "
                f"newest PID {pid}.",
                file=sys.stderr,
            )
        cpu_start = process_cpu_snapshot(pid) if pid is not None else None
        if pid is None:
            print("Warning: controller PID not found; CPU will be n/a.", file=sys.stderr)
        bridge_pid, bridge_matches = find_bridge_pid()
        if len(bridge_matches) > 1:
            print(
                "Warning: multiple RB10 bridge processes found; CPU is measured from "
                f"newest PID {bridge_pid}.",
                file=sys.stderr,
            )

        print(
            f"Recording {target_rate:.1f} Hz for {args.duration:.1f} s "
            f"(controller PID {pid if pid is not None else 'n/a'}, "
            f"bridge PID {bridge_pid if bridge_pid is not None else 'n/a'}) ..."
        )
        bridge_cpu_start = (
            process_cpu_snapshot(bridge_pid) if bridge_pid is not None else None
        )
        stop_time = time.monotonic() + args.duration
        while rclpy.ok() and time.monotonic() < stop_time:
            rclpy.spin_once(node, timeout_sec=0.1)
        cpu_end = process_cpu_snapshot(pid) if pid is not None else None
        bridge_cpu_end = (
            process_cpu_snapshot(bridge_pid) if bridge_pid is not None else None
        )

        if len(node.samples) < 2:
            print("Error: fewer than two valid timing samples received.", file=sys.stderr)
            return 2
        summary = summarize(
            node.samples,
            pid,
            cpu_start,
            cpu_end,
            bridge_pid,
            bridge_cpu_start,
            bridge_cpu_end,
            args.label,
            args.topic,
        )
        output_directory = Path(args.output_directory).expanduser()
        output_directory.mkdir(parents=True, exist_ok=True)
        rate_token = f"{summary['target_rate_hz']:g}".replace(".", "p")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_directory / (
            f"rmp_control_rate_{rate_token}Hz_{timestamp}.json"
        )
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        samples_path = None
        if args.save_samples:
            samples_path = output_path.with_name(output_path.stem + "_samples.csv")
            save_timing_samples(node.samples, samples_path)
        print_summary(summary)
        print(f"Saved: {output_path}")
        if samples_path is not None:
            print(f"Timing samples: {samples_path}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; no result saved.", file=sys.stderr)
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    args = parse_args()
    output_directory = Path(args.output_directory).expanduser()
    if args.compare_latest:
        return compare_latest(output_directory)
    return record(args)


if __name__ == "__main__":
    raise SystemExit(main())

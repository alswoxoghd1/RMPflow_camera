#!/usr/bin/env python3
"""Sweep the tangent-plane Escape metric isotropy over the fake-proximity scenario.

Each configuration is a separate headless `ros2 launch` run.  The launch is
started in its own process group, left to run for a fixed wall time, then sent
SIGINT so the trace logger closes its CSV cleanly.  Afterwards every run's CSV
is summarised into one comparison table.

The sweep is a measurement harness, not a proof.  It reports what the leaf did;
it does not certify stability, and the tangent-plane leaf is not a GDS.

Typical use:

    python3 scripts/run_tangent_plane_gamma_sweep.py --duration 40

    python3 scripts/run_tangent_plane_gamma_sweep.py \\
        --gammas 0.0 0.5 1.0 --with-risk-damped-baseline --duration 60
"""

import argparse
import csv
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

LAUNCH_PACKAGE = "rb10_rmpflow_rviz"
LAUNCH_FILE = "rb10_rmpflow_fake_proximity.launch.py"

NONFINITE = {"", "nan", "-nan", "+nan", "inf", "-inf", "+inf", "none"}

# Columns the summary reads.  Missing columns degrade a metric to None rather
# than aborting, so an older trace schema still produces a partial table.
COL_TIME = "time_ros_s"
COL_ACTIVE = "tangent_escape_rmp_active"
COL_SCHEMA = "tangent_escape_rmp_schema_id"
COL_CLEARANCE = "tangent_escape_rmp_clearance_m"
COL_VT = "tangent_escape_rmp_tangent_velocity_m_s"
COL_AY = "tangent_escape_rmp_desired_tangent_accel_m_s2"
COL_METRIC = "tangent_escape_rmp_effective_metric_scalar"
COL_PUSH_ALPHA = "tangent_escape_rmp_push_activation"
COL_METRIC_ALPHA = "tangent_escape_rmp_metric_activation"
COL_PUSH_ACCEL = "tangent_escape_rmp_push_accel_m_s2"
COL_DAMPING = "tangent_escape_rmp_tangent_damping_term_m_s2"
COL_DDOT = "tangent_escape_rmp_analytic_clearance_rate_m_s"
COL_PINJ = "tangent_escape_rmp_injection_power"
COL_GAMMA = "tangent_escape_rmp_metric_isotropy"
COL_TANGENT_NORMAL = "tangent_escape_rmp_tangent_normal_component"
COL_GENERATION = "tangent_escape_rmp_handoff_generation"
COL_QDD_NORM = "rmp_joint_accel_norm"
TANGENT_COLS = [
    "tangent_escape_rmp_tangent_x",
    "tangent_escape_rmp_tangent_y",
    "tangent_escape_rmp_tangent_z",
]
EE_COLS = ["rmp_ee_x", "rmp_ee_y", "rmp_ee_z"]
GOAL_COLS = ["controller_goal_x", "controller_goal_y", "controller_goal_z"]
QDD_COLS = [f"rmp_joint_accel_{index}_rad_s2" for index in range(1, 7)]


def to_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    stripped = text.strip()
    if stripped.lower() in NONFINITE:
        return None
    try:
        value = float(stripped)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def row_vector(row: Dict[str, str], columns: Sequence[str]) -> Optional[List[float]]:
    values = [to_float(row.get(column)) for column in columns]
    if any(value is None for value in values):
        return None
    return values  # type: ignore[return-value]


def norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


@dataclass
class RunConfig:
    name: str
    acceleration_model: str
    gamma: Optional[float]
    extra_args: List[str] = field(default_factory=list)


@dataclass
class RunSummary:
    name: str
    csv_path: Optional[Path]
    samples: int = 0
    schema_ids: List[int] = field(default_factory=list)
    reported_gamma: Optional[float] = None
    active_fraction: Optional[float] = None
    active_samples: int = 0
    duration_s: Optional[float] = None
    min_clearance_m: Optional[float] = None
    min_clearance_active_m: Optional[float] = None
    max_abs_vt: Optional[float] = None
    mean_abs_vt_active: Optional[float] = None
    max_abs_ay: Optional[float] = None
    max_push_accel: Optional[float] = None
    injected_energy_j: Optional[float] = None
    peak_injection_w: Optional[float] = None
    direction_reversals: int = 0
    max_direction_step_deg: Optional[float] = None
    handoff_generations: Optional[int] = None
    max_tangent_normal: Optional[float] = None
    metric_gate_violations: int = 0
    max_qdd_norm: Optional[float] = None
    max_qdd_jump: Optional[float] = None
    goal_error_start_m: Optional[float] = None
    goal_error_end_m: Optional[float] = None
    goal_error_min_m: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    @property
    def goal_error_delta_m(self) -> Optional[float]:
        if self.goal_error_start_m is None or self.goal_error_end_m is None:
            return None
        return self.goal_error_end_m - self.goal_error_start_m


def build_launch_command(
    config: RunConfig,
    args: argparse.Namespace,
    log_directory: Path,
    prefix: str,
) -> List[str]:
    command = [
        "ros2",
        "launch",
        LAUNCH_PACKAGE,
        LAUNCH_FILE,
        "use_rviz:=false",
        "use_rmpflow_trace_logger:=true",
        f"rmpflow_trace_log_directory:={log_directory}",
        f"rmpflow_trace_output_prefix:={prefix}",
        f"rmpflow_trace_log_rate:={args.trace_rate}",
        f"tangent_escape_rmp_acceleration_model:={config.acceleration_model}",
        "enable_tangent_escape_rmp:=true",
        f"goal_x:={args.goal_x}",
        f"goal_y:={args.goal_y}",
        f"goal_z:={args.goal_z}",
        # The launch file defaults this to false because the fake publisher is
        # normally started from a second terminal.  An unattended sweep has no
        # second terminal, so it must start it here or every run records a
        # scenario with no obstacle at all.
        f"start_fake_proximity:={'true' if args.start_fake_proximity else 'false'}",
        f"fake_scenario:={args.fake_scenario}",
        f"fake_sensor_name:={args.fake_sensor}",
        f"fake_range_m:={args.fake_range}",
        f"fake_start_range_m:={args.fake_start_range}",
        f"fake_end_range_m:={args.fake_end_range}",
        f"fake_start_s:={args.fake_start_s}",
        f"fake_duration_s:={args.fake_duration_s}",
        f"fake_period_s:={args.fake_period_s}",
        f"fake_hold_s:={args.fake_hold_s}",
    ]
    if args.params_file:
        command.append(f"params_file:={args.params_file}")
    if config.gamma is not None:
        command.append(f"tangent_escape_rmp_metric_isotropy:={config.gamma}")
    if args.metric_scalar is not None:
        command.append(f"tangent_escape_rmp_metric_scalar:={args.metric_scalar}")
    if args.max_accel is not None:
        command.append(f"tangent_escape_rmp_max_accel:={args.max_accel}")
    if args.max_speed is not None:
        command.append(f"tangent_escape_rmp_max_speed:={args.max_speed}")
    if args.tangent_damping_gain is not None:
        command.append(
            f"tangent_escape_rmp_risk_tangent_damping_gain:={args.tangent_damping_gain}"
        )
    command.extend(config.extra_args)
    command.extend(args.launch_arg or [])
    return command


def run_one(
    config: RunConfig,
    args: argparse.Namespace,
    output_directory: Path,
) -> Optional[Path]:
    log_directory = output_directory / config.name
    log_directory.mkdir(parents=True, exist_ok=True)
    prefix = f"trace_{config.name}"
    command = build_launch_command(config, args, log_directory, prefix)

    print(f"\n=== run: {config.name} ===")
    print("  " + " ".join(command))
    stdout_path = log_directory / "launch_stdout.log"

    with stdout_path.open("w", encoding="utf-8") as stdout_file:
        process = subprocess.Popen(
            command,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + args.duration
        exited_early = False
        while time.monotonic() < deadline:
            if process.poll() is not None:
                exited_early = True
                break
            time.sleep(0.25)

        if exited_early:
            print(
                f"  launch exited early with code {process.returncode};"
                f" see {stdout_path}"
            )
        else:
            # SIGINT the whole group so the trace logger flushes and closes.
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=args.shutdown_timeout)
            except subprocess.TimeoutExpired:
                print("  launch ignored SIGINT; sending SIGKILL")
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)

    candidates = sorted(
        log_directory.glob(f"{prefix}*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        print(f"  no trace CSV produced in {log_directory}")
        return None
    csv_path = candidates[-1]
    print(f"  trace: {csv_path} ({csv_path.stat().st_size} bytes)")
    return csv_path


def summarize(
    name: str,
    csv_path: Optional[Path],
    fallback_goal: Optional[Sequence[float]] = None,
) -> RunSummary:
    """Summarise one trace CSV.

    `controller_goal_*` is only populated when a goal is published on
    /goal_pose.  A launch that takes its goal as a node parameter leaves those
    columns empty, so the caller can supply the commanded goal instead.
    """
    summary = RunSummary(name=name, csv_path=csv_path)
    if csv_path is None or not csv_path.exists():
        summary.notes.append("missing trace CSV")
        return summary

    times: List[float] = []
    active_times: List[float] = []
    clearances: List[float] = []
    active_clearances: List[float] = []
    abs_vt: List[float] = []
    active_abs_vt: List[float] = []
    abs_ay: List[float] = []
    push_accels: List[float] = []
    injection: List[float] = []
    tangent_normals: List[float] = []
    qdd_norms: List[float] = []
    goal_errors: List[float] = []
    generations: set = set()
    schema_ids: set = set()
    gammas: set = set()

    previous_tangent: Optional[List[float]] = None
    previous_qdd: Optional[List[float]] = None
    max_direction_step = None
    reversals = 0
    max_qdd_jump = None
    injected_energy = 0.0
    previous_active_time: Optional[float] = None
    used_fallback_goal = False

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            summary.samples += 1
            timestamp = to_float(row.get(COL_TIME))
            if timestamp is not None:
                times.append(timestamp)

            schema = to_float(row.get(COL_SCHEMA))
            if schema is not None:
                schema_ids.add(int(round(schema)))
            gamma = to_float(row.get(COL_GAMMA))
            if gamma is not None:
                gammas.add(round(gamma, 6))

            clearance = to_float(row.get(COL_CLEARANCE))
            if clearance is not None:
                clearances.append(clearance)

            qdd = row_vector(row, QDD_COLS)
            if qdd is not None:
                qdd_norms.append(norm(qdd))
                if previous_qdd is not None:
                    jump = norm([qdd[i] - previous_qdd[i] for i in range(6)])
                    max_qdd_jump = jump if max_qdd_jump is None else max(max_qdd_jump, jump)
                previous_qdd = qdd
            else:
                qdd_norm = to_float(row.get(COL_QDD_NORM))
                if qdd_norm is not None:
                    qdd_norms.append(qdd_norm)

            ee = row_vector(row, EE_COLS)
            goal = row_vector(row, GOAL_COLS)
            if goal is None and fallback_goal is not None:
                goal = list(fallback_goal)
                used_fallback_goal = True
            if ee is not None and goal is not None:
                goal_errors.append(norm([ee[i] - goal[i] for i in range(3)]))

            generation = to_float(row.get(COL_GENERATION))
            if generation is not None:
                generations.add(int(round(generation)))

            active = to_float(row.get(COL_ACTIVE))
            if active is None or active <= 0.5:
                previous_active_time = None
                continue

            summary.active_samples += 1
            if timestamp is not None:
                active_times.append(timestamp)
            if clearance is not None:
                active_clearances.append(clearance)

            velocity = to_float(row.get(COL_VT))
            if velocity is not None:
                abs_vt.append(abs(velocity))
                active_abs_vt.append(abs(velocity))
            accel = to_float(row.get(COL_AY))
            if accel is not None:
                abs_ay.append(abs(accel))
            push = to_float(row.get(COL_PUSH_ACCEL))
            if push is not None:
                push_accels.append(push)
            tangent_normal = to_float(row.get(COL_TANGENT_NORMAL))
            if tangent_normal is not None:
                tangent_normals.append(abs(tangent_normal))

            push_alpha = to_float(row.get(COL_PUSH_ALPHA))
            metric_alpha = to_float(row.get(COL_METRIC_ALPHA))
            if push_alpha is not None and metric_alpha is not None:
                if metric_alpha > push_alpha + 1e-9:
                    summary.metric_gate_violations += 1

            power = to_float(row.get(COL_PINJ))
            if power is not None:
                injection.append(power)
                if timestamp is not None and previous_active_time is not None:
                    delta = timestamp - previous_active_time
                    if 0.0 < delta < 1.0:
                        injected_energy += power * delta
            if timestamp is not None:
                previous_active_time = timestamp

            tangent = row_vector(row, TANGENT_COLS)
            if tangent is not None and norm(tangent) > 1e-9:
                unit = [value / norm(tangent) for value in tangent]
                if previous_tangent is not None:
                    dot = sum(unit[i] * previous_tangent[i] for i in range(3))
                    dot = max(-1.0, min(1.0, dot))
                    step = math.degrees(math.acos(dot))
                    max_direction_step = (
                        step if max_direction_step is None else max(max_direction_step, step)
                    )
                    if dot < 0.0:
                        reversals += 1
                previous_tangent = unit

    summary.schema_ids = sorted(schema_ids)
    if len(gammas) == 1:
        summary.reported_gamma = next(iter(gammas))
    elif gammas:
        summary.notes.append(f"multiple gammas observed: {sorted(gammas)}")

    if times:
        summary.duration_s = max(times) - min(times)
    if summary.samples:
        summary.active_fraction = summary.active_samples / summary.samples
    if clearances:
        summary.min_clearance_m = min(clearances)
    if active_clearances:
        summary.min_clearance_active_m = min(active_clearances)
    if abs_vt:
        summary.max_abs_vt = max(abs_vt)
    if active_abs_vt:
        summary.mean_abs_vt_active = statistics.fmean(active_abs_vt)
    if abs_ay:
        summary.max_abs_ay = max(abs_ay)
    if push_accels:
        summary.max_push_accel = max(push_accels)
    if injection:
        summary.peak_injection_w = max(injection)
        summary.injected_energy_j = injected_energy
    if tangent_normals:
        summary.max_tangent_normal = max(tangent_normals)
    if qdd_norms:
        summary.max_qdd_norm = max(qdd_norms)
    summary.max_qdd_jump = max_qdd_jump
    summary.direction_reversals = reversals
    summary.max_direction_step_deg = max_direction_step
    if generations:
        summary.handoff_generations = max(generations) - min(generations)
    if used_fallback_goal:
        summary.notes.append("goal error uses the commanded goal, not /goal_pose")
    if goal_errors:
        summary.goal_error_start_m = goal_errors[0]
        summary.goal_error_end_m = goal_errors[-1]
        summary.goal_error_min_m = min(goal_errors)
    if summary.active_samples == 0:
        summary.notes.append("Escape never activated")
    return summary


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def print_table(summaries: Sequence[RunSummary]) -> None:
    rows = [
        ("run", [s.name for s in summaries]),
        ("schema", ["/".join(str(v) for v in s.schema_ids) or "-" for s in summaries]),
        ("gamma", [fmt(s.reported_gamma) for s in summaries]),
        ("samples", [str(s.samples) for s in summaries]),
        ("duration_s", [fmt(s.duration_s) for s in summaries]),
        ("active_frac", [fmt(s.active_fraction) for s in summaries]),
        ("min_clear_m", [fmt(s.min_clearance_m) for s in summaries]),
        ("min_clear_act_m", [fmt(s.min_clearance_active_m) for s in summaries]),
        ("max|v_t|", [fmt(s.max_abs_vt) for s in summaries]),
        ("mean|v_t|act", [fmt(s.mean_abs_vt_active) for s in summaries]),
        ("max|a_y|", [fmt(s.max_abs_ay) for s in summaries]),
        ("max a_push", [fmt(s.max_push_accel) for s in summaries]),
        ("E_inj", [fmt(s.injected_energy_j) for s in summaries]),
        ("peak_P_inj", [fmt(s.peak_injection_w) for s in summaries]),
        ("dir_reversals", [str(s.direction_reversals) for s in summaries]),
        ("max_dir_step_deg", [fmt(s.max_direction_step_deg) for s in summaries]),
        ("handoffs", [fmt(s.handoff_generations) for s in summaries]),
        ("max|t*.n|", [fmt(s.max_tangent_normal) for s in summaries]),
        ("gate_violations", [str(s.metric_gate_violations) for s in summaries]),
        ("max|qdd|", [fmt(s.max_qdd_norm) for s in summaries]),
        ("max qdd jump", [fmt(s.max_qdd_jump) for s in summaries]),
        ("goal_err_start", [fmt(s.goal_error_start_m) for s in summaries]),
        ("goal_err_end", [fmt(s.goal_error_end_m) for s in summaries]),
        ("goal_err_delta", [fmt(s.goal_error_delta_m) for s in summaries]),
    ]
    label_width = max(len(label) for label, _ in rows)
    column_width = max(
        12,
        max((len(cell) for _, cells in rows for cell in cells), default=12) + 2,
    )
    print("\n=== tangent-plane gamma sweep ===")
    for label, cells in rows:
        line = label.ljust(label_width) + " | "
        line += " ".join(cell.rjust(column_width) for cell in cells)
        print(line)
    for summary in summaries:
        for note in summary.notes:
            print(f"note[{summary.name}]: {note}")
    print(
        "\nMeasurements only. gate_violations must be 0 (alpha_M <= alpha_a is "
        "enforced in the solver). max|t*.n| is the latched tangent's drift out "
        "of the tangent plane, which the leaf re-projects away."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="metric isotropy values to sweep (default: 0.0 0.5 1.0)",
    )
    parser.add_argument(
        "--with-risk-damped-baseline",
        action="store_true",
        help="add a risk_damped run so the drive-law change is isolated too",
    )
    parser.add_argument("--duration", type=float, default=40.0, help="seconds per run")
    parser.add_argument("--shutdown-timeout", type=float, default=25.0)
    parser.add_argument("--trace-rate", type=float, default=100.0)
    parser.add_argument("--goal-x", type=float, default=0.6)
    parser.add_argument("--goal-y", type=float, default=-0.4)
    parser.add_argument("--goal-z", type=float, default=0.6)
    parser.add_argument("--params-file", default=None)
    parser.add_argument(
        "--no-start-fake-proximity",
        dest="start_fake_proximity",
        action="store_false",
        help="do not start the fake publisher (run it yourself in another terminal)",
    )
    parser.set_defaults(start_fake_proximity=True)
    parser.add_argument("--fake-scenario", default="approach_retreat")
    parser.add_argument("--fake-sensor", default="tof6_1_R")
    parser.add_argument("--fake-range", type=float, default=0.05)
    parser.add_argument("--fake-start-range", type=float, default=0.50)
    parser.add_argument("--fake-end-range", type=float, default=0.50)
    parser.add_argument("--fake-start-s", type=float, default=0.0)
    parser.add_argument("--fake-duration-s", type=float, default=10.0)
    parser.add_argument("--fake-period-s", type=float, default=10.0)
    parser.add_argument("--fake-hold-s", type=float, default=3.0)
    parser.add_argument("--metric-scalar", type=float, default=None)
    parser.add_argument("--max-accel", type=float, default=None)
    parser.add_argument("--max-speed", type=float, default=None)
    parser.add_argument("--tangent-damping-gain", type=float, default=None)
    parser.add_argument(
        "--launch-arg",
        action="append",
        default=[],
        help="extra raw name:=value launch argument, repeatable",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="default: ~/ros2_ws/log/tangent_plane_gamma_sweep/<timestamp>",
    )
    parser.add_argument(
        "--summarize-only",
        nargs="+",
        default=None,
        metavar="NAME=CSV",
        help="skip launching and summarize existing CSVs",
    )
    args = parser.parse_args()

    if args.summarize_only:
        summaries = []
        for entry in args.summarize_only:
            if "=" not in entry:
                print(f"--summarize-only expects NAME=CSV, got {entry!r}")
                return 2
            name, _, path = entry.partition("=")
            summaries.append(
                summarize(
                    name,
                    Path(path).expanduser(),
                    fallback_goal=(args.goal_x, args.goal_y, args.goal_z),
                )
            )
        print_table(summaries)
        return 0

    if args.output_dir:
        output_directory = Path(args.output_dir).expanduser()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = (
            Path.home() / "ros2_ws" / "log" / "tangent_plane_gamma_sweep" / stamp
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"output directory: {output_directory}")

    configs: List[RunConfig] = []
    if args.with_risk_damped_baseline:
        configs.append(
            RunConfig(name="risk_damped", acceleration_model="risk_damped", gamma=None)
        )
    for gamma in args.gammas:
        label = re.sub(r"[^0-9]", "_", f"{gamma:.2f}")
        configs.append(
            RunConfig(
                name=f"tangent_plane_g{label}",
                acceleration_model="tangent_plane",
                gamma=gamma,
            )
        )

    summaries: List[RunSummary] = []
    for config in configs:
        csv_path = run_one(config, args, output_directory)
        summaries.append(
            summarize(
                config.name,
                csv_path,
                fallback_goal=(args.goal_x, args.goal_y, args.goal_z),
            )
        )

    print_table(summaries)

    summary_path = output_directory / "sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in vars(summary).items()
                }
                for summary in summaries
            ],
            handle,
            indent=2,
        )
    print(f"\nsummary: {summary_path}")

    if any(summary.metric_gate_violations for summary in summaries):
        print("FAIL: alpha_M exceeded alpha_a in at least one run")
        return 1
    if all(summary.active_samples == 0 for summary in summaries):
        print("FAIL: Escape never activated in any run")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

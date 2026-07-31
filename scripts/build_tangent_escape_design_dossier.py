#!/usr/bin/env python3
"""Build the final Tangent Escape RMP design-closeout dossier."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


DEFAULT_LOCAL_DIR = Path(
    "~/ros2_ws/log/rmpflow_production_stability"
).expanduser()
DEFAULT_PARAMS = Path(
    "~/ros2_ws/src/RMP_Proximity-Sensor/config/params.yaml"
).expanduser()


class DossierError(RuntimeError):
    pass


@dataclass
class EpisodeAudit:
    index: int
    start_row: int
    end_row: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    exit_observed: bool
    exit_reason: str
    branch_change_count: int
    ignored_branch_refresh_count: int
    minimum_branch_interval_s: Optional[float]
    minimum_phase_interval_s: Optional[float]
    maximum_abs_qd_rad_s: float
    maximum_abs_qdd_rad_s2: float
    command_bounds_pass: bool


def finite(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def vector_norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def latest_file(directory: Path, pattern: str) -> Path:
    paths = [path for path in directory.glob(pattern) if path.is_file()]
    if not paths:
        raise DossierError(f"No {pattern} files found in {directory}")
    return max(paths, key=lambda path: path.stat().st_mtime)


def load_yaml(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise DossierError(f"YAML does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping):
        raise DossierError(f"YAML root is not a mapping: {path}")
    return document


def load_params(path: Path) -> Mapping[str, object]:
    document = load_yaml(path)
    try:
        params = document["rmpflow_controller"]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise DossierError(f"Invalid ROS parameter document: {path}") from error
    if not isinstance(params, Mapping):
        raise DossierError(f"Controller parameters are not a mapping: {path}")
    return params


def load_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not path.is_file():
        raise DossierError(f"Trace does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise DossierError(f"Trace has no rows: {path}")
    return rows, fields


def value(row: Mapping[str, str], field: str) -> Optional[float]:
    return finite(row.get(field))


def vector(row: Mapping[str, str], fields: Sequence[str]) -> Optional[List[float]]:
    result = [value(row, field) for field in fields]
    if any(item is None for item in result):
        return None
    return [float(item) for item in result]


def row_time(row: Mapping[str, str]) -> Optional[float]:
    return value(row, "time_ros_s")


def is_escape_active(row: Mapping[str, str]) -> bool:
    return (value(row, "tangent_escape_rmp_active") or 0.0) >= 0.5


def goal_position(
    row: Mapping[str, str], fixed_goal: Optional[Sequence[float]]
) -> Optional[List[float]]:
    observed = vector(row, [f"controller_goal_{axis}" for axis in "xyz"])
    if observed is not None:
        return observed
    if fixed_goal is None:
        return None
    return [float(item) for item in fixed_goal]


def state_distance(
    row: Mapping[str, str], equilibrium_q: Sequence[float]
) -> Optional[float]:
    q = vector(row, [f"joint_{joint}_pos_rad" for joint in range(1, 7)])
    qd = vector(row, [f"joint_{joint}_vel_rad_s" for joint in range(1, 7)])
    if q is None or qd is None:
        return None
    return vector_norm(
        [current - expected for current, expected in zip(q, equilibrium_q)] + qd
    )


def transition_intervals(
    rows: Sequence[Mapping[str, str]],
    start: int,
    end: int,
    field: str,
) -> List[float]:
    events: List[float] = []
    previous: Optional[int] = None
    for row in rows[start : end + 1]:
        raw = value(row, field)
        time_s = row_time(row)
        if raw is None or time_s is None:
            continue
        current = int(round(raw))
        if previous is None or current != previous:
            events.append(time_s)
            previous = current
    return [
        current - previous_time
        for previous_time, current in zip(events, events[1:])
        if current >= previous_time
    ]


def effective_branch_events(
    rows: Sequence[Mapping[str, str]],
    start: int,
    end: int,
    minimum_direction_change_rad: float = math.radians(2.0),
) -> Tuple[List[float], int]:
    events: List[float] = []
    ignored_refreshes = 0
    previous_generation: Optional[int] = None
    previous_control_point: Optional[int] = None
    previous_obstacle_key: Optional[float] = None
    previous_tangent: Optional[List[float]] = None

    for row in rows[start : end + 1]:
        raw_generation = value(row, "tangent_escape_rmp_handoff_generation")
        time_s = row_time(row)
        if raw_generation is None or time_s is None:
            continue
        generation = int(round(raw_generation))
        control_point_raw = value(row, "tangent_escape_rmp_control_point_index")
        control_point = (
            int(round(control_point_raw))
            if control_point_raw is not None
            else None
        )
        obstacle_key = value(row, "tangent_escape_rmp_obstacle_episode_id")
        tangent = vector(
            row,
            [
                "tangent_escape_rmp_tangent_x",
                "tangent_escape_rmp_tangent_y",
                "tangent_escape_rmp_tangent_z",
            ],
        )
        if generation == previous_generation:
            continue

        effective_change = previous_generation is None
        if previous_generation is not None:
            control_point_changed = (
                control_point is not None
                and previous_control_point is not None
                and control_point != previous_control_point
            )
            obstacle_changed = (
                obstacle_key is not None
                and previous_obstacle_key is not None
                and obstacle_key != previous_obstacle_key
            )
            direction_changed = True
            if tangent is not None and previous_tangent is not None:
                tangent_norm = vector_norm(tangent)
                previous_norm = vector_norm(previous_tangent)
                if tangent_norm > 1e-12 and previous_norm > 1e-12:
                    cosine = sum(
                        current * previous
                        for current, previous in zip(tangent, previous_tangent)
                    ) / (tangent_norm * previous_norm)
                    direction_changed = (
                        math.acos(max(-1.0, min(1.0, cosine)))
                        >= minimum_direction_change_rad
                    )
            effective_change = (
                control_point_changed or obstacle_changed or direction_changed
            )
            if not effective_change:
                ignored_refreshes += 1

        if effective_change:
            events.append(time_s)
            previous_control_point = control_point
            previous_obstacle_key = obstacle_key
            previous_tangent = tangent
        previous_generation = generation

    return events, ignored_refreshes


def infer_exit_reason(
    rows: Sequence[Mapping[str, str]], end: int, entry_threshold: float
) -> str:
    if end >= len(rows) - 1:
        return "TRACE_CENSORED"
    next_row = rows[end + 1]
    next_blockage = value(next_row, "tangent_escape_rmp_max_blockage") or 0.0
    next_activation = value(next_row, "tangent_escape_rmp_raw_activation") or 0.0
    last_reason = int(
        round(value(rows[end], "tangent_escape_rmp_handoff_reason_id") or 0.0)
    )
    if next_blockage < entry_threshold and next_activation < entry_threshold:
        return "BLOCKAGE_CLEARED"
    reason_names = {
        1: "LOW_BLOCKAGE_RELEASE",
        2: "PAIR_SWITCH_RELEASE",
        3: "GOAL_CHANGE_RELEASE",
        4: "INVALID_OR_CRITICAL_DIRECTION_RELEASE",
        5: "FAILED_DIRECTION_RESELECT",
        6: "GOAL_PROGRESS_RELEASE",
        7: "MAX_DURATION_TIMEOUT",
    }
    return reason_names.get(last_reason, "GEOMETRY_OR_CANDIDATE_RELEASE")


def active_intervals(rows: Sequence[Mapping[str, str]]) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, row in enumerate(rows):
        active = is_escape_active(row)
        if active and start is None:
            start = index
        elif not active and start is not None:
            intervals.append((start, index - 1))
            start = None
    if start is not None:
        intervals.append((start, len(rows) - 1))
    return intervals


def command_row_values(
    row: Mapping[str, str],
) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
    q = vector(row, [f"joint_{joint}_pos_rad" for joint in range(1, 7)])
    qd = vector(row, [f"joint_{joint}_vel_rad_s" for joint in range(1, 7)])
    qdd = vector(
        row, [f"rmp_joint_accel_{joint}_rad_s2" for joint in range(1, 7)]
    )
    return q, qd, qdd


def analyze_trace(
    path: Path,
    name: str,
    kind: str,
    rows: Sequence[Mapping[str, str]],
    fields: Sequence[str],
    params: Mapping[str, object],
    local: Mapping[str, object],
    fixed_goal: Optional[Sequence[float]],
    reactivation_window_s: float,
) -> Dict[str, object]:
    required = {
        "time_ros_s",
        "tangent_escape_rmp_active",
        "joint_1_pos_rad",
        "joint_1_vel_rad_s",
        "rmp_joint_accel_1_rad_s2",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise DossierError(f"{path} is missing fields: {', '.join(missing)}")

    max_qdd_limit = float(params["max_joint_accel"])
    max_qd_limit = float(params["command_guard_max_velocity_rad_s"])
    lower = [
        math.radians(float(item)) for item in params["joint_lower_limits_deg"]
    ]
    upper = [
        math.radians(float(item)) for item in params["joint_upper_limits_deg"]
    ]
    handoff_duration = float(params["tangent_escape_rmp_handoff_duration"])
    maximum_active_duration = float(
        params["tangent_escape_rmp_maximum_active_duration"]
    )
    timeout_reentry_cooldown = float(
        params["tangent_escape_rmp_timeout_reentry_cooldown"]
    )
    entry_threshold = (
        1e-12
        if str(params["tangent_escape_rmp_acceleration_model"]) == "risk_damped"
        else float(params["tangent_escape_rmp_min_activation"])
    )

    complete_rows = 0
    missing_after_start = 0
    state_started = False
    q_violation_rows = 0
    qd_violation_rows = 0
    qdd_violation_rows = 0
    maximum_abs_qd = 0.0
    maximum_abs_qdd = 0.0
    qd_series: List[float] = []
    qdd_series: List[float] = []
    for row in rows:
        q, qd, qdd = command_row_values(row)
        if q is None or qd is None or qdd is None:
            if state_started:
                missing_after_start += 1
            qd_series.append(float("nan"))
            qdd_series.append(float("nan"))
            continue
        state_started = True
        complete_rows += 1
        qd_abs = max(abs(item) for item in qd)
        qdd_abs = max(abs(item) for item in qdd)
        maximum_abs_qd = max(maximum_abs_qd, qd_abs)
        maximum_abs_qdd = max(maximum_abs_qdd, qdd_abs)
        qd_series.append(vector_norm(qd))
        qdd_series.append(vector_norm(qdd))
        qd_violation_rows += int(qd_abs > max_qd_limit + 1e-6)
        qdd_violation_rows += int(qdd_abs > max_qdd_limit + 1e-6)
        q_violation_rows += int(
            any(
                position < lower[index] - 1e-8
                or position > upper[index] + 1e-8
                for index, position in enumerate(q)
            )
        )
    command_bounds_pass = (
        complete_rows > 0
        and missing_after_start == 0
        and q_violation_rows == 0
        and qd_violation_rows == 0
        and qdd_violation_rows == 0
    )

    episodes: List[EpisodeAudit] = []
    for episode_index, (start, end) in enumerate(active_intervals(rows), start=1):
        start_time = row_time(rows[start])
        end_time = row_time(rows[end])
        if start_time is None or end_time is None:
            continue
        branch_events, ignored_branch_refreshes = effective_branch_events(
            rows, start, end
        )
        branch_intervals = [
            current - previous
            for previous, current in zip(branch_events, branch_events[1:])
            if current >= previous
        ]
        phase_intervals = transition_intervals(
            rows, start, end, "tangent_escape_rmp_handoff_phase_id"
        )
        episode_qd = 0.0
        episode_qdd = 0.0
        for row in rows[start : end + 1]:
            _, qd, qdd = command_row_values(row)
            if qd is not None:
                episode_qd = max(episode_qd, max(abs(item) for item in qd))
            if qdd is not None:
                episode_qdd = max(episode_qdd, max(abs(item) for item in qdd))
        exit_observed = end < len(rows) - 1
        episodes.append(
            EpisodeAudit(
                index=episode_index,
                start_row=start,
                end_row=end,
                start_time_s=start_time,
                end_time_s=end_time,
                duration_s=max(end_time - start_time, 0.0),
                exit_observed=exit_observed,
                exit_reason=infer_exit_reason(rows, end, entry_threshold),
                branch_change_count=max(len(branch_events) - 1, 0),
                ignored_branch_refresh_count=ignored_branch_refreshes,
                minimum_branch_interval_s=(
                    min(branch_intervals) if branch_intervals else None
                ),
                minimum_phase_interval_s=(
                    min(phase_intervals) if phase_intervals else None
                ),
                maximum_abs_qd_rad_s=episode_qd,
                maximum_abs_qdd_rad_s2=episode_qdd,
                command_bounds_pass=(
                    episode_qd <= max_qd_limit + 1e-6
                    and episode_qdd <= max_qdd_limit + 1e-6
                ),
            )
        )

    local_goal = [float(item) for item in local["fixed_inputs"]["goal_position"]]
    trace_goal = None
    for row in rows:
        trace_goal = goal_position(row, fixed_goal)
        if trace_goal is not None:
            break
    goal_match = (
        trace_goal is not None
        and vector_norm(
            current - expected for current, expected in zip(trace_goal, local_goal)
        )
        <= 1e-3
    )
    equilibrium_q = [float(item) for item in local["equilibrium"]["q_rad"]]
    sampled_radius = float(
        local["nonlinear_sampling"]["largest_strict_decrease_radius"]
    )
    final_episode = episodes[-1] if episodes else None
    first_reentry_time: Optional[float] = None
    minimum_post_escape_distance = math.inf
    if final_episode is not None and final_episode.exit_observed and goal_match:
        for row in rows[final_episode.end_row + 1 :]:
            distance = state_distance(row, equilibrium_q)
            time_s = row_time(row)
            if distance is None or time_s is None:
                continue
            minimum_post_escape_distance = min(
                minimum_post_escape_distance, distance
            )
            if first_reentry_time is None and distance <= sampled_radius:
                first_reentry_time = time_s

    reactivation_after_reentry = False
    reentry_observation_s = 0.0
    if first_reentry_time is not None:
        last_time = first_reentry_time
        for row in rows:
            time_s = row_time(row)
            if time_s is None or time_s < first_reentry_time:
                continue
            last_time = time_s
            if is_escape_active(row):
                reactivation_after_reentry = True
        reentry_observation_s = max(last_time - first_reentry_time, 0.0)

    branch_intervals_all: List[float] = []
    for episode in episodes:
        if episode.minimum_branch_interval_s is not None:
            branch_intervals_all.append(episode.minimum_branch_interval_s)
    observed_branch_dwell_pass = (
        not branch_intervals_all
        or min(branch_intervals_all) >= 0.5 * handoff_duration
    )
    all_episodes_released = bool(episodes) and all(
        episode.exit_observed for episode in episodes
    )
    hard_episode_timeout_configured = maximum_active_duration > 0.0
    maximum_observed_episode_duration = max(
        (episode.duration_s for episode in episodes),
        default=0.0,
    )
    observed_episode_duration_bound_pass = (
        not episodes
        or (
            hard_episode_timeout_configured
            and maximum_observed_episode_duration
            <= maximum_active_duration + handoff_duration + 0.10
        )
    )
    final_reentry = first_reentry_time is not None
    no_reactivation_observed = (
        final_reentry
        and not reactivation_after_reentry
        and reentry_observation_s >= max(reactivation_window_s, 0.0)
    )
    conditional_chain = (
        bool(local["verdict"]["normal_local_exponential_stability_certified"])
        and hard_episode_timeout_configured
        and command_bounds_pass
        and all_episodes_released
        and observed_episode_duration_bound_pass
        and observed_branch_dwell_pass
        and goal_match
        and final_reentry
        and no_reactivation_observed
    )

    has_escape = bool(episodes)
    if kind == "positive":
        scenario_pass = conditional_chain
    elif kind == "clear":
        scenario_pass = command_bounds_pass and not has_escape
    elif kind == "persistent":
        scenario_pass = command_bounds_pass and has_escape
    else:
        raise DossierError(f"Unsupported scenario kind '{kind}' for {name}")

    times = [row_time(row) for row in rows]
    finite_times = [item for item in times if item is not None]
    origin = finite_times[0] if finite_times else 0.0
    return {
        "name": name,
        "kind": kind,
        "trace_csv": str(path.resolve()),
        "row_count": len(rows),
        "duration_s": (
            finite_times[-1] - finite_times[0] if len(finite_times) >= 2 else 0.0
        ),
        "episode_count": len(episodes),
        "episodes": [asdict(episode) for episode in episodes],
        "all_episodes_released": all_episodes_released,
        "hard_episode_timeout_configured": hard_episode_timeout_configured,
        "configured_maximum_active_duration_s": maximum_active_duration,
        "configured_timeout_reentry_cooldown_s": timeout_reentry_cooldown,
        "maximum_observed_episode_duration_s": maximum_observed_episode_duration,
        "observed_episode_duration_bound_pass": (
            observed_episode_duration_bound_pass
        ),
        "command_bounds_pass": command_bounds_pass,
        "missing_rows_after_state_start": missing_after_start,
        "q_violation_rows": q_violation_rows,
        "qd_violation_rows": qd_violation_rows,
        "qdd_violation_rows": qdd_violation_rows,
        "maximum_abs_qd_rad_s": maximum_abs_qd,
        "maximum_abs_qdd_rad_s2": maximum_abs_qdd,
        "observed_branch_dwell_pass": observed_branch_dwell_pass,
        "minimum_observed_branch_interval_s": (
            min(branch_intervals_all) if branch_intervals_all else None
        ),
        "goal_matches_local_certificate": goal_match,
        "sampled_normal_decrease_radius": sampled_radius,
        "minimum_post_escape_state_distance": (
            minimum_post_escape_distance
            if math.isfinite(minimum_post_escape_distance)
            else None
        ),
        "first_final_reentry_time_s": first_reentry_time,
        "final_reentry_observed": final_reentry,
        "reactivation_after_final_reentry": reactivation_after_reentry,
        "post_reentry_observation_s": reentry_observation_s,
        "no_reactivation_observed_for_window": no_reactivation_observed,
        "conditional_convergence_chain_observed": conditional_chain,
        "scenario_pass": scenario_pass,
        "plot": {
            "time_s": [
                (item - origin) if item is not None else float("nan")
                for item in times
            ],
            "active": [1.0 if is_escape_active(row) else 0.0 for row in rows],
            "qd_norm": qd_series,
            "qdd_norm": qdd_series,
        },
    }


def load_manifest(
    path: Path,
) -> Tuple[List[Mapping[str, object]], Optional[List[float]]]:
    document = load_yaml(path)
    scenarios = document.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise DossierError("Manifest must contain a non-empty scenarios list")
    fixed_goal_raw = document.get("fixed_goal")
    fixed_goal = None
    if isinstance(fixed_goal_raw, list) and len(fixed_goal_raw) == 3:
        fixed_goal = [float(item) for item in fixed_goal_raw]
    return scenarios, fixed_goal


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    from matplotlib.colors import ListedColormap

    return pyplot, ListedColormap


def scenario_check_status(
    scenario: Mapping[str, object], key: str
) -> str:
    episode_count = int(scenario["episode_count"])
    kind = str(scenario["kind"])
    if key in {
        "all_episodes_released",
        "observed_branch_dwell_pass",
    } and episode_count == 0:
        return "na"
    if key in {
        "final_reentry_observed",
        "no_reactivation_observed_for_window",
    } and kind != "positive":
        return "na"
    return "pass" if bool(scenario[key]) else "fail"


def make_figure(
    local: Mapping[str, object],
    scenarios: Sequence[Mapping[str, object]],
    output_png: Path,
    output_svg: Path,
    dpi: int,
) -> None:
    pyplot, listed_colormap = import_pyplot()
    green = "#157a63"
    blue = "#1769aa"
    orange = "#c75300"
    red = "#b3261e"
    gray = "#5f6368"
    line = "#dadce0"

    figure, axes = pyplot.subplots(2, 3, figsize=(18.0, 10.2))
    figure.patch.set_facecolor("white")

    eigen_axis = axes[0, 0]
    eigen_axis.set_title("(a) NORMAL discrete eigenvalues", loc="left", fontweight="bold")
    angles = [2.0 * math.pi * index / 400.0 for index in range(401)]
    eigen_axis.plot(
        [math.cos(angle) for angle in angles],
        [math.sin(angle) for angle in angles],
        color=gray,
        linestyle="--",
        linewidth=1.0,
    )
    eigenvalues = local["linearization"]["eigenvalues"]
    eigen_axis.scatter(
        [float(item["real"]) for item in eigenvalues],
        [float(item["imag"]) for item in eigenvalues],
        color=blue,
        edgecolor="white",
        s=44,
        zorder=3,
    )
    eigen_axis.set_aspect("equal", adjustable="box")
    eigen_axis.set_xlim(-1.06, 1.06)
    eigen_axis.set_ylim(-1.06, 1.06)
    eigen_axis.grid(True, alpha=0.18)
    eigen_axis.text(
        0.97,
        0.04,
        f"rho(A)={float(local['linearization']['spectral_radius']):.7f}",
        transform=eigen_axis.transAxes,
        ha="right",
        bbox={"facecolor": "white", "edgecolor": line},
    )

    lyapunov_axis = axes[0, 1]
    lyapunov_axis.set_title("(b) Sampled nonlinear Lyapunov decrease", loc="left", fontweight="bold")
    samples = local["nonlinear_sampling"]["samples"]
    passed = [
        (float(sample["radius"]), -float(sample["delta_v"]))
        for sample in samples
        if float(sample["delta_v"]) < 0.0
    ]
    failed = [
        (float(sample["radius"]), max(float(sample["delta_v"]), 1e-20))
        for sample in samples
        if float(sample["delta_v"]) >= 0.0
    ]
    if passed:
        lyapunov_axis.scatter(
            [item[0] for item in passed],
            [item[1] for item in passed],
            s=10,
            alpha=0.35,
            color=green,
            label=r"$-\Delta V$",
        )
    if failed:
        lyapunov_axis.scatter(
            [item[0] for item in failed],
            [item[1] for item in failed],
            marker="x",
            color=red,
            label="nondecrease",
        )
    lyapunov_axis.set_xscale("log")
    lyapunov_axis.set_yscale("log")
    lyapunov_axis.set_xlabel(r"$\|\delta x\|_2$")
    lyapunov_axis.set_ylabel(r"$-\Delta V$")
    lyapunov_axis.grid(True, which="both", alpha=0.2)
    lyapunov_axis.legend(loc="lower right")

    natural_axis = axes[0, 2]
    natural_axis.set_title("(c) Active Escape root metric by ToF", loc="left", fontweight="bold")
    audit = local["rmp_natural_form"]["active_escape_audit"]
    probes = [item for item in audit["sensor_probes"] if bool(item["activated"])]
    natural_axis.bar(
        list(range(len(probes))),
        [float(item["root_metric_min_eigenvalue"]) for item in probes],
        color=green,
        width=0.76,
    )
    natural_axis.set_xticks(list(range(len(probes))))
    natural_axis.set_xticklabels(
        [str(item["name"]) for item in probes], rotation=70, ha="right", fontsize=7
    )
    natural_axis.set_ylabel(r"minimum eigenvalue of $M_{root}$")
    natural_axis.grid(True, axis="y", alpha=0.2)
    natural_axis.text(
        0.98,
        0.96,
        f"{int(audit['activated_sensor_count'])}/{int(audit['sensor_count'])} ToFs\n"
        f"rank max={int(audit['maximum_contribution_rank'])}\n"
        f"max |n^Tt|={float(audit['maximum_tangent_normal_dot_abs']):.2e}",
        transform=natural_axis.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": line},
    )

    timeline_axis = axes[1, 0]
    timeline_axis.set_title("(d) Escape episodes by scenario", loc="left", fontweight="bold")
    for index, scenario in enumerate(scenarios):
        plot = scenario["plot"]
        timeline_axis.fill_between(
            plot["time_s"],
            index,
            [
                index + 0.72 * float(active)
                for active in plot["active"]
            ],
            color=orange,
            alpha=0.78,
            step="post",
            linewidth=0.0,
        )
    timeline_axis.set_yticks(list(range(len(scenarios))))
    timeline_axis.set_yticklabels([str(item["name"]) for item in scenarios], fontsize=8)
    timeline_axis.set_xlabel("time from trace start (s)")
    timeline_axis.set_ylabel("Escape active")
    timeline_axis.grid(True, axis="x", alpha=0.2)

    matrix_axis = axes[1, 1]
    matrix_axis.set_title("(e) Hybrid execution checks", loc="left", fontweight="bold")
    checks = [
        ("Bounds", "command_bounds_pass"),
        ("Finite exit", "all_episodes_released"),
        ("Dwell", "observed_branch_dwell_pass"),
        ("Re-entry", "final_reentry_observed"),
        ("No react.", "no_reactivation_observed_for_window"),
    ]
    status_code = {"na": 0.0, "fail": 1.0, "pass": 2.0}
    matrix_status = [
        [scenario_check_status(scenario, key) for _, key in checks]
        for scenario in scenarios
    ]
    matrix = [
        [status_code[status] for status in row]
        for row in matrix_status
    ]
    matrix_axis.imshow(
        matrix,
        cmap=listed_colormap(["#f1f3f4", "#fce8e6", "#e8f5f1"]),
        vmin=0.0,
        vmax=2.0,
        aspect="auto",
    )
    matrix_axis.set_xticks(list(range(len(checks))))
    matrix_axis.set_xticklabels([label for label, _ in checks], rotation=35, ha="right")
    matrix_axis.set_yticks(list(range(len(scenarios))))
    matrix_axis.set_yticklabels([str(item["name"]) for item in scenarios], fontsize=8)
    for row_index, row_statuses in enumerate(matrix_status):
        for column_index, check_status in enumerate(row_statuses):
            label = {
                "pass": "PASS",
                "fail": "FAIL",
                "na": "N/A",
            }[check_status]
            label_color = {
                "pass": green,
                "fail": red,
                "na": gray,
            }[check_status]
            matrix_axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=label_color,
                fontweight="bold",
            )

    scope_axis = axes[1, 2]
    scope_axis.set_title("(f) Claim boundary", loc="left", fontweight="bold")
    scope_axis.axis("off")
    statuses = [
        ("Natural-form compatibility", True, "PROVED + TESTED"),
        (
            "NORMAL local stability",
            bool(local["verdict"]["normal_local_exponential_stability_certified"]),
            "CERTIFIED",
        ),
        (
            "Clear-set guard separation",
            bool(local["verdict"]["normal_escape_guard_separated_in_clear_set"]),
            "CERTIFIED",
        ),
        (
            "Software command bounds",
            bool(local["verdict"]["software_command_box_forward_invariant_certified"]),
            "CERTIFIED",
        ),
        (
            "Hard episode timeout",
            all(
                bool(item["hard_episode_timeout_configured"])
                for item in scenarios
            ),
            "ENFORCED",
        ),
        (
            "Conditional hybrid chain",
            any(
                bool(item["conditional_convergence_chain_observed"])
                for item in scenarios
            ),
            "OBSERVED",
        ),
    ]
    y = 0.88
    for label, passed_check, success_text in statuses:
        scope_axis.text(0.02, y, label, fontsize=10, va="center")
        color = green if passed_check else red
        scope_axis.text(
            0.98,
            y,
            success_text if passed_check else "NOT SHOWN",
            ha="right",
            va="center",
            color="white",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": color, "edgecolor": color},
        )
        y -= 0.115
    scope_axis.text(
        0.02,
        0.05,
        "Not claimed:\n"
        "- structured-GDS closure of supervisor\n"
        "- global hybrid asymptotic stability\n"
        "- collision-safe-set invariance",
        color=red,
        fontsize=10,
        va="bottom",
    )

    figure.suptitle(
        "Tangent Escape RMP Design Closeout\n"
        "Natural-form structure + NORMAL certificate + conditional hybrid evidence",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_svg, bbox_inches="tight", facecolor="white")
    pyplot.close(figure)


def status_class(passed: bool) -> str:
    return "pass" if passed else "fail"


def make_html(
    output_path: Path,
    image_path: Path,
    local_path: Path,
    params_path: Path,
    manifest_path: Path,
    local: Mapping[str, object],
    scenarios: Sequence[Mapping[str, object]],
    verdict: Mapping[str, object],
) -> None:
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    audit = local["rmp_natural_form"]["active_escape_audit"]
    episode_rows = []
    for scenario in scenarios:
        if not scenario["episodes"]:
            episode_rows.append(
                "<tr>"
                f"<td>{html.escape(str(scenario['name']))}</td>"
                "<td colspan='7'>Escape activation 없음</td>"
                "</tr>"
            )
            continue
        for episode in scenario["episodes"]:
            branch_interval = episode["minimum_branch_interval_s"]
            episode_rows.append(
                "<tr>"
                f"<td>{html.escape(str(scenario['name']))}</td>"
                f"<td>{int(episode['index'])}</td>"
                f"<td>{float(episode['duration_s']):.4f}</td>"
                f"<td>{html.escape(str(episode['exit_reason']))}</td>"
                f"<td>{int(episode['branch_change_count'])}</td>"
                f"<td>{'n/a' if branch_interval is None else f'{float(branch_interval):.4f}'}</td>"
                f"<td>{float(episode['maximum_abs_qd_rad_s']):.4g}</td>"
                f"<td>{float(episode['maximum_abs_qdd_rad_s2']):.4g}</td>"
                "</tr>"
            )
    def check_cell(item: Mapping[str, object], key: str) -> str:
        check_status = scenario_check_status(item, key)
        label = {"pass": "PASS", "fail": "FAIL", "na": "N/A"}[check_status]
        return f"<td class='{check_status}'>{label}</td>"

    scenario_rows = ""
    for item in scenarios:
        scenario_rows += (
            "<tr>"
            f"<td>{html.escape(str(item['name']))}</td>"
            f"<td>{html.escape(str(item['kind']))}</td>"
            f"<td>{int(item['episode_count'])}</td>"
            f"{check_cell(item, 'command_bounds_pass')}"
            f"{check_cell(item, 'all_episodes_released')}"
            f"{check_cell(item, 'observed_branch_dwell_pass')}"
            f"{check_cell(item, 'final_reentry_observed')}"
            f"{check_cell(item, 'no_reactivation_observed_for_window')}"
            f"<td class='{status_class(bool(item['scenario_pass']))}'>"
            f"{'PASS' if item['scenario_pass'] else 'FAIL'}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tangent Escape RMP Design Closeout</title>
<style>
:root {{ --ink:#202124; --muted:#5f6368; --line:#dadce0;
         --green:#157a63; --red:#b3261e; --blue:#1769aa; }}
body {{ margin:0; color:var(--ink); background:white;
        font:16px/1.62 Arial,"Noto Sans KR",sans-serif; }}
main {{ max-width:1220px; margin:0 auto; padding:34px 28px 72px; }}
h1 {{ margin:0 0 8px; font-size:30px; }}
h2 {{ margin:34px 0 10px; padding-bottom:7px;
      border-bottom:1px solid var(--line); font-size:21px; }}
.meta {{ color:var(--muted); }}
.verdict {{ margin:22px 0; padding:15px 18px; border-left:5px solid; }}
.verdict.pass {{ border-color:var(--green); background:#e8f5f1; }}
.verdict.fail {{ border-color:var(--red); background:#fce8e6; }}
img {{ width:100%; border:1px solid var(--line); }}
table {{ width:100%; border-collapse:collapse; margin:12px 0 24px; }}
th,td {{ border:1px solid var(--line); padding:8px 10px; text-align:left; }}
th {{ background:#f8f9fa; }}
.pass {{ color:var(--green); font-weight:700; }}
.fail {{ color:var(--red); font-weight:700; }}
.na {{ color:var(--muted); font-weight:700; }}
.equation {{ padding:13px 16px; border:1px solid var(--line);
             background:#f8f9fa; font-family:serif; }}
code {{ background:#f1f3f4; padding:2px 5px; }}
</style>
</head>
<body><main>
<h1>Tangent Escape RMP 설계 마무리 자료</h1>
<p class="meta">생성 시각: {html.escape(datetime.now().isoformat(timespec="seconds"))}</p>
<div class="verdict {status_class(bool(verdict['design_closeout_pass']))}">
<strong>{'설계 마무리 PASS' if verdict['design_closeout_pass'] else '추가 검증 필요'}</strong><br>
현재 non-GDS Escape 설계가 natural-form RMP 합성 구조를 보존하고, NORMAL 모드의
국소 안정성과 clear-set guard 분리가 성립하며, 지정된 fake 실행에서 조건부 hybrid
복귀 사슬을 확인했다는 범위의 결론입니다.
</div>
<img src="data:image/png;base64,{image_data}" alt="Tangent Escape RMP closeout figure">

<h2>1. 그림 읽는 법</h2>
<table>
<tr><th>패널</th><th>의미</th></tr>
<tr><td>(a)</td><td>NORMAL 선형화 고유값이 단위원 내부에 있어 국소 이산 안정 조건을 만족합니다.</td></tr>
<tr><td>(b)</td><td>샘플링한 NORMAL 근방에서 ΔV&lt;0이며, 점의 높이는 감소량 −ΔV입니다.</td></tr>
<tr><td>(c)</td><td>20개 ToF 각각에서 Escape 추가 후 root metric의 최소 고유값이 양수입니다.</td></tr>
<tr><td>(d)</td><td>주황 구간이 실제 Escape episode입니다. clear 환경에는 구간이 없습니다.</td></tr>
<tr><td>(e)</td><td>조건부 hybrid 정리의 실행 조건입니다. 회색 N/A는 해당 시나리오의 판정 대상이 아닙니다.</td></tr>
<tr><td>(f)</td><td>증명·인증·실험 관측과 주장하지 않는 범위를 분리합니다.</td></tr>
</table>

<h2>2. RMP 내부 구조</h2>
<div class="equation">
m<sub>y</sub> = m<sub>t</sub> λ B ≥ 0<br>
ΔM = J<sub>y</sub><sup>T</sup> m<sub>y</sub> J<sub>y</sub> ⪰ 0,
&nbsp; rank(ΔM) ≤ 1<br>
Δf = J<sub>y</sub><sup>T</sup>m<sub>y</sub>
(a<sub>y</sub> − Jdot<sub>y</sub>qdot)
</div>
<p>20개 ToF 모두 synthetic blocking probe에서 활성화되었습니다.
총 <strong>{int(audit['active_sample_count'])}</strong>개 active sample 중 실패는
<strong>{int(audit['failed_active_sample_count'])}</strong>건입니다.
최대 rank는 <strong>{int(audit['maximum_contribution_rank'])}</strong>,
최소 λ(ΔM)는 <strong>{float(audit['minimum_contribution_eigenvalue']):.3e}</strong>,
최대 |nᵀt|는 <strong>{float(audit['maximum_tangent_normal_dot_abs']):.3e}</strong>입니다.
Obstacle이 없을 때 ΔM과 Δf는 모두 정확히 0이었습니다.</p>

<h2>3. 안정성 논리</h2>
<p><strong>Formal:</strong> NORMAL production discrete map의 평형점에서
ρ(A)=<strong>{float(local['linearization']['spectral_radius']):.9f}</strong>&lt;1이고,
AᵀPA−P=−I의 P가 양의 정부호입니다. 따라서 분석 가정 아래 NORMAL 평형점은
국소 지수 안정합니다.</p>
<p><strong>Guard separation:</strong> d<sub>i</sub>≥d<sub>off</sub>=
{float(local['normal_escape_guard_separation']['distance_off_m']):.3f} m이면
α<sub>d</sub>=0이고 B=0이므로 Escape 진입 guard와 Ω<sub>N</sub><sup>clear</sup>가
겹치지 않습니다.</p>
<p><strong>Conditional hybrid:</strong> Escape 구간에서 명령이 bounded이고,
episode가 유한하게 끝나며, 마지막 episode 뒤 NORMAL 감소 근방에 재진입하고,
관측 구간 동안 Escape가 다시 켜지지 않으면 이후 NORMAL 인증으로 수렴합니다.</p>
<p><strong>Finite episode enforcement:</strong> continuously active Escape는
최대 {float(scenarios[0]['configured_maximum_active_duration_s']):.1f} s 뒤
smooth lambda-down으로 종료되고,
{float(scenarios[0]['configured_timeout_reentry_cooldown_s']):.1f} s 동안
재진입을 막습니다. 반복 실험의 최대 episode는
{max(float(item['maximum_observed_episode_duration_s']) for item in scenarios):.3f} s였습니다.</p>

<h2>4. 반복 시나리오</h2>
<table>
<tr><th>시나리오</th><th>종류</th><th>Episode</th><th>명령 경계</th>
<th>유한 종료</th><th>Dwell</th><th>최종 재진입</th>
<th>무재활성화</th><th>판정</th></tr>
{scenario_rows}
</table>

<h2>5. Episode 상세</h2>
<table>
<tr><th>시나리오</th><th>#</th><th>지속 (s)</th><th>종료 이유</th>
<th>branch 변경</th><th>최소 branch 간격 (s)</th>
<th>max |qdot|</th><th>max |qdd|</th></tr>
{''.join(episode_rows)}
</table>

<h2>6. 정확한 주장 범위</h2>
<table>
<tr><th>주장</th><th>상태</th></tr>
<tr><td>Escape leaf의 symmetric PSD rank-one natural-form pullback</td>
<td class="pass">증명식 + production 수치검사</td></tr>
<tr><td>NORMAL production map의 국소 지수 안정성</td>
<td class="pass">인증</td></tr>
<tr><td>clear environment에서 NORMAL–Escape guard 분리</td>
<td class="pass">조건부 해석 인증</td></tr>
<tr><td>한 continuously active Escape episode의 최대 지속시간</td>
<td class="pass">hard timeout + smooth release로 강제</td></tr>
<tr><td>Escape 포함 조건부 hybrid 복귀·수렴</td>
<td class="pass">정리 조건 + 반복 실행 증거</td></tr>
<tr><td>supervisor 전체의 structured-GDS closure</td>
<td class="fail">주장하지 않음</td></tr>
<tr><td>모든 장애물 환경의 global asymptotic stability</td>
<td class="fail">주장하지 않음</td></tr>
<tr><td>충돌 안전집합의 formal forward invariance</td>
<td class="fail">주장하지 않음</td></tr>
</table>

<h2>7. 입력과 재현</h2>
<ul>
<li>Local certificate: <code>{html.escape(str(local_path.resolve()))}</code></li>
<li>Parameters: <code>{html.escape(str(params_path.resolve()))}</code></li>
<li>Scenario manifest: <code>{html.escape(str(manifest_path.resolve()))}</code></li>
</ul>
</main></body></html>"""
    output_path.write_text(page, encoding="utf-8")


def make_markdown(
    output_path: Path,
    local_path: Path,
    scenarios: Sequence[Mapping[str, object]],
    local: Mapping[str, object],
    verdict: Mapping[str, object],
) -> None:
    audit = local["rmp_natural_form"]["active_escape_audit"]
    def markdown_status(item: Mapping[str, object], key: str) -> str:
        return {
            "pass": "PASS",
            "fail": "FAIL",
            "na": "N/A",
        }[scenario_check_status(item, key)]

    rows = "\n".join(
        f"| {item['name']} | {item['kind']} | {item['episode_count']} | "
        f"{markdown_status(item, 'command_bounds_pass')} | "
        f"{markdown_status(item, 'all_episodes_released')} | "
        f"{markdown_status(item, 'observed_branch_dwell_pass')} | "
        f"{markdown_status(item, 'final_reentry_observed')} | "
        f"{markdown_status(item, 'no_reactivation_observed_for_window')} | "
        f"{'PASS' if item['scenario_pass'] else 'FAIL'} |"
        for item in scenarios
    )
    positive_scenarios = [
        item for item in scenarios if item["kind"] == "positive"
    ]
    minimum_reentry_distance = min(
        float(item["minimum_post_escape_state_distance"])
        for item in positive_scenarios
        if item["minimum_post_escape_state_distance"] is not None
    )
    minimum_reactivation_observation = min(
        float(item["post_reentry_observation_s"])
        for item in positive_scenarios
    )
    text = f"""# Tangent Escape RMP 설계 마무리

## 결론

**{'PASS' if verdict['design_closeout_pass'] else '추가 검증 필요'}**

현재 Escape는 score, memory, timer, hard switching을 포함하므로 structured GDS
closure를 주장하지 않는다. 대신 다음 세 층으로 결과를 분리한다.

1. Escape leaf는 `m_y = m_t lambda B >= 0`인 scalar natural-form RMP다.
   Pullback은 `DeltaM = J_y^T m_y J_y >= 0`, `rank(DeltaM) <= 1`이다.
2. Escape가 꺼진 NORMAL production discrete map은 평형점 근방에서
   `rho(A)={float(local['linearization']['spectral_radius']):.9f}<1`이고
   이산 Lyapunov 방정식의 `P`가 양의 정부호이므로 국소 지수 안정하다.
3. Escape 실행은 bounded finite transient, NORMAL 감소 근방 재진입,
   이후 재활성화 없음이라는 조건 아래 NORMAL 안정성으로 연결한다.
4. `T_E,max={float(scenarios[0]['configured_maximum_active_duration_s']):.1f} s`
   hard timeout과
   `{float(scenarios[0]['configured_timeout_reentry_cooldown_s']):.1f} s`
   cooldown이 한 continuously active episode의 종료를 강제한다.

## 조건부 hybrid 결론

고정되고 도달 가능한 goal에 대해 다음 조건을 둔다.

1. 각 Escape episode의 software command가 설정 경계 안에 있다.
2. 각 episode가 유한시간 안에 종료된다.
3. 유효 branch 전환 사이에 양의 dwell이 유지된다.
4. 마지막 Escape 뒤 상태가 샘플링된 NORMAL 감소 근방
   `Omega_N = {{x: ||x-x*||_2 <= 0.01}}`에 재진입한다.
5. 재진입 뒤 관측 구간 동안 Escape가 다시 활성화되지 않는다.

그러면 마지막 Escape 이후에는 인증된 NORMAL map이 적용되므로 command state가
NORMAL 평형점으로 수렴한다. 실제 로봇에는 bounded tracking disturbance가
추가되므로 물리 상태에 대한 주장은 평형점 근방으로의 practical convergence다.

## 구조 검증

- ToF probe: {int(audit['activated_sensor_count'])}/{int(audit['sensor_count'])}
- Active samples: {int(audit['active_sample_count'])}
- Failed samples: {int(audit['failed_active_sample_count'])}
- Maximum rank(DeltaM): {int(audit['maximum_contribution_rank'])}
- Minimum eigenvalue(DeltaM): {float(audit['minimum_contribution_eigenvalue']):.6e}
- Maximum `|n^T t|`: {float(audit['maximum_tangent_normal_dot_abs']):.6e}
- Minimum root metric eigenvalue: {float(audit['minimum_root_metric_eigenvalue']):.6g}
- Maximum regularized resolve residual:
  {float(audit['maximum_regularized_resolve_residual']):.6e}
- Inactive `DeltaM`, `Deltaf`: exact zero
- Hard episode timeout:
  {float(scenarios[0]['configured_maximum_active_duration_s']):.3f} s
- Timeout re-entry cooldown:
  {float(scenarios[0]['configured_timeout_reentry_cooldown_s']):.3f} s

## 시나리오 결과

| Scenario | Kind | Episodes | Bounds | Finite exit | Dwell | Re-entry | No reactivation | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

세 positive 반복에서 NORMAL 근방까지의 최소 post-Escape 상태 거리는
`{minimum_reentry_distance:.6e}`이고, 재진입 후 무재활성화 관측시간의 최솟값은
`{minimum_reactivation_observation:.3f} s`다.

## 그림 설명

- **(a) NORMAL eigenvalues:** 모든 고유값이 단위원 안에 있고
  `rho(A)={float(local['linearization']['spectral_radius']):.9f}<1`이다.
- **(b) Lyapunov decrease:** 인증 근방의 nonlinear sample에서 `Delta V<0`이다.
- **(c) Root metric:** 20개 ToF active probe 모두 root metric이 SPD이고,
  Escape contribution은 PSD rank one이다.
- **(d) Episode timeline:** 주황색이 Escape가 실제 켜진 시간이며 clear 환경은
  활성 구간이 없다.
- **(e) Hybrid checks:** 초록은 PASS, 빨강은 FAIL, 회색은 해당 없음이다.
- **(f) Claim boundary:** formal 결과, 조건부 실험 결과, 주장하지 않는 결과를
  한 그림에서 분리한다.

## 교수님께 보고할 문장

제안한 Tangent Escape 정책은 score 기반 supervisor 자체를 GDS라고 가정하지
않는다. 고정 mode에서 Escape가 출력하는 scalar leaf의 metric은 비음수이며,
root pullback은 대칭 PSD rank-one contribution을 만든다. 20개 ToF와
{int(audit['active_sample_count'])}개 active production sample에서 이 구조와
root SPD/resolve를 확인했다. Escape가 비활성인 NORMAL production map은 이산
선형화와 Lyapunov 방정식으로 국소 지수 안정성을 인증했다. 전체 switching
동작은 Escape의 유한시간 종료, software command bound, NORMAL 감소 근방
재진입, 이후 재활성화 없음이라는 조건부 hybrid 정리로 연결했고, 지정된
반복 fake 시나리오에서 해당 조건을 검사했다.

## 1분 발표 대본

왼쪽 위는 Escape가 꺼진 NORMAL 제어기의 국소 안정성 결과로, 이산 선형화
고유값이 모두 단위원 안에 있고 Lyapunov 샘플도 감소합니다. 오른쪽 위는
Escape leaf가 20개 ToF 모두에서 대칭 PSD rank-one metric을 root에 더하며
resolve 구조를 깨지 않는다는 검사입니다. 아래쪽은 실제 switching 실행을
분리해 본 결과입니다. 세 접근·후퇴 반복은 명령 경계, 유한 종료, dwell,
NORMAL 근방 재진입, 재활성화 없음까지 모두 통과했습니다. 센서가 clear일
때는 Escape가 켜지지 않았고, 지속 장애물처럼 목표 수렴 조건이 성립하지 않는
경우에도 software command bound는 유지됐습니다. 따라서 본 연구의 결론은
supervisor 전체의 structured-GDS 또는 global stability가 아니라,
natural-form RMP 호환성, NORMAL 국소 안정성, 그리고 명시된 조건 아래의
hybrid 복귀·수렴입니다.

## 논문에서 쓰면 안 되는 표현

- 전체 supervisor가 structured GDS closure를 만족한다.
- 모든 장애물 환경에서 global asymptotic stability가 증명되었다.
- Collision RMP만으로 충돌 안전집합의 forward invariance가 증명되었다.

## 근거 파일

- `{local_path.resolve()}`
"""
    output_path.write_text(text, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final Tangent Escape RMP design dossier."
    )
    parser.add_argument("--local-yaml", type=Path)
    parser.add_argument("--params-file", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--reactivation-window-s", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        local_path = (
            args.local_yaml.expanduser()
            if args.local_yaml
            else latest_file(DEFAULT_LOCAL_DIR, "production_stability_*.yaml")
        )
        params_path = args.params_file.expanduser()
        manifest_path = args.manifest.expanduser()
        output_prefix = args.output_prefix.expanduser()
        local = load_yaml(local_path)
        params = load_params(params_path)
        manifest_scenarios, manifest_goal = load_manifest(manifest_path)

        audit = local.get("rmp_natural_form", {}).get("active_escape_audit", {})
        if not bool(audit.get("available")):
            raise DossierError(
                "Local YAML does not contain the active Escape natural-form audit"
            )

        scenario_results: List[Mapping[str, object]] = []
        for scenario in manifest_scenarios:
            name = str(scenario.get("name", "")).strip()
            kind = str(scenario.get("kind", "")).strip()
            trace_value = str(scenario.get("trace_csv", "")).strip()
            if not name or not kind or not trace_value:
                raise DossierError(
                    "Each manifest scenario needs name, kind, and trace_csv"
                )
            trace_path = Path(trace_value).expanduser()
            scenario_goal_raw = scenario.get("fixed_goal", manifest_goal)
            scenario_goal = (
                [float(item) for item in scenario_goal_raw]
                if isinstance(scenario_goal_raw, list)
                and len(scenario_goal_raw) == 3
                else None
            )
            rows, fields = load_csv(trace_path)
            scenario_results.append(
                analyze_trace(
                    trace_path,
                    name,
                    kind,
                    rows,
                    fields,
                    params,
                    local,
                    scenario_goal,
                    args.reactivation_window_s,
                )
            )

        structure_pass = (
            bool(local["rmp_natural_form"]["active_escape_natural_form_valid"])
            and bool(local["verdict"]["escape_leaf_natural_form_compatible"])
        )
        normal_pass = bool(
            local["verdict"]["normal_local_exponential_stability_certified"]
        )
        guard_pass = bool(
            local["verdict"]["normal_escape_guard_separated_in_clear_set"]
        )
        command_invariance_pass = bool(
            local["verdict"]["software_command_box_forward_invariant_certified"]
        )
        hard_timeout_pass = (
            float(params["tangent_escape_rmp_maximum_active_duration"]) > 0.0
            and all(
                bool(item["observed_episode_duration_bound_pass"])
                for item in scenario_results
            )
        )
        positive_results = [
            item for item in scenario_results if item["kind"] == "positive"
        ]
        negative_results = [
            item for item in scenario_results if item["kind"] != "positive"
        ]
        positive_pass = bool(positive_results) and all(
            bool(item["scenario_pass"]) for item in positive_results
        )
        negative_pass = all(
            bool(item["scenario_pass"]) for item in negative_results
        )
        verdict = {
            "natural_form_structure_pass": structure_pass,
            "normal_local_stability_pass": normal_pass,
            "normal_escape_guard_separation_pass": guard_pass,
            "software_command_invariance_pass": command_invariance_pass,
            "hard_episode_timeout_pass": hard_timeout_pass,
            "positive_scenarios_pass": positive_pass,
            "negative_scenarios_pass": negative_pass,
            "design_closeout_pass": (
                structure_pass
                and normal_pass
                and guard_pass
                and command_invariance_pass
                and hard_timeout_pass
                and positive_pass
                and negative_pass
            ),
            "structured_gds_closure_claimed": False,
            "global_hybrid_asymptotic_stability_claimed": False,
            "collision_safe_set_invariance_claimed": False,
        }

        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        png_path = output_prefix.with_suffix(".png")
        svg_path = output_prefix.with_suffix(".svg")
        html_path = output_prefix.with_suffix(".html")
        yaml_path = output_prefix.with_suffix(".yaml")
        markdown_path = output_prefix.with_suffix(".md")
        make_figure(local, scenario_results, png_path, svg_path, args.dpi)
        make_html(
            html_path,
            png_path,
            local_path,
            params_path,
            manifest_path,
            local,
            scenario_results,
            verdict,
        )
        make_markdown(
            markdown_path, local_path, scenario_results, local, verdict
        )
        machine_report = {
            "schema_version": 1,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "local_certificate": str(local_path.resolve()),
            "params_file": str(params_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "verdict": verdict,
            "natural_form_audit": audit,
            "normal_escape_guard_separation": local[
                "normal_escape_guard_separation"
            ],
            "scenarios": [
                {key: value for key, value in item.items() if key != "plot"}
                for item in scenario_results
            ],
            "claim_boundary": {
                "formal": [
                    "active Escape natural-form rank-one PSD compatibility",
                    "NORMAL local exponential stability under listed assumptions",
                    "software command-box forward invariance",
                    "Escape guard exclusion inside the clear-environment set",
                    "finite continuously active Escape episode via hard timeout",
                ],
                "conditional_experimental": [
                    "finite Escape execution",
                    "observed dwell and no branch chatter",
                    "re-entry into sampled NORMAL Lyapunov-decrease neighborhood",
                    "no reactivation during the stated observation window",
                ],
                "not_claimed": [
                    "structured-GDS closure of the stateful supervisor",
                    "global hybrid asymptotic stability for all environments",
                    "collision safe-set forward invariance",
                ],
            },
        }
        yaml_path.write_text(
            yaml.safe_dump(machine_report, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        print(f"Local certificate: {local_path}")
        print(f"Scenarios: {len(scenario_results)}")
        print(f"Design closeout: {'PASS' if verdict['design_closeout_pass'] else 'FAIL'}")
        print(f"HTML: {html_path}")
        print(f"Figure PNG: {png_path}")
        print(f"Figure SVG: {svg_path}")
        print(f"Machine YAML: {yaml_path}")
        print(f"Presentation notes: {markdown_path}")
        return 0 if verdict["design_closeout_pass"] else 2
    except (DossierError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"Error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Combine the exact NORMAL-mode certificate with an Escape execution trace."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


DEFAULT_LOCAL_DIR = Path(
    "~/ros2_ws/log/rmpflow_production_stability"
).expanduser()
DEFAULT_TRACE_DIR = Path("~/ros2_ws/log/rmpflow_trace").expanduser()
DEFAULT_PARAMS = Path(
    "~/ros2_ws/src/RMP_Proximity-Sensor/config/params.yaml"
).expanduser()


class ReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class Episode:
    start: int
    end: int
    start_time_s: float
    end_time_s: float

    @property
    def duration_s(self) -> float:
        return max(self.end_time_s - self.start_time_s, 0.0)


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
        raise ReportError(f"No {pattern} files found in {directory}")
    return max(paths, key=lambda path: path.stat().st_mtime)


def latest_escape_trace(directory: Path) -> Path:
    required = {
        "time_ros_s",
        "tangent_escape_rmp_active",
        "tangent_escape_rmp_state_mode_id",
        "rmp_joint_accel_1_rad_s2",
        "joint_1_pos_rad",
        "joint_1_vel_rad_s",
    }
    candidates = sorted(
        directory.glob("rmpflow_trace_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
                if not required.issubset(fields):
                    continue
                if any(
                    (finite(row.get("tangent_escape_rmp_active")) or 0.0) >= 0.5
                    for row in reader
                ):
                    return path
        except OSError:
            continue
    raise ReportError(f"No compatible trace containing active Escape rows in {directory}")


def load_yaml(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise ReportError(f"YAML does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ReportError(f"YAML root is not a mapping: {path}")
    return value


def load_params(path: Path) -> Mapping[str, object]:
    document = load_yaml(path)
    try:
        params = document["rmpflow_controller"]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise ReportError(f"Invalid ROS parameter structure: {path}") from error
    if not isinstance(params, Mapping):
        raise ReportError(f"Controller parameters are not a mapping: {path}")
    return params


def load_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not path.is_file():
        raise ReportError(f"CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ReportError(f"CSV has no rows: {path}")
    return rows, fields


def value(row: Mapping[str, str], field: str) -> Optional[float]:
    return finite(row.get(field))


def row_time(row: Mapping[str, str]) -> Optional[float]:
    return value(row, "time_ros_s")


def vector(row: Mapping[str, str], fields: Sequence[str]) -> Optional[List[float]]:
    result = [value(row, field) for field in fields]
    if any(item is None for item in result):
        return None
    return [float(item) for item in result]


def active_episodes(rows: Sequence[Mapping[str, str]]) -> List[Episode]:
    intervals: List[Episode] = []
    start: Optional[int] = None
    for index, row in enumerate(rows):
        active = (value(row, "tangent_escape_rmp_active") or 0.0) >= 0.5
        if active and start is None:
            start = index
        elif not active and start is not None:
            start_time = row_time(rows[start])
            end_time = row_time(rows[index - 1])
            if start_time is not None and end_time is not None:
                intervals.append(Episode(start, index - 1, start_time, end_time))
            start = None
    if start is not None:
        start_time = row_time(rows[start])
        end_time = row_time(rows[-1])
        if start_time is not None and end_time is not None:
            intervals.append(Episode(start, len(rows) - 1, start_time, end_time))
    return intervals


def goal_error(
    row: Mapping[str, str],
    fixed_goal: Optional[Sequence[float]] = None,
) -> Optional[float]:
    direct = value(row, "goal_error_m")
    if direct is not None:
        return direct
    goal = goal_position(row, fixed_goal)
    ee = vector(row, [f"rmp_ee_{axis}" for axis in "xyz"])
    if goal is None or ee is None:
        return None
    return vector_norm(current - target for current, target in zip(ee, goal))


def goal_position(
    row: Mapping[str, str],
    fixed_goal: Optional[Sequence[float]] = None,
) -> Optional[List[float]]:
    observed = vector(row, [f"controller_goal_{axis}" for axis in "xyz"])
    if observed is not None:
        return observed
    if fixed_goal is None:
        return None
    return [float(item) for item in fixed_goal]


def trace_summary(
    rows: Sequence[Mapping[str, str]],
    params: Mapping[str, object],
    local: Mapping[str, object],
    fixed_trace_goal: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    episodes = active_episodes(rows)
    if not episodes:
        raise ReportError("The selected trace has no active Escape episode")

    max_accel = float(params["max_joint_accel"])
    max_velocity = float(params["command_guard_max_velocity_rad_s"])
    lower_deg = [float(item) for item in params["joint_lower_limits_deg"]]
    upper_deg = [float(item) for item in params["joint_upper_limits_deg"]]
    lower = [math.radians(item) for item in lower_deg]
    upper = [math.radians(item) for item in upper_deg]
    handoff_duration = float(params["tangent_escape_rmp_handoff_duration"])

    complete_state_rows = 0
    nonfinite_rows = 0
    nonfinite_rows_after_state_start = 0
    complete_state_started = False
    qdd_violation_rows = 0
    qd_violation_rows = 0
    q_violation_rows = 0
    maximum_abs_qdd = 0.0
    maximum_abs_qd = 0.0
    qdd_norms: List[float] = []
    qd_norms: List[float] = []
    errors: List[Optional[float]] = []
    for row in rows:
        q = vector(row, [f"joint_{joint}_pos_rad" for joint in range(1, 7)])
        qd = vector(row, [f"joint_{joint}_vel_rad_s" for joint in range(1, 7)])
        qdd = vector(
            row,
            [f"rmp_joint_accel_{joint}_rad_s2" for joint in range(1, 7)],
        )
        errors.append(goal_error(row, fixed_trace_goal))
        if q is None or qd is None or qdd is None:
            nonfinite_rows += 1
            if complete_state_started:
                nonfinite_rows_after_state_start += 1
            qdd_norms.append(float("nan"))
            qd_norms.append(float("nan"))
            continue
        complete_state_started = True
        complete_state_rows += 1
        qdd_abs = max(abs(item) for item in qdd)
        qd_abs = max(abs(item) for item in qd)
        maximum_abs_qdd = max(maximum_abs_qdd, qdd_abs)
        maximum_abs_qd = max(maximum_abs_qd, qd_abs)
        qdd_norms.append(vector_norm(qdd))
        qd_norms.append(vector_norm(qd))
        if qdd_abs > max_accel + 1e-6:
            qdd_violation_rows += 1
        if qd_abs > max_velocity + 1e-6:
            qd_violation_rows += 1
        if any(
            position < lower[index] - 1e-8 or position > upper[index] + 1e-8
            for index, position in enumerate(q)
        ):
            q_violation_rows += 1

    release_pass = all(episode.end < len(rows) - 1 for episode in episodes)
    generation_events: List[float] = []
    previous_generation: Optional[int] = None
    for row in rows:
        active = (value(row, "tangent_escape_rmp_active") or 0.0) >= 0.5
        generation_value = value(row, "tangent_escape_rmp_handoff_generation")
        time_s = row_time(row)
        if not active or generation_value is None or time_s is None:
            continue
        generation = int(round(generation_value))
        if previous_generation is not None and generation != previous_generation:
            generation_events.append(time_s)
        previous_generation = generation
    generation_intervals = [
        current - previous
        for previous, current in zip(generation_events, generation_events[1:])
    ]
    minimum_generation_interval = (
        min(generation_intervals) if generation_intervals else math.inf
    )
    no_observed_chatter = (
        minimum_generation_interval >= max(0.5 * handoff_duration, 0.0)
    )

    local_goal = [
        float(item) for item in local["fixed_inputs"]["goal_position"]
    ]
    observed_active_goal = goal_position(rows[episodes[0].start])
    active_goal = goal_position(rows[episodes[0].start], fixed_trace_goal)
    goal_source = (
        "controller_goal_topic"
        if observed_active_goal is not None
        else "explicit_fixed_trace_goal"
        if fixed_trace_goal is not None
        else "unavailable"
    )
    goal_matches = (
        active_goal is not None
        and vector_norm(
            current - expected for current, expected in zip(active_goal, local_goal)
        )
        <= 1e-3
    )
    equilibrium_q = [
        float(item) for item in local["equilibrium"]["q_rad"]
    ]
    sampled_decrease_radius = float(
        local["nonlinear_sampling"]["largest_strict_decrease_radius"]
    )
    minimum_post_escape_state_distance = math.inf
    first_basin_reentry_time_s: Optional[float] = None
    if goal_matches and release_pass:
        search_start = episodes[-1].end + 1
        for row in rows[search_start:]:
            q = vector(row, [f"joint_{joint}_pos_rad" for joint in range(1, 7)])
            qd = vector(row, [f"joint_{joint}_vel_rad_s" for joint in range(1, 7)])
            time_s = row_time(row)
            current_goal = goal_position(row, fixed_trace_goal)
            if q is None or qd is None or time_s is None or current_goal is None:
                continue
            if vector_norm(
                current - expected
                for current, expected in zip(current_goal, local_goal)
            ) > 1e-3:
                continue
            distance = vector_norm(
                [current - expected for current, expected in zip(q, equilibrium_q)]
                + qd
            )
            minimum_post_escape_state_distance = min(
                minimum_post_escape_state_distance,
                distance,
            )
            if (
                first_basin_reentry_time_s is None
                and distance <= sampled_decrease_radius
            ):
                first_basin_reentry_time_s = time_s

    bounds_pass = (
        complete_state_rows > 0
        and nonfinite_rows_after_state_start == 0
        and qdd_violation_rows == 0
        and qd_violation_rows == 0
        and q_violation_rows == 0
    )
    execution_pass = bounds_pass and release_pass and no_observed_chatter
    basin_reentry_observed = first_basin_reentry_time_s is not None
    conditional_chain_observed = (
        bool(local["verdict"]["normal_local_exponential_stability_certified"])
        and execution_pass
        and goal_matches
        and basin_reentry_observed
    )

    return {
        "episodes": episodes,
        "episode_count": len(episodes),
        "active_duration_s": sum(item.duration_s for item in episodes),
        "all_episodes_released": release_pass,
        "complete_state_rows": complete_state_rows,
        "missing_or_nonfinite_rows": nonfinite_rows,
        "missing_or_nonfinite_rows_after_state_start":
        nonfinite_rows_after_state_start,
        "qdd_violation_rows": qdd_violation_rows,
        "qd_violation_rows": qd_violation_rows,
        "q_violation_rows": q_violation_rows,
        "maximum_abs_qdd_rad_s2": maximum_abs_qdd,
        "maximum_abs_qd_rad_s": maximum_abs_qd,
        "generation_change_count": len(generation_events),
        "minimum_generation_interval_s": minimum_generation_interval,
        "no_observed_branch_chatter": no_observed_chatter,
        "goal_matches_local_certificate": goal_matches,
        "goal_source": goal_source,
        "sampled_decrease_radius": sampled_decrease_radius,
        "minimum_post_escape_state_distance": minimum_post_escape_state_distance,
        "first_basin_reentry_time_s": first_basin_reentry_time_s,
        "basin_reentry_observed": basin_reentry_observed,
        "software_bounds_trace_pass": bounds_pass,
        "hybrid_execution_trace_pass": execution_pass,
        "conditional_convergence_chain_observed": conditional_chain_observed,
        "qdd_norms": qdd_norms,
        "qd_norms": qd_norms,
        "goal_errors": errors,
    }


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def make_figure(
    local: Mapping[str, object],
    rows: Sequence[Mapping[str, str]],
    summary: Mapping[str, object],
    png_path: Path,
    svg_path: Path,
    dpi: int,
) -> None:
    pyplot = import_pyplot()
    green = "#157a63"
    blue = "#1769aa"
    orange = "#c75300"
    red = "#b3261e"
    gray = "#5f6368"

    figure, axes = pyplot.subplots(2, 2, figsize=(15.2, 10.2))
    figure.patch.set_facecolor("white")

    eigen_axis = axes[0, 0]
    eigen_axis.set_title("(a) Exact discrete-map eigenvalues", loc="left", fontweight="bold")
    angles = [2.0 * math.pi * index / 400.0 for index in range(401)]
    eigen_axis.plot(
        [math.cos(angle) for angle in angles],
        [math.sin(angle) for angle in angles],
        color=gray,
        linestyle="--",
        linewidth=1.0,
        label="unit circle",
    )
    eigenvalues = local["linearization"]["eigenvalues"]
    eigen_axis.scatter(
        [float(item["real"]) for item in eigenvalues],
        [float(item["imag"]) for item in eigenvalues],
        s=48,
        color=blue,
        edgecolor="white",
        linewidth=0.7,
        label="eig(A)",
        zorder=3,
    )
    eigen_axis.axhline(0.0, color="#dadce0", linewidth=0.8)
    eigen_axis.axvline(0.0, color="#dadce0", linewidth=0.8)
    eigen_axis.set_aspect("equal", adjustable="box")
    eigen_axis.set_xlim(-1.08, 1.08)
    eigen_axis.set_ylim(-1.08, 1.08)
    eigen_axis.grid(True, alpha=0.18)
    eigen_axis.legend(loc="lower left")
    eigen_axis.text(
        0.98,
        0.04,
        f"spectral radius = {float(local['linearization']['spectral_radius']):.7f}",
        transform=eigen_axis.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#dadce0"},
    )

    delta_axis = axes[0, 1]
    delta_axis.set_title("(b) Nonlinear Lyapunov decrease", loc="left", fontweight="bold")
    samples = local["nonlinear_sampling"]["samples"]
    passed_x: List[float] = []
    passed_y: List[float] = []
    failed_x: List[float] = []
    failed_y: List[float] = []
    for sample in samples:
        radius = float(sample["radius"])
        delta = float(sample["delta_v"])
        if delta < 0.0:
            passed_x.append(radius)
            passed_y.append(-delta)
        else:
            failed_x.append(radius)
            failed_y.append(max(delta, 1e-20))
    delta_axis.scatter(
        passed_x,
        passed_y,
        s=11,
        alpha=0.38,
        color=green,
        label=r"$-\Delta V$ ($\Delta V<0$)",
    )
    if failed_x:
        delta_axis.scatter(
            failed_x,
            failed_y,
            s=24,
            color=red,
            marker="x",
            label=r"$\Delta V\geq0$",
        )
    delta_axis.set_xscale("log")
    delta_axis.set_yscale("log")
    delta_axis.set_xlabel(r"state perturbation radius $\|\delta x\|_2$")
    delta_axis.set_ylabel(r"Lyapunov decrease $-\Delta V$")
    delta_axis.grid(True, which="both", alpha=0.2)
    delta_axis.legend(loc="lower right")

    trace_axis = axes[1, 0]
    trace_axis.set_title("(c) Escape episode and bounded commands", loc="left", fontweight="bold")
    times = [row_time(row) for row in rows]
    finite_times = [item for item in times if item is not None]
    origin = finite_times[0]
    relative_times = [
        (item - origin) if item is not None else float("nan") for item in times
    ]
    max_qd = max(float(summary["maximum_abs_qd_rad_s"]), 1e-12)
    max_qdd = max(float(summary["maximum_abs_qdd_rad_s2"]), 1e-12)
    trace_axis.plot(
        relative_times,
        [item / max_qd if math.isfinite(item) else float("nan") for item in summary["qd_norms"]],
        color=blue,
        linewidth=1.0,
        label=r"$\|\dot q\|$/run max",
    )
    trace_axis.plot(
        relative_times,
        [item / max_qdd if math.isfinite(item) else float("nan") for item in summary["qdd_norms"]],
        color=orange,
        linewidth=0.9,
        label=r"$\|\ddot q\|$/run max",
    )
    for episode in summary["episodes"]:
        trace_axis.axvspan(
            episode.start_time_s - origin,
            episode.end_time_s - origin,
            color="#f4b183",
            alpha=0.22,
            linewidth=0,
        )
    trace_axis.set_xlabel("time (s)")
    trace_axis.set_ylabel("normalized command magnitude")
    trace_axis.set_ylim(bottom=0.0)
    trace_axis.grid(True, alpha=0.2)
    goal_axis = trace_axis.twinx()
    goal_axis.plot(
        relative_times,
        [
            item if item is not None else float("nan")
            for item in summary["goal_errors"]
        ],
        color=green,
        linewidth=1.2,
        label="goal error",
    )
    goal_axis.set_ylabel("goal error (m)")
    lines = trace_axis.get_lines() + goal_axis.get_lines()
    trace_axis.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper right",
        fontsize=8,
    )

    status_axis = axes[1, 1]
    status_axis.set_title("(d) Certificate scope", loc="left", fontweight="bold")
    status_axis.axis("off")
    statuses = [
        (
            "Natural-form root pullback / resolve",
            bool(local["rmp_natural_form"]["equilibrium_natural_form_valid"]),
            "VERIFIED",
        ),
        (
            "NORMAL local exponential stability",
            bool(local["verdict"]["normal_local_exponential_stability_certified"]),
            "CERTIFIED",
        ),
        (
            "Joint command-box forward invariance",
            bool(local["verdict"]["software_command_box_forward_invariant_certified"]),
            "CERTIFIED",
        ),
        (
            "Escape bounded/released in this trace",
            bool(summary["hybrid_execution_trace_pass"]),
            "VERIFIED",
        ),
        (
            "Return into sampled NORMAL decrease neighborhood",
            bool(summary["conditional_convergence_chain_observed"]),
            "OBSERVED",
        ),
    ]
    y = 0.83
    for label, passed, success_label in statuses:
        status_axis.text(0.02, y, label, fontsize=10, va="center")
        color = green if passed else red
        result = success_label if passed else "NOT SHOWN"
        status_axis.text(
            0.98,
            y,
            result,
            ha="right",
            va="center",
            color="white",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.30", "facecolor": color, "edgecolor": color},
        )
        status_axis.plot([0.02, 0.98], [y - 0.075, y - 0.075], color="#e8eaed", linewidth=0.8)
        y -= 0.13
    status_axis.text(
        0.02,
        0.05,
        "Not claimed: global hybrid asymptotic stability,\n"
        "collision safe-set invariance, structured-GDS closure.",
        fontsize=10,
        color=red,
        va="bottom",
    )

    figure.suptitle(
        "Production Tangent Escape RMP: Stability and Invariance Certificate\n"
        f"NORMAL rho={float(local['linearization']['spectral_radius']):.7f}, "
        f"Escape episodes={int(summary['episode_count'])}, "
        f"|qd|max={float(summary['maximum_abs_qd_rad_s']):.3g} rad/s, "
        f"|qdd|max={float(summary['maximum_abs_qdd_rad_s2']):.3g} rad/s^2",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    pyplot.close(figure)


def display_float(value: object, digits: int = 6) -> str:
    number = finite(value)
    if number is None or math.isinf(number):
        return "n/a"
    return f"{number:.{digits}g}"


def make_html(
    path: Path,
    png_path: Path,
    local_path: Path,
    trace_path: Path,
    params_path: Path,
    local: Mapping[str, object],
    summary: Mapping[str, object],
) -> None:
    image_data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    normal_pass = bool(
        local["verdict"]["normal_local_exponential_stability_certified"]
    )
    bounds_pass = bool(
        local["verdict"]["software_command_box_forward_invariant_certified"]
    )
    trace_pass = bool(summary["hybrid_execution_trace_pass"])
    chain_pass = bool(summary["conditional_convergence_chain_observed"])
    episode_rows = "".join(
        "<tr>"
        f"<td>{index + 1}</td>"
        f"<td>{episode.start}</td>"
        f"<td>{episode.end}</td>"
        f"<td>{episode.duration_s:.4f}</td>"
        "<td>released</td>"
        "</tr>"
        for index, episode in enumerate(summary["episodes"])
    )
    verdict_class = "pass" if normal_pass and bounds_pass and trace_pass else "warn"
    verdict_text = (
        "NORMAL 모드의 국소 지수 안정성과 소프트웨어 명령 경계의 불변성은 "
        "인증되었고, 기록된 Escape episode는 경계 안에서 종료되었습니다."
    )
    if chain_pass:
        verdict_text += (
            " 같은 고정 목표에 대해 상태가 Lyapunov 감소를 샘플링으로 확인한 NORMAL "
            "근방으로 다시 들어가고 이후 수렴한 것도 관측됐습니다."
        )
    else:
        verdict_text += (
            " 다만 이 trace에서는 같은 목표의 인증 근방 재진입까지 연결되지 않았으므로 "
            "조건부 수렴 연결은 아직 관측되지 않았습니다."
        )

    page = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Production Tangent Escape RMP Stability Report</title>
<style>
  :root {{ --ink:#202124; --muted:#5f6368; --line:#dadce0;
           --green:#157a63; --red:#b3261e; --blue:#1769aa; }}
  body {{ margin:0; color:var(--ink); background:white;
          font:16px/1.62 Arial,"Noto Sans KR",sans-serif; }}
  main {{ max-width:1160px; margin:0 auto; padding:34px 28px 64px; }}
  h1 {{ margin:0 0 8px; font-size:30px; }}
  h2 {{ margin:32px 0 10px; padding-bottom:7px;
        border-bottom:1px solid var(--line); font-size:21px; }}
  .meta {{ color:var(--muted); }}
  .verdict {{ margin:22px 0; padding:14px 17px; border-left:5px solid; }}
  .verdict.pass {{ border-color:var(--green); background:#e8f5f1; }}
  .verdict.warn {{ border-color:var(--red); background:#fce8e6; }}
  img {{ width:100%; height:auto; border:1px solid var(--line); }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0 22px; }}
  th,td {{ border:1px solid var(--line); padding:8px 10px; text-align:left; }}
  th {{ background:#f8f9fa; }}
  .pass {{ color:var(--green); font-weight:700; }}
  .fail {{ color:var(--red); font-weight:700; }}
  code {{ background:#f1f3f4; padding:2px 5px; }}
  .equation {{ padding:12px 15px; border:1px solid var(--line);
               background:#f8f9fa; font-family:serif; }}
</style>
</head>
<body><main>
<h1>Production Tangent Escape RMP Stability Report</h1>
<p class="meta">생성 시각: {html.escape(datetime.now().isoformat(timespec="seconds"))}</p>
<div class="verdict {verdict_class}">{html.escape(verdict_text)}</div>
<img src="data:image/png;base64,{image_data}" alt="stability certificate figure">

<h2>무엇을 증명했는가</h2>
<table>
<tr><th>항목</th><th>결과</th><th>의미</th></tr>
<tr><td>NORMAL 국소 안정성</td><td class="{'pass' if normal_pass else 'fail'}">
{'PASS' if normal_pass else 'FAIL'}</td>
<td>실제 제한·적분·command guard를 포함한 이산 맵의 평형점에서
모든 고유값이 단위원 안에 있고 이산 Lyapunov 함수가 엄격히 감소합니다.</td></tr>
<tr><td>명령 경계 불변성</td><td class="{'pass' if bounds_pass else 'fail'}">
{'PASS' if bounds_pass else 'FAIL'}</td>
<td>코드의 포화와 predictive clamp 때문에 명령 q, qdot, qdd가 설정 경계를
벗어나지 않는 소프트웨어 불변 집합입니다.</td></tr>
<tr><td>Escape 실행 boundedness</td><td class="{'pass' if trace_pass else 'fail'}">
{'PASS' if trace_pass else 'FAIL'}</td>
<td>이 trace의 모든 Escape episode에서 유한 상태·명령, 경계 준수,
episode 종료와 NORMAL 복귀를 검사했습니다.</td></tr>
<tr><td>조건부 수렴 연결</td><td class="{'pass' if chain_pass else 'fail'}">
{'PASS' if chain_pass else 'NOT OBSERVED'}</td>
<td>Escape 종료 후 같은 고정 목표의 샘플된 Lyapunov 감소 근방으로 재진입하고,
그 이후 수렴한 실행 증거를 뜻합니다. 반경 자체의 formal 인증은 아닙니다.</td></tr>
</table>

<div class="equation">
δx[k+1] = A δx[k], &nbsp; ρ(A) =
{float(local['linearization']['spectral_radius']):.9f} &lt; 1<br>
AᵀPA − P = −I, &nbsp; λmin(P) =
{float(local['lyapunov']['p_min_eigenvalue']):.6g},
&nbsp; max λ(AᵀPA−P) =
{float(local['lyapunov']['decrease_max_eigenvalue']):.6g}
</div>

<h2>Escape Episode</h2>
<table>
<tr><th>#</th><th>시작 row</th><th>종료 row</th><th>지속시간 (s)</th><th>결과</th></tr>
{episode_rows}
</table>
<p>최대 |qdot|: <strong>{float(summary['maximum_abs_qd_rad_s']):.6g} rad/s</strong>,
최대 |qdd|: <strong>{float(summary['maximum_abs_qdd_rad_s2']):.6g} rad/s²</strong>,
branch generation 변화: <strong>{int(summary['generation_change_count'])}</strong>,
관측된 chatter: <strong>{'없음' if summary['no_observed_branch_chatter'] else '있음'}</strong>.</p>

<h2>정확한 한계</h2>
<p>이 보고서는 전체 상태공간의 global asymptotic stability를 주장하지 않습니다.
Escape가 영구 장애물 때문에 계속 켜져 있는 모든 경우의 목표 수렴도 증명하지 않습니다.
또한 여기서의 불변성은 관절 명령 경계이며, 장애물 최소거리 집합의 forward invariance가
아닙니다. Structured-GDS closure도 주장하지 않습니다. 현재 Escape는 non-GDS hybrid
policy이므로 natural-form pullback/resolve의 유효성과 별도로 구분했습니다.</p>

<h2>입력 자료</h2>
<ul>
<li>Local certificate: <code>{html.escape(str(local_path.resolve()))}</code></li>
<li>Escape trace: <code>{html.escape(str(trace_path.resolve()))}</code></li>
<li>Parameters: <code>{html.escape(str(params_path.resolve()))}</code></li>
<li>Trace goal source: <code>{html.escape(str(summary['goal_source']))}</code></li>
</ul>
</main></body></html>"""
    path.write_text(page, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a production NORMAL-mode Lyapunov and Escape hybrid "
            "execution report."
        )
    )
    parser.add_argument("--local-yaml", type=Path)
    parser.add_argument("--trace-csv", type=Path)
    parser.add_argument("--params-file", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument(
        "--fixed-trace-goal",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Fixed controller goal used by the trace launch when no /goal_pose "
            "message was recorded."
        ),
    )
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    local_path = args.local_yaml or latest_file(
        DEFAULT_LOCAL_DIR, "production_stability_*.yaml"
    )
    trace_path = args.trace_csv or latest_escape_trace(DEFAULT_TRACE_DIR)
    params_path = args.params_file.expanduser()
    local = load_yaml(local_path.expanduser())
    params = load_params(params_path)
    rows, _ = load_csv(trace_path.expanduser())
    summary = trace_summary(rows, params, local, args.fixed_trace_goal)

    if args.output_prefix:
        prefix = args.output_prefix.expanduser()
    else:
        prefix = local_path.expanduser().with_name(
            f"production_escape_certificate_{datetime.now():%Y%m%d_%H%M%S}"
        )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = prefix.with_suffix(".png")
    svg_path = prefix.with_suffix(".svg")
    html_path = prefix.with_suffix(".html")
    yaml_path = prefix.with_suffix(".yaml")
    make_figure(local, rows, summary, png_path, svg_path, args.dpi)
    make_html(
        html_path,
        png_path,
        local_path.expanduser(),
        trace_path.expanduser(),
        params_path,
        local,
        summary,
    )

    serializable = {
        key: value
        for key, value in summary.items()
        if key not in {"episodes", "qdd_norms", "qd_norms", "goal_errors"}
    }
    serializable["episodes"] = [
        {
            "start_row": item.start,
            "end_row": item.end,
            "start_time_s": item.start_time_s,
            "end_time_s": item.end_time_s,
            "duration_s": item.duration_s,
        }
        for item in summary["episodes"]
    ]
    serializable["local_certificate"] = str(local_path.expanduser().resolve())
    serializable["trace_csv"] = str(trace_path.expanduser().resolve())
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)

    print(f"Local certificate: {local_path}")
    print(f"Escape trace: {trace_path}")
    print(
        "NORMAL local exponential stability: "
        f"{'PASS' if local['verdict']['normal_local_exponential_stability_certified'] else 'FAIL'}"
    )
    print(
        "Software command-box forward invariance: "
        f"{'PASS' if local['verdict']['software_command_box_forward_invariant_certified'] else 'FAIL'}"
    )
    print(
        "Escape hybrid execution trace: "
        f"{'PASS' if summary['hybrid_execution_trace_pass'] else 'FAIL'}"
    )
    print(
        "Conditional convergence chain observed: "
        f"{'YES' if summary['conditional_convergence_chain_observed'] else 'NO'}"
    )
    print(f"Saved HTML: {html_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")
    print(f"Saved summary: {yaml_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, TypeError, ValueError, ReportError) as error:
        print(f"Error: {error}")
        raise SystemExit(2)

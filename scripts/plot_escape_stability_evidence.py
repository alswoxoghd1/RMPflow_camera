#!/usr/bin/env python3
"""Create a paper-style Escape RMP stability evidence figure and HTML report."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import yaml


CERTIFICATE_FIELDS = (
    "stability_schema_version",
    "stability_base_gds_structural",
    "stability_environment_static",
    "stability_external_rmp_empty",
    "stability_guard_enabled",
    "stability_conditional_nonincrease",
    "stability_escape_scale",
    "stability_tank_energy",
    "stability_tank_capacity",
    "stability_escape_power_requested",
    "stability_escape_power_applied",
    "stability_solve_power",
    "stability_clamp_power",
    "stability_positive_escape_energy",
    "stability_negative_escape_energy",
    "stability_energy_bound_violation",
    "stability_nonincrease_upper_bound",
    "stability_initial_energy",
    "stability_escape_active",
)


@dataclass(frozen=True)
class Sample:
    time_s: float
    values: Dict[str, float]


@dataclass(frozen=True)
class Audit:
    path: Path
    samples: List[Sample]

    def values(self, field: str) -> List[float]:
        return [sample.values[field] for sample in self.samples]

    @property
    def times(self) -> List[float]:
        origin = self.samples[0].time_s
        return [sample.time_s - origin for sample in self.samples]


class ReportError(RuntimeError):
    pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine a finite-budget guarded run and a large-budget task-energy "
            "run into a stability evidence PNG, SVG, and HTML report."
        )
    )
    parser.add_argument("--guarded-csv", required=True, type=Path)
    parser.add_argument("--strict-csv", required=True, type=Path)
    parser.add_argument("--task-energy-csv", required=True, type=Path)
    parser.add_argument("--params-file", required=True, type=Path)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Output path without extension (default: beside guarded CSV)",
    )
    parser.add_argument(
        "--title",
        default="Tangent Escape RMP: Non-GDS Stability Evidence",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def finite_float(value: Optional[str]) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def load_audit(path: Path) -> Audit:
    if not path.is_file():
        raise ReportError(f"CSV does not exist: {path}")
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = [field for field in CERTIFICATE_FIELDS if field not in fields]
        if missing:
            raise ReportError(
                f"{path} is missing stability fields: {', '.join(missing)}"
            )
        if "time_ros_s" not in fields:
            raise ReportError(f"{path} is missing time_ros_s")
        for row in reader:
            time_s = finite_float(row.get("time_ros_s"))
            values = {
                field: finite_float(row.get(field)) for field in CERTIFICATE_FIELDS
            }
            if (
                time_s is None
                or any(value is None for value in values.values())
                or abs(float(values["stability_schema_version"]) - 1.0) > 1e-9
            ):
                continue
            samples.append(
                Sample(
                    time_s=time_s,
                    values={field: float(value) for field, value in values.items()},
                )
            )
    if not samples:
        raise ReportError(f"No complete schema-v1 certificate samples in {path}")
    return Audit(path=path.resolve(), samples=samples)


def load_params(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise ReportError(f"Parameter file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    try:
        params = data["rmpflow_controller"]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise ReportError(f"Unexpected ROS parameter structure in {path}") from error
    if not isinstance(params, Mapping):
        raise ReportError(f"ROS parameters are not a mapping in {path}")
    return params


def ratio(values: Iterable[float], threshold: float = 0.5) -> float:
    sequence = list(values)
    return sum(value > threshold for value in sequence) / max(len(sequence), 1)


def active_spans(audit: Audit) -> List[tuple[float, float]]:
    times = audit.times
    active = audit.values("stability_escape_active")
    spans: List[tuple[float, float]] = []
    start: Optional[float] = None
    for time_s, flag in zip(times, active):
        if flag > 0.5 and start is None:
            start = time_s
        if flag <= 0.5 and start is not None:
            spans.append((start, time_s))
            start = None
    if start is not None:
        spans.append((start, times[-1]))
    return spans


def param_float(params: Mapping[str, object], name: str) -> float:
    try:
        value = float(params[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ReportError(f"Missing or invalid parameter: {name}") from error
    if not math.isfinite(value):
        raise ReportError(f"Non-finite parameter: {name}")
    return value


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    from matplotlib.patches import FancyBboxPatch

    return pyplot, FancyBboxPatch


def add_box(axis, patch_type, xy, width, height, text, color):
    patch = patch_type(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        linewidth=1.4,
        edgecolor=color,
        facecolor="white",
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=9,
    )


def make_figure(
    guarded: Audit,
    strict: Audit,
    task: Audit,
    params: Mapping[str, object],
    title: str,
    png_path: Path,
    svg_path: Path,
    dpi: int,
) -> Dict[str, float]:
    pyplot, patch_type = import_pyplot()
    blue = "#1769aa"
    orange = "#e66a1f"
    green = "#1b9e77"
    red = "#c62828"
    purple = "#6a4c93"
    gray = "#5f6368"

    metric = param_float(params, "tangent_escape_rmp_metric_scalar")
    max_accel = param_float(params, "tangent_escape_rmp_max_accel")
    production_guard = bool(params.get("escape_stability_guard_enabled", False))
    configured_budget = param_float(params, "escape_stability_tank_capacity")

    guard_scale = guarded.values("stability_escape_scale")
    guard_violation = guarded.values("stability_energy_bound_violation")
    guard_residual = guarded.values("stability_nonincrease_upper_bound")
    guard_tank = guarded.values("stability_tank_energy")
    guard_initial = guarded.values("stability_initial_energy")[-1]
    guard_spent = max(guard_initial - guard_tank[-1], 0.0)
    guard_limited_ratio = ratio([1.0 if value < 0.999999 else 0.0 for value in guard_scale])

    task_tank = task.values("stability_tank_energy")
    task_initial = task.values("stability_initial_energy")[-1]
    task_energy = max(task_initial - task_tank[-1], 0.0)
    task_scale_min = min(task.values("stability_escape_scale"))
    budget_fraction = configured_budget / task_energy if task_energy > 0.0 else math.inf

    strict_scale = strict.values("stability_escape_scale")
    strict_applied_power = strict.values("stability_escape_power_applied")
    strict_positive_energy = strict.values("stability_positive_escape_energy")
    strict_violation = strict.values("stability_energy_bound_violation")
    static_ratio = ratio(strict.values("stability_environment_static"))
    structural_ratio = ratio(strict.values("stability_base_gds_structural"))
    conditional_ratio = ratio(strict.values("stability_conditional_nonincrease"))
    static_rows = [
        sample
        for sample in strict.samples
        if sample.values["stability_environment_static"] > 0.5
    ]
    static_conditional_ratio = ratio(
        sample.values["stability_conditional_nonincrease"]
        for sample in static_rows
    )

    figure, axes = pyplot.subplots(2, 2, figsize=(15.5, 10.5))
    figure.patch.set_facecolor("white")

    architecture = axes[0, 0]
    architecture.set_xlim(0.0, 1.0)
    architecture.set_ylim(0.0, 1.0)
    architecture.axis("off")
    architecture.set_title("(a) Non-GDS Escape interconnection", loc="left", fontweight="bold")
    add_box(
        architecture,
        patch_type,
        (0.03, 0.63),
        0.30,
        0.19,
        "Base controller\nstorage $V_0$, dissipation $D_0$",
        blue,
    )
    add_box(
        architecture,
        patch_type,
        (0.03, 0.19),
        0.30,
        0.22,
        "Arbitrary hybrid Escape\nscore, memory, switching",
        orange,
    )
    add_box(
        architecture,
        patch_type,
        (0.43, 0.38),
        0.23,
        0.22,
        "Power monitor\n$P_e=\\dot q^T(f_e-M_e\\ddot q)$",
        purple,
    )
    add_box(
        architecture,
        patch_type,
        (0.75, 0.38),
        0.22,
        0.22,
        "Metric/force scale\n$\\alpha\\in[0,1]$",
        green,
    )
    architecture.annotate("", (0.43, 0.53), (0.33, 0.72), arrowprops={"arrowstyle": "->"})
    architecture.annotate("", (0.43, 0.45), (0.33, 0.30), arrowprops={"arrowstyle": "->"})
    architecture.annotate("", (0.75, 0.49), (0.66, 0.49), arrowprops={"arrowstyle": "->"})
    architecture.text(
        0.50,
        0.06,
        "finite tank: $\\int\\max(P_e,0)dt\\leq E_0$ "
        "$\\Rightarrow V_0(t)\\leq V_0(0)+E_0$\n"
        "strict: $P_e\\leq0$ $\\Rightarrow\\dot V_0\\leq-D_0\\leq0$",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#202124",
    )

    power_axis = axes[0, 1]
    power_axis.set_title(
        "(b) Strict nonpositive-power execution", loc="left", fontweight="bold"
    )
    requested = strict.values("stability_escape_power_requested")
    applied = strict_applied_power
    power_axis.plot(strict.times, requested, color=orange, linewidth=1.0, label="requested $P_e$")
    power_axis.plot(strict.times, applied, color=blue, linewidth=1.1, label="applied $P_e$")
    power_axis.axhline(0.0, color="#202124", linewidth=0.8)
    power_axis.set_yscale("symlog", linthresh=1.0)
    power_axis.set_xlabel("time (s)")
    power_axis.set_ylabel("RMP power")
    for start, end in active_spans(strict):
        power_axis.axvspan(start, end, color="#f4b183", alpha=0.12, linewidth=0)
    scale_axis = power_axis.twinx()
    scale_axis.plot(
        strict.times,
        strict_scale,
        color=green,
        linewidth=1.2,
        label="guard scale $\\alpha$",
    )
    scale_axis.set_ylim(-0.04, 1.04)
    scale_axis.set_ylabel("scale")
    lines = power_axis.get_lines()[:2] + scale_axis.get_lines()
    power_axis.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=8)
    power_axis.grid(True, alpha=0.22)

    energy_axis = axes[1, 0]
    energy_axis.set_title("(c) Task-energy calibration", loc="left", fontweight="bold")
    task_positive = task.values("stability_positive_escape_energy")
    task_depletion = [max(task_initial - value, 0.0) for value in task_tank]
    energy_axis.plot(
        task.times,
        task_positive,
        color=orange,
        linewidth=1.6,
        label="cumulative positive Escape energy",
    )
    energy_axis.plot(
        task.times,
        task_depletion,
        color=blue,
        linestyle="--",
        linewidth=1.2,
        label="tank depletion",
    )
    energy_axis.axhline(
        configured_budget,
        color=red,
        linestyle=":",
        linewidth=1.5,
        label=f"configured budget {configured_budget:g}",
    )
    energy_axis.set_yscale("log")
    energy_axis.set_xlabel("time (s)")
    energy_axis.set_ylabel("RMP storage units (not joules)")
    energy_axis.grid(True, alpha=0.22)
    energy_axis.legend(loc="upper left", fontsize=8)
    energy_axis.text(
        0.98,
        0.06,
        f"measured demand in this run: {task_energy:.4g}\n"
        f"budget coverage: {100.0 * budget_fraction:.4g}%\n"
        f"large-budget scale min: {task_scale_min:.3g}",
        transform=energy_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#dadce0", "alpha": 0.92},
    )

    status_axis = axes[1, 1]
    status_axis.set_title("(d) Certificate scope and verdict", loc="left", fontweight="bold")
    status_axis.axis("off")
    statuses = [
        ("Rank-one metric PSD and power identity", "VERIFIED", green),
        ("Strict nonpositive-power stability", "VERIFIED", green),
        ("Finite positive-energy bound", "VERIFIED", green),
        (
            "Certificate pass within static rows",
            f"{100.0 * static_conditional_ratio:.1f}%",
            blue,
        ),
        ("Current production base profile", "NOT CERTIFIED", red),
        ("Global asymptotic hybrid stability", "NOT PROVEN", red),
    ]
    y = 0.80
    for label, result, color in statuses:
        status_axis.text(0.02, y, label, va="center", fontsize=10, color="#202124")
        status_axis.text(
            0.98,
            y,
            result,
            va="center",
            ha="right",
            fontsize=9,
            color="white",
            bbox={"boxstyle": "round,pad=0.32", "facecolor": color, "edgecolor": color},
        )
        status_axis.plot([0.02, 0.98], [y - 0.075, y - 0.075], color="#e8eaed", linewidth=0.8)
        y -= 0.125
    status_axis.text(
        0.02,
        0.005,
        "Current conclusion: the non-GDS Escape interconnection has a valid\n"
        "strict stability and finite-energy certificate in the proof profile. The normal\n"
        "production controller is not yet a formally certified Lyapunov-stable system.",
        va="bottom",
        fontsize=9,
        color="#202124",
    )

    subtitle = (
        f"$m_t={metric:g}$, $|a_t|\\leq {max_accel:g}$ m/s$^2$; "
        f"strict positive energy={strict_positive_energy[-1]:.3g}, "
        f"violation max={max(strict_violation):.3g}, "
        f"proof static-row pass={100.0 * static_conditional_ratio:.1f}%, "
        f"production guard={'ON' if production_guard else 'OFF'}"
    )
    figure.suptitle(f"{title}\n{subtitle}", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    pyplot.close(figure)

    return {
        "metric": metric,
        "max_accel": max_accel,
        "production_guard": float(production_guard),
        "configured_budget": configured_budget,
        "guard_spent": guard_spent,
        "guard_scale_min": min(guard_scale),
        "guard_limited_ratio": guard_limited_ratio,
        "guard_violation_max": max(guard_violation),
        "guard_residual_abs_max": max(abs(value) for value in guard_residual),
        "strict_scale_min": min(strict_scale),
        "strict_applied_positive_max": max(
            max(value, 0.0) for value in strict_applied_power
        ),
        "strict_positive_energy": strict_positive_energy[-1],
        "strict_violation_max": max(strict_violation),
        "task_energy": task_energy,
        "task_scale_min": task_scale_min,
        "budget_fraction": budget_fraction,
        "structural_ratio": structural_ratio,
        "static_ratio": static_ratio,
        "conditional_ratio": conditional_ratio,
        "static_conditional_ratio": static_conditional_ratio,
    }


def make_html(
    output: Path,
    png_path: Path,
    guarded: Audit,
    strict: Audit,
    task: Audit,
    params_path: Path,
    stats: Mapping[str, float],
    title: str,
) -> None:
    png_data = base64.b64encode(png_path.read_bytes()).decode("ascii")

    def percent(value: float) -> str:
        return f"{100.0 * value:.2f}%"

    production_guard = stats["production_guard"] > 0.5
    current_verdict = (
        "현재 운용 profile 전체의 Lyapunov stability는 아직 증명되지 않았다."
    )
    body = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{
    color-scheme: light;
    --ink: #202124; --muted: #5f6368; --line: #dadce0;
    --blue: #1769aa; --green: #147d64; --orange: #c75300; --red: #b3261e;
  }}
  body {{ margin: 0; font: 16px/1.62 Arial, sans-serif; color: var(--ink); background: #fff; }}
  main {{ max-width: 1180px; margin: 0 auto; padding: 36px 28px 64px; }}
  h1 {{ font-size: 30px; margin: 0 0 8px; }}
  h2 {{ font-size: 21px; margin: 34px 0 10px; border-bottom: 1px solid var(--line); padding-bottom: 7px; }}
  h3 {{ font-size: 17px; margin: 22px 0 7px; }}
  p, li {{ max-width: 980px; }}
  .meta {{ color: var(--muted); margin-bottom: 24px; }}
  .verdict {{ border-left: 5px solid var(--red); padding: 13px 16px; background: #fce8e6; font-weight: 700; }}
  .verified {{ color: var(--green); font-weight: 700; }}
  .conditional {{ color: var(--blue); font-weight: 700; }}
  .unproven {{ color: var(--red); font-weight: 700; }}
  .equation {{ padding: 12px 16px; background: #f8f9fa; border: 1px solid var(--line); overflow-x: auto; font-family: serif; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0 20px; }}
  th, td {{ border-bottom: 1px solid var(--line); padding: 10px 9px; text-align: left; vertical-align: top; }}
  th {{ background: #f8f9fa; }}
  figure {{ margin: 24px 0; }}
  figure img {{ width: 100%; height: auto; border: 1px solid var(--line); }}
  figcaption {{ color: var(--muted); margin-top: 8px; font-size: 14px; }}
  code {{ background: #f1f3f4; padding: 2px 5px; }}
  .small {{ color: var(--muted); font-size: 14px; }}
</style>
</head>
<body><main>
<h1>{html.escape(title)}</h1>
<p class="meta">finite-tank 로그: {html.escape(str(guarded.path))}<br>
strict stability 로그: {html.escape(str(strict.path))}<br>
대규모 tank 로그: {html.escape(str(task.path))}<br>
파라미터: {html.escape(str(params_path.resolve()))}</p>
<p class="verdict">{html.escape(current_verdict)}</p>

<figure>
  <img src="data:image/png;base64,{png_data}" alt="Escape RMP stability evidence">
  <figcaption>
    (a) 비-GDS Escape를 power port로 연결한 증명 구조.
    (b) zero-capacity strict guard에서 요청/적용 power와 scale.
    (c) 큰 tank 실행에서 측정한 task energy.
    (d) 현재 증명 범위. 에너지 값은 RMP metric에 따른 storage 단위이며 물리적 joule이 아니다.
  </figcaption>
</figure>

<h2>1. Escape RMP가 왜 단일 GDS가 아닌가</h2>
<p>
현재 Escape는 후보 score, 이전 방향, blocked memory, stuck timer와 mode switching으로
방향을 고른다. 따라서 force가 현재의 <code>(q, qdot)</code>만으로 정해지는 하나의
smooth potential-gradient/damping field가 아니다. 각 고정 branch를 별도 GDS로
재설계할 수는 있지만, 모든 memory와 switching을 유지한 전체 정책을 원래 RMPflow의
단일 smooth structured-GDS라고 부를 수는 없다. 즉 수학적으로 영원히 불가능한 것이
아니라, <strong>현재 기능을 그대로 둔 표현이 단일 GDS가 아니다.</strong>
</p>

<h2>2. 비-GDS leaf를 사용해도 되는가</h2>
<p>
RMP natural form의 계산 자체는 임의의 대칭 PSD metric과 finite natural force를
pullback할 수 있다. 따라서 구현상 가능하다. 다만 원래 structured-GDS closure
정리가 자동 적용되지 않으므로, 추가 leaf가 base controller에 주입하는 power를
별도로 제한해야 한다.
</p>

<h2>3. 사용한 Lyapunov/passivity 증명</h2>
<p>Escape가 없는 base controller에 저장함수 <code>V0</code>와 소산 <code>D0</code>가 있다고 가정한다.</p>
<div class="equation">V̇₀ = −D₀, &nbsp; D₀ ≥ 0</div>
<p>Escape의 root natural-form 기여를 <code>(fe, Me)</code>라 하면 정확한 interconnection power는 다음이다.</p>
<div class="equation">Pₑ = q̇ᵀ(fe − Me q̈), &nbsp; V̇₀ = −D₀ + Pₑ</div>
<p>
따라서 strict guard가 <code>Pₑ ≤ σD₀</code>, <code>0 ≤ σ &lt; 1</code>을 항상
만족시키면 Escape가 GDS가 아니어도 다음이 성립한다.
</p>
<div class="equation">V̇₀ ≤ −(1−σ)D₀ ≤ 0</div>
<p>
현재 production 기본값은 finite-energy tank guard도 꺼져 있다. proof 실행에서는
같은 scaling 구현의 tank capacity를 0으로 설정하여 strict guard를 구성했다.
일반 finite-tank 모드는 양의 power의
누적합을 초기 tank보다 작게 제한하여 <code>V0(t) ≤ V0(0)+E0</code>를 보장한다.
이것은 <span class="verified">유한 에너지 boundedness/passivity 보장</span>이지만,
고정된 양의 <code>E0</code>만으로 물리 상태의 global asymptotic stability까지
자동 증명되지는 않는다.
</p>

<h2>4. 무엇을 왜 검증했는가</h2>
<table>
<thead><tr><th>검증</th><th>목적</th><th>방법</th><th>결과</th></tr></thead>
<tbody>
<tr><td>Power identity</td><td>guard가 올바른 에너지를 재는지 확인</td><td>direct solve와 rank-one 식 비교</td><td class="verified">PASS</td></tr>
<tr><td>Metric 구조</td><td>root metric을 망가뜨리지 않는지 확인</td><td>대칭성, PSD, rank-one 고유값 검사</td><td class="verified">PASS</td></tr>
<tr><td>Strict power guard</td><td>비-GDS Escape가 V0를 증가시키지 않는지 확인</td><td>zero-capacity 실행에서 applied positive power와 energy 검사</td><td class="verified">PASS</td></tr>
<tr><td>Tank non-negativity</td><td>허용 에너지 초과 방지</td><td>metric/force 동시 scale 및 누적합 검사</td><td class="verified">PASS</td></tr>
<tr><td>Base proof profile</td><td>V0가 존재하는 조건 확인</td><td>GDS leaf, smooth task map, SPD root, exact solve 검사</td><td class="conditional">CONDITIONAL PASS</td></tr>
<tr><td>Runtime residual</td><td>수치 구현이 식을 지키는지 확인</td><td>solve/clamp power 및 energy-bound residual 기록</td><td class="verified">PASS</td></tr>
</tbody>
</table>

<h2>5. 현재 결과</h2>
<table>
<tbody>
<tr><th>현재 Escape metric</th><td>{stats['metric']:.6g}</td></tr>
<tr><th>현재 tangent acceleration limit</th><td>{stats['max_accel']:.6g} m/s²</td></tr>
<tr><th>proof base structural ratio</th><td>{percent(stats['structural_ratio'])}</td></tr>
<tr><th>환경 고정 row 비율</th><td>{percent(stats['static_ratio'])}</td></tr>
<tr><th>환경 고정 row에서 certificate 통과</th><td>{percent(stats['static_conditional_ratio'])}</td></tr>
<tr><th>strict 적용 양의 power 최대</th><td>{stats['strict_applied_positive_max']:.6g}</td></tr>
<tr><th>strict 누적 양의 energy</th><td>{stats['strict_positive_energy']:.6g}</td></tr>
<tr><th>strict energy-bound violation 최대</th><td>{stats['strict_violation_max']:.6g}</td></tr>
<tr><th>finite-tank run 최소 scale</th><td>{stats['guard_scale_min']:.6g}</td></tr>
<tr><th>finite-tank guard가 제한한 row 비율</th><td>{percent(stats['guard_limited_ratio'])}</td></tr>
<tr><th>측정 task energy</th><td>{stats['task_energy']:.6g} RMP storage units</td></tr>
<tr><th>현재 설정 tank</th><td>{stats['configured_budget']:.6g} ({percent(stats['budget_fraction'])} of measured demand)</td></tr>
<tr><th>production에서 guard</th><td class="{'verified' if production_guard else 'unproven'}">{'ON' if production_guard else 'OFF'}</td></tr>
</tbody>
</table>
<p>
strict proof profile에서는 적용된 양의 Escape energy가
<code>{stats['strict_positive_energy']:.3g}</code>이고 위반도
<code>{stats['strict_violation_max']:.3g}</code>이었다. 따라서 환경 고정 및 base
storage 가정 아래에서는 <code>V0</code> 비증가를 보존했다. 반면 현재
<code>0.25</code> finite tank는 빠르게 고갈되어 scale이
<code>{stats['guard_scale_min']:.3g}</code>까지 감소했다. 현재 계수로 Escape 성능과
certificate를 동시에 유지하려면 task-energy calibration 또는 damping power를
사용하는 strict budget 설계가 필요하다.
</p>

<h2>6. 지금 증명된 것과 증명되지 않은 것</h2>
<p class="verified">증명/검증됨: proof profile에서 Escape의 PSD rank-one 결합, 정확한 port-power 계측, strict Lyapunov 비증가, 양의 에너지 상한, 수치 residual.</p>
<p class="conditional">조건부: 고정 goal/obstacle와 exact continuous root solve를 가정한 base Lyapunov storage.</p>
<p class="unproven">미증명: 현재 production profile 전체와 물리 상태의 global asymptotic hybrid stability.</p>

<h2>7. 논문에 사용할 주장</h2>
<p>
“The score- and memory-driven Escape policy is not assumed to be a GDS.
Instead, it is connected to a Lyapunov-stable base RMPflow controller through
an exactly measured natural-form power port. A metric/force scaling guard bounds
the positive energy injected by Escape. Under a fixed environment, the
base-storage assumptions, and a zero-capacity strict guard, the base Lyapunov
function is nonincreasing; both the recorded positive Escape energy and the
energy-bound violation remained zero in the reported experiment.”
</p>
<p class="small">
이 문장에는 production profile 전체의 global asymptotic stability를 증명했다는
표현을 넣으면 안 된다.
</p>

<h2>8. 다음 작업</h2>
<ol>
  <li>production과 proof base의 차이를 제거하거나, 현재 no-Escape production controller의 별도 CLF/storage를 확립한다.</li>
  <li>현재 검증한 <code>Pₑ ≤ 0</code> strict mode를 production-selectable parameter로 명시하고, 이후 <code>Pₑ ≤ σD₀</code>로 확장해 성능을 확보한다.</li>
  <li>현재 계수에 필요한 task energy를 여러 장애물/초기 자세에서 반복 측정하고 percentile 기반 tank budget을 정한다.</li>
  <li>goal/obstacle가 바뀌는 구간의 explicit-time power를 포함한다.</li>
  <li>zero-effect handoff와 최소 dwell time을 이용해 Zeno가 없고 state reset energy가 0임을 정리한다.</li>
  <li>시뮬레이션에서 통과한 동일 certificate를 실제 로봇 로그에서 재검증한다.</li>
</ol>

<h2>참고 이론</h2>
<ul>
  <li><a href="https://arxiv.org/abs/2007.14256">RMPflow: A Geometric Framework for Generation of Multi-Task Motion Policies</a></li>
  <li><a href="https://arxiv.org/abs/2301.12759">Learning Passive Policies with Virtual Energy Tanks in Robotics</a></li>
  <li><a href="https://arxiv.org/abs/1903.12605">Stable, Concurrent Controller Composition for Multi-Objective Robotic Tasks</a></li>
</ul>
</main></body></html>
"""
    output.write_text(body, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        guarded = load_audit(args.guarded_csv.expanduser())
        strict = load_audit(args.strict_csv.expanduser())
        task = load_audit(args.task_energy_csv.expanduser())
        params_path = args.params_file.expanduser()
        params = load_params(params_path)
        prefix = (
            args.output_prefix.expanduser()
            if args.output_prefix
            else guarded.path.with_name(f"{guarded.path.stem}_stability_evidence")
        )
        png_path = prefix.with_suffix(".png")
        svg_path = prefix.with_suffix(".svg")
        html_path = prefix.with_suffix(".html")
        for path in (png_path, svg_path, html_path):
            if path.exists() and not args.overwrite:
                raise ReportError(f"Output exists; use --overwrite: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
        stats = make_figure(
            guarded,
            strict,
            task,
            params,
            args.title,
            png_path,
            svg_path,
            args.dpi,
        )
        make_html(
            html_path,
            png_path,
            guarded,
            strict,
            task,
            params_path,
            stats,
            args.title,
        )
    except (OSError, ReportError, yaml.YAMLError) as error:
        print(f"Error: {error}")
        return 2

    print(f"Guarded CSV: {guarded.path}")
    print(f"Strict CSV: {strict.path}")
    print(f"Task-energy CSV: {task.path}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")
    print(f"Saved HTML: {html_path}")
    print(
        "Verdict: strict proof-profile stability PASS; "
        "finite-energy certificate PASS; "
        "production Lyapunov stability NOT CERTIFIED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a focused same-state Tangent Escape ON/OFF comparison report."""

import argparse
import base64
import csv
import html
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TIME_FIELD = "time_ros_s"
ACTIVE_FIELD = "tangent_escape_dual_solve_active"
ACTIVATION_FIELD = "tangent_escape_dual_activation"
METRIC_FIELD = "tangent_escape_dual_effective_metric_scalar"
ORTHOGONALITY_FIELD = "tangent_escape_rmp_normal_dot_tangent"

QDD_WITH_FIELDS = [
    f"tangent_escape_qdd_with_{joint}_rad_s2" for joint in range(1, 7)
]
QDD_WITHOUT_FIELDS = [
    f"tangent_escape_qdd_without_{joint}_rad_s2" for joint in range(1, 7)
]
QDD_DELTA_FIELDS = [
    f"tangent_escape_delta_qdd_{joint}_rad_s2" for joint in range(1, 7)
]
CP_WITH_FIELDS = [
    f"tangent_escape_cp_accel_with_{axis}_m_s2" for axis in ("x", "y", "z")
]
CP_WITHOUT_FIELDS = [
    f"tangent_escape_cp_accel_without_{axis}_m_s2" for axis in ("x", "y", "z")
]
CP_DELTA_FIELDS = [
    f"tangent_escape_delta_cp_accel_{axis}_m_s2" for axis in ("x", "y", "z")
]
TANGENT_FIELDS = [
    f"tangent_escape_rmp_tangent_{axis}" for axis in ("x", "y", "z")
]
NORMAL_FIELDS = [
    f"tangent_escape_rmp_normal_{axis}" for axis in ("x", "y", "z")
]

SCALAR_FIELDS = [
    "tangent_escape_qdd_with_norm",
    "tangent_escape_qdd_without_norm",
    "tangent_escape_delta_qdd_norm",
    "tangent_escape_delta_tcp_accel_norm_m_s2",
    "tangent_escape_delta_cp_accel_norm_m_s2",
    "tangent_escape_delta_tcp_accel_dot_tangent_m_s2",
    "tangent_escape_delta_tcp_accel_dot_normal_m_s2",
    "tangent_escape_delta_cp_accel_dot_tangent_m_s2",
    "tangent_escape_delta_cp_accel_dot_normal_m_s2",
    ACTIVE_FIELD,
    ACTIVATION_FIELD,
    METRIC_FIELD,
]

REQUIRED_FIELDS = {
    TIME_FIELD,
    ORTHOGONALITY_FIELD,
    *SCALAR_FIELDS,
    *QDD_WITH_FIELDS,
    *QDD_WITHOUT_FIELDS,
    *QDD_DELTA_FIELDS,
    *CP_WITH_FIELDS,
    *CP_WITHOUT_FIELDS,
    *CP_DELTA_FIELDS,
    *TANGENT_FIELDS,
    *NORMAL_FIELDS,
}


class ComparisonError(RuntimeError):
    """Raised when a trace cannot provide an exact same-state comparison."""


@dataclass(frozen=True)
class Sample:
    time_s: float
    values: Dict[str, float]


@dataclass(frozen=True)
class Stats:
    maximum: float
    mean: float
    p95: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot an exact same-state Tangent Escape ON/OFF comparison from "
            "rmpflow_trace dual-solve columns."
        )
    )
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        help=(
            "Input rmpflow_trace CSV. If omitted, use the newest compatible "
            "CSV below ~/ros2_ws/log/rmpflow_trace."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        help="Output directory. Defaults to the input CSV directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing PNG and HTML report.",
    )
    return parser.parse_args(argv)


def finite_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = float(stripped)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def header_fields(path: Path) -> set[str]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return set(next(csv.reader(stream), []))
    except (OSError, csv.Error):
        return set()


def latest_compatible_csv(root: Path) -> Path:
    if not root.exists():
        raise ComparisonError(f"Trace directory does not exist: {root}")
    candidates = sorted(
        root.rglob("*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if REQUIRED_FIELDS.issubset(header_fields(path)):
            return path
    raise ComparisonError(
        f"No CSV with complete Tangent Escape dual-solve columns found in {root}"
    )


def load_samples(path: Path) -> List[Sample]:
    if not path.is_file():
        raise ComparisonError(f"Input CSV does not exist: {path}")
    samples: List[Sample] = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_FIELDS - fieldnames)
            if missing:
                raise ComparisonError(
                    "Missing same-state dual-solve columns: " + ", ".join(missing)
                )
            first_time: Optional[float] = None
            for row in reader:
                stamp = finite_float(row.get(TIME_FIELD))
                if stamp is None:
                    continue
                values: Dict[str, float] = {}
                complete = True
                for field in REQUIRED_FIELDS - {TIME_FIELD}:
                    value = finite_float(row.get(field))
                    if value is None:
                        complete = False
                        break
                    values[field] = value
                if not complete:
                    continue
                if first_time is None:
                    first_time = stamp
                samples.append(Sample(stamp - first_time, values))
    except (OSError, csv.Error) as error:
        raise ComparisonError(f"Could not read {path}: {error}") from error
    if not samples:
        raise ComparisonError(f"No complete dual-solve rows found in {path}")
    if not any(sample.values[ACTIVE_FIELD] >= 0.5 for sample in samples):
        raise ComparisonError(f"No active Tangent Escape dual-solve rows found in {path}")
    return samples


def vector(sample: Sample, fields: Sequence[str]) -> List[float]:
    return [sample.values[field] for field in fields]


def norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def dot(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    return sum(left * right for left, right in zip(lhs, rhs))


def scaled(values: Sequence[float], scale: float) -> List[float]:
    return [scale * value for value in values]


def subtract(lhs: Sequence[float], rhs: Sequence[float]) -> List[float]:
    return [left - right for left, right in zip(lhs, rhs)]


def normalized(values: Sequence[float]) -> List[float]:
    magnitude = norm(values)
    if magnitude <= 1.0e-12:
        return [0.0 for _ in values]
    return [value / magnitude for value in values]


def angle_degrees(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    denominator = norm(lhs) * norm(rhs)
    if denominator <= 1.0e-12:
        return 0.0
    cosine = max(-1.0, min(1.0, dot(lhs, rhs) / denominator))
    return math.degrees(math.acos(cosine))


def percentile(values: Sequence[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = ratio * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: Sequence[float]) -> Stats:
    if not values:
        return Stats(0.0, 0.0, 0.0)
    return Stats(max(values), statistics.fmean(values), percentile(values, 0.95))


def active_samples(samples: Sequence[Sample]) -> List[Sample]:
    return [sample for sample in samples if sample.values[ACTIVE_FIELD] >= 0.5]


def active_spans(samples: Sequence[Sample]) -> List[Tuple[float, float]]:
    times = [sample.time_s for sample in samples]
    gaps = [
        current - previous
        for previous, current in zip(times, times[1:])
        if current > previous
    ]
    fallback = statistics.median(gaps) if gaps else 0.01
    spans: List[Tuple[float, float]] = []
    start: Optional[float] = None
    for index, sample in enumerate(samples):
        active = sample.values[ACTIVE_FIELD] >= 0.5
        if active and start is None:
            start = sample.time_s
        if not active and start is not None:
            spans.append((start, sample.time_s))
            start = None
        if start is not None and index == len(samples) - 1:
            spans.append((start, sample.time_s + fallback))
    return spans


def shade(axis: object, spans: Iterable[Tuple[float, float]]) -> None:
    for start, end in spans:
        axis.axvspan(start, end, color="#F59E0B", alpha=0.10, linewidth=0)


def style_axis(axis: object, title: str, ylabel: str = "") -> None:
    axis.set_title(title, fontsize=11, fontweight="bold")
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def create_figure(
    samples: Sequence[Sample],
    source: Path,
    output: Path,
) -> Dict[str, float]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except (ImportError, ModuleNotFoundError) as error:
        raise ComparisonError("python3-matplotlib is required") from error

    active = active_samples(samples)
    spans = active_spans(samples)
    times = [sample.time_s for sample in samples]

    qdd_delta_values = [
        sample.values["tangent_escape_delta_qdd_norm"] for sample in active
    ]
    tcp_delta_values = [
        sample.values["tangent_escape_delta_tcp_accel_norm_m_s2"]
        for sample in active
    ]
    cp_delta_values = [
        sample.values["tangent_escape_delta_cp_accel_norm_m_s2"]
        for sample in active
    ]
    cp_tangent_values = [
        abs(sample.values["tangent_escape_delta_cp_accel_dot_tangent_m_s2"])
        for sample in active
    ]
    cp_normal_values = [
        abs(sample.values["tangent_escape_delta_cp_accel_dot_normal_m_s2"])
        for sample in active
    ]
    qdd_stats = stats(qdd_delta_values)
    tcp_stats = stats(tcp_delta_values)
    cp_stats = stats(cp_delta_values)
    tangent_stats = stats(cp_tangent_values)
    normal_stats = stats(cp_normal_values)

    peak = max(
        active,
        key=lambda sample: sample.values[
            "tangent_escape_delta_cp_accel_norm_m_s2"
        ],
    )
    peak_qdd_angle = angle_degrees(
        vector(peak, QDD_WITH_FIELDS),
        vector(peak, QDD_WITHOUT_FIELDS),
    )
    peak_cp_angle = angle_degrees(
        vector(peak, CP_WITH_FIELDS),
        vector(peak, CP_WITHOUT_FIELDS),
    )

    figure = pyplot.figure(figsize=(17, 14), constrained_layout=True)
    grid = figure.add_gridspec(3, 2)
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, 0]),
    ]
    vector_axis = figure.add_subplot(grid[2, 1], projection="3d")

    axis = axes[0]
    axis.plot(
        times,
        [sample.values["tangent_escape_qdd_without_norm"] for sample in samples],
        color="#6B7280",
        linestyle="--",
        linewidth=1.2,
        label="Escape OFF",
    )
    axis.plot(
        times,
        [sample.values["tangent_escape_qdd_with_norm"] for sample in samples],
        color="#2563EB",
        linewidth=1.2,
        label="Escape ON",
    )
    delta_axis = axis.twinx()
    delta_axis.plot(
        times,
        [sample.values["tangent_escape_delta_qdd_norm"] for sample in samples],
        color="#DC2626",
        linewidth=1.0,
        label="ON - OFF",
    )
    delta_axis.set_ylabel("Difference (rad/s^2)", color="#991B1B")
    style_axis(axis, "Joint acceleration norm at the same state", "qdd norm (rad/s^2)")
    shade(axis, spans)
    handles, labels = axis.get_legend_handles_labels()
    other_handles, other_labels = delta_axis.get_legend_handles_labels()
    axis.legend(handles + other_handles, labels + other_labels, loc="best")

    axis = axes[1]
    colors = ["#2563EB", "#DC2626", "#16A34A", "#9333EA", "#EA580C", "#0891B2"]
    for joint, (field, color) in enumerate(zip(QDD_DELTA_FIELDS, colors), start=1):
        axis.plot(
            times,
            [sample.values[field] for sample in samples],
            color=color,
            linewidth=1.0,
            label=f"joint {joint}",
        )
    axis.axhline(0.0, color="#111827", linewidth=0.8)
    style_axis(axis, "Escape-added acceleration by joint", "delta qdd (rad/s^2)")
    shade(axis, spans)
    axis.legend(loc="best", ncol=3, fontsize=8)

    axis = axes[2]
    axis.plot(
        times,
        [
            sample.values["tangent_escape_delta_tcp_accel_norm_m_s2"]
            for sample in samples
        ],
        color="#2563EB",
        linewidth=1.2,
        label="TCP difference",
    )
    axis.plot(
        times,
        [
            sample.values["tangent_escape_delta_cp_accel_norm_m_s2"]
            for sample in samples
        ],
        color="#DC2626",
        linewidth=1.2,
        label="sensor CP difference",
    )
    activation_axis = axis.twinx()
    activation_axis.plot(
        times,
        [sample.values[ACTIVATION_FIELD] for sample in samples],
        color="#16A34A",
        linewidth=1.0,
        alpha=0.8,
        label="activation",
    )
    activation_axis.set_ylim(-0.05, 1.05)
    activation_axis.set_ylabel("Activation")
    style_axis(axis, "Task-space effect of Tangent Escape", "acceleration difference (m/s^2)")
    shade(axis, spans)
    handles, labels = axis.get_legend_handles_labels()
    other_handles, other_labels = activation_axis.get_legend_handles_labels()
    axis.legend(handles + other_handles, labels + other_labels, loc="best")

    axis = axes[3]
    axis.plot(
        times,
        [
            sample.values["tangent_escape_delta_cp_accel_dot_tangent_m_s2"]
            for sample in samples
        ],
        color="#16A34A",
        linewidth=1.2,
        label="delta CP dot tangent",
    )
    axis.plot(
        times,
        [
            sample.values["tangent_escape_delta_cp_accel_dot_normal_m_s2"]
            for sample in samples
        ],
        color="#EA580C",
        linestyle="--",
        linewidth=1.2,
        label="delta CP dot normal",
    )
    axis.axhline(0.0, color="#111827", linewidth=0.8)
    style_axis(
        axis,
        "Direction of the added sensor-control-point acceleration",
        "signed projection (m/s^2)",
    )
    shade(axis, spans)
    axis.legend(loc="best")

    axis = axes[4]
    axis.plot(
        times,
        [sample.values[ACTIVATION_FIELD] for sample in samples],
        color="#16A34A",
        linewidth=1.2,
        label="activation",
    )
    if all(ORTHOGONALITY_FIELD in sample.values for sample in samples):
        axis.plot(
            times,
            [sample.values[ORTHOGONALITY_FIELD] for sample in samples],
            color="#7C3AED",
            linewidth=1.0,
            label="normal dot tangent",
        )
    axis.axhline(0.2, color="#DC2626", linestyle=":", linewidth=0.8)
    axis.axhline(-0.2, color="#DC2626", linestyle=":", linewidth=0.8)
    axis.set_ylim(-1.05, 1.05)
    metric_axis = axis.twinx()
    metric_axis.plot(
        times,
        [sample.values[METRIC_FIELD] for sample in samples],
        color="#EA580C",
        linewidth=1.0,
        alpha=0.75,
        label="effective metric",
    )
    metric_axis.set_ylabel("Metric scalar")
    style_axis(axis, "When and how strongly Escape is fused", "activation / n dot t")
    shade(axis, spans)
    handles, labels = axis.get_legend_handles_labels()
    other_handles, other_labels = metric_axis.get_legend_handles_labels()
    axis.legend(handles + other_handles, labels + other_labels, loc="best")
    axis.set_xlabel("time from first dual-solve row (s)")

    delta_vector = vector(peak, CP_DELTA_FIELDS)
    tangent = normalized(vector(peak, TANGENT_FIELDS))
    normal = normalized(vector(peak, NORMAL_FIELDS))
    tangent_component = scaled(tangent, dot(delta_vector, tangent))
    normal_component = scaled(normal, dot(delta_vector, normal))
    residual_component = subtract(
        subtract(delta_vector, tangent_component),
        normal_component,
    )
    vector_items = [
        ("total delta", delta_vector, "#DC2626"),
        ("tangent part", tangent_component, "#16A34A"),
        ("normal part", normal_component, "#EA580C"),
        ("remaining part", residual_component, "#7C3AED"),
    ]
    for label, values_, color in vector_items:
        vector_axis.quiver(
            0.0,
            0.0,
            0.0,
            values_[0],
            values_[1],
            values_[2],
            color=color,
            linewidth=2.0,
            arrow_length_ratio=0.12,
            label=label,
        )
    vector_limit = max(
        1.0e-6,
        1.2 * max(norm(values_) for _, values_, _ in vector_items),
    )
    vector_axis.set_xlim(-vector_limit, vector_limit)
    vector_axis.set_ylim(-vector_limit, vector_limit)
    vector_axis.set_zlim(-vector_limit, vector_limit)
    vector_axis.set_xlabel("base X")
    vector_axis.set_ylabel("base Y")
    vector_axis.set_zlabel("base Z")
    vector_axis.set_title(
        f"Peak Escape-added CP acceleration at t={peak.time_s:.3f} s",
        fontsize=11,
        fontweight="bold",
    )
    vector_axis.legend(loc="best", fontsize=8)

    active_duration = sum(end - start for start, end in spans)
    tangent_ratio = (
        tangent_stats.mean / normal_stats.mean
        if normal_stats.mean > 1.0e-12
        else math.inf
    )
    subtitle = (
        f"{source.name} | active rows {len(active)}/{len(samples)}, "
        f"active duration {active_duration:.3f} s | "
        f"max delta qdd {qdd_stats.maximum:.5g} rad/s^2, "
        f"max delta CP {cp_stats.maximum:.5g} m/s^2 | "
        f"peak direction change qdd {peak_qdd_angle:.4g} deg, "
        f"CP {peak_cp_angle:.4g} deg"
    )
    figure.suptitle(
        "Tangent Escape RMP: exact same-state ON versus OFF\n" + subtitle,
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output, dpi=160, bbox_inches="tight")
    pyplot.close(figure)

    return {
        "active_rows": float(len(active)),
        "total_rows": float(len(samples)),
        "active_duration_s": active_duration,
        "qdd_max": qdd_stats.maximum,
        "qdd_mean": qdd_stats.mean,
        "qdd_p95": qdd_stats.p95,
        "tcp_max": tcp_stats.maximum,
        "tcp_mean": tcp_stats.mean,
        "tcp_p95": tcp_stats.p95,
        "cp_max": cp_stats.maximum,
        "cp_mean": cp_stats.mean,
        "cp_p95": cp_stats.p95,
        "cp_tangent_max": tangent_stats.maximum,
        "cp_tangent_mean": tangent_stats.mean,
        "cp_tangent_p95": tangent_stats.p95,
        "cp_normal_max": normal_stats.maximum,
        "cp_normal_mean": normal_stats.mean,
        "cp_normal_p95": normal_stats.p95,
        "tangent_to_normal_mean_ratio": tangent_ratio,
        "peak_time_s": peak.time_s,
        "peak_qdd_angle_deg": peak_qdd_angle,
        "peak_cp_angle_deg": peak_cp_angle,
    }


def create_html(
    source: Path,
    png_path: Path,
    html_path: Path,
    summary: Dict[str, float],
) -> None:
    image_data = base64.b64encode(png_path.read_bytes()).decode("ascii")

    def row(label: str, prefix: str, unit: str) -> str:
        return (
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{summary[prefix + '_max']:.6g} {unit}</td>"
            f"<td>{summary[prefix + '_mean']:.6g} {unit}</td>"
            f"<td>{summary[prefix + '_p95']:.6g} {unit}</td>"
            "</tr>"
        )

    ratio = summary["tangent_to_normal_mean_ratio"]
    ratio_text = f"{ratio:.2f}x" if math.isfinite(ratio) else "infinite"
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tangent Escape ON versus OFF</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2933; background: #f5f7f8; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .source {{ color: #59636e; overflow-wrap: anywhere; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 20px 0; }}
    .metric {{ background: white; border: 1px solid #d8dee4; border-radius: 6px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 23px; margin-top: 5px; }}
    figure {{ margin: 0; background: white; border: 1px solid #d8dee4; padding: 12px; }}
    img {{ width: 100%; height: auto; display: block; }}
    section {{ background: white; border-top: 1px solid #d8dee4; margin-top: 18px; padding: 18px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e2e7eb; padding: 9px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    code {{ overflow-wrap: anywhere; }}
    @media (max-width: 800px) {{ .metrics {{ grid-template-columns: 1fr 1fr; }} main {{ padding: 14px; }} }}
  </style>
</head>
<body>
<main>
  <h1>Tangent Escape RMP: same-state ON versus OFF</h1>
  <div class="source"><code>{html.escape(str(source.resolve()))}</code></div>
  <div class="metrics">
    <div class="metric">Active dual-solve rows<strong>{int(summary['active_rows'])}/{int(summary['total_rows'])}</strong></div>
    <div class="metric">Maximum delta qdd<strong>{summary['qdd_max']:.5g} rad/s^2</strong></div>
    <div class="metric">Maximum delta CP accel<strong>{summary['cp_max']:.5g} m/s^2</strong></div>
    <div class="metric">Mean tangent/normal effect<strong>{ratio_text}</strong></div>
  </div>
  <figure>
    <img src="data:image/png;base64,{image_data}" alt="Tangent Escape exact same-state ON OFF comparison">
  </figure>
  <section>
    <h2>Active-row difference statistics</h2>
    <table>
      <thead><tr><th>Quantity</th><th>Maximum</th><th>Mean</th><th>P95</th></tr></thead>
      <tbody>
        {row('Joint acceleration difference', 'qdd', 'rad/s^2')}
        {row('TCP acceleration difference', 'tcp', 'm/s^2')}
        {row('Sensor CP acceleration difference', 'cp', 'm/s^2')}
        {row('Absolute CP tangent projection', 'cp_tangent', 'm/s^2')}
        {row('Absolute CP normal projection', 'cp_normal', 'm/s^2')}
      </tbody>
    </table>
  </section>
  <section>
    <h2>How to read this report</h2>
    <p>The controller evaluates Escape ON and Escape OFF at exactly the same
    q, qdot, goal, and obstacle snapshot. Their difference isolates the
    instantaneous contribution of Tangent Escape without trajectory drift.</p>
    <p>This report does not show the eventual path difference after integrating
    both controllers. That requires two synchronized closed-loop runs with
    identical goal and sensor inputs.</p>
  </section>
</main>
</body>
</html>
"""
    html_path.write_text(document, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        source = args.csv
        if source is None:
            source = latest_compatible_csv(
                Path.home() / "ros2_ws" / "log" / "rmpflow_trace"
            )
        source = source.expanduser().resolve()
        samples = load_samples(source)
        output_directory = (
            args.output_directory.expanduser().resolve()
            if args.output_directory
            else source.parent
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        stem = f"{source.stem}_tangent_escape_on_off"
        png_path = output_directory / f"{stem}.png"
        html_path = output_directory / f"{stem}.html"
        if not args.overwrite:
            existing = [path for path in (png_path, html_path) if path.exists()]
            if existing:
                raise ComparisonError(
                    "Output already exists; pass --overwrite: "
                    + ", ".join(str(path) for path in existing)
                )
        summary = create_figure(samples, source, png_path)
        create_html(source, png_path, html_path, summary)
    except ComparisonError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Input CSV: {source}")
    print(
        "Active rows: "
        f"{int(summary['active_rows'])}/{int(summary['total_rows'])}"
    )
    print(
        "Maximum same-state difference: "
        f"qdd={summary['qdd_max']:.6g} rad/s^2, "
        f"TCP={summary['tcp_max']:.6g} m/s^2, "
        f"CP={summary['cp_max']:.6g} m/s^2"
    )
    print(f"Saved PNG: {png_path}")
    print(f"Saved HTML: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

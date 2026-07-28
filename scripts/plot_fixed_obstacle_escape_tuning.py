#!/usr/bin/env python3
"""Aggregate valid fixed-obstacle Escape ON/OFF benchmark repetitions."""

import argparse
import html
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Config = Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        default="~/ros2_ws/log/tangent_escape_fixed_benchmark",
    )
    parser.add_argument("--pattern", default="*settled*.json")
    parser.add_argument("--selected-gain", type=float, default=0.30)
    parser.add_argument("--selected-metric", type=float, default=25000.0)
    parser.add_argument("--selected-floor", type=float, default=0.20)
    parser.add_argument("--output")
    return parser.parse_args()


def number_from_name(
    name: str,
    pattern: str,
    scale: float,
    default: Optional[float] = None,
) -> float:
    match = re.search(pattern, name)
    if match:
        return float(match.group(1)) * scale
    if default is not None:
        return default
    raise ValueError(f"Cannot parse configuration from {name}: {pattern}")


def parse_config(path: Path) -> Config:
    name = path.stem.lower()
    gain = number_from_name(name, r"gain(\d{3})", 0.01, 0.30)
    metric = number_from_name(name, r"metric(\d+)k", 1000.0)
    floor = number_from_name(name, r"floor(\d{3})", 0.01)
    return gain, metric, floor


def inferred_invalid_reasons(data: Dict[str, object]) -> List[str]:
    reasons = list(data.get("invalid_reasons", []))
    if not data.get("start_pose_reached", False):
        reasons.append("start pose did not settle")
    checks = (
        ("on_path_length_m", "Escape ON controller did not move"),
        ("off_path_length_m", "Escape OFF controller did not move"),
        (
            "maximum_on_joint_velocity_norm_rad_s",
            "Escape ON joint-state stream did not show motion",
        ),
        (
            "maximum_off_joint_velocity_norm_rad_s",
            "Escape OFF joint-state stream did not show motion",
        ),
    )
    for field, reason in checks:
        if float(data.get(field, 0.0)) < 0.05:
            reasons.append(reason)
    return sorted(set(reasons))


def finite_difference(
    data: Dict[str, object],
    on_field: str,
    off_field: str,
) -> Optional[float]:
    on_value = data.get(on_field)
    off_value = data.get(off_field)
    if on_value is None or off_value is None:
        return None
    difference = float(on_value) - float(off_value)
    return difference if math.isfinite(difference) else None


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.median(finite) if finite else None


def format_value(value: Optional[float], scale: float, digits: int) -> str:
    if value is None:
        return "n/a"
    return f"{value * scale:.{digits}f}"


def config_name(config: Config) -> str:
    gain, metric, floor = config
    return f"gain={gain:.2f}, metric={metric:,.0f}, floor={floor:.2f}"


def delta_chart(
    rows: Sequence[Dict[str, object]],
    field: str,
    title: str,
) -> str:
    width = 900
    row_height = 46
    margin_left = 270
    margin_right = 90
    margin_top = 38
    height = margin_top + row_height * len(rows) + 28
    values = [float(row[field]) for row in rows if row[field] is not None]
    limit = max([abs(value) for value in values] + [0.01]) * 1.15
    plot_width = width - margin_left - margin_right

    def sx(value: float) -> float:
        return margin_left + plot_width * (value + limit) / (2.0 * limit)

    zero_x = sx(0.0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">',
        f'<line class="zero" x1="{zero_x:.1f}" y1="{margin_top - 18}" '
        f'x2="{zero_x:.1f}" y2="{height - 20}"/>',
        f'<text x="{margin_left}" y="18">{-1000.0 * limit:.0f} ms</text>',
        f'<text x="{zero_x - 18}" y="18">0</text>',
        f'<text x="{width - margin_right - 38}" y="18">'
        f'{1000.0 * limit:.0f} ms</text>',
    ]
    for index, row in enumerate(rows):
        y = margin_top + index * row_height
        value = row[field]
        parts.append(
            f'<text class="row-label" x="8" y="{y + 18}">'
            f'{html.escape(str(row["configuration"]))}</text>'
        )
        if value is None:
            parts.append(f'<text x="{zero_x + 8}" y="{y + 18}">n/a</text>')
            continue
        value = float(value)
        value_x = sx(value)
        x = min(zero_x, value_x)
        bar_width = max(abs(value_x - zero_x), 1.0)
        color = "#16856b" if value <= 0.0 else "#c94f44"
        parts.append(
            f'<rect x="{x:.1f}" y="{y + 5}" width="{bar_width:.1f}" '
            f'height="18" fill="{color}"/>'
        )
        label_x = value_x - 58 if value < 0.0 else value_x + 6
        parts.append(
            f'<text x="{label_x:.1f}" y="{y + 19}">{1000.0 * value:+.1f} ms</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def load_trials(
    directory: Path,
    pattern: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    valid: List[Dict[str, object]] = []
    invalid: List[Dict[str, object]] = []
    for path in sorted(directory.glob(pattern)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            config = parse_config(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            invalid.append({"path": str(path), "reasons": [str(error)]})
            continue
        reasons = inferred_invalid_reasons(data)
        trial = {
            "path": str(path),
            "config": config,
            "data": data,
            "reach_delta_s": finite_difference(
                data, "on_reach_time_s", "off_reach_time_s"
            ),
            "dwell_delta_s": finite_difference(
                data,
                "on_local_minimum_dwell_s",
                "off_local_minimum_dwell_s",
            ),
        }
        if reasons:
            trial["reasons"] = reasons
            invalid.append(trial)
        else:
            valid.append(trial)
    return valid, invalid


def aggregate(valid: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Config, List[Dict[str, object]]] = defaultdict(list)
    for trial in valid:
        grouped[trial["config"]].append(trial)
    rows: List[Dict[str, object]] = []
    for config, trials in sorted(grouped.items()):
        data_rows = [trial["data"] for trial in trials]
        rows.append(
            {
                "config": config,
                "configuration": config_name(config),
                "repetitions": len(trials),
                "reach_delta_s": median(
                    trial["reach_delta_s"] for trial in trials
                ),
                "dwell_delta_s": median(
                    trial["dwell_delta_s"] for trial in trials
                ),
                "minimum_tof_range_m": min(
                    float(data["minimum_tof_range_m"]) for data in data_rows
                ),
                "maximum_scalar_accel_m_s2": max(
                    float(data["maximum_scalar_accel_m_s2"])
                    for data in data_rows
                ),
                "maximum_risk_accel_m_s2": max(
                    float(data["maximum_risk_accel_m_s2"])
                    for data in data_rows
                ),
                "maximum_stuck_accel_m_s2": max(
                    float(data["maximum_stuck_accel_m_s2"])
                    for data in data_rows
                ),
                "maximum_on_off_separation_m": max(
                    float(data["maximum_on_off_separation_m"])
                    for data in data_rows
                ),
            }
        )
    return rows


def write_report(
    rows: Sequence[Dict[str, object]],
    valid: Sequence[Dict[str, object]],
    invalid: Sequence[Dict[str, object]],
    selected: Config,
    output: Path,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    table_rows = []
    for row in rows:
        selected_class = " selected" if row["config"] == selected else ""
        table_rows.append(
            f'<tr class="{selected_class.strip()}">'
            f'<td>{html.escape(str(row["configuration"]))}</td>'
            f'<td>{row["repetitions"]}</td>'
            f'<td>{format_value(row["reach_delta_s"], 1000.0, 1)}</td>'
            f'<td>{format_value(row["dwell_delta_s"], 1000.0, 1)}</td>'
            f'<td>{format_value(row["minimum_tof_range_m"], 100.0, 1)}</td>'
            f'<td>{format_value(row["maximum_scalar_accel_m_s2"], 1.0, 3)}</td>'
            f'<td>{format_value(row["maximum_stuck_accel_m_s2"], 1.0, 3)}</td>'
            f'<td>{format_value(row["maximum_on_off_separation_m"], 100.0, 1)}</td>'
            "</tr>"
        )
    invalid_rows = "".join(
        "<tr>"
        f'<td>{html.escape(Path(str(trial["path"])).name)}</td>'
        f'<td>{html.escape("; ".join(trial.get("reasons", [])))}</td>'
        "</tr>"
        for trial in invalid
    )
    selected_row = next((row for row in rows if row["config"] == selected), None)
    if selected_row is None:
        selected_text = "Selected configuration has no valid repetitions."
    else:
        selected_text = (
            f'{config_name(selected)}: median ON-OFF reach delta '
            f'{format_value(selected_row["reach_delta_s"], 1000.0, 1)} ms, '
            f'local-minimum dwell delta '
            f'{format_value(selected_row["dwell_delta_s"], 1000.0, 1)} ms.'
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tangent Escape tuning comparison</title>
<style>
body {{ margin: 0; background: #f2f4f5; color: #20272c;
  font: 14px/1.45 Arial, sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 24px; margin: 0 0 8px; }}
h2 {{ font-size: 17px; margin: 0 0 12px; }}
p {{ margin: 6px 0 16px; }}
section {{ background: #fff; border: 1px solid #d6dcdf; border-radius: 6px;
  padding: 18px; margin-bottom: 16px; overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; min-width: 900px; }}
th, td {{ border-bottom: 1px solid #e3e7e9; padding: 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: #45515a; }}
tr.selected {{ background: #e7f4ef; font-weight: 700; }}
svg {{ display: block; width: 100%; min-width: 800px; background: #fbfcfc; }}
.zero {{ stroke: #6d7880; stroke-width: 1; }}
text {{ fill: #4d5961; font-size: 11px; }}
.row-label {{ font-size: 12px; }}
.note {{ border-left: 4px solid #16856b; padding-left: 12px; }}
</style></head><body><main>
<h1>Fixed-obstacle Tangent Escape tuning</h1>
<p>Valid trials: {len(valid)}. Excluded trials: {len(invalid)}.
Negative ON-OFF deltas mean Escape ON was faster or spent less time stopped.</p>
<section><h2>Selected provisional setting</h2>
<p class="note">{html.escape(selected_text)}</p></section>
<section><h2>Configuration summary</h2>
<table><thead><tr>
<th>Configuration</th><th>N</th><th>Reach delta (ms)</th>
<th>Stopped dwell delta (ms)</th><th>Min ToF (cm)</th>
<th>Max scalar accel (m/s2)</th><th>Max stuck term (m/s2)</th>
<th>Max ON/OFF separation (cm)</th>
</tr></thead><tbody>{''.join(table_rows)}</tbody></table></section>
<section><h2>Goal reach delta: ON minus OFF</h2>
{delta_chart(rows, "reach_delta_s", "Goal reach delta")}</section>
<section><h2>Stopped dwell delta: ON minus OFF</h2>
{delta_chart(rows, "dwell_delta_s", "Stopped dwell delta")}</section>
<section><h2>Excluded trials</h2>
<table><thead><tr><th>File</th><th>Reason</th></tr></thead>
<tbody>{invalid_rows or '<tr><td colspan="2">None</td></tr>'}</tbody></table>
</section></main></body></html>"""
    output.write_text(document, encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "selected": {
                    "gain": selected[0],
                    "metric": selected[1],
                    "floor": selected[2],
                },
                "valid_trials": len(valid),
                "invalid_trials": len(invalid),
                "configurations": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def main() -> int:
    args = parse_args()
    directory = Path(args.directory).expanduser()
    valid, invalid = load_trials(directory, args.pattern)
    rows = aggregate(valid)
    if not rows:
        raise RuntimeError(f"No valid benchmark JSON files found in {directory}")
    selected = (
        args.selected_gain,
        args.selected_metric,
        args.selected_floor,
    )
    output = (
        Path(args.output).expanduser()
        if args.output
        else directory
        / f"tangent_escape_tuning_summary_{datetime.now():%Y%m%d_%H%M%S}.html"
    )
    report = write_report(rows, valid, invalid, selected, output)
    print(f"Valid trials: {len(valid)}")
    print(f"Excluded trials: {len(invalid)}")
    print(f"Saved report: {report}")
    print(f"Saved data: {report.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

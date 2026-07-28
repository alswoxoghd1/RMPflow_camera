#!/usr/bin/env python3

import argparse
import base64
import html
import json
import math
from pathlib import Path


SCENARIO_SPECS = (
    (
        "Incident",
        "Problem run",
        "rmpflow_rate_benchmark/rmp_control_rate_200Hz_20260722_153930.json",
        "#C43C39",
    ),
    (
        "Pre-cleanup",
        "Controlled A/B",
        "rmpflow_rate_benchmark/pre_runtime_cleanup/"
        "rmp_control_rate_200Hz_20260723_091219.json",
        "#D98D21",
    ),
    (
        "Post-cleanup",
        "Controlled A/B",
        "rmpflow_rate_benchmark/post_dead_graph_cleanup/"
        "rmp_control_rate_200Hz_20260723_092803.json",
        "#16857A",
    ),
    (
        "Final fake OFF",
        "Final validation",
        "rmpflow_trace/escape_ab_20260723_171816/"
        "rmp_control_rate_200Hz_20260723_172644.json",
        "#2D6FA3",
    ),
    (
        "Final fake ON",
        "Final validation",
        "rmpflow_trace/escape_ab_20260723_171816/"
        "rmp_control_rate_200Hz_20260723_172733.json",
        "#4B8F3A",
    ),
    (
        "Final real OFF",
        "Final validation",
        "rmpflow_rate_benchmark/rmp_control_rate_200Hz_20260723_170449.json",
        "#41464D",
    ),
)


def load_scenarios(log_root: Path) -> list[dict]:
    scenarios = []
    for short_label, group, relative_path, color in SCENARIO_SPECS:
        path = log_root / relative_path
        with path.open("r", encoding="utf-8") as stream:
            values = json.load(stream)
        values["_short_label"] = short_label
        values["_group"] = group
        values["_path"] = path
        values["_color"] = color
        scenarios.append(values)
    return scenarios


def finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def annotate_bars(axis, bars, values, fmt: str, na_indices: set[int] | None = None) -> None:
    na_indices = na_indices or set()
    for index, (bar, value) in enumerate(zip(bars, values)):
        if index in na_indices:
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                2.0,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#41464D",
                fontweight="bold",
            )
            continue
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            format(value, fmt),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#25282C",
        )


def style_axis(axis, title: str) -> None:
    axis.set_title(title, fontsize=12, fontweight="bold", pad=10, color="#202328")
    axis.grid(axis="y", color="#D9DEE3", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#AEB6BE")
    axis.spines["bottom"].set_color("#AEB6BE")


def create_figure(scenarios: list[dict], png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [item["_short_label"] for item in scenarios]
    colors = [item["_color"] for item in scenarios]
    x = np.arange(len(scenarios))

    cpu_values = [
        finite_number(item.get("process_cpu_percent")) for item in scenarios
    ]
    cpu_plot = [value if value is not None else 0.0 for value in cpu_values]
    rate_values = [float(item["achieved_rate_hz"]) for item in scenarios]
    work_mean = [float(item["work_mean_ms"]) for item in scenarios]
    work_max = [float(item["work_max_ms"]) for item in scenarios]
    miss_values = [float(item["deadline_miss_percent"]) for item in scenarios]
    variability = [
        max(maximum - mean, 1e-4) for maximum, mean in zip(work_max, work_mean)
    ]
    pre_solve = [float(item["pre_solve_mean_ms"]) for item in scenarios]
    solve = [float(item["solve_mean_ms"]) for item in scenarios]
    post_solve = [float(item["post_solve_mean_ms"]) for item in scenarios]
    component_totals = [
        pre + middle + post
        for pre, middle, post in zip(pre_solve, solve, post_solve)
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": "#30343A",
            "xtick.color": "#4C535A",
            "ytick.color": "#4C535A",
            "figure.facecolor": "#F7F8FA",
            "axes.facecolor": "#FFFFFF",
        }
    )

    figure, axes = plt.subplots(3, 2, figsize=(16, 15), constrained_layout=True)
    figure.suptitle(
        "RMPflow Real-Time Performance: Before vs After Optimization",
        fontsize=20,
        fontweight="bold",
        color="#1B1E22",
    )
    figure.text(
        0.5,
        0.972,
        "200 Hz target | CPU is process usage (about one fully used core at 100%)",
        ha="center",
        fontsize=10,
        color="#5A626A",
    )

    axis = axes[0, 0]
    bars = axis.bar(x, cpu_plot, color=colors, width=0.68)
    axis.axhline(100.0, color="#C43C39", linestyle="--", linewidth=1.0)
    axis.set_ylim(0.0, 112.0)
    axis.set_ylabel("Process CPU (%)")
    axis.set_xticks(x, labels, rotation=23, ha="right")
    style_axis(axis, "CPU load")
    annotate_bars(
        axis,
        bars,
        cpu_plot,
        ".1f",
        {index for index, value in enumerate(cpu_values) if value is None},
    )

    axis = axes[0, 1]
    bars = axis.bar(x, rate_values, color=colors, width=0.68)
    axis.axhline(
        200.0,
        color="#22262A",
        linestyle="--",
        linewidth=1.2,
        label="200 Hz target",
    )
    axis.set_ylim(0.0, 225.0)
    axis.set_ylabel("Achieved control rate (Hz)")
    axis.set_xticks(x, labels, rotation=23, ha="right")
    axis.legend(frameon=False, loc="lower right")
    style_axis(axis, "Effective control rate")
    annotate_bars(axis, bars, rate_values, ".1f")

    axis = axes[1, 0]
    width = 0.34
    mean_bars = axis.bar(
        x - width / 2.0,
        work_mean,
        width=width,
        color="#2D6FA3",
        label="Mean work",
    )
    max_bars = axis.bar(
        x + width / 2.0,
        work_max,
        width=width,
        color="#D98D21",
        label="Max work",
    )
    axis.axhline(
        5.0,
        color="#C43C39",
        linestyle="--",
        linewidth=1.2,
        label="5 ms budget",
    )
    axis.set_yscale("log")
    axis.set_ylim(0.1, 180.0)
    axis.set_ylabel("Controller work latency (ms, log scale)")
    axis.set_xticks(x, labels, rotation=23, ha="right")
    axis.legend(frameon=False, loc="upper right")
    style_axis(axis, "Computation latency")
    for bars, values in ((mean_bars, work_mean), (max_bars, work_max)):
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value * 1.10,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    axis = axes[1, 1]
    bars = axis.bar(x, miss_values, color=colors, width=0.68)
    axis.set_ylim(0.0, 108.0)
    axis.set_ylabel("Deadline misses (%)")
    axis.set_xticks(x, labels, rotation=23, ha="right")
    style_axis(axis, "Real-time deadline compliance")
    annotate_bars(axis, bars, miss_values, ".2f")

    axis = axes[2, 0]
    bars = axis.bar(x, variability, color=colors, width=0.68)
    axis.set_yscale("log")
    axis.set_ylim(0.1, 100.0)
    axis.set_ylabel("Max work - mean work (ms, log scale)")
    axis.set_xticks(x, labels, rotation=23, ha="right")
    style_axis(axis, "Timing variability proxy")
    for bar, value in zip(bars, variability):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * 1.10,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axis = axes[2, 1]
    pre_share = [
        100.0 * value / total for value, total in zip(pre_solve, component_totals)
    ]
    solve_share = [
        100.0 * value / total for value, total in zip(solve, component_totals)
    ]
    post_share = [
        100.0 * value / total for value, total in zip(post_solve, component_totals)
    ]
    axis.bar(x, pre_share, color="#668FAE", width=0.68, label="Pre-solve")
    axis.bar(
        x,
        solve_share,
        bottom=pre_share,
        color="#16857A",
        width=0.68,
        label="Solve",
    )
    stacked_bottom = [a + b for a, b in zip(pre_share, solve_share)]
    axis.bar(
        x,
        post_share,
        bottom=stacked_bottom,
        color="#D98D21",
        width=0.68,
        label="Post-solve",
    )
    axis.set_ylim(0.0, 112.0)
    axis.set_ylabel("Share of measured work (%)")
    axis.set_xticks(x, labels, rotation=23, ha="right")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    style_axis(axis, "Where controller time was spent")
    for index, total in enumerate(component_totals):
        axis.text(
            index,
            102.0,
            f"{total:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def percentage_change(before: float, after: float) -> float:
    return 100.0 * (after - before) / before


def format_value(value: object, suffix: str = "", digits: int = 3) -> str:
    number = finite_number(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}{suffix}"


def create_html(scenarios: list[dict], png_path: Path, html_path: Path) -> None:
    image_data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    pre_cleanup = scenarios[1]
    post_cleanup = scenarios[2]
    final_fake_off = scenarios[3]
    final_fake_on = scenarios[4]
    incident = scenarios[0]
    final_real = scenarios[5]

    cpu_change = percentage_change(
        float(pre_cleanup["process_cpu_percent"]),
        float(post_cleanup["process_cpu_percent"]),
    )
    work_change = percentage_change(
        float(pre_cleanup["work_mean_ms"]),
        float(post_cleanup["work_mean_ms"]),
    )
    escape_overhead = (
        float(final_fake_on["work_mean_ms"]) -
        float(final_fake_off["work_mean_ms"])
    )

    rows = []
    for item in scenarios:
        rows.append(
            "<tr>"
            f"<td><span class='dot' style='background:{item['_color']}'></span>"
            f"{html.escape(item['_short_label'])}</td>"
            f"<td>{html.escape(item['_group'])}</td>"
            f"<td>{format_value(item.get('process_cpu_percent'), '%', 1)}</td>"
            f"<td>{format_value(item.get('achieved_rate_hz'), ' Hz', 2)}</td>"
            f"<td>{format_value(item.get('work_mean_ms'), ' ms')}</td>"
            f"<td>{format_value(item.get('work_max_ms'), ' ms')}</td>"
            f"<td>{format_value(item.get('deadline_miss_percent'), '%', 2)}</td>"
            "</tr>"
        )

    source_items = "\n".join(
        f"<li><code>{html.escape(str(item['_path']))}</code></li>"
        for item in scenarios
    )

    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RMPflow 실시간 성능 개선 보고서</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2328;
      --muted: #5b6570;
      --line: #d9dee3;
      --paper: #ffffff;
      --band: #f4f6f8;
      --accent: #16857a;
      --danger: #c43c39;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: Arial, "Noto Sans KR", sans-serif;
      line-height: 1.55;
    }}
    header {{
      padding: 44px max(24px, calc((100vw - 1180px) / 2));
      border-bottom: 1px solid var(--line);
      background: var(--band);
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 18px; font-size: 21px; letter-spacing: 0; }}
    p {{ margin: 8px 0; }}
    .subtitle {{ color: var(--muted); max-width: 920px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 56px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 30px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 18px;
      min-height: 112px;
    }}
    .metric strong {{ display: block; font-size: 25px; margin: 4px 0; }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    figure {{ margin: 0 0 36px; }}
    figure img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 10px; color: var(--muted); font-size: 13px; }}
    section {{ margin-top: 36px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-variant-numeric: tabular-nums;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 9px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{
      text-align: left;
    }}
    th {{ background: var(--band); font-weight: 700; }}
    .dot {{
      display: inline-block;
      width: 9px;
      height: 9px;
      margin-right: 8px;
      border-radius: 50%;
    }}
    .finding {{
      border-left: 4px solid var(--accent);
      padding: 4px 0 4px 16px;
      margin: 16px 0;
    }}
    .warning {{
      border-left-color: var(--danger);
    }}
    code {{
      overflow-wrap: anywhere;
      white-space: normal;
      font-family: Consolas, monospace;
      font-size: 12px;
    }}
    li {{ margin: 5px 0; }}
    @media (max-width: 760px) {{
      header {{ padding: 28px 20px; }}
      h1 {{ font-size: 24px; }}
      main {{ padding: 24px 16px 40px; }}
      .summary {{ grid-template-columns: 1fr; }}
      .table-wrap {{ overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>RMPflow 실시간 성능 개선</h1>
    <p class="subtitle">
      불필요한 운동학, 이중 root solve 및 진단 계산을 제거하기 전후의 CPU 사용량,
      계산 레이턴스, deadline miss와 timing variability를 비교한다.
    </p>
  </header>
  <main>
    <div class="summary">
      <div class="metric">
        <span>문제 당시 유효 제어율</span>
        <strong>{float(incident['achieved_rate_hz']):.2f} → {float(final_real['achieved_rate_hz']):.2f} Hz</strong>
        <span>최종 실제 검증은 조건이 다른 deployment validation</span>
      </div>
      <div class="metric">
        <span>동일조건 cleanup A/B 평균 작업</span>
        <strong>{float(pre_cleanup['work_mean_ms']):.2f} → {float(post_cleanup['work_mean_ms']):.2f} ms</strong>
        <span>{abs(work_change):.1f}% 감소</span>
      </div>
      <div class="metric">
        <span>동일조건 cleanup A/B CPU</span>
        <strong>{float(pre_cleanup['process_cpu_percent']):.1f} → {float(post_cleanup['process_cpu_percent']):.1f}%</strong>
        <span>{abs(cpu_change):.1f}% 상대 감소</span>
      </div>
    </div>

    <figure>
      <img src="data:image/png;base64,{image_data}" alt="RMPflow CPU, latency, control rate and deadline comparison">
      <figcaption>
        Timing variability는 이전 실험에 per-cycle 표준편차가 저장되지 않아
        <code>max work - mean work</code>를 대용 지표로 사용했다. 이는 네트워크
        패킷 지터가 아니라 controller workload의 상단 변동폭이다.
      </figcaption>
    </figure>

    <section>
      <h2>측정값</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>실험</th>
              <th>용도</th>
              <th>CPU</th>
              <th>Actual rate</th>
              <th>Mean latency</th>
              <th>Max latency</th>
              <th>Deadline miss</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>해석</h2>
      <div class="finding warning">
        <strong>문제 당시:</strong> 평균 작업 시간이
        {float(incident['work_mean_ms']):.2f} ms로 5 ms 예산의
        {float(incident['work_budget_percent']):.1f}%였고, deadline miss가
        {float(incident['deadline_miss_percent']):.0f}% 발생했다.
      </div>
      <div class="finding">
        <strong>동일조건 cleanup A/B:</strong> dead graph와 상시 진단 계산을 제거해
        평균 작업은 {abs(work_change):.1f}%, CPU는 {abs(cpu_change):.1f}% 감소했다.
      </div>
      <div class="finding">
        <strong>최종 fake A/B:</strong> Escape ON의 추가 평균 작업은
        {escape_overhead:.3f} ms였으며 OFF와 ON 모두 약 200 Hz를 유지했다.
      </div>
      <div class="finding">
        <strong>최종 실제 검증:</strong> 평균 {float(final_real['work_mean_ms']):.3f} ms,
        최대 {float(final_real['work_max_ms']):.3f} ms, deadline miss 0%를 기록했다.
        이 로그에는 process CPU가 저장되지 않아 CPU 값은 N/A로 남겼다.
      </div>
    </section>

    <section>
      <h2>보고 시 주의점</h2>
      <p>
        문제 당시와 최종 실제 검증은 궤적 및 진단 설정이 동일하지 않다. 따라서
        두 값을 엄밀한 A/B 효과 크기로 사용하지 않고, 문제 재현과 최종 deployment
        validation으로 제시한다. 정량적인 최적화 효과는 동일조건
        <strong>Pre-cleanup 대 Post-cleanup</strong> 비교를 사용한다.
      </p>
    </section>

    <section>
      <h2>원본 데이터</h2>
      <ul>
        {source_items}
      </ul>
    </section>
  </main>
</body>
</html>
"""
    html_path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot before/after RMP controller timing and CPU benchmarks."
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("~/ros2_ws/log").expanduser(),
        help="Root containing rmpflow_rate_benchmark and rmpflow_trace.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "~/ros2_ws/log/rmpflow_rate_benchmark/"
            "rmp_timing_improvement_20260724"
        ).expanduser(),
        help="Output path without extension.",
    )
    args = parser.parse_args()

    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios(args.log_root.expanduser().resolve())
    png_path = output_prefix.with_suffix(".png")
    html_path = output_prefix.with_suffix(".html")
    create_figure(scenarios, png_path)
    create_html(scenarios, png_path, html_path)
    print(f"Saved PNG: {png_path}")
    print(f"Saved HTML: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

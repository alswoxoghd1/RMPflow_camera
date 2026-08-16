#!/usr/bin/env python3
"""Plot the TCP path recorded by rmp_data_recorder as a speed-colored 3D curve.

The recorder CSV carries ee_pose_x/y/z but no goal columns, so the goal is
supplied separately with --goal or --goal-pose.
"""

import argparse
import csv
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

DEFAULT_DATA_DIR = os.path.expanduser("~/ros2_ws/data/rmp_datasets")
DEFAULT_POSES_FILE = os.path.expanduser(
    "~/ros2_ws/src/RMP_Proximity-Sensor/config/experiment_goal_poses.yaml"
)
EE_COLUMNS = ("ee_pose_x", "ee_pose_y", "ee_pose_z")
TIME_COLUMN = "timestamp_unix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a recorded TCP path in 3D, colored by TCP speed.",
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="Recorder CSV. Defaults to the newest CSV in the dataset directory.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory searched when csv_path is omitted.",
    )
    parser.add_argument(
        "--format",
        choices=["html", "png", "both"],
        default="html",
        help="html writes an interactive self-contained page, png a static image.",
    )
    parser.add_argument(
        "--output",
        help="Output path. Defaults to <csv>_tcp_path_3d.<ext> next to the CSV.",
    )
    parser.add_argument(
        "--goal",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Goal position drawn as a marker.",
    )
    parser.add_argument(
        "--goal-pose",
        help="Goal pose name looked up in --poses-file instead of --goal.",
    )
    parser.add_argument(
        "--poses-file",
        default=DEFAULT_POSES_FILE,
        help="YAML file used by --goal-pose.",
    )
    parser.add_argument(
        "--goal-tolerance-m",
        type=float,
        default=0.0,
        help="Draw a tolerance sphere of this radius around the goal.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=9,
        help="Moving-average window in samples used for the speed color. 1 disables it.",
    )
    parser.add_argument(
        "--arrows",
        type=int,
        default=12,
        help="Number of direction arrows along the path. 0 disables them.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=22.0,
        help="3D view elevation in degrees.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-58.0,
        help="3D view azimuth in degrees.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output resolution.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive window instead of writing a PNG only.",
    )
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be at least 1.")
    if args.arrows < 0:
        parser.error("--arrows must be non-negative.")
    if args.goal_tolerance_m < 0.0:
        parser.error("--goal-tolerance-m must be non-negative.")
    return args


def newest_csv(directory: str) -> str:
    expanded = os.path.expanduser(directory)
    candidates = [
        os.path.join(expanded, name)
        for name in os.listdir(expanded)
        if name.endswith(".csv")
    ]
    if not candidates:
        raise RuntimeError(f"No CSV files found in {expanded}")
    return max(candidates, key=os.path.getmtime)


def read_csv_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    data_lines = []
    with open(path, newline="") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            data_lines.append(line)
    reader = csv.DictReader(data_lines)
    return list(reader.fieldnames or []), list(reader)


def as_float(text: object) -> float:
    try:
        value = float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def load_path(path: str) -> Tuple[np.ndarray, np.ndarray]:
    fieldnames, rows = read_csv_rows(path)
    missing = [name for name in EE_COLUMNS if name not in fieldnames]
    if missing:
        raise RuntimeError(
            f"{path} has no TCP pose columns {missing}. "
            "Record with publish_rmp_ee_pose:=true so /rmp_ee_pose is published."
        )

    times: List[float] = []
    points: List[List[float]] = []
    for index, row in enumerate(rows):
        xyz = [as_float(row.get(name)) for name in EE_COLUMNS]
        if any(math.isnan(value) for value in xyz):
            continue
        stamp = as_float(row.get(TIME_COLUMN))
        times.append(float(index) if math.isnan(stamp) else stamp)
        points.append(xyz)

    if len(points) < 2:
        raise RuntimeError(
            f"{path} holds {len(points)} usable TCP samples. "
            "The ee_pose columns are empty, so nothing can be drawn."
        )

    time_array = np.asarray(times, dtype=float)
    return time_array - time_array[0], np.asarray(points, dtype=float)


def load_goal(args: argparse.Namespace) -> Optional[np.ndarray]:
    if args.goal is not None:
        return np.asarray(args.goal, dtype=float)
    if not args.goal_pose:
        return None

    import yaml

    with open(os.path.expanduser(args.poses_file), "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    poses = document.get("poses", {})
    if args.goal_pose not in poses:
        available = ", ".join(sorted(str(name) for name in poses))
        raise RuntimeError(f"Unknown pose '{args.goal_pose}'. Available: {available}")
    entry = poses[args.goal_pose]
    position = entry.get("position", entry.get("position_xyz"))
    if not isinstance(position, list) or len(position) != 3:
        raise RuntimeError(f"Pose '{args.goal_pose}' has no 3-element position.")
    return np.asarray([float(value) for value in position], dtype=float)


def speed_profile(times: np.ndarray, points: np.ndarray, window: int) -> np.ndarray:
    deltas = np.diff(points, axis=0)
    step_distance = np.linalg.norm(deltas, axis=1)
    dt = np.diff(times)
    dt[dt <= 0.0] = np.nan
    segment_speed = step_distance / dt
    segment_speed = np.nan_to_num(segment_speed, nan=0.0, posinf=0.0)

    if window > 1 and segment_speed.size >= window:
        kernel = np.ones(window) / float(window)
        padded = np.pad(segment_speed, (window // 2, window // 2), mode="edge")
        segment_speed = np.convolve(padded, kernel, mode="same")[
            window // 2 : window // 2 + step_distance.size
        ]
    return segment_speed


def equal_cube_limits(
    points: np.ndarray,
    extra: Optional[np.ndarray],
) -> List[Tuple[float, float]]:
    """Axis ranges of one cube containing the data, so 1 m looks the same on every axis."""
    stacked = points if extra is None else np.vstack([points, extra.reshape(1, 3)])
    lower = stacked.min(axis=0)
    upper = stacked.max(axis=0)
    center = (lower + upper) / 2.0
    span = max(float(np.max(upper - lower)), 1e-3) * 1.15
    return [(float(center[axis] - span / 2), float(center[axis] + span / 2)) for axis in range(3)]


def set_equal_aspect(axis, points: np.ndarray, extra: Optional[np.ndarray]) -> None:
    limits = equal_cube_limits(points, extra)
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_zlim(*limits[2])
    axis.set_box_aspect((1.0, 1.0, 1.0))


def draw_sphere(axis, center: np.ndarray, radius: float) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 16)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    axis.plot_wireframe(x, y, z, color="#2ca02c", linewidth=0.4, alpha=0.35)


def summarize(
    times: np.ndarray,
    points: np.ndarray,
    speeds: np.ndarray,
    goal: Optional[np.ndarray],
) -> List[str]:
    step_distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
    path_length = float(step_distance.sum())
    straight = float(np.linalg.norm(points[-1] - points[0]))
    ratio = path_length / straight if straight > 1e-9 else float("nan")
    lines = [
        f"samples      {points.shape[0]}",
        f"duration     {times[-1] - times[0]:.2f} s",
        f"path length  {path_length:.4f} m",
        f"straight     {straight:.4f} m",
        f"detour ratio {ratio:.3f}",
        f"speed mean   {float(np.mean(speeds)):.4f} m/s",
        f"speed max    {float(np.max(speeds)):.4f} m/s",
    ]
    if goal is not None:
        lines.append(f"final error  {float(np.linalg.norm(points[-1] - goal)):.4f} m")
    return lines


def build_figure(
    args: argparse.Namespace,
    csv_path: str,
    times: np.ndarray,
    points: np.ndarray,
    speeds: np.ndarray,
    goal: Optional[np.ndarray],
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    figure = plt.figure(figsize=(15.0, 8.0))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.55, 1.0], hspace=0.28, wspace=0.2)
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_xy = figure.add_subplot(grid[0, 1])
    axis_speed = figure.add_subplot(grid[1, 1])

    segments = np.stack([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(float(np.min(speeds)), max(float(np.max(speeds)), 1e-6))
    collection = Line3DCollection(segments, cmap="viridis", norm=norm, linewidth=2.6)
    collection.set_array(speeds)
    axis_3d.add_collection3d(collection)

    axis_3d.scatter(*points[0], color="#1f77b4", s=70, marker="o", label="start", depthshade=False)
    axis_3d.scatter(*points[-1], color="#d62728", s=110, marker="*", label="end", depthshade=False)
    if goal is not None:
        axis_3d.scatter(*goal, color="#2ca02c", s=110, marker="X", label="goal", depthshade=False)
        axis_3d.plot(
            [points[-1, 0], goal[0]],
            [points[-1, 1], goal[1]],
            [points[-1, 2], goal[2]],
            color="#2ca02c",
            linestyle=":",
            linewidth=1.2,
        )
        if args.goal_tolerance_m > 0.0:
            draw_sphere(axis_3d, goal, args.goal_tolerance_m)

    if args.arrows > 0 and points.shape[0] > 2:
        stride = max(1, points.shape[0] // args.arrows)
        anchors = points[:-1:stride]
        deltas = np.diff(points, axis=0)[::stride]
        # Scale arrows to the plotted extent so they stay visible on long paths.
        arrow_length = 0.05 * max(float(np.max(points.max(axis=0) - points.min(axis=0))), 1e-3)
        axis_3d.quiver(
            anchors[:, 0],
            anchors[:, 1],
            anchors[:, 2],
            deltas[:, 0],
            deltas[:, 1],
            deltas[:, 2],
            length=arrow_length,
            normalize=True,
            color="#444444",
            linewidth=0.9,
            arrow_length_ratio=0.35,
        )

    set_equal_aspect(axis_3d, points, goal)
    axis_3d.set_xlabel("x [m]")
    axis_3d.set_ylabel("y [m]")
    axis_3d.set_zlabel("z [m]")
    axis_3d.view_init(elev=args.elev, azim=args.azim)
    axis_3d.set_title("TCP path (color = speed)")
    axis_3d.legend(loc="upper left")

    bar = figure.colorbar(collection, ax=axis_3d, shrink=0.62, pad=0.09)
    bar.set_label("speed [m/s]")

    axis_xy.plot(points[:, 0], points[:, 1], color="#1f77b4", linewidth=1.8)
    axis_xy.scatter(points[0, 0], points[0, 1], color="#1f77b4", s=45, marker="o")
    axis_xy.scatter(points[-1, 0], points[-1, 1], color="#d62728", s=70, marker="*")
    if goal is not None:
        axis_xy.scatter(goal[0], goal[1], color="#2ca02c", s=70, marker="X")
    axis_xy.set_aspect("equal", adjustable="datalim")
    axis_xy.set_xlabel("x [m]")
    axis_xy.set_ylabel("y [m]")
    axis_xy.set_title("top view (xy)")
    axis_xy.grid(alpha=0.3)

    axis_speed.plot(times[1:], speeds, color="#6a3d9a", linewidth=1.4)
    axis_speed.set_xlabel("time [s]")
    axis_speed.set_ylabel("speed [m/s]")
    axis_speed.set_title("TCP speed")
    axis_speed.grid(alpha=0.3)

    text = "\n".join(summarize(times, points, speeds, goal))
    axis_speed.text(
        1.02,
        1.0,
        text,
        transform=axis_speed.transAxes,
        family="monospace",
        fontsize=8.5,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "#f4f4f4", "edgecolor": "#bbbbbb"},
    )

    figure.suptitle(os.path.basename(csv_path), fontsize=12)
    figure.subplots_adjust(right=0.82)
    return figure


def build_plotly_figure(
    args: argparse.Namespace,
    csv_path: str,
    times: np.ndarray,
    points: np.ndarray,
    speeds: np.ndarray,
    goal: Optional[np.ndarray],
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # speeds is one shorter than points; repeat the first sample so every
    # vertex carries a color and a hover value.
    vertex_speed = np.concatenate([speeds[:1], speeds])

    figure = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.55, 0.45],
        row_heights=[0.5, 0.5],
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}],
        ],
        subplot_titles=("", "top view (xy)", "TCP speed"),
        horizontal_spacing=0.11,
        vertical_spacing=0.12,
    )

    hover = (
        "t=%{customdata[0]:.2f}s<br>"
        "x=%{x:.4f} y=%{y:.4f} z=%{z:.4f}<br>"
        "speed=%{customdata[1]:.4f} m/s<extra></extra>"
    )
    figure.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="lines",
            name="TCP path",
            line={
                "width": 6,
                "color": vertex_speed,
                "colorscale": "Viridis",
                "colorbar": {"title": "speed [m/s]", "x": 0.50, "len": 0.7, "thickness": 14},
            },
            customdata=np.stack([times, vertex_speed], axis=1),
            hovertemplate=hover,
        ),
        row=1,
        col=1,
    )
    for label, index, color, symbol in (
        ("start", 0, "#1f77b4", "circle"),
        ("end", -1, "#d62728", "diamond"),
    ):
        figure.add_trace(
            go.Scatter3d(
                x=[points[index, 0]],
                y=[points[index, 1]],
                z=[points[index, 2]],
                mode="markers",
                name=label,
                marker={"size": 6, "color": color, "symbol": symbol},
                hovertemplate=f"{label}<br>x=%{{x:.4f}} y=%{{y:.4f}} z=%{{z:.4f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if goal is not None:
        figure.add_trace(
            go.Scatter3d(
                x=[goal[0]],
                y=[goal[1]],
                z=[goal[2]],
                mode="markers",
                name="goal",
                marker={"size": 7, "color": "#2ca02c", "symbol": "x"},
                hovertemplate="goal<br>x=%{x:.4f} y=%{y:.4f} z=%{z:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    figure.add_trace(
        go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode="lines",
            name="xy",
            line={"color": "#1f77b4", "width": 2},
            showlegend=False,
            hovertemplate="x=%{x:.4f} y=%{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    if goal is not None:
        figure.add_trace(
            go.Scatter(
                x=[goal[0]],
                y=[goal[1]],
                mode="markers",
                marker={"size": 10, "color": "#2ca02c", "symbol": "x"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )

    figure.add_trace(
        go.Scatter(
            x=times[1:],
            y=speeds,
            mode="lines",
            name="speed",
            line={"color": "#6a3d9a", "width": 1.6},
            showlegend=False,
            hovertemplate="t=%{x:.2f}s<br>speed=%{y:.4f} m/s<extra></extra>",
        ),
        row=2,
        col=2,
    )

    stats = "<br>".join(
        line.replace(" ", "&nbsp;") for line in summarize(times, points, speeds, goal)
    )
    figure.add_annotation(
        text=f"<b>summary</b><br>{stats}",
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.09,
        showarrow=False,
        align="left",
        font={"family": "monospace", "size": 12},
        bgcolor="#f4f4f4",
        bordercolor="#bbbbbb",
        borderwidth=1,
    )

    # Equal ranges on all three axes + aspectmode="cube" keeps 1 m identical on
    # every axis and guarantees the whole path stays inside the scene box.
    limits = equal_cube_limits(points, goal)
    figure.update_scenes(
        aspectmode="cube",
        xaxis={"title": "x [m]", "range": list(limits[0])},
        yaxis={"title": "y [m]", "range": list(limits[1])},
        zaxis={"title": "z [m]", "range": list(limits[2])},
        camera={"eye": {"x": 1.6, "y": -1.8, "z": 1.0}},
    )
    figure.update_xaxes(title_text="x [m]", row=1, col=2)
    figure.update_yaxes(
        title_text="y [m]", scaleanchor="x", scaleratio=1.0, row=1, col=2
    )
    figure.update_xaxes(title_text="time [s]", row=2, col=2)
    figure.update_yaxes(title_text="speed [m/s]", row=2, col=2)
    figure.update_layout(
        title=f"TCP path — {os.path.basename(csv_path)}",
        template="plotly_white",
        height=820,
        margin={"l": 40, "r": 30, "t": 70, "b": 120},
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
    )
    return figure


def main() -> int:
    args = parse_args()

    csv_path = os.path.expanduser(args.csv_path) if args.csv_path else newest_csv(args.data_dir)
    times, points = load_path(csv_path)
    goal = load_goal(args)
    speeds = speed_profile(times, points, args.smooth)

    stem = os.path.splitext(csv_path)[0]
    want_html = args.format in {"html", "both"}
    want_png = args.format in {"png", "both"}
    if args.output and args.format == "both":
        raise RuntimeError("--output cannot be combined with --format both.")

    if want_html:
        figure = build_plotly_figure(args, csv_path, times, points, speeds, goal)
        html_path = os.path.expanduser(args.output or f"{stem}_tcp_path_3d.html")
        # include_plotlyjs=True inlines the library so the page works offline.
        figure.write_html(html_path, include_plotlyjs=True, full_html=True)
        print(f"Saved: {html_path}")
        if args.show:
            figure.show()

    if want_png:
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mpl_figure = build_figure(args, csv_path, times, points, speeds, goal)
        png_path = os.path.expanduser(args.output or f"{stem}_tcp_path_3d.png")
        mpl_figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {png_path}")
        if args.show and not want_html:
            plt.show()
        plt.close(mpl_figure)

    for line in summarize(times, points, speeds, goal):
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

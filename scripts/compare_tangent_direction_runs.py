#!/usr/bin/env python3
"""Compare two log_tg_rmp_observation.py CSVs for one fixed-goal trial each.

All ranges are sensor-derived equivalent distances, not ground-truth geometric
clearance. This script describes recorded behavior; it does not infer safety.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def number(row, field):
    try:
        value = float(row[field])
        return value if math.isfinite(value) else None
    except (KeyError, TypeError, ValueError):
        return None


def vector(row, prefix):
    result = [number(row, prefix + axis) for axis in ("x", "y", "z")]
    return result if all(value is not None for value in result) else None


def summarize(path, reach_tolerance=0.01, low_speed=0.005, max_gap=0.1):
    with Path(path).open(newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        required = {"t", "active", "tcp_x", "tcp_y", "tcp_z", "goal_distance"}
        if not required.issubset(fields):
            raise ValueError(f"{path}: expected log_tg_rmp_observation.py CSV columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty recording")
    times = [number(row, "t") for row in rows]
    if any(t is None for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"{path}: timestamps must be finite and strictly increasing")
    # Restrict outcome metrics to the first approach, ending at first arrival.
    reached_index = next((i for i, row in enumerate(rows)
                          if number(row, "goal_distance") is not None
                          and number(row, "goal_distance") <= reach_tolerance), None)
    end = len(rows) if reached_index is None else reached_index + 1
    trial = rows[:end]
    path_length = active_time = low_progress_time = gap_time = 0.0
    turn_rates = []
    owner_switches = 0
    for before, after in zip(trial, trial[1:]):
        dt = number(after, "t") - number(before, "t")
        if dt > max_gap:
            gap_time += dt
            continue
        p0, p1 = vector(before, "tcp_"), vector(after, "tcp_")
        distance = math.dist(p0, p1) if p0 is not None and p1 is not None else None
        if distance is not None:
            path_length += distance
        active = (number(before, "active") or 0.0) > 0.5
        if active:
            active_time += dt
            if distance is not None and distance / dt < low_speed:
                low_progress_time += dt
        if active and (number(after, "active") or 0.0) > 0.5:
            owner_switches += number(before, "sensor_index") != number(after, "sensor_index")
            t0, t1 = vector(before, "t_"), vector(after, "t_")
            g0, g1 = number(before, "handoff_gain"), number(after, "handoff_gain")
            if t0 and t1 and g0 is not None and g1 is not None and min(g0, g1) >= 0.9:
                norm0 = math.sqrt(sum(x * x for x in t0))
                norm1 = math.sqrt(sum(x * x for x in t1))
                if min(norm0, norm1) > 1e-8:
                    cosine = sum(a * b for a, b in zip(t0, t1)) / (norm0 * norm1)
                    turn_rates.append(math.acos(max(-1.0, min(1.0, cosine))) / dt)
    range_fields = [field for field in fields if field.startswith("d__")]
    ranges = [value for row in trial for field in range_fields
              if (value := number(row, field)) is not None and value >= 0.0]
    confidence = [value for row in trial
                  if (number(row, "active") or 0.0) > 0.5
                  and (value := number(row, "history_confidence")) is not None]
    modes = sorted({int(value) for row in rows
                    if (value := number(row, "direction_mode")) is not None})
    sorted_rates = sorted(turn_rates)
    return {
        "file": str(path),
        "modes": modes,  # 0=legacy, 1=sensor_history, 2=fixed, 3=velocity-normal simple
        "samples": len(trial),
        "trial_duration_s": number(trial[-1], "t") - times[0],
        "reached_goal": reached_index is not None,
        "time_to_goal_s": None if reached_index is None else times[reached_index] - times[0],
        "final_goal_error_m": number(trial[-1], "goal_distance"),
        "tcp_path_length_m": path_length,
        "tg_active_s": active_time,
        "tg_active_low_tcp_speed_s": low_progress_time,
        "unobserved_gap_s": gap_time,
        "minimum_equivalent_sensor_range_m": min(ranges) if ranges else None,
        "owner_switches": owner_switches,
        "world_tangent_rate_p95_rad_s": sorted_rates[min(len(sorted_rates) - 1,
            math.ceil(0.95 * len(sorted_rates)) - 1)] if sorted_rates else None,
        "mean_history_direction_confidence": statistics.mean(confidence) if confidence else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_csv", type=Path)
    parser.add_argument("baseline_csv", type=Path)
    parser.add_argument("--baseline-mode", choices=("simple", "fixed"), default="simple",
                        help="simple: velocity-normal steering; fixed: former fixed-angle baseline")
    parser.add_argument("--reach-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    if args.reach_tolerance <= 0 or not math.isfinite(args.reach_tolerance):
        parser.error("--reach-tolerance must be finite and positive")
    try:
        result = {name: summarize(path, args.reach_tolerance)
                  for name, path in (("sensor_history", args.history_csv),
                                     (args.baseline_mode, args.baseline_csv))}
        baseline_mode_id = 3 if args.baseline_mode == "simple" else 2
        for name, expected_mode in (("sensor_history", 1), (args.baseline_mode, baseline_mode_id)):
            if result[name]["modes"] != [expected_mode]:
                raise ValueError(f"{name}: expected mode {expected_mode}, "
                                 f"got {result[name]['modes']}; check recording and argument order")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

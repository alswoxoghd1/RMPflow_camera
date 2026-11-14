#!/usr/bin/env python3
"""Record the TG-RMP blockage observation to CSV.

Pairs with the overlay's "observe" variant, where the leaf is evaluated and
logged in full but contributes nothing to the root RMP.  The trajectory is the
base policy's, so the two candidate blockage references can be compared without
either of them having shaped the run they are measured on:

    dot_tcp  = o_k . normalize(goal - p_k)               the TCP goal line
    dot_nom  = o_k . normalize(J_k qdd_nominal + cdot_k) where the goal-driven
                                                         policy, with every
                                                         obstacle removed,
                                                         accelerates this point

Both are raw -- no activation shape applied -- and the nominal magnitude is kept
alongside so the dead-band can be chosen from the data.

Usage:
    ros2 launch rb10_rmpflow_rviz rb10_rmpflow_escape_overlay.launch.py \
        variants:=observe,off
    python3 log_tg_rmp_observation.py -o run.csv
"""

import argparse
import csv
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# rb10_model.hpp:118-138, in index order.
SENSORS = [
    ("tof6_1_L", "link5"), ("tof6_1_F", "link5"),
    ("tof6_1_R", "link5"), ("tof6_1_U", "link5"),
    ("tof_S", "link3_5"), ("tof_E", "link3_5"),
    ("tof_N", "link3_5"), ("tof_W", "link3_5"),
    ("tof3_1_S", "link3_5"), ("tof3_1_W", "link3_5"),
    ("tof3_1_N", "link3_5"), ("tof3_1_E", "link3_5"),
    ("tof2_1_E", "link2"), ("tof2_1_S", "link2"),
    ("tof2_1_W", "link2"), ("tof2_1_N", "link2"),
    ("tof2_E", "link2"), ("tof2_S", "link2"),
    ("tof2_W", "link2"), ("tof2_N", "link2"),
]

# tangent_escape_supervisor.cpp: fixed prefix, 4 values per sensor, then the
# I1 scalars.
FIXED = 56
STRIDE = 4
SENSOR_LAYOUTS = {
    20: SENSORS,
    12: [SENSORS[k] for k in (0, 1, 2, 3, 4, 5, 6, 7, 19, 18, 17, 16)],
}

# Scalar slots this run cares about.
SCALARS = [
    ("active", 0), ("sensor_index", 1), ("clearance", 2), ("theta", 3),
    ("alpha_free", 15),
    ("alpha_d", 42), ("active_link", 43), ("observe_only", 44),
    ("observation_valid", 48), ("observation_count", 49),
    ("alpha_block", 46), ("alpha_tg", 51),
    ("blockage", 52), ("nominal_magnitude", 53), ("metric", 10),
    # Direction diagnosis: which coefficient moved, and did the owner change.
    ("theta_rate", 4), ("score_a", 8), ("score_b", 14),
    ("score_c", 27), ("score_e", 30),
    ("stall", 33), ("stall_penalty", 41),
    # v_t and what acts on it. With tangent_damping at 0 these say whether the
    # joint Damping RMP alone is holding the tangential rate down, which is the
    # only thing left dissipating along y.
    ("tangent_rate", 7), ("desired_accel", 9),
    # Owner control point, for the path it traces.
    ("cp_x", 11), ("cp_y", 12), ("cp_z", 13),
    # The escape direction in world, so "why is it going up" is one column.
    ("t_x", 20), ("t_y", 21), ("t_z", 22),
    ("o_out_x", 17), ("o_out_y", 18), ("o_out_z", 19),
    ("push_accel", 28), ("damping_accel", 31),
    ("budget_remaining", 6), ("injected_power", 16),
    ("leaf_jacobian_norm", 38), ("root_metric_norm", 39),
    ("root_force_norm", 40), ("natural_force", 32), ("jdot_qdot", 34),
]

# I1 whole-body kinematic consequence, appended after the per-sensor block.
I1_SCALARS = [
    ("a_i1", 0), ("b_i1", 1), ("i1_count", 2), ("i1_largest_gain", 3),
]

# Closing-rate acceleration term.
ACCEL_SCALARS = [
    ("d_dot_hat", 0), ("gamma_v", 1), ("closing_term", 2), ("closing_shape", 3),
    ("gamma_m", 4), ("goal_e1", 5), ("goal_e2", 6), ("goal_from_nominal", 7),
]

# Every dipole term of the direction score, weights applied, so that
#   score_a == a_goal + a_velocity + a_risk + a_i1 - a_stall
# holds exactly and can be checked instead of assumed.
TERM_SCALARS = [
    ("a_goal", 0), ("b_goal", 1), ("a_velocity", 2), ("b_velocity", 3),
    ("a_risk", 4), ("b_risk", 5), ("a_stall", 6), ("b_stall", 7),
]

# Is the arm getting anywhere. Path length over net displacement is what tells
# travelling along an obstacle apart from circling a pocket.
PROGRESS_SCALARS = [
    ("tcp_x", 0), ("tcp_y", 1), ("tcp_z", 2), ("goal_distance", 3),
]

# The tangent basis and the goal reference in world, the same vectors RViz
# draws, so a screen observation and a log row can be checked against each
# other.
BASIS_SCALARS = [
    ("e1_x", 0), ("e1_y", 1), ("e1_z", 2),
    ("e2_x", 3), ("e2_y", 4), ("e2_z", 5),
    ("goal_plane_x", 6), ("goal_plane_y", 7), ("goal_plane_z", 8),
]

# Appended after the existing basis block; absent in older schema-9 logs.
PREDICTION_SCALARS = [
    ("predicted_risk", 0), ("predicted_risk_score", 1),
    ("predicted_risk_gradient", 2), ("predicted_min_clearance", 3),
    ("predicted_critical_sensor", 4), ("predicted_obstacle_sensor", 5),
    ("predicted_pair_count", 6), ("predicted_approaching_pair_count", 7),
    ("total_direction_score", 8),
]


HISTORY_SCALARS = [
    ("direction_mode", 0), ("history_risk", 1), ("history_risk_score", 2),
    ("history_risk_gradient", 3), ("unknown_motion_risk", 4),
    ("history_min_predicted_range", 5), ("history_confidence", 6),
    ("history_model_count", 7), ("history_unknown_count", 8),
    ("history_critical_sensor", 9), ("handoff_gain", 10),
]

STEERING_SCALARS = [
    ("steering_speed", 0), ("steering_velocity_dot", 1),
    ("steering_velocity_source", 2), ("steering_fallback", 3),
    ("steering_turn_rate", 4),
]


class Logger(Node):
    def __init__(self, topic, path):
        super().__init__("tg_rmp_observation_logger")
        self.rows = 0
        self.skipped = 0
        self.path = path
        self.file = open(path, "w", newline="")
        self.sensors = None
        self.writer = csv.writer(self.file)
        self.create_subscription(Float64MultiArray, topic, self.on_data, 50)
        self.get_logger().info(f"logging {topic} -> {path}")

    def write_header(self, sensors):
        self.sensors = sensors
        header = ["t"] + [name for name, _ in SCALARS]
        header += [name for name, _ in I1_SCALARS]
        header += [name for name, _ in ACCEL_SCALARS]
        header += [name for name, _ in TERM_SCALARS]
        header += [name for name, _ in PROGRESS_SCALARS]
        header += [name for name, _ in BASIS_SCALARS]
        for name, link in sensors:
            tag = f"{name}__{link}"
            header += [f"d__{tag}", f"dot_tcp__{tag}",
                       f"dot_nom__{tag}", f"mag_nom__{tag}"]
        header += [name for name, _ in PREDICTION_SCALARS]
        header += [name for name, _ in HISTORY_SCALARS]
        for frame, _ in sensors:
            header += [f"history_count__{frame}", f"history_age__{frame}",
                       f"history_noise__{frame}", f"history_gradient_norm__{frame}"]
        header += [name for name, _ in STEERING_SCALARS]
        self.writer.writerow(header)

    def on_data(self, msg):
        values = msg.data
        # An idle cycle publishes the single-element sentinel.
        if len(values) < FIXED or values[26] != 9.0 or not math.isfinite(values[50]):
            self.skipped += 1
            return
        count = int(values[50])
        sensors = SENSOR_LAYOUTS.get(count)
        i1_base = FIXED + STRIDE * count
        accel_base = i1_base + 4
        term_base = accel_base + 8
        progress_base = term_base + 8
        basis_base = progress_base + 4
        prediction_base = basis_base + 9
        if (sensors is None or count != values[50] or len(values) < prediction_base
                or (self.sensors is not None and self.sensors != sensors)):
            self.skipped += 1
            return
        if self.sensors is None:
            self.write_header(sensors)
        stamp = self.get_clock().now().nanoseconds * 1e-9
        row = [stamp] + [values[slot] for _, slot in SCALARS]
        row += [values[i1_base + offset] for _, offset in I1_SCALARS]
        row += [values[accel_base + offset] for _, offset in ACCEL_SCALARS]
        row += [values[term_base + offset] for _, offset in TERM_SCALARS]
        row += [values[progress_base + offset] for _, offset in PROGRESS_SCALARS]
        row += [values[basis_base + offset] for _, offset in BASIS_SCALARS]
        for k in range(count):
            base = FIXED + STRIDE * k
            row += list(values[base:base + STRIDE])
        row += [values[prediction_base + offset]
                if prediction_base + offset < len(values) else ""
                for _, offset in PREDICTION_SCALARS]
        history_base = prediction_base + len(PREDICTION_SCALARS)
        row += [values[history_base + offset]
                if history_base + offset < len(values) else ""
                for _, offset in HISTORY_SCALARS]
        history_sensor_base = history_base + len(HISTORY_SCALARS)
        row += [values[history_sensor_base + offset]
                if history_sensor_base + offset < len(values) else ""
                for offset in range(4 * count)]
        steering_base = history_sensor_base + 4 * count
        row += [values[steering_base + offset]
                if steering_base + offset < len(values) else ""
                for _, offset in STEERING_SCALARS]
        self.writer.writerow(row)
        self.rows += 1

    def close(self):
        self.file.close()
        self.get_logger().info(
            f"wrote {self.rows} rows to {self.path} "
            f"({self.skipped} idle cycles skipped)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--topic", default="/escape_on/tangent_escape_rmp_data")
    parser.add_argument("-o", "--output", default="tg_rmp_observation.csv")
    args = parser.parse_args()

    rclpy.init()
    node = Logger(args.topic, args.output)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

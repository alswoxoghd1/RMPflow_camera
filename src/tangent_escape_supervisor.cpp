#include "rb10_rmpflow_rviz/tangent_escape_supervisor.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "rb10_rmpflow_rviz/tangent_escape_ode.hpp"

namespace rb10_rmpflow_rviz
{

TangentEscapeSupervisor::TangentEscapeSupervisor(EigenRmpConfig config)
: config_(std::move(config))
{}

TangentEscapeCommand TangentEscapeSupervisor::evaluate(
  const JointVector & q,
  const KinematicsContext & context,
  const JointVector & qd,
  const Eigen::Vector3d & goal,
  const std::vector<ObstacleSphere> & obstacles,
  const JointVector & nominal_qdd,
  const Matrix6 & base_metric,
  const JointVector & base_force) const
{
  (void)q;

  Matrix6 metric = base_metric;
  JointVector force = base_force;
  TangentEscapeCommand command;
  std::vector<double> debug_data;
  std::vector<double> * debug_data_ptr =
    (config_.enable_tangent_escape_diagnostics ||
    config_.enable_tangent_escape_dual_solve_diagnostics) ?
    &debug_data : nullptr;

  evaluate_ode(
    context, qd, nominal_qdd, goal, obstacles, metric, force, command,
    debug_data_ptr);

  command.metric_contribution = metric - base_metric;
  command.force_contribution = force - base_force;
  if (debug_data_ptr != nullptr) {
    command.debug_data = std::move(debug_data);
  }
  return command;
}

void TangentEscapeSupervisor::set_previous_qdd(const JointVector & qdd) const
{
  // The direction ODE does not use the previous solve; kept so the solver
  // interface does not have to change.
  (void)qdd;
}

void TangentEscapeSupervisor::evaluate_ode(
  const KinematicsContext & context,
  const JointVector & qd,
  const JointVector & nominal_qdd,
  const Eigen::Vector3d & goal,
  const std::vector<ObstacleSphere> & obstacles,
  Matrix6 & metric,
  JointVector & force,
  TangentEscapeCommand & command,
  std::vector<double> * debug_data) const
{
  const auto & params = config_.tangent_escape;
  const auto & ode = config_.tangent_escape_ode;

  // Per-sensor clearance, using the same geometry and the same tagged-obstacle
  // rule the other models use so that a fair A/B keeps the inputs identical.
  const std::size_t point_count = std::min(
    context.control_points.size(),
    RB10Model::sensor_control_points.size());
  std::vector<double> clearances(
    point_count, std::numeric_limits<double>::infinity());
  // The rate contributed by the robot's own motion, taken from the obstacle
  // that actually sets the clearance.  d = |p - c| - r, so for a stationary
  // obstacle ddot = (p - c)/|p - c| . v exactly.
  std::vector<double> kinematic_rates(point_count, 0.0);
  for (const auto & obstacle : obstacles) {
    if (obstacle.radius <= 0.0 || !obstacle.center.allFinite()) {
      continue;
    }
    std::size_t first = 0;
    std::size_t last = point_count;
    if (obstacle.proximity_control_point_index >= 0) {
      first = static_cast<std::size_t>(obstacle.proximity_control_point_index);
      if (first >= point_count) {
        continue;
      }
      last = first + 1;
    } else if (obstacle.source_id < 0) {
      // Unlabelled far-away sphere: the controller's empty-obstacle sentinel.
      continue;
    }
    for (std::size_t index = first; index < last; ++index) {
      const Eigen::Vector3d delta =
        context.control_points[index].position - obstacle.center;
      const double center_distance = delta.norm();
      if (center_distance <= 1e-9) {
        continue;
      }
      const double clearance = center_distance -
        (context.control_points[index].radius + obstacle.radius);
      if (clearance < clearances[index]) {
        clearances[index] = clearance;
        kinematic_rates[index] =
          (delta / center_distance).dot(context.control_point_velocities[index]);
      }
    }
  }

  // Residual = everything the robot's own motion does not explain.  For a
  // stationary obstacle that is pure sensing noise, so it is low-passed and
  // dead-banded; for a moving one it is the obstacle's own approach, which is
  // the only way a distance-only sensor can see it at all.
  std::vector<double> clearance_rates(point_count, 0.0);
  const double dt = std::max(params.control_dt, 1e-4);
  const double tau = std::max(ode.clearance_rate_filter_tau, 0.0);
  const double blend = tau > dt ? dt / tau : 1.0;
  for (std::size_t index = 0; index < point_count; ++index) {
    auto & seeded = clearance_rate_state_.seeded[index];
    auto & previous = clearance_rate_state_.previous_clearance[index];
    auto & residual = clearance_rate_state_.filtered_residual[index];
    const double clearance = clearances[index];
    if (!std::isfinite(clearance)) {
      seeded = false;
      residual = 0.0;
      clearance_rates[index] = 0.0;
      continue;
    }
    double rate = kinematic_rates[index];
    if (ode.use_obstacle_motion) {
      if (seeded) {
        const double total_rate = (clearance - previous) / dt;
        residual += blend * (total_rate - kinematic_rates[index] - residual);
      } else {
        residual = 0.0;
      }
      rate += residual;
    }
    previous = clearance;
    seeded = true;
    if (std::abs(rate) < std::max(ode.clearance_rate_deadband, 0.0)) {
      rate = 0.0;
    }
    clearance_rates[index] = rate;
  }

  tangent_escape_ode::LeafParams leaf_params;
  leaf_params.enabled = params.enabled;
  leaf_params.metric_mu = ode.metric_mu;
  leaf_params.metric_length = ode.metric_length;
  leaf_params.metric_epsilon = ode.metric_epsilon;
  leaf_params.metric_rate_scale = ode.metric_rate_scale;
  leaf_params.accel_gain = ode.accel_gain;
  leaf_params.accel_length = ode.accel_length;
  leaf_params.progress_scale = ode.progress_scale;
  leaf_params.progress_recovery_time = ode.progress_recovery_time;
  leaf_params.tangent_damping = ode.tangent_damping;
  leaf_params.direction_rate = ode.direction_rate;
  leaf_params.risk_length = ode.risk_length;
  leaf_params.pocket_length = ode.pocket_length;
  leaf_params.weight_goal = ode.weight_goal;
  leaf_params.weight_continuity = ode.weight_continuity;
  leaf_params.e1_bias = ode.e1_bias;
  leaf_params.weight_risk = ode.weight_risk;
  leaf_params.weight_axis = ode.weight_axis;
  leaf_params.velocity_epsilon = ode.velocity_epsilon;
  leaf_params.accel_saturation = ode.accel_saturation;
  leaf_params.closing_gain = ode.closing_gain;
  leaf_params.closing_length = ode.closing_length;
  leaf_params.closing_epsilon = ode.closing_epsilon;
  leaf_params.closing_rate_scale = ode.closing_rate_scale;
  leaf_params.enable_progress_attenuation = ode.enable_progress_attenuation;
  leaf_params.weight_i1 = ode.weight_i1;
  leaf_params.i1_length = ode.i1_length;
  leaf_params.i1_damping = ode.i1_damping;
  leaf_params.i1_exclude_parent_link = ode.i1_exclude_parent_link;
  leaf_params.axial_preference = ode.axial_preference;
  leaf_params.weight_axial_block = ode.weight_axial_block;
  leaf_params.pocket_threshold = ode.pocket_threshold;
  leaf_params.pocket_epsilon = ode.pocket_epsilon;
  leaf_params.alpha_on_distance = ode.alpha_on_distance;
  leaf_params.alpha_off_distance = ode.alpha_off_distance;
  leaf_params.owner_hysteresis = ode.owner_hysteresis;
  leaf_params.weight_stall = ode.weight_stall;
  leaf_params.stall_time_constant = ode.stall_time_constant;
  leaf_params.stall_speed = ode.stall_speed;
  leaf_params.clearance_margin = params.clearance_margin;
  leaf_params.control_dt = params.control_dt;
  leaf_params.observe_only = ode.observe_only;
  leaf_params.block_on_dot = ode.block_on_dot;
  leaf_params.block_off_dot = ode.block_off_dot;
  leaf_params.nominal_motion_epsilon = ode.nominal_motion_epsilon;

  const double nominal_tau =
    std::max(ode.nominal_filter_tau, 0.0);
  if (!nominal_qdd_seeded_ || nominal_tau <= dt) {
    filtered_nominal_qdd_ = nominal_qdd;
    nominal_qdd_seeded_ = true;
  } else {
    filtered_nominal_qdd_ +=
      (dt / nominal_tau) * (nominal_qdd - filtered_nominal_qdd_);
  }

  const auto leaf = tangent_escape_ode::evaluate(
    ode_state_, leaf_params, context, qd, filtered_nominal_qdd_, goal,
    clearances, clearance_rates);

  // Observation mode still runs the whole leaf and still logs, it just never
  // reaches the root.  That is the point: the trajectory underneath the
  // measurement has to be the base policy's, not one this leaf shaped.
  if (leaf.active && !ode.observe_only) {
    metric += leaf.metric_contribution;
    force += leaf.force_contribution;
  }

  command.active = leaf.active;
  command.control_point_index = leaf.sensor_index;
  command.tangent = leaf.tangent;
  command.metric_scalar = leaf.metric;
  command.scalar_acceleration =
    leaf.metric > 1e-12 ? leaf.natural_force / leaf.metric : 0.0;

  if (debug_data == nullptr) {
    return;
  }
  // An inactive cycle still carries a blockage observation whenever any sensor
  // sees anything, and the approach into an obstacle is precisely the part the
  // activation comparison needs, so the record is emitted for it too.
  if (!leaf.active && !leaf.observation_valid) {
    *debug_data = std::vector<double>{0.0};
    return;
  }

  // Slots 0, 9-13, 17-26, 28, 29, 45, 47 keep the meaning schemas 6-8 give
  // them, so the controller's existing tangent/normal/acceleration arrow
  // markers render for v2 without a second code path.  The v2-specific
  // quantities live in the slots those schemas leave free.
  std::vector<double> values(tangent_escape_debug::kSize, 0.0);
  values[0] = leaf.active ? 1.0 : 0.0;
  values[1] = static_cast<double>(leaf.sensor_index);
  values[2] = leaf.clearance;
  values[3] = leaf.theta;
  values[4] = leaf.theta_rate;
  values[5] = leaf.progress;
  values[6] = leaf.progress_remaining;
  values[7] = leaf.tangent_rate;
  values[8] = leaf.score_a;
  values[9] = leaf.desired_acceleration;   // shared: scalar acceleration
  values[10] = leaf.metric;                // shared: effective metric
  values[11] = leaf.control_point.x();     // shared: control point
  values[12] = leaf.control_point.y();
  values[13] = leaf.control_point.z();
  values[14] = leaf.score_b;
  values[15] = leaf.alpha_free;
  values[16] = leaf.injected_power;
  // shared: outward normal, i.e. away from the obstacle.
  values[17] = -leaf.obstacle_direction.x();
  values[18] = -leaf.obstacle_direction.y();
  values[19] = -leaf.obstacle_direction.z();
  values[20] = leaf.tangent.x();           // shared: escape direction
  values[21] = leaf.tangent.y();
  values[22] = leaf.tangent.z();
  values[23] = leaf.desired_acceleration * leaf.tangent.x();  // shared: a vector
  values[24] = leaf.desired_acceleration * leaf.tangent.y();
  values[25] = leaf.desired_acceleration * leaf.tangent.z();
  values[26] = 9.0;                        // schema id
  values[27] = leaf.score_c;
  values[28] = leaf.push_acceleration;     // shared slot: reference drive
  values[29] = leaf.tangent_rate;          // shared slot: tangent speed
  values[30] = leaf.score_e;
  values[33] = leaf.stall;
  values[41] = leaf.stall_penalty;
  values[31] = leaf.damping_acceleration;
  values[32] = leaf.natural_force;
  values[34] = leaf.jdot_qdot;
  values[35] = leaf.jdot_theta_term;
  values[36] = leaf.jdot_omega_term;
  values[37] = leaf.tangent_normal_residual;
  values[38] = leaf.leaf_jacobian.norm();
  values[39] = leaf.metric_contribution.norm();
  values[40] = leaf.force_contribution.norm();
  values[45] = leaf.alpha_free;            // shared slot: activation ramp
  values[47] = leaf.active ? 1.0 : 0.0;    // shared slot: high-level state

  // --- activation / blockage observation -----------------------------------
  // Slots 42-44 and 48-50 are free in schemas 6-9; 46, 51-55 are left alone
  // because the controller still reads 54 as `z`.
  values[42] = leaf.alpha_distance;
  values[43] = static_cast<double>(leaf.active_link);
  values[44] = ode.observe_only ? 1.0 : 0.0;
  values[46] = leaf.alpha_block;
  values[48] = leaf.observation_valid ? 1.0 : 0.0;
  values[49] = static_cast<double>(leaf.observation_count);
  values[50] = static_cast<double>(tangent_escape_ode::kSensorCount);
  values[51] = leaf.alpha_total;
  values[52] = leaf.blockage;
  values[53] = leaf.nominal_magnitude;

  for (std::size_t k = 0; k < tangent_escape_ode::kSensorCount; ++k) {
    const std::size_t base = tangent_escape_debug::kFixed + tangent_escape_debug::kStride * k;
    values[base + 0] = leaf.sensor_clearance[k];
    values[base + 1] = leaf.sensor_dot_goal_tcp[k];
    values[base + 2] = leaf.sensor_dot_goal_nominal[k];
    values[base + 3] = leaf.sensor_nominal_magnitude[k];
  }

  // I1 scalars ride past the per-sensor block so the block keeps its offsets.
  values[tangent_escape_debug::kI1] = leaf.score_a_i1;
  values[tangent_escape_debug::kI1 + 1] = leaf.score_b_i1;
  values[tangent_escape_debug::kI1 + 2] = static_cast<double>(leaf.i1_count);
  values[tangent_escape_debug::kI1 + 3] = leaf.i1_largest_gain;
  values[tangent_escape_debug::kAccel] = leaf.clearance_rate;
  values[tangent_escape_debug::kAccel + 1] = leaf.closing_gate;
  values[tangent_escape_debug::kAccel + 2] = leaf.closing_term;
  values[tangent_escape_debug::kAccel + 3] = leaf.closing_shape;
  values[tangent_escape_debug::kAccel + 4] = leaf.metric_gate;
  values[tangent_escape_debug::kAccel + 5] = leaf.goal_e1;
  values[tangent_escape_debug::kAccel + 6] = leaf.goal_e2;
  values[tangent_escape_debug::kAccel + 7] = leaf.goal_from_nominal ? 1.0 : 0.0;
  values[tangent_escape_debug::kTerms] = leaf.score_a_goal;
  values[tangent_escape_debug::kTerms + 1] = leaf.score_b_goal;
  values[tangent_escape_debug::kTerms + 2] = leaf.score_a_velocity;
  values[tangent_escape_debug::kTerms + 3] = leaf.score_b_velocity;
  values[tangent_escape_debug::kTerms + 4] = leaf.score_a_risk;
  values[tangent_escape_debug::kTerms + 5] = leaf.score_b_risk;
  values[tangent_escape_debug::kTerms + 6] = leaf.score_a_stall;
  values[tangent_escape_debug::kTerms + 7] = leaf.score_b_stall;
  values[tangent_escape_debug::kProgress] = leaf.tcp_position.x();
  values[tangent_escape_debug::kProgress + 1] = leaf.tcp_position.y();
  values[tangent_escape_debug::kProgress + 2] = leaf.tcp_position.z();
  values[tangent_escape_debug::kProgress + 3] = leaf.goal_distance;
  for (int axis = 0; axis < 3; ++axis) {
    values[tangent_escape_debug::kBasis + axis] = leaf.basis_e1[axis];
    values[tangent_escape_debug::kBasis + 3 + axis] = leaf.basis_e2[axis];
    values[tangent_escape_debug::kBasis + 6 + axis] = leaf.goal_in_plane[axis];
  }
  *debug_data = std::move(values);
}

}  // namespace rb10_rmpflow_rviz

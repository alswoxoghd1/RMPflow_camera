#include "rb10_rmpflow_rviz/tangent_escape_supervisor.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "rb10_rmpflow_rviz/tangent_escape_ode.hpp"

namespace rb10_rmpflow_rviz
{
namespace
{
// Fixed prefix of the schema-9 diagnostic record.  Slots 0, 9-13, 17-26, 28,
// 29, 45 and 47 keep the meaning the controller's marker builder expects, so
// the tangent/normal/acceleration arrows render without a second code path.
constexpr std::size_t kTangentOdeDebugSize = 56;
}  // namespace

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
  (void)nominal_qdd;

  Matrix6 metric = base_metric;
  JointVector force = base_force;
  TangentEscapeCommand command;
  std::vector<double> debug_data;
  std::vector<double> * debug_data_ptr =
    (config_.enable_tangent_escape_diagnostics ||
    config_.enable_tangent_escape_dual_solve_diagnostics) ?
    &debug_data : nullptr;

  evaluate_ode(context, qd, goal, obstacles, metric, force, command, debug_data_ptr);

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
      clearances[index] = std::min(clearances[index], clearance);
    }
  }

  tangent_escape_ode::LeafParams leaf_params;
  leaf_params.enabled = params.enabled;
  leaf_params.metric_max = ode.metric_max;
  leaf_params.metric_sigma = ode.metric_sigma;
  leaf_params.accel_max = ode.accel_max;
  leaf_params.accel_sigma = ode.accel_sigma;
  leaf_params.budget_scale = ode.budget_scale;
  leaf_params.budget_recovery_time = ode.budget_recovery_time;
  leaf_params.tangent_damping = ode.tangent_damping;
  leaf_params.direction_rate = ode.direction_rate;
  leaf_params.risk_sigma = ode.risk_sigma;
  leaf_params.weight_goal = ode.weight_goal;
  leaf_params.weight_velocity = ode.weight_velocity;
  leaf_params.weight_risk = ode.weight_risk;
  leaf_params.weight_curvature = ode.weight_curvature;
  leaf_params.velocity_epsilon = ode.velocity_epsilon;
  leaf_params.axial_preference = ode.axial_preference;
  leaf_params.pocket_threshold = ode.pocket_threshold;
  leaf_params.pocket_epsilon = ode.pocket_epsilon;
  leaf_params.activation_distance = ode.activation_distance;
  leaf_params.release_distance = ode.release_distance;
  leaf_params.switch_hysteresis = ode.switch_hysteresis;
  leaf_params.direction_memory_time = ode.direction_memory_time;
  leaf_params.weight_stall = ode.weight_stall;
  leaf_params.stall_time_constant = ode.stall_time_constant;
  leaf_params.stall_speed = ode.stall_speed;
  leaf_params.clearance_margin = params.clearance_margin;
  leaf_params.control_dt = params.control_dt;

  const auto leaf = tangent_escape_ode::evaluate(
    ode_state_, leaf_params, context, qd, goal, clearances);

  if (leaf.active) {
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
  if (!leaf.active) {
    *debug_data = std::vector<double>{0.0};
    return;
  }

  // Slots 0, 9-13, 17-26, 28, 29, 45, 47 keep the meaning schemas 6-8 give
  // them, so the controller's existing tangent/normal/acceleration arrow
  // markers render for v2 without a second code path.  The v2-specific
  // quantities live in the slots those schemas leave free.
  std::vector<double> values(kTangentOdeDebugSize, 0.0);
  values[0] = 1.0;
  values[1] = static_cast<double>(leaf.sensor_index);
  values[2] = leaf.clearance;
  values[3] = leaf.theta;
  values[4] = leaf.theta_rate;
  values[5] = leaf.budget;
  values[6] = leaf.budget_remaining;
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
  values[47] = 1.0;                        // shared slot: high-level state
  *debug_data = std::move(values);
}

}  // namespace rb10_rmpflow_rviz

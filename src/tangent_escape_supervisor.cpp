#include "rb10_rmpflow_rviz/tangent_escape_supervisor.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "rb10_rmpflow_rviz/tangent_escape_ode.hpp"
#include "rb10_rmpflow_rviz/simple_tangent_leaf.hpp"

namespace rb10_rmpflow_rviz
{

TangentEscapeSupervisor::TangentEscapeSupervisor(EigenRmpConfig config)
: config_(std::move(config)), sensor_history_(config_.tangent_escape_ode.history)
{}

void TangentEscapeSupervisor::set_observations(const tangent_sensor_history::Input & input) const
{
  if (config_.tangent_escape_ode.direction_mode != tangent_sensor_history::DirectionMode::kLegacy) {
    sensor_history_.update(input);
  }
}

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
  using Mode = tangent_sensor_history::DirectionMode;
  const bool measured_mode = ode.direction_mode != Mode::kLegacy;
  const bool have_measurements = measured_mode && sensor_history_.current().stamp >= 0.0 &&
    sensor_history_.current().source_available;

  // Legacy and geometric-only simulations retain tagged sphere clearances.
  // The two experimental modes replace these with scalar observations below
  // whenever the bridge's observation stream is available.
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

  if (have_measurements) {
    const auto measured_ranges = sensor_history_.ranges();
    for (std::size_t i = 0; i < point_count; ++i) {
      // Distances are sensor-centred equivalent ranges. Do not subtract a
      // robot sphere radius or infer a measured obstacle bearing here.
      clearances[i] = measured_ranges[i];
    }
  }

  const double dt = std::max(params.control_dt, 1e-4);
  std::array<bool, tangent_escape_ode::kSensorCount> clearance_held{};
  for (std::size_t index = 0; index < point_count; ++index) {
    if (!std::isfinite(clearances[index])) {
      continue;
    }
    sensor_hold_state_.last_clearance[index] = clearances[index];
    sensor_hold_state_.missing_elapsed[index] = 0.0;
    sensor_hold_state_.valid[index] = true;
  }

  // A hold is only allowed after the leaf has selected an owner.  It therefore
  // cannot activate TG from a stale pre-activation sample.  During the hold,
  // move the effective clearance smoothly to the release edge; alpha_distance
  // reaches zero at the deadline instead of dropping the metric in one cycle.
  const double sensor_hold_time = std::max(ode.sensor_hold_time, 0.0);
  if (sensor_hold_time > 0.0 && ode_state_.active) {
    const std::size_t owner = ode_state_.sensor_index;
    if (
      owner < point_count &&
      !std::isfinite(clearances[owner]) &&
      sensor_hold_state_.valid[owner])
    {
      auto & missing_elapsed = sensor_hold_state_.missing_elapsed[owner];
      missing_elapsed += dt;
      if (missing_elapsed <= sensor_hold_time) {
        const double held_clearance = sensor_hold_state_.last_clearance[owner];
        clearances[owner] = tangent_escape_ode::held_sensor_clearance(
          held_clearance,
          ode.alpha_on_distance,
          ode.alpha_off_distance,
          missing_elapsed,
          sensor_hold_time);
        kinematic_rates[owner] = 0.0;
        clearance_held[owner] = true;
      } else {
        sensor_hold_state_.valid[owner] = false;
      }
    }
  }

  // Residual = everything the robot's own motion does not explain.  For a
  // stationary obstacle that is pure sensing noise, so it is low-passed and
  // dead-banded; for a moving one it is the obstacle's own approach, which is
  // the only way a distance-only sensor can see it at all.
  std::vector<double> clearance_rates(point_count, 0.0);
  const double tau = std::max(ode.clearance_rate_filter_tau, 0.0);
  const double blend = tau > dt ? dt / tau : 1.0;
  for (std::size_t index = 0; index < point_count; ++index) {
    auto & seeded = clearance_rate_state_.seeded[index];
    auto & previous = clearance_rate_state_.previous_clearance[index];
    auto & residual = clearance_rate_state_.filtered_residual[index];
    const double clearance = clearances[index];
    if (clearance_held[index]) {
      // The synthetic release ramp is not obstacle motion.  Feeding it through
      // the derivative estimator would double-attenuate the held metric.
      seeded = false;
      residual = 0.0;
      clearance_rates[index] = 0.0;
      continue;
    }
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
  if (measured_mode) {
    const auto measured_rates = sensor_history_.rates();
    for (std::size_t i = 0; i < point_count; ++i) {
      clearance_rates[i] = have_measurements && !clearance_held[i] ? measured_rates[i] : 0.0;
    }
  }

  tangent_escape_ode::LeafParams leaf_params;
  leaf_params.capture_history_decision = config_.record_history_diagnostics && debug_data != nullptr;
  leaf_params.direction_mode = ode.direction_mode;
  leaf_params.history = ode.history;
  leaf_params.response_input = sensor_history_.current();
  leaf_params.enabled = params.enabled;
  leaf_params.metric_mu = ode.metric_mu;
  leaf_params.metric_length = ode.metric_length;
  leaf_params.metric_epsilon = ode.metric_epsilon;
  leaf_params.metric_rate_scale = ode.metric_rate_scale;
  leaf_params.history_metric_rate_floor = ode.direction_mode ==
    tangent_sensor_history::DirectionMode::kSensorHistory ? ode.history_metric_rate_floor : 0.0;
  leaf_params.authority_boost = ode.authority_boost;
  leaf_params.authority_boost.enabled = ode.authority_boost.enabled &&
    ode.direction_mode == tangent_sensor_history::DirectionMode::kSensorHistory;
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
  leaf_params.e2_bias = ode.e2_bias;
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
  leaf_params.weight_predicted_risk = ode.weight_predicted_risk;
  leaf_params.prediction_distance = ode.prediction_distance;
  leaf_params.prediction_risk_length = ode.prediction_risk_length;
  leaf_params.prediction_damping = ode.prediction_damping;
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
  leaf_params.block_start_angle_deg = ode.block_start_angle_deg;
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

  // A sensor tag limits the existing clearance calculation to its owner.
  // Legacy geometric prediction treats each detection as a fixed world
  // sphere. Capacitive observations do not establish such sphere positions;
  // both experimental modes skip this path entirely.
  std::vector<tangent_escape_prediction::StaticObstacle> prediction_obstacles;
  if (!measured_mode && ode.weight_predicted_risk > 0.0) {
    prediction_obstacles.reserve(obstacles.size());
    for (const auto & obstacle : obstacles) {
      if (!std::isfinite(obstacle.radius) || obstacle.radius <= 0.0 ||
        !obstacle.center.allFinite())
      {
        continue;
      }
      const int source = obstacle.proximity_control_point_index;
      if (source >= 0 ? static_cast<std::size_t>(source) >= point_count :
        obstacle.source_id < 0)
      {
        continue;
      }
      prediction_obstacles.push_back({obstacle.center, obstacle.radius, source});
    }
  }

  // Common dimensionless prior for initialization and provisional rotation sign:
  // goal progress + predicted tangent motion - local range risk. Subsequent
  // angular advance depends on observed response, not this predicted maximum.
  leaf_params.response_mobility = [&](double angle, std::size_t owner) {
    auto trial_state = ode_state_;
    trial_state.active = true; trial_state.theta_valid = true;
    trial_state.sensor_index = owner; trial_state.theta = angle;
    trial_state.filtered_theta_rate = 0.0;
    // Candidate and actual policies both use full ownership authority.
    trial_state.handoff_gain = 1.0;
    auto trial_params = leaf_params;
    trial_params.response_mobility = {};
    trial_params.capture_history_decision = false; trial_params.freeze_owner = true;
    const auto trial = simple_tangent_leaf::evaluate_fixed(trial_state, trial_params,
      context, qd, filtered_nominal_qdd_, goal, clearances, clearance_rates);
    if (!trial.active || trial.metric <= 1e-9) return TgDirectionCandidate{};
    Matrix6 M = metric + trial.metric_contribution;
    const double scale = std::max(M.cwiseAbs().maxCoeff()*.01,1.0);
    M /= scale; M.diagonal().array() += std::max(config_.solve_offset,1e-9);
    JointVector a = M.ldlt().solve((force+trial.force_contribution)/scale);
    if (!a.allFinite()) return TgDirectionCandidate{};
    const double peak=a.cwiseAbs().maxCoeff();
    if (peak>config_.max_joint_accel && peak>0) a*=config_.max_joint_accel/peak;
    const double h=ode.history.horizon;
    JointVector displacement=h*qd+.5*h*h*a;
    const double norm=displacement.norm();
    if(norm>ode.history.max_joint_step)displacement*=ode.history.max_joint_step/norm;
    const auto risk=sensor_history_.evaluate(displacement,context);
    const Eigen::Vector3d error=goal-context.tcp_position;
    const double length=std::max(ode.history.escape_length,1e-9);
    TgDirectionCandidate candidate;candidate.valid=true;
    candidate.goal=ode.history.weight_goal*std::tanh(
      error.dot(context.tcp_jacobian*displacement)/(std::max(error.norm(),1e-9)*length));
    candidate.execution=ode.history.weight_escape*std::tanh(
      trial.tangent.dot(context.control_point_jacobians[owner]*displacement)/length);
    // The old risk remains separately configurable (default zero). The new
    // non-owner cost only uses validated, direction-supported closing evidence.
    // Existing shadow fields retain the original diagnostic model semantics.
    const auto neighbor_risk=sensor_history_.evaluate_neighbors(displacement,owner);
    candidate.risk=ode.history.weight_risk*risk.value +
      ode.history.weight_neighbor_risk*neighbor_risk.value;
    candidate.neighbor={ode.history.weight_neighbor_risk*neighbor_risk.value,
      neighbor_risk.confidence,static_cast<double>(neighbor_risk.model_count),
      static_cast<double>(neighbor_risk.unknown_count),static_cast<double>(neighbor_risk.critical_sensor),
      neighbor_risk.critical_change,neighbor_risk.critical_sigma};
    candidate.shadow={risk.nominal_cost,risk.uncertainty_cost,risk.unknown_motion,
      risk.shadow_support,static_cast<double>(risk.shadow_sensor),risk.critical_change,risk.critical_sigma};
    return candidate;
  };

  const auto objective = [&](double theta, std::size_t owner) {
      const auto cost = [&](double angle) {
          auto trial_state = ode_state_;
          trial_state.active = true;
          trial_state.theta_valid = true;
          trial_state.sensor_index = owner;
          trial_state.theta = angle;
          trial_state.filtered_theta_rate = 0.0;
          trial_state.handoff_gain = 1.0;
          auto trial_params = leaf_params;
          trial_params.capture_history_decision = false;
          trial_params.freeze_owner = true;
          // Freeze direction only for this short-horizon counterfactual.
          const auto trial = simple_tangent_leaf::evaluate_fixed(
            trial_state, trial_params, context, qd, filtered_nominal_qdd_, goal,
            clearances, clearance_rates);
          Matrix6 root_metric = metric + trial.metric_contribution;
          const double scale = std::max(root_metric.cwiseAbs().maxCoeff() * 0.01, 1.0);
          root_metric /= scale;
          root_metric.diagonal().array() += std::max(config_.solve_offset, 1e-9);
          const JointVector root_force = (force + trial.force_contribution) / scale;
          JointVector acceleration = root_metric.ldlt().solve(root_force);
          if (!acceleration.allFinite()) { acceleration.setZero(); }
          const double peak = acceleration.cwiseAbs().maxCoeff();
          if (peak > config_.max_joint_accel && peak > 0.0) {
            acceleration *= config_.max_joint_accel / peak;
          }
          const double h = ode.history.horizon;
          JointVector displacement = h * qd + 0.5 * h * h * acceleration;
          const double norm = displacement.norm();
          if (norm > ode.history.max_joint_step) {
            displacement *= ode.history.max_joint_step / norm;
          }
          auto risk = sensor_history_.evaluate(displacement, context);
          // Fixed physical scale: do not normalize each candidate by its own motion.
          risk.escape = std::tanh(trial.tangent.dot(context.control_point_jacobians[owner] * displacement) /
            ode.history.escape_length);
          risk.predicted_displacement = displacement;
          risk.unclipped_displacement_norm = norm;
          return risk;
        };
      auto result = cost(theta);
      constexpr double epsilon = 1e-3;
      const auto plus = cost(theta + epsilon), minus = cost(theta - epsilon);
      result.gradient = (plus.value - minus.value) / (2.0 * epsilon);
      result.escape_gradient = (plus.escape - minus.escape) / (2.0 * epsilon);
      return result;
    };
  std::vector<Eigen::Vector3d> measured_velocities;
  const auto & measured = sensor_history_.current();
  if (ode.direction_mode == Mode::kSimple && measured.stamp >= 0.0 &&
    measured.velocity_available && measured.qd.allFinite())
  {
    const auto measured_context = RB10Model::forward_context(
      measured.q, ControlPointJacobianMode::kAnalytic);
    for (const auto & jacobian : measured_context.control_point_jacobians) {
      measured_velocities.push_back(jacobian * measured.qd);
    }
  }
  const auto leaf = ode.direction_mode == Mode::kSimple ?
    simple_tangent_leaf::evaluate(
    ode_state_, leaf_params, context, qd, filtered_nominal_qdd_, goal, clearances, clearance_rates,
    measured_velocities) :
    tangent_escape_ode::evaluate(
    ode_state_, leaf_params, context, qd, filtered_nominal_qdd_, goal,
    clearances, clearance_rates, prediction_obstacles,
    ode.direction_mode == Mode::kSensorHistory ?
    tangent_escape_ode::DirectionObjective(objective) : tangent_escape_ode::DirectionObjective{});

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
  if (!measured_mode && !leaf.active && !leaf.observation_valid) {
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
  values[tangent_escape_debug::kProgress] = context.tcp_position.x();
  values[tangent_escape_debug::kProgress + 1] = context.tcp_position.y();
  values[tangent_escape_debug::kProgress + 2] = context.tcp_position.z();
  values[tangent_escape_debug::kProgress + 3] = (goal - context.tcp_position).norm();
  for (int axis = 0; axis < 3; ++axis) {
    values[tangent_escape_debug::kBasis + axis] = leaf.basis_e1[axis];
    values[tangent_escape_debug::kBasis + 3 + axis] = leaf.basis_e2[axis];
    values[tangent_escape_debug::kBasis + 6 + axis] = leaf.goal_in_plane[axis];
  }
  const std::size_t prediction_base = tangent_escape_debug::kPrediction;
  values[prediction_base] = leaf.predicted_risk.value;
  values[prediction_base + 1] = leaf.predicted_risk_score;
  values[prediction_base + 2] = leaf.predicted_risk.gradient;
  values[prediction_base + 3] = leaf.predicted_risk.min_clearance;
  values[prediction_base + 4] = leaf.predicted_risk.critical_sensor_index;
  values[prediction_base + 5] = leaf.predicted_risk.obstacle_sensor_index;
  values[prediction_base + 6] = static_cast<double>(leaf.predicted_risk.pair_count);
  values[prediction_base + 7] = static_cast<double>(leaf.predicted_risk.approaching_pair_count);
  values[prediction_base + 8] = leaf.total_direction_score;
  const std::size_t history_base = tangent_escape_debug::kHistory;
  values[history_base] = static_cast<double>(ode.direction_mode);
  values[history_base + 1] = leaf.history_risk.value;
  values[history_base + 2] = leaf.history_risk_score;
  values[history_base + 3] = leaf.history_risk.gradient;
  values[history_base + 4] = leaf.history_risk.unknown_motion;
  values[history_base + 5] = leaf.history_risk.min_predicted_range;
  values[history_base + 6] = leaf.history_risk.confidence;
  values[history_base + 7] = static_cast<double>(leaf.history_risk.model_count);
  values[history_base + 8] = static_cast<double>(leaf.history_risk.unknown_count);
  values[history_base + 9] = leaf.history_risk.critical_sensor;
  values[history_base + 10] = leaf.handoff_gain;
  for (std::size_t i = 0; i < tangent_escape_ode::kSensorCount; ++i) {
    const auto & model = sensor_history_.models()[i];
    const auto base = tangent_escape_debug::kHistorySensors + 4 * i;
    values[base] = static_cast<double>(model.count);
    values[base + 1] = model.stamp >= 0.0 ? sensor_history_.current().stamp - model.stamp : -1.0;
    values[base + 2] = model.noise;
    values[base + 3] = model.gradient.norm();
  }
  values[tangent_escape_debug::kSteering] = leaf.steering_velocity.norm();
  values[tangent_escape_debug::kSteering + 1] = leaf.tangent.dot(leaf.steering_velocity);
  values[tangent_escape_debug::kSteering + 2] = leaf.steering_velocity_source;
  values[tangent_escape_debug::kSteering + 3] = leaf.steering_fallback;
  values[tangent_escape_debug::kSteering + 4] = leaf.steering_turn_rate;
  if (ode.direction_mode == Mode::kSensorHistory && config_.record_history_diagnostics) {
    // Append-only extension. Existing schema-9 readers retain their offsets.
    // Lightweight decisions accompany every cycle; models on change plus
    // full keyframes every 0.5 s allow recovery after a dropped packet.
    const std::size_t start = values.size();
    values.insert(values.end(), {45045.0, 11.0, 0.0}); // magic, version, payload length
    const auto & d = leaf.decision;
    values.insert(values.end(), {
      static_cast<double>(++history_record_sequence_), dt,
      static_cast<double>(d.events), d.active_before ? 1.0 : 0.0,
      d.theta_valid_before ? 1.0 : 0.0, static_cast<double>(d.owner_before),
      static_cast<double>(d.closest), d.closest_clearance, d.owner_clearance,
      d.theta_before, d.theta_evaluated, ode_state_.theta,
      d.filtered_rate_before, d.raw_rate, d.clamped_rate, ode_state_.filtered_theta_rate,
      d.handoff_before, ode_state_.handoff_gain,
      d.goal_gradient, d.continuity_gradient, d.history_gradient, d.total_gradient,
      d.risk_before.value, d.risk_before.gradient, d.risk_before.measured,
      d.risk_before.unknown_motion, d.risk_before.min_predicted_range,
      d.risk_before.confidence, static_cast<double>(d.risk_before.model_count),
      static_cast<double>(d.risk_before.unknown_count), static_cast<double>(d.risk_before.critical_sensor),
      have_measurements ? 1.0 : 0.0, measured.stamp, measured.source_available ? 1.0 : 0.0,
      measured.velocity_available ? 1.0 : 0.0});
    const auto append_vector = [&values](const auto & vector) {
        for (int i = 0; i < vector.size(); ++i) { values.push_back(vector[i]); }
      };
    append_vector(measured.q); append_vector(measured.qd);
    append_vector(nominal_qdd); append_vector(filtered_nominal_qdd_);
    append_vector(d.risk_before.predicted_displacement);
    values.push_back(d.risk_before.unclipped_displacement_norm);
    values.push_back(static_cast<double>(d.seeds.size()));
    for (const auto & seed : d.seeds) { values.insert(values.end(), seed.begin(), seed.end()); }
    const auto & h = ode.history;
    // Includes compiled estimator constants that ROS parameter services cannot report.
    values.insert(values.end(), {h.window, h.horizon, h.clearance, h.risk_length,
      h.weight_goal, h.weight_risk, h.weight_continuity, h.unknown_motion_weight,
      h.max_direction_rate, h.direction_filter_tau, h.handoff_time, h.initial_angle,
      h.fresh_time, h.expiry, h.sample_interval, h.ridge, h.noise_floor, h.max_joint_step,
      ode.direction_rate, ode.owner_hysteresis, ode.sensor_hold_time, config_.solve_offset});
    values.push_back(static_cast<double>(point_count));
    for (std::size_t i = 0; i < point_count; ++i) {
      values.insert(values.end(), {measured.samples[i].stamp, measured.samples[i].range,
        clearances[i], clearance_rates[i], clearance_held[i] ? 1.0 : 0.0,
        sensor_hold_state_.missing_elapsed[i],
        static_cast<double>(sensor_history_.update_reasons()[i]), sensor_history_.update_stamps()[i]});
    }
    history_keyframe_elapsed_ += dt;
    const bool keyframe = history_keyframe_elapsed_ >= 0.5;
    if (keyframe) { history_keyframe_elapsed_ = 0.0; }
    std::vector<std::size_t> changed;
    for (std::size_t i = 0; i < tangent_escape_ode::kSensorCount; ++i) {
      const auto & model = sensor_history_.models()[i];
      const std::array<double, 3> key{model.stamp, static_cast<double>(model.count), model.valid ? 1.0 : 0.0};
      if (keyframe || key != recorded_model_keys_[i]) { changed.push_back(i); }
      recorded_model_keys_[i] = key;
    }
    if (!changed.empty()) { ++history_model_revision_; }
    values.insert(values.end(), {keyframe ? 1.0 : 0.0,
      static_cast<double>(history_model_revision_), static_cast<double>(changed.size())});
    for (auto i : changed) {
      const auto & model = sensor_history_.models()[i];
      values.insert(values.end(), {static_cast<double>(i), model.valid ? 1.0 : 0.0,
        model.stamp, static_cast<double>(model.count), model.range, model.rate, model.noise});
      append_vector(model.anchor_q); append_vector(model.gradient);
      for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 6; ++col) { values.push_back(model.information_inverse(row, col)); }
      }
    }
    // Version 2: record the actual activation reference independently of the
    // legacy nominal-acceleration diagnostics, even on compact snapshots.
    values.insert(values.end(), {leaf.block_from_goal_progress ? 2.0 : 0.0,
      leaf.block_reference_valid ? 1.0 : 0.0, ode.nominal_motion_epsilon});
    append_vector(leaf.goal_progress_joint_motion);
    append_vector(leaf.block_reference);
    values.insert(values.end(), {h.weight_escape, h.escape_length, h.reselect_interval,
      d.escape_gradient, d.risk_before.escape, leaf.history_risk.escape,
      static_cast<double>(d.seed_escape.size())});
    values.insert(values.end(), d.seed_escape.begin(), d.seed_escape.end());
    const auto & response=ode_state_.response;
    values.insert(values.end(), {static_cast<double>(response.phase),static_cast<double>(response.result),
      response.target,response.accepted,response.speed,response.baseline,
      static_cast<double>(response.failures),measured.encoder_stamp,
      response.best_speed, response.seeded?1.0:0.0});
    values.insert(values.end(), {response.anchor,response.has_accepted?1.0:0.0,
      static_cast<double>(response.sign),static_cast<double>(response.sweep_sequence),
      response.current_score,response.best_score,response.chosen_angle,response.sweep_us,
      response.scan_metric,static_cast<double>(response.tried),response.bound_clamped?1.0:0.0});
    values.insert(values.end(),response.sweep_scores.begin(),response.sweep_scores.end());
    values.insert(values.end(),{response.observation_confidence,response.goal_speed,
      response.poor_evidence,response.progress_evidence,response.pending_angle,
      static_cast<double>(response.pending_count)});
    for(const auto * component:{&response.sweep_goal,&response.sweep_execution,
        &response.sweep_risk,&response.sweep_continuity})
      values.insert(values.end(),component->begin(),component->end());
    values.insert(values.end(),{static_cast<double>(response.reason),response.turning?1.:0.,
      response.turn_travel,response.turn_elapsed,static_cast<double>(response.reverse_count),
      response.range_closing,response.min_range,response.adverse_evidence,response.retreat_evidence,
      response.positive_score,response.negative_score,response.sign_margin,response.window_dt,
      static_cast<double>(response.paired_samples),response.turn_rate_limit,
      TgResponseDirection::failure_weight,response.evaluated_angle});
    values.insert(values.end(),response.sweep_failure.begin(),response.sweep_failure.end());
    const auto shadow_start=values.size();
    for(const auto & component:response.shadow)
      values.insert(values.end(),component.begin(),component.end());
    values.insert(values.end(),{response.shadow_window_stamp,response.shadow_stalled,response.shadow_worsening});
    values.push_back(static_cast<double>(sensor_history_.prediction_checks().size()));
    for(const auto & c:sensor_history_.prediction_checks())
      values.insert(values.end(),{c.stamp,c.model_stamp,c.predicted,c.observed,c.error,
        c.sigma,c.support,c.valid?1.:0.});
    values.push_back(static_cast<double>(values.size()-shadow_start+1));

    values.insert(values.end(), {response.command_support,response.command_speed,
      response.execution_supported?1.:0.,response.unsupported_elapsed,response.net_travel,
      response.absolute_travel,response.progress_duration,response.progress_ratio,
      measured.command_stamp,measured.command_available?1.:0.,ode.history_metric_rate_floor});
    append_vector(measured.command_qd);
    const auto & boost=response.authority;
    const auto & bp=ode.authority_boost;
    values.insert(values.end(),{leaf.metric_base,leaf.authority_gain,boost.gain,
      static_cast<double>(boost.phase),boost.deficit_time,boost.cap_time,
      boost.command_coverage,boost.exhausted?1.:0.,bp.enabled?1.:0.,bp.maximum,
      bp.rise_rate,bp.decay_rate,bp.trigger_time,bp.settle_time});
    values.push_back(ode.history.weight_neighbor_risk);
    for (const auto & component:response.neighbor)
      values.insert(values.end(),component.begin(),component.end());
    values[start + 2] = static_cast<double>(values.size() - start - 3);
  }
  *debug_data = std::move(values);
}

}  // namespace rb10_rmpflow_rviz

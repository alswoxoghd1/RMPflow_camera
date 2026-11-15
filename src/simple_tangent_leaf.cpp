#include "rb10_rmpflow_rviz/simple_tangent_leaf.hpp"

#include <algorithm>
#include <cmath>

namespace rb10_rmpflow_rviz::simple_tangent_leaf
{
tangent_escape_ode::LeafOutput evaluate_fixed(
  tangent_escape_ode::LeafState & state,
  tangent_escape_ode::LeafParams params,
  const KinematicsContext & context,
  const RB10Model::JointVector & qd,
  const RB10Model::JointVector & nominal_qdd,
  const Eigen::Vector3d & goal,
  const std::vector<double> & ranges,
  const std::vector<double> & rates)
{
  params.goal_progress_blockage = params.goal_progress_blockage ||
    params.direction_mode == tangent_sensor_history::DirectionMode::kSensorHistory;
  params.direction_mode = tangent_sensor_history::DirectionMode::kFixed;
  return tangent_escape_ode::evaluate(state, params, context, qd, nominal_qdd, goal, ranges, rates);
}

tangent_escape_ode::LeafOutput evaluate(
  tangent_escape_ode::LeafState & state,
  tangent_escape_ode::LeafParams params,
  const KinematicsContext & context,
  const RB10Model::JointVector & qd,
  const RB10Model::JointVector & nominal_qdd,
  const Eigen::Vector3d & goal,
  const std::vector<double> & ranges,
  const std::vector<double> & rates,
  const std::vector<Eigen::Vector3d> & measured_velocities)
{
  const auto previous = state.steering_direction;
  const bool previous_valid = state.active && state.steering_direction_valid;
  const auto previous_owner = state.sensor_index;
  const double previous_rate = state.steering_rate;
  // Pure steering has zero instantaneous work at nonzero velocity. The old
  // progress integral of a sensor-plane tangent is not its energy coordinate.
  params.enable_progress_attenuation = false;
  state.progress = 0.0;
  auto output = evaluate_fixed(state, params, context, qd, nominal_qdd, goal, ranges, rates);
  output.direction_mode = static_cast<int>(tangent_sensor_history::DirectionMode::kSimple);
  if (!output.active) { return output; }
  state.progress = output.progress = 0.0;
  output.progress_remaining = 1.0;

  const auto owner = output.sensor_index;
  const auto & jacobian = context.control_point_jacobians[owner];
  const bool measured = owner < measured_velocities.size() && measured_velocities[owner].allFinite();
  const Eigen::Vector3d velocity = measured ? measured_velocities[owner] :
    context.control_point_velocities[owner];
  output.steering_velocity = velocity;
  output.steering_velocity_source = measured ? 1 : 0;
  const double speed = velocity.norm();
  const double epsilon = std::max(params.velocity_epsilon, 1e-8);
  Eigen::Vector3d axis = velocity;
  if (speed < epsilon) {
    axis = jacobian * nominal_qdd + context.control_point_curvatures[owner];
    output.steering_fallback |= 1;
    if (axis.norm() <= std::max(params.nominal_motion_epsilon, 1e-9)) {
      axis = output.obstacle_direction;
      output.steering_fallback = 2;
    }
    // Connect the startup reference continuously to the measured velocity
    // plane. An abrupt switch at epsilon would rotate the force in one tick.
    // A plane is unchanged by flipping its normal, so use the shorter arc.
    if (speed > 1e-12 && axis.allFinite() && axis.norm() > 1e-9) {
      const Eigen::Vector3d startup_axis = axis.normalized();
      Eigen::Vector3d velocity_axis = velocity / speed;
      if (startup_axis.dot(velocity_axis) < 0.0) { velocity_axis = -velocity_axis; }
      const double fraction = speed / epsilon;
      const double blend = fraction * fraction * (3.0 - 2.0 * fraction);
      axis = (1.0 - blend) * startup_axis + blend * velocity_axis;
    }
  }
  if (!axis.allFinite() || axis.norm() < 1e-9) {
    state = tangent_escape_ode::LeafState{};
    output.active = false;
    output.metric = 0.0;
    output.metric_contribution.setZero();
    output.force_contribution.setZero();
    return output;
  }
  axis.normalize();
  const auto project = [&axis](const Eigen::Vector3d & vector) -> Eigen::Vector3d {
      return vector - axis * axis.dot(vector);
    };

  // Define a deterministic basis in the velocity-normal plane for the
  // singular head-on case and diagnostics. initial_angle only breaks ties.
  Eigen::Vector3d basis1 = project(output.basis_e1);
  if (basis1.norm() < 1e-6) { basis1 = project(output.basis_e2); }
  if (basis1.norm() < 1e-6) { basis1 = axis.unitOrthogonal(); }
  basis1.normalize();
  const Eigen::Vector3d basis2 = axis.cross(basis1).normalized();
  Eigen::Vector3d seed = std::cos(params.history.initial_angle) * basis1 +
    std::sin(params.history.initial_angle) * basis2;
  Eigen::Vector3d transported = previous_valid ? project(previous) : seed;
  if (transported.norm() < 1e-6) {
    transported = seed;
    // A discontinuous change of velocity can destroy the previous projection.
    // Choose the deterministic seed without an authority fade.
    state.handoff_gain = 1.0;
  }
  transported.normalize();

  // v x (v x n) / |v|^2 = -(I - u u^T)n, for u=v/|v|.
  // n is explicitly the sensor mounting-axis proxy; no obstacle position is
  // reconstructed from the omnidirectional capacitive scalar measurement.
  Eigen::Vector3d desired = project(-output.obstacle_direction);
  const double lateral = desired.norm();
  double angle = 0.0;
  if (lateral > 1e-8) {
    desired /= lateral;
    angle = std::atan2(axis.dot(transported.cross(desired)), transported.dot(desired));
  }
  // Near parallel, retain the previous side rather than amplify tiny noise
  // into a 180-degree change. Transition gradually to the mounting-axis rule.
  double blend = std::clamp((lateral - 0.05) / 0.10, 0.0, 1.0);
  blend = blend * blend * (3.0 - 2.0 * blend);
  if (blend < 1.0) { output.steering_fallback |= 4; }
  const double dt = std::max(params.control_dt, 1e-4);
  const double tau = std::max(params.history.direction_filter_tau, dt);
  const double limit = std::max(params.history.max_direction_rate, 0.0);
  const double target_rate = std::clamp(blend * angle / tau, -limit, limit);
  double rate = previous_valid && previous_owner == owner ? previous_rate : 0.0;
  rate += dt / (tau + dt) * (target_rate - rate);
  if (blend == 0.0) { rate = 0.0; }
  if (!previous_valid) {
    // Initial choice can use the current velocity directly; no score search.
    transported = Eigen::AngleAxisd(blend * angle, axis) * transported;
    rate = 0.0;
  }
  Eigen::Vector3d direction = Eigen::AngleAxisd(rate * dt, axis) * transported;
  direction = project(direction).normalized();
  state.steering_direction = direction;
  state.steering_direction_valid = true;
  state.steering_rate = rate;

  output.tangent = direction;
  output.tangent_perpendicular = axis.cross(direction);
  output.basis_e1 = basis1;
  output.basis_e2 = basis2;
  output.theta = std::atan2(direction.dot(basis2), direction.dot(basis1));
  output.theta_rate = rate;
  output.steering_turn_rate = rate;
  output.goal_in_plane = project(jacobian * nominal_qdd + context.control_point_curvatures[owner]);
  output.tangent_rate = direction.dot(velocity);
  state.previous_tangent_rate = output.tangent_rate;
  output.damping_acceleration = std::max(params.tangent_damping, 0.0) * output.tangent_rate;
  output.desired_acceleration = output.push_acceleration - output.damping_acceleration;
  output.handoff_gain = state.handoff_gain;
  output.natural_force = output.metric * output.desired_acceleration;
  output.injected_power = output.natural_force * output.tangent_rate;
  output.tangent_normal_residual = std::abs(direction.dot(output.obstacle_direction));

  // Cartesian task x=p_i(q), M_x=m*t*t^T, f_x=m*a*t. Only the curvature of
  // p_i enters pullback. Differentiating t(v)^T*v=0 as the old 1-D coordinate
  // would cancel the requested steering acceleration.
  output.leaf_jacobian = direction.transpose() * jacobian;
  output.jdot_theta_term = output.jdot_omega_term = 0.0;
  output.jdot_qdot = direction.dot(context.control_point_curvatures[owner]);
  output.metric_contribution = output.metric * output.leaf_jacobian.transpose() * output.leaf_jacobian;
  output.force_contribution = output.leaf_jacobian.transpose() *
    (output.natural_force - output.metric * output.jdot_qdot);
  if (!output.metric_contribution.allFinite() || !output.force_contribution.allFinite()) {
    output.active = false;
    output.metric = 0.0;
    output.metric_contribution.setZero();
    output.force_contribution.setZero();
    state = tangent_escape_ode::LeafState{};
  }
  return output;
}
}

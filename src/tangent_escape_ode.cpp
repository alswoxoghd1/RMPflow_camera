#include "rb10_rmpflow_rviz/tangent_escape_ode.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

namespace rb10_rmpflow_rviz
{
namespace tangent_escape_ode
{
namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kRingSize = 4;

double sigmoid(double value)
{
  return 1.0 / (1.0 + std::exp(-std::clamp(value, -60.0, 60.0)));
}

double smoothstep(double value, double lower, double upper)
{
  if (upper <= lower + 1e-12) {
    return value >= upper ? 1.0 : 0.0;
  }
  const double t = std::clamp((value - lower) / (upper - lower), 0.0, 1.0);
  return t * t * t * (10.0 + t * (-15.0 + 6.0 * t));
}

double risk_weight(double clearance, double sigma)
{
  if (!std::isfinite(clearance)) {
    return 0.0;
  }
  return std::exp(-std::max(clearance, 0.0) / std::max(sigma, 1e-9));
}

// Sensors on the same link and close together along the link axis form one
// neighbourhood.  The 200 mm gap between the two stations on links 2 and 3
// keeps them separate rings, while the 60 mm gap on the wrist merges its
// lateral trio with its forward sensor into a single group of four.
constexpr double kStationMergeDistance = 0.10;

std::array<SensorNeighborhood, kSensorCount> build_neighborhood_table()
{
  std::array<SensorNeighborhood, kSensorCount> table{};

  std::map<std::size_t, std::vector<std::size_t>> by_link;
  for (std::size_t index = 0; index < kSensorCount; ++index) {
    by_link[RB10Model::sensor_control_points[index].parent_link].push_back(index);
  }

  for (auto & entry : by_link) {
    auto members = entry.second;
    std::sort(
      members.begin(), members.end(),
      [](std::size_t lhs, std::size_t rhs) {
        return RB10Model::sensor_control_points[lhs].offset.z() <
        RB10Model::sensor_control_points[rhs].offset.z();
      });

    // Single-linkage cluster along the link axis.
    std::vector<std::vector<std::size_t>> groups;
    for (std::size_t member : members) {
      const double station = RB10Model::sensor_control_points[member].offset.z();
      if (groups.empty()) {
        groups.push_back({member});
        continue;
      }
      const double previous_station =
        RB10Model::sensor_control_points[groups.back().back()].offset.z();
      if (std::abs(station - previous_station) <= kStationMergeDistance) {
        groups.back().push_back(member);
      } else {
        groups.push_back({member});
      }
    }

    for (const auto & group : groups) {
      if (group.size() < 2) {
        // A lone sensor has no neighbour risk to draw on; the direction would
        // be driven by goal and velocity alone, which is not enough to pick a
        // side.  Leave it out rather than let it own the leaf blindly.
        continue;
      }
      for (std::size_t index : group) {
        const auto & self = RB10Model::sensor_control_points[index];
        const Eigen::Vector3d normal = self.local_normal.normalized();
        const Eigen::Vector3d axis1 = self.local_tangent_bias.normalized();
        const Eigen::Vector3d axis2 = normal.cross(axis1);

        auto & neighbourhood = table[index];
        neighbourhood.count = 0;
        for (std::size_t other : group) {
          if (other == index || neighbourhood.count >= kMaxNeighbors) {
            continue;
          }
          const Eigen::Vector3d other_normal =
            RB10Model::sensor_control_points[other].local_normal.normalized();
          const double alignment = std::clamp(normal.dot(other_normal), -1.0, 1.0);

          NeighborContribution contribution;
          contribution.index = other;
          if (alignment < -0.999) {
            // Antipodal: no tangent-plane direction, only "far side blocked".
            contribution.axial_block = true;
            contribution.weight = 1.0;
          } else if (alignment > 0.999) {
            // Same direction: says nothing about which tangent to take.
            continue;
          } else {
            const Eigen::Vector3d projected = other_normal - alignment * normal;
            contribution.axial_block = false;
            contribution.weight = projected.norm();
            contribution.phi = std::atan2(
              axis2.dot(other_normal), axis1.dot(other_normal));
          }
          neighbourhood.entries[neighbourhood.count++] = contribution;
        }

        if (neighbourhood.count == 0) {
          continue;
        }
        neighbourhood.valid = true;

        // eps_i orients the axial escape preference toward the distal end.
        const double axial = self.local_tangent_bias.z();
        if (std::abs(axial) < 1e-9) {
          neighbourhood.axial_sign = 1.0;
          neighbourhood.axial_preference_defined = false;
        } else {
          neighbourhood.axial_sign = axial >= 0.0 ? 1.0 : -1.0;
          neighbourhood.axial_preference_defined = true;
        }
      }
    }
  }

  return table;
}

}  // namespace

const std::array<SensorNeighborhood, kSensorCount> & neighborhood_table()
{
  static const std::array<SensorNeighborhood, kSensorCount> table =
    build_neighborhood_table();
  return table;
}

double distance_activation(double clearance, double on, double off)
{
  if (!std::isfinite(clearance)) {
    return 0.0;
  }
  // The quintic is flat to second order at both ends, so the gate it will
  // eventually multiply reaches zero without a kink in its time derivative.
  return 1.0 - smoothstep(clearance, on, off);
}

double held_sensor_clearance(
  double last_clearance,
  double alpha_on_distance,
  double alpha_off_distance,
  double missing_elapsed,
  double hold_time)
{
  if (!std::isfinite(last_clearance)) {
    return std::numeric_limits<double>::infinity();
  }
  if (hold_time <= 0.0) {
    return last_clearance;
  }
  const double release_clearance =
    std::max(alpha_on_distance, alpha_off_distance);
  const double fade = std::clamp(missing_elapsed / hold_time, 0.0, 1.0);
  return last_clearance +
         fade * (std::max(last_clearance, release_clearance) - last_clearance);
}

double direction_score(double theta, double a, double b, double c, double e)
{
  return a * std::cos(theta) + b * std::sin(theta) +
         c * std::cos(2.0 * theta) + e * std::sin(2.0 * theta);
}

double direction_score_gradient(double theta, double a, double b, double c, double e)
{
  return -a * std::sin(theta) + b * std::cos(theta) -
         2.0 * c * std::sin(2.0 * theta) + 2.0 * e * std::cos(2.0 * theta);
}

LeafOutput evaluate(
  LeafState & state,
  const LeafParams & params,
  const KinematicsContext & context,
  const JointVector & qd,
  const JointVector & nominal_qdd,
  const Eigen::Vector3d & goal,
  const std::vector<double> & clearances,
  const std::vector<double> & clearance_rates,
  const std::vector<tangent_escape_prediction::StaticObstacle> & prediction_obstacles,
  const DirectionObjective & objective)
{
  LeafOutput output;
  using Mode = tangent_sensor_history::DirectionMode;
  const bool legacy = params.direction_mode == Mode::kLegacy;
  const bool adaptive = params.direction_mode == Mode::kSensorHistory;
  const bool progress_block = adaptive || params.goal_progress_blockage;
  output.block_from_goal_progress = progress_block;
  if (progress_block) { output.alpha_block = 0.0; }
  output.direction_mode = static_cast<int>(params.direction_mode);
  const bool capture = adaptive && params.capture_history_decision;
  auto & decision = output.decision;
  if (capture) {
    decision.active_before = state.active;
    decision.theta_valid_before = state.theta_valid;
    decision.owner_before = state.sensor_index;
    decision.theta_before = state.theta;
    decision.filtered_rate_before = state.filtered_theta_rate;
    decision.handoff_before = state.handoff_gain;
  }

  const auto & neighbours_table = neighborhood_table();
  const std::size_t point_count = std::min(
    {
      context.control_points.size(),
      context.control_point_jacobians.size(),
      context.control_point_velocities.size(),
      context.control_point_curvatures.size(),
      clearances.size(),
      kSensorCount
    });

  if (!params.enabled || params.metric_mu <= 0.0 || point_count == 0) {
    if (capture) { decision.events |= 64; }
    state = LeafState{};
    return output;
  }

  const double dt = std::max(params.control_dt, 1e-4);

  // One reference for both direction preference and history activation. It
  // remains meaningful at zero actual velocity and during collision retreat.
  // The sensor mounting axis is only a bearing proxy, not a measured normal.
  JointVector goal_progress = JointVector::Zero();
  if (progress_block) {
    constexpr double inverse_damping = 0.01;
    const Eigen::Matrix3d gram =
      context.tcp_jacobian * context.tcp_jacobian.transpose() +
      inverse_damping * inverse_damping * Eigen::Matrix3d::Identity();
    goal_progress = context.tcp_jacobian.transpose() *
      gram.ldlt().solve(goal - context.tcp_position);
    if (!goal_progress.allFinite()) { goal_progress.setZero(); }
    output.goal_progress_joint_motion = goal_progress;
  }

  // Ownership is restricted to goal-blocking detections in history mode.
  // This uses the same mounting-axis proxy and angle gate as activation; it
  // does not turn a scalar capacitive measurement into an obstacle bearing.
  const auto goal_block = [&](std::size_t k) {
      const Eigen::Vector3d motion = context.control_point_jacobians[k] * goal_progress;
      const double norm = motion.norm();
      const auto & sensor = RB10Model::sensor_control_points[k];
      const Eigen::Vector3d axis =
        (context.link_rotations[sensor.parent_link] * sensor.local_normal).normalized();
      const double angle = params.block_start_angle_deg;
      if (!motion.allFinite() || norm <= std::max(params.nominal_motion_epsilon, 1e-12) ||
        !std::isfinite(angle) || angle <= 0.0 || angle > 180.0) { return 0.0; }
      const double degrees = std::acos(std::clamp(axis.dot(motion)/norm, -1.0, 1.0))*180.0/kPi;
      return std::clamp(1.0-degrees/angle, 0.0, 1.0);
    };
  const auto eligible = [&](std::size_t k) {
      if (k >= point_count || !neighbours_table[k].valid) { return false; }
      const double d = clearances[k]-params.clearance_margin;
      return std::isfinite(d) && distance_activation(d,params.alpha_on_distance,
        params.alpha_off_distance)>0.0 && (!progress_block || goal_block(k)>0.0);
    };
  // --- ownership: one leaf for the whole robot -----------------------------
  std::size_t closest = 0;
  double closest_clearance = std::numeric_limits<double>::infinity();
  bool found = false;
  for (std::size_t index = 0; index < point_count; ++index) {
    if (!neighbours_table[index].valid) {
      continue;
    }
    const double clearance = clearances[index] - params.clearance_margin;
    if (!std::isfinite(clearance)) {
      continue;
    }
    if (!found || clearance < closest_clearance) {
      closest = index;
      closest_clearance = clearance;
      found = true;
    }
  }

  // Keep nearest detection for diagnostics when no sensor blocks the goal.
  // Such a detection receives zero activation, never a fallback TG direction.
  bool have_blocker = false;
  if (progress_block) {
    for (std::size_t k=0;k<point_count;++k) {
      if (!eligible(k)) { continue; }
      const double d=clearances[k]-params.clearance_margin;
      if (!have_blocker || d<closest_clearance) { closest=k;closest_clearance=d; }
      have_blocker=true;
    }
  }

  const auto reset = [&state]() {
      state = LeafState{};
    };

  if (!found) {
    if (capture) { decision.events |= 16; }
    reset();
    return output;
  }

  // --- blockage observation ------------------------------------------------
  // Deliberately ahead of the activation gate and covering every sensor, not
  // just the owner: the question this answers is which reference direction
  // detects a blocked path, and the interesting part of that is the approach,
  // before any gate has fired, on links that are not the closest one.
  for (std::size_t k = 0; k < point_count; ++k) {
    const double sensor_clearance = clearances[k] - params.clearance_margin;
    if (!std::isfinite(sensor_clearance)) {
      continue;
    }
    const auto & sensor = RB10Model::sensor_control_points[k];
    const Eigen::Vector3d normal =
      (context.link_rotations[sensor.parent_link] * sensor.local_normal).normalized();
    const Eigen::Vector3d position = context.control_points[k].position;

    const Eigen::Vector3d to_goal = goal - position;
    const double to_goal_norm = to_goal.norm();

    // Where the goal-driven policy accelerates this very point once every
    // obstacle is removed.  The obstacles have to be removed: a local minimum
    // is target pull and obstacle push cancelling, so a reference taken from
    // the full solve would collapse to zero exactly where it is needed.
    const Eigen::Vector3d nominal_acceleration =
      context.control_point_jacobians[k] * nominal_qdd +
      context.control_point_curvatures[k];
    const double nominal_norm = nominal_acceleration.norm();

    output.sensor_clearance[k] = sensor_clearance;
    output.sensor_dot_goal_tcp[k] =
      to_goal_norm > 1e-9 ? normal.dot(to_goal) / to_goal_norm : 0.0;
    output.sensor_dot_goal_nominal[k] =
      nominal_norm > std::max(params.nominal_motion_epsilon, 1e-12) ?
      normal.dot(nominal_acceleration) / nominal_norm :
      0.0;
    output.sensor_nominal_magnitude[k] = nominal_norm;
    ++output.observation_count;
  }
  output.observation_valid = output.observation_count > 0;
  if (capture) {
    decision.closest = static_cast<int>(closest);
    decision.closest_clearance = closest_clearance;
  }
  output.alpha_distance = distance_activation(
    closest_clearance, params.alpha_on_distance, params.alpha_off_distance);

  // Past alpha_off the activation is identically zero, so the metric is too and
  // the leaf has no say whatever its internal state says.  That is what lets
  // the state be dropped right here with no discontinuity in the command, and
  // it is why the old direction-memory timer is gone: the memory existed only
  // to stop a hard release edge from turning the band into a relaxation
  // oscillator, and there is no edge left to oscillate on.  The overshoot is
  // just to keep the leaf from re-seeding every cycle while it sits on the
  // boundary; the metric is zero across it either way.
  const double release = std::max(params.alpha_off_distance, params.alpha_on_distance) +
    std::max(params.owner_hysteresis, 0.0);
  if (closest_clearance > release) {
    if (capture) { decision.events |= 32; }
    reset();
    return output;
  }

  // Sensor hand-off keeps the current owner until another sensor is clearly
  // closer, so a near-tie does not chatter between two ring neighbours.
  std::size_t owner = state.active ? state.sensor_index : closest;
  if (!state.active || owner >= point_count || !neighbours_table[owner].valid) {
    owner = closest;
  }
  const double owner_clearance = clearances[owner] - params.clearance_margin;
  bool switch_owner =
    !state.active ||
    !std::isfinite(owner_clearance) ||
    (progress_block && have_blocker && !eligible(owner)) ||
    (owner_clearance - closest_clearance) > std::max(params.owner_hysteresis, 0.0);
  if (params.freeze_owner && state.active) { switch_owner = false; }
  if (capture) {
    decision.owner_clearance = owner_clearance;
    if (state.active && switch_owner && owner != closest) { decision.events |= 2; }
  }

  // No metric fade on ownership changes. Retain the serialized field at one
  // for old trace readers; switching still uses distance hysteresis above.
  state.handoff_gain = 1.0;

  // --- sensor frame --------------------------------------------------------
  const auto sensor_basis = [&context](std::size_t index) {
      const auto & sensor = RB10Model::sensor_control_points[index];
      const Eigen::Matrix3d & rotation = context.link_rotations[sensor.parent_link];
      const Eigen::Vector3d normal = (rotation * sensor.local_normal).normalized();
      const Eigen::Vector3d axis1 = (rotation * sensor.local_tangent_bias).normalized();
      // e2 = o x e1 is orthonormal by construction, so t . o == 0 exactly and
      // no re-projection is ever needed.
      const Eigen::Vector3d axis2 = normal.cross(axis1);
      return std::array<Eigen::Vector3d, 3>{normal, axis1, axis2};
    };

  if (switch_owner) {
    if (adaptive && state.active && owner != closest) { state.theta_valid = false; }
    if (capture && state.active && owner != closest) { decision.events |= 4; }
    if (state.active && state.sensor_index < point_count &&
      neighbours_table[state.sensor_index].valid)
    {
      // Carry the direction across by re-expressing the old world tangent in
      // the new sensor basis; the axial component is preserved exactly.
      const auto old_basis = sensor_basis(state.sensor_index);
      const Eigen::Vector3d old_tangent =
        std::cos(state.theta) * old_basis[1] + std::sin(state.theta) * old_basis[2];
      const auto new_basis = sensor_basis(closest);
      const double x = new_basis[1].dot(old_tangent);
      const double y = new_basis[2].dot(old_tangent);
      if (capture && std::hypot(x, y) <= 1e-6) { decision.events |= 128; }
      state.theta = legacy || std::hypot(x, y) > 1e-6 ?
        std::atan2(y, x) : params.history.initial_angle;
      if (!legacy) {
        state.handoff_gain = 1.0;
        state.filtered_theta_rate = 0.0;
      }
    }
    owner = closest;
  }

  const std::size_t index = owner;
  const auto & neighbours = neighbours_table[index];
  const auto basis = sensor_basis(index);
  const Eigen::Vector3d & obstacle_direction = basis[0];
  const Eigen::Vector3d & axis1 = basis[1];
  const Eigen::Vector3d & axis2 = basis[2];

  const double clearance = std::max(clearances[index] - params.clearance_margin, 0.0);
  // An absent or non-finite rate reads as zero, which makes the closing-rate
  // term vanish instead of firing on a number nobody supplied.
  const double clearance_rate =
    index < clearance_rates.size() && std::isfinite(clearance_rates[index]) ?
    clearance_rates[index] :
    0.0;

  // --- neighbour risk, reduced to tangent-plane angles ---------------------
  // Each neighbour argues against the direction its own normal points at, as a
  // dipole plus a quadrupole:
  //     S(theta) -= w_j [ cos(theta - phi_j) + cos(2(theta - phi_j)) ]
  // Expanding gives the a, b, c, e coefficients below.  On a regular cross
  // ring (phi in {0, +-90, 180}) this collapses exactly to the hand-written
  // (next, previous, antipodal) form, and e vanishes.
  double risk_a = 0.0;
  double risk_b = 0.0;
  double risk_c = 0.0;
  double risk_e = 0.0;
  double axial_block_risk = 0.0;

  for (std::size_t slot = 0; slot < neighbours.count; ++slot) {
    const auto & contribution = neighbours.entries[slot];
    if (contribution.index >= clearances.size()) {
      continue;
    }
    const double neighbour_clearance =
      clearances[contribution.index] - params.clearance_margin;
    const double weight = risk_weight(neighbour_clearance, params.risk_length);

    if (contribution.axial_block) {
      axial_block_risk += weight;
      continue;
    }
    const double scaled = weight * contribution.weight;
    risk_a -= scaled * std::cos(contribution.phi);
    risk_b -= scaled * std::sin(contribution.phi);
    risk_c -= scaled * std::cos(2.0 * contribution.phi);
    risk_e -= scaled * std::sin(2.0 * contribution.phi);
  }

  // --- score coefficients --------------------------------------------------
  const Eigen::Matrix<double, 3, 6> & point_jacobian =
    context.control_point_jacobians[index];
  const Eigen::Vector3d velocity = context.control_point_velocities[index];
  const Eigen::Vector3d to_goal = goal - context.control_points[index].position;

  // Where the goal-driven policy accelerates THIS control point once every
  // obstacle is removed. Retained for diagnostics and non-history modes.
  //
  // The obstacles have to be removed: a local minimum is target pull and
  // obstacle push cancelling, so a reference taken from the full solve would
  // collapse to zero at exactly the configuration this leaf exists for.
  const Eigen::Vector3d nominal_acceleration =
    point_jacobian * nominal_qdd + context.control_point_curvatures[index];
  const double nominal_norm = nominal_acceleration.norm();
  const bool nominal_usable =
    nominal_norm > std::max(params.nominal_motion_epsilon, 1e-12);

  // The goal term used to be the straight line from this sensor to the TCP
  // goal.  For a link-2 or link-3 point that is a place it is never asked to
  // reach, and measuring both on one run showed the difference is not
  // academic: the two disagree in sign on 52% of the cycles tof3_1_E owns the
  // leaf and on all 139 that tof6_1_L owns it.  The disagreement is per
  // sensor, not per link -- averaging over a link hides it.
  //
  // Near the goal the nominal acceleration vanishes and its direction is
  // noise; the TCP line is the fallback there, being at least well defined.
  Eigen::Vector3d goal_direction =
    nominal_usable ?
    Eigen::Vector3d(nominal_acceleration / nominal_norm) :
    (to_goal.norm() > 1e-9 ?
    Eigen::Vector3d(to_goal.normalized()) :
    Eigen::Vector3d::Zero());
  if (adaptive) {
    // Direction preference must describe goal progress, not the braking
    // acceleration in nominal_qdd. Map the TCP position error through a
    // damped inverse to joint motion, then to this sensor's motion. In
    // particular, do not aim a link sensor directly at the TCP goal.
    const Eigen::Vector3d progress = point_jacobian * goal_progress;
    goal_direction.setZero();
    if (progress.norm() > 1e-9 && progress.allFinite()) {
      goal_direction = progress.normalized();
    }
  }
  const Eigen::Vector3d velocity_direction =
    velocity.norm() > std::max(params.velocity_epsilon, 1e-12) ?
    Eigen::Vector3d(velocity.normalized()) :
    Eigen::Vector3d::Zero();


  // Activation does not depend on the selected tangent. Track observations
  // and ownership while off, but defer optimization until authority is positive.
  // alpha_free is a compatibility diagnostic only. Neighbour detections
  // constrain candidate scores, never the amplitude of every tangent.
  const double alpha_free = 1.0;

  // --- blockage gate -------------------------------------------------------
  // History asks whether goal progress moves this sensor toward its mounting
  // axis. Braking is not evidence that the goal route is obstructed. Keep the
  // existing nominal-acceleration rule in the other experimental modes.
  const Eigen::Vector3d block_reference = progress_block ?
    Eigen::Vector3d(point_jacobian * goal_progress) : nominal_acceleration;
  const double block_norm = block_reference.norm();
  const bool block_usable = block_reference.allFinite() &&
    block_norm > std::max(params.nominal_motion_epsilon, 1e-12);
  output.block_reference = block_reference;
  output.block_reference_valid = block_usable;
  const double blockage = block_usable ?
    obstacle_direction.dot(block_reference) / block_norm :
    0.0;
  double alpha_block = progress_block ? 0.0 : 1.0;
  if (block_usable) {
    if (progress_block) {
      const double start_angle = params.block_start_angle_deg;
      if (std::isfinite(start_angle) && start_angle > 0.0 && start_angle <= 180.0) {
        const double angle_deg = std::acos(std::clamp(blockage, -1.0, 1.0)) * 180.0 / kPi;
        alpha_block = std::clamp(1.0 - angle_deg / start_angle, 0.0, 1.0);
      }
    } else {
      alpha_block = smoothstep(blockage, params.block_on_dot, params.block_off_dot);
    }
  }

  // From the owner, which the hand-off hysteresis may hold a little further
  // out than the closest sensor.  The same distance drives the push and the
  // metric, so the whole leaf has to speak about one sensor.
  const double alpha_distance = distance_activation(
    clearance, params.alpha_on_distance, params.alpha_off_distance);
  const double alpha_total = alpha_distance * alpha_block;

  const double raw_metric_gate = std::min(
    2.0 * (1.0 - sigmoid(clearance_rate / std::max(params.metric_rate_scale, 1e-9))),
    1.0);
  // Preserve detour authority on a transient opening, without bypassing block
  // or distance release. Frozen history trials preserve the same floor.
  const double metric_gate = (adaptive || params.goal_progress_blockage) ?
    std::max(raw_metric_gate, std::clamp(params.history_metric_rate_floor, 0.0, 1.0)) : raw_metric_gate;
  const double metric_base =
    alpha_total * metric_gate *
    std::max(params.metric_mu, 0.0) /
    (clearance / std::max(params.metric_length, 1e-9) +
    std::max(params.metric_epsilon, 1e-9));

  if (adaptive && !params.freeze_owner) {
    const auto & feedback=params.response_input;
    const bool fresh=feedback.encoder_stamp>=0 && feedback.stamp>=feedback.encoder_stamp &&
      feedback.stamp-feedback.encoder_stamp<=.05 && feedback.command_available &&
      feedback.command_qd.allFinite() && feedback.stamp>=feedback.command_stamp &&
      feedback.stamp-feedback.command_stamp<=.05;
    if (!fresh || !state.response.seeded || state.response.owner!=index ||
      (state.response.goal-goal).norm()>1e-6 || alpha_block<=.1 || !(metric_base>1e-9))
    {
      state.response.authority.reset();
    } else {
      state.response.authority.advance(dt,params.authority_boost);
    }
  }
  // Frozen history candidates use the same multiplier, without updating it.
  // Multiplying both M and f below preserves the canonical TG acceleration.
  const double authority_gain=params.authority_boost.enabled &&
    (adaptive || params.goal_progress_blockage) ? state.response.authority.gain : 1.0;
  const double metric=metric_base*authority_gain;
  output.metric_base=metric_base;
  output.authority_gain=authority_gain;

  if (adaptive && !(metric > 1e-9)) {
    state.response.pause();
    state.active = true;
    state.sensor_index = index;
    state.filtered_theta_rate = 0.0;
    output.active = true;
    output.sensor_index = index;
    output.active_link = RB10Model::sensor_control_points[index].parent_link;
    output.clearance = clearance;
    output.alpha_distance = alpha_distance;
    output.alpha_free = alpha_free;
    output.alpha_block = alpha_block;
    output.alpha_total = alpha_total;
    output.metric_gate = metric_gate;
    output.blockage = blockage;
    output.obstacle_direction = obstacle_direction;
    output.goal_e1 = axis1.dot(goal_direction);
    output.goal_e2 = axis2.dot(goal_direction);
    output.goal_from_nominal = false;
    output.theta = state.theta;
    output.control_point = context.control_points[index].position;
    output.tcp_position = context.tcp_position;
    output.goal_distance = (goal - context.tcp_position).norm();
    output.clearance_rate = clearance_rate;
    if (state.theta_valid) {
      output.basis_e1 = axis1;
      output.basis_e2 = axis2;
      output.tangent = std::cos(state.theta) * axis1 + std::sin(state.theta) * axis2;
    }
    if (capture) {
      decision.theta_evaluated = state.theta;
      decision.filtered_rate_before = 0.0;
    }
    return output;
  }

  // --- I1: whole-body kinematic consequence --------------------------------
  // Local risk asks what the obstacles look like around this sensor.  I1 asks
  // a different question: if the leaf pushes the owner along t, where does the
  // rest of the arm go, and does any of it go into something.
  //
  //   qdot_theta = Ji# t(theta)          the joint motion that would realise t
  //   v_k        = Jk qdot_theta         what that does to control point k
  //   z_k        = o_k . v_k             ...toward k's own obstacle, or away
  //   S_I1       = -w_I sum_k rho_k z_k  penalise sending a body part inward
  //
  // z_k is linear in t(theta), so S_I1 is a pure dipole and folds into a and b
  // with no new harmonic.  Expanding t = cos e1 + sin e2 gives the two
  // coefficients directly.
  //
  // Ji# t is the minimum-norm joint velocity that realises t, not the motion
  // the robot will actually make -- that comes out of the whole root solve.
  // I1 is therefore a score, not a prediction, and it only has to rank
  // candidate directions against each other.
  //
  // Cost: one 3x3 solve and two 6-vectors for the owner, then a single 6x3
  // product and two dot products per body sensor.  Gki is never formed.
  double score_a_i1 = 0.0;
  double score_b_i1 = 0.0;
  std::size_t i1_count = 0;
  double i1_largest_gain = 0.0;
  if (legacy) {
    const double damping = std::max(params.i1_damping, 1e-6);
    const Eigen::Matrix3d gram =
      point_jacobian * point_jacobian.transpose() +
      damping * damping * Eigen::Matrix3d::Identity();
    // Ji# = Ji^T (Ji Ji^T + lambda^2 I)^-1, applied to the two basis tangents.
    const Eigen::Matrix<double, 6, 3> damped_pseudoinverse =
      point_jacobian.transpose() * gram.inverse();
    const JointVector joint_motion_e1 = damped_pseudoinverse * axis1;
    const JointVector joint_motion_e2 = damped_pseudoinverse * axis2;

    const std::size_t owner_link =
      RB10Model::sensor_control_points[index].parent_link;
    for (std::size_t k = 0; k < point_count; ++k) {
      if (k == index) {
        continue;
      }
      // The owner's own ring is what local risk already speaks for.  Leaving
      // it out is what keeps the two terms answering different questions
      // instead of double-counting the same neighbours.
      bool in_owner_group = false;
      for (std::size_t slot = 0; slot < neighbours.count; ++slot) {
        if (neighbours.entries[slot].index == k) {
          in_owner_group = true;
          break;
        }
      }
      if (in_owner_group) {
        continue;
      }
      const auto & body_sensor = RB10Model::sensor_control_points[k];
      if (params.i1_exclude_parent_link && body_sensor.parent_link == owner_link) {
        continue;
      }
      const double body_clearance = clearances[k] - params.clearance_margin;
      if (!std::isfinite(body_clearance)) {
        continue;
      }
      const double proximity = risk_weight(body_clearance, params.i1_length);
      if (proximity <= 0.0) {
        continue;
      }
      const Eigen::Vector3d body_normal =
        (context.link_rotations[body_sensor.parent_link] *
        body_sensor.local_normal).normalized();
      // o_k^T Jk Ji# e = (Jk^T o_k) . (Ji# e), so Gki never has to be built.
      const JointVector body_row =
        context.control_point_jacobians[k].transpose() * body_normal;
      const double gain_e1 = body_row.dot(joint_motion_e1);
      const double gain_e2 = body_row.dot(joint_motion_e2);
      score_a_i1 -= proximity * gain_e1;
      score_b_i1 -= proximity * gain_e2;
      i1_largest_gain =
        std::max(i1_largest_gain, std::hypot(gain_e1, gain_e2));
      ++i1_count;
    }
    score_a_i1 *= params.weight_i1;
    score_b_i1 *= params.weight_i1;
  }

  // A sensor whose e1 carries no axial component has no meaningful "toward the
  // distal end", so it takes no axial preference from a blocked far side.
  const double axial_sign = neighbours.axial_preference_defined ?
    neighbours.axial_sign * (params.axial_preference >= 0.0 ? 1.0 : -1.0) :
    0.0;

  const double goal_weight = legacy ? params.weight_goal :
    (adaptive ? params.history.weight_goal : 0.0);
  const double continuity_weight = legacy ? params.weight_continuity :
    (adaptive ? params.history.weight_continuity : 0.0);
  const double score_a_goal = goal_weight * axis1.dot(goal_direction);
  const double score_b_goal = goal_weight * axis2.dot(goal_direction);
  const double score_a_velocity =
    continuity_weight * axis1.dot(velocity_direction);
  const double score_b_velocity =
    continuity_weight * axis2.dot(velocity_direction);
  const double score_a_risk =
    legacy ? params.weight_risk * risk_a +
    params.weight_axial_block * axial_sign * axial_block_risk : 0.0;
  const double score_b_risk = legacy ? params.weight_risk * risk_b : 0.0;

  // Signed sensor-frame biases provide explicit preferences along either
  // tangent basis axis without overloading the data-dependent goal/risk weights.
  double score_a =
    score_a_goal + score_a_velocity + score_a_risk + score_a_i1 +
    (legacy ? params.e1_bias : 0.0);
  double score_b =
    score_b_goal + score_b_velocity + score_b_risk + score_b_i1 +
    (legacy ? params.e2_bias : 0.0);
  // The axial block now ADDS to c.  c cos(2 theta) is +c along e1 and -c along
  // e2, so preferring the link axis -- which is what a blocked far side is
  // supposed to argue for, and what the dipole term above already does with
  // the same signal -- needs c positive.  Subtracting sent it the other way,
  // to the lateral axis, so the two uses of one signal disagreed about which
  // axis to leave along.
  //
  // It was not a small disagreement.  In a run that locked up, c came to
  // -0.580 with -0.407 of that from the antipodal sensor alone: 70% of the
  // axis decision was made by the one neighbour that carries no tangent-plane
  // direction at all.  The lateral axis it picked happened to be vertical in
  // that pose, which is why the arm kept escaping downwards instead of round.
  const double score_c =
    legacy ? params.weight_axis * risk_c +
    params.weight_axial_block * axial_block_risk : 0.0;
  const double score_e = legacy ? params.weight_axis * risk_e : 0.0;

  // Keep every existing score term intact. The new one-sided lookahead cost
  // need not be a dipole, so evaluate it separately in both the initial
  // candidate ranking and the continuous direction ODE.
  const double prediction_weight = legacy && std::isfinite(params.weight_predicted_risk) ?
    std::max(params.weight_predicted_risk, 0.0) : 0.0;
  tangent_escape_prediction::StaticObstaclePredictor prediction;
  if (prediction_weight > 0.0) {
    prediction = tangent_escape_prediction::StaticObstaclePredictor(
      context, index, axis1, axis2, prediction_obstacles,
      params.prediction_distance, params.prediction_risk_length,
      params.prediction_damping, params.clearance_margin);
  }

  // --- direction state -----------------------------------------------------
  if (adaptive ? !state.theta_valid : !state.active) {
    if (capture) { decision.events |= 1; }
    // The gradient vanishes at minima as well as maxima, so seed the ODE from
    // a coarse maximum instead of letting it settle into the wrong stationary
    // point. A sensor switch may re-seed at nonzero metric; no zero-authority
    // transition is assumed, and the existing joint command limits remain.
    double best_score = -std::numeric_limits<double>::infinity();
    double best_theta = legacy ? 0.0 : params.history.initial_angle;
    for (int seed = 0; seed < (legacy || adaptive ? 8 : 0); ++seed) {
      const double candidate = (legacy ? 0.0 : params.history.initial_angle) +
        2.0 * kPi * static_cast<double>(seed) / 8.0;
      const auto candidate_risk = adaptive && objective ? objective(candidate, index) :
        tangent_sensor_history::Risk{};
      const double history_cost = params.history.weight_risk * candidate_risk.value;
      const double escape_score = adaptive ? params.history.weight_escape *
        alpha_block * candidate_risk.escape : 0.0;
      double score =
        direction_score(candidate, score_a, score_b, score_c, score_e) -
        prediction_weight * prediction.evaluate(candidate).value -
        history_cost + escape_score;
      const auto unified = adaptive && params.response_mobility ?
        params.response_mobility(candidate,index) : TgDirectionCandidate{};
      if(unified.valid)score=unified.total();
      if (capture) {
        decision.seed_escape.push_back(unified.valid?unified.execution:escape_score);
        decision.seeds.push_back({candidate,
          unified.valid?unified.goal:score_a_goal * std::cos(candidate) + score_b_goal * std::sin(candidate),
          unified.valid?0.0:score_a_velocity * std::cos(candidate) + score_b_velocity * std::sin(candidate),
          unified.valid?-unified.risk:-history_cost, score});
      }
      if (score > best_score + (legacy ? 0.0 : 1e-9)) {
        best_score = score;
        best_theta = candidate;
      }
    }
    if (!state.theta_valid) {
      if (capture) { decision.events |= 256; }
      state.theta = best_theta;
      state.progress = 0.0;
    }
    state.active = true;
    state.theta_valid = true;
  }
  state.sensor_index = index;

  // Stall: the leaf is commanding a push but the control point is not making
  // tangential progress.  Uses the previous cycle's tangent rate, which is the
  // motion the previous command actually produced.
  // The distance-driven push supplies the legacy stall reference.
  const double commanded_push =
    std::max(params.accel_gain, 0.0) *
    std::exp(-clearance / std::max(params.accel_length, 1e-9));
  const double commanding = smoothstep(
    commanded_push, 0.0, std::max(params.accel_gain, 1e-9) * 0.05);
  const double moving = smoothstep(
    std::abs(state.previous_tangent_rate), 0.0,
    std::max(params.stall_speed, 1e-9));
  const double stall = std::clamp(commanding * (1.0 - moving), 0.0, 1.0);

  // Accumulate the directions visited while stalled; the accumulator decays
  // once the leaf is moving again, so a solved obstacle is forgotten.
  const double stall_alpha =
    std::clamp(dt / std::max(params.stall_time_constant, dt), 0.0, 1.0);
  state.stall_x += stall_alpha * (stall * std::cos(state.theta) - state.stall_x);
  state.stall_y += stall_alpha * (stall * std::sin(state.theta) - state.stall_y);
  const double stall_penalty = std::hypot(state.stall_x, state.stall_y);

  // Subtract the dipole: same algebra as the neighbour risk terms.
  const double score_a_stall = legacy ? std::max(params.weight_stall, 0.0) * state.stall_x : 0.0;
  const double score_b_stall = legacy ? std::max(params.weight_stall, 0.0) * state.stall_y : 0.0;
  score_a -= score_a_stall;
  score_b -= score_b_stall;

  const auto update_risk = adaptive && objective ? objective(state.theta, index) :
    tangent_sensor_history::Risk{};
  const double total_gradient =
    (direction_score_gradient(state.theta, score_a, score_b, score_c, score_e) -
    prediction_weight * prediction.evaluate(state.theta).gradient -
    (adaptive && objective ? params.history.weight_risk * update_risk.gradient : 0.0));
  double ascent_gradient = total_gradient;
  if (adaptive) {
    const auto score_at = [&](double angle) {
      return params.response_mobility ? params.response_mobility(angle,index) :
        TgDirectionCandidate{};
    };
    const Eigen::Vector3d actual_tangent = std::cos(state.theta)*axis1+std::sin(state.theta)*axis2;
    const double desired = state.response.update(state.theta,state.filtered_theta_rate,index,goal,
      actual_tangent,context.control_point_jacobians[index],params.response_input,alpha_block,score_at,metric,context.tcp_jacobian,
      (goal-context.tcp_position)/std::max((goal-context.tcp_position).norm(),1e-9),
      params.history.weight_continuity,params.history.reselect_interval,params.history.max_direction_rate,
      params.authority_boost);
    ascent_gradient = 5.0*std::remainder(desired-state.theta,2*kPi);
    // A worsening range response changes steering; only missing/invalid
    // observations freeze it. Collision RMP independently supplies repulsion.
    if (state.response.phase==5) {
      state.filtered_theta_rate=0.0;ascent_gradient=0.0;
    }
    if (!state.response.seeded || (goal-context.tcp_position).norm()<.01) {
      state.response.pause();state.filtered_theta_rate=0.0;ascent_gradient=0.0;
    }
  }
  const double escape_gradient = ascent_gradient-total_gradient;
  const double probe_gradient = 0.0;
  double theta_rate = std::max(params.direction_rate, 0.0) * ascent_gradient;
  if (capture) {
    decision.theta_evaluated = state.theta;
    // The handoff may have reset the filter since entry to evaluate().
    decision.filtered_rate_before = state.filtered_theta_rate;
    decision.goal_gradient = -score_a_goal * std::sin(state.theta) + score_b_goal * std::cos(state.theta);
    decision.continuity_gradient = -score_a_velocity * std::sin(state.theta) + score_b_velocity * std::cos(state.theta);
    decision.history_gradient = -params.history.weight_risk * update_risk.gradient;
    decision.escape_gradient = escape_gradient + probe_gradient;
    decision.total_gradient = ascent_gradient;
    decision.raw_rate = theta_rate;
    decision.risk_before = update_risk;
  }
  if (!legacy) {
    const double limit = std::max(params.history.max_direction_rate, 0.0);
    const double target = std::clamp(theta_rate, -limit, limit);
    if (capture) { decision.clamped_rate = target; }
    state.filtered_theta_rate += dt / (std::max(params.history.direction_filter_tau, 0.0) + dt) *
      (target - state.filtered_theta_rate);
    theta_rate = adaptive ? state.filtered_theta_rate : 0.0;
  }
  state.response.bound_clamped=false; // v5 compatibility; full-circle steering
  state.theta += theta_rate * dt;

  if (!std::isfinite(state.theta)) {
    state.theta = 0.0;
  }
  state.theta = std::remainder(state.theta, 2.0 * kPi);

  const Eigen::Vector3d tangent =
    std::cos(state.theta) * axis1 + std::sin(state.theta) * axis2;
  const Eigen::Vector3d tangent_perpendicular =
    -std::sin(state.theta) * axis1 + std::cos(state.theta) * axis2;

  // --- leaf coordinate rate ------------------------------------------------
  const RowVector6 leaf_jacobian = tangent.transpose() * point_jacobian;
  const double tangent_rate = leaf_jacobian * qd;

  state.previous_tangent_rate = tangent_rate;

  // --- progress --------------------------------------------------------------
  // Only forward progress spends the progress; being pushed backwards does not
  // refund it.  The slow recovery term lets a new obstacle episode start with
  // authority again without ever making the well infinitely deep.
  const double recovery = state.progress /
    std::max(params.progress_recovery_time, 1e-6);
  state.progress += (std::max(0.0, tangent_rate) - recovery) * dt;
  state.progress = std::max(state.progress, 0.0);
  const double progress_scale = std::max(params.progress_scale, 1e-6);
  const double progress_remaining = std::exp(-state.progress / progress_scale);

  // --- Jdot_e * qdot, analytic ---------------------------------------------
  // tdot = theta_dot * t_perp + omega x t.  The direction only changes because
  // the ODE turns it or because the link rotates; both are closed form, so no
  // numerical differentiation is involved.
  const auto & sensor = RB10Model::sensor_control_points[index];
  const Eigen::Vector3d angular_velocity =
    context.link_angular_jacobians[sensor.parent_link] * qd;
  const Eigen::Vector3d tangent_derivative =
    theta_rate * tangent_perpendicular + angular_velocity.cross(tangent);
  const double jdot_theta_term =
    (theta_rate * tangent_perpendicular).dot(velocity);
  const double jdot_omega_term =
    angular_velocity.cross(tangent).dot(velocity);
  const double jdot_qdot =
    tangent_derivative.dot(velocity) +
    tangent.dot(context.control_point_curvatures[index]);

  // --- canonical form ------------------------------------------------------
  // The leaf is designed as (a, M) and only converted to (f, M) for the
  // pullback.  Designing f directly was a mistake: with the force gate wider
  // than the metric gate, the implied desired acceleration f/m grows like
  // exp(+d (1/sigma_m - 1/sigma_a)) and blows up with distance -- a meaningless
  // policy, and a 0/0 as the metric vanishes.  Gating the acceleration instead
  // keeps a bounded by accel_gain everywhere while the metric stays narrow.
  // No activation factor here.  f = M a, so gating both the acceleration and
  // the metric puts alpha into the transported force twice; worse, the damping
  // term is not gated at all, so as alpha shrank the leaf's desired
  // acceleration collapsed onto -b_e ydot and it braked tangential motion
  // instead of yielding.  Activation belongs on the metric alone: it decides
  // how much authority the leaf has, not what it is asking for.
  const double distance_push =
    std::max(params.accel_gain, 0.0) *
    std::exp(-clearance / std::max(params.accel_length, 1e-9)) *
    (params.enable_progress_attenuation ? progress_remaining : 1.0);

  // Closing-rate term.  gamma_v is a soft rectifier on the closing direction,
  // and the 1/(d/l + eps) factor makes the same closing speed count for more
  // the nearer it happens; eps is what keeps it finite at contact.
  const double closing_gate =
    1.0 - sigmoid(clearance_rate / std::max(params.closing_rate_scale, 1e-9));
  const double closing_shape = std::max(
    -closing_gate * clearance_rate /
    (clearance / std::max(params.closing_length, 1e-9) +
    std::max(params.closing_epsilon, 1e-9)),
    0.0);
  const double closing_term = std::max(params.closing_gain, 0.0) * closing_shape;

  // Saturation covers the push only.  The damping below has to stay outside
  // it: clamping the sum to [0, a_max] would delete every negative value the
  // leaf can produce, and -b_e ydot is the only dissipation it has left now
  // that the progress attenuation is off.  A leaf that can never decelerate
  // tangential motion can inject energy along y without bound.
  const double push_acceleration = std::clamp(
    distance_push + closing_term, 0.0, std::max(params.accel_saturation, 0.0));
  // b_e is a rate here, not a force gain: 1/b_e is the tangential settling
  // time and must stay well above the control period.
  const double damping_acceleration =
    std::max(params.tangent_damping, 0.0) * tangent_rate;
  const double desired_acceleration = push_acceleration - damping_acceleration;
  // Metric.  Hyperbolic in d rather than exponential, with its own closing-rate
  // gate, so that authority tracks both how close the obstacle is and whether
  // the point is actually driving into it.
  //
  //   m = alpha_TG gamma_m(ddot) mu / (d / l_m + eps_m)
  //
  // The distance shaping is deliberately not repeated: alpha_d already decides
  // the range over which the leaf exists at all, and this factor only grades
  // proximity inside it.  eps_m is what bounds m at contact -- the peak is
  // mu / eps_m, not mu.
  //
  // The tail is far fatter than the old exponential, which is the reason the
  // Schmitt trigger had to become alpha_d first: this metric does not vanish
  // on its own anywhere, so without a smooth gate it would step to zero at the
  // release edge.
  // gamma_m, rescaled so that standing still counts as full authority.
  //
  // The plain 1 - sigmoid(ddot/v_m) is exactly 0.5 at ddot = 0, and ddot = 0 is
  // where this leaf earns its keep: a local minimum is a robot that has stopped
  // against an obstacle.  Measured, ddot's median over a run was 0, so the
  // metric ran at half strength essentially all the time -- the same failure
  // the velocity activation gate was deleted for.  Doubling and clamping keeps
  // the intent, which is that RECEDING costs authority, and drops the
  // side effect, which was that being stationary did too.
  // f = M a is the representation change the tree transports.
  const double natural_force = metric * desired_acceleration;

  // --- pullback ------------------------------------------------------------
  const Matrix6 metric_contribution =
    metric * leaf_jacobian.transpose() * leaf_jacobian;
  const JointVector force_contribution =
    leaf_jacobian.transpose() * (natural_force - metric * jdot_qdot);

  output.active = true;
  output.sensor_index = index;
  output.clearance = clearance;
  output.active_link = RB10Model::sensor_control_points[index].parent_link;
  output.alpha_distance = alpha_distance;
  output.alpha_block = alpha_block;
  output.alpha_total = alpha_total;
  output.blockage = blockage;
  output.nominal_magnitude = nominal_norm;
  output.obstacle_direction = obstacle_direction;
  output.tangent = tangent;
  output.tangent_perpendicular = tangent_perpendicular;
  output.theta = state.theta;
  output.theta_rate = theta_rate;
  output.progress = state.progress;
  output.progress_remaining = progress_remaining;
  output.tangent_rate = tangent_rate;
  output.score_a = score_a;
  output.score_b = score_b;
  output.score_c = score_c;
  output.score_e = score_e;
  output.stall = stall;
  output.stall_penalty = stall_penalty;
  output.score_a_goal = score_a_goal;
  output.score_b_goal = score_b_goal;
  output.score_a_velocity = score_a_velocity;
  output.score_b_velocity = score_b_velocity;
  output.score_a_risk = score_a_risk;
  output.score_b_risk = score_b_risk;
  output.score_a_stall = score_a_stall;
  output.score_b_stall = score_b_stall;
  output.score_a_i1 = score_a_i1;
  output.goal_e1 = axis1.dot(goal_direction);
  output.goal_e2 = axis2.dot(goal_direction);
  output.goal_from_nominal = !adaptive && nominal_usable;
  output.score_b_i1 = score_b_i1;
  output.i1_count = i1_count;
  output.i1_largest_gain = i1_largest_gain;
  output.predicted_risk = prediction.evaluate(state.theta);
  output.predicted_risk_score = -prediction_weight * output.predicted_risk.value;
  output.total_direction_score =
    direction_score(state.theta, score_a, score_b, score_c, score_e) +
    output.predicted_risk_score;
  if (adaptive && objective) {
    output.history_risk = objective(state.theta, index);
    output.history_risk_score = -params.history.weight_risk * output.history_risk.value;
    output.total_direction_score += output.history_risk_score +
      params.history.weight_escape * output.alpha_block * output.history_risk.escape;
  }
  output.handoff_gain = state.handoff_gain;
  output.clearance_rate = clearance_rate;
  output.closing_gate = closing_gate;
  output.closing_term = closing_term;
  output.closing_shape = closing_shape;
  output.push_acceleration = push_acceleration;
  output.damping_acceleration = damping_acceleration;
  output.desired_acceleration = desired_acceleration;
  output.natural_force = natural_force;
  output.metric = metric;
  output.metric_gate = metric_gate;
  output.alpha_free = alpha_free;
  output.injected_power = natural_force * tangent_rate;
  output.jdot_qdot = jdot_qdot;
  output.jdot_theta_term = jdot_theta_term;
  output.jdot_omega_term = jdot_omega_term;
  output.tangent_normal_residual = std::abs(tangent.dot(obstacle_direction));
  output.control_point = context.control_points[index].position;
  output.tcp_position = context.tcp_position;
  output.basis_e1 = axis1;
  output.basis_e2 = axis2;
  output.goal_in_plane =
    axis1.dot(goal_direction) * axis1 + axis2.dot(goal_direction) * axis2;
  output.goal_distance = (goal - context.tcp_position).norm();
  output.leaf_jacobian = leaf_jacobian;

  if (metric_contribution.allFinite() && force_contribution.allFinite()) {
    output.metric_contribution = metric_contribution;
    output.force_contribution = force_contribution;
  } else {
    output.active = false;
    state = LeafState{};
  }

  return output;
}

}  // namespace tangent_escape_ode
}  // namespace rb10_rmpflow_rviz

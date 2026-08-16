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
  const std::vector<double> & clearance_rates)
{
  LeafOutput output;

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
    state = LeafState{};
    return output;
  }

  const double dt = std::max(params.control_dt, 1e-4);

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

  const auto reset = [&state]() {
      state = LeafState{};
    };

  if (!found) {
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
    reset();
    return output;
  }

  // Sensor hand-off keeps the current owner until another sensor is clearly
  // closer, so a near-tie does not chatter between two ring neighbours.
  std::size_t owner = state.active ? state.sensor_index : closest;
  if (!state.active || !neighbours_table[owner].valid || owner >= point_count) {
    owner = closest;
  }
  const double owner_clearance = clearances[owner] - params.clearance_margin;
  const bool switch_owner =
    !state.active ||
    !std::isfinite(owner_clearance) ||
    (owner_clearance - closest_clearance) > std::max(params.owner_hysteresis, 0.0);

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
    if (state.active && neighbours_table[state.sensor_index].valid) {
      // Carry the direction across by re-expressing the old world tangent in
      // the new sensor basis; the axial component is preserved exactly.
      const auto old_basis = sensor_basis(state.sensor_index);
      const Eigen::Vector3d old_tangent =
        std::cos(state.theta) * old_basis[1] + std::sin(state.theta) * old_basis[2];
      const auto new_basis = sensor_basis(closest);
      state.theta = std::atan2(
        new_basis[2].dot(old_tangent),
        new_basis[1].dot(old_tangent));
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
  double pocket_product = 1.0;
  std::size_t pocket_count = 0;

  for (std::size_t slot = 0; slot < neighbours.count; ++slot) {
    const auto & contribution = neighbours.entries[slot];
    if (contribution.index >= clearances.size()) {
      continue;
    }
    const double neighbour_clearance =
      clearances[contribution.index] - params.clearance_margin;
    const double weight = risk_weight(neighbour_clearance, params.risk_length);
    pocket_product *= risk_weight(neighbour_clearance, params.pocket_length);
    ++pocket_count;

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
  // obstacle is removed.  Also what the blockage gate reads further down.
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
  const Eigen::Vector3d goal_direction =
    nominal_usable ?
    Eigen::Vector3d(nominal_acceleration / nominal_norm) :
    (to_goal.norm() > 1e-9 ?
    Eigen::Vector3d(to_goal.normalized()) :
    Eigen::Vector3d::Zero());
  const Eigen::Vector3d velocity_direction =
    velocity.norm() > std::max(params.velocity_epsilon, 1e-12) ?
    Eigen::Vector3d(velocity.normalized()) :
    Eigen::Vector3d::Zero();


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
  {
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

  const double score_a_goal = params.weight_goal * axis1.dot(goal_direction);
  const double score_b_goal = params.weight_goal * axis2.dot(goal_direction);
  const double score_a_velocity =
    params.weight_continuity * axis1.dot(velocity_direction);
  const double score_b_velocity =
    params.weight_continuity * axis2.dot(velocity_direction);
  const double score_a_risk =
    params.weight_risk * risk_a +
    params.weight_axial_block * axial_sign * axial_block_risk;
  const double score_b_risk = params.weight_risk * risk_b;

  double score_a = score_a_goal + score_a_velocity + score_a_risk + score_a_i1;
  double score_b = score_b_goal + score_b_velocity + score_b_risk + score_b_i1;
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
    params.weight_axis * risk_c +
    params.weight_axial_block * axial_block_risk;
  const double score_e = params.weight_axis * risk_e;

  // --- direction state -----------------------------------------------------
  if (!state.active) {
    // The gradient vanishes at minima as well as maxima, so seed the ODE from
    // a coarse maximum instead of letting it settle into the wrong stationary
    // point.  The metric is ~0 at this instant, so the discrete jump in theta
    // has no effect on the commanded acceleration.
    double best_score = -std::numeric_limits<double>::infinity();
    double best_theta = 0.0;
    for (int seed = 0; seed < 8; ++seed) {
      const double candidate = 2.0 * kPi * static_cast<double>(seed) / 8.0;
      const double score =
        direction_score(candidate, score_a, score_b, score_c, score_e);
      if (score > best_score) {
        best_score = score;
        best_theta = candidate;
      }
    }
    if (!state.theta_valid) {
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
  // The distance-driven part of the push is already known here; the progress and
  // pocket factors only scale it down, so this is the right "is it trying"
  // signal without reordering the rest of the cycle.
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
  const double score_a_stall = std::max(params.weight_stall, 0.0) * state.stall_x;
  const double score_b_stall = std::max(params.weight_stall, 0.0) * state.stall_y;
  score_a -= score_a_stall;
  score_b -= score_b_stall;

  const double theta_rate = std::max(params.direction_rate, 0.0) *
    direction_score_gradient(state.theta, score_a, score_b, score_c, score_e);
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

  // --- pocket guard --------------------------------------------------------
  // Geometric mean over however many neighbours this sensor actually has, so
  // a three-neighbour wrist sensor and a four-neighbour ring sensor are judged
  // on the same scale.
  const double mean_risk = pocket_count > 0 ?
    std::pow(std::max(pocket_product, 0.0), 1.0 / static_cast<double>(pocket_count)) :
    0.0;
  const double alpha_free = std::clamp(
    1.0 - sigmoid((mean_risk - params.pocket_threshold) /
    std::max(params.pocket_epsilon, 1e-9)),
    0.0,
    1.0);

  // --- blockage gate -------------------------------------------------------
  // Is this control point actually being driven into its own obstacle?  The
  // reference is where the goal-driven policy accelerates this point with the
  // obstacles removed -- not the line from here to the TCP goal, which is a
  // place a link-2 or link-3 point is never asked to reach.  Measured on this
  // scenario the two disagree in sign on 52% of the cycles tof3_1_E owns the
  // leaf and on all 139 that tof6_1_L owns it.
  //
  // This is not the old velocity gate coming back.  That one died because
  // qdot -> 0 at a local minimum, so it switched the leaf off exactly where it
  // was needed.  Neither half of that applies here: this is an acceleration,
  // which the policy keeps commanding while the robot stands still, and it is
  // taken with the obstacles removed, so the target pull has nothing left to
  // cancel against.
  const double blockage = nominal_usable ?
    obstacle_direction.dot(nominal_acceleration) / nominal_norm :
    0.0;
  // Fail open.  Near the goal the nominal acceleration vanishes and its
  // direction is noise; answering "not blocked" there would reproduce the very
  // failure the velocity gate was deleted for.  When the signal cannot be
  // read, the gate does not close.
  const double alpha_block = nominal_usable ?
    smoothstep(blockage, params.block_on_dot, params.block_off_dot) :
    1.0;

  // From the owner, which the hand-off hysteresis may hold a little further
  // out than the closest sensor.  The same distance drives the push and the
  // metric, so the whole leaf has to speak about one sensor.
  const double alpha_distance = distance_activation(
    clearance, params.alpha_on_distance, params.alpha_off_distance);
  const double alpha_total = alpha_distance * alpha_block * alpha_free;

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
  const double metric_gate = std::min(
    2.0 * (1.0 - sigmoid(clearance_rate / std::max(params.metric_rate_scale, 1e-9))),
    1.0);
  const double metric =
    alpha_total * metric_gate *
    std::max(params.metric_mu, 0.0) /
    (clearance / std::max(params.metric_length, 1e-9) +
    std::max(params.metric_epsilon, 1e-9));
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
  output.goal_from_nominal = nominal_usable;
  output.score_b_i1 = score_b_i1;
  output.i1_count = i1_count;
  output.i1_largest_gain = i1_largest_gain;
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

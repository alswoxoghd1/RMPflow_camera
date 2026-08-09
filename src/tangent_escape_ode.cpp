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
constexpr std::size_t kSensorCount = RB10Model::sensor_control_points.size();
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
  const Eigen::Vector3d & goal,
  const std::vector<double> & clearances)
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

  if (!params.enabled || params.metric_max <= 0.0 || point_count == 0) {
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

  // Deactivating keeps the direction and the spent budget for a short while.
  // Wiping them immediately makes the release boundary a relaxation
  // oscillator: the leaf pushes out of the band, switches off, gets pulled
  // back in, re-seeds a fresh direction with a refunded budget, and repeats.
  const auto deactivate = [&state, &params, dt]() {
      state.active = false;
      state.inactive_time += dt;
      if (state.inactive_time > std::max(params.direction_memory_time, 0.0)) {
        state.theta_valid = false;
        state.budget = 0.0;
      }
    };

  const double release = std::max(params.release_distance, params.activation_distance);
  if (!found || closest_clearance > release) {
    deactivate();
    return output;
  }
  if (!state.active && closest_clearance > params.activation_distance) {
    // Inside the release band but not yet close enough to engage.
    deactivate();
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
    (owner_clearance - closest_clearance) > std::max(params.switch_hysteresis, 0.0);

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
    const double weight = risk_weight(
      clearances[contribution.index] - params.clearance_margin, params.risk_sigma);
    pocket_product *= weight;
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

  const Eigen::Vector3d goal_direction =
    to_goal.norm() > 1e-9 ? Eigen::Vector3d(to_goal.normalized()) : Eigen::Vector3d::Zero();
  const Eigen::Vector3d velocity_direction =
    velocity.norm() > std::max(params.velocity_epsilon, 1e-12) ?
    Eigen::Vector3d(velocity.normalized()) :
    Eigen::Vector3d::Zero();


  // A sensor whose e1 carries no axial component has no meaningful "toward the
  // distal end", so it takes no axial preference from a blocked far side.
  const double axial_sign = neighbours.axial_preference_defined ?
    neighbours.axial_sign * (params.axial_preference >= 0.0 ? 1.0 : -1.0) :
    0.0;

  double score_a =
    params.weight_goal * axis1.dot(goal_direction) +
    params.weight_velocity * axis1.dot(velocity_direction) +
    params.weight_risk * (risk_a + axial_sign * axial_block_risk);
  double score_b =
    params.weight_goal * axis2.dot(goal_direction) +
    params.weight_velocity * axis2.dot(velocity_direction) +
    params.weight_risk * risk_b;
  const double score_c =
    params.weight_curvature * (risk_c - axial_block_risk);
  const double score_e = params.weight_curvature * risk_e;

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
      state.budget = 0.0;
    }
    state.active = true;
    state.theta_valid = true;
    state.inactive_time = 0.0;
  }
  state.sensor_index = index;

  // Stall: the leaf is commanding a push but the control point is not making
  // tangential progress.  Uses the previous cycle's tangent rate, which is the
  // motion the previous command actually produced.
  // The distance-driven part of the push is already known here; the budget and
  // pocket factors only scale it down, so this is the right "is it trying"
  // signal without reordering the rest of the cycle.
  const double commanded_push =
    std::max(params.accel_max, 0.0) *
    std::exp(-clearance / std::max(params.accel_sigma, 1e-9));
  const double commanding = smoothstep(
    commanded_push, 0.0, std::max(params.accel_max, 1e-9) * 0.05);
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
  score_a -= std::max(params.weight_stall, 0.0) * state.stall_x;
  score_b -= std::max(params.weight_stall, 0.0) * state.stall_y;

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

  // --- budget --------------------------------------------------------------
  // Only forward progress spends the budget; being pushed backwards does not
  // refund it.  The slow recovery term lets a new obstacle episode start with
  // authority again without ever making the well infinitely deep.
  const double recovery = state.budget /
    std::max(params.budget_recovery_time, 1e-6);
  state.budget += (std::max(0.0, tangent_rate) - recovery) * dt;
  state.budget = std::max(state.budget, 0.0);
  const double budget_scale = std::max(params.budget_scale, 1e-6);
  const double budget_remaining = std::exp(-state.budget / budget_scale);

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
  // keeps a bounded by accel_max everywhere while the metric stays narrow.
  const double push_acceleration =
    std::max(params.accel_max, 0.0) *
    std::exp(-clearance / std::max(params.accel_sigma, 1e-9)) *
    budget_remaining * alpha_free;
  // b_e is a rate here, not a force gain: 1/b_e is the tangential settling
  // time and must stay well above the control period.
  const double damping_acceleration =
    std::max(params.tangent_damping, 0.0) * tangent_rate;
  const double desired_acceleration = push_acceleration - damping_acceleration;
  const double metric =
    std::max(params.metric_max, 0.0) *
    std::exp(-clearance / std::max(params.metric_sigma, 1e-9)) * alpha_free;
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
  output.obstacle_direction = obstacle_direction;
  output.tangent = tangent;
  output.tangent_perpendicular = tangent_perpendicular;
  output.theta = state.theta;
  output.theta_rate = theta_rate;
  output.budget = state.budget;
  output.budget_remaining = budget_remaining;
  output.tangent_rate = tangent_rate;
  output.score_a = score_a;
  output.score_b = score_b;
  output.score_c = score_c;
  output.score_e = score_e;
  output.stall = stall;
  output.stall_penalty = stall_penalty;
  output.push_acceleration = push_acceleration;
  output.damping_acceleration = damping_acceleration;
  output.desired_acceleration = desired_acceleration;
  output.natural_force = natural_force;
  output.metric = metric;
  output.alpha_free = alpha_free;
  output.injected_power = natural_force * tangent_rate;
  output.jdot_qdot = jdot_qdot;
  output.jdot_theta_term = jdot_theta_term;
  output.jdot_omega_term = jdot_omega_term;
  output.tangent_normal_residual = std::abs(tangent.dot(obstacle_direction));
  output.control_point = context.control_points[index].position;
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

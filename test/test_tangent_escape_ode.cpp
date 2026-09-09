// Escape RMP v2 leaf tests.
//
// These check the properties the design relies on rather than a trajectory:
// exact tangent/normal orthogonality, rank-one PSD pullback, the analytic
// Jdot decomposition, the finite energy progress, and the ring table derived
// from the sensor geometry.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "rb10_rmpflow_rviz/casadi_rb10_kinematics.hpp"
#include "rb10_rmpflow_rviz/rb10_model.hpp"
#include "rb10_rmpflow_rviz/tangent_escape_ode.hpp"

namespace
{

using rb10_rmpflow_rviz::CasadiRb10Kinematics;
using rb10_rmpflow_rviz::KinematicsContext;
using rb10_rmpflow_rviz::RB10Model;
namespace ode = rb10_rmpflow_rviz::tangent_escape_ode;

using JointVector = RB10Model::JointVector;
using Matrix6 = Eigen::Matrix<double, 6, 6>;

[[noreturn]] void fail(const std::string & message)
{
  throw std::runtime_error(message);
}

void require(bool condition, const std::string & message)
{
  if (!condition) {
    fail(message);
  }
}

ode::LeafParams make_params()
{
  ode::LeafParams params;
  params.enabled = true;
  params.metric_mu = 1000.0;
  params.metric_length = 0.03;
  params.accel_gain = 50.0;
  params.accel_length = 0.08;
  params.progress_scale = 0.2;
  params.progress_recovery_time = 3.0;
  params.tangent_damping = 40.0;
  params.direction_rate = 3.0;
  params.risk_length = 0.06;
  params.alpha_on_distance = 0.30;
  params.alpha_off_distance = 0.40;
  params.owner_hysteresis = 0.005;
  params.control_dt = 0.005;
  return params;
}

// Clearances default to "nothing seen"; the caller pokes in the ones it wants.
std::vector<double> clear_scene()
{
  return std::vector<double>(
    RB10Model::sensor_control_points.size(),
    std::numeric_limits<double>::infinity());
}

void test_neighborhood_table()
{
  const auto & table = ode::neighborhood_table();
  std::size_t valid = 0;
  std::size_t regular_rings = 0;
  for (std::size_t index = 0; index < table.size(); ++index) {
    const auto & entry = table[index];
    require(entry.valid, "every ToF sensor must have a usable neighbourhood");
    ++valid;
    require(entry.count >= 1, "valid neighbourhood has no neighbours");

    const auto & self = RB10Model::sensor_control_points[index];
    const Eigen::Vector3d normal = self.local_normal.normalized();
    const Eigen::Vector3d axis1 = self.local_tangent_bias.normalized();
    const Eigen::Vector3d axis2 = normal.cross(axis1);

    require(
      std::abs(normal.dot(axis1)) <= 1e-12,
      "sensor normal and tangent bias are not orthogonal");

    std::size_t axial_blocks = 0;
    std::size_t in_plane = 0;
    for (std::size_t slot = 0; slot < entry.count; ++slot) {
      const auto & contribution = entry.entries[slot];
      require(contribution.index != index, "neighbour points at itself");
      const auto & other = RB10Model::sensor_control_points[contribution.index];
      require(
        other.parent_link == self.parent_link,
        "neighbour is on a different link");
      require(
        std::abs(other.offset.z() - self.offset.z()) <= 0.10 + 1e-9,
        "neighbour is outside the station merge distance");

      const Eigen::Vector3d other_normal = other.local_normal.normalized();
      const double alignment = normal.dot(other_normal);
      if (contribution.axial_block) {
        ++axial_blocks;
        require(alignment < -0.999, "axial block neighbour is not antipodal");
      } else {
        ++in_plane;
        require(
          contribution.weight > 1e-6 && contribution.weight <= 1.0 + 1e-9,
          "in-plane neighbour weight left (0, 1]");
        const double expected = std::atan2(
          axis2.dot(other_normal), axis1.dot(other_normal));
        require(
          std::abs(std::remainder(contribution.phi - expected, 2.0 * M_PI)) <= 1e-9,
          "in-plane neighbour angle is wrong");
      }
    }

    if (entry.axial_preference_defined) {
      require(
        entry.axial_sign * self.local_tangent_bias.z() > 0.0,
        "axial sign does not orient e1 toward the distal end");
    } else {
      require(
        std::abs(self.local_tangent_bias.z()) < 1e-9,
        "axial preference marked undefined for a sensor with an axial e1");
    }

    // A regular four-sensor cross ring must still reduce to the original
    // (next, previous, antipodal) shape: one antipodal plus two lateral
    // neighbours at exactly +-90 degrees.
    if (axial_blocks == 1 && in_plane == 2) {
      bool plus = false;
      bool minus = false;
      for (std::size_t slot = 0; slot < entry.count; ++slot) {
        const auto & contribution = entry.entries[slot];
        if (contribution.axial_block) {
          continue;
        }
        const double degrees = contribution.phi * 180.0 / M_PI;
        if (std::abs(degrees - 90.0) <= 1e-6) {
          plus = true;
        }
        if (std::abs(degrees + 90.0) <= 1e-6) {
          minus = true;
        }
      }
      if (plus && minus) {
        ++regular_rings;
      }
    }
  }
  require(
    valid == RB10Model::sensor_control_points.size(),
    "not every ToF sensor is covered");
  require(
    regular_rings == 16,
    "the four cross rings no longer reduce to the +-90 / 180 pattern");
  std::cout << "  neighbourhood table: " << valid << " sensors covered, "
            << regular_rings << " on regular cross rings\n";
}

void test_score_gradient()
{
  const double a = 0.7, b = -0.4, c = 0.25, e = -0.15;
  for (double theta = -3.0; theta < 3.0; theta += 0.37) {
    const double h = 1e-6;
    const double numeric =
      (ode::direction_score(theta + h, a, b, c, e) -
      ode::direction_score(theta - h, a, b, c, e)) / (2.0 * h);
    const double analytic = ode::direction_score_gradient(theta, a, b, c, e);
    require(
      std::abs(numeric - analytic) <= 1e-6,
      "direction score gradient disagrees with finite difference");
  }
  std::cout << "  score gradient matches finite difference\n";
}

void test_sensor_hold_release()
{
  constexpr double last = 0.10;
  constexpr double on = 0.15;
  constexpr double off = 0.40;
  constexpr double hold = 0.20;
  require(
    std::abs(ode::held_sensor_clearance(last, on, off, 0.0, hold) - last) <= 1e-12,
    "sensor hold does not start at the last clearance");
  require(
    std::abs(ode::held_sensor_clearance(last, on, off, 0.10, hold) - 0.25) <= 1e-12,
    "sensor hold midpoint is not on the release ramp");
  require(
    std::abs(ode::held_sensor_clearance(last, on, off, hold, hold) - off) <= 1e-12,
    "sensor hold does not reach the release distance at the deadline");
  require(
    ode::distance_activation(
      ode::held_sensor_clearance(last, on, off, hold, hold), on, off) <= 1e-12,
    "sensor hold deadline does not have zero TG distance authority");
  std::cout << "  sensor hold reaches zero authority continuously\n";
}

void test_leaf(const KinematicsContext & context)
{
  const auto & table = ode::neighborhood_table();
  std::size_t ring_sensor = 0;
  bool found = false;
  for (std::size_t index = 0; index < table.size(); ++index) {
    if (table[index].valid && index < context.control_point_jacobians.size()) {
      ring_sensor = index;
      found = true;
      break;
    }
  }
  require(found, "no usable sensor in the kinematics context");

  const auto set_neighbours = [&table, ring_sensor](
    std::vector<double> & scene, double clearance) {
      const auto & entry = table[ring_sensor];
      for (std::size_t slot = 0; slot < entry.count; ++slot) {
        scene[entry.entries[slot].index] = clearance;
      }
    };

  const ode::LeafParams params = make_params();
  const Eigen::Vector3d goal(0.6, -0.4, 0.6);
  JointVector qd = JointVector::Zero();
  qd << 0.10, -0.05, 0.03, 0.02, -0.01, 0.04;
  // Blockage observation input only; it never reaches (a, M).
  // Empty rates read as zero, which switches the closing-rate term off.
  const std::vector<double> no_rates;
  JointVector nominal_qdd = JointVector::Zero();
  nominal_qdd << 0.5, -0.2, 0.1, 0.05, -0.03, 0.02;

  // --- inactive when nothing is near ---------------------------------------
  {
    ode::LeafState state;
    const auto out = ode::evaluate(state, params, context, qd, nominal_qdd, goal, clear_scene(), no_rates);
    require(!out.active, "leaf activated with no obstacle in range");
    require(
      out.metric_contribution.norm() <= 0.0 && out.force_contribution.norm() <= 0.0,
      "inactive leaf contributed to the root RMP");
  }

  // --- active: structural properties ---------------------------------------
  ode::LeafState state;
  auto clearances = clear_scene();
  clearances[ring_sensor] = 0.05;
  set_neighbours(clearances, 0.30);

  // --- signed e2 bias selects the matching tangent-basis direction ---------
  for (const double e2_bias : {0.75, -0.75}) {
    ode::LeafParams biased = params;
    biased.weight_goal = 0.0;
    biased.weight_continuity = 0.0;
    biased.e1_bias = 0.0;
    biased.e2_bias = e2_bias;
    biased.weight_risk = 0.0;
    biased.weight_axis = 0.0;
    biased.weight_axial_block = 0.0;
    biased.weight_i1 = 0.0;
    biased.weight_stall = 0.0;
    biased.direction_rate = 0.0;
    ode::LeafState biased_state;
    const auto out = ode::evaluate(
      biased_state, biased, context, qd, nominal_qdd, goal, clearances, no_rates);
    require(out.active, "e2-bias probe did not activate the leaf");
    require(
      std::abs(out.score_b - e2_bias) <= 1e-12,
      "configured e2 bias was not added to score_b");
    require(
      std::copysign(1.0, out.tangent.dot(out.basis_e2)) ==
      std::copysign(1.0, e2_bias),
      "e2 bias selected the wrong tangent-basis direction");
  }
  std::cout << "  signed e2 bias selects +/-e2\n";

  double previous_budget = 0.0;
  double injected_energy = 0.0;
  bool saw_active = false;

  for (int cycle = 0; cycle < 200; ++cycle) {
    const auto out = ode::evaluate(state, params, context, qd, nominal_qdd, goal, clearances, no_rates);
    require(out.active, "leaf did not stay active with a close obstacle");
    saw_active = true;

    // 1. t . o == 0 by construction, not by re-projection.
    require(
      out.tangent_normal_residual <= 1e-12,
      "tangent is not orthogonal to the obstacle normal");
    require(
      std::abs(out.tangent.norm() - 1.0) <= 1e-12,
      "tangent is not a unit vector");
    require(
      std::abs(out.tangent.dot(out.tangent_perpendicular)) <= 1e-12,
      "tangent and its perpendicular are not orthogonal");

    // 2. Pullback is symmetric PSD with rank at most one.
    const Matrix6 & delta = out.metric_contribution;
    require(
      (delta - delta.transpose()).norm() <= 1e-9 * std::max(1.0, delta.norm()),
      "metric contribution is not symmetric");
    Eigen::SelfAdjointEigenSolver<Matrix6> solver(delta);
    require(solver.info() == Eigen::Success, "eigendecomposition failed");
    const auto eigenvalues = solver.eigenvalues();
    const double tolerance = 1e-9 * std::max(1.0, delta.norm());
    require(eigenvalues.minCoeff() >= -tolerance, "metric contribution is not PSD");
    int rank = 0;
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
      if (eigenvalues[i] > tolerance) {
        ++rank;
      }
    }
    require(rank <= 1, "metric contribution has rank above one");

    // 3. The force lies along the same single axis as the metric.
    if (rank == 1 && out.force_contribution.norm() > 1e-9) {
      Eigen::Index dominant = 0;
      eigenvalues.maxCoeff(&dominant);
      const JointVector axis = solver.eigenvectors().col(dominant);
      const JointVector residual =
        out.force_contribution - axis * axis.dot(out.force_contribution);
      require(
        residual.norm() <= 1e-6 * std::max(1.0, out.force_contribution.norm()),
        "force contribution is not along the leaf axis");
    }

    // 4. Jdot decomposition adds up.
    require(
      std::abs(out.jdot_qdot) < std::numeric_limits<double>::infinity(),
      "Jdot * qdot is not finite");

    // 5. Budget only grows while moving forward, and never goes negative.
    require(out.progress >= -1e-12, "progress went negative");
    if (out.tangent_rate <= 0.0) {
      require(
        out.progress <= previous_budget + 1e-12,
        "progress grew while tangential progress was not positive");
    }
    previous_budget = out.progress;

    require(
      out.progress_remaining > 0.0 && out.progress_remaining <= 1.0 + 1e-12,
      "remaining progress left the unit interval");

    // 6. The leaf is designed in canonical form, so the desired acceleration
    // is bounded by accel_gain at every distance -- this is exactly what a
    // force-designed leaf failed to guarantee.
    require(
      out.push_acceleration <= params.accel_gain + 1e-9,
      "push acceleration exceeded accel_gain");
    require(
      std::abs(out.natural_force - out.metric * out.desired_acceleration) <=
      1e-6 * std::max(1.0, std::abs(out.natural_force)),
      "natural force is not m * a");

    injected_energy += std::max(out.injected_power, 0.0) * params.control_dt;
  }
  require(saw_active, "leaf never became active");

  // The progress is still integrated but no longer attenuates the push unless
  // enable_progress_attenuation is set, so the well depth below is what the
  // leaf WOULD be bounded by, not what currently bounds it.  With the
  // attenuation off the only dissipation left is the -b_e ydot damping term,
  // which is why that term stays outside the push saturation.  Reported rather
  // than asserted: the number is what says whether the attenuation is still
  // needed.
  const double well_depth =
    params.metric_mu * params.accel_gain * params.progress_scale;
  std::cout << "  injected energy " << injected_energy
            << " vs well depth " << well_depth << "\n";

  // --- pocket guard shuts the leaf down ------------------------------------
  // The guard is a smooth sigmoid on the geometric mean neighbour risk, so the
  // meaningful property is monotonicity as the pocket closes, not one value.
  {
    double previous_alpha = 2.0;
    double tightest_alpha = 1.0;
    for (const double neighbour_clearance : {0.30, 0.15, 0.06, 0.02, 0.005}) {
      ode::LeafState pocket_state;
      auto pocket = clear_scene();
      pocket[ring_sensor] = 0.02;
      set_neighbours(pocket, neighbour_clearance);
      const auto out = ode::evaluate(pocket_state, params, context, qd, nominal_qdd, goal, pocket, no_rates);
      require(out.active, "pocket probe did not activate the leaf");
      require(
        out.alpha_free <= previous_alpha + 1e-12,
        "pocket guard did not tighten monotonically as the pocket closed");
      require(
        out.alpha_free >= -1e-12 && out.alpha_free <= 1.0 + 1e-12,
        "pocket guard left the unit interval");
      // Activation lives on the metric and nowhere else.  Gating the push as
      // well was the older arrangement, and it was wrong twice over: alpha
      // entered the transported force f = M a squared, and since the damping
      // term -b_e ydot carried no gate at all, a shrinking alpha drove the
      // desired acceleration onto pure damping -- the leaf braked tangential
      // motion instead of yielding.
      require(
        std::abs(
          out.metric - out.alpha_total * out.metric_gate * params.metric_mu /
          (out.clearance / params.metric_length + params.metric_epsilon)) <=
        1e-6 * std::max(1.0, out.metric),
        "leaf metric does not carry the full activation");
      require(
        std::abs(
          out.alpha_total -
          out.alpha_distance * out.alpha_block * out.alpha_free) <= 1e-12,
        "alpha_TG is not the product of its three factors");
      // With the progress attenuation off and no clearance rate supplied, the
      // push is the distance term alone -- and in particular carries no
      // activation factor.
      require(
        std::abs(
          out.push_acceleration - std::min(
            params.accel_gain * std::exp(-out.clearance / params.accel_length),
            params.accel_saturation)) <=
        1e-6 * std::max(1.0, out.push_acceleration),
        "push acceleration still carries an activation factor");
      require(
        out.closing_term == 0.0 && out.clearance_rate == 0.0,
        "closing-rate term fired without a clearance rate");
      previous_alpha = out.alpha_free;
      tightest_alpha = out.alpha_free;
    }
    require(
      tightest_alpha < 0.10,
      "fully enclosed pocket did not shut the leaf down");
  }
  std::cout << "  pocket guard closes monotonically when boxed in\n";

  // --- distance activation reaches zero smoothly ---------------------------
  // The old Schmitt trigger let the metric step to whatever it happened to be
  // at the release edge.  With the exponential metric that step was ~1e-7 of
  // the peak and nobody noticed, but the metric is about to become hyperbolic
  // in d, where the same edge is a real discontinuity.  This pins the property
  // the smoothstep is here to provide -- and with it the reason the
  // direction-memory timer could be deleted: the state is dropped at a point
  // where the leaf already has no authority.
  {
    require(
      std::abs(ode::distance_activation(
        params.alpha_on_distance, params.alpha_on_distance,
        params.alpha_off_distance) - 1.0) <= 1e-12 &&
      ode::distance_activation(
        params.alpha_off_distance, params.alpha_on_distance,
        params.alpha_off_distance) <= 1e-12,
      "alpha_d does not span [0, 1] across the activation band");

    double previous = 2.0;
    double previous_metric = std::numeric_limits<double>::infinity();
    double largest_metric_step = 0.0;
    const double span = params.alpha_off_distance - params.alpha_on_distance;
    for (int step = 0; step <= 200; ++step) {
      const double probe =
        params.alpha_on_distance + span * 1.2 * static_cast<double>(step) / 200.0;
      const double alpha = ode::distance_activation(
        probe, params.alpha_on_distance, params.alpha_off_distance);
      require(
        alpha <= previous + 1e-12 && alpha >= -1e-12 && alpha <= 1.0 + 1e-12,
        "alpha_d is not a monotone map into [0, 1]");
      previous = alpha;

      ode::LeafState probe_state;
      auto probe_scene = clear_scene();
      probe_scene[ring_sensor] = probe;
      set_neighbours(probe_scene, probe + 0.10);
      const auto out = ode::evaluate(
        probe_state, params, context, qd, nominal_qdd, goal, probe_scene, no_rates);
      const double norm = out.metric_contribution.norm();
      if (std::isfinite(previous_metric)) {
        largest_metric_step =
          std::max(largest_metric_step, std::abs(norm - previous_metric));
      }
      previous_metric = norm;
    }
    require(previous <= 1e-12, "alpha_d did not reach zero past the band");
    require(
      previous_metric <= 1e-12,
      "metric contribution is non-zero past the release distance");
    require(
      largest_metric_step < 0.02 * params.metric_mu,
      "metric contribution steps at the release edge");
  }
  std::cout << "  distance activation reaches zero without a step\n";

  // --- theta stays finite and the leaf survives a sign flip ----------------
  {
    ode::LeafParams flipped = params;
    flipped.axial_preference = -1.0;
    ode::LeafState flipped_state;
    const auto out =
      ode::evaluate(flipped_state, flipped, context, qd, nominal_qdd, goal, clearances, no_rates);
    require(out.active && std::isfinite(out.theta), "axial preference flip broke the leaf");
  }
  std::cout << "  axial preference flip is accepted\n";
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " /path/to/robot.urdf\n";
    return EXIT_FAILURE;
  }
  try {
    test_neighborhood_table();
    test_score_gradient();
    test_sensor_hold_release();

    // The CasADi context is used because it populates link_angular_jacobians,
    // which the leaf needs for the omega x t part of Jdot.
    JointVector q = JointVector::Zero();
    q << 0.2, -0.5, 0.9, 0.1, 0.4, -0.3;
    JointVector qd = JointVector::Zero();
    qd << 0.10, -0.05, 0.03, 0.02, -0.01, 0.04;
    CasadiRb10Kinematics model(argv[1]);
    const KinematicsContext context = model.forward_context(q, qd);
    require(
      context.link_angular_jacobians[RB10Model::sensor_control_points[0].parent_link]
      .norm() > 0.0,
      "angular jacobian is empty; omega x t would be silently dropped");
    test_leaf(context);

    std::cout << "tangent_ode Escape leaf tests passed\n";
  } catch (const std::exception & error) {
    std::cerr << "test_tangent_escape_ode: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

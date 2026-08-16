#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "rb10_rmpflow_rviz/rmp_eigen_solver.hpp"
#include "rb10_rmpflow_rviz/tangent_escape_ode.hpp"

namespace rb10_rmpflow_rviz
{

// Layout of the schema-9 diagnostic record.  Shared rather than local to the
// supervisor because the controller's marker builder indexes the same record,
// and a hand-copied offset there silently drifts the moment a block is added.
//
// Slots 0, 9-13, 17-26, 28, 29, 45 and 47 of the fixed prefix keep the meaning
// the marker builder expects, so the tangent/normal/acceleration arrows render
// without a second code path.  Every consumer checks size with >=, so blocks
// are only ever appended.
namespace tangent_escape_debug
{
constexpr std::size_t kFixed = 56;
// Per sensor: clearance, TCP-goal dot, nominal-motion dot, nominal magnitude.
constexpr std::size_t kStride = 4;
constexpr std::size_t kI1 = kFixed + kStride * tangent_escape_ode::kSensorCount;
// a_I1, b_I1, contributing body sensors, largest |Gki t|.
constexpr std::size_t kAccel = kI1 + 4;
// ddot_hat, gamma_v, closing term, that term ungained, gamma_m, goal_e1,
// goal_e2, whether the goal reference came from the nominal motion.
constexpr std::size_t kTerms = kAccel + 8;
// Every dipole term of the direction score, weights applied.
constexpr std::size_t kProgress = kTerms + 8;
// TCP position and remaining goal distance.
constexpr std::size_t kBasis = kProgress + 4;
// e1, e2 and the goal reference projected into the tangent plane, in world.
constexpr std::size_t kSize = kBasis + 9;
}  // namespace tangent_escape_debug

struct TangentEscapeCommand
{
  using JointVector = RB10Model::JointVector;
  using Matrix6 = Eigen::Matrix<double, 6, 6>;

  bool active{false};
  std::size_t control_point_index{0};
  Eigen::Vector3d tangent{Eigen::Vector3d::UnitX()};
  double metric_scalar{0.0};
  double scalar_acceleration{0.0};
  Matrix6 metric_contribution{Matrix6::Zero()};
  JointVector force_contribution{JointVector::Zero()};
  std::vector<double> debug_data;
};

// Thin adapter between the solver and the Escape leaf.  The direction state
// lives in tangent_escape_ode::LeafState, outside the RMP tree; this class only
// turns the solver's obstacle list into per-sensor clearances and forwards the
// leaf's pullback.
class TangentEscapeSupervisor
{
public:
  using JointVector = RB10Model::JointVector;
  using Matrix6 = Eigen::Matrix<double, 6, 6>;
  using RowVector6 = Eigen::Matrix<double, 1, 6>;

  explicit TangentEscapeSupervisor(EigenRmpConfig config);

  TangentEscapeCommand evaluate(
    const JointVector & q,
    const KinematicsContext & context,
    const JointVector & qd,
    const Eigen::Vector3d & goal,
    const std::vector<ObstacleSphere> & obstacles,
    const JointVector & nominal_qdd,
    const Matrix6 & base_metric,
    const JointVector & base_force) const;

  void set_previous_qdd(const JointVector & qdd) const;

private:
  void evaluate_ode(
    const KinematicsContext & context,
    const JointVector & qd,
    const JointVector & nominal_qdd,
    const Eigen::Vector3d & goal,
    const std::vector<ObstacleSphere> & obstacles,
    Matrix6 & metric,
    JointVector & force,
    TangentEscapeCommand & command,
    std::vector<double> * debug_data) const;

  // Clearance-rate estimator.
  //
  // Differencing the raw clearance is the obvious way and the wrong one: the
  // clearance is rebuilt every cycle from proximity readings, so the
  // difference carries their noise amplified by 1/dt.  Most of the rate is not
  // noisy at all, though.  Split it:
  //
  //   ddot = u . v_k                    the robot's own motion, exact from
  //                                     encoders for a stationary obstacle
  //        + (obstacle's own motion)    everything left over
  //
  // The first part is taken analytically, unfiltered.  Only the residual --
  // which is where an actually moving obstacle shows up, along with all the
  // sensor noise -- goes through the low-pass and the dead-band.  This is the
  // same decomposition the Collision leaf already uses for its own rate.
  struct ClearanceRateState
  {
    std::array<double, tangent_escape_ode::kSensorCount> previous_clearance{};
    std::array<bool, tangent_escape_ode::kSensorCount> seeded{};
    std::array<double, tangent_escape_ode::kSensorCount> filtered_residual{};
  };

  // qdd_nominal is a fresh root solve every cycle with no continuity between
  // them, so the direction it implies for a control point jitters: measured,
  // the goal reference swung inside the tangent plane at 2.6 rad/s at the 90th
  // percentile and 22 rad/s at the 99th, against a theta ODE that tops out
  // near 5.  The score's target was moving an order of magnitude faster than
  // the state chasing it could follow.  One first-order lag on the joint
  // acceleration fixes every control point at once and keeps the reference
  // exact kinematics of a smoothed nominal motion.
  EigenRmpConfig config_;
  mutable tangent_escape_ode::LeafState ode_state_{};
  mutable JointVector filtered_nominal_qdd_{JointVector::Zero()};
  mutable bool nominal_qdd_seeded_{false};
  mutable ClearanceRateState clearance_rate_state_{};
};

}  // namespace rb10_rmpflow_rviz

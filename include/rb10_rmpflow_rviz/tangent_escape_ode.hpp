#pragma once

// Escape RMP v2 -- a one-dimensional leaf laid out next to the Collision leaf.
//
//   Collision : x = d(q),        J_c = -o^T Jp,   infinite-depth barrier
//   Escape    : ydot = t^T v,    J_e =  t^T Jp,   finite-depth well
//
// The leaf coordinate is the tangential progress rate.  It is deliberately NOT
// a holonomic task map: t depends on the internal direction state theta, so y
// is a path integral rather than a function of q.  RMPflow's pullback only
// needs J and Jdot*qdot, so the algorithm is unaffected; the consequence is
// that the standard GDS theorem cannot be cited directly and stability is
// argued through the finite energy budget instead.
//
// Two properties are structural rather than tuned:
//   * t is built from the sensor frame's own tangent basis, so t . o == 0
//     exactly.  There is no world-frame latch and no re-projection.
//   * The metric is a function of clearance only.  The direction state never
//     enters the metric, which keeps the curvature term identically zero.

#include <array>
#include <cstddef>

#include <Eigen/Dense>

#include "rb10_rmpflow_rviz/rb10_model.hpp"

namespace rb10_rmpflow_rviz
{
namespace tangent_escape_ode
{

using JointVector = RB10Model::JointVector;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using RowVector6 = Eigen::Matrix<double, 1, 6>;

// How one neighbouring sensor informs the direction choice of another.
//
// The four-sensor cross rings admit the hand-written (next, previous,
// antipodal) table, but the wrist cluster does not: it is three lateral
// sensors plus one axial sensor.  Rather than special-case it, every
// neighbour is reduced to the angle at which its normal enters this sensor's
// tangent plane.  For a regular cross ring this reproduces the hand-written
// table exactly, so nothing changes on the links that already worked.
struct NeighborContribution
{
  std::size_t index{0};
  // The neighbour looks the opposite way, so it carries no tangent-plane
  // direction at all -- it only says "the far side is blocked too", which
  // argues for leaving along the link axis.
  bool axial_block{false};
  double phi{0.0};      // tangent-plane angle in the (e1, e2) basis
  double weight{0.0};   // |P_T o_j|, so a partly aligned neighbour counts less
};

// Up to (group size - 1); every current group holds four sensors.
inline constexpr std::size_t kMaxNeighbors = 7;

struct SensorNeighborhood
{
  bool valid{false};
  std::size_t count{0};
  std::array<NeighborContribution, kMaxNeighbors> entries{};
  // eps_i: sign that makes eps * e1 point toward the distal end of the link.
  double axial_sign{1.0};
  // False for a sensor whose e1 has no axial component (the wrist's forward
  // sensor), where "toward the distal end" is not meaningful.
  bool axial_preference_defined{true};
};

// Built once from RB10Model::sensor_control_points.  Every sensor that shares
// its neighbourhood with at least one other sensor is valid, so all 20 ToF
// sensors can own the leaf.
const std::array<SensorNeighborhood, RB10Model::sensor_control_points.size()> &
neighborhood_table();

struct LeafParams
{
  bool enabled{false};

  // Metric.  Kept deliberately narrower than the force gate: the metric is a
  // constraint on tangential acceleration, so a wide metric would cap what the
  // goal leaf may do inside the tangent plane even when Escape has little to
  // say.  Force is what should reach far; metric is what should stay local.
  double metric_max{100000.0};
  double metric_sigma{0.03};

  // Acceleration.  The leaf is designed in canonical (a, M) form; f = M a is
  // only the representation the pullback transports.  accel_max is a real
  // tangential acceleration in m/s^2, so a stays bounded by it at every
  // distance no matter how narrow the metric gate is.
  double accel_max{50.0};
  double accel_sigma{0.08};
  double budget_scale{1.0};          // s0, finite well depth in metres
  double budget_recovery_time{3.0};  // tau
  // b_e as a rate [1/s]: 1/b_e is the tangential settling time and must stay
  // well above the control period, so keep it far below the control rate.
  double tangent_damping{40.0};

  // Direction ODE.
  double direction_rate{3.0};     // k_theta
  double risk_sigma{0.06};        // sigma of the neighbour risk weights
  double weight_goal{1.0};
  double weight_velocity{0.3};
  double weight_risk{1.0};
  double weight_curvature{1.5};
  double velocity_epsilon{1e-4};
  // Flips the preferred axial escape direction for every sensor at once.
  double axial_preference{1.0};

  // Pocket guard.  When the two ring neighbours and the antipodal sensor are
  // all close there is no useful direction left inside the tangent plane, and
  // the only way out is along -o, which belongs to Collision.
  double pocket_threshold{0.6};
  double pocket_epsilon{0.1};

  // Ownership.
  double activation_distance{0.30};
  double release_distance{0.40};
  double switch_hysteresis{0.005};
  // How long a direction and its spent budget survive a deactivation.  The
  // release boundary is a hard edge, so without this the leaf re-seeds its
  // direction and refunds its budget every time it briefly leaves the band --
  // which turns the activation band into a limit cycle.
  double direction_memory_time{1.0};

  // Stall memory.  Pushing hard while going nowhere is the signature of a
  // local minimum, and the plain gradient ascent has no way to notice: it
  // oscillates between two maxima of S forever.  Directions visited while
  // stalled are accumulated as a dipole and subtracted from the score, so
  // theta is pushed out of the region it has already been failing in.  This
  // lives entirely inside the direction ODE, which is outside the RMP tree,
  // so it changes nothing about (f, M).
  double weight_stall{2.0};
  double stall_time_constant{1.5};   // seconds to build up / forget
  double stall_speed{0.02};          // |ydot| below this counts as stalled
  double clearance_margin{0.0};

  double control_dt{0.005};
};

struct LeafState
{
  bool active{false};
  bool theta_valid{false};
  std::size_t sensor_index{0};
  double theta{0.0};
  double budget{0.0};
  double inactive_time{0.0};
  // Decaying accumulator of the tangent directions visited while stalled.
  double stall_x{0.0};
  double stall_y{0.0};
  // Last cycle's tangential rate: the motion the previous command
  // actually produced, which is what stall has to judge.
  double previous_tangent_rate{0.0};
};

struct LeafOutput
{
  bool active{false};
  std::size_t sensor_index{0};
  double clearance{0.0};

  Eigen::Vector3d obstacle_direction{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d tangent{Eigen::Vector3d::UnitY()};
  Eigen::Vector3d tangent_perpendicular{Eigen::Vector3d::UnitZ()};

  double theta{0.0};
  double theta_rate{0.0};
  double budget{0.0};
  double budget_remaining{1.0};   // exp(-s/s0)
  double tangent_rate{0.0};       // ydot

  double score_a{0.0};
  double score_b{0.0};
  double score_c{0.0};
  // sin(2 theta) coefficient.  Identically zero on a regular cross ring; it
  // only becomes non-zero for irregular neighbourhoods such as the wrist.
  double score_e{0.0};
  double stall{0.0};           // 1 while pushing but not moving
  double stall_penalty{0.0};   // magnitude of the accumulated dipole

  double push_acceleration{0.0};     // a_push
  double damping_acceleration{0.0};  // b_e * ydot
  double desired_acceleration{0.0};  // a_e, bounded by accel_max
  double natural_force{0.0};         // f_e = m a_e
  double metric{0.0};             // m
  double alpha_free{1.0};
  double injected_power{0.0};     // f_e * ydot

  // Jdot_e * qdot, split so the log can show whether the direction ODE or the
  // link rotation dominates.  A theta term far above the omega term means
  // direction_rate is too high.
  double jdot_qdot{0.0};
  double jdot_theta_term{0.0};
  double jdot_omega_term{0.0};

  double tangent_normal_residual{0.0};  // |t . o|, expected ~1e-16
  Eigen::Vector3d control_point{Eigen::Vector3d::Zero()};

  RowVector6 leaf_jacobian{RowVector6::Zero()};
  Matrix6 metric_contribution{Matrix6::Zero()};
  JointVector force_contribution{JointVector::Zero()};
};

// One control cycle.  `clearances` is indexed like sensor_control_points; a
// non-finite or negative-infinity entry means that sensor saw nothing.
LeafOutput evaluate(
  LeafState & state,
  const LeafParams & params,
  const KinematicsContext & context,
  const JointVector & qd,
  const Eigen::Vector3d & goal,
  const std::vector<double> & clearances);

// Score whose gradient the direction ODE ascends:
//   S(theta) = a cos(theta) + b sin(theta) + c cos(2 theta) + e sin(2 theta)
double direction_score(double theta, double a, double b, double c, double e);
double direction_score_gradient(double theta, double a, double b, double c, double e);

}  // namespace tangent_escape_ode
}  // namespace rb10_rmpflow_rviz

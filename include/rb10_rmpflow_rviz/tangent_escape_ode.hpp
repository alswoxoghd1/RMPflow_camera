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
// argued through the finite energy progress instead.
//
// Two properties are structural rather than tuned:
//   * t is built from the sensor frame's own tangent basis, so t . o == 0
//     exactly.  There is no world-frame latch and no re-projection.
//   * The direction state never enters the metric, which keeps the curvature
//     term identically zero.  The metric does depend on the clearance and on
//     its rate -- m(d, ddot) -- but not on theta, and that is what the
//     curvature term needs.

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

inline constexpr std::size_t kSensorCount =
  RB10Model::sensor_control_points.size();

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

  // Observation mode.  Ownership, the direction ODE, the activation gate and
  // the blockage measurement all run exactly as they would in a live episode,
  // but nothing reaches the root RMP, so the trajectory belongs to the base
  // policy alone.  An activation signal scored on a trajectory the leaf itself
  // shaped would be circular; this is what gives the candidate signals an
  // uncontaminated reference to be judged against.
  bool observe_only{false};

  // Blockage activation.  alpha_block = smoothstep(o . ahat_nominal) between
  // these two, where ahat_nominal is where the goal-driven policy accelerates
  // the owner's control point with every obstacle removed.  Zero means the
  // point is not being driven into its obstacle at all, so the leaf has
  // nothing to prevent.
  double block_on_dot{0.0};
  double block_off_dot{0.3};

  // Below this the nominal control-point acceleration carries no trustworthy
  // direction -- near the goal it vanishes.  The blockage gate then fails
  // OPEN: reporting "not blocked" there would switch the leaf off for want of
  // a reading, which is exactly how the deleted velocity gate failed.
  double nominal_motion_epsilon{1e-3};

  // Metric.  Kept deliberately narrower than the force gate: the metric is a
  // constraint on tangential acceleration, so a wide metric would cap what the
  // goal leaf may do inside the tangent plane even when Escape has little to
  // say.  Force is what should reach far; metric is what should stay local.
  // m = alpha_TG gamma_m(ddot) mu / (d / l_m + eps_m).
  //
  // `metric_mu` is mu, the numerator -- it is no longer the peak, which is
  // mu / eps_m.  `metric_length` is l_m.  Both keep their old names so the
  // launch arguments and the parameter file do not have to churn.
  double metric_mu{10000.0};      // mu
  double metric_length{0.03};       // l_m
  double metric_epsilon{0.1};      // eps_m, what bounds m at contact
  // v_m, its own width -- the metric and the acceleration are allowed to react
  // to the closing rate on different scales.
  double metric_rate_scale{0.05};

  // Acceleration.  The leaf is designed in canonical (a, M) form; f = M a is
  // only the representation the pullback transports.  accel_gain is a real
  // tangential acceleration in m/s^2, so a stays bounded by it at every
  // distance no matter how narrow the metric gate is.
  double accel_gain{50.0};     // k_p, gain of the distance term
  double accel_length{0.08};   // l_p
  // Cap on the whole push, k_p + the closing term.  It has to be separate from
  // k_p: with one number doing both, the exponential already sits at the cap
  // where the closing term matters most and the term would be clipped away
  // exactly at contact.
  double accel_saturation{50.0};

  // Closing-rate term.  Distance alone cannot tell a control point sitting
  // still at 5 cm from one driving into the obstacle at 5 cm, and those two
  // want very different pushes.
  //
  //   gamma_v(ddot) = 1 - sigmoid(ddot / closing_rate_scale)   soft rectifier
  //   a_close       = k_v [ -gamma_v ddot ]_+ / (d / l_v + eps_v)
  //
  // gamma_v is already a soft [.]_+ on the closing direction; the hard clamp
  // on top of it only guarantees that receding never subtracts from the push,
  // which the saturation below would otherwise hide.
  double closing_gain{0.0};        // k_v; zero leaves the term logged but unused
  double closing_length{0.05};     // l_v
  double closing_epsilon{0.1};     // eps_v, and what bounds the term as d -> 0
  double closing_rate_scale{0.05}; // v_d, width of the gamma_v transition

  // Progress attenuation e^{-s/s0}.  Off by default: the closing-rate term now
  // reads the same "am I actually getting anywhere" question off the sensor
  // directly, which is what s was standing in for.  The state is still
  // integrated and logged so it can be switched back on without a rebuild.
  //
  // Worth knowing before leaving it off: the finite well was the leaf's
  // stability argument.  y is a path integral, so the GDS theorem cannot be
  // cited here, and with s gone the only dissipation left is the -b_e ydot
  // damping term.  That term is the reason the damping was kept out of the new
  // acceleration's saturation.
  bool enable_progress_attenuation{false};
  double progress_scale{1.0};          // s0, finite well depth in metres
  double progress_recovery_time{3.0};  // tau
  // b_e as a rate [1/s]: 1/b_e is the tangential settling time and must stay
  // well above the control period, so keep it far below the control rate.
  double tangent_damping{40.0};

  // Direction ODE.
  double direction_rate{3.0};     // k_theta
  // sigma of the neighbour risk weights.  This decides at what range the
  // obstacle distribution has any say in the direction at all, and 0.06 was
  // far too tight for it: with every sensor at 0.09 m or more, exp(-d/0.06)
  // put the whole risk dipole at 0.05 against a goal term of 0.56, so the
  // tangent was chosen by goal and current motion with no argument for going
  // around the obstacle rather than over it.
  double risk_length{0.15};
  // The pocket guard reads the same neighbour clearances but asks a different
  // question -- "am I boxed in", which is a near-contact condition -- so it
  // gets its own length scale rather than inheriting the direction term's.
  double pocket_length{0.06};
  double weight_goal{1.0};
  double weight_continuity{0.3};
  // Signed dipole bias in the sensor tangent basis.  Negative values prefer
  // -e1, positive values prefer +e1, and zero preserves the geometric score.
  double e1_bias{0.0};
  double weight_risk{1.0};
  double weight_axis{1.5};
  double velocity_epsilon{1e-4};

  // I1: whole-body kinematic consequence.  Penalises a tangent that would send
  // some other part of the arm toward its own obstacle.
  //
  // Defaults to zero, i.e. computed and logged but not applied.  The gain
  // o_k^T Jk Ji# t is dimensionless but not bounded by one -- moving a control
  // point near the base swings a distal one much further -- so the weight has
  // to be read off the logged coefficients rather than guessed.
  double weight_i1{0.0};
  double i1_length{0.10};      // rho_k = exp(-d_k / sigma_I)
  double i1_damping{0.05};    // lambda_J in the damped pseudoinverse
  // Keep I1 to sensors on other links.  The owner's own ring is what the local
  // risk term already speaks for; overlapping the two would double-count the
  // same neighbours and blur what each term is answering.
  bool i1_exclude_parent_link{true};
  // Flips the preferred axial escape direction for every sensor at once.
  double axial_preference{1.0};

  // Weight on the antipodal neighbour's "the far side is blocked too" signal,
  // which enters the dipole as a push along +e1 and the quadrupole as a
  // preference for the e1 axis.
  //
  // Zero, because the signal is not real.  A clearance here is the geometric
  // distance from a control point to an obstacle with no regard for which way
  // its sensor faces, so the antipodal sensor reads the SAME obstacle from the
  // far side of the link.  Measured over 11843 cycles owned by tof6_1_L, its
  // antipodal partner reported a finite clearance on 100% of them, sitting a
  // median 0.1175 m further out -- which is exactly the 0.12 m that separates
  // the two sensors on the wrist -- and correlating with the owner at 0.982.
  // Nothing was ever behind the arm.  The term ran at a steady 0.24 and pulled
  // theta toward the link axis the whole time.
  //
  // Restoring it means fixing the clearance first: a sensor should only report
  // an obstacle that lies in front of it.
  double weight_axial_block{0.0};

  // Pocket guard.  When the two ring neighbours and the antipodal sensor are
  // all close there is no useful direction left inside the tangent plane, and
  // the only way out is along -o, which belongs to Collision.
  double pocket_threshold{0.6};
  double pocket_epsilon{0.1};

  // Distance activation.  These were the two edges of a Schmitt trigger; they
  // are now the two ends of a quintic smoothstep, so the same numbers keep
  // roughly the same meaning without the latch.  alpha_d is 1 at or below
  // alpha_on_distance, 0 at or above alpha_off_distance.  The latch was worth
  // removing on its own -- once engaged it never let go, measured active on
  // 11291 of 11291 cycles -- but it has to go before the metric becomes
  // hyperbolic in d, because that metric does not vanish on its own at the
  // release edge and would step to zero there.
  double alpha_on_distance{0.30};
  double alpha_off_distance{0.40};
  double owner_hysteresis{0.005};

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
  double progress{0.0};
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
  double progress{0.0};
  double progress_remaining{1.0};   // exp(-s/s0)
  double tangent_rate{0.0};       // ydot

  double score_a{0.0};
  double score_b{0.0};
  double score_c{0.0};
  // sin(2 theta) coefficient.  Identically zero on a regular cross ring; it
  // only becomes non-zero for irregular neighbourhoods such as the wrist.
  double score_e{0.0};
  double stall{0.0};           // 1 while pushing but not moving
  double stall_penalty{0.0};   // magnitude of the accumulated dipole

  // Every dynamic dipole term as it actually enters score_a / score_b, weights
  // already applied.  The configured e1_bias is added separately to score_a;
  // after that, the stall term is subtracted.  Without this decomposition the
  // log can only lump velocity together with risk, which is precisely the pair
  // that has to be told apart: one is meant to hold a working direction and the
  // other to argue against the obstacle.
  double score_a_goal{0.0};
  double score_b_goal{0.0};
  double score_a_velocity{0.0};
  double score_b_velocity{0.0};
  double score_a_risk{0.0};
  double score_b_risk{0.0};
  double score_a_stall{0.0};
  double score_b_stall{0.0};
  double score_a_i1{0.0};
  double score_b_i1{0.0};
  // The goal reference resolved into the tangent basis, i.e. a_goal and b_goal
  // before weight_goal.  Logged so a sudden turn can be attributed: the
  // nominal-motion reference moves faster than the old TCP line did.
  double goal_e1{0.0};
  double goal_e2{0.0};
  bool goal_from_nominal{false};
  std::size_t i1_count{0};       // body sensors that contributed
  double i1_largest_gain{0.0};   // max |Gki t| seen, for scaling weight_i1

  double clearance_rate{0.0};        // ddot_hat for the owner
  double closing_gate{0.0};          // gamma_v(ddot_hat)
  double closing_term{0.0};          // the k_v term as it enters the push
  double closing_shape{0.0};         // the same without k_v, for scaling it

  double push_acceleration{0.0};     // a_push, after saturation
  double damping_acceleration{0.0};  // b_e * ydot
  double desired_acceleration{0.0};  // a_e, bounded by accel_gain
  double natural_force{0.0};         // f_e = m a_e
  double metric{0.0};             // m
  double metric_gate{0.0};        // gamma_m(ddot_hat)
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

  // Where the robot actually is, and how far it still has to go.  Without
  // these the log can say the direction keeps turning but not whether the arm
  // is getting anywhere -- and circling a pocket is exactly the failure that
  // looks healthy on every other channel: the clearance holds steady, the
  // tangential rate is non-zero, and the leaf reports itself active the whole
  // time.  Path length over net displacement is what separates useful travel
  // along an obstacle from going round in circles, and it needs a position.
  Eigen::Vector3d tcp_position{Eigen::Vector3d::Zero()};
  double goal_distance{0.0};

  // The tangent basis and the goal reference, in world, so RViz can show what
  // theta is measured against.  A lone escape arrow says which way the leaf
  // went but not whether that was with the goal or against it, and the angle
  // between them is the thing worth watching -- it sat past 120 degrees on
  // half the cycles of one run.
  Eigen::Vector3d basis_e1{Eigen::Vector3d::Zero()};   // theta = 0
  Eigen::Vector3d basis_e2{Eigen::Vector3d::Zero()};   // theta = 90 deg
  Eigen::Vector3d goal_in_plane{Eigen::Vector3d::Zero()};

  // --- activation / blockage observation -----------------------------------
  // Filled whenever any sensor sees something, including on the cycles where
  // the leaf itself stays inactive, so the approach into an obstacle is
  // covered and not just the engaged episode.
  // alpha_TG = alpha_distance * alpha_block * alpha_free, and it multiplies the
  // metric only.  All four are logged: the product alone cannot say which
  // factor closed the gate.
  double alpha_distance{0.0};   // owner distance
  double alpha_block{1.0};      // owner driven into its own obstacle?
  double alpha_total{0.0};      // alpha_TG
  double blockage{0.0};         // o . ahat_nominal, ungated
  double nominal_magnitude{0.0};
  std::size_t active_link{0};   // parent link of the owner
  bool observation_valid{false};
  std::size_t observation_count{0};

  // Blockage is  g(o_k . dhat), and the open question is what dhat should be:
  //   tcp     : dhat = normalize(goal - p_k)
  //             the straight line from this sensor to the TCP goal, which is
  //             not a direction a link-2 or link-3 point is ever asked to move
  //   nominal : dhat = normalize(J_k qdd_nominal + cdot_k)
  //             where this point is accelerated by the goal-driven policy with
  //             every obstacle removed -- what the point actually wants
  // Both are recorded raw, for every sensor, with no g() applied and with the
  // nominal magnitude alongside, so the reference, the shape of g and the
  // dead-band all get chosen from the data rather than fixed here.
  std::array<double, kSensorCount> sensor_clearance{};
  std::array<double, kSensorCount> sensor_dot_goal_tcp{};
  std::array<double, kSensorCount> sensor_dot_goal_nominal{};
  std::array<double, kSensorCount> sensor_nominal_magnitude{};

  RowVector6 leaf_jacobian{RowVector6::Zero()};
  Matrix6 metric_contribution{Matrix6::Zero()};
  JointVector force_contribution{JointVector::Zero()};
};

// One control cycle.  `clearances` is indexed like sensor_control_points; a
// non-finite or negative-infinity entry means that sensor saw nothing.
// `nominal_qdd` is the root solve over the goal-driven leaves with the
// obstacle list emptied, i.e. the motion the robot would command if nothing
// were in the way.  It drives the blockage gate and the per-sensor blockage
// record; it never enters (a, M) directly.
LeafOutput evaluate(
  LeafState & state,
  const LeafParams & params,
  const KinematicsContext & context,
  const JointVector & qd,
  const JointVector & nominal_qdd,
  const Eigen::Vector3d & goal,
  const std::vector<double> & clearances,
  // Filtered rate of change of each clearance, indexed like `clearances`.
  // Negative means closing.  Empty is treated as all-zero, which makes the
  // closing-rate term vanish rather than misfire.
  const std::vector<double> & clearance_rates);

// alpha_d: 1 at or below `on`, 0 at or above `off`, quintic in between.
double distance_activation(double clearance, double on, double off);

// Score whose gradient the direction ODE ascends:
//   S(theta) = a cos(theta) + b sin(theta) + c cos(2 theta) + e sin(2 theta)
double direction_score(double theta, double a, double b, double c, double e);
double direction_score_gradient(double theta, double a, double b, double c, double e);

}  // namespace tangent_escape_ode
}  // namespace rb10_rmpflow_rviz

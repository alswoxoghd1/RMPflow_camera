#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "rb10_rmpflow_rviz/joint_acceleration_limiter.hpp"
#include "rb10_rmpflow_rviz/pinocchio_direct_solver.hpp"
#include "rb10_rmpflow_rviz/pinocchio_model.hpp"
#include "rb10_rmpflow_rviz/rmp_stability_profile.hpp"

namespace rb10_rmpflow_rviz
{
namespace
{

using JointVector = RB10Model::JointVector;
using Matrix6 = Eigen::Matrix<double, 6, 6>;

constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kCertificateBaseGdsStructural = 1;
constexpr std::size_t kCertificateEnvironmentStatic = 2;
constexpr std::size_t kCertificateExternalRmpEmpty = 3;
constexpr std::size_t kCertificateGuardEnabled = 4;
constexpr std::size_t kCertificateConditionalNonincrease = 5;
constexpr std::size_t kCertificateEnergyViolation = 28;
constexpr std::size_t kCertificateBaseConfigProfileValid = 31;
constexpr std::size_t kCertificateBaseDomainValid = 32;
constexpr std::size_t kCertificateBaseMetricSpd = 33;
constexpr std::size_t kCertificateBaseSolveValid = 34;
constexpr std::size_t kCertificateSize = 35;

std::string test_urdf_path()
{
  return RB10_TEST_URDF_PATH;
}

double radians(double degrees)
{
  return degrees * kPi / 180.0;
}

JointVector reference_q()
{
  JointVector q;
  q <<
    0.624269511868643,
    -0.42053732097686486,
    -1.6277205440557674,
    0.749218624408137,
    -1.429386695978297,
    0.5760347969017805;
  return q;
}

std::vector<RmpNodeConfig> complete_graph()
{
  return {
    make_rmp_node_config(
      "cspace_target", "root", "cspace_target", "cspace_target", true),
    make_rmp_node_config(
      "joint_limits", "root", "joint_limit", "joint_limit", true),
    make_rmp_node_config(
      "joint_velocity_cap", "root", "identity", "joint_velocity_cap", true),
    make_rmp_node_config(
      "tcp_position", "root", "tcp_position", "none", true),
    make_rmp_node_config(
      "target", "tcp_position", "identity", "target", true, "goal"),
    make_rmp_node_config(
      "control_points", "root", "control_points", "none", true),
    make_rmp_node_config(
      "collision", "control_points", "collision_distance", "collision", true),
    make_rmp_node_config(
      "tangent_escape", "control_points", "identity", "tangent_escape", true, "goal"),
    make_rmp_node_config(
      "damping", "root", "identity", "damping", true),
    make_rmp_node_config(
      "tcp_axis_target_x", "root", "link_orientation_axis", "axis_target", true,
      "orientation_goal_x", "tcp_rmp", "x"),
    make_rmp_node_config(
      "tcp_axis_target_y", "root", "link_orientation_axis", "axis_target", true,
      "orientation_goal_y", "tcp_rmp", "y"),
    make_rmp_node_config(
      "tcp_axis_target_z", "root", "link_orientation_axis", "axis_target", true,
      "orientation_goal_z", "tcp_rmp", "z")
  };
}

EigenRmpConfig operational_config(bool escape_enabled)
{
  EigenRmpConfig config;
  config.solve_method = "rmp2";
  config.rmp_type = "canonical";
  config.solve_offset = 0.001;
  config.max_joint_accel = 10.0;
  config.preserve_joint_accel_direction = true;
  config.default_q = {
    0.624269511868643,
    -0.42053732097686486,
    -1.6277205440557674,
    0.749218624408137,
    -1.429386695978297,
    0.5760347969017805};
  config.joint_lower_limits = {
    radians(-211.68), radians(-70.0), radians(-150.0),
    radians(-80.0), radians(-180.0), radians(-360.0)};
  config.joint_upper_limits = {
    radians(211.68), radians(70.0), radians(-50.0),
    radians(80.0), radians(120.0), radians(360.0)};
  config.joint_limit_buffers.fill(0.01);
  config.graph_nodes = complete_graph();

  config.cspace_target.metric_scalar = 1000.0;
  config.cspace_target.position_gain = 5.0;
  config.cspace_target.damping_gain = 30.0;
  config.cspace_target.robust_position_term_thresh = 0.5;
  config.cspace_target.inertia = 0.0;

  config.joint_limit.policy = "lula_canonical";
  config.joint_limit.metric_scalar = 1000.0;
  config.joint_limit.metric_length_scale = 0.1;
  config.joint_limit.metric_exploder_eps = 0.001;
  config.joint_limit.metric_velocity_gate_length_scale = 0.01;
  config.joint_limit.accel_damper_gain = 100.0;
  config.joint_limit.accel_potential_gain = 100.0;
  config.joint_limit.accel_potential_exploder_eps = 0.001;
  config.joint_limit.accel_potential_exploder_length_scale = 0.01;
  config.joint_limit.gds_center_fraction = 0.5;
  config.joint_limit.gds_domain_epsilon = 1e-9;

  config.joint_velocity_cap.max_velocity = 1.0;
  config.joint_velocity_cap.velocity_damping_region = 0.5;
  config.joint_velocity_cap.damping_gain = 500.0;
  config.joint_velocity_cap.metric_weight = 100.0;

  config.target.accel_p_gain = 40.0;
  config.target.accel_d_gain = 120.0;
  config.target.accel_norm_eps = 0.08;
  config.target.metric_alpha_length_scale = 0.01;
  config.target.min_metric_alpha = 0.1;
  config.target.max_metric_scalar = 100000.0;
  config.target.min_metric_scalar = 25000.0;
  config.target.proximity_metric_boost_scalar = 2.0;
  config.target.proximity_metric_boost_length_scale = 0.02;

  config.axis_target.accel_p_gain = 80.0;
  config.axis_target.accel_d_gain = 80.0;
  config.axis_target.metric_scalar = 100000.0;
  config.axis_target.proximity_metric_boost_scalar = 10.0;
  config.axis_target.proximity_metric_boost_length_scale = 0.01;

  config.damping.accel_d_gain = 120.0;
  config.damping.metric_scalar = 0.0;
  config.damping.inertia = 300.0;

  config.collision.policy = "lula_canonical";
  config.collision.margin = 0.0;
  config.collision.damping_gain = 50.0;
  config.collision.damping_std_dev = 0.01;
  config.collision.damping_robustness_eps = 0.01;
  config.collision.damping_velocity_gate_length_scale = 0.03;
  config.collision.repulsion_gain = 500.0;
  config.collision.repulsion_std_dev = 0.05;
  config.collision.metric_modulation_radius = 0.3;
  config.collision.metric_scalar = 100000.0;
  config.collision.metric_exploder_std_dev = 0.02;
  config.collision.metric_exploder_eps = 0.8;
  config.collision.gds_clearance_smoothing = 1e-4;
  config.collision.gds_metric_velocity_floor = 0.5;
  config.collision.gds_metric_velocity_scale = 0.03;
  config.collision.gds_damping_velocity_scale = 0.03;

  auto & escape = config.tangent_escape;
  escape.enabled = escape_enabled;
  escape.clearance_margin = 0.0;
  escape.control_dt = 0.005;

  return config;
}

EigenRmpConfig proof_config(bool escape_enabled)
{
  auto config = operational_config(escape_enabled);
  config.solve_offset = 0.0;
  config.max_joint_accel = 1e6;
  config.collision.policy = "paper_gds";
  config.joint_limit.policy = "paper_gds";
  config.target.min_metric_alpha = 1.0;
  config.target.proximity_metric_boost_scalar = 1.0;
  config.axis_target.proximity_metric_boost_scalar = 1.0;
  config.damping.metric_scalar = 0.0;
  for (auto & node : config.graph_nodes) {
    if (node.name == "joint_velocity_cap") {
      node.enabled = false;
    }
  }
  config.enable_tangent_escape_diagnostics = escape_enabled;
  config.enable_tangent_escape_dual_solve_diagnostics = escape_enabled;
  config.enable_stability_certificate_diagnostics = true;
  config.escape_energy_certificate.enabled = true;
  config.escape_energy_certificate.tank_capacity = 1000.0;
  config.escape_energy_certificate.initial_energy = 1000.0;
  config.escape_energy_certificate.control_dt = config.tangent_escape.control_dt;
  config.escape_energy_certificate.scale_search_iterations = 20;
  config.escape_energy_certificate.power_tolerance = 1e-8;
  return config;
}

std::unordered_map<std::string, Eigen::Vector3d> targets_for(
  const KinematicsContext & context,
  const Eigen::Vector3d & goal)
{
  const Eigen::Matrix3d & rotation =
    context.link_rotations[RB10Model::TCP_RMP];
  return {
    {"goal", goal},
    {"body_goal", context.link_positions[RB10Model::LINK4]},
    {"orientation_goal_x", rotation.col(0)},
    {"orientation_goal_y", rotation.col(1)},
    {"orientation_goal_z", rotation.col(2)}
  };
}

struct BlockingScenario
{
  std::unordered_map<std::string, Eigen::Vector3d> targets;
  ObstacleSphere obstacle;
  std::size_t control_point_index{0};
};

BlockingScenario find_blocking_scenario(
  const EigenRmpConfig & no_escape_config,
  const JointVector & q)
{
  const PinocchioModel model(test_urdf_path());
  const JointVector qd = JointVector::Zero();
  const auto context = model.forward_context(q, qd);
  const PinocchioDirectRmpSolver solver(no_escape_config, test_urdf_path());
  const std::array<Eigen::Vector3d, 6> directions{
    Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX(),
    Eigen::Vector3d::UnitY(), -Eigen::Vector3d::UnitY(),
    Eigen::Vector3d::UnitZ(), -Eigen::Vector3d::UnitZ()};

  double best_speed = -1.0;
  Eigen::Vector3d best_goal = context.tcp_position;
  Eigen::Vector3d best_motion = Eigen::Vector3d::UnitX();
  std::size_t best_index = 0;
  for (const auto & direction : directions) {
    const Eigen::Vector3d goal = context.tcp_position + 0.20 * direction;
    const auto targets = targets_for(context, goal);
    const auto result = solver.solve(q, qd, targets, {});
    for (std::size_t index = 0; index < context.control_points.size(); ++index) {
      const Eigen::Vector3d motion =
        context.control_point_jacobians[index] *
        (qd + 0.01 * result.qdd);
      if (motion.norm() > best_speed) {
        best_speed = motion.norm();
        best_goal = goal;
        best_motion = motion;
        best_index = index;
      }
    }
  }

  EXPECT_GT(best_speed, 1e-8);
  const Eigen::Vector3d obstacle_direction =
    best_motion.norm() > 1e-12 ? best_motion.normalized() : Eigen::Vector3d::UnitX();
  constexpr double obstacle_radius = 0.04;
  constexpr double clearance = 0.04;
  const auto & point = context.control_points[best_index];
  ObstacleSphere obstacle;
  obstacle.center =
    point.position +
    (point.radius + obstacle_radius + clearance) * obstacle_direction;
  obstacle.radius = obstacle_radius;
  obstacle.proximity_control_point_index = static_cast<int>(best_index);
  obstacle.source_id = 9001;
  return {targets_for(context, best_goal), obstacle, best_index};
}

void expect_finite_spd_result(
  const RmpSolveResult & result,
  bool expect_exact_root_solve = true)
{
  ASSERT_TRUE(result.qdd.allFinite());
  ASSERT_TRUE(result.metric.allFinite());
  ASSERT_TRUE(result.force.allFinite());
  ASSERT_TRUE(result.metric.isApprox(result.metric.transpose(), 1e-9));

  Eigen::SelfAdjointEigenSolver<Matrix6> eigensolver(result.metric);
  ASSERT_EQ(eigensolver.info(), Eigen::Success);
  EXPECT_GT(eigensolver.eigenvalues().minCoeff(), 0.0);
  if (expect_exact_root_solve) {
    EXPECT_LE(
      (result.metric * result.qdd - result.force).norm(),
      1e-7 * (1.0 + result.force.norm()));
  }
}

TEST(FullRmpProperties, OperationalAndProofProfilesAreNotConflated)
{
  const auto operational = operational_config(true);
  const auto proof = proof_config(true);

  EXPECT_FALSE(base_graph_has_structured_gds_form(operational));
  EXPECT_TRUE(base_graph_has_structured_gds_form(proof));
}

TEST(FullRmpProperties, ProofProfileFullGraphHasClosureCertificate)
{
  const JointVector q = reference_q();
  const JointVector qd = JointVector::Zero();
  auto no_escape_config = proof_config(false);
  const auto scenario = find_blocking_scenario(no_escape_config, q);

  const PinocchioDirectRmpSolver no_escape_solver(no_escape_config, test_urdf_path());
  const auto no_escape = no_escape_solver.solve(
    q, qd, scenario.targets, {scenario.obstacle});
  expect_finite_spd_result(no_escape);

  auto enabled_config = proof_config(true);
  // Keep the GDS proof profile unchanged while making this synthetic
  // activation probe permissive enough to exercise the Escape leaf itself.
  // The disabled joint-velocity leaf remains absent from the proof graph.
  enabled_config.joint_velocity_cap.max_velocity = 1e6;
  const PinocchioDirectRmpSolver enabled_solver(enabled_config, test_urdf_path());
  bool saw_active_escape = false;
  bool saw_nonzero_rank_one_escape = false;
  bool saw_conditional_certificate = false;
  for (int cycle = 0; cycle < 12; ++cycle) {
    const auto result = enabled_solver.solve(
      q, qd, scenario.targets, {scenario.obstacle});
    expect_finite_spd_result(result);
    ASSERT_EQ(result.stability_certificate_data.size(), kCertificateSize);
    const auto & certificate = result.stability_certificate_data;
    EXPECT_EQ(certificate[kCertificateBaseGdsStructural], 1.0);
    EXPECT_EQ(certificate[kCertificateExternalRmpEmpty], 1.0);
    EXPECT_EQ(certificate[kCertificateGuardEnabled], 1.0);
    EXPECT_EQ(certificate[kCertificateBaseConfigProfileValid], 1.0);
    EXPECT_EQ(certificate[kCertificateBaseDomainValid], 1.0);
    EXPECT_EQ(certificate[kCertificateBaseMetricSpd], 1.0);
    EXPECT_EQ(certificate[kCertificateBaseSolveValid], 1.0);
    EXPECT_LE(certificate[kCertificateEnergyViolation], 1e-12);
    if (cycle > 0) {
      EXPECT_EQ(certificate[kCertificateEnvironmentStatic], 1.0);
      EXPECT_EQ(certificate[kCertificateConditionalNonincrease], 1.0);
      saw_conditional_certificate = true;
    }

    if (
      result.tangent_escape_rmp_data.size() > 10 &&
      result.tangent_escape_rmp_data[0] > 0.5 &&
      result.tangent_escape_rmp_data[10] > 1e-9)
    {
      saw_active_escape = true;
      const Matrix6 delta_metric = result.metric - no_escape.metric;
      const Matrix6 symmetric_delta =
        0.5 * (delta_metric + delta_metric.transpose());
      Eigen::SelfAdjointEigenSolver<Matrix6> eigensolver(symmetric_delta);
      ASSERT_EQ(eigensolver.info(), Eigen::Success);
      const double maximum = eigensolver.eigenvalues().maxCoeff();
      const int numerical_rank =
        (eigensolver.eigenvalues().array() > std::max(1e-8, maximum * 1e-8))
        .count();
      EXPECT_GE(eigensolver.eigenvalues().minCoeff(), -1e-7);
      EXPECT_EQ(numerical_rank, 1);
      saw_nonzero_rank_one_escape = true;
    }
  }

  EXPECT_TRUE(saw_active_escape);
  EXPECT_TRUE(saw_nonzero_rank_one_escape);
  EXPECT_TRUE(saw_conditional_certificate);
}

TEST(FullRmpProperties, ProofProfileRemainsWellPosedAcrossInteriorStates)
{
  auto config = proof_config(false);
  const JointVector q_reference = reference_q();
  const PinocchioModel model(test_urdf_path());
  const auto reference_context =
    model.forward_context(q_reference, JointVector::Zero());
  const auto scenario = find_blocking_scenario(config, q_reference);
  const auto fixed_targets = targets_for(
    reference_context,
    reference_context.tcp_position + Eigen::Vector3d(0.04, -0.03, 0.02));
  const PinocchioDirectRmpSolver solver(config, test_urdf_path());

  std::mt19937 generator(20260723U);
  std::uniform_real_distribution<double> position_fraction(0.30, 0.70);
  std::uniform_real_distribution<double> velocity_distribution(-0.08, 0.08);
  for (int sample = 0; sample < 40; ++sample) {
    JointVector q;
    JointVector qd;
    for (int joint = 0; joint < 6; ++joint) {
      const auto index = static_cast<std::size_t>(joint);
      const double lower =
        config.joint_lower_limits[index] + config.joint_limit_buffers[index];
      const double upper =
        config.joint_upper_limits[index] - config.joint_limit_buffers[index];
      const double fraction = position_fraction(generator);
      q[joint] = lower + fraction * (upper - lower);
      qd[joint] = velocity_distribution(generator);
    }

    const auto result = solver.solve(
      q, qd, fixed_targets, {scenario.obstacle});
    expect_finite_spd_result(result);
    ASSERT_EQ(result.stability_certificate_data.size(), kCertificateSize);
    const auto & certificate = result.stability_certificate_data;
    EXPECT_EQ(certificate[kCertificateBaseGdsStructural], 1.0);
    EXPECT_EQ(certificate[kCertificateBaseConfigProfileValid], 1.0);
    EXPECT_EQ(certificate[kCertificateBaseDomainValid], 1.0);
    EXPECT_EQ(certificate[kCertificateBaseMetricSpd], 1.0);
    EXPECT_EQ(certificate[kCertificateBaseSolveValid], 1.0);
    if (sample > 0) {
      EXPECT_EQ(certificate[kCertificateEnvironmentStatic], 1.0);
      EXPECT_EQ(certificate[kCertificateConditionalNonincrease], 1.0);
    }
  }
}

TEST(FullRmpProperties, OperationalProfileIsFiniteButOutsideGdsCertificate)
{
  auto config = operational_config(true);
  config.enable_stability_certificate_diagnostics = true;
  const JointVector q = reference_q();
  const JointVector qd = JointVector::Zero();
  const auto scenario = find_blocking_scenario(operational_config(false), q);
  const PinocchioDirectRmpSolver solver(config, test_urdf_path());

  const auto first = solver.solve(q, qd, scenario.targets, {scenario.obstacle});
  const auto second = solver.solve(q, qd, scenario.targets, {scenario.obstacle});
  expect_finite_spd_result(first, false);
  expect_finite_spd_result(second, false);
  ASSERT_EQ(second.stability_certificate_data.size(), kCertificateSize);
  const auto & certificate = second.stability_certificate_data;
  EXPECT_EQ(certificate[kCertificateEnvironmentStatic], 1.0);
  EXPECT_EQ(certificate[kCertificateBaseConfigProfileValid], 0.0);
  EXPECT_EQ(certificate[kCertificateBaseGdsStructural], 0.0);
  EXPECT_EQ(certificate[kCertificateConditionalNonincrease], 0.0);
}

TEST(FullRmpProperties, DistanceOnlyCollisionSetHasBoundaryCounterexample)
{
  auto config = operational_config(true);
  config.enable_stability_certificate_diagnostics = false;
  const JointVector q = reference_q();
  const PinocchioModel model(test_urdf_path());
  const auto static_context = model.forward_context(q, JointVector::Zero());

  double best_motion_norm = -1.0;
  std::size_t control_point_index = 0;
  Eigen::Vector3d clearance_normal = Eigen::Vector3d::UnitX();
  for (std::size_t point = 0; point < static_context.control_points.size(); ++point) {
    const auto & jacobian = static_context.control_point_jacobians[point];
    for (int joint = 0; joint < 6; ++joint) {
      const Eigen::Vector3d motion = jacobian.col(joint);
      if (motion.norm() > best_motion_norm) {
        best_motion_norm = motion.norm();
        control_point_index = point;
        clearance_normal = motion.normalized();
      }
    }
  }
  ASSERT_GT(best_motion_norm, 1e-6);

  const auto & point = static_context.control_points[control_point_index];
  constexpr double obstacle_radius = 0.04;
  ObstacleSphere obstacle;
  obstacle.center =
    point.position -
    (point.radius + obstacle_radius) * clearance_normal;
  obstacle.radius = obstacle_radius;
  obstacle.proximity_control_point_index =
    static_cast<int>(control_point_index);
  obstacle.source_id = 9901;

  const Eigen::Matrix<double, 1, 6> clearance_jacobian =
    clearance_normal.transpose() *
    static_context.control_point_jacobians[control_point_index];
  ASSERT_GT(clearance_jacobian.squaredNorm(), 1e-8);
  constexpr double approach_speed = 0.05;
  const JointVector qd =
    -clearance_jacobian.transpose() *
    (approach_speed / clearance_jacobian.squaredNorm());
  const auto context = model.forward_context(q, qd);
  const double initial_clearance =
    (context.control_points[control_point_index].position - obstacle.center).norm() -
    context.control_points[control_point_index].radius -
    obstacle.radius;
  const double initial_clearance_rate =
    clearance_normal.dot(
    context.control_point_velocities[control_point_index]);
  ASSERT_NEAR(initial_clearance, 0.0, 1e-12);
  ASSERT_NEAR(initial_clearance_rate, -approach_speed, 1e-10);

  const auto targets = targets_for(
    context,
    context.tcp_position + Eigen::Vector3d(0.05, -0.02, 0.03));
  const PinocchioDirectRmpSolver solver(config, test_urdf_path());
  const auto result = solver.solve(q, qd, targets, {obstacle});
  ASSERT_TRUE(result.qdd.allFinite());
  const JointVector command_qdd = limit_joint_acceleration(
    result.qdd,
    config.max_joint_accel,
    config.preserve_joint_accel_direction);
  ASSERT_TRUE(command_qdd.allFinite());

  // The controller is acceleration controlled. At h=0 with hdot<0, any
  // finite acceleration leaves the distance-only set immediately.
  constexpr double dt = 1e-5;
  const JointVector qd_next = qd + command_qdd * dt;
  const JointVector q_next = q + qd_next * dt;
  const auto next_context = model.forward_context(q_next, qd_next);
  const double next_clearance =
    (next_context.control_points[control_point_index].position - obstacle.center).norm() -
    next_context.control_points[control_point_index].radius -
    obstacle.radius;

  EXPECT_LT(next_clearance, -1e-8);
}

}  // namespace
}  // namespace rb10_rmpflow_rviz

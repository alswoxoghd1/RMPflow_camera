#include <algorithm>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "rb10_rmpflow_rviz/casadi_rmp2_solver.hpp"
#include "rb10_rmpflow_rviz/pinocchio_direct_solver.hpp"

namespace rb10_rmpflow_rviz
{
namespace
{

using JointVector = RB10Model::JointVector;

std::string test_urdf_path()
{
  return RB10_TEST_URDF_PATH;
}

EigenRmpConfig base_config()
{
  EigenRmpConfig config;
  config.solve_method = "rmp2";
  config.rmp_type = "canonical";
  config.solve_offset = 1e-3;
  config.max_joint_accel = 1e6;
  config.casadi_rmp2_max_obstacles = 2;
  config.casadi_rmp2_jit = false;
  config.tangent_escape.enabled = false;
  config.enable_tangent_escape_diagnostics = false;
  config.enable_tangent_escape_dual_solve_diagnostics = false;
  config.enable_leaf_ablation_diagnostics = false;
  config.enable_stability_certificate_diagnostics = false;
  config.escape_energy_certificate.enabled = false;
  config.default_q = {0.1, -0.25, -1.35, 0.2, -0.65, 0.1};
  return config;
}

EigenRmpConfig escape_config(bool enabled)
{
  auto config = base_config();
  config.solve_offset = 1e-4;
  config.enable_tangent_escape_diagnostics = true;
  config.graph_nodes = {
    make_rmp_node_config("tcp_position", "root", "tcp_position", "none", true),
    make_rmp_node_config("target", "tcp_position", "identity", "target", true, "goal"),
    make_rmp_node_config("control_points", "root", "control_points", "none", true),
    make_rmp_node_config(
      "tangent_escape", "control_points", "identity", "tangent_escape", true, "goal"),
    make_rmp_node_config("damping", "root", "identity", "damping", true)
  };
  config.target.accel_p_gain = 30.0;
  config.target.accel_d_gain = 10.0;
  config.target.accel_norm_eps = 0.05;
  config.target.proximity_metric_boost_scalar = 1.0;
  config.damping.accel_d_gain = 2.0;
  config.damping.metric_scalar = 0.001;
  config.damping.inertia = 0.01;
  auto & escape = config.tangent_escape;
  escape.enabled = enabled;
  escape.control_dt = 0.01;
  return config;
}

std::unordered_map<std::string, Eigen::Vector3d> targets()
{
  return {
    {"goal", Eigen::Vector3d(0.45, -0.25, 0.75)},
    {"orientation_goal_x", Eigen::Vector3d::UnitX()},
    {"orientation_goal_y", Eigen::Vector3d::UnitY()},
    {"orientation_goal_z", Eigen::Vector3d::UnitZ()}
  };
}

ObstacleSphere obstacle_for_control_point(
  const KinematicsContext & context,
  std::size_t control_point_index,
  double clearance)
{
  const auto & spec = RB10Model::sensor_control_points.at(control_point_index);
  const Eigen::Vector3d normal =
    (context.link_rotations.at(spec.parent_link) * spec.local_normal).normalized();
  ObstacleSphere obstacle;
  obstacle.radius = 0.03;
  obstacle.center = context.control_points.at(control_point_index).position + normal * (
    context.control_points.at(control_point_index).radius + obstacle.radius + clearance);
  obstacle.proximity_control_point_index = static_cast<int>(control_point_index);
  obstacle.source_id = 7;
  return obstacle;
}

void expect_equivalent(
  const RmpSolveResult & pinocchio,
  const RmpSolveResult & casadi,
  const std::string & label)
{
  const double metric_scale = std::max(pinocchio.metric.norm(), casadi.metric.norm());
  const double force_scale = std::max(pinocchio.force.norm(), casadi.force.norm());
  const double qdd_scale = std::max(pinocchio.qdd.norm(), casadi.qdd.norm());
  const double metric_relative_error =
    (pinocchio.metric - casadi.metric).norm() / (1.0 + metric_scale);
  const double force_relative_error =
    (pinocchio.force - casadi.force).norm() / (1.0 + force_scale);
  const double qdd_relative_error =
    (pinocchio.qdd - casadi.qdd).norm() / (1.0 + qdd_scale);

  EXPECT_LT(metric_relative_error, 1e-8) << label;
  EXPECT_LT(force_relative_error, 1e-8) << label;
  EXPECT_LT(qdd_relative_error, 1e-7) << label;
}

void compare_config(
  const EigenRmpConfig & config,
  const std::vector<ObstacleSphere> & obstacles)
{
  const PinocchioDirectRmpSolver pinocchio(config, test_urdf_path());
  const CasadiRmp2Solver casadi(config, test_urdf_path());
  const std::array<JointVector, 3> positions{
    (JointVector() << 0.2, -0.35, -1.45, 0.25, -0.75, 0.15).finished(),
    (JointVector() << -0.1, 0.2, -1.15, -0.35, -0.45, -0.2).finished(),
    (JointVector() << 0.45, -0.15, -1.7, 0.4, -0.9, 0.3).finished()
  };
  const std::array<JointVector, 3> velocities{
    (JointVector() << 0.08, -0.06, 0.04, -0.03, 0.02, -0.01).finished(),
    (JointVector() << -0.12, 0.09, -0.07, 0.05, -0.03, 0.02).finished(),
    (JointVector() << 0.02, 0.03, -0.05, 0.07, -0.04, -0.06).finished()
  };

  for (std::size_t sample = 0; sample < positions.size(); ++sample) {
    const auto pinocchio_result = pinocchio.solve(
      positions[sample], velocities[sample], targets(), obstacles);
    const auto casadi_result = casadi.solve(
      positions[sample], velocities[sample], targets(), obstacles);
    expect_equivalent(
      pinocchio_result, casadi_result, "sample=" + std::to_string(sample));
  }
}

TEST(SolverBackendEquivalence, TargetAndCurvature)
{
  auto config = base_config();
  config.graph_nodes = {
    make_rmp_node_config("tcp_position", "root", "tcp_position", "none", true),
    make_rmp_node_config("target", "tcp_position", "identity", "target", true, "goal")
  };
  compare_config(config, {});
}

TEST(SolverBackendEquivalence, Collision)
{
  auto config = base_config();
  config.graph_nodes = {
    make_rmp_node_config("control_points", "root", "control_points", "none", true),
    make_rmp_node_config(
      "collision", "control_points", "collision_distance", "collision", true)
  };
  const CasadiRmp2Solver geometry_solver(config, test_urdf_path());
  JointVector q;
  q << 0.2, -0.35, -1.45, 0.25, -0.75, 0.15;
  const auto context = geometry_solver.forward_context(q, JointVector::Zero());
  compare_config(config, {obstacle_for_control_point(context, 2, 0.08)});
}

TEST(SolverBackendEquivalence, CompleteNonEscapeGraph)
{
  auto config = base_config();
  config.graph_nodes = {
    make_rmp_node_config("cspace_target", "root", "cspace_target", "cspace_target", true),
    make_rmp_node_config("joint_limits", "root", "joint_limit", "joint_limit", true),
    make_rmp_node_config(
      "joint_velocity_cap", "root", "identity", "joint_velocity_cap", true),
    make_rmp_node_config("tcp_position", "root", "tcp_position", "none", true),
    make_rmp_node_config("target", "tcp_position", "identity", "target", true, "goal"),
    make_rmp_node_config("control_points", "root", "control_points", "none", true),
    make_rmp_node_config(
      "collision", "control_points", "collision_distance", "collision", true),
    make_rmp_node_config("damping", "root", "identity", "damping", true),
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
  const CasadiRmp2Solver geometry_solver(config, test_urdf_path());
  JointVector q;
  q << 0.2, -0.35, -1.45, 0.25, -0.75, 0.15;
  const auto context = geometry_solver.forward_context(q, JointVector::Zero());
  compare_config(config, {obstacle_for_control_point(context, 2, 0.08)});
}

TEST(SolverBackendEquivalence, StaticObstaclePrediction)
{
  auto config = escape_config(true);
  config.tangent_escape_ode.weight_predicted_risk = 1.0;
  const CasadiRmp2Solver geometry_solver(config, test_urdf_path());
  const auto context = geometry_solver.forward_context(
    JointVector::Zero(), JointVector::Zero());
  compare_config(config, {
    obstacle_for_control_point(context, 0, 0.02),
    obstacle_for_control_point(context, RB10Model::sensor_control_points.size() - 1, 0.04)});
}

TEST(SolverBackendEquivalence, StatefulTangentEscape)
{
  const auto nominal_config = escape_config(false);
  const CasadiRmp2Solver nominal_solver(nominal_config, test_urdf_path());
  const JointVector q = JointVector::Zero();
  const JointVector qd = JointVector::Zero();
  const auto context = nominal_solver.forward_context(q, qd);

  const std::array<Eigen::Vector3d, 6> directions{
    Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX(),
    Eigen::Vector3d::UnitY(), -Eigen::Vector3d::UnitY(),
    Eigen::Vector3d::UnitZ(), -Eigen::Vector3d::UnitZ()};
  double best_score = 0.0;
  std::size_t best_point = 0;
  Eigen::Vector3d best_motion = Eigen::Vector3d::Zero();
  Eigen::Vector3d best_goal = context.tcp_position;
  for (const auto & direction : directions) {
    const Eigen::Vector3d goal = context.tcp_position + 0.25 * direction;
    const auto nominal = nominal_solver.solve(q, qd, {{"goal", goal}}, {});
    for (std::size_t point = 0; point < context.control_points.size(); ++point) {
      const Eigen::Vector3d motion = context.control_point_jacobians[point] *
        (qd + 0.01 * nominal.qdd);
      if (motion.norm() <= 1e-10) {
        continue;
      }
      const Eigen::Vector3d approach = motion.normalized();
      const double score = motion.norm() *
        ((Eigen::Matrix3d::Identity() - approach * approach.transpose()) *
        context.control_point_jacobians[point]).norm();
      if (score > best_score) {
        best_score = score;
        best_point = point;
        best_motion = motion;
        best_goal = goal;
      }
    }
  }
  ASSERT_GT(best_score, 1e-8);

  const auto & point = context.control_points[best_point];
  constexpr double obstacle_radius = 0.04;
  constexpr double clearance = 0.02;
  const ObstacleSphere obstacle{
    point.position +
    (point.radius + obstacle_radius + clearance) * best_motion.normalized(),
    obstacle_radius,
    static_cast<int>(best_point),
    101};
  const std::unordered_map<std::string, Eigen::Vector3d> goal{{"goal", best_goal}};

  const auto config = escape_config(true);
  PinocchioDirectRmpSolver pinocchio(config, test_urdf_path());
  CasadiRmp2Solver casadi(config, test_urdf_path());
  bool escape_activated = false;
  for (int cycle = 0; cycle < 8; ++cycle) {
    const auto pinocchio_result = pinocchio.solve(q, qd, goal, {obstacle});
    const auto casadi_result = casadi.solve(q, qd, goal, {obstacle});
    expect_equivalent(
      pinocchio_result, casadi_result, "escape cycle=" + std::to_string(cycle));
    if (
      !pinocchio_result.tangent_escape_rmp_data.empty() &&
      pinocchio_result.tangent_escape_rmp_data[0] > 0.5)
    {
      escape_activated = true;
    }
  }
  EXPECT_TRUE(escape_activated);
}

TEST(SolverBackendEquivalence, SensorHistoryAndSimpleTangentWithCollision)
{
  for (const auto mode : {tangent_sensor_history::DirectionMode::kSensorHistory,
      tangent_sensor_history::DirectionMode::kSimple})
  {
    auto config = escape_config(true);
    config.graph_nodes.insert(config.graph_nodes.begin() + 3,
      make_rmp_node_config("collision", "control_points", "identity", "collision", true));
    config.tangent_escape_ode.direction_mode = mode;
    config.tangent_escape_ode.block_on_dot = -2.0;
    config.tangent_escape_ode.block_off_dot = -1.0;
    config.tangent_escape_ode.alpha_off_distance = 0.45;
    PinocchioDirectRmpSolver pinocchio(config, test_urdf_path());
    CasadiRmp2Solver casadi(config, test_urdf_path());
    JointVector q;
    q << 0.2, -0.35, -1.45, 0.25, -0.75, 0.15;
    for (int cycle = 0; cycle < 55; ++cycle) {
      tangent_sensor_history::Input input;
      input.stamp = 100.0 + cycle * 0.025;
      input.q = q;
      input.q[0] += 0.02 * std::sin(0.12 * cycle);
      input.velocity_available = true;
      input.qd[0] = 0.096 * std::cos(0.12 * cycle);
      input.samples[0] = {input.stamp, 0.08 - 0.4 * (input.q[0] - q[0])};
      input.samples[5] = {input.stamp, 0.15 + 0.3 * (input.q[0] - q[0])};
      pinocchio.set_tangent_observations(input);
      casadi.set_tangent_observations(input);
      if (cycle < 40) { continue; }
      const auto context = RB10Model::forward_context(input.q);
      const std::vector<ObstacleSphere> obstacles{
        obstacle_for_control_point(context, 0, input.samples[0].range),
        obstacle_for_control_point(context, 5, input.samples[5].range)};
      // Exercise an actually blocked goal under the history angle gate.
      // Legacy block_on/off values do not force that gate open.
      auto blocked_targets = targets();
      const auto & spec = RB10Model::sensor_control_points[0];
      const Eigen::Vector3d normal = context.link_rotations[spec.parent_link] * spec.local_normal;
      const Eigen::Matrix3d regularized = context.tcp_jacobian * context.tcp_jacobian.transpose() +
        .0001 * Eigen::Matrix3d::Identity();
      const Eigen::Matrix3d mapping = context.control_point_jacobians[0] *
        context.tcp_jacobian.transpose() * regularized.ldlt().solve(Eigen::Matrix3d::Identity());
      const Eigen::Vector3d blocked_motion = .2 * normal + .06 *
        (context.link_rotations[spec.parent_link] * spec.local_tangent_bias);
      const Eigen::Vector3d goal_step = mapping.fullPivLu().solve(blocked_motion);
      ASSERT_LT((mapping * goal_step - blocked_motion).norm(),1e-8);
      blocked_targets["goal"] = context.tcp_position + goal_step;
      const auto first = pinocchio.solve(input.q, JointVector::Zero(), blocked_targets, obstacles);
      const auto second = casadi.solve(input.q, JointVector::Zero(), blocked_targets, obstacles);
      expect_equivalent(first, second, "mode=" + std::to_string(static_cast<int>(mode)) +
        " cycle=" + std::to_string(cycle));
      ASSERT_GT(first.tangent_escape_rmp_data.size(), tangent_escape_debug::kHistory);
      EXPECT_GT(first.tangent_escape_rmp_data[10], 0.0);
      EXPECT_DOUBLE_EQ(first.tangent_escape_rmp_data[tangent_escape_debug::kHistory],
        static_cast<int>(mode));
      if (mode == tangent_sensor_history::DirectionMode::kSimple) {
        EXPECT_DOUBLE_EQ(first.tangent_escape_rmp_data[tangent_escape_debug::kSteering + 2], 1.0);
        EXPECT_NEAR(first.tangent_escape_rmp_data[tangent_escape_debug::kSteering + 1], 0.0, 1e-10);
      }
    }
  }
}

TEST(SolverBackendEquivalence, PinocchioDiagnosticsAreOptIn)
{
  auto config = base_config();
  config.graph_nodes = {
    make_rmp_node_config("tcp_position", "root", "tcp_position", "none", true),
    make_rmp_node_config("target", "tcp_position", "identity", "target", true, "goal"),
    make_rmp_node_config("damping", "root", "identity", "damping", true)
  };
  JointVector q;
  q << 0.2, -0.35, -1.45, 0.25, -0.75, 0.15;
  const JointVector qd = JointVector::Zero();

  const PinocchioDirectRmpSolver lean_solver(config, test_urdf_path());
  const auto lean = lean_solver.solve(q, qd, targets(), {});
  EXPECT_TRUE(lean.tangent_escape_dual_solve_data.empty());
  EXPECT_TRUE(lean.leaf_ablation_data.empty());
  EXPECT_TRUE(lean.stability_certificate_data.empty());

  config.enable_tangent_escape_dual_solve_diagnostics = true;
  config.enable_leaf_ablation_diagnostics = true;
  config.enable_stability_certificate_diagnostics = true;
  const PinocchioDirectRmpSolver diagnostic_solver(config, test_urdf_path());
  (void)diagnostic_solver.solve(q, qd, targets(), {});
  const auto diagnostic = diagnostic_solver.solve(q, qd, targets(), {});
  EXPECT_EQ(diagnostic.tangent_escape_dual_solve_data.size(), 43U);
  EXPECT_GE(diagnostic.leaf_ablation_data.size(), 25U);
  EXPECT_EQ(diagnostic.stability_certificate_data.size(), 35U);
  const auto verify_recorded_groups = [](const auto & solution) {
      const auto & data = solution.leaf_ablation_data;
      ASSERT_GE(data.size(), 25U);
      const auto groups = static_cast<std::size_t>(data[1]);
      std::size_t offset = 25 + 17 * groups;
      ASSERT_GT(data.size(), offset + 1);
      ASSERT_DOUBLE_EQ(data[offset], 42042.0);
      ASSERT_DOUBLE_EQ(data[offset + 1], groups);
      offset += 2;
      Eigen::Matrix<double,6,6> metric = Eigen::Matrix<double,6,6>::Zero();
      JointVector force = JointVector::Zero();
      for (std::size_t i=0; i<groups; ++i) {
        ASSERT_GE(data.size(), offset + 43);
        ++offset;
        for (int row=0; row<6; ++row) {
          for (int col=0; col<6; ++col) { metric(row,col) += data[offset++]; }
        }
        for (int j=0; j<6; ++j) { force[j] += data[offset++]; }
      }
      EXPECT_LT((metric - solution.metric).norm(), 1e-6);
      EXPECT_LT((force - solution.force).norm(), 1e-6);
      ASSERT_GT(data.size(), offset);
      ASSERT_DOUBLE_EQ(data[offset], 44044.0);
    };
  verify_recorded_groups(diagnostic);
  const CasadiRmp2Solver diagnostic_casadi(config, test_urdf_path());
  verify_recorded_groups(diagnostic_casadi.solve(q, qd, targets(), {}));
}

}  // namespace
}  // namespace rb10_rmpflow_rviz

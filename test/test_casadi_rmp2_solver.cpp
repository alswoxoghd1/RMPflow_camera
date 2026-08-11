#include <array>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>

#include "rb10_rmpflow_rviz/casadi_rmp2_solver.hpp"

namespace rb10_rmpflow_rviz
{
namespace
{

std::string test_urdf_path()
{
  return RB10_TEST_URDF_PATH;
}

EigenRmpConfig minimal_target_collision_config()
{
  EigenRmpConfig config;
  config.solve_method = "rmp2";
  config.rmp_type = "canonical";
  config.casadi_rmp2_max_obstacles = 2;
  config.graph_nodes = {
    make_rmp_node_config("tcp_position", "root", "tcp_position", "none", true),
    make_rmp_node_config("target", "tcp_position", "identity", "target", true, "goal"),
    make_rmp_node_config("control_points", "root", "control_points", "none", true),
    make_rmp_node_config("collision", "control_points", "collision_distance", "collision", true)
  };
  return config;
}

EigenRmpConfig full_static_config()
{
  EigenRmpConfig config = minimal_target_collision_config();
  config.graph_nodes = {
    make_rmp_node_config("cspace_target", "root", "cspace_target", "cspace_target", true),
    make_rmp_node_config("joint_limits", "root", "joint_limit", "joint_limit", true),
    make_rmp_node_config(
      "joint_velocity_cap", "root", "identity", "joint_velocity_cap", true),
    make_rmp_node_config("tcp_position", "root", "tcp_position", "none", true),
    make_rmp_node_config("target", "tcp_position", "identity", "target", true, "goal"),
    make_rmp_node_config("control_points", "root", "control_points", "none", true),
    make_rmp_node_config("collision", "control_points", "collision_distance", "collision", true),
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
  config.default_q = {0.2, -0.3, -1.2, 0.4, -0.8, 0.1};
  config.joint_lower_limits = {-3.0, -1.3, -2.8, -1.5, -3.0, -6.0};
  config.joint_upper_limits = {3.0, 1.3, -0.6, 1.5, 2.0, 6.0};
  config.joint_limit_buffers = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
  return config;
}

EigenRmpConfig escape_parity_config(bool enabled)
{
  EigenRmpConfig config;
  config.solve_method = "rmp2";
  config.rmp_type = "canonical";
  config.solve_offset = 1e-4;
  config.max_joint_accel = 1e6;
  config.casadi_rmp2_max_obstacles = 2;
  config.enable_tangent_escape_diagnostics = true;
  config.enable_tangent_escape_dual_solve_diagnostics = true;
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

std::vector<ObstacleSphere> positive_clearance_obstacles(
  const KinematicsContext & context,
  double margin = 0.0)
{
  constexpr double obstacle_radius = 0.03;
  constexpr std::array<double, 4> desired_clearances{0.04, 0.08, 0.12, 0.18};
  for (std::size_t source = 0; source < RB10Model::sensor_control_points.size(); ++source) {
    const auto & spec = RB10Model::sensor_control_points[source];
    const Eigen::Vector3d normal =
      (context.link_rotations[spec.parent_link] * spec.local_normal).normalized();
    for (const double desired_clearance : desired_clearances) {
      ObstacleSphere obstacle;
      obstacle.center = context.control_points[source].position + normal * (
        context.control_points[source].radius + obstacle_radius + margin + desired_clearance);
      obstacle.radius = obstacle_radius;
      obstacle.proximity_control_point_index = static_cast<int>(source);
      obstacle.source_id = 11;

      double minimum_clearance = std::numeric_limits<double>::infinity();
      for (const auto & point : context.control_points) {
        minimum_clearance = std::min(
          minimum_clearance,
          (point.position - obstacle.center).norm() - point.radius - obstacle.radius - margin);
      }
      if (minimum_clearance > 1e-3 && minimum_clearance < 0.45) {
        return {obstacle};
      }
    }
  }
  throw std::runtime_error("Unable to construct a positive-clearance collision test obstacle");
}

void expect_valid_root(const RmpSolveResult & result)
{
  EXPECT_TRUE(result.metric.allFinite());
  EXPECT_TRUE(result.force.allFinite());
  EXPECT_TRUE(result.qdd.allFinite());
  EXPECT_TRUE(result.metric.isApprox(result.metric.transpose(), 1e-9));
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver(
    0.5 * (result.metric + result.metric.transpose()));
  ASSERT_EQ(eigensolver.info(), Eigen::Success);
  EXPECT_GE(eigensolver.eigenvalues().minCoeff(), -1e-8);
}

void exercise_config(const EigenRmpConfig & config, int sample_count)
{
  const CasadiRmp2Solver solver(config, test_urdf_path());
  std::mt19937 generator(20260722);
  std::uniform_real_distribution<double> small(-0.2, 0.2);

  for (int sample = 0; sample < sample_count; ++sample) {
    RB10Model::JointVector q;
    q << 0.3, -0.4, -1.4, 0.35, -0.7, 0.2;
    RB10Model::JointVector qd;
    for (int joint = 0; joint < 6; ++joint) {
      q[joint] += small(generator);
      qd[joint] = small(generator);
    }
    SCOPED_TRACE(
      "sample=" + std::to_string(sample) + " q=" +
      std::to_string(q[0]) + "," + std::to_string(q[1]) + "," +
      std::to_string(q[2]) + "," + std::to_string(q[3]) + "," +
      std::to_string(q[4]) + "," + std::to_string(q[5]));
    const auto sample_obstacles = positive_clearance_obstacles(
      solver.forward_context(q, qd), config.collision.margin);
    const auto result = solver.solve(q, qd, targets(), sample_obstacles);
    expect_valid_root(result);
  }
}

TEST(CasadiRmp2Solver, TargetCollisionGraphIsFiniteAndPsd)
{
  exercise_config(minimal_target_collision_config(), 8);
}

TEST(CasadiRmp2Solver, TargetOnlyGraphIsFiniteAndPsd)
{
  auto config = minimal_target_collision_config();
  config.graph_nodes.resize(2);
  exercise_config(config, 2);
}

TEST(CasadiRmp2Solver, CollisionOnlyGraphIsFiniteAndPsd)
{
  auto config = minimal_target_collision_config();
  config.graph_nodes.erase(config.graph_nodes.begin(), config.graph_nodes.begin() + 2);
  exercise_config(config, 8);
}

TEST(CasadiRmp2Solver, CombinedRootEqualsSumOfIndependentRoots)
{
  auto target_config = minimal_target_collision_config();
  target_config.graph_nodes.resize(2);
  auto collision_config = minimal_target_collision_config();
  collision_config.graph_nodes.erase(
    collision_config.graph_nodes.begin(), collision_config.graph_nodes.begin() + 2);
  const CasadiRmp2Solver target_solver(target_config, test_urdf_path());
  const CasadiRmp2Solver collision_solver(collision_config, test_urdf_path());
  const CasadiRmp2Solver combined_solver(minimal_target_collision_config(), test_urdf_path());
  RB10Model::JointVector q;
  q << 0.3, -0.4, -1.4, 0.35, -0.7, 0.2;
  RB10Model::JointVector qd;
  qd << 0.1, -0.1, 0.08, -0.04, 0.02, -0.03;
  const auto sample_obstacles = positive_clearance_obstacles(
    combined_solver.forward_context(q, qd));
  const auto target = target_solver.solve(q, qd, targets(), sample_obstacles);
  const auto collision = collision_solver.solve(q, qd, targets(), sample_obstacles);
  const auto combined = combined_solver.solve(q, qd, targets(), sample_obstacles);
  EXPECT_LT((combined.metric - target.metric - collision.metric).norm(), 1e-8) <<
    "target=" << target.metric.norm() << " collision=" << collision.metric.norm() <<
    " combined=" << combined.metric.norm();
  EXPECT_LT((combined.force - target.force - collision.force).norm(), 1e-7) <<
    "target=" << target.force.norm() << " collision=" << collision.force.norm() <<
    " combined=" << combined.force.norm();

}

TEST(CasadiRmp2Solver, FullStaticCanonicalGraphIsFiniteAndPsd)
{
  exercise_config(full_static_config(), 8);
}

TEST(CasadiRmp2Solver, ZeroVelocityDoesNotPoisonInactiveVelocityCap)
{
  EigenRmpConfig config = full_static_config();
  config.joint_velocity_cap.max_velocity = 1.0;
  config.joint_velocity_cap.velocity_damping_region = 0.5;
  config.joint_velocity_cap.eps = 1e-3;

  const RB10Model::JointVector q =
    (RB10Model::JointVector() << 0.2, -0.3, -1.2, 0.4, -0.8, 0.1).finished();
  const RB10Model::JointVector qd = RB10Model::JointVector::Zero();
  CasadiRmp2Solver solver(config, test_urdf_path());

  const auto result = solver.solve(q, qd, targets(), {});
  expect_valid_root(result);
  EXPECT_GT(result.qdd.norm(), 1e-6);
}

TEST(CasadiRmp2Solver, SignedClearanceRemainsRepulsiveDuringPenetration)
{
  auto config = minimal_target_collision_config();
  config.graph_nodes.erase(config.graph_nodes.begin(), config.graph_nodes.begin() + 2);
  const CasadiRmp2Solver solver(config, test_urdf_path());
  RB10Model::JointVector q;
  q << 0.3, -0.4, -1.4, 0.35, -0.7, 0.2;
  RB10Model::JointVector qd;
  qd << 0.1, -0.1, 0.08, -0.04, 0.02, -0.03;
  const auto context = solver.forward_context(q, qd);
  const std::size_t point_index = 7;
  const auto & spec = RB10Model::sensor_control_points[point_index];
  const Eigen::Vector3d normal =
    (context.link_rotations[spec.parent_link] * spec.local_normal).normalized();
  const ObstacleSphere obstacle{
    context.control_points[point_index].position + 0.03 * normal,
    0.03,
    static_cast<int>(point_index),
    22};

  const auto result = solver.solve(q, qd, targets(), {obstacle});
  EXPECT_TRUE(result.metric.allFinite());
  EXPECT_TRUE(result.force.allFinite());
  EXPECT_TRUE(result.qdd.allFinite());
  EXPECT_GT(result.metric.norm(), 1e-6);
  EXPECT_GT(result.force.norm(), 1e-6);
  EXPECT_GE(result.metric.selfadjointView<Eigen::Lower>().eigenvalues().minCoeff(), -1e-8);
}

TEST(CasadiRmp2Solver, EscapeSupervisorProducesTangentRankOneContribution)
{
  const auto disabled_config = escape_parity_config(false);
  const CasadiRmp2Solver nominal_solver(disabled_config, test_urdf_path());
  RB10Model::JointVector q = RB10Model::JointVector::Zero();
  RB10Model::JointVector qd = RB10Model::JointVector::Zero();
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
      const double speed = motion.norm();
      if (speed <= 1e-10) {
        continue;
      }
      const Eigen::Vector3d approach = motion / speed;
      const Eigen::Matrix3d tangent_projector =
        Eigen::Matrix3d::Identity() - approach * approach.transpose();
      const double score = speed *
        (tangent_projector * context.control_point_jacobians[point]).norm();
      if (score > best_score) {
        best_score = score;
        best_point = point;
        best_motion = motion;
        best_goal = goal;
      }
    }
  }
  ASSERT_GT(best_score, 1e-8);

  const double obstacle_radius = 0.04;
  const double clearance = 0.02;
  const auto & point = context.control_points[best_point];
  const ObstacleSphere obstacle{
    point.position +
    (point.radius + obstacle_radius + clearance) * best_motion.normalized(),
    obstacle_radius,
    static_cast<int>(best_point),
    101};
  const std::unordered_map<std::string, Eigen::Vector3d> goal{{"goal", best_goal}};

  const auto enabled_config = escape_parity_config(true);
  CasadiRmp2Solver solver(enabled_config, test_urdf_path());
  auto lean_config = enabled_config;
  lean_config.enable_tangent_escape_diagnostics = false;
  lean_config.enable_tangent_escape_dual_solve_diagnostics = false;
  CasadiRmp2Solver lean_solver(lean_config, test_urdf_path());
  const auto without_escape = nominal_solver.solve(q, qd, goal, {obstacle});
  bool observed_active = false;
  for (int cycle = 0; cycle < 8; ++cycle) {
    const auto result = solver.solve(q, qd, goal, {obstacle});
    const auto lean_result = lean_solver.solve(q, qd, goal, {obstacle});
    expect_valid_root(result);
    expect_valid_root(lean_result);
    EXPECT_TRUE(lean_result.metric.isApprox(result.metric, 1e-10));
    EXPECT_TRUE(lean_result.force.isApprox(result.force, 1e-10));
    EXPECT_TRUE(lean_result.qdd.isApprox(result.qdd, 1e-10));
    EXPECT_TRUE(lean_result.tangent_escape_rmp_data.empty());
    EXPECT_TRUE(lean_result.tangent_escape_dual_solve_data.empty());
    if (
      result.tangent_escape_rmp_data.size() >= 23 &&
      result.tangent_escape_rmp_data[0] > 0.5)
    {
      observed_active = true;
      const Eigen::Vector3d normal(
        result.tangent_escape_rmp_data[17],
        result.tangent_escape_rmp_data[18],
        result.tangent_escape_rmp_data[19]);
      const Eigen::Vector3d tangent(
        result.tangent_escape_rmp_data[20],
        result.tangent_escape_rmp_data[21],
        result.tangent_escape_rmp_data[22]);
      EXPECT_NEAR(tangent.norm(), 1.0, 5e-3);
      EXPECT_NEAR(normal.dot(tangent), 0.0, 2e-2);
      EXPECT_GT((result.metric - without_escape.metric).norm(), 1e-8);
      EXPECT_EQ(result.tangent_escape_dual_solve_data.size(), 43U);
    }
  }
  EXPECT_TRUE(observed_active);
}

TEST(CasadiRmp2Solver, GeneratesStandaloneCSource)
{
  EigenRmpConfig config = minimal_target_collision_config();
  config.casadi_rmp2_max_obstacles = 1;
  const CasadiRmp2Solver solver(config, test_urdf_path());
  const std::filesystem::path output_base =
    std::filesystem::temp_directory_path() / "rb10_rmp2_test_generated";
  const std::string generated = solver.generate_c_code(output_base.string());
  EXPECT_FALSE(generated.empty());
  EXPECT_TRUE(std::filesystem::exists(output_base.string() + ".c"));
  EXPECT_TRUE(std::filesystem::exists(output_base.string() + ".h"));
  std::filesystem::remove(output_base.string() + ".c");
  std::filesystem::remove(output_base.string() + ".h");
}

TEST(CasadiRmp2Solver, SupportsDiagnosticModes)
{
  EigenRmpConfig config = minimal_target_collision_config();
  config.enable_leaf_ablation_diagnostics = true;
  config.enable_stability_certificate_diagnostics = true;
  config.escape_energy_certificate.enabled = true;
  const CasadiRmp2Solver solver(config, test_urdf_path());
  RB10Model::JointVector q;
  q << 0.3, -0.4, -1.4, 0.35, -0.7, 0.2;
  const RB10Model::JointVector qd = RB10Model::JointVector::Zero();
  const auto sample_obstacles = positive_clearance_obstacles(
    solver.forward_context(q, qd));
  (void)solver.solve(q, qd, targets(), sample_obstacles);
  const auto result = solver.solve(q, qd, targets(), sample_obstacles);
  EXPECT_GE(result.leaf_ablation_data.size(), 25U);
  EXPECT_EQ(result.stability_certificate_data.size(), 35U);
  EXPECT_GT(result.stability_certificate_data[2], 0.5);
}

}  // namespace
}  // namespace rb10_rmpflow_rviz

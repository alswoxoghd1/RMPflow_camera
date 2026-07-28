#include <random>
#include <string>

#include <gtest/gtest.h>

#include "rb10_rmpflow_rviz/casadi_rb10_kinematics.hpp"
#include "rb10_rmpflow_rviz/rb10_model.hpp"

namespace rb10_rmpflow_rviz
{
namespace
{

std::string test_urdf_path()
{
  return RB10_TEST_URDF_PATH;
}

Eigen::Matrix<double, 3, 6> expected_angular_jacobian(
  const KinematicsContext & context,
  std::size_t link_index)
{
  Eigen::Matrix<double, 3, 6> jacobian =
    Eigen::Matrix<double, 3, 6>::Zero();
  const int active_joints = RB10Model::active_joint_count_for_link(link_index);
  for (int joint = 0; joint < active_joints; ++joint) {
    jacobian.col(joint) = context.joint_axes[static_cast<std::size_t>(joint)];
  }
  return jacobian;
}

TEST(CasadiRb10Kinematics, MatchesExplicitGeometryAndDifferentialIdentities)
{
  const CasadiRb10Kinematics model(test_urdf_path());
  std::mt19937 generator(20260722);
  std::uniform_real_distribution<double> q_distribution(-1.5, 1.5);
  std::uniform_real_distribution<double> qd_distribution(-0.4, 0.4);
  constexpr double finite_difference_step = 1e-6;

  for (int sample = 0; sample < 24; ++sample) {
    RB10Model::JointVector q;
    RB10Model::JointVector qd;
    for (int joint = 0; joint < 6; ++joint) {
      q[joint] = q_distribution(generator);
      qd[joint] = qd_distribution(generator);
    }

    const auto expected = RB10Model::forward_context(
      q, ControlPointJacobianMode::kAnalytic);
    const auto actual = model.forward_context(q, qd);
    const auto plus = model.forward_context(q + finite_difference_step * qd, qd);
    const auto minus = model.forward_context(q - finite_difference_step * qd, qd);

    for (std::size_t index = 0; index < RB10Model::LINK_COUNT; ++index) {
      EXPECT_TRUE(actual.link_positions[index].isApprox(expected.link_positions[index], 1e-10));
      EXPECT_TRUE(actual.link_rotations[index].isApprox(expected.link_rotations[index], 1e-10));
      EXPECT_TRUE(actual.link_jacobians[index].isApprox(expected.link_jacobians[index], 1e-9));
      EXPECT_TRUE(
        actual.link_velocities[index].isApprox(
          actual.link_jacobians[index] * qd, 1e-10));
      const Eigen::Vector3d finite_difference_curvature =
        ((plus.link_jacobians[index] - minus.link_jacobians[index]) /
        (2.0 * finite_difference_step)) * qd;
      EXPECT_LT(
        (actual.link_curvatures[index] - finite_difference_curvature).norm(),
        2e-7);

      const auto angular_jacobian = expected_angular_jacobian(expected, index);
      EXPECT_TRUE(
        actual.link_angular_jacobians[index].isApprox(angular_jacobian, 1e-9));
      EXPECT_TRUE(
        actual.link_angular_velocities[index].isApprox(
          actual.link_angular_jacobians[index] * qd, 1e-10));
      const Eigen::Vector3d finite_difference_angular_curvature =
        ((plus.link_angular_jacobians[index] - minus.link_angular_jacobians[index]) /
        (2.0 * finite_difference_step)) * qd;
      EXPECT_LT(
        (actual.link_angular_curvatures[index] -
        finite_difference_angular_curvature).norm(),
        2e-7) <<
        "sample=" << sample << " link=" << index << " error=" <<
        (actual.link_angular_curvatures[index] -
        finite_difference_angular_curvature).norm();
    }

    ASSERT_EQ(actual.control_points.size(), expected.control_points.size());
    for (std::size_t index = 0; index < actual.control_points.size(); ++index) {
      EXPECT_LT(
        (actual.control_points[index].position -
        expected.control_points[index].position).norm(),
        1e-6) <<
        "sample=" << sample << " point=" << index << " error=" <<
        (actual.control_points[index].position -
        expected.control_points[index].position).norm();
      EXPECT_LT(
        (actual.control_point_jacobians[index] -
        expected.control_point_jacobians[index]).norm(),
        1e-6) <<
        "sample=" << sample << " point=" << index << " error=" <<
        (actual.control_point_jacobians[index] -
        expected.control_point_jacobians[index]).norm();
      EXPECT_TRUE(
        actual.control_point_velocities[index].isApprox(
          actual.control_point_jacobians[index] * qd, 1e-10));
      const Eigen::Vector3d finite_difference_curvature =
        ((plus.control_point_jacobians[index] -
        minus.control_point_jacobians[index]) /
        (2.0 * finite_difference_step)) * qd;
      EXPECT_LT(
        (actual.control_point_curvatures[index] -
        finite_difference_curvature).norm(),
        2e-7);
    }
  }
}

}  // namespace
}  // namespace rb10_rmpflow_rviz

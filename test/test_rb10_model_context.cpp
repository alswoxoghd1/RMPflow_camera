#include <array>

#include <gtest/gtest.h>

#include "rb10_rmpflow_rviz/rb10_model.hpp"

namespace rb10_rmpflow_rviz
{
namespace
{

using JointVector = RB10Model::JointVector;

constexpr double kGeometryTolerance = 1e-12;
constexpr double kJacobianTolerance = 2e-5;

const std::array<JointVector, 5> kTestConfigurations{{
  JointVector::Zero(),
  (JointVector() << 0.2, -0.6, 0.8, -0.3, 0.4, -0.2).finished(),
  (JointVector() << -0.8, 0.5, -1.0, 0.7, -0.4, 0.9).finished(),
  (JointVector() << 1.2, -1.1, 0.3, 0.9, -0.7, -0.5).finished(),
  (JointVector() << -1.4, 0.9, -0.5, -0.8, 1.0, 0.6).finished(),
}};

void expect_same_geometry(
  const KinematicsContext & reference,
  const KinematicsContext & candidate)
{
  for (std::size_t index = 0; index < RB10Model::LINK_COUNT; ++index) {
    EXPECT_LE(
      (reference.link_positions[index] - candidate.link_positions[index]).norm(),
      kGeometryTolerance);
    EXPECT_LE(
      (reference.link_rotations[index] - candidate.link_rotations[index]).norm(),
      kGeometryTolerance);
  }

  for (std::size_t index = 0; index < reference.joint_origins.size(); ++index) {
    EXPECT_LE(
      (reference.joint_origins[index] - candidate.joint_origins[index]).norm(),
      kGeometryTolerance);
    EXPECT_LE(
      (reference.joint_axes[index] - candidate.joint_axes[index]).norm(),
      kGeometryTolerance);
  }

  ASSERT_EQ(reference.control_points.size(), candidate.control_points.size());
  for (std::size_t index = 0; index < reference.control_points.size(); ++index) {
    EXPECT_LE(
      (reference.control_points[index].position - candidate.control_points[index].position).norm(),
      kGeometryTolerance);
    EXPECT_DOUBLE_EQ(reference.control_points[index].radius, candidate.control_points[index].radius);
  }
  EXPECT_LE((reference.tcp_position - candidate.tcp_position).norm(), kGeometryTolerance);
  EXPECT_LE((reference.tcp_jacobian - candidate.tcp_jacobian).norm(), kGeometryTolerance);
}

TEST(RB10ModelContext, PoseOnlyModePreservesAllGeometry)
{
  for (const auto & q : kTestConfigurations) {
    const auto numerical = RB10Model::forward_context(
      q, ControlPointJacobianMode::kNumerical);
    const auto pose_only = RB10Model::forward_context(
      q, ControlPointJacobianMode::kNone);

    expect_same_geometry(numerical, pose_only);
    EXPECT_EQ(numerical.control_point_jacobians.size(), RB10Model::sensor_control_points.size());
    EXPECT_TRUE(pose_only.control_point_jacobians.empty());
  }
}

TEST(RB10ModelContext, AnalyticSensorJacobiansMatchLegacyNumericalJacobians)
{
  for (const auto & q : kTestConfigurations) {
    const auto numerical = RB10Model::forward_context(
      q, ControlPointJacobianMode::kNumerical);
    const auto analytic = RB10Model::forward_context(
      q, ControlPointJacobianMode::kAnalytic);

    expect_same_geometry(numerical, analytic);
    ASSERT_EQ(numerical.control_point_jacobians.size(), analytic.control_point_jacobians.size());
    for (std::size_t index = 0; index < numerical.control_point_jacobians.size(); ++index) {
      EXPECT_LE(
        (numerical.control_point_jacobians[index] -
        analytic.control_point_jacobians[index]).cwiseAbs().maxCoeff(),
        kJacobianTolerance)
        << "sensor index " << index;
    }
  }
}

}  // namespace
}  // namespace rb10_rmpflow_rviz

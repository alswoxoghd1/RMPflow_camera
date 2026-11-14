#include <array>
#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "rb10_rmpflow_rviz/casadi_rb10_kinematics.hpp"
#include "rb10_rmpflow_rviz/pinocchio_model.hpp"
#include "rb10_rmpflow_rviz/tangent_escape_ode.hpp"

namespace rb10_rmpflow_rviz
{
namespace
{

TEST(SoCSensorLayout, UrdfAndBothBackendsHaveTwelveCenteredSensors)
{
  const PinocchioModel pinocchio(RB10_TEST_URDF_PATH);
  const CasadiRb10Kinematics casadi(RB10_TEST_URDF_PATH);
  RB10Model::JointVector q;
  q << 0.2, -0.4, -1.2, 0.3, -0.7, 0.1;
  const auto pin_context = pinocchio.forward_context(q, RB10Model::JointVector::Zero());
  const auto casadi_context = casadi.forward_context(q, RB10Model::JointVector::Zero());
  ASSERT_EQ(pin_context.control_points.size(), 12u);
  ASSERT_EQ(casadi_context.control_points.size(), 12u);
  std::array<std::size_t, RB10Model::LINK_COUNT> counts{};

  for (std::size_t index = 0; index < 12; ++index) {
    const auto & spec = RB10Model::sensor_control_points[index];
    ++counts[spec.parent_link];
    EXPECT_LT(
      (pin_context.control_points[index].position -
      casadi_context.control_points[index].position).norm(), 1e-10);
    EXPECT_LT(
      (pin_context.control_point_jacobians[index] -
      casadi_context.control_point_jacobians[index]).norm(), 1e-10);
    const Eigen::Vector3d local_position =
      pin_context.link_rotations[spec.parent_link].transpose() *
      (pin_context.control_points[index].position -
      pin_context.link_positions[spec.parent_link]);
    if (spec.parent_link == RB10Model::LINK2) {
      EXPECT_NEAR(local_position.z(), 0.6127 / 2.0, 1e-12);
      EXPECT_NEAR(std::hypot(local_position.x(), local_position.y() + 0.1875), 0.079, 1e-12);
    } else if (spec.parent_link == RB10Model::LINK3_5) {
      EXPECT_NEAR(local_position.z(), 0.57015 / 2.0, 1e-12);
      EXPECT_NEAR(std::hypot(local_position.x(), local_position.y()), 0.0645, 1e-12);
    }
  }
  EXPECT_EQ(counts[RB10Model::LINK2], 4u);
  EXPECT_EQ(counts[RB10Model::LINK3_5], 4u);
  EXPECT_EQ(counts[RB10Model::LINK5], 4u);

  for (const auto * removed : {
      "sensor_link2_1", "sensor_link3_1", "tof2_1_N", "tof2_1_S",
      "tof2_1_E", "tof2_1_W", "tof3_1_N", "tof3_1_S", "tof3_1_E", "tof3_1_W"})
  {
    EXPECT_FALSE(pinocchio.model().existFrame(removed)) << removed;
  }
}

TEST(SoCSensorLayout, TangentEscapeNeighborhoodsUseOnlyTheRetainedSensors)
{
  const auto & neighborhoods = tangent_escape_ode::neighborhood_table();
  ASSERT_EQ(neighborhoods.size(), 12u);
  for (std::size_t index = 0; index < neighborhoods.size(); ++index) {
    const auto & entry = neighborhoods[index];
    ASSERT_TRUE(entry.valid) << index;
    EXPECT_EQ(entry.count, 3u) << index;
    for (std::size_t neighbor = 0; neighbor < entry.count; ++neighbor) {
      const auto other = entry.entries[neighbor].index;
      ASSERT_LT(other, 12u);
      EXPECT_NE(index, other);
      EXPECT_EQ(
        RB10Model::sensor_control_points[index].parent_link,
        RB10Model::sensor_control_points[other].parent_link);
    }
  }
}

}  // namespace
}  // namespace rb10_rmpflow_rviz

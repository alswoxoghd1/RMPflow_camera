#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "rb10_rmpflow_rviz/tangent_escape_ode.hpp"
#include "rb10_rmpflow_rviz/tangent_escape_prediction.hpp"
#include "rb10_rmpflow_rviz/tangent_escape_supervisor.hpp"

namespace rb10_rmpflow_rviz
{
namespace
{

namespace prediction = tangent_escape_prediction;
namespace ode = tangent_escape_ode;
constexpr double kPi = 3.14159265358979323846;

KinematicsContext scene()
{
  KinematicsContext context;
  for (std::size_t k = 0; k < context.link_rotations.size(); ++k) {
    context.link_rotations[k].setIdentity();
    context.link_angular_jacobians[k].setZero();
  }
  for (std::size_t k = 0; k < RB10Model::sensor_control_points.size(); ++k) {
    context.control_points.push_back({Eigen::Vector3d(10.0 + k, 10.0, 10.0), 0.01});
    context.control_point_jacobians.push_back(RB10Model::Jacobian::Zero());
    context.control_point_velocities.push_back(Eigen::Vector3d::Zero());
    context.control_point_curvatures.push_back(Eigen::Vector3d::Zero());
  }
  context.control_points[0].position = Eigen::Vector3d(1.0, 0.0, 0.0);
  context.control_points[1].position.setZero();
  context.control_point_jacobians[0].leftCols<3>().setIdentity();
  context.control_point_jacobians[1].leftCols<3>().setIdentity();
  return context;
}

prediction::StaticObstaclePredictor predictor(
  const KinematicsContext & context,
  const std::vector<prediction::StaticObstacle> & obstacles,
  double distance = 0.02)
{
  return prediction::StaticObstaclePredictor(
    context, 0, Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(),
    obstacles, distance, 0.10, 0.05, 0.0);
}

ode::LeafParams leaf_params()
{
  ode::LeafParams params;
  params.enabled = true;
  params.weight_goal = 0.0;
  params.weight_continuity = 0.0;
  params.weight_risk = 0.0;
  params.weight_axis = 0.0;
  params.weight_axial_block = 0.0;
  params.weight_i1 = 0.0;
  params.weight_stall = 0.0;
  params.e1_bias = 1.0;
  return params;
}

TEST(TangentEscapePrediction, PenalizesOtherSensorApproachWithoutRewardingRetreat)
{
  const auto context = scene();
  const auto model = predictor(context, {{{0.10, 0.0, 0.0}, 0.01, 0}});
  const auto toward = model.evaluate(0.0);
  const auto away = model.evaluate(kPi);
  EXPECT_GT(toward.value, 0.1);
  EXPECT_DOUBLE_EQ(away.value, 0.0);
  EXPECT_LT(toward.min_clearance, away.min_clearance);
  EXPECT_EQ(toward.critical_sensor_index, 1);
  EXPECT_EQ(toward.obstacle_sensor_index, 0);
  EXPECT_EQ(toward.pair_count, RB10Model::sensor_control_points.size() - 1);
}

TEST(TangentEscapePrediction, RetreatAtAnotherSensorCannotCancelApproach)
{
  auto context = scene();
  const std::vector<prediction::StaticObstacle> obstacles{{{0.10, 0.0, 0.0}, 0.01, 0}};
  const double before = predictor(context, obstacles).evaluate(0.0).value;
  context.control_points[2].position.setZero();
  context.control_point_jacobians[2] = -10.0 * context.control_point_jacobians[1];
  const double after = predictor(context, obstacles).evaluate(0.0).value;
  EXPECT_GT(before, 0.0);
  EXPECT_DOUBLE_EQ(after, before);
}

TEST(TangentEscapePrediction, ChecksWholeSweptSegmentWhenEndpointIsClear)
{
  auto context = scene();
  context.control_points[1].radius = 0.002;
  const auto model = predictor(context, {{{0.01, 0.0, 0.0}, 0.002, 0}}, 0.04);
  const auto risk = model.evaluate(0.0);
  EXPECT_GT(risk.value, 0.0);
  EXPECT_NEAR(risk.min_clearance, -0.004, 1e-12);
  EXPECT_DOUBLE_EQ(model.evaluate(kPi).value, 0.0);
}

TEST(TangentEscapePrediction, AnalyticGradientMatchesFiniteDifferences)
{
  auto context = scene();
  context.control_points[1].radius = 0.001;
  // The first obstacle exercises endpoint minima; the second exercises
  // interior minima of the swept path as theta varies.
  for (const Eigen::Vector3d & center : {
      Eigen::Vector3d(0.10, 0.025, 0.01), Eigen::Vector3d(0.01, 0.012, 0.0)})
  {
    const auto model = predictor(context, {{center, 0.001, 0}}, 0.04);
    for (int k = -30; k <= 30; ++k) {
      const double theta = 0.1 * k + 0.023;
      constexpr double h = 1e-6;
      const double numerical =
        (model.evaluate(theta + h).value - model.evaluate(theta - h).value) / (2.0 * h);
      EXPECT_NEAR(model.evaluate(theta).gradient, numerical, 2e-6)
        << "theta=" << theta << " obstacle=" << center.transpose();
    }
  }
}

TEST(TangentEscapePrediction, SingularKinematicsAndInvalidDetectionsStayFinite)
{
  auto context = scene();
  context.control_point_jacobians[0].setZero();
  const auto stationary = predictor(context, {{{0.10, 0.0, 0.0}, 0.01, 0}}).evaluate(0.4);
  EXPECT_DOUBLE_EQ(stationary.value, 0.0);
  EXPECT_DOUBLE_EQ(stationary.gradient, 0.0);
  EXPECT_TRUE(std::isfinite(stationary.min_clearance));
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto invalid = predictor(context, {
    {{nan, 0.0, 0.0}, 0.01, 0}, {{0.1, 0.0, 0.0}, nan, 0},
    {{0.1, 0.0, 0.0}, -1.0, 0}}).evaluate(0.0);
  EXPECT_EQ(invalid.pair_count, 0U);
  EXPECT_DOUBLE_EQ(invalid.value, 0.0);
  EXPECT_EQ(predictor(context, {}).evaluate(0.0).pair_count, 0U);
}

TEST(TangentEscapePrediction, RiskSelectsSaferInitialDirectionAndChangesContinuousUpdate)
{
  auto context = scene();
  const auto & sensor = RB10Model::sensor_control_points[0];
  const Eigen::Vector3d e1 = sensor.local_tangent_bias.normalized();
  const std::vector<prediction::StaticObstacle> obstacles{{0.03 * e1, 0.005, 0}};
  std::vector<double> clearances(
    RB10Model::sensor_control_points.size(), std::numeric_limits<double>::infinity());
  clearances[0] = 0.01;
  const RB10Model::JointVector zero = RB10Model::JointVector::Zero();
  const Eigen::Vector3d goal(0.4, 0.3, 0.5);
  auto params = leaf_params();
  ode::LeafState old_state;
  const auto old = ode::evaluate(old_state, params, context, zero, zero, goal, clearances, {});
  params.weight_predicted_risk = 10.0;
  ode::LeafState new_state;
  const auto added = ode::evaluate(
    new_state, params, context, zero, zero, goal, clearances, {}, obstacles);
  ASSERT_TRUE(old.active && added.active);
  EXPECT_NEAR(old.tangent.dot(e1), 1.0, 1e-12);
  EXPECT_LT(added.tangent.dot(e1), 0.8);
  EXPECT_EQ(added.predicted_risk.pair_count, RB10Model::sensor_control_points.size() - 1);
  // The pre-existing scalar policy and activation values do not change.
  EXPECT_DOUBLE_EQ(old.alpha_total, added.alpha_total);
  EXPECT_DOUBLE_EQ(old.push_acceleration, added.push_acceleration);
  EXPECT_DOUBLE_EQ(old.metric, added.metric);

  old_state.theta = 0.5;
  new_state = old_state;
  params.weight_predicted_risk = 0.0;
  const auto old_update = ode::evaluate(
    old_state, params, context, zero, zero, goal, clearances, {}, obstacles);
  params.weight_predicted_risk = 1.0;
  const auto new_update = ode::evaluate(
    new_state, params, context, zero, zero, goal, clearances, {}, obstacles);
  EXPECT_GT(new_update.theta_rate, old_update.theta_rate);
}

TEST(TangentEscapePrediction, ZeroWeightAndEmptyDetectionsPreserveOriginalLeaf)
{
  const auto context = scene();
  const auto params = leaf_params();
  std::vector<double> clearances(
    RB10Model::sensor_control_points.size(), std::numeric_limits<double>::infinity());
  clearances[0] = 0.01;
  const RB10Model::JointVector zero = RB10Model::JointVector::Zero();
  const Eigen::Vector3d goal(0.4, 0.3, 0.5);
  ode::LeafState reference_state;
  ode::LeafState disabled_state;
  ode::LeafState empty_state;
  auto enabled = params;
  enabled.weight_predicted_risk = 1.0;
  for (int cycle = 0; cycle < 20; ++cycle) {
    const auto reference = ode::evaluate(
      reference_state, params, context, zero, zero, goal, clearances, {});
    const auto disabled = ode::evaluate(
      disabled_state, params, context, zero, zero, goal, clearances, {},
      {{{0.03, 0.0, 0.0}, 0.005, 0}});
    const auto empty = ode::evaluate(
      empty_state, enabled, context, zero, zero, goal, clearances, {});
    EXPECT_DOUBLE_EQ(reference.theta, disabled.theta);
    EXPECT_DOUBLE_EQ(reference.theta, empty.theta);
    EXPECT_DOUBLE_EQ((reference.metric_contribution - disabled.metric_contribution).norm(), 0.0);
    EXPECT_DOUBLE_EQ((reference.force_contribution - disabled.force_contribution).norm(), 0.0);
    EXPECT_DOUBLE_EQ((reference.force_contribution - empty.force_contribution).norm(), 0.0);
  }
}

TEST(TangentEscapePrediction, SupervisorSharesDetectionOnlyForNewRiskAndAppendsDiagnostics)
{
  auto context = scene();
  const auto & sensor = RB10Model::sensor_control_points[0];
  const Eigen::Vector3d normal = sensor.local_normal.normalized();
  const Eigen::Vector3d e1 = sensor.local_tangent_bias.normalized();
  const ObstacleSphere obstacle{
    context.control_points[0].position + 0.03 * normal, 0.01, 0, 7};
  context.control_points[1].position = obstacle.center - 0.03 * e1;
  EigenRmpConfig config;
  config.tangent_escape.enabled = true;
  config.tangent_escape_ode.weight_predicted_risk = 1.0;
  config.enable_tangent_escape_diagnostics = true;
  const TangentEscapeSupervisor supervisor(config);
  const RB10Model::JointVector zero = RB10Model::JointVector::Zero();
  const auto output = supervisor.evaluate(
    zero, context, zero, {0.4, 0.3, 0.5}, {obstacle}, zero,
    Eigen::Matrix<double, 6, 6>::Zero(), zero);
  ASSERT_TRUE(output.active);
  ASSERT_EQ(output.debug_data.size(), tangent_escape_debug::kSize);
  EXPECT_EQ(output.debug_data[26], 9.0);
  EXPECT_EQ(output.debug_data[tangent_escape_debug::kPrediction + 6],
    RB10Model::sensor_control_points.size() - 1);
  // The detection still belongs only to sensor 0 in the original clearance
  // input. Sensor 1 is considered by the new geometry prediction anyway.
  EXPECT_DOUBLE_EQ(output.debug_data[49], 1.0);  // one original sensor observation
  EXPECT_TRUE(std::isfinite(output.debug_data[tangent_escape_debug::kPrediction + 3]));
}

}  // namespace
}  // namespace rb10_rmpflow_rviz

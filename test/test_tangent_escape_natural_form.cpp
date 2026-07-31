#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "rb10_rmpflow_rviz/tangent_escape_natural_form.hpp"

namespace rb10_rmpflow_rviz
{
namespace
{

TangentEscapeJointVector regularized_resolve(
  const TangentEscapeRootMatrix & metric,
  const TangentEscapeJointVector & force,
  double offset)
{
  const double scale = std::max(metric.cwiseAbs().maxCoeff() * 0.01, 1.0);
  TangentEscapeRootMatrix regularized = metric / scale;
  regularized += offset * TangentEscapeRootMatrix::Identity();
  return regularized.ldlt().solve(force / scale);
}

TEST(TangentEscapeNaturalForm, AcceptsRankOneScalarTaskPullback)
{
  const TangentEscapeRootMatrix base_metric =
    20.0 * TangentEscapeRootMatrix::Identity();
  const TangentEscapeJointVector base_force =
    (TangentEscapeJointVector() << 1.0, -2.0, 0.5, 1.5, -0.7, 0.2).finished();
  const Eigen::Matrix<double, 1, 6> scalar_jacobian =
    (Eigen::Matrix<double, 1, 6>() << 0.4, -0.2, 0.7, 0.0, 0.3, -0.1).finished();
  constexpr double metric_scalar = 100000.0;
  constexpr double scalar_natural_acceleration = 1.7;
  const TangentEscapeRootMatrix escape_metric =
    scalar_jacobian.transpose() * metric_scalar * scalar_jacobian;
  const TangentEscapeJointVector escape_force =
    scalar_jacobian.transpose() *
    (metric_scalar * scalar_natural_acceleration);
  const TangentEscapeRootMatrix full_metric = base_metric + escape_metric;
  const TangentEscapeJointVector full_force = base_force + escape_force;
  constexpr double solve_offset = 0.001;
  const TangentEscapeJointVector qdd =
    regularized_resolve(full_metric, full_force, solve_offset);

  const auto audit = audit_tangent_escape_natural_form(
    base_metric,
    base_force,
    full_metric,
    full_force,
    qdd,
    solve_offset,
    Eigen::Vector3d::UnitY(),
    Eigen::Vector3d::UnitX());

  EXPECT_TRUE(audit.contribution_active);
  EXPECT_TRUE(audit.finite);
  EXPECT_TRUE(audit.contribution_symmetric);
  EXPECT_TRUE(audit.contribution_psd);
  EXPECT_TRUE(audit.contribution_rank_one_or_zero);
  EXPECT_EQ(audit.contribution_numerical_rank, 1);
  EXPECT_TRUE(audit.root_metric_spd);
  EXPECT_TRUE(audit.regularized_root_resolve_consistent);
  EXPECT_TRUE(audit.tangent_normal_orthogonal);
  EXPECT_TRUE(audit.natural_form_compatible);
}

TEST(TangentEscapeNaturalForm, RejectsIndefiniteContribution)
{
  const TangentEscapeRootMatrix base_metric =
    20.0 * TangentEscapeRootMatrix::Identity();
  TangentEscapeRootMatrix full_metric = base_metric;
  full_metric(0, 0) -= 30.0;
  const TangentEscapeJointVector force = TangentEscapeJointVector::Zero();

  const auto audit = audit_tangent_escape_natural_form(
    base_metric,
    force,
    full_metric,
    force,
    TangentEscapeJointVector::Zero(),
    0.001,
    Eigen::Vector3d::UnitY(),
    Eigen::Vector3d::UnitX());

  EXPECT_FALSE(audit.contribution_psd);
  EXPECT_FALSE(audit.root_metric_spd);
  EXPECT_FALSE(audit.natural_form_compatible);
}

}  // namespace
}  // namespace rb10_rmpflow_rviz

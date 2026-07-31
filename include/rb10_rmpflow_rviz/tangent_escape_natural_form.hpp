#pragma once

#include <limits>

#include <Eigen/Dense>

namespace rb10_rmpflow_rviz
{

using TangentEscapeJointVector = Eigen::Matrix<double, 6, 1>;
using TangentEscapeRootMatrix = Eigen::Matrix<double, 6, 6>;

struct TangentEscapeNaturalFormAudit
{
  bool contribution_active{false};
  bool finite{false};
  bool contribution_symmetric{false};
  bool contribution_psd{false};
  bool contribution_rank_one_or_zero{false};
  bool root_metric_spd{false};
  bool regularized_root_resolve_consistent{false};
  bool tangent_normal_orthogonal{false};
  bool natural_form_compatible{false};
  int contribution_numerical_rank{0};
  double contribution_metric_norm{0.0};
  double contribution_force_norm{0.0};
  double contribution_symmetry_residual{
    std::numeric_limits<double>::infinity()};
  double contribution_min_eigenvalue{
    -std::numeric_limits<double>::infinity()};
  double contribution_max_eigenvalue{
    std::numeric_limits<double>::infinity()};
  double root_metric_min_eigenvalue{
    -std::numeric_limits<double>::infinity()};
  double root_metric_symmetry_residual{
    std::numeric_limits<double>::infinity()};
  double regularized_root_resolve_residual{
    std::numeric_limits<double>::infinity()};
  double tangent_normal_dot_abs{
    std::numeric_limits<double>::infinity()};
};

TangentEscapeNaturalFormAudit audit_tangent_escape_natural_form(
  const TangentEscapeRootMatrix & base_metric,
  const TangentEscapeJointVector & base_force,
  const TangentEscapeRootMatrix & full_metric,
  const TangentEscapeJointVector & full_force,
  const TangentEscapeJointVector & resolved_acceleration,
  double root_solve_offset,
  const Eigen::Vector3d & tangent,
  const Eigen::Vector3d & obstacle_normal,
  double active_tolerance = 1e-10,
  double symmetry_tolerance = 1e-9,
  double eigenvalue_tolerance = 1e-8,
  double orthogonality_tolerance = 1e-5);

}  // namespace rb10_rmpflow_rviz

#include "rb10_rmpflow_rviz/tangent_escape_natural_form.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Eigenvalues>

namespace rb10_rmpflow_rviz
{

TangentEscapeNaturalFormAudit audit_tangent_escape_natural_form(
  const TangentEscapeRootMatrix & base_metric,
  const TangentEscapeJointVector & base_force,
  const TangentEscapeRootMatrix & full_metric,
  const TangentEscapeJointVector & full_force,
  const TangentEscapeJointVector & resolved_acceleration,
  double root_solve_offset,
  const Eigen::Vector3d & tangent,
  const Eigen::Vector3d & obstacle_normal,
  double active_tolerance,
  double symmetry_tolerance,
  double eigenvalue_tolerance,
  double orthogonality_tolerance)
{
  TangentEscapeNaturalFormAudit result;
  const TangentEscapeRootMatrix contribution_metric =
    full_metric - base_metric;
  const TangentEscapeJointVector contribution_force =
    full_force - base_force;
  result.contribution_metric_norm = contribution_metric.norm();
  result.contribution_force_norm = contribution_force.norm();
  result.contribution_active =
    result.contribution_metric_norm > std::max(active_tolerance, 0.0) ||
    result.contribution_force_norm > std::max(active_tolerance, 0.0);
  result.finite =
    base_metric.allFinite() &&
    base_force.allFinite() &&
    full_metric.allFinite() &&
    full_force.allFinite() &&
    resolved_acceleration.allFinite() &&
    tangent.allFinite() &&
    obstacle_normal.allFinite();
  if (!result.finite) {
    return result;
  }

  const TangentEscapeRootMatrix symmetric_contribution =
    0.5 * (contribution_metric + contribution_metric.transpose());
  result.contribution_symmetry_residual =
    (contribution_metric - contribution_metric.transpose()).norm();
  result.contribution_symmetric =
    result.contribution_symmetry_residual <= std::max(symmetry_tolerance, 0.0);

  Eigen::SelfAdjointEigenSolver<TangentEscapeRootMatrix> contribution_solver(
    symmetric_contribution);
  if (contribution_solver.info() == Eigen::Success) {
    result.contribution_min_eigenvalue =
      contribution_solver.eigenvalues().minCoeff();
    result.contribution_max_eigenvalue =
      contribution_solver.eigenvalues().maxCoeff();
    const double rank_threshold = std::max(
      std::max(eigenvalue_tolerance, 0.0),
      std::abs(result.contribution_max_eigenvalue) * 1e-8);
    result.contribution_numerical_rank =
      (contribution_solver.eigenvalues().array() > rank_threshold).count();
    result.contribution_psd =
      result.contribution_min_eigenvalue >= -std::max(eigenvalue_tolerance, 0.0);
    result.contribution_rank_one_or_zero =
      result.contribution_numerical_rank <= 1;
  }

  const TangentEscapeRootMatrix symmetric_root =
    0.5 * (full_metric + full_metric.transpose());
  result.root_metric_symmetry_residual =
    (full_metric - full_metric.transpose()).norm();
  Eigen::SelfAdjointEigenSolver<TangentEscapeRootMatrix> root_solver(
    symmetric_root);
  if (root_solver.info() == Eigen::Success) {
    result.root_metric_min_eigenvalue =
      root_solver.eigenvalues().minCoeff();
    result.root_metric_spd =
      result.root_metric_min_eigenvalue > std::max(eigenvalue_tolerance, 0.0) &&
      result.root_metric_symmetry_residual <= std::max(symmetry_tolerance, 0.0);
  }

  const double root_scale =
    std::max(full_metric.cwiseAbs().maxCoeff() * 0.01, 1.0);
  TangentEscapeRootMatrix regularized_metric = full_metric / root_scale;
  regularized_metric +=
    std::max(root_solve_offset, 0.0) * TangentEscapeRootMatrix::Identity();
  const TangentEscapeJointVector regularized_force = full_force / root_scale;
  result.regularized_root_resolve_residual =
    (regularized_metric * resolved_acceleration - regularized_force).norm();
  result.regularized_root_resolve_consistent =
    result.regularized_root_resolve_residual <=
    1e-8 * (1.0 + regularized_force.norm());

  const double tangent_norm = tangent.norm();
  const double normal_norm = obstacle_normal.norm();
  if (tangent_norm > 1e-12 && normal_norm > 1e-12) {
    result.tangent_normal_dot_abs = std::abs(
      tangent.dot(obstacle_normal) / (tangent_norm * normal_norm));
    result.tangent_normal_orthogonal =
      result.tangent_normal_dot_abs <= std::max(orthogonality_tolerance, 0.0);
  }

  result.natural_form_compatible =
    result.contribution_active &&
    result.contribution_symmetric &&
    result.contribution_psd &&
    result.contribution_rank_one_or_zero &&
    result.root_metric_spd &&
    result.regularized_root_resolve_consistent &&
    result.tangent_normal_orthogonal;
  return result;
}

}  // namespace rb10_rmpflow_rviz

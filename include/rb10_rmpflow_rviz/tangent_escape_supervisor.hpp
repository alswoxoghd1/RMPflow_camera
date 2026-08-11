#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "rb10_rmpflow_rviz/rmp_eigen_solver.hpp"
#include "rb10_rmpflow_rviz/tangent_escape_ode.hpp"

namespace rb10_rmpflow_rviz
{

struct TangentEscapeCommand
{
  using JointVector = RB10Model::JointVector;
  using Matrix6 = Eigen::Matrix<double, 6, 6>;

  bool active{false};
  std::size_t control_point_index{0};
  Eigen::Vector3d tangent{Eigen::Vector3d::UnitX()};
  double metric_scalar{0.0};
  double scalar_acceleration{0.0};
  Matrix6 metric_contribution{Matrix6::Zero()};
  JointVector force_contribution{JointVector::Zero()};
  std::vector<double> debug_data;
};

// Thin adapter between the solver and the Escape leaf.  The direction state
// lives in tangent_escape_ode::LeafState, outside the RMP tree; this class only
// turns the solver's obstacle list into per-sensor clearances and forwards the
// leaf's pullback.
class TangentEscapeSupervisor
{
public:
  using JointVector = RB10Model::JointVector;
  using Matrix6 = Eigen::Matrix<double, 6, 6>;
  using RowVector6 = Eigen::Matrix<double, 1, 6>;

  explicit TangentEscapeSupervisor(EigenRmpConfig config);

  TangentEscapeCommand evaluate(
    const JointVector & q,
    const KinematicsContext & context,
    const JointVector & qd,
    const Eigen::Vector3d & goal,
    const std::vector<ObstacleSphere> & obstacles,
    const JointVector & nominal_qdd,
    const Matrix6 & base_metric,
    const JointVector & base_force) const;

  void set_previous_qdd(const JointVector & qdd) const;

private:
  void evaluate_ode(
    const KinematicsContext & context,
    const JointVector & qd,
    const Eigen::Vector3d & goal,
    const std::vector<ObstacleSphere> & obstacles,
    Matrix6 & metric,
    JointVector & force,
    TangentEscapeCommand & command,
    std::vector<double> * debug_data) const;

  EigenRmpConfig config_;
  mutable tangent_escape_ode::LeafState ode_state_{};
};

}  // namespace rb10_rmpflow_rviz

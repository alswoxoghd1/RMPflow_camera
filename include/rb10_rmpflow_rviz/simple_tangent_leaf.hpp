#pragma once
#include "rb10_rmpflow_rviz/tangent_escape_ode.hpp"

namespace rb10_rmpflow_rviz::simple_tangent_leaf
{
// Score-free velocity-normal steering. The mounting normal is a directional
// proxy, not a measured obstacle bearing. Uses a Cartesian point task map;
// its pullback must not differentiate the velocity-dependent direction as
// though it were the original one-dimensional progress coordinate.
tangent_escape_ode::LeafOutput evaluate(
  tangent_escape_ode::LeafState & state,
  tangent_escape_ode::LeafParams params,
  const KinematicsContext & context,
  const RB10Model::JointVector & qd,
  const RB10Model::JointVector & nominal_qdd,
  const Eigen::Vector3d & goal,
  const std::vector<double> & ranges,
  const std::vector<double> & rates,
  const std::vector<Eigen::Vector3d> & measured_velocities = {});

// Fixed sensor-plane leaf retained for history-mode counterfactuals and the
// explicit fixed comparison mode. No velocity-steering state is advanced.
tangent_escape_ode::LeafOutput evaluate_fixed(
  tangent_escape_ode::LeafState & state,
  tangent_escape_ode::LeafParams params,
  const KinematicsContext & context,
  const RB10Model::JointVector & qd,
  const RB10Model::JointVector & nominal_qdd,
  const Eigen::Vector3d & goal,
  const std::vector<double> & ranges,
  const std::vector<double> & rates);
}

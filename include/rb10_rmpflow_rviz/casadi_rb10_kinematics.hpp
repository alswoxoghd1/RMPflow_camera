#pragma once

#include <memory>
#include <string>

#include <casadi/casadi.hpp>

#include "rb10_rmpflow_rviz/rb10_model.hpp"

namespace rb10_rmpflow_rviz
{

// RB10 URDF kinematics. The URDF tree is translated once
// into a CasADi DAG and all runtime derivatives come from that graph.
class CasadiRb10Kinematics
{
public:
  using JointVector = RB10Model::JointVector;

  explicit CasadiRb10Kinematics(const std::string & urdf_path);

  KinematicsContext forward_context(
    const JointVector & q,
    const JointVector & qd) const;

  // Inputs: q[6]. Outputs: stacked link positions [36], stacked link
  // rotations [108, column-major per frame], and sensor positions [60].
  const casadi::Function & task_map_function() const;

private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

}  // namespace rb10_rmpflow_rviz

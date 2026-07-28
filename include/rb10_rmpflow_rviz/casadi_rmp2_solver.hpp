#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rb10_rmpflow_rviz/casadi_rb10_kinematics.hpp"
#include "rb10_rmpflow_rviz/rmp_solver_interface.hpp"

namespace rb10_rmpflow_rviz
{

// Algorithm-3 backend: task coordinates, JVP curvature, and mirrored
// reverse-accumulation are all represented in one CasADi DAG.
class CasadiRmp2Solver : public RmpSolverInterface
{
public:
  using JointVector = RB10Model::JointVector;
  using Matrix6 = Eigen::Matrix<double, 6, 6>;

  CasadiRmp2Solver(EigenRmpConfig config, std::string urdf_path);

  RmpSolveResult solve(
    const JointVector & q,
    const JointVector & qd,
    const std::unordered_map<std::string, Eigen::Vector3d> & vector_targets,
    const std::vector<ObstacleSphere> & obstacles,
    const std::unordered_map<std::string, ExternalRmpFeature> & external_rmps = {}) const override;

  std::vector<RmpSolveResult> solve_batch(
    const std::vector<RmpBatchInput> & batch_inputs) const override;

  KinematicsContext forward_context(
    const JointVector & q,
    const JointVector & qd) const;

  const std::vector<std::string> & target_keys() const;
  int max_obstacles() const;
  bool jit_enabled() const;

  // Writes a self-contained CasADi-generated C implementation of the root
  // policy. This is an offline artifact; the optional JIT path executes the
  // same generated representation in-process.
  std::string generate_c_code(const std::string & output_base_name) const;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace rb10_rmpflow_rviz

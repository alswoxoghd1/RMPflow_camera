#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "rb10_rmpflow_rviz/rmp_eigen_solver.hpp"

namespace rb10_rmpflow_rviz
{

bool base_graph_has_structured_gds_form(const EigenRmpConfig & config);

bool same_rmp_environment(
  const std::unordered_map<std::string, Eigen::Vector3d> & previous_targets,
  const std::unordered_map<std::string, Eigen::Vector3d> & current_targets,
  const std::vector<ObstacleSphere> & previous_obstacles,
  const std::vector<ObstacleSphere> & current_obstacles);

}  // namespace rb10_rmpflow_rviz

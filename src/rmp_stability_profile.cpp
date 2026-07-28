#include "rb10_rmpflow_rviz/rmp_stability_profile.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>

#include "rb10_rmpflow_rviz/paper_gds_collision.hpp"
#include "rb10_rmpflow_rviz/paper_gds_joint_limit.hpp"

namespace rb10_rmpflow_rviz
{
namespace
{

bool all_finite(std::initializer_list<double> values)
{
  return std::all_of(
    values.begin(), values.end(),
    [](double value) {return std::isfinite(value);});
}

bool target_leaf_has_constant_gds_metric(const TargetRmpParams & params)
{
  return
    all_finite({
      params.accel_p_gain,
      params.accel_d_gain,
      params.accel_norm_eps,
      params.metric_alpha_length_scale,
      params.min_metric_alpha,
      params.max_metric_scalar,
      params.min_metric_scalar,
      params.proximity_metric_boost_scalar,
      params.proximity_metric_boost_length_scale}) &&
    params.min_metric_alpha == 1.0 &&
    params.proximity_metric_boost_scalar == 1.0 &&
    params.accel_norm_eps > 0.0 &&
    params.metric_alpha_length_scale > 0.0 &&
    params.proximity_metric_boost_length_scale > 0.0 &&
    params.max_metric_scalar > 0.0 &&
    params.min_metric_scalar >= 0.0 &&
    params.accel_p_gain >= 0.0 &&
    params.accel_d_gain >= 0.0;
}

bool axis_leaf_has_constant_gds_metric(const AxisTargetParams & params)
{
  return
    all_finite({
      params.accel_p_gain,
      params.accel_d_gain,
      params.metric_scalar,
      params.proximity_metric_boost_scalar,
      params.proximity_metric_boost_length_scale}) &&
    params.proximity_metric_boost_scalar == 1.0 &&
    params.proximity_metric_boost_length_scale > 0.0 &&
    params.metric_scalar >= 0.0 &&
    params.accel_p_gain >= 0.0 &&
    params.accel_d_gain >= 0.0;
}

bool supported_smooth_task_map(const std::string & task_map_type)
{
  return
    task_map_type == "cspace_target" ||
    task_map_type == "identity" ||
    task_map_type == "joint_limit" ||
    task_map_type == "tcp_position" ||
    task_map_type == "link_position" ||
    task_map_type == "link_orientation_axis" ||
    task_map_type == "control_points" ||
    task_map_type == "collision_distance" ||
    task_map_type == "affine" ||
    task_map_type == "elem_multiply" ||
    task_map_type == "sin" ||
    task_map_type == "cos" ||
    task_map_type == "tanh" ||
    task_map_type == "square" ||
    task_map_type == "sum" ||
    task_map_type == "weighted_sum" ||
    task_map_type == "difference" ||
    task_map_type == "concat" ||
    task_map_type == "slice";
}

bool leaf_attachment_supported(
  const RmpNodeConfig & node,
  const std::string & leaf_type)
{
  if (leaf_type == "cspace_target") {
    return node.task_map_type == "cspace_target" || node.task_map_type == "identity";
  }
  if (leaf_type == "joint_limit") {
    return node.task_map_type == "joint_limit";
  }
  if (leaf_type == "collision") {
    return node.task_map_type == "collision_distance";
  }
  if (leaf_type == "axis_target") {
    return node.task_map_type == "link_orientation_axis";
  }
  return true;
}

bool leaf_has_structured_gds_form(
  const EigenRmpConfig & config,
  const std::string & leaf_type)
{
  if (leaf_type.empty() || leaf_type == "none" || leaf_type == "tangent_escape") {
    return true;
  }
  if (leaf_type == "cspace_target") {
    return
      all_finite({
        config.cspace_target.metric_scalar,
        config.cspace_target.position_gain,
        config.cspace_target.damping_gain,
        config.cspace_target.robust_position_term_thresh,
        config.cspace_target.inertia}) &&
      config.cspace_target.metric_scalar + config.cspace_target.inertia > 0.0 &&
      config.cspace_target.position_gain >= 0.0 &&
      config.cspace_target.damping_gain >= 0.0 &&
      config.cspace_target.robust_position_term_thresh >= 0.0;
  }
  if (leaf_type == "joint_limit") {
    if (config.joint_limit.policy != "paper_gds") {
      return false;
    }
    for (std::size_t joint = 0; joint < 6; ++joint) {
      PaperGdsJointLimitConfig params;
      params.lower = config.joint_lower_limits[joint] + config.joint_limit_buffers[joint];
      params.upper = config.joint_upper_limits[joint] - config.joint_limit_buffers[joint];
      params.center_fraction = config.joint_limit.gds_center_fraction;
      params.task_metric = config.joint_limit.metric_scalar;
      params.potential_gain =
        config.joint_limit.metric_scalar * config.joint_limit.accel_potential_gain;
      params.damping =
        config.joint_limit.metric_scalar * config.joint_limit.accel_damper_gain;
      params.boundary_epsilon = config.joint_limit.gds_domain_epsilon;
      if (!valid_paper_gds_joint_limit_config(params)) {
        return false;
      }
    }
    return true;
  }
  if (leaf_type == "target") {
    return target_leaf_has_constant_gds_metric(config.target);
  }
  if (leaf_type == "axis_target") {
    return axis_leaf_has_constant_gds_metric(config.axis_target);
  }
  if (leaf_type == "wrist_axis_target") {
    return false;
  }
  if (leaf_type == "collision") {
    if (config.collision.policy != "paper_gds") {
      return false;
    }
    paper_gds_collision::Params params;
    params.metric_scalar = config.collision.metric_scalar;
    params.metric_modulation_radius = config.collision.metric_modulation_radius;
    params.metric_exploder_std_dev = config.collision.metric_exploder_std_dev;
    params.metric_exploder_eps = config.collision.metric_exploder_eps;
    params.clearance_smoothing = config.collision.gds_clearance_smoothing;
    params.metric_velocity_floor = config.collision.gds_metric_velocity_floor;
    params.metric_velocity_scale = config.collision.gds_metric_velocity_scale;
    params.repulsion_gain = config.collision.repulsion_gain;
    params.repulsion_std_dev = config.collision.repulsion_std_dev;
    params.damping_gain = config.collision.damping_gain;
    params.damping_std_dev = config.collision.damping_std_dev;
    params.damping_robustness_eps = config.collision.damping_robustness_eps;
    params.damping_velocity_scale = config.collision.gds_damping_velocity_scale;
    return paper_gds_collision::parameters_are_valid(params);
  }
  if (leaf_type == "damping") {
    return
      all_finite({
        config.damping.metric_scalar,
        config.damping.inertia,
        config.damping.accel_d_gain}) &&
      config.damping.metric_scalar == 0.0 &&
      config.damping.inertia >= 0.0 &&
      config.damping.accel_d_gain >= 0.0;
  }
  return false;
}

}  // namespace

bool base_graph_has_structured_gds_form(const EigenRmpConfig & config)
{
  if (config.solve_offset != 0.0) {
    return false;
  }
  for (const auto & node : config.graph_nodes) {
    if (!node.enabled) {
      continue;
    }
    if (!supported_smooth_task_map(node.task_map_type)) {
      return false;
    }
    if (
      !leaf_attachment_supported(node, node.leaf_rmp_type) ||
      !leaf_attachment_supported(node, node.handcrafted_leaf_rmp_type) ||
      !leaf_has_structured_gds_form(config, node.leaf_rmp_type) ||
      !leaf_has_structured_gds_form(config, node.handcrafted_leaf_rmp_type))
    {
      return false;
    }
  }
  return true;
}

bool same_rmp_environment(
  const std::unordered_map<std::string, Eigen::Vector3d> & previous_targets,
  const std::unordered_map<std::string, Eigen::Vector3d> & current_targets,
  const std::vector<ObstacleSphere> & previous_obstacles,
  const std::vector<ObstacleSphere> & current_obstacles)
{
  if (
    previous_targets.size() != current_targets.size() ||
    previous_obstacles.size() != current_obstacles.size())
  {
    return false;
  }
  for (const auto & target : current_targets) {
    const auto previous = previous_targets.find(target.first);
    if (
      previous == previous_targets.end() ||
      (previous->second - target.second).norm() > 1e-12)
    {
      return false;
    }
  }
  for (std::size_t index = 0; index < current_obstacles.size(); ++index) {
    const auto & previous = previous_obstacles[index];
    const auto & current = current_obstacles[index];
    if (
      previous.source_id != current.source_id ||
      previous.proximity_control_point_index != current.proximity_control_point_index ||
      previous.radius != current.radius ||
      (previous.center - current.center).norm() > 1e-12)
    {
      return false;
    }
  }
  return true;
}

}  // namespace rb10_rmpflow_rviz

#include "rb10_rmpflow_rviz/casadi_rmp2_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <casadi/casadi.hpp>
#include <Eigen/Eigenvalues>

#include "rb10_rmpflow_rviz/casadi_task_graph.hpp"
#include "rb10_rmpflow_rviz/joint_acceleration_limiter.hpp"
#include "rb10_rmpflow_rviz/rmp_stability_profile.hpp"
#include "rb10_rmpflow_rviz/tangent_escape_supervisor.hpp"

namespace rb10_rmpflow_rviz
{

namespace
{

using JointVector = RB10Model::JointVector;
using Matrix6 = Eigen::Matrix<double, 6, 6>;

struct SymbolicLeaf
{
  casadi::MX x;
  casadi::MX x0;
  casadi::MX x00;
  casadi::MX curvature;
  casadi::MX metric;
  casadi::MX natural_force;
  bool nominal{false};
  int ablation_group{-1};
};

enum class LeafAblationGroup : int
{
  CspaceTarget = 1,
  JointLimits = 2,
  JointVelocityCap = 3,
  Target = 4,
  Collision = 5,
  TangentEscape = 6,
  Damping = 7,
  BodyTarget = 8,
  AxisTargetX = 9,
  AxisTargetY = 10,
  AxisTargetZ = 11,
  WristAxisTarget = 12,
  External = 13,
  Other = 99
};

int leaf_ablation_group_id(
  const RmpNodeConfig & node,
  const std::string & leaf_type)
{
  if (leaf_type == "cspace_target") {
    return static_cast<int>(LeafAblationGroup::CspaceTarget);
  }
  if (leaf_type == "joint_limit") {
    return static_cast<int>(LeafAblationGroup::JointLimits);
  }
  if (leaf_type == "joint_velocity_cap") {
    return static_cast<int>(LeafAblationGroup::JointVelocityCap);
  }
  if (leaf_type == "target") {
    return static_cast<int>(
      node.name == "body_link4" ? LeafAblationGroup::BodyTarget : LeafAblationGroup::Target);
  }
  if (leaf_type == "collision") {
    return static_cast<int>(LeafAblationGroup::Collision);
  }
  if (leaf_type == "tangent_escape") {
    return static_cast<int>(LeafAblationGroup::TangentEscape);
  }
  if (leaf_type == "damping") {
    return static_cast<int>(LeafAblationGroup::Damping);
  }
  if (leaf_type == "axis_target") {
    if (node.axis == "x") {
      return static_cast<int>(LeafAblationGroup::AxisTargetX);
    }
    if (node.axis == "y") {
      return static_cast<int>(LeafAblationGroup::AxisTargetY);
    }
    return static_cast<int>(LeafAblationGroup::AxisTargetZ);
  }
  if (leaf_type == "wrist_axis_target") {
    return static_cast<int>(LeafAblationGroup::WristAxisTarget);
  }
  if (leaf_type == "external") {
    return static_cast<int>(LeafAblationGroup::External);
  }
  return static_cast<int>(LeafAblationGroup::Other);
}

struct EscapeEnergyCertificateState
{
  bool initialized{false};
  double tank_energy{0.0};
  double positive_energy_integral{0.0};
  double negative_energy_integral{0.0};
  double net_energy_integral{0.0};
  std::uint64_t sample_count{0};
  bool previous_environment_valid{false};
  std::unordered_map<std::string, Eigen::Vector3d> previous_vector_targets;
  std::vector<ObstacleSphere> previous_obstacles;
};

casadi::MX scalar(double value)
{
  return casadi::DM(value);
}

casadi::MX squared_norm(const casadi::MX & value)
{
  return dot(value, value);
}

casadi::MX vector_norm(const casadi::MX & value)
{
  return sqrt(squared_norm(value));
}

casadi::MX sigmoid(const casadi::MX & value)
{
  return 1.0 / (1.0 + exp(-value));
}

casadi::MX vector_from_array(const std::array<double, 6> & values)
{
  return casadi::DM(std::vector<double>(values.begin(), values.end()));
}

casadi::DM joint_vector_to_dm(const JointVector & value)
{
  casadi::DM out = casadi::DM::zeros(6, 1);
  for (int index = 0; index < 6; ++index) {
    out(index) = value[index];
  }
  return out;
}

Eigen::VectorXd dm_to_vector(const casadi::DM & value)
{
  const casadi::DM dense = casadi::DM::densify(value);
  const auto elements = dense.get_elements();
  Eigen::VectorXd out(static_cast<Eigen::Index>(elements.size()));
  for (std::size_t index = 0; index < elements.size(); ++index) {
    out[static_cast<Eigen::Index>(index)] = elements[index];
  }
  return out;
}

Eigen::MatrixXd dm_to_matrix(const casadi::DM & value)
{
  const casadi::DM dense = casadi::DM::densify(value);
  const auto elements = dense.get_elements();
  Eigen::MatrixXd out(dense.size1(), dense.size2());
  for (casadi_int col = 0; col < dense.size2(); ++col) {
    for (casadi_int row = 0; row < dense.size1(); ++row) {
      const std::size_t flat = static_cast<std::size_t>(col * dense.size1() + row);
      out(row, col) = elements[flat];
    }
  }
  return out;
}

JointVector resolve_root(const Matrix6 & metric, const JointVector & force, double solve_offset)
{
  const double max_abs = std::max(metric.cwiseAbs().maxCoeff() * 0.01, 1.0);
  Matrix6 scaled_metric = metric / max_abs;
  JointVector scaled_force = force / max_abs;
  scaled_metric += std::max(solve_offset, 0.0) * Matrix6::Identity();

  JointVector qdd = scaled_metric.ldlt().solve(scaled_force).eval();
  if (!qdd.allFinite()) {
    qdd = scaled_metric.completeOrthogonalDecomposition().solve(scaled_force).eval();
  }
  if (!qdd.allFinite()) {
    qdd.setZero();
  }
  return qdd;
}

int link_index_from_name(const std::string & name)
{
  for (std::size_t index = 0; index < RB10Model::link_names.size(); ++index) {
    if (name == RB10Model::link_names[index]) {
      return static_cast<int>(index);
    }
  }
  throw std::runtime_error("Unsupported RB10 link name in CasADi RMP2 graph: " + name);
}

int axis_index(const std::string & name)
{
  if (name == "x") {
    return 0;
  }
  if (name == "y") {
    return 1;
  }
  if (name == "z") {
    return 2;
  }
  throw std::runtime_error("Unsupported orientation axis in CasADi RMP2 graph: " + name);
}

bool is_symbolic_task_map(const std::string & type)
{
  static const std::unordered_set<std::string> types{
    "affine", "elem_multiply", "elem_divide", "sin", "cos", "tanh",
    "square", "abs", "sum", "weighted_sum", "difference", "concat",
    "slice", "norm", "normalize"
  };
  return types.count(type) != 0;
}

std::vector<std::size_t> topological_order(const std::vector<RmpNodeConfig> & nodes)
{
  std::unordered_map<std::string, std::size_t> enabled;
  enabled.emplace("root", nodes.size());
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    if (!nodes[index].enabled) {
      continue;
    }
    if (!enabled.emplace(nodes[index].name, index).second) {
      throw std::runtime_error("Duplicate enabled RMP graph node: " + nodes[index].name);
    }
  }

  std::vector<std::size_t> order;
  std::unordered_set<std::string> resolved{"root"};
  while (order.size() + 1 < enabled.size()) {
    bool made_progress = false;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
      const auto & node = nodes[index];
      if (!node.enabled || resolved.count(node.name) != 0) {
        continue;
      }
      bool ready = true;
      for (const auto & parent : node.parents) {
        if (enabled.count(parent) == 0) {
          throw std::runtime_error(
                  "RMP graph node " + node.name + " has disabled/missing parent " + parent);
        }
        ready = ready && resolved.count(parent) != 0;
      }
      if (ready) {
        order.push_back(index);
        resolved.insert(node.name);
        made_progress = true;
      }
    }
    if (!made_progress) {
      throw std::runtime_error("Cycle detected in CasADi RMP2 graph");
    }
  }
  return order;
}

std::vector<std::string> collect_target_keys(const std::vector<RmpNodeConfig> & nodes)
{
  std::set<std::string> keys;
  for (const auto & node : nodes) {
    if (!node.enabled) {
      continue;
    }
    const auto collect = [&keys, &node](const std::string & leaf) {
        if (
          leaf == "target" || leaf == "axis_target" ||
          leaf == "wrist_axis_target" || leaf == "tangent_escape")
        {
          keys.insert(node.target_key);
        }
      };
    collect(node.leaf_rmp_type);
    collect(node.handcrafted_leaf_rmp_type);
  }
  return {keys.begin(), keys.end()};
}

casadi::MX collision_task_map(
  const casadi::MX & control_points,
  const casadi::MX & obstacle_centers,
  const casadi::MX & obstacle_radii,
  int max_obstacles,
  const CollisionRmpParams & params)
{
  std::vector<casadi::MX> clearances;
  clearances.reserve(RB10Model::sensor_control_points.size() * max_obstacles);
  for (std::size_t point = 0; point < RB10Model::sensor_control_points.size(); ++point) {
    const casadi::MX position = control_points(casadi::Slice(
        static_cast<casadi_int>(3 * point),
        static_cast<casadi_int>(3 * point + 3)));
    for (int obstacle = 0; obstacle < max_obstacles; ++obstacle) {
      const casadi::MX center = obstacle_centers(casadi::Slice(
          3 * obstacle, 3 * obstacle + 3));
      const casadi::MX delta = position - center;
      const casadi::MX center_distance = fmax(vector_norm(delta), 1e-9);
      const casadi::MX signed_clearance =
        center_distance -
        RB10Model::sensor_control_points[point].radius -
        obstacle_radii(obstacle) - params.margin;
      // Keep the task map differentiable and geometrically meaningful through
      // contact.  The legacy explicit backend clamps the coordinate but keeps
      // the signed-distance Jacobian, which cannot be represented by one
      // differentiable task map.  Policy terms that require a nonnegative
      // clearance clamp locally inside the leaf below.
      clearances.push_back(signed_clearance);
    }
  }
  return casadi::MX::vertcat(clearances);
}

casadi::MX orientation_axis_from_stacked_rotations(
  const casadi::MX & rotations,
  const std::string & link_name,
  const std::string & axis)
{
  const int link = link_index_from_name(link_name);
  const casadi_int offset = 9 * link;
  const casadi::MX rotation = casadi::MX::reshape(
    rotations(casadi::Slice(offset, offset + 9)), 3, 3);
  return rotation(casadi::Slice(), axis_index(axis));
}

std::unordered_map<std::string, casadi::MX> build_node_outputs(
  const casadi::MX & q,
  const casadi::MX & obstacle_centers,
  const casadi::MX & obstacle_radii,
  const casadi::Function & robot_task_map,
  const EigenRmpConfig & config,
  const std::vector<std::size_t> & order)
{
  const std::vector<casadi::MX> kinematics_inputs{q};
  const std::vector<casadi::MX> kinematics = robot_task_map(kinematics_inputs);
  const casadi::MX & link_positions = kinematics.at(0);
  const casadi::MX & link_rotations = kinematics.at(1);
  const casadi::MX & control_points = kinematics.at(2);

  std::unordered_map<std::string, casadi::MX> outputs;
  outputs.emplace("root", q);
  for (const std::size_t index : order) {
    const auto & node = config.graph_nodes[index];
    std::vector<casadi::MX> parents;
    parents.reserve(node.parents.size());
    for (const auto & name : node.parents) {
      parents.push_back(outputs.at(name));
    }

    casadi::MX output;
    if (node.task_map_type == "identity" || node.task_map_type == "cspace_target") {
      output = node.scale * parents.front();
    } else if (node.task_map_type == "joint_limit") {
      std::vector<casadi::MX> values;
      values.reserve(12);
      for (int joint = 0; joint < 6; ++joint) {
        const double lower =
          config.joint_lower_limits[static_cast<std::size_t>(joint)] +
          config.joint_limit_buffers[static_cast<std::size_t>(joint)];
        const double upper =
          config.joint_upper_limits[static_cast<std::size_t>(joint)] -
          config.joint_limit_buffers[static_cast<std::size_t>(joint)];
        values.push_back(upper - parents.front()(joint));
        values.push_back(parents.front()(joint) - lower);
      }
      output = casadi::MX::vertcat(values);
    } else if (node.task_map_type == "tcp_position") {
      const casadi_int offset = 3 * static_cast<int>(RB10Model::TCP_RMP);
      output = link_positions(casadi::Slice(offset, offset + 3));
    } else if (node.task_map_type == "link_position") {
      const casadi_int offset = 3 * link_index_from_name(node.link_name);
      output = link_positions(casadi::Slice(offset, offset + 3));
    } else if (node.task_map_type == "link_orientation_axis") {
      output = orientation_axis_from_stacked_rotations(
        link_rotations, node.link_name, node.axis);
    } else if (node.task_map_type == "control_points") {
      output = control_points;
    } else if (node.task_map_type == "collision_distance") {
      output = collision_task_map(
        parents.front(), obstacle_centers, obstacle_radii,
        config.casadi_rmp2_max_obstacles, config.collision);
    } else if (is_symbolic_task_map(node.task_map_type)) {
      output = CasadiTaskMapLibrary::apply_task_map(node, parents);
    } else {
      throw std::runtime_error(
              "Unsupported task map in CasADi RMP2 backend: " + node.task_map_type);
    }
    outputs.emplace(node.name, output);
  }
  return outputs;
}

casadi::MX target_value(
  const casadi::MX & values,
  const std::vector<std::string> & keys,
  const std::string & key)
{
  const auto found = std::find(keys.begin(), keys.end(), key);
  if (found == keys.end()) {
    throw std::runtime_error("Missing symbolic target key: " + key);
  }
  const casadi_int index = static_cast<casadi_int>(std::distance(keys.begin(), found));
  return values(casadi::Slice(3 * index, 3 * index + 3));
}

casadi::MX target_active(
  const casadi::MX & active,
  const std::vector<std::string> & keys,
  const std::string & key)
{
  const auto found = std::find(keys.begin(), keys.end(), key);
  if (found == keys.end()) {
    throw std::runtime_error("Missing symbolic target key: " + key);
  }
  return active(static_cast<casadi_int>(std::distance(keys.begin(), found)));
}

void append_leaf(
  std::vector<SymbolicLeaf> & leaves,
  const casadi::MX & q,
  const casadi::MX & qd,
  casadi::MX x,
  casadi::MX x0,
  casadi::MX x00,
  casadi::MX metric,
  casadi::MX natural_force,
  bool nominal,
  const casadi::MX * curvature_override = nullptr)
{
  const casadi::MX velocity = jtimes(x, q, qd);
  const casadi::MX curvature = curvature_override == nullptr ?
    jtimes(velocity, q, qd) : *curvature_override;
  leaves.push_back(SymbolicLeaf{
    std::move(x), std::move(x0), std::move(x00), curvature,
    std::move(metric), std::move(natural_force), nominal, -1
  });
}

casadi::MX diagonal_matrix(const std::vector<casadi::MX> & values)
{
  return casadi::MX::diag(casadi::MX::vertcat(values));
}

void append_cspace_target(
  std::vector<SymbolicLeaf> & leaves,
  const casadi::MX & q,
  const casadi::MX & qd,
  const casadi::MX & x,
  const casadi::MX & x0,
  const casadi::MX & x00,
  const EigenRmpConfig & config)
{
  const casadi::MX velocity = jtimes(x, q, qd);
  const casadi::MX delta = x - vector_from_array(config.default_q);
  const casadi::MX norm = vector_norm(delta);
  const casadi::MX robust_scale = if_else(
    norm > config.cspace_target.robust_position_term_thresh,
    config.cspace_target.robust_position_term_thresh / fmax(norm, 1e-12),
    1.0);
  const casadi::MX acceleration =
    -config.cspace_target.position_gain * robust_scale * delta -
    config.cspace_target.damping_gain * velocity;
  const casadi::MX metric =
    (config.cspace_target.metric_scalar + config.cspace_target.inertia) *
    casadi::DM::eye(x.size1());
  append_leaf(
    leaves, q, qd, x, x0, x00, metric,
    casadi::MX::mtimes(metric, acceleration), true);
}

void append_joint_limits(
  std::vector<SymbolicLeaf> & leaves,
  const casadi::MX & q,
  const casadi::MX & qd,
  const casadi::MX & x,
  const casadi::MX & x0,
  const casadi::MX & x00,
  const EigenRmpConfig & config)
{
  if (config.joint_limit.policy == "paper_gds") {
    std::vector<casadi::MX> z;
    std::vector<casadi::MX> z0;
    std::vector<casadi::MX> z00;
    std::vector<casadi::MX> natural_forces;
    z.reserve(6);
    z0.reserve(6);
    z00.reserve(6);
    natural_forces.reserve(6);
    for (int joint = 0; joint < 6; ++joint) {
      const casadi_int upper = 2 * joint;
      const casadi_int lower = upper + 1;
      z.push_back(log(x(lower)) - log(x(upper)));
      z0.push_back(log(x0(lower)) - log(x0(upper)));
      z00.push_back(log(x00(lower)) - log(x00(upper)));
    }
    const casadi::MX task = casadi::MX::vertcat(z);
    const casadi::MX velocity = jtimes(task, q, qd);
    for (int joint = 0; joint < 6; ++joint) {
      const double center = config.joint_limit.gds_center_fraction;
      const double center_z = std::log(center) - std::log1p(-center);
      const double potential_gain =
        config.joint_limit.metric_scalar * config.joint_limit.accel_potential_gain;
      const double damping =
        config.joint_limit.metric_scalar * config.joint_limit.accel_damper_gain;
      natural_forces.push_back(
        -potential_gain * (task(joint) - center_z) - damping * velocity(joint));
    }
    append_leaf(
      leaves, q, qd,
      task, casadi::MX::vertcat(z0), casadi::MX::vertcat(z00),
      config.joint_limit.metric_scalar * casadi::DM::eye(6),
      casadi::MX::vertcat(natural_forces), true);
    return;
  }
  if (config.joint_limit.policy != "lula_canonical") {
    throw std::runtime_error("Unsupported CasADi RMP2 joint-limit policy");
  }

  const casadi::MX velocity = jtimes(x, q, qd);
  std::vector<casadi::MX> metrics;
  std::vector<casadi::MX> accelerations;
  metrics.reserve(12);
  accelerations.reserve(12);
  for (casadi_int row = 0; row < x.size1(); ++row) {
    const casadi::MX clearance = fmax(x(row), 0.0);
    const casadi::MX rate = velocity(row);
    const casadi::MX metric_before_gate =
      config.joint_limit.metric_scalar /
      (clearance / config.joint_limit.metric_length_scale +
      config.joint_limit.metric_exploder_eps);
    metrics.push_back(
      (1.0 - sigmoid(rate / config.joint_limit.metric_velocity_gate_length_scale)) *
      metric_before_gate);
    const casadi::MX scaled =
      clearance / config.joint_limit.accel_potential_exploder_length_scale;
    accelerations.push_back(
      config.joint_limit.accel_potential_gain /
      (scaled * scaled + config.joint_limit.accel_potential_exploder_eps) -
      config.joint_limit.accel_damper_gain * rate);
  }
  const casadi::MX metric = diagonal_matrix(metrics);
  append_leaf(
    leaves, q, qd, x, x0, x00, metric,
    casadi::MX::mtimes(metric, casadi::MX::vertcat(accelerations)), true);
}

void append_joint_velocity_cap(
  std::vector<SymbolicLeaf> & leaves,
  const casadi::MX & q,
  const casadi::MX & qd,
  const casadi::MX & x,
  const casadi::MX & x0,
  const casadi::MX & x00,
  const EigenRmpConfig & config)
{
  const casadi::MX velocity = jtimes(x, q, qd);
  std::vector<casadi::MX> metrics;
  std::vector<casadi::MX> accelerations;
  metrics.reserve(6);
  accelerations.reserve(6);
  const double threshold =
    config.joint_velocity_cap.max_velocity -
    config.joint_velocity_cap.velocity_damping_region;
  for (int joint = 0; joint < 6; ++joint) {
    const casadi::MX speed = fabs(velocity(joint));
    const casadi::MX enabled = if_else(speed >= threshold, 1.0, 0.0);
    const casadi::MX delta =
      speed - config.joint_velocity_cap.max_velocity +
      config.joint_velocity_cap.velocity_damping_region;
    const casadi::MX clipped = fmin(
      delta,
      config.joint_velocity_cap.velocity_damping_region -
      config.joint_velocity_cap.eps);
    const casadi::MX ratio =
      clipped / config.joint_velocity_cap.velocity_damping_region;
    const casadi::MX denominator = fmax(
      1.0 - ratio * ratio,
      std::max(config.joint_velocity_cap.eps, 1e-12));
    metrics.push_back(if_else(
        enabled > 0.5,
        config.joint_velocity_cap.metric_weight / denominator,
        0.0));
    accelerations.push_back(
      enabled * -fabs(config.joint_velocity_cap.damping_gain * delta) *
      if_else(velocity(joint) >= 0.0, 1.0, -1.0));
  }
  const casadi::MX metric = diagonal_matrix(metrics);
  append_leaf(
    leaves, q, qd, x, x0, x00, metric,
    casadi::MX::mtimes(metric, casadi::MX::vertcat(accelerations)), true);
}

void append_target(
  std::vector<SymbolicLeaf> & leaves,
  const casadi::MX & q,
  const casadi::MX & qd,
  const casadi::MX & x,
  const casadi::MX & x0,
  const casadi::MX & x00,
  const casadi::MX & goal,
  const casadi::MX & active,
  const EigenRmpConfig & config)
{
  const casadi::MX velocity = jtimes(x, q, qd);
  const casadi::MX delta = goal - x;
  const casadi::MX distance = vector_norm(delta);
  const casadi::MX soft_distance = fmax(
    distance, config.target.accel_norm_eps / 10.0);
  const casadi::MX direction = delta / soft_distance;
  const casadi::MX acceleration =
    config.target.accel_p_gain * delta /
    (distance + config.target.accel_norm_eps) -
    config.target.accel_d_gain * velocity;
  const casadi::MX alpha =
    (1.0 - config.target.min_metric_alpha) *
    exp(-0.5 * pow(distance / config.target.metric_alpha_length_scale, 2)) +
    config.target.min_metric_alpha;
  casadi::MX metric =
    alpha * config.target.max_metric_scalar * casadi::DM::eye(3) +
    (1.0 - alpha) * config.target.min_metric_scalar *
    casadi::MX::mtimes(direction, direction.T());
  const casadi::MX boost_alpha = exp(
    -0.5 * pow(distance / config.target.proximity_metric_boost_length_scale, 2));
  metric *= active * (
    boost_alpha * config.target.proximity_metric_boost_scalar +
    (1.0 - boost_alpha));
  append_leaf(
    leaves, q, qd, x, x0, x00, metric,
    casadi::MX::mtimes(metric, acceleration), true);
}

void append_axis_target(
  std::vector<SymbolicLeaf> & leaves,
  const casadi::MX & q,
  const casadi::MX & qd,
  const casadi::MX & x,
  const casadi::MX & x0,
  const casadi::MX & x00,
  const casadi::MX & goal,
  const casadi::MX & active,
  const casadi::MX * tcp_position,
  const casadi::MX * position_goal,
  const AxisTargetParams & params,
  bool nominal,
  bool zero_curvature)
{
  const casadi::MX velocity = jtimes(x, q, qd);
  const casadi::MX x_unit = x / vector_norm(x);
  const casadi::MX goal_unit = goal / vector_norm(goal);
  const casadi::MX acceleration =
    params.accel_p_gain * (goal_unit - x_unit) -
    params.accel_d_gain * velocity;
  casadi::MX metric_scalar = active * params.metric_scalar;
  if (tcp_position != nullptr && position_goal != nullptr) {
    const casadi::MX distance = vector_norm(*position_goal - *tcp_position);
    const casadi::MX boost_alpha = exp(
      -0.5 * pow(
        distance / fmax(params.proximity_metric_boost_length_scale, 1e-9), 2));
    metric_scalar *=
      boost_alpha * params.proximity_metric_boost_scalar + (1.0 - boost_alpha);
  }
  const casadi::MX metric = metric_scalar * casadi::DM::eye(3);
  const casadi::MX natural_force = casadi::MX::mtimes(metric, acceleration);
  if (zero_curvature) {
    const casadi::MX curvature = casadi::MX::zeros(3, 1);
    append_leaf(
      leaves, q, qd, x, x0, x00, metric, natural_force,
      nominal, &curvature);
  } else {
    append_leaf(
      leaves, q, qd, x, x0, x00, metric, natural_force, nominal);
  }
}

void append_collision(
  std::vector<SymbolicLeaf> & leaves,
  const casadi::MX & q,
  const casadi::MX & qd,
  const casadi::MX & x,
  const casadi::MX & x0,
  const casadi::MX & x00,
  const casadi::MX & obstacle_active,
  const EigenRmpConfig & config)
{
  const casadi::MX velocity = jtimes(x, q, qd);
  std::vector<casadi::MX> metrics;
  std::vector<casadi::MX> forces;
  metrics.reserve(static_cast<std::size_t>(x.size1()));
  forces.reserve(static_cast<std::size_t>(x.size1()));
  for (casadi_int row = 0; row < x.size1(); ++row) {
    const int obstacle = static_cast<int>(row) % config.casadi_rmp2_max_obstacles;
    const casadi::MX mask = obstacle_active(obstacle);
    const casadi::MX signed_clearance = x(row);
    const casadi::MX rate = velocity(row);
    if (config.collision.policy == "paper_gds") {
      const casadi::MX clearance = signed_clearance;
      const double smoothing = config.collision.gds_clearance_smoothing;
      const casadi::MX radius = sqrt(clearance * clearance + smoothing * smoothing);
      const casadi::MX positive = 0.5 * (clearance + radius);
      const casadi::MX positive_derivative = 0.5 * (1.0 + clearance / radius);
      const double modulation_radius = config.collision.metric_modulation_radius;
      const casadi::MX z = (modulation_radius - clearance) / modulation_radius;
      const casadi::MX smoothstep = pow(z, 3) * (10.0 + z * (-15.0 + 6.0 * z));
      const casadi::MX smoothstep_derivative =
        30.0 * z * z * (1.0 - z) * (1.0 - z);
      const casadi::MX gate = if_else(
        clearance <= 0.0, 1.0,
        if_else(clearance >= modulation_radius, 0.0, smoothstep * smoothstep));
      const casadi::MX gate_derivative = if_else(
        clearance <= 0.0, 0.0,
        if_else(
          clearance >= modulation_radius, 0.0,
          -2.0 * smoothstep * smoothstep_derivative / modulation_radius));
      const casadi::MX denominator =
        config.collision.metric_exploder_eps +
        positive / config.collision.metric_exploder_std_dev;
      const casadi::MX denominator_derivative =
        positive_derivative / config.collision.metric_exploder_std_dev;
      const casadi::MX w = config.collision.metric_scalar * gate / denominator;
      const casadi::MX w_derivative = config.collision.metric_scalar *
        (gate_derivative * denominator - gate * denominator_derivative) /
        (denominator * denominator);
      const casadi::MX negative_rate = fmin(rate, -1e-12);
      const casadi::MX ratio =
        config.collision.gds_metric_velocity_scale / (-negative_rate);
      const casadi::MX approach_profile = exp(-ratio * ratio);
      const casadi::MX u = if_else(
        rate >= 0.0,
        config.collision.gds_metric_velocity_floor,
        config.collision.gds_metric_velocity_floor +
        (1.0 - config.collision.gds_metric_velocity_floor) * approach_profile);
      const casadi::MX rate_times_u_derivative = if_else(
        rate >= 0.0, 0.0,
        2.0 * (1.0 - config.collision.gds_metric_velocity_floor) *
        approach_profile * ratio * ratio);
      const casadi::MX G = w * u;
      const casadi::MX Xi = 0.5 * w * rate_times_u_derivative;
      const casadi::MX metric = G + Xi;
      const casadi::MX xi = 0.5 * u * w_derivative * rate * rate;
      const casadi::MX repulsion = config.collision.repulsion_gain *
        exp(-positive / config.collision.repulsion_std_dev);
      const casadi::MX potential_force =
        w * config.collision.gds_metric_velocity_floor * repulsion;
      const casadi::MX damping_gate = sigmoid(
        -rate / config.collision.gds_damping_velocity_scale);
      const casadi::MX damping_denominator =
        config.collision.damping_robustness_eps +
        positive / config.collision.damping_std_dev;
      const casadi::MX damping =
        metric * config.collision.damping_gain *
        damping_gate / damping_denominator;
      metrics.push_back(mask * metric);
      forces.push_back(mask * (potential_force - xi - damping * rate));
      continue;
    }
    if (
      config.collision.policy != "lula_canonical" &&
      config.collision.policy != "repulsive")
    {
      throw std::runtime_error("Unsupported CasADi RMP2 collision policy");
    }
    const casadi::MX clearance = fmax(signed_clearance, 0.0);
    casadi::MX metric = config.collision.metric_scalar /
      (clearance / config.collision.metric_exploder_std_dev +
      config.collision.metric_exploder_eps);
    const double radius = config.collision.metric_modulation_radius;
    const casadi::MX gate = if_else(
      clearance > radius,
      0.0,
      clearance * clearance / (radius * radius) -
      2.0 * clearance / radius + 1.0);
    metric *= gate * (1.0 - sigmoid(
        rate / config.collision.damping_velocity_gate_length_scale));
    const casadi::MX repel = config.collision.repulsion_gain *
      exp(-clearance / config.collision.repulsion_std_dev);
    const casadi::MX damping =
      -(1.0 - sigmoid(
        rate / config.collision.damping_velocity_gate_length_scale)) *
      config.collision.damping_gain * rate /
      (clearance / config.collision.damping_std_dev +
      config.collision.damping_robustness_eps);
    metrics.push_back(mask * metric);
    forces.push_back(mask * metric * (repel + damping));
  }
  append_leaf(
    leaves, q, qd, x, x0, x00,
    diagonal_matrix(metrics), casadi::MX::vertcat(forces), false);
}

void append_damping(
  std::vector<SymbolicLeaf> & leaves,
  const casadi::MX & q,
  const casadi::MX & qd,
  const casadi::MX & x,
  const casadi::MX & x0,
  const casadi::MX & x00,
  const EigenRmpConfig & config)
{
  const casadi::MX velocity = jtimes(x, q, qd);
  const casadi::MX speed = vector_norm(velocity);
  const casadi::MX acceleration =
    -config.damping.accel_d_gain * speed * velocity;
  const casadi::MX metric =
    (config.damping.metric_scalar * speed + config.damping.inertia) *
    casadi::DM::eye(x.size1());
  append_leaf(
    leaves, q, qd, x, x0, x00, metric,
    casadi::MX::mtimes(metric, acceleration), true);
}

enum class LeafSelection
{
  All,
  Nominal,
  NonNominal
};

std::pair<casadi::MX, casadi::MX> algorithm3_pullback(
  const std::vector<SymbolicLeaf> & leaves,
  const casadi::MX & q0,
  const casadi::MX & q00,
  LeafSelection selection,
  int ablation_group = -1)
{
  casadi::MX metric_bilinear = scalar(0.0);
  casadi::MX force_pairing = scalar(0.0);
  for (const auto & leaf : leaves) {
    if (selection == LeafSelection::Nominal && !leaf.nominal) {
      continue;
    }
    if (selection == LeafSelection::NonNominal && leaf.nominal) {
      continue;
    }
    if (ablation_group >= 0 && leaf.ablation_group != ablation_group) {
      continue;
    }
    metric_bilinear += casadi::MX::mtimes({leaf.x0.T(), leaf.metric, leaf.x00})(0);
    force_pairing += dot(
      leaf.x0,
      leaf.natural_force - casadi::MX::mtimes(leaf.metric, leaf.curvature));
  }
  const casadi::MX force = gradient(force_pairing, q0);
  const casadi::MX first_reverse = gradient(metric_bilinear, q0);
  const casadi::MX metric = casadi::MX::jacobian(first_reverse, q00);
  return {metric, force};
}

}  // namespace

struct CasadiRmp2Solver::Impl
{
  Impl(EigenRmpConfig solver_config, std::string urdf_path)
  : config(std::move(solver_config)),
    model(std::move(urdf_path)),
    tangent_escape_supervisor(config)
  {
    if (config.solve_method != "rmp2") {
      throw std::runtime_error("CasADi RMP2 backend requires solve_method=rmp2");
    }
    if (config.casadi_rmp2_max_obstacles <= 0) {
      throw std::runtime_error("casadi_rmp2_max_obstacles must be positive");
    }
    if (config.rmp_type != "canonical" && config.rmp_type != "natural") {
      throw std::runtime_error("CasADi RMP2 backend requires canonical or natural RMP type");
    }
    const auto & energy = config.escape_energy_certificate;
    if (
      !std::isfinite(energy.tank_capacity) ||
      !std::isfinite(energy.initial_energy) ||
      !std::isfinite(energy.control_dt) ||
      !std::isfinite(energy.power_tolerance) ||
      energy.tank_capacity < 0.0 ||
      energy.initial_energy < 0.0 ||
      energy.initial_energy > energy.tank_capacity ||
      energy.control_dt <= 0.0 ||
      energy.scale_search_iterations < 1 ||
      energy.power_tolerance < 0.0)
    {
      throw std::runtime_error("Invalid Escape stability energy-certificate parameters");
    }
    for (const auto & node : config.graph_nodes) {
      if (
        node.enabled &&
        (node.leaf_rmp_type == "external" ||
        node.handcrafted_leaf_rmp_type == "external"))
      {
        throw std::runtime_error(
                "CasADi RMP2 backend does not yet support dynamic external leaf dimensions");
      }
    }

    targets = collect_target_keys(config.graph_nodes);
    const auto order = topological_order(config.graph_nodes);
    for (const auto & node : config.graph_nodes) {
      if (!node.enabled) {
        continue;
      }
      tangent_escape_leaf_enabled =
        tangent_escape_leaf_enabled ||
        node.leaf_rmp_type == "tangent_escape" ||
        node.handcrafted_leaf_rmp_type == "tangent_escape";
    }

    const casadi::MX q = casadi::MX::sym("q", 6, 1);
    const casadi::MX qd = casadi::MX::sym("qd", 6, 1);
    const casadi::MX q0 = casadi::MX::sym("q0", 6, 1);
    const casadi::MX q00 = casadi::MX::sym("q00", 6, 1);
    const casadi::MX target_values = casadi::MX::sym(
      "target_values", 3 * static_cast<int>(targets.size()), 1);
    const casadi::MX target_mask = casadi::MX::sym(
      "target_mask", static_cast<int>(targets.size()), 1);
    const casadi::MX obstacle_centers = casadi::MX::sym(
      "obstacle_centers", 3 * config.casadi_rmp2_max_obstacles, 1);
    const casadi::MX obstacle_radii = casadi::MX::sym(
      "obstacle_radii", config.casadi_rmp2_max_obstacles, 1);
    const casadi::MX obstacle_mask = casadi::MX::sym(
      "obstacle_mask", config.casadi_rmp2_max_obstacles, 1);
    const auto nodes = build_node_outputs(
      q, obstacle_centers, obstacle_radii,
      model.task_map_function(), config, order);
    const auto nodes0 = build_node_outputs(
      q0, obstacle_centers, obstacle_radii,
      model.task_map_function(), config, order);
    const auto nodes00 = build_node_outputs(
      q00, obstacle_centers, obstacle_radii,
      model.task_map_function(), config, order);

    std::vector<SymbolicLeaf> leaves;
    bool target_node_enabled = false;
    const casadi::MX * tcp_position = nullptr;
    const casadi::MX * position_goal = nullptr;
    const auto tcp_node = nodes.find("tcp_position");
    if (tcp_node != nodes.end()) {
      tcp_position = &tcp_node->second;
    }
    if (std::find(targets.begin(), targets.end(), "goal") != targets.end()) {
      position_goal_value = target_value(target_values, targets, "goal");
      position_goal = &position_goal_value;
    }
    for (const auto & node : config.graph_nodes) {
      target_node_enabled = target_node_enabled ||
        (node.enabled && node.name == "target");
    }

    for (const std::size_t index : order) {
      const auto & node = config.graph_nodes[index];
      const auto & x = nodes.at(node.name);
      const auto & x0_value = nodes0.at(node.name);
      const auto & x00_value = nodes00.at(node.name);
      const auto add = [&](const std::string & leaf_type) {
          if (leaf_type.empty() || leaf_type == "none") {
            return;
          }
          const std::size_t first_leaf = leaves.size();
          if (leaf_type == "cspace_target") {
            append_cspace_target(leaves, q, qd, x, x0_value, x00_value, config);
          } else if (leaf_type == "joint_limit") {
            append_joint_limits(leaves, q, qd, x, x0_value, x00_value, config);
          } else if (leaf_type == "joint_velocity_cap") {
            append_joint_velocity_cap(leaves, q, qd, x, x0_value, x00_value, config);
          } else if (leaf_type == "target") {
            append_target(
              leaves, q, qd, x, x0_value, x00_value,
              target_value(target_values, targets, node.target_key),
              target_active(target_mask, targets, node.target_key), config);
          } else if (leaf_type == "axis_target") {
            append_axis_target(
              leaves, q, qd, x, x0_value, x00_value,
              target_value(target_values, targets, node.target_key),
              target_active(target_mask, targets, node.target_key),
              target_node_enabled ? tcp_position : nullptr,
              target_node_enabled ? position_goal : nullptr,
              config.axis_target, true, false);
          } else if (leaf_type == "wrist_axis_target") {
            append_axis_target(
              leaves, q, qd, x, x0_value, x00_value,
              target_value(target_values, targets, node.target_key),
              target_active(target_mask, targets, node.target_key),
              target_node_enabled ? tcp_position : nullptr,
              target_node_enabled ? position_goal : nullptr,
              config.wrist_axis_target, true, true);
          } else if (leaf_type == "collision") {
            append_collision(
              leaves, q, qd, x, x0_value, x00_value,
              obstacle_mask, config);
          } else if (leaf_type == "damping") {
            append_damping(leaves, q, qd, x, x0_value, x00_value, config);
          } else if (leaf_type == "tangent_escape") {
            // The stateful supervisor contributes this rank-one leaf directly
            // to the root natural form. Its symbolic inputs were always zero,
            // so adding a second graph leaf only repeated dead kinematics.
          } else {
            throw std::runtime_error(
                    "Unsupported leaf in CasADi RMP2 backend: " + leaf_type);
          }
          const int group_id = leaf_ablation_group_id(node, leaf_type);
          for (std::size_t leaf = first_leaf; leaf < leaves.size(); ++leaf) {
            leaves[leaf].ablation_group = group_id;
          }
        };
      add(node.leaf_rmp_type);
      add(node.handcrafted_leaf_rmp_type);
    }

    nominal_root_enabled = config.tangent_escape.enabled && tangent_escape_leaf_enabled;
    std::pair<casadi::MX, casadi::MX> root;
    std::pair<casadi::MX, casadi::MX> nominal_root;
    if (nominal_root_enabled) {
      nominal_root = algorithm3_pullback(
        leaves, q0, q00, LeafSelection::Nominal);
      const auto non_nominal_root = algorithm3_pullback(
        leaves, q0, q00, LeafSelection::NonNominal);
      root = {
        nominal_root.first + non_nominal_root.first,
        nominal_root.second + non_nominal_root.second};
    } else {
      root = algorithm3_pullback(leaves, q0, q00, LeafSelection::All);
    }
    std::vector<casadi::MX> function_outputs{root.first, root.second};
    std::vector<std::string> function_output_names{"root_metric", "root_force"};
    if (nominal_root_enabled) {
      function_outputs.push_back(nominal_root.first);
      function_outputs.push_back(nominal_root.second);
      function_output_names.push_back("nominal_metric");
      function_output_names.push_back("nominal_force");
    }
    if (config.enable_leaf_ablation_diagnostics) {
      std::set<int> unique_groups;
      for (const auto & leaf : leaves) {
        if (leaf.ablation_group >= 0) {
          unique_groups.insert(leaf.ablation_group);
        }
      }
      if (tangent_escape_leaf_enabled) {
        unique_groups.insert(static_cast<int>(LeafAblationGroup::TangentEscape));
      }
      ablation_group_ids.assign(unique_groups.begin(), unique_groups.end());
      for (const int group_id : ablation_group_ids) {
        const auto contribution =
          group_id == static_cast<int>(LeafAblationGroup::TangentEscape) ?
          std::make_pair(casadi::MX::zeros(6, 6), casadi::MX::zeros(6, 1)) :
          algorithm3_pullback(
          leaves, q0, q00, LeafSelection::All, group_id);
        function_outputs.push_back(contribution.first);
        function_outputs.push_back(contribution.second);
        function_output_names.push_back("ablation_metric_" + std::to_string(group_id));
        function_output_names.push_back("ablation_force_" + std::to_string(group_id));
      }
    }
    casadi::Dict options;
    if (config.casadi_rmp2_jit) {
      options["jit"] = true;
      options["compiler"] = "shell";
    }
    function = casadi::Function(
      "rb10_casadi_rmp2_policy",
      {
        q, qd, target_values, target_mask,
        obstacle_centers, obstacle_radii, obstacle_mask,
        q0, q00
      },
      function_outputs,
      {
        "q", "qd", "target_values", "target_mask",
        "obstacle_centers", "obstacle_radii", "obstacle_mask",
        "q0", "q00"
      },
      function_output_names,
      options);
  }

  EigenRmpConfig config;
  CasadiRb10Kinematics model;
  mutable TangentEscapeSupervisor tangent_escape_supervisor;
  mutable EscapeEnergyCertificateState escape_energy_certificate_state;
  bool tangent_escape_leaf_enabled{false};
  bool nominal_root_enabled{false};
  std::vector<std::string> targets;
  std::vector<int> ablation_group_ids;
  casadi::MX position_goal_value;
  casadi::Function function;
};

CasadiRmp2Solver::CasadiRmp2Solver(EigenRmpConfig config, std::string urdf_path)
: impl_(std::make_shared<Impl>(std::move(config), std::move(urdf_path)))
{}

void CasadiRmp2Solver::set_tangent_observations(const tangent_sensor_history::Input & input) const
{
  impl_->tangent_escape_supervisor.set_observations(input);
}

RmpSolveResult CasadiRmp2Solver::solve(
  const JointVector & q,
  const JointVector & qd,
  const std::unordered_map<std::string, Eigen::Vector3d> & vector_targets,
  const std::vector<ObstacleSphere> & obstacles,
  const std::unordered_map<std::string, ExternalRmpFeature> & external_rmps) const
{
  if (!external_rmps.empty()) {
    throw std::runtime_error(
            "CasADi RMP2 backend does not support runtime external RMPs yet");
  }
  if (obstacles.size() > static_cast<std::size_t>(impl_->config.casadi_rmp2_max_obstacles)) {
    throw std::runtime_error(
            "Obstacle count exceeds casadi_rmp2_max_obstacles");
  }

  casadi::DM targets = casadi::DM::zeros(
    3 * static_cast<int>(impl_->targets.size()), 1);
  casadi::DM target_mask = casadi::DM::zeros(
    static_cast<int>(impl_->targets.size()), 1);
  for (std::size_t index = 0; index < impl_->targets.size(); ++index) {
    const auto found = vector_targets.find(impl_->targets[index]);
    if (found == vector_targets.end()) {
      continue;
    }
    for (int axis = 0; axis < 3; ++axis) {
      targets(static_cast<casadi_int>(3 * index + axis)) = found->second[axis];
    }
    target_mask(static_cast<casadi_int>(index)) = 1.0;
  }

  const int max_obstacles = impl_->config.casadi_rmp2_max_obstacles;
  casadi::DM centers = casadi::DM::zeros(3 * max_obstacles, 1);
  casadi::DM radii = casadi::DM::zeros(max_obstacles, 1);
  casadi::DM obstacle_mask = casadi::DM::zeros(max_obstacles, 1);
  for (std::size_t index = 0; index < obstacles.size(); ++index) {
    const auto & obstacle = obstacles[index];
    for (int axis = 0; axis < 3; ++axis) {
      centers(static_cast<casadi_int>(3 * index + axis)) = obstacle.center[axis];
    }
    radii(static_cast<casadi_int>(index)) = obstacle.radius;
    obstacle_mask(static_cast<casadi_int>(index)) = 1.0;
  }

  const casadi::DM q_dm = joint_vector_to_dm(q);
  const std::vector<casadi::DM> inputs{
    q_dm,
    joint_vector_to_dm(qd),
    targets,
    target_mask,
    centers,
    radii,
    obstacle_mask,
    q_dm,
    q_dm
  };
  const std::vector<casadi::DM> outputs = impl_->function(inputs);
  const std::size_t root_output_count = impl_->nominal_root_enabled ? 4U : 2U;
  const std::size_t expected_output_count =
    root_output_count + 2 * impl_->ablation_group_ids.size();
  if (outputs.size() != expected_output_count) {
    throw std::runtime_error("Unexpected CasADi RMP2 policy output count");
  }

  const Eigen::MatrixXd metric_dynamic = dm_to_matrix(outputs[0]);
  const Eigen::VectorXd force_dynamic = dm_to_vector(outputs[1]);
  if (
    metric_dynamic.rows() != 6 || metric_dynamic.cols() != 6 ||
    force_dynamic.size() != 6)
  {
    throw std::runtime_error("CasADi RMP2 root output dimension mismatch");
  }
  const Matrix6 base_metric = metric_dynamic;
  const JointVector base_force = force_dynamic;
  Matrix6 nominal_metric = Matrix6::Zero();
  JointVector nominal_force = JointVector::Zero();
  if (impl_->nominal_root_enabled) {
    const Eigen::MatrixXd nominal_metric_dynamic = dm_to_matrix(outputs[2]);
    const Eigen::VectorXd nominal_force_dynamic = dm_to_vector(outputs[3]);
    if (
      nominal_metric_dynamic.rows() != 6 || nominal_metric_dynamic.cols() != 6 ||
      nominal_force_dynamic.size() != 6)
    {
      throw std::runtime_error("CasADi RMP2 nominal root output dimension mismatch");
    }
    nominal_metric = nominal_metric_dynamic;
    nominal_force = nominal_force_dynamic;
  }

  struct RootContribution
  {
    Matrix6 metric{Matrix6::Zero()};
    JointVector force{JointVector::Zero()};
  };
  std::unordered_map<int, RootContribution> leaf_contributions;
  for (std::size_t index = 0; index < impl_->ablation_group_ids.size(); ++index) {
    const Eigen::MatrixXd contribution_metric =
      dm_to_matrix(outputs[root_output_count + 2 * index]);
    const Eigen::VectorXd contribution_force =
      dm_to_vector(outputs[root_output_count + 2 * index + 1]);
    if (
      contribution_metric.rows() != 6 || contribution_metric.cols() != 6 ||
      contribution_force.size() != 6)
    {
      throw std::runtime_error("CasADi RMP2 ablation output dimension mismatch");
    }
    leaf_contributions.emplace(
      impl_->ablation_group_ids[index],
      RootContribution{Matrix6(contribution_metric), JointVector(contribution_force)});
  }

  TangentEscapeCommand escape_command;
  KinematicsContext context;
  bool context_valid = false;
  if (impl_->nominal_root_enabled) {
    const auto goal = vector_targets.find("goal");
    if (goal != vector_targets.end()) {
      context = impl_->model.forward_context(q, qd);
      context_valid = true;
      const JointVector nominal_qdd = resolve_root(
        nominal_metric, nominal_force, impl_->config.solve_offset);
      escape_command = impl_->tangent_escape_supervisor.evaluate(
        q,
        context,
        qd,
        goal->second,
        obstacles,
        nominal_qdd,
        base_metric,
        base_force);
    }
  }

  const Matrix6 requested_escape_metric = escape_command.metric_contribution;
  const JointVector requested_escape_force = escape_command.force_contribution;
  const auto & energy_params = impl_->config.escape_energy_certificate;
  auto & energy_state = impl_->escape_energy_certificate_state;
  const bool certificate_enabled =
    impl_->config.enable_stability_certificate_diagnostics || energy_params.enabled;
  const bool dual_solve_enabled =
    impl_->config.enable_tangent_escape_dual_solve_diagnostics;
  const bool limited_solve_needed =
    dual_solve_enabled || certificate_enabled ||
    impl_->config.enable_leaf_ablation_diagnostics;
  struct ScaledEscapeSolve
  {
    Matrix6 metric{Matrix6::Zero()};
    JointVector force{JointVector::Zero()};
    JointVector raw_qdd{JointVector::Zero()};
    JointVector command_qdd{JointVector::Zero()};
    double escape_power{0.0};
    double solve_power{0.0};
    double clamp_power{0.0};
  };
  const auto solve_scaled_escape = [&](double scale) {
      ScaledEscapeSolve sample;
      const double bounded_scale = std::clamp(scale, 0.0, 1.0);
      const Matrix6 escape_metric = bounded_scale * requested_escape_metric;
      const JointVector escape_force = bounded_scale * requested_escape_force;
      sample.metric = base_metric + escape_metric;
      sample.force = base_force + escape_force;
      sample.raw_qdd = resolve_root(
        sample.metric, sample.force, impl_->config.solve_offset);
      sample.command_qdd = sample.raw_qdd;
      if (limited_solve_needed) {
        sample.command_qdd = limit_joint_acceleration(
          sample.raw_qdd,
          impl_->config.max_joint_accel,
          impl_->config.preserve_joint_accel_direction);
      }
      if (certificate_enabled) {
        sample.escape_power =
          qd.dot(escape_force - escape_metric * sample.command_qdd);
        sample.solve_power = qd.dot(sample.metric * sample.raw_qdd - sample.force);
        sample.clamp_power =
          qd.dot(sample.metric * (sample.command_qdd - sample.raw_qdd));
      }
      return sample;
    };

  if (certificate_enabled && !energy_state.initialized) {
    energy_state.initialized = true;
    energy_state.tank_energy = std::clamp(
      energy_params.initial_energy, 0.0, energy_params.tank_capacity);
  }

  const ScaledEscapeSolve requested_solve = solve_scaled_escape(1.0);
  double escape_scale = 1.0;
  ScaledEscapeSolve final_solve = requested_solve;
  if (energy_params.enabled && requested_solve.escape_power > 0.0) {
    const double positive_power_budget =
      energy_state.tank_energy / std::max(energy_params.control_dt, 1e-12);
    if (requested_solve.escape_power > positive_power_budget) {
      double feasible_scale = 0.0;
      double infeasible_scale = 1.0;
      for (int iteration = 0; iteration < energy_params.scale_search_iterations; ++iteration) {
        const double trial_scale = 0.5 * (feasible_scale + infeasible_scale);
        const auto trial = solve_scaled_escape(trial_scale);
        if (trial.escape_power <= positive_power_budget) {
          feasible_scale = trial_scale;
        } else {
          infeasible_scale = trial_scale;
        }
      }
      escape_scale = feasible_scale;
      final_solve = solve_scaled_escape(escape_scale);
      const double required_energy =
        energy_params.control_dt * std::max(final_solve.escape_power, 0.0);
      if (!std::isfinite(required_energy) || required_energy > energy_state.tank_energy) {
        escape_scale = 0.0;
        final_solve = solve_scaled_escape(0.0);
      }
    }
  }
  ScaledEscapeSolve no_escape_solve;
  if (dual_solve_enabled || certificate_enabled) {
    no_escape_solve = solve_scaled_escape(0.0);
  }

  const double positive_escape_energy =
    energy_params.control_dt * std::max(final_solve.escape_power, 0.0);
  const double negative_escape_energy =
    energy_params.control_dt * std::max(-final_solve.escape_power, 0.0);
  bool environment_static = false;
  if (certificate_enabled) {
    environment_static =
      energy_state.previous_environment_valid &&
      external_rmps.empty() &&
      same_rmp_environment(
      energy_state.previous_vector_targets,
      vector_targets,
      energy_state.previous_obstacles,
      obstacles);
    energy_state.previous_vector_targets = vector_targets;
    energy_state.previous_obstacles = obstacles;
    energy_state.previous_environment_valid = true;
    if (energy_params.enabled) {
      energy_state.tank_energy = std::clamp(
        energy_state.tank_energy - positive_escape_energy,
        0.0,
        energy_params.tank_capacity);
    }
    energy_state.positive_energy_integral += positive_escape_energy;
    energy_state.negative_energy_integral += negative_escape_energy;
    energy_state.net_energy_integral +=
      energy_params.control_dt * final_solve.escape_power;
    ++energy_state.sample_count;
  }

  if (impl_->config.enable_leaf_ablation_diagnostics) {
    const int escape_group = static_cast<int>(LeafAblationGroup::TangentEscape);
    auto escape_contribution = leaf_contributions.find(escape_group);
    if (escape_contribution != leaf_contributions.end()) {
      escape_contribution->second.metric = escape_scale * requested_escape_metric;
      escape_contribution->second.force = escape_scale * requested_escape_force;
    }
  }

  if (impl_->config.tangent_escape.enabled) {
    impl_->tangent_escape_supervisor.set_previous_qdd(
      limit_joint_acceleration(
        final_solve.raw_qdd,
        impl_->config.max_joint_accel,
        impl_->config.preserve_joint_accel_direction));
  }
  RmpSolveResult result;
  result.metric = final_solve.metric;
  result.force = final_solve.force;
  result.qdd = final_solve.raw_qdd;
  if (impl_->config.enable_tangent_escape_diagnostics) {
    result.tangent_escape_rmp_data = escape_command.debug_data;
  }

  if (dual_solve_enabled && context_valid) {
    std::vector<double> dual(43, 0.0);
    dual[0] =
      requested_escape_metric.norm() > 1e-12 || requested_escape_force.norm() > 1e-12 ?
      1.0 : 0.0;
    for (int joint = 0; joint < 6; ++joint) {
      dual[1 + joint] = final_solve.command_qdd[joint];
      dual[7 + joint] = no_escape_solve.command_qdd[joint];
      dual[13 + joint] =
        final_solve.command_qdd[joint] - no_escape_solve.command_qdd[joint];
    }
    const Eigen::Vector3d tcp_with =
      context.tcp_jacobian * final_solve.command_qdd + context.tcp_curvature;
    const Eigen::Vector3d tcp_without =
      context.tcp_jacobian * no_escape_solve.command_qdd + context.tcp_curvature;
    const Eigen::Vector3d tcp_delta = tcp_with - tcp_without;
    for (int axis = 0; axis < 3; ++axis) {
      dual[19 + axis] = tcp_with[axis];
      dual[22 + axis] = tcp_without[axis];
      dual[25 + axis] = tcp_delta[axis];
    }
    if (escape_command.active) {
      const auto point = escape_command.control_point_index;
      Eigen::Vector3d tangent = escape_command.tangent.normalized();
      dual[28] = tcp_delta.dot(tangent);
      dual[30] = escape_command.debug_data.size() > 6 ?
        escape_command.debug_data[6] : 0.0;
      dual[31] = escape_scale * escape_command.metric_scalar;
      const Eigen::Vector3d cp_with =
        context.control_point_jacobians[point] * final_solve.command_qdd +
        context.control_point_curvatures[point];
      const Eigen::Vector3d cp_without =
        context.control_point_jacobians[point] * no_escape_solve.command_qdd +
        context.control_point_curvatures[point];
      const Eigen::Vector3d cp_delta = cp_with - cp_without;
      for (int axis = 0; axis < 3; ++axis) {
        dual[32 + axis] = cp_with[axis];
        dual[35 + axis] = cp_without[axis];
        dual[38 + axis] = cp_delta[axis];
      }
      dual[41] = cp_delta.dot(tangent);
      if (escape_command.debug_data.size() >= 20) {
        Eigen::Vector3d normal{
          escape_command.debug_data[17],
          escape_command.debug_data[18],
          escape_command.debug_data[19]};
        if (normal.norm() > 1e-12) {
          normal.normalize();
          dual[29] = tcp_delta.dot(normal);
          dual[42] = cp_delta.dot(normal);
        }
      }
    }
    result.tangent_escape_dual_solve_data = std::move(dual);
  }

  if (certificate_enabled) {
    if (!context_valid) {
      context = impl_->model.forward_context(q, qd);
      context_valid = true;
    }
    bool base_domain_valid =
      q.allFinite() && qd.allFinite() &&
      base_metric.allFinite() && base_force.allFinite();
    for (const auto & target : vector_targets) {
      base_domain_valid = base_domain_valid && target.second.allFinite();
    }
    for (const auto & obstacle : obstacles) {
      base_domain_valid =
        base_domain_valid && obstacle.center.allFinite() &&
        std::isfinite(obstacle.radius) && obstacle.radius >= 0.0;
    }
    for (std::size_t index = 0; index < RB10Model::LINK_COUNT; ++index) {
      base_domain_valid =
        base_domain_valid &&
        context.link_positions[index].allFinite() &&
        context.link_rotations[index].allFinite() &&
        context.link_jacobians[index].allFinite() &&
        context.link_velocities[index].allFinite() &&
        context.link_curvatures[index].allFinite();
    }
    for (std::size_t index = 0; index < context.control_points.size(); ++index) {
      base_domain_valid =
        base_domain_valid &&
        context.control_points[index].position.allFinite() &&
        context.control_point_jacobians[index].allFinite() &&
        context.control_point_velocities[index].allFinite() &&
        context.control_point_curvatures[index].allFinite();
    }
    if (impl_->config.collision.policy == "paper_gds") {
      for (const auto & point : context.control_points) {
        for (const auto & obstacle : obstacles) {
          if ((point.position - obstacle.center).norm() <= 1e-9) {
            base_domain_valid = false;
          }
        }
      }
    }
    for (const auto & node : impl_->config.graph_nodes) {
      if (
        !node.enabled ||
        (node.leaf_rmp_type != "axis_target" &&
        node.handcrafted_leaf_rmp_type != "axis_target"))
      {
        continue;
      }
      const auto target = vector_targets.find(node.target_key);
      if (target == vector_targets.end() || target->second.norm() <= 1e-9) {
        base_domain_valid = false;
      }
    }

    const Matrix6 symmetric_base_metric =
      0.5 * (base_metric + base_metric.transpose());
    Eigen::SelfAdjointEigenSolver<Matrix6> eigensolver;
    if (symmetric_base_metric.allFinite()) {
      eigensolver.compute(symmetric_base_metric, Eigen::EigenvaluesOnly);
    }
    const bool base_metric_spd =
      base_metric.allFinite() &&
      base_metric.isApprox(base_metric.transpose(), 1e-10) &&
      eigensolver.info() == Eigen::Success &&
      eigensolver.eigenvalues().minCoeff() >
      1e-12 * std::max(1.0, eigensolver.eigenvalues().maxCoeff());
    const double base_solve_residual =
      (base_metric * no_escape_solve.raw_qdd - base_force).norm();
    const bool base_solve_valid =
      no_escape_solve.raw_qdd.allFinite() &&
      std::isfinite(base_solve_residual) &&
      base_solve_residual <= 1e-8 * (1.0 + base_force.norm());
    const bool base_config_profile_valid =
      base_graph_has_structured_gds_form(impl_->config);
    const bool base_gds_structural =
      base_config_profile_valid && base_domain_valid &&
      base_metric_spd && base_solve_valid;
    const double numerical_power =
      final_solve.solve_power + final_solve.clamp_power;
    const double conditional_nonincrease_upper_bound =
      numerical_power +
      (energy_params.enabled ? 0.0 : std::max(final_solve.escape_power, 0.0));
    const bool conditional_nonincrease_certificate =
      base_gds_structural && environment_static && external_rmps.empty() &&
      conditional_nonincrease_upper_bound <= energy_params.power_tolerance;
    const bool clamp_active =
      (final_solve.command_qdd - final_solve.raw_qdd).norm() > 1e-12;
    const double tank_spent = energy_params.initial_energy - energy_state.tank_energy;
    const double guarded_energy_residual = energy_params.enabled ?
      tank_spent - energy_state.positive_energy_integral : 0.0;
    const double guarded_energy_violation = energy_params.enabled ?
      std::max(
      energy_state.positive_energy_integral - energy_params.initial_energy,
      0.0) : 0.0;
    result.stability_certificate_data = {
      1.0,
      base_gds_structural ? 1.0 : 0.0,
      environment_static ? 1.0 : 0.0,
      external_rmps.empty() ? 1.0 : 0.0,
      energy_params.enabled ? 1.0 : 0.0,
      conditional_nonincrease_certificate ? 1.0 : 0.0,
      escape_scale,
      energy_state.tank_energy,
      energy_params.tank_capacity,
      requested_solve.escape_power,
      final_solve.escape_power,
      final_solve.solve_power,
      final_solve.clamp_power,
      numerical_power,
      energy_state.positive_energy_integral,
      energy_state.negative_energy_integral,
      energy_state.net_energy_integral,
      static_cast<double>(energy_state.sample_count),
      requested_escape_metric.trace(),
      (escape_scale * requested_escape_metric).trace(),
      requested_escape_force.norm(),
      (escape_scale * requested_escape_force).norm(),
      final_solve.raw_qdd.norm(),
      final_solve.command_qdd.norm(),
      clamp_active ? 1.0 : 0.0,
      impl_->config.solve_offset,
      requested_escape_metric.norm() > 1e-12 ? 1.0 : 0.0,
      guarded_energy_residual,
      guarded_energy_violation,
      conditional_nonincrease_upper_bound,
      energy_params.initial_energy,
      base_config_profile_valid ? 1.0 : 0.0,
      base_domain_valid ? 1.0 : 0.0,
      base_metric_spd ? 1.0 : 0.0,
      base_solve_valid ? 1.0 : 0.0};
  }

  if (impl_->config.enable_leaf_ablation_diagnostics) {
    constexpr double active_tolerance = 1e-12;
    constexpr std::size_t header_size = 25;
    constexpr std::size_t record_size = 17;
    std::vector<double> ablation;
    ablation.reserve(
      header_size + record_size * impl_->ablation_group_ids.size());
    const double raw_norm = final_solve.raw_qdd.norm();
    const double command_norm = final_solve.command_qdd.norm();
    double direction_cosine = 1.0;
    if (raw_norm > active_tolerance && command_norm > active_tolerance) {
      direction_cosine = std::clamp(
        final_solve.raw_qdd.dot(final_solve.command_qdd) /
        (raw_norm * command_norm), -1.0, 1.0);
    }
    int saturated_joint_count = 0;
    for (int joint = 0; joint < 6; ++joint) {
      if (std::abs(final_solve.raw_qdd[joint]) >
        impl_->config.max_joint_accel + active_tolerance)
      {
        ++saturated_joint_count;
      }
    }
    ablation = {
      1.0,
      static_cast<double>(impl_->ablation_group_ids.size()),
      impl_->config.max_joint_accel,
      raw_norm,
      command_norm,
      direction_cosine,
      static_cast<double>(saturated_joint_count)};
    for (int joint = 0; joint < 6; ++joint) {
      ablation.push_back(final_solve.raw_qdd[joint]);
    }
    for (int joint = 0; joint < 6; ++joint) {
      ablation.push_back(final_solve.command_qdd[joint]);
    }
    for (int joint = 0; joint < 6; ++joint) {
      ablation.push_back(
        final_solve.command_qdd[joint] - final_solve.raw_qdd[joint]);
    }
    for (const int group_id : impl_->ablation_group_ids) {
      const auto contribution = leaf_contributions.find(group_id);
      if (contribution == leaf_contributions.end()) {
        throw std::runtime_error("Missing CasADi leaf-ablation contribution");
      }
      const Matrix6 without_metric = final_solve.metric - contribution->second.metric;
      const JointVector without_force = final_solve.force - contribution->second.force;
      const JointVector without_raw = resolve_root(
        without_metric, without_force, impl_->config.solve_offset);
      const JointVector without_command = limit_joint_acceleration(
        without_raw,
        impl_->config.max_joint_accel,
        impl_->config.preserve_joint_accel_direction);
      const JointVector delta_raw = final_solve.raw_qdd - without_raw;
      const JointVector delta_command = final_solve.command_qdd - without_command;
      const bool active =
        contribution->second.metric.norm() > active_tolerance ||
        contribution->second.force.norm() > active_tolerance;
      ablation.push_back(static_cast<double>(group_id));
      ablation.push_back(active ? 1.0 : 0.0);
      ablation.push_back(contribution->second.metric.trace());
      ablation.push_back(contribution->second.metric.norm());
      ablation.push_back(contribution->second.force.norm());
      for (int joint = 0; joint < 6; ++joint) {
        ablation.push_back(delta_raw[joint]);
      }
      for (int joint = 0; joint < 6; ++joint) {
        ablation.push_back(delta_command[joint]);
      }
    }
    // Extension: marker, count, then group id, row-major M[36], f[6].
    ablation.push_back(42042.0);
    ablation.push_back(static_cast<double>(impl_->ablation_group_ids.size()));
    for (const int group_id : impl_->ablation_group_ids) {
      const auto & item = leaf_contributions.at(group_id);
      ablation.push_back(group_id);
      for (int row=0; row<6; ++row) {
        for (int col=0; col<6; ++col) { ablation.push_back(item.metric(row,col)); }
      }
      for (int joint=0; joint<6; ++joint) { ablation.push_back(item.force[joint]); }
    }

    if (!context_valid) { context = impl_->model.forward_context(q, qd); context_valid = true; }

    ablation.push_back(44044.0);
    ablation.push_back(context.control_points.size());
    for (std::size_t i=0; i<context.control_points.size(); ++i) {
      for (int row=0; row<3; ++row) {
        for (int col=0; col<6; ++col) { ablation.push_back(context.control_point_jacobians[i](row,col)); }
        ablation.push_back(context.control_point_curvatures[i][row]);
      }
    }
    for (int row=0; row<3; ++row) {
      ablation.push_back(context.tcp_position[row]);
      for (int col=0; col<6; ++col) { ablation.push_back(context.tcp_jacobian(row,col)); }
      ablation.push_back(context.tcp_curvature[row]);
    }

    result.leaf_ablation_data = std::move(ablation);
  }
  return result;
}

std::vector<RmpSolveResult> CasadiRmp2Solver::solve_batch(
  const std::vector<RmpBatchInput> & batch_inputs) const
{
  const auto supervisor_snapshot = impl_->tangent_escape_supervisor;
  const auto energy_snapshot = impl_->escape_energy_certificate_state;
  const auto restore_state = [this, &supervisor_snapshot, &energy_snapshot]() {
      impl_->tangent_escape_supervisor = supervisor_snapshot;
      impl_->escape_energy_certificate_state = energy_snapshot;
    };

  std::vector<RmpSolveResult> results;
  results.reserve(batch_inputs.size());
  try {
    for (const auto & input : batch_inputs) {
      restore_state();
      results.push_back(solve(
          input.q,
          input.qd,
          input.vector_targets,
          input.obstacles,
          input.external_rmps));
    }
  } catch (...) {
    restore_state();
    throw;
  }
  restore_state();
  return results;
}

KinematicsContext CasadiRmp2Solver::forward_context(
  const JointVector & q,
  const JointVector & qd) const
{
  return impl_->model.forward_context(q, qd);
}

const std::vector<std::string> & CasadiRmp2Solver::target_keys() const
{
  return impl_->targets;
}

int CasadiRmp2Solver::max_obstacles() const
{
  return impl_->config.casadi_rmp2_max_obstacles;
}

bool CasadiRmp2Solver::jit_enabled() const
{
  return impl_->config.casadi_rmp2_jit;
}

std::string CasadiRmp2Solver::generate_c_code(const std::string & output_base_name) const
{
  if (output_base_name.empty()) {
    throw std::runtime_error("CasADi RMP2 code generation requires an output base name");
  }
  const std::filesystem::path output_path(output_base_name);
  const std::string file_name = output_path.filename().string();
  if (file_name.empty()) {
    throw std::runtime_error("CasADi RMP2 code generation requires a file name");
  }
  casadi::Dict options;
  options["with_header"] = true;
  casadi::CodeGenerator generator(file_name, options);
  generator.add(impl_->function);
  std::string prefix;
  if (output_path.has_parent_path()) {
    prefix = output_path.parent_path().string() +
      std::string(1, std::filesystem::path::preferred_separator);
  }
  return generator.generate(prefix);
}

}  // namespace rb10_rmpflow_rviz

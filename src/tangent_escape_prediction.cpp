#include "rb10_rmpflow_rviz/tangent_escape_prediction.hpp"

#include <algorithm>
#include <cmath>

namespace rb10_rmpflow_rviz
{
namespace tangent_escape_prediction
{

StaticObstaclePredictor::StaticObstaclePredictor(
  const KinematicsContext & context,
  std::size_t owner,
  const Eigen::Vector3d & e1,
  const Eigen::Vector3d & e2,
  const std::vector<StaticObstacle> & obstacles,
  double distance,
  double risk_length,
  double damping,
  double clearance_margin)
{
  const std::size_t count = std::min({
    context.control_points.size(), context.control_point_jacobians.size(),
    RB10Model::sensor_control_points.size()});
  if (owner >= count || obstacles.empty() ||
    !std::isfinite(distance) || distance <= 0.0 ||
    !std::isfinite(risk_length) || risk_length <= 0.0 ||
    !std::isfinite(damping) || damping <= 0.0 ||
    !std::isfinite(clearance_margin) || !e1.allFinite() || !e2.allFinite())
  {
    return;
  }
  distance_ = distance;
  risk_length_ = risk_length;
  const auto & owner_jacobian = context.control_point_jacobians[owner];
  if (!owner_jacobian.allFinite()) {
    return;
  }
  const Eigen::Matrix3d gram = owner_jacobian * owner_jacobian.transpose() +
    damping * damping * Eigen::Matrix3d::Identity();
  const auto decomposition = gram.ldlt();
  const RB10Model::JointVector motion_e1 =
    distance * owner_jacobian.transpose() * decomposition.solve(e1);
  const RB10Model::JointVector motion_e2 =
    distance * owner_jacobian.transpose() * decomposition.solve(e2);
  if (!motion_e1.allFinite() || !motion_e2.allFinite()) {
    return;
  }

  pairs_.reserve((count - 1) * obstacles.size());
  for (std::size_t k = 0; k < count; ++k) {
    const auto & point = context.control_points[k];
    const auto & jacobian = context.control_point_jacobians[k];
    if (k == owner || !point.position.allFinite() ||
      !std::isfinite(point.radius) || point.radius < 0.0 || !jacobian.allFinite())
    {
      continue;
    }
    const Eigen::Vector3d displacement_e1 = jacobian * motion_e1;
    const Eigen::Vector3d displacement_e2 = jacobian * motion_e2;
    for (const auto & obstacle : obstacles) {
      if (!obstacle.center.allFinite() ||
        !std::isfinite(obstacle.radius) || obstacle.radius <= 0.0)
      {
        continue;
      }
      const Eigen::Vector3d relative = point.position - obstacle.center;
      const double radius = point.radius + obstacle.radius + clearance_margin;
      pairs_.push_back({
        relative, displacement_e1, displacement_e2, radius,
        relative.norm() - radius, static_cast<int>(k), obstacle.source_sensor_index});
    }
  }
}

Risk StaticObstaclePredictor::evaluate(double theta) const
{
  Risk result;
  if (!std::isfinite(theta)) {
    return result;
  }
  const double cosine = std::cos(theta);
  const double sine = std::sin(theta);
  double largest_risk = 0.0;
  for (const auto & pair : pairs_) {
    const Eigen::Vector3d displacement =
      cosine * pair.displacement_e1 + sine * pair.displacement_e2;
    const Eigen::Vector3d derivative =
      -sine * pair.displacement_e1 + cosine * pair.displacement_e2;
    const double squared_length = displacement.squaredNorm();
    const double fraction = squared_length > 1e-18 ?
      std::clamp(-pair.relative.dot(displacement) / squared_length, 0.0, 1.0) : 0.0;
    const Eigen::Vector3d nearest = pair.relative + fraction * displacement;
    const double center_distance = nearest.norm();
    const double clearance = center_distance - pair.radius;
    result.min_clearance = std::min(result.min_clearance, clearance);
    ++result.pair_count;

    // Squaring the positive clearance loss makes the cost and its gradient
    // vanish at the transition between approaching and receding directions.
    const double loss = std::max(pair.clearance - clearance, 0.0) / distance_;
    if (loss <= 0.0) {
      continue;
    }
    ++result.approaching_pair_count;
    const double proximity = std::exp(-std::max(clearance, 0.0) / risk_length_);
    const double cost = proximity * loss * loss;
    result.value += cost;
    if (cost > largest_risk) {
      largest_risk = cost;
      result.critical_sensor_index = pair.sensor_index;
      result.obstacle_sensor_index = pair.obstacle_sensor_index;
    }
    // At an interior closest point, nearest.dot(displacement) == 0, so the
    // derivative of the minimizing fraction drops out (envelope theorem).
    // At either endpoint the fraction is constant. Contact at the sphere
    // center has no unique normal; use a finite zero subgradient there.
    const double clearance_derivative = center_distance > 1e-12 ?
      nearest.dot(fraction * derivative) / center_distance : 0.0;
    result.gradient -= proximity * (
      2.0 * loss / distance_ + (clearance > 0.0 ? loss * loss / risk_length_ : 0.0)) *
      clearance_derivative;
  }
  return result;
}

}  // namespace tangent_escape_prediction
}  // namespace rb10_rmpflow_rviz

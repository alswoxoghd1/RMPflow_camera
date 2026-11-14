#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "rb10_rmpflow_rviz/rb10_model.hpp"

namespace rb10_rmpflow_rviz
{
namespace tangent_escape_prediction
{

// A detected obstacle frozen in world coordinates for this prediction only.
// Its source is diagnostic: every other sensor point is tested against it,
// even if that sensor did not itself detect the obstacle.
struct StaticObstacle
{
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  double radius{0.0};
  int source_sensor_index{-1};
};

struct Risk
{
  double value{0.0};
  double gradient{0.0};              // d(value) / d(theta)
  double min_clearance{std::numeric_limits<double>::infinity()};
  int critical_sensor_index{-1};    // pair with the largest risk contribution
  int obstacle_sensor_index{-1};    // sensor that detected that obstacle
  std::size_t pair_count{0};
  std::size_t approaching_pair_count{0};
};

// Local kinematic prediction, not a prediction of the final RMP root solve:
//   dq(theta) = distance * Ji# * (cos(theta) e1 + sin(theta) e2)
//   dp_k      = Jk dq(theta).
// The obstacle is stationary and the minimum clearance over p_k + s dp_k,
// 0 <= s <= 1, is evaluated. Only decreases in clearance incur a penalty;
// retreat at one point cannot cancel another point's approach.
class StaticObstaclePredictor
{
public:
  StaticObstaclePredictor() = default;
  StaticObstaclePredictor(
    const KinematicsContext & context,
    std::size_t owner,
    const Eigen::Vector3d & e1,
    const Eigen::Vector3d & e2,
    const std::vector<StaticObstacle> & obstacles,
    double distance,
    double risk_length,
    double damping,
    double clearance_margin);

  Risk evaluate(double theta) const;

private:
  struct Pair
  {
    Eigen::Vector3d relative;
    Eigen::Vector3d displacement_e1;
    Eigen::Vector3d displacement_e2;
    double radius;
    double clearance;
    int sensor_index;
    int obstacle_sensor_index;
  };

  std::vector<Pair> pairs_;
  double distance_{0.02};
  double risk_length_{0.10};
};

}  // namespace tangent_escape_prediction
}  // namespace rb10_rmpflow_rviz

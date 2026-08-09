#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>
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

// Stateful selection and memory stay outside the differentiable RMP graph.
// Within each fixed mode, this returns one ordinary scalar tangent leaf.
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
  static constexpr std::size_t failure_memory_count_ = 4;

  enum class AccelerationModel : int
  {
    CanonicalVelocity = 0,
    RiskDamped = 1,
    TangentPlane = 2,
    TangentOde = 3
  };

  enum class CanonicalPhase : int
  {
    Off = 0,
    Engage = 1,
    Drive = 2,
    ReleaseDriveDown = 3,
    ReleaseBrake = 4,
    ReleaseLambdaDown = 5,
    ReselectDriveDown = 6,
    ReselectBrake = 7,
    ReselectLambdaDown = 8
  };

  struct FailureMemory
  {
    bool valid{false};
    Eigen::Vector3d tangent{Eigen::Vector3d::UnitX()};
    double strength{0.0};
  };

  struct ClearanceRateState
  {
    std::size_t control_point_index{0};
    std::int64_t obstacle_key{-1};
    double previous_clearance{0.0};
    double filtered_rate{0.0};
    bool previous_clearance_valid{false};
    bool seen_this_cycle{false};
  };

  struct CanonicalState
  {
    CanonicalPhase phase{CanonicalPhase::Off};
    bool active_pair_valid{false};
    std::size_t control_point_index{0};
    std::int64_t obstacle_key{-1};
    bool pending_pair_valid{false};
    std::size_t pending_control_point_index{0};
    std::int64_t pending_obstacle_key{-1};
    bool tangent_valid{false};
    Eigen::Vector3d tangent{Eigen::Vector3d::UnitX()};
    Eigen::Vector3d obstacle_direction_at_selection{Eigen::Vector3d::UnitX()};
    bool pending_failure_memory{false};
    bool force_direction_change{false};
    int handoff_reason{0};
    double current_score{-std::numeric_limits<double>::infinity()};
    double z{0.0};
    double lambda{0.0};
    double drive_ramp{0.0};
    double release_brake{0.0};
    double desired_velocity{0.0};
    double phase_elapsed_s{0.0};
    double phase_start_lambda{0.0};
    double phase_start_drive_ramp{0.0};
    double phase_start_release_brake{0.0};
    double active_age_s{0.0};
    double episode_age_s{0.0};
    double previous_goal_error{0.0};
    bool previous_goal_error_valid{false};
    Eigen::Vector3d previous_goal{Eigen::Vector3d::Zero()};
    bool previous_goal_valid{false};
    double filtered_goal_progress{0.0};
    double last_alpha_stuck{0.0};
    double last_raw_activation{0.0};
    double last_blockage{0.0};
    double last_clearance{0.0};
    double last_beta{0.0};
    double command_distance{0.0};
    double actual_distance{0.0};
    double episode_start_sector_risk{0.0};
    double last_sector_risk{0.0};
    std::array<FailureMemory, failure_memory_count_> failure_memory{};
    std::size_t failure_memory_cursor{0};
    JointVector previous_qdd{JointVector::Zero()};
    bool previous_qdd_valid{false};
    std::uint64_t handoff_generation{0};
    std::vector<ClearanceRateState> clearance_rate_states{};
  };

  void evaluate_ode(
    const KinematicsContext & context,
    const JointVector & qd,
    const Eigen::Vector3d & goal,
    const std::vector<ObstacleSphere> & obstacles,
    Matrix6 & metric,
    JointVector & force,
    TangentEscapeCommand & command,
    std::vector<double> * debug_data) const;

  void evaluate_impl(
    CanonicalState & state,
    AccelerationModel model,
    const JointVector & q,
    const KinematicsContext & context,
    const JointVector & qd,
    const Eigen::Vector3d & goal,
    const std::vector<ObstacleSphere> & obstacles,
    const JointVector & nominal_qdd,
    const Matrix6 & base_metric,
    const JointVector & base_force,
    Matrix6 & metric,
    JointVector & force,
    TangentEscapeCommand & command,
    std::vector<double> * debug_data) const;

  // Each acceleration model owns its hybrid state so that switching models
  // between runs never carries a latched pair, tangent, or failure memory
  // across the model boundary.
  static AccelerationModel parse_acceleration_model(const std::string & name);
  CanonicalState & state_for(AccelerationModel model) const;

  EigenRmpConfig config_;
  mutable CanonicalState canonical_state_{};
  mutable CanonicalState risk_damped_state_{};
  mutable CanonicalState tangent_plane_state_{};
  mutable tangent_escape_ode::LeafState ode_state_{};
};

}  // namespace rb10_rmpflow_rviz

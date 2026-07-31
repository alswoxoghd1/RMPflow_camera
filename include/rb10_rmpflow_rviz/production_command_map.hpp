#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

#include <Eigen/Dense>

namespace rb10_rmpflow_rviz
{

using ProductionJointVector = Eigen::Matrix<double, 6, 1>;
using ProductionStateVector = Eigen::Matrix<double, 12, 1>;
using ProductionStateMatrix = Eigen::Matrix<double, 12, 12>;

enum class ProductionCommandMode
{
  kPosition,
  kVelocity
};

struct ProductionJointState
{
  ProductionJointVector q{ProductionJointVector::Zero()};
  ProductionJointVector qd{ProductionJointVector::Zero()};
};

struct ProductionCommandMapConfig
{
  double dt{0.005};
  double max_joint_accel{10.0};
  bool preserve_joint_accel_direction{true};
  ProductionCommandMode command_mode{ProductionCommandMode::kVelocity};
  double command_guard_max_step_rad{0.00436332313};
  double command_guard_max_velocity_rad_s{1.0};
  bool command_guard_velocity_slew_enabled{false};
  double command_guard_max_accel_rad_s2{10.0};
  bool predictive_joint_limit_guard{true};
  std::array<double, 6> joint_lower_limits{};
  std::array<double, 6> joint_upper_limits{};
  std::array<double, 6> joint_limit_buffers{};
  double near_limit_damping_region_rad{0.1};
  double near_limit_velocity_scale{0.5};
};

struct ProductionCommandMemory
{
  ProductionJointVector previous_velocity_command{ProductionJointVector::Zero()};
  bool previous_velocity_command_initialized{false};
};

struct ProductionCommandDiagnostics
{
  bool finite{true};
  bool acceleration_limited{false};
  bool integration_position_clamped{false};
  bool near_limit_velocity_damped{false};
  bool command_guard_clamped{false};
  bool velocity_slew_clamped{false};
  bool predictive_joint_limit_clamped{false};
  ProductionJointVector raw_qdd{ProductionJointVector::Zero()};
  ProductionJointVector applied_qdd{ProductionJointVector::Zero()};
  ProductionJointVector pre_guard_command{ProductionJointVector::Zero()};
  ProductionJointVector command_step{ProductionJointVector::Zero()};

  bool any_nonsmooth_branch_active() const
  {
    return
      acceleration_limited ||
      integration_position_clamped ||
      near_limit_velocity_damped ||
      command_guard_clamped;
  }
};

struct ProductionCommandStep
{
  ProductionJointState state;
  ProductionCommandMemory memory;
  ProductionCommandDiagnostics diagnostics;
};

ProductionStateVector pack_production_state(const ProductionJointState & state);

ProductionJointState unpack_production_state(const ProductionStateVector & state);

ProductionJointState integrate_production_command(
  const ProductionJointState & state,
  const ProductionJointVector & applied_qdd,
  const ProductionCommandMapConfig & config,
  ProductionCommandDiagnostics * diagnostics = nullptr);

void apply_production_command_guard(
  const ProductionJointState & measured_state,
  ProductionJointState & command_state,
  const ProductionCommandMapConfig & config,
  ProductionCommandMemory & memory,
  ProductionCommandDiagnostics * diagnostics = nullptr);

ProductionCommandStep production_command_step(
  const ProductionJointState & measured_state,
  const ProductionJointVector & raw_qdd,
  const ProductionCommandMapConfig & config,
  const ProductionCommandMemory & memory);

struct EquilibriumSearchOptions
{
  int maximum_iterations{80};
  double acceleration_tolerance{1e-8};
  double finite_difference_step{1e-5};
  double maximum_newton_step_norm{0.25};
  int line_search_steps{14};
  double joint_limit_margin{1e-4};
};

struct EquilibriumSearchResult
{
  bool converged{false};
  int iterations{0};
  ProductionJointVector q{ProductionJointVector::Zero()};
  ProductionJointVector acceleration{ProductionJointVector::Zero()};
  double acceleration_norm{std::numeric_limits<double>::infinity()};
};

using StaticAccelerationFunction =
  std::function<ProductionJointVector(const ProductionJointVector &)>;

EquilibriumSearchResult find_production_equilibrium(
  const StaticAccelerationFunction & acceleration,
  const ProductionJointVector & initial_q,
  const std::array<double, 6> & joint_lower_limits,
  const std::array<double, 6> & joint_upper_limits,
  const EquilibriumSearchOptions & options = {});

struct DiscreteLyapunovResult
{
  bool linearization_finite{false};
  bool schur_stable{false};
  bool lyapunov_solved{false};
  bool lyapunov_positive_definite{false};
  bool lyapunov_decrease_strict{false};
  bool equilibrium_nonsmooth_branch_active{false};
  bool perturbation_nonsmooth_branch_active{false};
  ProductionStateMatrix jacobian{ProductionStateMatrix::Zero()};
  ProductionStateMatrix lyapunov_matrix{ProductionStateMatrix::Zero()};
  ProductionStateMatrix lyapunov_residual{ProductionStateMatrix::Zero()};
  std::vector<Eigen::Vector2d> eigenvalues;
  double spectral_radius{std::numeric_limits<double>::infinity()};
  double equilibrium_map_residual_norm{std::numeric_limits<double>::infinity()};
  double lyapunov_min_eigenvalue{-std::numeric_limits<double>::infinity()};
  double lyapunov_max_eigenvalue{std::numeric_limits<double>::infinity()};
  double decrease_max_eigenvalue{std::numeric_limits<double>::infinity()};
  double residual_norm{std::numeric_limits<double>::infinity()};
};

using ProductionStepFunction =
  std::function<ProductionCommandStep(const ProductionJointState &)>;

DiscreteLyapunovResult analyze_production_local_stability(
  const ProductionStepFunction & step,
  const ProductionJointState & equilibrium,
  double position_difference_step = 1e-5,
  double velocity_difference_step = 1e-5);

struct NonlinearLyapunovSample
{
  double radius{0.0};
  int direction_index{0};
  double state_norm{0.0};
  double value{0.0};
  double next_value{0.0};
  double delta_value{0.0};
  double relative_delta{0.0};
  bool nonsmooth_branch_active{false};
};

std::vector<NonlinearLyapunovSample> sample_production_lyapunov_region(
  const ProductionStepFunction & step,
  const ProductionJointState & equilibrium,
  const ProductionStateMatrix & lyapunov_matrix,
  const std::vector<double> & radii,
  int random_directions_per_radius = 64,
  unsigned int random_seed = 20260728U);

}  // namespace rb10_rmpflow_rviz

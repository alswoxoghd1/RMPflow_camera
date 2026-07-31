#include "rb10_rmpflow_rviz/production_command_map.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#include <Eigen/Eigenvalues>

#include "rb10_rmpflow_rviz/joint_acceleration_limiter.hpp"

namespace rb10_rmpflow_rviz
{
namespace
{

ProductionJointVector clamp_joint_positions(
  const ProductionJointVector & q,
  const ProductionCommandMapConfig & config,
  bool * clamped)
{
  ProductionJointVector result = q;
  for (Eigen::Index index = 0; index < result.size(); ++index) {
    const std::size_t joint = static_cast<std::size_t>(index);
    const double limited = std::clamp(
      result[index],
      config.joint_lower_limits[joint],
      config.joint_upper_limits[joint]);
    if (clamped != nullptr && std::abs(limited - result[index]) > 1e-12) {
      *clamped = true;
    }
    result[index] = limited;
  }
  return result;
}

ProductionJointVector clamp_equilibrium_position(
  const ProductionJointVector & q,
  const std::array<double, 6> & lower,
  const std::array<double, 6> & upper,
  double margin)
{
  ProductionJointVector result = q;
  for (Eigen::Index index = 0; index < result.size(); ++index) {
    const std::size_t joint = static_cast<std::size_t>(index);
    const double available_half_width =
      0.5 * std::max(upper[joint] - lower[joint], 0.0);
    const double safe_margin = std::min(std::max(margin, 0.0), available_half_width);
    result[index] = std::clamp(
      result[index],
      lower[joint] + safe_margin,
      upper[joint] - safe_margin);
  }
  return result;
}

bool finite_state(const ProductionJointState & state)
{
  return state.q.allFinite() && state.qd.allFinite();
}

}  // namespace

ProductionStateVector pack_production_state(const ProductionJointState & state)
{
  ProductionStateVector packed;
  packed.head<6>() = state.q;
  packed.tail<6>() = state.qd;
  return packed;
}

ProductionJointState unpack_production_state(const ProductionStateVector & state)
{
  ProductionJointState unpacked;
  unpacked.q = state.head<6>();
  unpacked.qd = state.tail<6>();
  return unpacked;
}

ProductionJointState integrate_production_command(
  const ProductionJointState & state,
  const ProductionJointVector & applied_qdd,
  const ProductionCommandMapConfig & config,
  ProductionCommandDiagnostics * diagnostics)
{
  ProductionJointState command_state = state;
  command_state.qd += applied_qdd * config.dt;
  command_state.q += command_state.qd * config.dt;

  bool position_clamped = false;
  command_state.q = clamp_joint_positions(command_state.q, config, &position_clamped);
  if (diagnostics != nullptr) {
    diagnostics->integration_position_clamped |= position_clamped;
  }

  for (Eigen::Index index = 0; index < command_state.q.size(); ++index) {
    const std::size_t joint = static_cast<std::size_t>(index);
    const bool moving_below_lower_region =
      command_state.q[index] <=
      config.joint_lower_limits[joint] + config.near_limit_damping_region_rad &&
      command_state.qd[index] < 0.0;
    const bool moving_above_upper_region =
      command_state.q[index] >=
      config.joint_upper_limits[joint] - config.near_limit_damping_region_rad &&
      command_state.qd[index] > 0.0;
    if (moving_below_lower_region || moving_above_upper_region) {
      command_state.qd[index] *= config.near_limit_velocity_scale;
      if (diagnostics != nullptr) {
        diagnostics->near_limit_velocity_damped = true;
      }
    }
  }

  if (diagnostics != nullptr) {
    diagnostics->finite &= finite_state(command_state);
  }
  return command_state;
}

void apply_production_command_guard(
  const ProductionJointState & measured_state,
  ProductionJointState & command_state,
  const ProductionCommandMapConfig & config,
  ProductionCommandMemory & memory,
  ProductionCommandDiagnostics * diagnostics)
{
  ProductionCommandDiagnostics local;
  ProductionCommandDiagnostics & output = diagnostics != nullptr ? *diagnostics : local;
  output.pre_guard_command =
    config.command_mode == ProductionCommandMode::kVelocity ?
    command_state.qd :
    command_state.q;

  if (config.command_mode == ProductionCommandMode::kVelocity) {
    const double safe_dt = std::max(config.dt, 1e-6);
    if (!memory.previous_velocity_command_initialized) {
      memory.previous_velocity_command.setZero();
      memory.previous_velocity_command_initialized = true;
    }
    const ProductionJointVector previous_command = memory.previous_velocity_command;
    const double maximum_velocity_step =
      config.command_guard_velocity_slew_enabled ?
      config.command_guard_max_accel_rad_s2 * safe_dt :
      std::numeric_limits<double>::infinity();

    for (Eigen::Index index = 0; index < command_state.q.size(); ++index) {
      const std::size_t joint = static_cast<std::size_t>(index);
      const double requested_velocity = command_state.qd[index];
      double limited_velocity = std::clamp(
        requested_velocity,
        -config.command_guard_max_velocity_rad_s,
        config.command_guard_max_velocity_rad_s);
      if (std::abs(limited_velocity - requested_velocity) > 1e-9) {
        output.command_guard_clamped = true;
      }

      if (config.command_guard_velocity_slew_enabled) {
        const double slew_limited_velocity = std::clamp(
          limited_velocity,
          previous_command[index] - maximum_velocity_step,
          previous_command[index] + maximum_velocity_step);
        if (std::abs(slew_limited_velocity - limited_velocity) > 1e-9) {
          output.command_guard_clamped = true;
          output.velocity_slew_clamped = true;
        }
        limited_velocity = slew_limited_velocity;
      }

      double guarded_lower = config.joint_lower_limits[joint];
      double guarded_upper = config.joint_upper_limits[joint];
      if (config.predictive_joint_limit_guard) {
        const double raw_lower = config.joint_lower_limits[joint];
        const double raw_upper = config.joint_upper_limits[joint];
        const double maximum_buffer = std::max(0.0, 0.45 * (raw_upper - raw_lower));
        const double buffer = std::min(
          std::max(config.joint_limit_buffers[joint], 0.0),
          maximum_buffer);
        guarded_lower = raw_lower + buffer;
        guarded_upper = raw_upper - buffer;
        const double velocity_before_joint_guard = limited_velocity;
        if (measured_state.q[index] < guarded_lower) {
          limited_velocity = std::max(limited_velocity, 0.0);
        } else if (measured_state.q[index] > guarded_upper) {
          limited_velocity = std::min(limited_velocity, 0.0);
        } else {
          const double lower_velocity =
            (guarded_lower - measured_state.q[index]) / safe_dt;
          const double upper_velocity =
            (guarded_upper - measured_state.q[index]) / safe_dt;
          limited_velocity = std::clamp(
            limited_velocity,
            lower_velocity,
            upper_velocity);
        }
        if (std::abs(limited_velocity - velocity_before_joint_guard) > 1e-9) {
          output.command_guard_clamped = true;
          output.predictive_joint_limit_clamped = true;
        }
      } else {
        const bool moving_out_below =
          measured_state.q[index] <=
          config.joint_lower_limits[joint] + config.near_limit_damping_region_rad &&
          limited_velocity < 0.0;
        const bool moving_out_above =
          measured_state.q[index] >=
          config.joint_upper_limits[joint] - config.near_limit_damping_region_rad &&
          limited_velocity > 0.0;
        if (moving_out_below || moving_out_above) {
          limited_velocity = 0.0;
          output.command_guard_clamped = true;
        }
      }

      if (std::abs(limited_velocity - requested_velocity) > 1e-9) {
        output.command_guard_clamped = true;
      }
      command_state.qd[index] = limited_velocity;
      command_state.q[index] =
        measured_state.q[index] + command_state.qd[index] * config.dt;
      if (config.predictive_joint_limit_guard) {
        const double unclamped_position = command_state.q[index];
        command_state.q[index] = std::clamp(
          command_state.q[index],
          guarded_lower,
          guarded_upper);
        if (std::abs(command_state.q[index] - unclamped_position) > 1e-9) {
          output.command_guard_clamped = true;
          output.predictive_joint_limit_clamped = true;
        }
      }
    }

    bool raw_limit_clamped = false;
    command_state.q =
      clamp_joint_positions(command_state.q, config, &raw_limit_clamped);
    output.command_guard_clamped |= raw_limit_clamped;
    output.command_step = command_state.qd - previous_command;
    memory.previous_velocity_command = command_state.qd;
    output.finite &= finite_state(command_state);
    return;
  }

  const double maximum_step =
    std::max(
    config.command_guard_max_step_rad,
    config.command_guard_max_velocity_rad_s * config.dt);
  for (Eigen::Index index = 0; index < command_state.q.size(); ++index) {
    const double delta = command_state.q[index] - measured_state.q[index];
    const double limited_delta = std::clamp(delta, -maximum_step, maximum_step);
    if (std::abs(limited_delta - delta) > 1e-9) {
      output.command_guard_clamped = true;
    }
    command_state.q[index] = measured_state.q[index] + limited_delta;

    const double limited_velocity = std::clamp(
      command_state.qd[index],
      -config.command_guard_max_velocity_rad_s,
      config.command_guard_max_velocity_rad_s);
    if (std::abs(limited_velocity - command_state.qd[index]) > 1e-9) {
      output.command_guard_clamped = true;
    }
    command_state.qd[index] = limited_velocity;
  }

  bool position_clamped = false;
  command_state.q = clamp_joint_positions(command_state.q, config, &position_clamped);
  output.command_guard_clamped |= position_clamped;
  output.command_step = command_state.q - measured_state.q;
  output.finite &= finite_state(command_state);
}

ProductionCommandStep production_command_step(
  const ProductionJointState & measured_state,
  const ProductionJointVector & raw_qdd,
  const ProductionCommandMapConfig & config,
  const ProductionCommandMemory & memory)
{
  ProductionCommandStep result;
  result.memory = memory;
  result.diagnostics.raw_qdd = raw_qdd;
  result.diagnostics.applied_qdd = limit_joint_acceleration(
    raw_qdd,
    config.max_joint_accel,
    config.preserve_joint_accel_direction);
  result.diagnostics.acceleration_limited =
    (result.diagnostics.applied_qdd - raw_qdd).cwiseAbs().maxCoeff() > 1e-9;
  result.diagnostics.finite =
    measured_state.q.allFinite() &&
    measured_state.qd.allFinite() &&
    raw_qdd.allFinite() &&
    result.diagnostics.applied_qdd.allFinite();

  result.state = integrate_production_command(
    measured_state,
    result.diagnostics.applied_qdd,
    config,
    &result.diagnostics);
  apply_production_command_guard(
    measured_state,
    result.state,
    config,
    result.memory,
    &result.diagnostics);
  return result;
}

EquilibriumSearchResult find_production_equilibrium(
  const StaticAccelerationFunction & acceleration,
  const ProductionJointVector & initial_q,
  const std::array<double, 6> & joint_lower_limits,
  const std::array<double, 6> & joint_upper_limits,
  const EquilibriumSearchOptions & options)
{
  if (!acceleration) {
    throw std::invalid_argument("Equilibrium acceleration callback is empty");
  }

  EquilibriumSearchResult result;
  result.q = clamp_equilibrium_position(
    initial_q,
    joint_lower_limits,
    joint_upper_limits,
    options.joint_limit_margin);

  const double difference_step =
    std::max(options.finite_difference_step, 1e-9);
  for (int iteration = 0; iteration < std::max(options.maximum_iterations, 1); ++iteration) {
    result.iterations = iteration + 1;
    result.acceleration = acceleration(result.q);
    result.acceleration_norm = result.acceleration.norm();
    if (!result.acceleration.allFinite()) {
      return result;
    }
    if (result.acceleration_norm <= std::max(options.acceleration_tolerance, 0.0)) {
      result.converged = true;
      return result;
    }

    Eigen::Matrix<double, 6, 6> jacobian;
    for (int column = 0; column < 6; ++column) {
      ProductionJointVector plus = result.q;
      ProductionJointVector minus = result.q;
      plus[column] += difference_step;
      minus[column] -= difference_step;
      plus = clamp_equilibrium_position(
        plus,
        joint_lower_limits,
        joint_upper_limits,
        options.joint_limit_margin);
      minus = clamp_equilibrium_position(
        minus,
        joint_lower_limits,
        joint_upper_limits,
        options.joint_limit_margin);
      const double denominator = plus[column] - minus[column];
      if (std::abs(denominator) <= 1e-12) {
        jacobian.col(column).setZero();
      } else {
        jacobian.col(column) =
          (acceleration(plus) - acceleration(minus)) / denominator;
      }
    }
    if (!jacobian.allFinite()) {
      return result;
    }

    ProductionJointVector newton_step =
      jacobian.completeOrthogonalDecomposition().solve(-result.acceleration);
    if (!newton_step.allFinite()) {
      return result;
    }
    const double maximum_step_norm = std::max(options.maximum_newton_step_norm, 1e-6);
    if (newton_step.norm() > maximum_step_norm) {
      newton_step *= maximum_step_norm / newton_step.norm();
    }

    bool accepted = false;
    double scale = 1.0;
    for (int line_search = 0;
      line_search < std::max(options.line_search_steps, 1);
      ++line_search)
    {
      const ProductionJointVector trial = clamp_equilibrium_position(
        result.q + scale * newton_step,
        joint_lower_limits,
        joint_upper_limits,
        options.joint_limit_margin);
      const ProductionJointVector trial_acceleration = acceleration(trial);
      if (
        trial_acceleration.allFinite() &&
        trial_acceleration.norm() < result.acceleration_norm)
      {
        result.q = trial;
        accepted = true;
        break;
      }
      scale *= 0.5;
    }

    if (!accepted) {
      // A small acceleration descent step gives the search a deterministic
      // fallback when the local qdd Jacobian is rank deficient.
      const ProductionJointVector trial = clamp_equilibrium_position(
        result.q + 1e-3 * result.acceleration,
        joint_lower_limits,
        joint_upper_limits,
        options.joint_limit_margin);
      const ProductionJointVector trial_acceleration = acceleration(trial);
      if (
        !trial_acceleration.allFinite() ||
        trial_acceleration.norm() >= result.acceleration_norm)
      {
        return result;
      }
      result.q = trial;
    }
  }

  result.acceleration = acceleration(result.q);
  result.acceleration_norm = result.acceleration.norm();
  result.converged =
    result.acceleration.allFinite() &&
    result.acceleration_norm <= std::max(options.acceleration_tolerance, 0.0);
  return result;
}

DiscreteLyapunovResult analyze_production_local_stability(
  const ProductionStepFunction & step,
  const ProductionJointState & equilibrium,
  double position_difference_step,
  double velocity_difference_step)
{
  if (!step) {
    throw std::invalid_argument("Production step callback is empty");
  }

  DiscreteLyapunovResult result;
  const ProductionStateVector equilibrium_vector = pack_production_state(equilibrium);
  const ProductionCommandStep equilibrium_step = step(equilibrium);
  result.equilibrium_map_residual_norm =
    (pack_production_state(equilibrium_step.state) - equilibrium_vector).norm();
  result.equilibrium_nonsmooth_branch_active =
    equilibrium_step.diagnostics.any_nonsmooth_branch_active();

  for (int column = 0; column < 12; ++column) {
    const double difference_step = std::max(
      column < 6 ? position_difference_step : velocity_difference_step,
      1e-9);
    ProductionStateVector plus_vector = equilibrium_vector;
    ProductionStateVector minus_vector = equilibrium_vector;
    plus_vector[column] += difference_step;
    minus_vector[column] -= difference_step;
    const ProductionCommandStep plus = step(unpack_production_state(plus_vector));
    const ProductionCommandStep minus = step(unpack_production_state(minus_vector));
    result.perturbation_nonsmooth_branch_active |=
      plus.diagnostics.any_nonsmooth_branch_active() ||
      minus.diagnostics.any_nonsmooth_branch_active();
    result.jacobian.col(column) =
      (pack_production_state(plus.state) - pack_production_state(minus.state)) /
      (2.0 * difference_step);
  }
  result.linearization_finite =
    result.jacobian.allFinite() &&
    std::isfinite(result.equilibrium_map_residual_norm);
  if (!result.linearization_finite) {
    return result;
  }

  Eigen::EigenSolver<ProductionStateMatrix> eigen_solver(
    result.jacobian,
    false);
  if (eigen_solver.info() != Eigen::Success) {
    return result;
  }
  result.eigenvalues.reserve(12);
  result.spectral_radius = 0.0;
  for (int index = 0; index < eigen_solver.eigenvalues().size(); ++index) {
    const auto value = eigen_solver.eigenvalues()[index];
    result.eigenvalues.emplace_back(value.real(), value.imag());
    result.spectral_radius = std::max(result.spectral_radius, std::abs(value));
  }
  result.schur_stable = result.spectral_radius < 1.0 - 1e-9;
  if (!result.schur_stable) {
    return result;
  }

  constexpr int state_dimension = 12;
  constexpr int squared_dimension = state_dimension * state_dimension;
  Eigen::MatrixXd equation =
    Eigen::MatrixXd::Identity(squared_dimension, squared_dimension);
  // Eigen stores matrices column-major, so vec(P)[i + j*n] corresponds to
  // P(i,j). For P - A^T P A = I, the coefficient of P(k,l) in equation
  // (i,j) is delta(i,k)delta(j,l) - A(k,i)A(l,j).
  for (int i = 0; i < state_dimension; ++i) {
    for (int j = 0; j < state_dimension; ++j) {
      const int equation_row = i + j * state_dimension;
      for (int k = 0; k < state_dimension; ++k) {
        for (int l = 0; l < state_dimension; ++l) {
          const int unknown_column = k + l * state_dimension;
          equation(equation_row, unknown_column) -=
            result.jacobian(k, i) * result.jacobian(l, j);
        }
      }
    }
  }

  const ProductionStateMatrix identity = ProductionStateMatrix::Identity();
  const Eigen::VectorXd right_hand_side =
    Eigen::Map<const Eigen::VectorXd>(identity.data(), squared_dimension);
  const Eigen::FullPivLU<Eigen::MatrixXd> decomposition(equation);
  if (!decomposition.isInvertible()) {
    return result;
  }
  const Eigen::VectorXd vectorized = decomposition.solve(right_hand_side);
  result.lyapunov_matrix =
    Eigen::Map<const ProductionStateMatrix>(vectorized.data());
  result.lyapunov_matrix =
    0.5 * (result.lyapunov_matrix + result.lyapunov_matrix.transpose());
  result.lyapunov_solved = result.lyapunov_matrix.allFinite();
  if (!result.lyapunov_solved) {
    return result;
  }

  result.lyapunov_residual =
    result.jacobian.transpose() * result.lyapunov_matrix * result.jacobian -
    result.lyapunov_matrix +
    ProductionStateMatrix::Identity();
  result.residual_norm = result.lyapunov_residual.norm();

  Eigen::SelfAdjointEigenSolver<ProductionStateMatrix> p_eigen_solver(
    result.lyapunov_matrix);
  const ProductionStateMatrix decrease =
    result.jacobian.transpose() * result.lyapunov_matrix * result.jacobian -
    result.lyapunov_matrix;
  Eigen::SelfAdjointEigenSolver<ProductionStateMatrix> decrease_eigen_solver(decrease);
  if (
    p_eigen_solver.info() != Eigen::Success ||
    decrease_eigen_solver.info() != Eigen::Success)
  {
    return result;
  }

  result.lyapunov_min_eigenvalue = p_eigen_solver.eigenvalues().minCoeff();
  result.lyapunov_max_eigenvalue = p_eigen_solver.eigenvalues().maxCoeff();
  result.decrease_max_eigenvalue =
    decrease_eigen_solver.eigenvalues().maxCoeff();
  result.lyapunov_positive_definite = result.lyapunov_min_eigenvalue > 1e-10;
  result.lyapunov_decrease_strict = result.decrease_max_eigenvalue < -1e-8;
  return result;
}

std::vector<NonlinearLyapunovSample> sample_production_lyapunov_region(
  const ProductionStepFunction & step,
  const ProductionJointState & equilibrium,
  const ProductionStateMatrix & lyapunov_matrix,
  const std::vector<double> & radii,
  int random_directions_per_radius,
  unsigned int random_seed)
{
  if (!step) {
    throw std::invalid_argument("Production step callback is empty");
  }

  std::vector<ProductionStateVector> directions;
  directions.reserve(
    24 + static_cast<std::size_t>(std::max(random_directions_per_radius, 0)));
  for (int index = 0; index < 12; ++index) {
    ProductionStateVector positive = ProductionStateVector::Zero();
    positive[index] = 1.0;
    directions.push_back(positive);
    directions.push_back(-positive);
  }

  std::mt19937 generator(random_seed);
  std::normal_distribution<double> normal_distribution(0.0, 1.0);
  for (int sample = 0; sample < std::max(random_directions_per_radius, 0); ++sample) {
    ProductionStateVector direction;
    for (int index = 0; index < direction.size(); ++index) {
      direction[index] = normal_distribution(generator);
    }
    if (direction.norm() > 1e-12) {
      directions.push_back(direction.normalized());
    }
  }

  const ProductionStateVector equilibrium_vector = pack_production_state(equilibrium);
  std::vector<NonlinearLyapunovSample> samples;
  samples.reserve(radii.size() * directions.size());
  for (double radius : radii) {
    if (!std::isfinite(radius) || radius <= 0.0) {
      continue;
    }
    for (std::size_t direction_index = 0;
      direction_index < directions.size();
      ++direction_index)
    {
      const ProductionStateVector error = radius * directions[direction_index];
      const ProductionJointState state =
        unpack_production_state(equilibrium_vector + error);
      const ProductionCommandStep next = step(state);
      const ProductionStateVector next_error =
        pack_production_state(next.state) - equilibrium_vector;
      NonlinearLyapunovSample sample;
      sample.radius = radius;
      sample.direction_index = static_cast<int>(direction_index);
      sample.state_norm = error.norm();
      sample.value = error.dot(lyapunov_matrix * error);
      sample.next_value = next_error.dot(lyapunov_matrix * next_error);
      sample.delta_value = sample.next_value - sample.value;
      sample.relative_delta =
        sample.delta_value / std::max(sample.value, 1e-18);
      sample.nonsmooth_branch_active =
        next.diagnostics.any_nonsmooth_branch_active();
      samples.push_back(sample);
    }
  }
  return samples;
}

}  // namespace rb10_rmpflow_rviz

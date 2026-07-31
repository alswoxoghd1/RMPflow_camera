#include <array>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "rb10_rmpflow_rviz/production_command_map.hpp"

namespace rb10_rmpflow_rviz
{
namespace
{

ProductionCommandMapConfig base_config()
{
  ProductionCommandMapConfig config;
  config.dt = 0.005;
  config.max_joint_accel = 10.0;
  config.preserve_joint_accel_direction = true;
  config.command_mode = ProductionCommandMode::kVelocity;
  config.command_guard_max_velocity_rad_s = 1.0;
  config.command_guard_velocity_slew_enabled = false;
  config.command_guard_max_accel_rad_s2 = 10.0;
  config.predictive_joint_limit_guard = true;
  config.joint_lower_limits.fill(-2.0);
  config.joint_upper_limits.fill(2.0);
  config.joint_limit_buffers.fill(0.01);
  return config;
}

TEST(ProductionCommandMap, MatchesVelocityModeSemiImplicitUpdate)
{
  const auto config = base_config();
  ProductionJointState state;
  state.q.setConstant(0.2);
  state.qd.setConstant(0.1);
  ProductionJointVector qdd;
  qdd << 1.0, -2.0, 3.0, -4.0, 5.0, -6.0;

  ProductionCommandMemory memory;
  memory.previous_velocity_command = state.qd;
  memory.previous_velocity_command_initialized = true;
  const auto result = production_command_step(state, qdd, config, memory);

  const ProductionJointVector expected_qd = state.qd + qdd * config.dt;
  const ProductionJointVector expected_q = state.q + expected_qd * config.dt;
  EXPECT_TRUE(result.state.qd.isApprox(expected_qd, 1e-14));
  EXPECT_TRUE(result.state.q.isApprox(expected_q, 1e-14));
  EXPECT_FALSE(result.diagnostics.any_nonsmooth_branch_active());
  EXPECT_TRUE(result.diagnostics.finite);
}

TEST(ProductionCommandMap, ReportsEveryActiveLimit)
{
  auto config = base_config();
  config.max_joint_accel = 1.0;
  config.command_guard_max_velocity_rad_s = 0.05;
  config.command_guard_velocity_slew_enabled = true;
  config.command_guard_max_accel_rad_s2 = 0.1;

  ProductionJointState state;
  state.q.setZero();
  state.qd.setZero();
  const ProductionJointVector qdd = ProductionJointVector::Constant(100.0);
  const auto result = production_command_step(
    state,
    qdd,
    config,
    ProductionCommandMemory{});

  EXPECT_TRUE(result.diagnostics.acceleration_limited);
  EXPECT_TRUE(result.diagnostics.command_guard_clamped);
  EXPECT_TRUE(result.diagnostics.velocity_slew_clamped);
  EXPECT_LE(result.state.qd.cwiseAbs().maxCoeff(), 0.1 * config.dt + 1e-14);
}

TEST(ProductionCommandMap, FindsStaticEquilibrium)
{
  const ProductionJointVector expected =
    (ProductionJointVector() << 0.2, -0.3, 0.4, -0.5, 0.6, -0.7).finished();
  const auto acceleration = [expected](const ProductionJointVector & q) {
      return -4.0 * (q - expected);
    };
  const std::array<double, 6> lower{-2.0, -2.0, -2.0, -2.0, -2.0, -2.0};
  const std::array<double, 6> upper{2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

  const auto result = find_production_equilibrium(
    acceleration,
    ProductionJointVector::Zero(),
    lower,
    upper);
  EXPECT_TRUE(result.converged);
  EXPECT_LT(result.acceleration_norm, 1e-8);
  EXPECT_TRUE(result.q.isApprox(expected, 1e-8));
}

TEST(ProductionCommandMap, CertifiesStableDiscreteClosedLoop)
{
  const auto config = base_config();
  ProductionJointState equilibrium;
  equilibrium.q.setZero();
  equilibrium.qd.setZero();

  const auto step = [config](const ProductionJointState & state) {
      const ProductionJointVector qdd = -20.0 * state.q - 10.0 * state.qd;
      ProductionCommandMemory memory;
      memory.previous_velocity_command = state.qd;
      memory.previous_velocity_command_initialized = true;
      return production_command_step(state, qdd, config, memory);
    };

  const auto certificate = analyze_production_local_stability(
    step,
    equilibrium,
    1e-6,
    1e-6);
  EXPECT_TRUE(certificate.linearization_finite);
  EXPECT_TRUE(certificate.schur_stable);
  EXPECT_TRUE(certificate.lyapunov_solved);
  EXPECT_TRUE(certificate.lyapunov_positive_definite);
  EXPECT_TRUE(certificate.lyapunov_decrease_strict);
  EXPECT_FALSE(certificate.equilibrium_nonsmooth_branch_active);
  EXPECT_FALSE(certificate.perturbation_nonsmooth_branch_active);
  EXPECT_LT(certificate.spectral_radius, 1.0);
  EXPECT_LT(certificate.residual_norm, 1e-7);

  const auto samples = sample_production_lyapunov_region(
    step,
    equilibrium,
    certificate.lyapunov_matrix,
    {1e-4, 1e-3, 1e-2},
    16);
  ASSERT_FALSE(samples.empty());
  for (const auto & sample : samples) {
    EXPECT_LT(sample.delta_value, 0.0);
    EXPECT_FALSE(sample.nonsmooth_branch_active);
  }
}

TEST(ProductionCommandMap, VelocityCommandBoxIsForwardInvariant)
{
  auto config = base_config();
  config.max_joint_accel = 10.0;
  config.command_guard_max_velocity_rad_s = 1.0;
  config.command_guard_velocity_slew_enabled = false;
  config.predictive_joint_limit_guard = true;

  std::mt19937 generator(20260728U);
  std::uniform_real_distribution<double> position(-3.0, 3.0);
  std::uniform_real_distribution<double> velocity(-20.0, 20.0);
  std::uniform_real_distribution<double> acceleration(-1000.0, 1000.0);
  for (int sample = 0; sample < 10000; ++sample) {
    ProductionJointState state;
    ProductionJointVector qdd;
    for (int joint = 0; joint < 6; ++joint) {
      state.q[joint] = position(generator);
      state.qd[joint] = velocity(generator);
      qdd[joint] = acceleration(generator);
    }
    ProductionCommandMemory memory;
    memory.previous_velocity_command = state.qd;
    memory.previous_velocity_command_initialized = true;
    const auto result = production_command_step(state, qdd, config, memory);
    ASSERT_TRUE(result.diagnostics.finite);
    EXPECT_LE(
      result.diagnostics.applied_qdd.cwiseAbs().maxCoeff(),
      config.max_joint_accel + 1e-12);
    EXPECT_LE(
      result.state.qd.cwiseAbs().maxCoeff(),
      config.command_guard_max_velocity_rad_s + 1e-12);
    for (int joint = 0; joint < 6; ++joint) {
      const double guarded_lower =
        config.joint_lower_limits[static_cast<std::size_t>(joint)] +
        config.joint_limit_buffers[static_cast<std::size_t>(joint)];
      const double guarded_upper =
        config.joint_upper_limits[static_cast<std::size_t>(joint)] -
        config.joint_limit_buffers[static_cast<std::size_t>(joint)];
      EXPECT_GE(result.state.q[joint], guarded_lower - 1e-12);
      EXPECT_LE(result.state.q[joint], guarded_upper + 1e-12);
    }
  }
}

}  // namespace
}  // namespace rb10_rmpflow_rviz

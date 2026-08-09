#include <pthread.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_box.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "rb10_rmpflow_rviz/rb10_socket_client.hpp"
#include "rb10_rmpflow_rviz/casadi_rmp2_solver.hpp"
#include "rb10_rmpflow_rviz/joint_acceleration_limiter.hpp"
#include "rb10_rmpflow_rviz/pinocchio_direct_solver.hpp"
#include "rb10_rmpflow_rviz/production_command_map.hpp"
#include "rb10_rmpflow_rviz/rb10_model.hpp"
#include "rb10_rmpflow_rviz/rmp_solver_interface.hpp"
#include "rb10_rmpflow_rviz/tangent_escape_natural_form.hpp"

namespace rb10_rmpflow_rviz
{

namespace
{

using JointVector = RB10Model::JointVector;
using JointStateRtPublisher = realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>;
using Float64ArrayRtPublisher =
  realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>;
using PoseRtPublisher = realtime_tools::RealtimePublisher<geometry_msgs::msg::Pose>;
constexpr double kPi = 3.14159265358979323846;

std::int64_t obstacle_source_id(const visualization_msgs::msg::Marker & marker)
{
  constexpr std::uint64_t kExactlyRepresentableDoubleMask = (1ULL << 52) - 1ULL;
  std::uint64_t seed = static_cast<std::uint64_t>(std::hash<std::string>{}(marker.ns));
  seed ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(marker.id)) +
    0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
  return static_cast<std::int64_t>(seed & kExactlyRepresentableDoubleMask);
}

double degrees_to_radians(double degrees)
{
  return degrees * kPi / 180.0;
}

double radians_to_degrees(double radians)
{
  return radians * 180.0 / kPi;
}

std::filesystem::path expand_home_path(const std::string & value)
{
  if (value.empty() || value.front() != '~') {
    return std::filesystem::path(value);
  }
  const char * home = std::getenv("HOME");
  if (home == nullptr || std::string(home).empty()) {
    throw std::runtime_error("HOME is not set; cannot expand output path: " + value);
  }
  if (value.size() == 1) {
    return std::filesystem::path(home);
  }
  if (value[1] != '/') {
    throw std::runtime_error("Only ~/... paths are supported: " + value);
  }
  return std::filesystem::path(home) / value.substr(2);
}

std::string file_timestamp()
{
  const std::time_t now = std::time(nullptr);
  std::tm local_time{};
  localtime_r(&now, &local_time);
  std::ostringstream stream;
  stream << std::put_time(&local_time, "%Y%m%d_%H%M%S");
  return stream.str();
}

YAML::Node yaml_joint_vector(const JointVector & vector)
{
  YAML::Node node(YAML::NodeType::Sequence);
  for (Eigen::Index index = 0; index < vector.size(); ++index) {
    node.push_back(vector[index]);
  }
  return node;
}

template<typename MatrixType>
YAML::Node yaml_matrix(const MatrixType & matrix)
{
  YAML::Node node(YAML::NodeType::Sequence);
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    YAML::Node output_row(YAML::NodeType::Sequence);
    for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
      output_row.push_back(matrix(row, column));
    }
    node.push_back(output_row);
  }
  return node;
}

struct RobotState
{
  JointVector q{JointVector::Zero()};
  JointVector qd{JointVector::Zero()};
};

struct CommandGuardDiagnostics
{
  bool clamped{false};
  bool velocity_slew_clamped{false};
  JointVector pre_guard_command{JointVector::Zero()};
  JointVector command_step{JointVector::Zero()};
};

struct BackendTimingSnapshot
{
  std::uint64_t state_interval_count{0U};
  double state_interval_latest_sec{0.0};
  double state_interval_total_sec{0.0};
  double state_interval_max_sec{0.0};
  double state_interval_window_max_sec{0.0};
  std::uint64_t state_gap_over_5ms_count{0U};
  std::uint64_t state_gap_over_20ms_count{0U};
  std::uint64_t state_gap_over_100ms_count{0U};
  std::uint64_t velocity_filter_update_count{0U};
  std::uint64_t velocity_filter_input_reject_count{0U};
  std::uint64_t velocity_filter_output_reject_count{0U};
  double velocity_filter_synced_dt_window_max_sec{0.0};
  std::uint64_t controller_packet_time_used_count{0U};
  std::uint64_t controller_packet_time_fallback_count{0U};
  std::uint64_t controller_packet_time_duplicate_count{0U};
  std::uint64_t controller_packet_time_reset_count{0U};
  double controller_packet_time_dt_window_max_sec{0.0};
  JointVector raw_position_step_abs_window_max{JointVector::Zero()};
  JointVector raw_velocity_abs_window_max{JointVector::Zero()};
  JointVector velocity_filter_residual_abs_window_max{JointVector::Zero()};
  JointVector filtered_velocity_abs_window_max{JointVector::Zero()};
};

struct ControlTimingAccumulator
{
  std::uint64_t active_cycle_count{0U};
  bool previous_active_cycle_valid{false};
  std::chrono::steady_clock::time_point previous_active_cycle_start{};
  std::chrono::steady_clock::time_point last_publish_time{};
  double cycle_interval_latest_sec{0.0};
  double cycle_interval_total_sec{0.0};
  double cycle_interval_max_sec{0.0};
  double cycle_interval_window_max_sec{0.0};
  double solve_latest_sec{0.0};
  double solve_total_sec{0.0};
  double solve_max_sec{0.0};
  double solve_window_max_sec{0.0};
  double pre_solve_total_sec{0.0};
  double pre_solve_window_max_sec{0.0};
  double post_solve_total_sec{0.0};
  double post_solve_window_max_sec{0.0};
  double work_latest_sec{0.0};
  double work_total_sec{0.0};
  double work_max_sec{0.0};
  double work_window_max_sec{0.0};
  double start_lateness_latest_sec{0.0};
  double start_lateness_total_sec{0.0};
  double start_lateness_max_sec{0.0};
  double start_lateness_window_max_sec{0.0};
  std::uint64_t deadline_miss_count{0U};
  double measured_qd_norm_max{0.0};
  double measured_qd_max_abs_max{0.0};
  double qdd_norm_max{0.0};
  double qdd_max_abs_max{0.0};
  double command_norm_max{0.0};
  double command_max_abs_max{0.0};
};

void update_atomic_max(std::atomic<std::int64_t> & target, std::int64_t value)
{
  std::int64_t current = target.load(std::memory_order_relaxed);
  while (
    value > current &&
    !target.compare_exchange_weak(
      current, value, std::memory_order_relaxed, std::memory_order_relaxed))
  {
  }
}

std::int64_t diagnostic_value_to_nano(double value)
{
  constexpr double kMaximumDiagnosticMagnitude = 1.0e6;
  if (!std::isfinite(value)) {
    return 0;
  }
  return static_cast<std::int64_t>(
    std::llround(std::min(std::abs(value), kMaximumDiagnosticMagnitude) * 1.0e9));
}

enum class CommandMode
{
  kPosition,
  kVelocity
};

std::string normalized_command_mode(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return value;
}

CommandMode parse_command_mode(const std::string & value)
{
  const std::string normalized = normalized_command_mode(value);
  if (normalized == "position" || normalized == "servo_j" || normalized == "servoj") {
    return CommandMode::kPosition;
  }
  if (normalized == "velocity" || normalized == "speed_j" || normalized == "speedj") {
    return CommandMode::kVelocity;
  }
  throw std::runtime_error(
    "command_mode must be one of: position, velocity, servo_j, speed_j");
}

const char * command_mode_to_string(CommandMode command_mode)
{
  return command_mode == CommandMode::kVelocity ? "velocity" : "position";
}

const char * rb10_stream_command_name(CommandMode command_mode)
{
  return command_mode == CommandMode::kVelocity ? "move_speed_j" : "move_servo_j";
}

struct TcpAccelerationSample
{
  std::array<double, 3> acceleration{0.0, 0.0, 0.0};
  double norm{0.0};
  bool valid{false};
};

struct TangentEscapeGeometrySample
{
  bool active{false};
  std::size_t sensor_index{0};
  Eigen::Vector3d control_point{Eigen::Vector3d::Zero()};
  Eigen::Vector3d obstacle_center{Eigen::Vector3d::Zero()};
  Eigen::Vector3d sensor_normal{Eigen::Vector3d::UnitZ()};
  Eigen::Vector3d obstacle_direction{Eigen::Vector3d::UnitZ()};
  Eigen::Vector3d collision_normal{Eigen::Vector3d::UnitZ()};
  Eigen::Vector3d tangent_bias{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d control_point_velocity{Eigen::Vector3d::Zero()};
  double clearance{0.0};
  double center_distance{0.0};
  double sensor_obstacle_dot{0.0};
  double collision_obstacle_dot{0.0};
  double bias_sensor_dot{0.0};
  double bias_obstacle_dot{0.0};
  double jacobian_frobenius_norm{0.0};
  double sensor_normal_jacobian_norm{0.0};
  double obstacle_direction_jacobian_norm{0.0};
  double tangent_bias_jacobian_norm{0.0};
  double velocity_obstacle_dot{0.0};
  double velocity_tangent_dot{0.0};
};

struct VelocityFilterDiagnostics
{
  bool valid_sample{false};
  bool output_updated{false};
  bool input_rejected{false};
  bool output_rejected{false};
  double synced_dt_sec{0.0};
  JointVector position_step{JointVector::Zero()};
  JointVector raw_velocity{JointVector::Zero()};
  JointVector residual{JointVector::Zero()};
  JointVector filtered_velocity{JointVector::Zero()};
};

class SynchronizedVelocityFilter
{
public:
  enum class FilterType
  {
    kMovingAverage,
    kAlphaBeta
  };

  static FilterType parse_filter_type(const std::string & filter_type)
  {
    if (
      filter_type == "alphabeta" ||
      filter_type == "alpha-beta" ||
      filter_type == "alpha_beta")
    {
      return FilterType::kAlphaBeta;
    }
    return FilterType::kMovingAverage;
  }

  static const char * filter_type_to_string(FilterType filter_type)
  {
    if (filter_type == FilterType::kAlphaBeta) {
      return "alpha-beta";
    }
    return "moving-average";
  }

  SynchronizedVelocityFilter(
    double input_rate_hz,
    double output_rate_hz,
    double lowpass_alpha,
    double ratio_tolerance,
    FilterType filter_type = FilterType::kMovingAverage,
    double alpha_beta_beta = 0.0,
    double max_raw_velocity_rad_s = 6.0,
    double position_step_margin_rad = 0.01,
    double max_filtered_velocity_rad_s = 2.0)
  : input_rate_hz_(std::max(input_rate_hz, 1.0)),
    output_rate_hz_(std::max(output_rate_hz, 1.0)),
    lowpass_alpha_(std::clamp(lowpass_alpha, 0.0, 1.0)),
    ratio_tolerance_(std::max(ratio_tolerance, 0.0)),
    filter_type_(filter_type),
    alpha_beta_beta_(std::clamp(alpha_beta_beta, 0.0, 1.0)),
    max_raw_velocity_rad_s_(std::max(max_raw_velocity_rad_s, 0.0)),
    position_step_margin_rad_(std::max(position_step_margin_rad, 0.0)),
    max_filtered_velocity_rad_s_(std::max(max_filtered_velocity_rad_s, 0.0))
  {
    const double raw_ratio = input_rate_hz_ / output_rate_hz_;
    const double rounded_ratio = std::round(raw_ratio);
    if (
      rounded_ratio >= 1.0 &&
      std::abs(raw_ratio - rounded_ratio) <= ratio_tolerance_)
    {
      sync_multiple_ = static_cast<int>(rounded_ratio);
      integer_ratio_aligned_ = true;
    } else {
      sync_multiple_ = std::max(1, static_cast<int>(std::lround(std::max(raw_ratio, 1.0))));
      integer_ratio_aligned_ = false;
    }
  }

  void reset(
    const JointVector & position,
    const std::chrono::steady_clock::time_point & current_time)
  {
    previous_position_ = position;
    previous_time_ = current_time;
    accumulated_position_step_.setZero();
    filtered_velocity_.setZero();
    accumulated_time_sec_ = 0.0;
    alpha_beta_position_.setZero();
    alpha_beta_initialized_ = false;
    accumulated_samples_ = 0;
    initialized_ = true;
    output_initialized_ = false;
    diagnostics_ = VelocityFilterDiagnostics{};
  }

  JointVector update(
    const JointVector & position,
    const std::chrono::steady_clock::time_point & current_time)
  {
    diagnostics_ = VelocityFilterDiagnostics{};
    if (!initialized_) {
      reset(position, current_time);
      return JointVector::Zero();
    }

    const std::chrono::duration<double> dt_duration = current_time - previous_time_;
    const double dt_sec = dt_duration.count();
    if (dt_sec <= std::numeric_limits<double>::epsilon() || dt_sec > 0.1) {
      reset(position, current_time);
      return JointVector::Zero();
    }

    const JointVector position_step = position - previous_position_;
    const JointVector raw_velocity = position_step / dt_sec;
    diagnostics_.valid_sample = true;
    diagnostics_.position_step = position_step;
    diagnostics_.raw_velocity = raw_velocity;
    diagnostics_.filtered_velocity = filtered_velocity_;
    const double allowed_position_step =
      max_raw_velocity_rad_s_ * dt_sec + position_step_margin_rad_;
    if (
      !position.allFinite() || !raw_velocity.allFinite() ||
      position_step.cwiseAbs().maxCoeff() > allowed_position_step)
    {
      diagnostics_.input_rejected = true;
      if (!output_initialized_) {
        return JointVector::Zero();
      }
      return filtered_velocity_;
    }
    previous_position_ = position;
    previous_time_ = current_time;
    accumulated_position_step_ += position_step;
    accumulated_time_sec_ += dt_sec;
    ++accumulated_samples_;

    if (accumulated_samples_ >= sync_multiple_) {
      const JointVector measured_velocity =
        accumulated_position_step_ /
        std::max(accumulated_time_sec_, std::numeric_limits<double>::epsilon());
      const double synced_dt_sec =
        std::max(accumulated_time_sec_, std::numeric_limits<double>::epsilon());

      if (filter_type_ == FilterType::kAlphaBeta) {
        if (!alpha_beta_initialized_) {
          alpha_beta_position_ = position;
          filtered_velocity_ = measured_velocity;
          alpha_beta_initialized_ = true;
        } else {
          const JointVector predicted_position =
            alpha_beta_position_ + filtered_velocity_ * synced_dt_sec;
          const JointVector residual = position - predicted_position;
          diagnostics_.residual = residual;
          alpha_beta_position_ = predicted_position + lowpass_alpha_ * residual;
          filtered_velocity_ += (alpha_beta_beta_ * residual) / synced_dt_sec;
        }
      } else {
        if (!output_initialized_) {
          filtered_velocity_ = measured_velocity;
        } else {
          filtered_velocity_ =
            lowpass_alpha_ * measured_velocity +
            (1.0 - lowpass_alpha_) * filtered_velocity_;
        }
      }
      output_initialized_ = true;
      diagnostics_.output_updated = true;
      diagnostics_.synced_dt_sec = synced_dt_sec;
      if (
        !filtered_velocity_.allFinite() ||
        filtered_velocity_.cwiseAbs().maxCoeff() > max_filtered_velocity_rad_s_)
      {
        VelocityFilterDiagnostics rejected_diagnostics = diagnostics_;
        rejected_diagnostics.output_rejected = true;
        rejected_diagnostics.filtered_velocity = filtered_velocity_;
        reset(position, current_time);
        diagnostics_ = rejected_diagnostics;
        return JointVector::Zero();
      }
      accumulated_position_step_.setZero();
      accumulated_time_sec_ = 0.0;
      accumulated_samples_ = 0;
    }

    if (!output_initialized_) {
      return JointVector::Zero();
    }
    diagnostics_.filtered_velocity = filtered_velocity_;
    return filtered_velocity_;
  }

  const VelocityFilterDiagnostics & diagnostics() const
  {
    return diagnostics_;
  }

  int sync_multiple() const
  {
    return sync_multiple_;
  }

  bool integer_ratio_aligned() const
  {
    return integer_ratio_aligned_;
  }

private:
  double input_rate_hz_{100.0};
  double output_rate_hz_{100.0};
  double lowpass_alpha_{0.25};
  double ratio_tolerance_{0.05};
  FilterType filter_type_{FilterType::kMovingAverage};
  double alpha_beta_beta_{0.0};
  double max_raw_velocity_rad_s_{6.0};
  double position_step_margin_rad_{0.01};
  double max_filtered_velocity_rad_s_{2.0};
  bool alpha_beta_initialized_{false};
  JointVector alpha_beta_position_{JointVector::Zero()};
  int sync_multiple_{1};
  bool integer_ratio_aligned_{true};
  bool initialized_{false};
  bool output_initialized_{false};
  int accumulated_samples_{0};
  JointVector previous_position_{JointVector::Zero()};
  JointVector accumulated_position_step_{JointVector::Zero()};
  JointVector filtered_velocity_{JointVector::Zero()};
  std::chrono::steady_clock::time_point previous_time_{};
  double accumulated_time_sec_{0.0};
  VelocityFilterDiagnostics diagnostics_{};
};

class ControllerPacketSampleClock
{
public:
  struct Update
  {
    std::chrono::steady_clock::time_point sample_time{};
    double controller_dt_sec{0.0};
    bool advanced{false};
    bool reset_filter{false};
    bool used_controller_time{false};
    bool used_fallback_time{false};
    bool duplicate_controller_time{false};
    bool controller_time_reset{false};
  };

  explicit ControllerPacketSampleClock(double nominal_rate_hz)
  : nominal_period_sec_(1.0 / std::max(nominal_rate_hz, 1.0))
  {
  }

  Update update(
    double controller_time_sec,
    const std::chrono::steady_clock::time_point & arrival_time)
  {
    Update result;
    result.sample_time = sample_time_;
    if (!initialized_) {
      initialized_ = true;
      sample_time_ = arrival_time;
      detection_start_time_ = arrival_time;
      previous_controller_time_sec_ = controller_time_sec;
      previous_controller_time_valid_ = std::isfinite(controller_time_sec);
      result.sample_time = sample_time_;
      result.reset_filter = true;
      return result;
    }

    if (mode_ == Mode::kFallback) {
      sample_time_ = std::max(
        sample_time_ + duration(nominal_period_sec_),
        arrival_time);
      result.sample_time = sample_time_;
      result.advanced = true;
      result.used_fallback_time = true;
      return result;
    }

    const bool controller_time_valid = std::isfinite(controller_time_sec);
    const double controller_dt_sec =
      controller_time_valid && previous_controller_time_valid_ ?
      controller_time_sec - previous_controller_time_sec_ : 0.0;
    if (controller_time_valid) {
      previous_controller_time_sec_ = controller_time_sec;
      previous_controller_time_valid_ = true;
    } else {
      previous_controller_time_valid_ = false;
    }

    constexpr double kMinimumControllerDtSec = 1.0e-6;
    constexpr double kMaximumControllerDtSec = 0.1;
    if (
      controller_time_valid &&
      controller_dt_sec > kMinimumControllerDtSec &&
      controller_dt_sec <= kMaximumControllerDtSec)
    {
      mode_ = Mode::kControllerTime;
      sample_time_ += duration(controller_dt_sec);
      result.sample_time = sample_time_;
      result.controller_dt_sec = controller_dt_sec;
      result.advanced = true;
      result.used_controller_time = true;
      return result;
    }

    const bool duplicate =
      controller_time_valid &&
      std::abs(controller_dt_sec) <= kMinimumControllerDtSec;
    result.duplicate_controller_time = duplicate;
    if (mode_ == Mode::kControllerTime && duplicate) {
      result.sample_time = sample_time_;
      return result;
    }

    const bool reset =
      !controller_time_valid ||
      (previous_controller_time_valid_ &&
      (controller_dt_sec < -kMinimumControllerDtSec ||
      controller_dt_sec > kMaximumControllerDtSec));
    if (mode_ == Mode::kControllerTime && reset) {
      mode_ = Mode::kDetecting;
      detection_start_time_ = arrival_time;
      sample_time_ = arrival_time;
      result.sample_time = sample_time_;
      result.reset_filter = true;
      result.controller_time_reset = true;
      return result;
    }

    constexpr double kControllerClockDetectionTimeoutSec = 0.05;
    if (
      std::chrono::duration<double>(arrival_time - detection_start_time_).count() <
      kControllerClockDetectionTimeoutSec)
    {
      result.sample_time = sample_time_;
      return result;
    }

    mode_ = Mode::kFallback;
    sample_time_ = arrival_time;
    result.sample_time = sample_time_;
    result.reset_filter = true;
    result.used_fallback_time = true;
    return result;
  }

private:
  enum class Mode
  {
    kDetecting,
    kControllerTime,
    kFallback
  };

  static std::chrono::steady_clock::duration duration(double seconds)
  {
    return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(seconds));
  }

  double nominal_period_sec_{0.002};
  bool initialized_{false};
  bool previous_controller_time_valid_{false};
  double previous_controller_time_sec_{0.0};
  Mode mode_{Mode::kDetecting};
  std::chrono::steady_clock::time_point sample_time_{};
  std::chrono::steady_clock::time_point detection_start_time_{};
};

struct ExternalRmpBuffer
{
  int dim{0};
  Eigen::MatrixXd metric_sqrt{Eigen::MatrixXd::Zero(0, 0)};
  Eigen::VectorXd acceleration{Eigen::VectorXd::Zero(0)};
  bool has_metric{false};
  bool has_acceleration{false};
};

struct GoalTarget
{
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};

bool command_blocked_by_robot_safety(const Rb10SystemState & state)
{
  return state.op_stat_soft_estop_occur != 0 ||
         state.op_stat_ems_flag != 0 ||
         state.op_stat_collision_occur != 0 ||
         state.op_stat_self_collision != 0 ||
         state.op_stat_sos_flag != 0;
}

std::string robot_safety_summary(const Rb10SystemState & state)
{
  std::string summary;
  const auto append_flag = [&summary](const char * label, std::int32_t value) {
    if (value == 0) {
      return;
    }
    if (!summary.empty()) {
      summary += ", ";
    }
    summary += label;
  };

  append_flag("soft_estop", state.op_stat_soft_estop_occur);
  append_flag("ems", state.op_stat_ems_flag);
  append_flag("collision", state.op_stat_collision_occur);
  append_flag("self_collision", state.op_stat_self_collision);
  append_flag("sos", state.op_stat_sos_flag);

  if (summary.empty()) {
    summary = "none";
  }
  return summary;
}

class ControllerBackend
{
public:
  virtual ~ControllerBackend() = default;

  virtual RobotState read_state() = 0;
  virtual bool ready() const
  {
    return true;
  }
  virtual BackendTimingSnapshot take_timing_snapshot()
  {
    return BackendTimingSnapshot{};
  }
  virtual void apply_command(
    const RobotState & command_state,
    const std::array<const char *, 6> & joint_names) = 0;
};

bool joint_state_to_robot_state(const sensor_msgs::msg::JointState & msg, RobotState & state)
{
  if (msg.position.size() < RB10Model::joint_names.size()) {
    return false;
  }

  if (msg.name.size() >= RB10Model::joint_names.size()) {
    std::unordered_map<std::string, std::size_t> name_to_index;
    name_to_index.reserve(msg.name.size());
    for (std::size_t index = 0; index < msg.name.size(); ++index) {
      name_to_index.emplace(msg.name[index], index);
    }

    bool matched_all = true;
    for (std::size_t index = 0; index < RB10Model::joint_names.size(); ++index) {
      const auto it = name_to_index.find(RB10Model::joint_names[index]);
      if (it == name_to_index.end() || it->second >= msg.position.size()) {
        matched_all = false;
        break;
      }
      state.q[static_cast<int>(index)] = msg.position[it->second];
      state.qd[static_cast<int>(index)] =
        it->second < msg.velocity.size() ? msg.velocity[it->second] : 0.0;
    }

    if (matched_all) {
      return true;
    }
  }

  for (std::size_t index = 0; index < RB10Model::joint_names.size(); ++index) {
    state.q[static_cast<int>(index)] = msg.position[index];
    state.qd[static_cast<int>(index)] =
      index < msg.velocity.size() ? msg.velocity[index] : 0.0;
  }
  return true;
}

class SimulationBackend : public ControllerBackend
{
public:
  explicit SimulationBackend(const JointVector & initial_q)
  {
    state_.q = initial_q;
  }

  RobotState read_state() override
  {
    std::scoped_lock lock(mutex_);
    return state_;
  }

  void apply_command(
    const RobotState & command_state,
    const std::array<const char *, 6> &) override
  {
    std::scoped_lock lock(mutex_);
    state_ = command_state;
  }

private:
  std::mutex mutex_;
  RobotState state_;
};

class HardwareBridgeBackend : public ControllerBackend
{
public:
  HardwareBridgeBackend(
    rclcpp::Node * node,
    const JointVector & initial_q,
    const std::string & state_topic,
    const std::string & command_topic)
  : node_(node)
  {
    RobotState initial_state;
    initial_state.q = initial_q;
    state_buffer_.initRT(initial_state);
    command_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(command_topic, 10);
    state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
      state_topic,
      10,
      std::bind(&HardwareBridgeBackend::on_state, this, std::placeholders::_1));
  }

  RobotState read_state() override
  {
    return *state_buffer_.readFromRT();
  }

  void apply_command(
    const RobotState & command_state,
    const std::array<const char *, 6> & joint_names) override
  {
    sensor_msgs::msg::JointState command;
    command.header.stamp = node_->now();
    command.name.assign(joint_names.begin(), joint_names.end());
    command.position.assign(command_state.q.data(), command_state.q.data() + command_state.q.size());
    command.velocity.assign(command_state.qd.data(), command_state.qd.data() + command_state.qd.size());

    command_pub_->publish(command);
  }

private:
  void on_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    RobotState next_state;
    if (!joint_state_to_robot_state(*msg, next_state)) {
      return;
    }

    state_buffer_.writeFromNonRT(next_state);
  }

  rclcpp::Node * node_;
  realtime_tools::RealtimeBuffer<RobotState> state_buffer_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr command_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
};

class JointCommandTopicsBackend : public ControllerBackend
{
public:
  JointCommandTopicsBackend(
    rclcpp::Node * node,
    const JointVector & initial_q,
    const std::string & state_topic,
    const std::string & command_topic,
    CommandMode command_mode,
    bool collect_timing_debug,
    double state_rate_hz,
    double control_rate_hz,
    bool use_synced_input_velocity_filter,
    double synced_input_velocity_filter_alpha,
    double synced_input_velocity_filter_beta,
    const std::string & synced_input_velocity_filter_type,
    double synced_input_velocity_ratio_tolerance,
    double synced_input_velocity_max_raw_rad_s,
    double synced_input_velocity_position_step_margin_rad,
    double synced_input_velocity_max_output_rad_s)
  : node_(node)
  , state_topic_(state_topic)
  , command_mode_(command_mode)
  , collect_timing_debug_(collect_timing_debug)
  , use_synced_input_velocity_filter_(use_synced_input_velocity_filter)
  {
    RobotState initial_state;
    initial_state.q = initial_q;
    state_buffer_.initRT(initial_state);
    if (use_synced_input_velocity_filter_) {
      const auto filter_type =
        SynchronizedVelocityFilter::parse_filter_type(synced_input_velocity_filter_type);
      synced_input_velocity_filter_ = std::make_unique<SynchronizedVelocityFilter>(
        state_rate_hz,
        control_rate_hz,
        synced_input_velocity_filter_alpha,
        synced_input_velocity_ratio_tolerance,
        filter_type,
        synced_input_velocity_filter_beta,
        synced_input_velocity_max_raw_rad_s,
        synced_input_velocity_position_step_margin_rad,
        synced_input_velocity_max_output_rad_s);
      RCLCPP_INFO(
        node_->get_logger(),
        "Joint-state topic velocity sync filter enabled (%s, alpha=%.3f beta=%.3f, raw limit=%.2f rad/s, output limit=%.2f rad/s): %.1f Hz input -> %d-sample synced velocity for %.1f Hz control",
        SynchronizedVelocityFilter::filter_type_to_string(filter_type),
        synced_input_velocity_filter_alpha,
        synced_input_velocity_filter_beta,
        synced_input_velocity_max_raw_rad_s,
        synced_input_velocity_max_output_rad_s,
        state_rate_hz,
        synced_input_velocity_filter_->sync_multiple(),
        control_rate_hz);
      if (!synced_input_velocity_filter_->integer_ratio_aligned()) {
        RCLCPP_WARN(
          node_->get_logger(),
          "joint-state input/control rate is not close to an integer multiple; using the nearest %d-sample velocity sync window",
          synced_input_velocity_filter_->sync_multiple());
      }
    }
    command_pub_ = node_->create_publisher<std_msgs::msg::Float64MultiArray>(command_topic, 10);
    state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
      state_topic,
      10,
      std::bind(&JointCommandTopicsBackend::on_state, this, std::placeholders::_1));
  }

  RobotState read_state() override
  {
    return *state_buffer_.readFromRT();
  }

  bool ready() const override
  {
    return state_received_.load();
  }

  BackendTimingSnapshot take_timing_snapshot() override
  {
    constexpr double kNanosecondsToSeconds = 1e-9;
    BackendTimingSnapshot snapshot;
    snapshot.state_interval_count =
      state_interval_count_.load(std::memory_order_relaxed);
    snapshot.state_interval_latest_sec =
      static_cast<double>(state_interval_latest_ns_.load(std::memory_order_relaxed)) *
      kNanosecondsToSeconds;
    snapshot.state_interval_total_sec =
      static_cast<double>(state_interval_total_ns_.load(std::memory_order_relaxed)) *
      kNanosecondsToSeconds;
    snapshot.state_interval_max_sec =
      static_cast<double>(state_interval_max_ns_.load(std::memory_order_relaxed)) *
      kNanosecondsToSeconds;
    snapshot.state_interval_window_max_sec =
      static_cast<double>(
      state_interval_window_max_ns_.exchange(0, std::memory_order_relaxed)) *
      kNanosecondsToSeconds;
    snapshot.state_gap_over_5ms_count =
      state_gap_over_5ms_count_.load(std::memory_order_relaxed);
    snapshot.state_gap_over_20ms_count =
      state_gap_over_20ms_count_.load(std::memory_order_relaxed);
    snapshot.state_gap_over_100ms_count =
      state_gap_over_100ms_count_.load(std::memory_order_relaxed);
    snapshot.velocity_filter_update_count =
      velocity_filter_update_count_.load(std::memory_order_relaxed);
    snapshot.velocity_filter_input_reject_count =
      velocity_filter_input_reject_count_.load(std::memory_order_relaxed);
    snapshot.velocity_filter_output_reject_count =
      velocity_filter_output_reject_count_.load(std::memory_order_relaxed);
    snapshot.velocity_filter_synced_dt_window_max_sec =
      static_cast<double>(
      velocity_filter_synced_dt_window_max_ns_.exchange(0, std::memory_order_relaxed)) *
      kNanosecondsToSeconds;
    for (int joint = 0; joint < 6; ++joint) {
      snapshot.raw_position_step_abs_window_max[joint] =
        static_cast<double>(
        raw_position_step_abs_window_max_nano_[joint].exchange(
          0, std::memory_order_relaxed)) * kNanosecondsToSeconds;
      snapshot.raw_velocity_abs_window_max[joint] =
        static_cast<double>(
        raw_velocity_abs_window_max_nano_[joint].exchange(
          0, std::memory_order_relaxed)) * kNanosecondsToSeconds;
      snapshot.velocity_filter_residual_abs_window_max[joint] =
        static_cast<double>(
        velocity_filter_residual_abs_window_max_nano_[joint].exchange(
          0, std::memory_order_relaxed)) * kNanosecondsToSeconds;
      snapshot.filtered_velocity_abs_window_max[joint] =
        static_cast<double>(
        filtered_velocity_abs_window_max_nano_[joint].exchange(
          0, std::memory_order_relaxed)) * kNanosecondsToSeconds;
    }
    return snapshot;
  }

  void apply_command(
    const RobotState & command_state,
    const std::array<const char *, 6> &) override
  {
    if (!ready()) {
      return;
    }

    std_msgs::msg::Float64MultiArray command;
    const JointVector & command_vector =
      command_mode_ == CommandMode::kVelocity ? command_state.qd : command_state.q;
    command.data.assign(command_vector.data(), command_vector.data() + command_vector.size());

    command_pub_->publish(command);
  }

private:
  void on_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    const auto callback_time = std::chrono::steady_clock::now();
    if (collect_timing_debug_) {
      record_state_interval(callback_time);
    }
    RobotState next_state;
    if (!joint_state_to_robot_state(*msg, next_state)) {
      return;
    }

    bool velocity_input_rejected = false;
    bool velocity_output_rejected = false;
    VelocityFilterDiagnostics velocity_diagnostics;
    if (use_synced_input_velocity_filter_ && synced_input_velocity_filter_) {
      auto sample_time = callback_time;
      const std::int64_t stamp_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
      if (stamp_ns > 0) {
        sample_time = std::chrono::steady_clock::time_point(
          std::chrono::nanoseconds(stamp_ns));
      }
      next_state.qd = synced_input_velocity_filter_->update(next_state.q, sample_time);
      velocity_diagnostics = synced_input_velocity_filter_->diagnostics();
      velocity_input_rejected = velocity_diagnostics.input_rejected;
      velocity_output_rejected = velocity_diagnostics.output_rejected;
      if (collect_timing_debug_) {
        record_velocity_filter_diagnostics(velocity_diagnostics);
      }
    }

    if (velocity_input_rejected) {
      RCLCPP_ERROR_THROTTLE(
        node_->get_logger(),
        *node_->get_clock(),
        1000,
        "Rejected implausible topic joint-state step before velocity estimation (raw max=%.3f rad/s, step max=%.6f rad)",
        velocity_diagnostics.raw_velocity.cwiseAbs().maxCoeff(),
        velocity_diagnostics.position_step.cwiseAbs().maxCoeff());
      return;
    }
    if (velocity_output_rejected) {
      RCLCPP_ERROR_THROTTLE(
        node_->get_logger(),
        *node_->get_clock(),
        1000,
        "Reset topic velocity filter after implausible output (max=%.3f rad/s)",
        velocity_diagnostics.filtered_velocity.cwiseAbs().maxCoeff());
    }

    state_buffer_.writeFromNonRT(next_state);
    state_received_.store(true);
    if (!logged_initial_state_) {
      RCLCPP_INFO(
        node_->get_logger(),
        "Received initial robot state on %s; enabling RMP commands",
        state_topic_.c_str());
      logged_initial_state_ = true;
    }
  }

  void record_state_interval(const std::chrono::steady_clock::time_point & current_time)
  {
    const auto current_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      current_time.time_since_epoch()).count();
    const std::int64_t previous_ns =
      previous_state_time_ns_.exchange(current_ns, std::memory_order_relaxed);
    if (previous_ns <= 0 || current_ns <= previous_ns) {
      return;
    }

    const std::int64_t interval_ns = current_ns - previous_ns;
    state_interval_latest_ns_.store(interval_ns, std::memory_order_relaxed);
    state_interval_total_ns_.fetch_add(interval_ns, std::memory_order_relaxed);
    state_interval_count_.fetch_add(1U, std::memory_order_relaxed);
    update_atomic_max(state_interval_max_ns_, interval_ns);
    update_atomic_max(state_interval_window_max_ns_, interval_ns);
    if (interval_ns > 5'000'000) {
      state_gap_over_5ms_count_.fetch_add(1U, std::memory_order_relaxed);
    }
    if (interval_ns > 20'000'000) {
      state_gap_over_20ms_count_.fetch_add(1U, std::memory_order_relaxed);
    }
    if (interval_ns > 100'000'000) {
      state_gap_over_100ms_count_.fetch_add(1U, std::memory_order_relaxed);
    }
  }

  void record_velocity_filter_diagnostics(const VelocityFilterDiagnostics & diagnostics)
  {
    if (!diagnostics.valid_sample) {
      return;
    }
    if (diagnostics.input_rejected) {
      velocity_filter_input_reject_count_.fetch_add(1U, std::memory_order_relaxed);
    }
    if (diagnostics.output_rejected) {
      velocity_filter_output_reject_count_.fetch_add(1U, std::memory_order_relaxed);
    }
    for (int joint = 0; joint < 6; ++joint) {
      update_atomic_max(
        raw_position_step_abs_window_max_nano_[joint],
        diagnostic_value_to_nano(diagnostics.position_step[joint]));
      update_atomic_max(
        raw_velocity_abs_window_max_nano_[joint],
        diagnostic_value_to_nano(diagnostics.raw_velocity[joint]));
      update_atomic_max(
        filtered_velocity_abs_window_max_nano_[joint],
        diagnostic_value_to_nano(diagnostics.filtered_velocity[joint]));
      if (diagnostics.output_updated) {
        update_atomic_max(
          velocity_filter_residual_abs_window_max_nano_[joint],
          diagnostic_value_to_nano(diagnostics.residual[joint]));
      }
    }
    if (diagnostics.output_updated) {
      velocity_filter_update_count_.fetch_add(1U, std::memory_order_relaxed);
      update_atomic_max(
        velocity_filter_synced_dt_window_max_ns_,
        diagnostic_value_to_nano(diagnostics.synced_dt_sec));
    }
  }

  rclcpp::Node * node_;
  std::string state_topic_;
  CommandMode command_mode_{CommandMode::kPosition};
  bool collect_timing_debug_{false};
  bool use_synced_input_velocity_filter_{false};
  realtime_tools::RealtimeBuffer<RobotState> state_buffer_;
  std::atomic<bool> state_received_{false};
  bool logged_initial_state_{false};
  std::atomic<std::int64_t> previous_state_time_ns_{0};
  std::atomic<std::int64_t> state_interval_latest_ns_{0};
  std::atomic<std::int64_t> state_interval_total_ns_{0};
  std::atomic<std::int64_t> state_interval_max_ns_{0};
  std::atomic<std::int64_t> state_interval_window_max_ns_{0};
  std::atomic<std::uint64_t> state_interval_count_{0U};
  std::atomic<std::uint64_t> state_gap_over_5ms_count_{0U};
  std::atomic<std::uint64_t> state_gap_over_20ms_count_{0U};
  std::atomic<std::uint64_t> state_gap_over_100ms_count_{0U};
  std::atomic<std::uint64_t> velocity_filter_update_count_{0U};
  std::atomic<std::uint64_t> velocity_filter_input_reject_count_{0U};
  std::atomic<std::uint64_t> velocity_filter_output_reject_count_{0U};
  std::atomic<std::int64_t> velocity_filter_synced_dt_window_max_ns_{0};
  std::array<std::atomic<std::int64_t>, 6> raw_position_step_abs_window_max_nano_{};
  std::array<std::atomic<std::int64_t>, 6> raw_velocity_abs_window_max_nano_{};
  std::array<std::atomic<std::int64_t>, 6>
  velocity_filter_residual_abs_window_max_nano_{};
  std::array<std::atomic<std::int64_t>, 6> filtered_velocity_abs_window_max_nano_{};
  std::unique_ptr<SynchronizedVelocityFilter> synced_input_velocity_filter_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr command_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
};

class DirectRb10Backend : public ControllerBackend
{
public:
  DirectRb10Backend(
    rclcpp::Node * node,
    const JointVector & initial_q,
    const std::string & robot_ip,
    bool simulation_mode,
    const std::string & real_joint_state_source,
    bool publish_debug_joint_state_sources,
    CommandMode command_mode,
    bool hardware_command_enabled,
    bool collect_timing_debug,
    double data_request_rate_hz,
    double control_rate_hz,
    double servo_t1,
    double servo_t2,
    double servo_gain,
    double servo_alpha,
    double speedj_t1,
    double speedj_t2,
    double speedj_gain,
    double speedj_alpha,
    bool use_synced_input_velocity_filter,
    double synced_input_velocity_filter_alpha,
    double synced_input_velocity_filter_beta,
    const std::string & synced_input_velocity_filter_type,
    double synced_input_velocity_ratio_tolerance,
    double synced_input_velocity_max_raw_rad_s,
    double synced_input_velocity_position_step_margin_rad,
    double synced_input_velocity_max_output_rad_s,
    bool startup_move_to_default_pose,
    const std::vector<double> & startup_home_joints_deg,
    double startup_movej_speed,
    double startup_movej_accel,
    double startup_release_tolerance_deg,
    double startup_release_timeout_sec,
    bool enable_socket_realtime,
    int socket_realtime_priority,
    bool stop_on_shutdown,
    const std::string & shutdown_action)
  : node_(node),
    simulation_mode_(simulation_mode),
    real_joint_state_source_(real_joint_state_source),
    publish_debug_joint_state_sources_(publish_debug_joint_state_sources),
    command_mode_(command_mode),
    hardware_command_enabled_(hardware_command_enabled),
    collect_timing_debug_(collect_timing_debug),
    controller_packet_sample_clock_(data_request_rate_hz),
    socket_realtime_enabled_(enable_socket_realtime),
    socket_realtime_priority_(std::max(socket_realtime_priority, 1)),
    servo_t1_(servo_t1),
    servo_t2_(servo_t2),
    servo_gain_(servo_gain),
    servo_alpha_(servo_alpha),
    speedj_t1_(speedj_t1),
    speedj_t2_(speedj_t2),
    speedj_gain_(speedj_gain),
    speedj_alpha_(speedj_alpha),
    use_synced_input_velocity_filter_(use_synced_input_velocity_filter),
    startup_move_to_default_pose_(startup_move_to_default_pose),
    startup_movej_speed_(startup_movej_speed),
    startup_movej_accel_(startup_movej_accel),
    startup_release_tolerance_deg_(startup_release_tolerance_deg),
    startup_release_timeout_sec_(startup_release_timeout_sec),
    stop_on_shutdown_(stop_on_shutdown),
    shutdown_action_(shutdown_action)
  {
    RobotState initial_state;
    initial_state.q = initial_q;
    state_buffer_.initRT(initial_state);
    if (startup_home_joints_deg.size() != 6U) {
      throw std::runtime_error("startup_home_joints_deg must contain exactly 6 values");
    }
    for (std::size_t index = 0; index < 6U; ++index) {
      startup_home_joints_deg_[index] = startup_home_joints_deg[index];
    }

    if (!socket_client_.connect(
        robot_ip,
        data_request_rate_hz,
        std::bind(&DirectRb10Backend::on_state, this, std::placeholders::_1),
        std::bind(&DirectRb10Backend::on_cmd_log, this, std::placeholders::_1),
        enable_socket_realtime,
        socket_realtime_priority,
        hardware_command_enabled_))
    {
      throw std::runtime_error(
              hardware_command_enabled_ ?
              "Failed to open RB10 command/data sockets from direct backend" :
              "Failed to open RB10 read-only data socket from direct backend");
    }

    if (hardware_command_enabled_) {
      if (!socket_client_.initialize_robot(simulation_mode_)) {
        socket_client_.disconnect();
        throw std::runtime_error("Failed to initialize RB10 from direct backend");
      }
    } else {
      RCLCPP_WARN(
        node_->get_logger(),
        "RB10 READ-ONLY DRY-RUN: command socket, robot initialization, startup motion, "
        "streaming commands, and shutdown commands are disabled");
    }

    if (use_synced_input_velocity_filter_) {
      const auto filter_type =
        SynchronizedVelocityFilter::parse_filter_type(synced_input_velocity_filter_type);
      synced_input_velocity_filter_ = std::make_unique<SynchronizedVelocityFilter>(
        data_request_rate_hz,
        control_rate_hz,
        synced_input_velocity_filter_alpha,
        synced_input_velocity_ratio_tolerance,
        filter_type,
        synced_input_velocity_filter_beta,
        synced_input_velocity_max_raw_rad_s,
        synced_input_velocity_position_step_margin_rad,
        synced_input_velocity_max_output_rad_s);
      RCLCPP_INFO(
        node_->get_logger(),
        "Direct RB10 backend high-rate velocity sync filter enabled (%s, alpha=%.3f beta=%.3f, raw limit=%.2f rad/s, output limit=%.2f rad/s): %.1f Hz input -> %d-sample synced velocity for %.1f Hz control",
        SynchronizedVelocityFilter::filter_type_to_string(filter_type),
        synced_input_velocity_filter_alpha,
        synced_input_velocity_filter_beta,
        synced_input_velocity_max_raw_rad_s,
        synced_input_velocity_max_output_rad_s,
        data_request_rate_hz,
        synced_input_velocity_filter_->sync_multiple(),
        control_rate_hz);
      if (!synced_input_velocity_filter_->integer_ratio_aligned()) {
        RCLCPP_WARN(
          node_->get_logger(),
          "hardware_data_request_rate/control_rate is not close to an integer multiple; using the nearest %d-sample velocity sync window",
          synced_input_velocity_filter_->sync_multiple());
      }
    }

    if (publish_debug_joint_state_sources_) {
      reference_joint_state_publisher_ =
        node_->create_publisher<sensor_msgs::msg::JointState>("/rb10/reference_joint_states", 10);
      measured_joint_state_publisher_ =
        node_->create_publisher<sensor_msgs::msg::JointState>("/rb10/measured_joint_states", 10);
      tracking_error_publisher_ =
        node_->create_publisher<std_msgs::msg::Float64MultiArray>("/rb10/joint_tracking_error_deg", 10);
    }

    if (startup_move_to_default_pose_ && hardware_command_enabled_) {
      if (!socket_client_.send_movej_degrees(
          startup_home_joints_deg_, startup_movej_speed_, startup_movej_accel_))
      {
        socket_client_.disconnect();
        throw std::runtime_error("Failed to command RB10 startup move_j from direct backend");
      }
      startup_waiting_for_home_.store(true);
      startup_release_deadline_ =
        std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(startup_release_timeout_sec_));
      RCLCPP_INFO(
        node_->get_logger(),
        "Direct RB10 backend commanded startup move_j; waiting to release the control loop until home is reached or timeout expires");
    } else if (startup_move_to_default_pose_) {
      RCLCPP_WARN(
        node_->get_logger(),
        "Ignoring startup_move_to_default_pose because hardware commands are disabled");
    }

    if (hardware_command_enabled_) {
      servo_command_thread_running_ = true;
      servo_command_thread_ = std::thread(&DirectRb10Backend::servo_command_loop, this);
    }
  }

  ~DirectRb10Backend() override
  {
    stop_servo_command_thread();
    if (
      hardware_command_enabled_ && command_mode_ == CommandMode::kVelocity &&
      socket_client_.is_connected())
    {
      std::array<double, 6> zero_velocity_deg_s{};
      if (!socket_client_.send_speedj_degrees_per_sec(
          zero_velocity_deg_s, speedj_t1_, speedj_t2_, speedj_gain_, speedj_alpha_))
      {
        RCLCPP_WARN(node_->get_logger(), "Direct RB10 backend failed to send zero move_speed_j on shutdown");
      }
    }
    if (hardware_command_enabled_ && stop_on_shutdown_ && socket_client_.is_connected()) {
      const bool halt_first = shutdown_action_ != "pause";
      const bool stopped = socket_client_.send_shutdown_sequence(halt_first);
      if (stopped) {
        RCLCPP_INFO(node_->get_logger(), "Direct RB10 backend sent shutdown stop/clear sequence");
      } else {
        RCLCPP_WARN(node_->get_logger(), "Direct RB10 backend shutdown sequence did not complete cleanly");
      }
    }
    socket_client_.disconnect();
  }

  RobotState read_state() override
  {
    return *state_buffer_.readFromRT();
  }

  bool ready() const override
  {
    return socket_client_.is_connected() &&
           state_received_.load() &&
           !startup_waiting_for_home_.load();
  }

  BackendTimingSnapshot take_timing_snapshot() override
  {
    constexpr double kNanosecondsToSeconds = 1e-9;
    BackendTimingSnapshot snapshot;
    snapshot.state_interval_count =
      state_interval_count_.load(std::memory_order_relaxed);
    snapshot.state_interval_latest_sec =
      static_cast<double>(state_interval_latest_ns_.load(std::memory_order_relaxed)) *
      kNanosecondsToSeconds;
    snapshot.state_interval_total_sec =
      static_cast<double>(state_interval_total_ns_.load(std::memory_order_relaxed)) *
      kNanosecondsToSeconds;
    snapshot.state_interval_max_sec =
      static_cast<double>(state_interval_max_ns_.load(std::memory_order_relaxed)) *
      kNanosecondsToSeconds;
    snapshot.state_interval_window_max_sec =
      static_cast<double>(
      state_interval_window_max_ns_.exchange(0, std::memory_order_relaxed)) *
      kNanosecondsToSeconds;
    snapshot.state_gap_over_5ms_count =
      state_gap_over_5ms_count_.load(std::memory_order_relaxed);
    snapshot.state_gap_over_20ms_count =
      state_gap_over_20ms_count_.load(std::memory_order_relaxed);
    snapshot.state_gap_over_100ms_count =
      state_gap_over_100ms_count_.load(std::memory_order_relaxed);
    snapshot.velocity_filter_update_count =
      velocity_filter_update_count_.load(std::memory_order_relaxed);
    snapshot.velocity_filter_input_reject_count =
      velocity_filter_input_reject_count_.load(std::memory_order_relaxed);
    snapshot.velocity_filter_output_reject_count =
      velocity_filter_output_reject_count_.load(std::memory_order_relaxed);
    snapshot.velocity_filter_synced_dt_window_max_sec =
      static_cast<double>(
      velocity_filter_synced_dt_window_max_ns_.exchange(0, std::memory_order_relaxed)) *
      kNanosecondsToSeconds;
    snapshot.controller_packet_time_used_count =
      controller_packet_time_used_count_.load(std::memory_order_relaxed);
    snapshot.controller_packet_time_fallback_count =
      controller_packet_time_fallback_count_.load(std::memory_order_relaxed);
    snapshot.controller_packet_time_duplicate_count =
      controller_packet_time_duplicate_count_.load(std::memory_order_relaxed);
    snapshot.controller_packet_time_reset_count =
      controller_packet_time_reset_count_.load(std::memory_order_relaxed);
    snapshot.controller_packet_time_dt_window_max_sec =
      static_cast<double>(
      controller_packet_time_dt_window_max_ns_.exchange(0, std::memory_order_relaxed)) *
      kNanosecondsToSeconds;
    for (int joint = 0; joint < 6; ++joint) {
      snapshot.raw_position_step_abs_window_max[joint] =
        static_cast<double>(
        raw_position_step_abs_window_max_nano_[joint].exchange(
          0, std::memory_order_relaxed)) * kNanosecondsToSeconds;
      snapshot.raw_velocity_abs_window_max[joint] =
        static_cast<double>(
        raw_velocity_abs_window_max_nano_[joint].exchange(
          0, std::memory_order_relaxed)) * kNanosecondsToSeconds;
      snapshot.velocity_filter_residual_abs_window_max[joint] =
        static_cast<double>(
        velocity_filter_residual_abs_window_max_nano_[joint].exchange(
          0, std::memory_order_relaxed)) * kNanosecondsToSeconds;
      snapshot.filtered_velocity_abs_window_max[joint] =
        static_cast<double>(
        filtered_velocity_abs_window_max_nano_[joint].exchange(
          0, std::memory_order_relaxed)) * kNanosecondsToSeconds;
    }
    return snapshot;
  }

  void apply_command(
    const RobotState & command_state,
    const std::array<const char *, 6> &) override
  {
    if (!hardware_command_enabled_ || !ready()) {
      return;
    }
    if (command_blocked_due_to_safety_.load()) {
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(),
        *node_->get_clock(),
        1000,
        "Direct RB10 backend is blocking %s while a robot safety stop is active",
        rb10_stream_command_name(command_mode_));
      return;
    }

    std::array<double, 6> joint_deg{};
    const JointVector & command_vector =
      command_mode_ == CommandMode::kVelocity ? command_state.qd : command_state.q;
    for (std::size_t index = 0; index < 6U; ++index) {
      joint_deg[index] = radians_to_degrees(command_vector[static_cast<int>(index)]);
    }
    queue_servo_command(joint_deg);
  }

private:
  void queue_servo_command(const std::array<double, 6> & joint_deg)
  {
    bool notify = false;
    {
      std::scoped_lock lock(servo_command_wait_mutex_);
      latest_servo_joint_deg_.set(joint_deg);
      servo_command_pending_ = true;
      notify = true;
    }
    if (notify) {
      servo_command_cv_.notify_one();
    }
  }

  void clear_pending_servo_command()
  {
    std::scoped_lock lock(servo_command_wait_mutex_);
    servo_command_pending_ = false;
  }

  void stop_servo_command_thread()
  {
    {
      std::scoped_lock lock(servo_command_wait_mutex_);
      servo_command_thread_running_ = false;
      servo_command_pending_ = false;
    }
    servo_command_cv_.notify_all();
    if (servo_command_thread_.joinable()) {
      servo_command_thread_.join();
    }
  }

  void servo_command_loop()
  {
    if (socket_realtime_enabled_) {
      sched_param params{};
      params.sched_priority = socket_realtime_priority_;
      const int result = pthread_setschedparam(pthread_self(), SCHED_FIFO, &params);
      if (result == 0) {
        RCLCPP_INFO(
          node_->get_logger(),
          "Direct RB10 command sender running with SCHED_FIFO priority %d",
          socket_realtime_priority_);
      } else {
        RCLCPP_WARN(
          node_->get_logger(),
          "Failed to switch direct RB10 command sender to SCHED_FIFO priority %d (error=%d)",
          socket_realtime_priority_,
          result);
      }
    }

    while (true) {
      std::array<double, 6> joint_deg{};
      {
        std::unique_lock<std::mutex> lock(servo_command_wait_mutex_);
        servo_command_cv_.wait(lock, [this]() {
          return !servo_command_thread_running_ || servo_command_pending_;
        });
        if (!servo_command_thread_running_ && !servo_command_pending_) {
          return;
        }
        servo_command_pending_ = false;
        latest_servo_joint_deg_.get(joint_deg);
      }

      const bool sent =
        command_mode_ == CommandMode::kVelocity ?
        socket_client_.send_speedj_degrees_per_sec(
          joint_deg, speedj_t1_, speedj_t2_, speedj_gain_, speedj_alpha_) :
        socket_client_.send_servoj_degrees(
          joint_deg, servo_t1_, servo_t2_, servo_gain_, servo_alpha_);
      if (!sent)
      {
        RCLCPP_WARN_THROTTLE(
          node_->get_logger(),
          *node_->get_clock(),
          1000,
          "Direct RB10 backend failed to send %s command",
          rb10_stream_command_name(command_mode_));
      }
    }
  }

  void on_state(const Rb10SystemState & state)
  {
    RobotState next_state;
    const auto current_time = std::chrono::steady_clock::now();
    if (collect_timing_debug_) {
      record_state_interval(current_time);
    }
    std::array<double, 6> reference_position_rad{};
    std::array<double, 6> measured_position_rad{};
    const auto & joint_source_deg = select_joint_source_deg(state);
    for (std::size_t index = 0; index < 6U; ++index) {
      reference_position_rad[index] = degrees_to_radians(state.joint_ref_deg[index]);
      measured_position_rad[index] = degrees_to_radians(state.joint_ang_deg[index]);
      next_state.q[static_cast<int>(index)] = degrees_to_radians(joint_source_deg[index]);
    }

    bool velocity_input_rejected = false;
    bool velocity_output_rejected = false;
    VelocityFilterDiagnostics velocity_diagnostics;
    auto velocity_sample_time = current_time;
    if (use_synced_input_velocity_filter_ && synced_input_velocity_filter_) {
      const auto sample_clock_update =
        controller_packet_sample_clock_.update(state.controller_time_sec, current_time);
      velocity_sample_time = sample_clock_update.sample_time;
      if (collect_timing_debug_) {
        record_controller_packet_clock_diagnostics(sample_clock_update);
      }

      if (sample_clock_update.reset_filter) {
        synced_input_velocity_filter_->reset(next_state.q, velocity_sample_time);
        next_state.qd.setZero();
      } else if (sample_clock_update.advanced) {
        next_state.qd =
          synced_input_velocity_filter_->update(next_state.q, velocity_sample_time);
        velocity_diagnostics = synced_input_velocity_filter_->diagnostics();
        velocity_input_rejected = velocity_diagnostics.input_rejected;
        velocity_output_rejected = velocity_diagnostics.output_rejected;
        if (collect_timing_debug_) {
          record_velocity_filter_diagnostics(velocity_diagnostics);
        }
      } else {
        next_state.qd = state_buffer_.readFromRT()->qd;
      }
    }

    if (!velocity_input_rejected) {
      state_buffer_.writeFromNonRT(next_state);
      state_received_.store(true);
    }

    if (hardware_command_enabled_) {
      const bool command_blocked = command_blocked_by_robot_safety(state);
      const bool was_command_blocked = command_blocked_due_to_safety_.exchange(command_blocked);
      if (command_blocked && !was_command_blocked) {
        clear_pending_servo_command();
        RCLCPP_ERROR(
          node_->get_logger(),
          "RB10 safety stop active; blocking %s commands (%s)",
          rb10_stream_command_name(command_mode_),
          robot_safety_summary(state).c_str());
      } else if (!command_blocked && was_command_blocked) {
        RCLCPP_INFO(node_->get_logger(), "RB10 safety stop cleared; servo commands may resume");
      }
    }

    if (velocity_input_rejected) {
      RCLCPP_ERROR_THROTTLE(
        node_->get_logger(),
        *node_->get_clock(),
        1000,
        "Rejected implausible RB10 joint-state step before velocity estimation (raw max=%.3f rad/s, step max=%.6f rad)",
        velocity_diagnostics.raw_velocity.cwiseAbs().maxCoeff(),
        velocity_diagnostics.position_step.cwiseAbs().maxCoeff());
      return;
    }
    if (velocity_output_rejected) {
      RCLCPP_ERROR_THROTTLE(
        node_->get_logger(),
        *node_->get_clock(),
        1000,
        "Reset velocity filter after implausible output (max=%.3f rad/s)",
        velocity_diagnostics.filtered_velocity.cwiseAbs().maxCoeff());
    }

    if (publish_debug_joint_state_sources_) {
      publish_debug_joint_states(reference_position_rad, measured_position_rad);
    }

    if (!startup_waiting_for_home_.load()) {
      return;
    }

    bool home_reached = true;
    const auto & startup_joint_deg =
      simulation_mode_ ? state.joint_ref_deg : state.joint_ang_deg;
    for (std::size_t index = 0; index < 6U; ++index) {
      if (
        std::fabs(startup_joint_deg[index] - startup_home_joints_deg_[index]) >
        startup_release_tolerance_deg_)
      {
        home_reached = false;
        break;
      }
    }

    if (home_reached || current_time >= startup_release_deadline_) {
      startup_waiting_for_home_.store(false);
      if (use_synced_input_velocity_filter_ && synced_input_velocity_filter_) {
        synced_input_velocity_filter_->reset(next_state.q, velocity_sample_time);
        next_state.qd.setZero();
        state_buffer_.writeFromNonRT(next_state);
      }
      if (home_reached) {
        RCLCPP_INFO(node_->get_logger(), "Direct RB10 backend startup home pose reached; enabling RMP commands");
      } else {
        RCLCPP_WARN(node_->get_logger(), "Direct RB10 backend startup release timeout expired; enabling RMP commands");
      }
    }
  }

  void record_state_interval(const std::chrono::steady_clock::time_point & current_time)
  {
    const auto current_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      current_time.time_since_epoch()).count();
    const std::int64_t previous_ns =
      previous_state_time_ns_.exchange(current_ns, std::memory_order_relaxed);
    if (previous_ns <= 0 || current_ns <= previous_ns) {
      return;
    }

    const std::int64_t interval_ns = current_ns - previous_ns;
    state_interval_latest_ns_.store(interval_ns, std::memory_order_relaxed);
    state_interval_total_ns_.fetch_add(interval_ns, std::memory_order_relaxed);
    state_interval_count_.fetch_add(1U, std::memory_order_relaxed);
    update_atomic_max(state_interval_max_ns_, interval_ns);
    update_atomic_max(state_interval_window_max_ns_, interval_ns);
    if (interval_ns > 5'000'000) {
      state_gap_over_5ms_count_.fetch_add(1U, std::memory_order_relaxed);
    }
    if (interval_ns > 20'000'000) {
      state_gap_over_20ms_count_.fetch_add(1U, std::memory_order_relaxed);
    }
    if (interval_ns > 100'000'000) {
      state_gap_over_100ms_count_.fetch_add(1U, std::memory_order_relaxed);
    }
  }

  void record_velocity_filter_diagnostics(const VelocityFilterDiagnostics & diagnostics)
  {
    if (!diagnostics.valid_sample) {
      return;
    }
    if (diagnostics.input_rejected) {
      velocity_filter_input_reject_count_.fetch_add(1U, std::memory_order_relaxed);
    }
    if (diagnostics.output_rejected) {
      velocity_filter_output_reject_count_.fetch_add(1U, std::memory_order_relaxed);
    }
    for (int joint = 0; joint < 6; ++joint) {
      update_atomic_max(
        raw_position_step_abs_window_max_nano_[joint],
        diagnostic_value_to_nano(diagnostics.position_step[joint]));
      update_atomic_max(
        raw_velocity_abs_window_max_nano_[joint],
        diagnostic_value_to_nano(diagnostics.raw_velocity[joint]));
      update_atomic_max(
        filtered_velocity_abs_window_max_nano_[joint],
        diagnostic_value_to_nano(diagnostics.filtered_velocity[joint]));
      if (diagnostics.output_updated) {
        update_atomic_max(
          velocity_filter_residual_abs_window_max_nano_[joint],
          diagnostic_value_to_nano(diagnostics.residual[joint]));
      }
    }
    if (diagnostics.output_updated) {
      velocity_filter_update_count_.fetch_add(1U, std::memory_order_relaxed);
      update_atomic_max(
        velocity_filter_synced_dt_window_max_ns_,
        diagnostic_value_to_nano(diagnostics.synced_dt_sec));
    }
  }

  void record_controller_packet_clock_diagnostics(
    const ControllerPacketSampleClock::Update & update)
  {
    if (update.used_controller_time) {
      controller_packet_time_used_count_.fetch_add(1U, std::memory_order_relaxed);
      update_atomic_max(
        controller_packet_time_dt_window_max_ns_,
        diagnostic_value_to_nano(update.controller_dt_sec));
    }
    if (update.used_fallback_time) {
      controller_packet_time_fallback_count_.fetch_add(1U, std::memory_order_relaxed);
    }
    if (update.duplicate_controller_time) {
      controller_packet_time_duplicate_count_.fetch_add(1U, std::memory_order_relaxed);
    }
    if (update.controller_time_reset) {
      controller_packet_time_reset_count_.fetch_add(1U, std::memory_order_relaxed);
    }
  }

  void on_cmd_log(const std::string & text)
  {
    if (text.empty()) {
      return;
    }
    if (text.rfind("[socket]", 0U) == 0U) {
      RCLCPP_WARN(node_->get_logger(), "%s", text.c_str());
      return;
    }
    if (text == "The command was executed") {
      RCLCPP_DEBUG(node_->get_logger(), "RB10 CMD: %s", text.c_str());
      return;
    }
    RCLCPP_INFO(node_->get_logger(), "RB10 CMD: %s", text.c_str());
  }

  void publish_debug_joint_states(
    const std::array<double, 6> & reference_position_rad,
    const std::array<double, 6> & measured_position_rad)
  {
    const auto stamp = node_->now();

    if (reference_joint_state_publisher_) {
      sensor_msgs::msg::JointState ref_msg;
      ref_msg.header.stamp = stamp;
      ref_msg.name.assign(RB10Model::joint_names.begin(), RB10Model::joint_names.end());
      ref_msg.position.assign(reference_position_rad.begin(), reference_position_rad.end());
      reference_joint_state_publisher_->publish(ref_msg);
    }

    if (measured_joint_state_publisher_) {
      sensor_msgs::msg::JointState measured_msg;
      measured_msg.header.stamp = stamp;
      measured_msg.name.assign(RB10Model::joint_names.begin(), RB10Model::joint_names.end());
      measured_msg.position.assign(measured_position_rad.begin(), measured_position_rad.end());
      measured_joint_state_publisher_->publish(measured_msg);
    }

    if (tracking_error_publisher_) {
      std_msgs::msg::Float64MultiArray tracking_error;
      tracking_error.data.resize(6U);
      for (std::size_t index = 0; index < 6U; ++index) {
        tracking_error.data[index] =
          radians_to_degrees(reference_position_rad[index] - measured_position_rad[index]);
      }
      tracking_error_publisher_->publish(tracking_error);
    }
  }

  const std::array<float, 6> & select_joint_source_deg(const Rb10SystemState & state) const
  {
    if (simulation_mode_) {
      return state.joint_ref_deg;
    }
    if (real_joint_state_source_ == "reference") {
      return state.joint_ref_deg;
    }
    return state.joint_ang_deg;
  }

  rclcpp::Node * node_;
  realtime_tools::RealtimeBuffer<RobotState> state_buffer_;
  Rb10SocketClient socket_client_;
  bool simulation_mode_{false};
  std::string real_joint_state_source_{"measured"};
  bool publish_debug_joint_state_sources_{false};
  CommandMode command_mode_{CommandMode::kPosition};
  bool hardware_command_enabled_{true};
  bool collect_timing_debug_{false};
  ControllerPacketSampleClock controller_packet_sample_clock_;
  bool socket_realtime_enabled_{false};
  int socket_realtime_priority_{60};
  double servo_t1_{0.002};
  double servo_t2_{0.1};
  double servo_gain_{0.02};
  double servo_alpha_{0.2};
  double speedj_t1_{0.02};
  double speedj_t2_{0.2};
  double speedj_gain_{0.05};
  double speedj_alpha_{0.1};
  bool use_synced_input_velocity_filter_{false};
  bool startup_move_to_default_pose_{true};
  std::array<double, 6> startup_home_joints_deg_{};
  double startup_movej_speed_{20.0};
  double startup_movej_accel_{20.0};
  double startup_release_tolerance_deg_{2.0};
  double startup_release_timeout_sec_{12.0};
  std::atomic<bool> startup_waiting_for_home_{false};
  std::chrono::steady_clock::time_point startup_release_deadline_{};
  bool stop_on_shutdown_{true};
  std::string shutdown_action_{"halt"};
  std::atomic<bool> state_received_{false};
  std::atomic<bool> command_blocked_due_to_safety_{false};
  std::atomic<std::int64_t> previous_state_time_ns_{0};
  std::atomic<std::int64_t> state_interval_latest_ns_{0};
  std::atomic<std::int64_t> state_interval_total_ns_{0};
  std::atomic<std::int64_t> state_interval_max_ns_{0};
  std::atomic<std::int64_t> state_interval_window_max_ns_{0};
  std::atomic<std::uint64_t> state_interval_count_{0U};
  std::atomic<std::uint64_t> state_gap_over_5ms_count_{0U};
  std::atomic<std::uint64_t> state_gap_over_20ms_count_{0U};
  std::atomic<std::uint64_t> state_gap_over_100ms_count_{0U};
  std::atomic<std::uint64_t> velocity_filter_update_count_{0U};
  std::atomic<std::uint64_t> velocity_filter_input_reject_count_{0U};
  std::atomic<std::uint64_t> velocity_filter_output_reject_count_{0U};
  std::atomic<std::int64_t> velocity_filter_synced_dt_window_max_ns_{0};
  std::atomic<std::uint64_t> controller_packet_time_used_count_{0U};
  std::atomic<std::uint64_t> controller_packet_time_fallback_count_{0U};
  std::atomic<std::uint64_t> controller_packet_time_duplicate_count_{0U};
  std::atomic<std::uint64_t> controller_packet_time_reset_count_{0U};
  std::atomic<std::int64_t> controller_packet_time_dt_window_max_ns_{0};
  std::array<std::atomic<std::int64_t>, 6> raw_position_step_abs_window_max_nano_{};
  std::array<std::atomic<std::int64_t>, 6> raw_velocity_abs_window_max_nano_{};
  std::array<std::atomic<std::int64_t>, 6>
  velocity_filter_residual_abs_window_max_nano_{};
  std::array<std::atomic<std::int64_t>, 6> filtered_velocity_abs_window_max_nano_{};
  realtime_tools::RealtimeBox<std::array<double, 6>> latest_servo_joint_deg_;
  std::mutex servo_command_wait_mutex_;
  std::condition_variable servo_command_cv_;
  std::thread servo_command_thread_;
  bool servo_command_pending_{false};
  bool servo_command_thread_running_{false};
  std::unique_ptr<SynchronizedVelocityFilter> synced_input_velocity_filter_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr reference_joint_state_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr measured_joint_state_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr tracking_error_publisher_;
};

bool enable_realtime(
  rclcpp::Logger logger,
  bool enabled,
  int priority,
  bool lock_memory)
{
  if (!enabled) {
    return false;
  }

  if (lock_memory && mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    RCLCPP_WARN(logger, "mlockall failed; continuing without memory locking");
  }

  sched_param params{};
  params.sched_priority = priority;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &params) != 0) {
    RCLCPP_WARN(logger, "Failed to switch control thread to SCHED_FIFO priority %d", priority);
    return false;
  }

  RCLCPP_INFO(logger, "Control thread running with SCHED_FIFO priority %d", priority);
  return true;
}

class RmpflowControllerNode : public rclcpp::Node
{
public:
  RmpflowControllerNode()
  : Node("rmpflow_controller"), running_(true)
  {
    declare_parameter("control_rate", 100.0);
    declare_parameter("publish_control_timing_debug", false);
    declare_parameter("control_timing_debug_topic", "/rmp_control_timing");
    declare_parameter("control_timing_publish_rate", 10.0);
    declare_parameter("visualization_rate", 20.0);
    declare_parameter("goal_x", 0.6);
    declare_parameter("goal_y", -0.4);
    declare_parameter("goal_z", 0.6);
    declare_parameter("min_goal_z", 0.05);
    declare_parameter("initialize_goal_from_first_state", true);
    declare_parameter("safety_stop_on_min_z", true);
    declare_parameter("workspace_floor_z", 0.0);
    declare_parameter("min_link6_z", 0.03);
    declare_parameter("min_tcp_z", 0.05);
    declare_parameter("body_goal", std::vector<double>{0.45, 0.0, 0.9});
    declare_parameter("orientation_goal", std::vector<double>{0.0, 0.0, 1.0});
    declare_parameter("initial_q", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    declare_parameter("default_q", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    declare_parameter(
      "joint_lower_limits_deg",
      std::vector<double>{-180.0, -180.0, -180.0, -180.0, -180.0, -180.0});
    declare_parameter(
      "joint_upper_limits_deg",
      std::vector<double>{180.0, 180.0, 180.0, 180.0, 180.0, 180.0});
    declare_parameter("joint_limit_buffers", std::vector<double>{0.01, 0.01, 0.01, 0.01, 0.01, 0.01});
    declare_parameter("backend_mode", "simulation");
    declare_parameter("command_mode", "velocity");
    declare_parameter("hardware_command_enabled", true);
    declare_parameter("dry_run_command_topic", "/rmp_dry_run_command");
    declare_parameter("robot_ip", "192.168.111.50");
    declare_parameter("simulation_mode", false);
    declare_parameter("real_joint_state_source", "measured");
    declare_parameter("publish_debug_joint_state_sources", false);
    declare_parameter("hardware_data_request_rate", 500.0);
    declare_parameter("use_synced_input_velocity_filter", false);
    declare_parameter("synced_input_velocity_filter_alpha", 0.35);
    declare_parameter("synced_input_velocity_filter_beta", 0.02);
    declare_parameter("synced_input_velocity_filter_type", std::string("alpha-beta"));
    declare_parameter("synced_input_velocity_ratio_tolerance", 0.05);
    declare_parameter("synced_input_velocity_max_raw_rad_s", 6.0);
    declare_parameter("synced_input_velocity_position_step_margin_rad", 0.01);
    declare_parameter("synced_input_velocity_max_output_rad_s", 2.0);
    declare_parameter("hardware_state_topic", "hardware_joint_states");
    declare_parameter("hardware_command_topic", "hardware_joint_command");
    declare_parameter("joint_state_topic", "joint_states");
    declare_parameter("rmp_flag_gate_enabled", false);
    declare_parameter("rmp_flag_topic", "/RMP_flag");
    declare_parameter("rmp_active_flag_value", 1);
    declare_parameter("position_command_topic", "position_controllers/commands");
    declare_parameter("publish_position_command", true);
    declare_parameter("position_command_state_topic", "/rmp_position_command");
    declare_parameter("publish_target_q", false);
    declare_parameter("target_q_topic", "/target_q");
    declare_parameter("publish_target_metric", false);
    declare_parameter("target_metric_topic", "/target_metric");
    declare_parameter("publish_rmp_accel_debug", false);
    declare_parameter("rmp_joint_accel_topic", "/rmp_joint_accel");
    declare_parameter("rmp_tcp_accel_topic", "/rmp_tcp_accel");
    declare_parameter("publish_joint_states", true);
    declare_parameter("joint_state_publish_topic", "joint_states");
    declare_parameter("publish_visualization", true);
    declare_parameter("publish_goal_marker", true);
    declare_parameter("publish_control_point_markers", true);
    declare_parameter("publish_body_obstacle_markers", true);
    declare_parameter("publish_end_effector_pose", true);
    declare_parameter("publish_debug_state", true);
    declare_parameter("publish_repulsion_metric_markers", true);
    declare_parameter("control_point_marker_diameter", 0.10);
    declare_parameter("repulsion_metric_marker_topic", "repulsion_metric_markers");
    declare_parameter("repulsion_metric_marker_min_norm", 0.01);
    declare_parameter("repulsion_metric_marker_dot_diameter", 0.04);
    declare_parameter("publish_tcp_accel_marker", true);
    declare_parameter("tcp_accel_marker_topic", "tcp_accel_marker");
    declare_parameter("tcp_accel_marker_max_length", 0.15);
    declare_parameter("tcp_accel_marker_norm_for_max_length", 2.0);
    declare_parameter("tcp_accel_marker_min_norm", 0.001);
    declare_parameter("enable_tangent_escape_rmp", false);
    declare_parameter(
      "tangent_escape_rmp_acceleration_model",
      std::string("canonical_velocity"));
    declare_parameter("tangent_escape_rmp_metric_scalar", 50000.0);
    declare_parameter("tangent_escape_rmp_clearance_margin", 0.0);
    declare_parameter("tangent_escape_rmp_velocity_gain", 20.0);
    declare_parameter("tangent_escape_rmp_max_accel", 0.6);
    declare_parameter("tangent_escape_rmp_goal_block_beta_on", 0.5);
    declare_parameter("tangent_escape_rmp_goal_block_beta_full", 0.94);
    declare_parameter("tangent_escape_rmp_nominal_prediction_dt", 0.01);
    declare_parameter("tangent_escape_rmp_normal_tolerance", 0.20);
    declare_parameter("tangent_escape_rmp_blocked_memory_penalty_weight", 2.0);
    declare_parameter("tangent_escape_rmp_prevent_weight", 0.5);
    declare_parameter("tangent_escape_rmp_activation_time_constant", 0.15);
    declare_parameter("tangent_escape_rmp_activation_rise_rate", 4.0);
    declare_parameter("tangent_escape_rmp_activation_fall_rate", 2.0);
    declare_parameter("tangent_escape_rmp_progress_filter_time_constant", 0.25);
    declare_parameter("tangent_escape_rmp_progress_low_threshold", 0.001);
    declare_parameter("tangent_escape_rmp_progress_ok_threshold", 0.01);
    declare_parameter("tangent_escape_rmp_still_speed_threshold", 0.003);
    declare_parameter("tangent_escape_rmp_moving_speed_threshold", 0.02);
    declare_parameter("tangent_escape_rmp_intent_on_speed", 0.005);
    declare_parameter("tangent_escape_rmp_intent_full_speed", 0.03);
    declare_parameter("tangent_escape_rmp_prevent_speed", 0.02);
    declare_parameter("tangent_escape_rmp_recovery_speed", 0.06);
    declare_parameter("tangent_escape_rmp_max_speed", 0.06);
    declare_parameter("tangent_escape_rmp_desired_velocity_time_constant", 0.12);
    declare_parameter("tangent_escape_rmp_release_stop_speed", 0.003);
    declare_parameter("tangent_escape_rmp_release_hold_speed", 0.02);
    declare_parameter("tangent_escape_rmp_drive_ramp_duration", 0.15);
    declare_parameter("tangent_escape_rmp_handoff_duration", 0.15);
    declare_parameter("tangent_escape_rmp_minimum_drive_duration", 0.2);
    declare_parameter("tangent_escape_rmp_maximum_active_duration", 0.0);
    declare_parameter("tangent_escape_rmp_timeout_reentry_cooldown", 1.0);
    declare_parameter("tangent_escape_rmp_release_blockage_threshold", 0.03);
    declare_parameter("tangent_escape_rmp_pair_switch_margin", 0.10);
    declare_parameter("tangent_escape_rmp_direction_switch_margin", 0.08);
    declare_parameter("tangent_escape_rmp_goal_score_scale", 0.02);
    declare_parameter("tangent_escape_rmp_sector_risk_scale", 1.0);
    declare_parameter("tangent_escape_rmp_sector_risk_hard_limit", 0.98);
    declare_parameter("tangent_escape_rmp_sector_risk_change_threshold", 0.10);
    declare_parameter("tangent_escape_rmp_accel_jump_weight", 0.20);
    declare_parameter("tangent_escape_rmp_accel_jump_scale", 10.0);
    declare_parameter("tangent_escape_rmp_blocked_memory_sigma", 0.35);
    declare_parameter("tangent_escape_rmp_candidate_min_displacement", 1e-5);
    declare_parameter("tangent_escape_rmp_refinement_iterations", 5);
    declare_parameter("tangent_escape_rmp_command_test_distance", 0.01);
    declare_parameter("tangent_escape_rmp_minimum_move_ratio", 0.20);
    declare_parameter("tangent_escape_rmp_goal_change_reset_distance", 0.02);
    declare_parameter("tangent_escape_rmp_risk_distance_gain", 0.30);
    declare_parameter("tangent_escape_rmp_risk_distance_scale", 0.08);
    declare_parameter("tangent_escape_rmp_risk_approach_gain", 0.50);
    declare_parameter("tangent_escape_rmp_risk_approach_distance_scale", 0.08);
    declare_parameter("tangent_escape_rmp_risk_approach_epsilon", 0.25);
    declare_parameter("tangent_escape_rmp_risk_velocity_gate_scale", 0.03);
    declare_parameter(
      "tangent_escape_rmp_risk_clearance_rate_filter_time_constant", 0.05);
    declare_parameter("tangent_escape_rmp_risk_tangent_damping_gain", 10.0);
    declare_parameter("tangent_escape_rmp_risk_stuck_accel_floor", 0.20);
    declare_parameter("tangent_escape_rmp_risk_stuck_dwell_time", 0.30);
    declare_parameter("tangent_escape_rmp_risk_stuck_ramp_time", 0.30);
    // "tangent_plane" model only.  Non-positive distances inherit the gate one
    // level up: push inherits safe/influence, metric inherits push.
    // Escape RMP v2 ("tangent_ode"); ignored by the other three models.
    declare_parameter("tangent_escape_ode_metric_max", 100000.0);
    declare_parameter("tangent_escape_ode_metric_sigma", 0.03);
    declare_parameter("tangent_escape_ode_accel_max", 50.0);
    declare_parameter("tangent_escape_ode_accel_sigma", 0.08);
    declare_parameter("tangent_escape_ode_budget_scale", 1.0);
    declare_parameter("tangent_escape_ode_budget_recovery_time", 3.0);
    declare_parameter("tangent_escape_ode_tangent_damping", 40.0);
    declare_parameter("tangent_escape_ode_direction_rate", 3.0);
    declare_parameter("tangent_escape_ode_risk_sigma", 0.06);
    declare_parameter("tangent_escape_ode_weight_goal", 1.0);
    declare_parameter("tangent_escape_ode_weight_velocity", 0.3);
    declare_parameter("tangent_escape_ode_weight_risk", 1.0);
    declare_parameter("tangent_escape_ode_weight_curvature", 1.5);
    declare_parameter("tangent_escape_ode_velocity_epsilon", 0.0001);
    declare_parameter("tangent_escape_ode_axial_preference", 1.0);
    declare_parameter("tangent_escape_ode_pocket_threshold", 0.6);
    declare_parameter("tangent_escape_ode_pocket_epsilon", 0.1);
    declare_parameter("tangent_escape_ode_activation_distance", 0.3);
    declare_parameter("tangent_escape_ode_release_distance", 0.4);
    declare_parameter("tangent_escape_ode_switch_hysteresis", 0.005);
    declare_parameter("tangent_escape_ode_weight_stall", 2.0);
    declare_parameter("tangent_escape_ode_stall_time_constant", 1.5);
    declare_parameter("tangent_escape_ode_stall_speed", 0.02);
    declare_parameter("tangent_escape_rmp_metric_isotropy", 1.0);
    declare_parameter("tangent_escape_rmp_push_safe_distance", 0.0);
    declare_parameter("tangent_escape_rmp_push_influence_distance", 0.0);
    declare_parameter("tangent_escape_rmp_metric_safe_distance", 0.0);
    declare_parameter("tangent_escape_rmp_metric_influence_distance", 0.0);
    declare_parameter("tangent_escape_rmp_safe_distance", 0.08);
    declare_parameter("tangent_escape_rmp_influence_distance", 0.20);
    declare_parameter("tangent_escape_rmp_candidate_lookahead", 0.08);
    declare_parameter("tangent_escape_rmp_goal_weight", 2.0);
    declare_parameter("tangent_escape_rmp_continuity_weight", 0.3);
    declare_parameter("tangent_escape_rmp_sector_risk_weight", 1.5);
    declare_parameter("tangent_escape_rmp_min_activation", 0.05);
    declare_parameter("tangent_escape_rmp_min_tangent_norm", 1e-4);
    declare_parameter("publish_tangent_escape_rmp_data", false);
    declare_parameter("tangent_escape_rmp_data_topic", "/tangent_escape_rmp_data");
    declare_parameter("publish_tangent_escape_dual_solve_data", false);
    declare_parameter(
      "tangent_escape_dual_solve_topic",
      "/tangent_escape_dual_solve");
    declare_parameter("publish_leaf_ablation_data", false);
    declare_parameter("leaf_ablation_topic", "/rmp_leaf_ablation");
    declare_parameter("publish_tangent_escape_geometry_debug", false);
    declare_parameter("tangent_escape_geometry_data_topic", "/tangent_escape_geometry_data");
    declare_parameter("tangent_escape_geometry_marker_topic", "tangent_escape_geometry_markers");
    declare_parameter("tangent_escape_geometry_marker_length", 0.12);
    declare_parameter("publish_rmp_ee_pose", false);
    declare_parameter("publish_goal_tf", true);
    declare_parameter("goal_tf_parent_frame", "base_link");
    declare_parameter("goal_tf_child_frame", "rmp_goal_target");
    declare_parameter("servo_t1", 0.002);
    declare_parameter("servo_t2", 0.1);
    declare_parameter("servo_gain", 0.02);
    declare_parameter("servo_alpha", 0.2);
    declare_parameter("speedj_t1", 0.02);
    declare_parameter("speedj_t2", 0.2);
    declare_parameter("speedj_gain", 0.05);
    declare_parameter("speedj_alpha", 0.1);
    declare_parameter("startup_move_to_default_pose", true);
    declare_parameter(
      "startup_home_joints_deg",
      std::vector<double>{88.82315826, 1.57005262, -108.45492554, 16.88487434, -89.99609375, 1.24207485});
    declare_parameter("startup_movej_speed", 20.0);
    declare_parameter("startup_movej_accel", 20.0);
    declare_parameter("startup_release_tolerance_deg", 2.0);
    declare_parameter("startup_release_timeout_sec", 12.0);
    declare_parameter("stop_on_shutdown", true);
    declare_parameter("shutdown_action", "halt");
    declare_parameter("enable_socket_realtime", false);
    declare_parameter("socket_realtime_priority", 60);
    declare_parameter("enable_realtime", false);
    declare_parameter("realtime_priority", 80);
    declare_parameter("lock_memory", false);
    declare_parameter("graph.node_names", default_rmp_graph_node_names());
    declare_parameter("root_solve_offset", 1e-3);
    declare_parameter("solver_backend", "pinocchio_direct");
    declare_parameter("solve_method", "rmp2");
    declare_parameter("rmp_type", "canonical");
    declare_parameter("casadi_rmp2_max_obstacles", 24);
    declare_parameter("casadi_rmp2_jit", true);
    declare_parameter("robot_urdf_path", "");
    declare_parameter(
      "external_rmp.topic_prefix",
      std::string("external_rmp"));
    declare_parameter(
      "body_obstacles.names",
      rclcpp::ParameterValue(std::vector<std::string>{}));
    declare_parameter("cspace_target_metric_scalar", 0.005);
    declare_parameter("cspace_target_position_gain", 100.0);
    declare_parameter("cspace_target_damping_gain", 50.0);
    declare_parameter("cspace_target_robust_position_term_thresh", 0.5);
    declare_parameter("cspace_target_inertia", 0.0001);
    declare_parameter("joint_limit_metric_scalar", 0.1);
    declare_parameter("joint_limit_policy", "lula_canonical");
    declare_parameter("joint_limit_metric_length_scale", 0.01);
    declare_parameter("joint_limit_metric_exploder_eps", 0.001);
    declare_parameter("joint_limit_metric_velocity_gate_length_scale", 0.01);
    declare_parameter("joint_limit_accel_damper_gain", 200.0);
    declare_parameter("joint_limit_accel_potential_gain", 1.0);
    declare_parameter("joint_limit_accel_potential_exploder_eps", 0.01);
    declare_parameter("joint_limit_accel_potential_exploder_length_scale", 0.1);
    declare_parameter("joint_limit_gds_center_fraction", 0.5);
    declare_parameter("joint_limit_gds_domain_epsilon", 1e-9);
    declare_parameter("joint_velocity_cap_max_velocity", 1.7);
    declare_parameter("joint_velocity_cap_velocity_damping_region", 0.15);
    declare_parameter("joint_velocity_cap_damping_gain", 5.0);
    declare_parameter("joint_velocity_cap_metric_weight", 0.05);
    declare_parameter("max_joint_accel", 20.0);
    declare_parameter("preserve_joint_accel_direction", true);
    declare_parameter("measured_position_feedback_blend", 1.0);
    declare_parameter("measured_velocity_feedback_blend", 0.35);
    declare_parameter("estimate_velocity_in_controller", false);
    declare_parameter("controller_velocity_filter_alpha", 0.25);
    declare_parameter("use_velocity_feedback_in_solver", true);
    declare_parameter("target_rmp_accel_p_gain", 50.0);
    declare_parameter("target_rmp_accel_d_gain", 70.0);
    declare_parameter("command_guard_max_step_rad", 0.00436332313);
    declare_parameter("command_guard_max_velocity_rad_s", 0.436332313);
    declare_parameter("command_guard_velocity_slew_enabled", true);
    declare_parameter("command_guard_max_accel_rad_s2", 10.0);
    declare_parameter("predictive_joint_limit_guard", true);
    declare_parameter("target_rmp_accel_norm_eps", 0.075);
    declare_parameter("target_rmp_metric_alpha_length_scale", 0.05);
    declare_parameter("target_rmp_min_metric_alpha", 0.03);
    declare_parameter("target_rmp_max_metric_scalar", 1.0);
    declare_parameter("target_rmp_min_metric_scalar", 0.5);
    declare_parameter("target_rmp_proximity_metric_boost_scalar", 3.0);
    declare_parameter("target_rmp_proximity_metric_boost_length_scale", 0.02);
    declare_parameter("axis_target_rmp_accel_p_gain", 1000.0);
    declare_parameter("axis_target_rmp_accel_d_gain", 500.0);
    declare_parameter("axis_target_rmp_metric_scalar", 50.0);
    declare_parameter("axis_target_rmp_proximity_metric_boost_scalar", 10.0);
    declare_parameter("axis_target_rmp_proximity_metric_boost_length_scale", 0.1);
    declare_parameter("wrist_axis_target_rmp_accel_p_gain", 50.0);
    declare_parameter("wrist_axis_target_rmp_accel_d_gain", 50.0);
    declare_parameter("wrist_axis_target_rmp_metric_scalar", 1000.0);
    declare_parameter("wrist_axis_target_rmp_proximity_metric_boost_scalar", 1.0);
    declare_parameter("wrist_axis_target_rmp_proximity_metric_boost_length_scale", 0.01);
    declare_parameter("collision_policy", "lula_canonical");
    declare_parameter("collision_rmp_margin", 0.0);
    declare_parameter("collision_rmp_damping_gain", 50.0);
    declare_parameter("collision_rmp_damping_std_dev", 0.04);
    declare_parameter("collision_rmp_damping_robustness_eps", 0.01);
    declare_parameter("collision_rmp_damping_velocity_gate_length_scale", 0.01);
    declare_parameter("collision_rmp_repulsion_gain", 800.0);
    declare_parameter("collision_rmp_repulsion_std_dev", 0.01);
    declare_parameter("collision_rmp_metric_modulation_radius", 0.5);
    declare_parameter("collision_rmp_metric_scalar", 1.0);
    declare_parameter("collision_rmp_metric_exploder_std_dev", 0.02);
    declare_parameter("collision_rmp_metric_exploder_eps", 0.001);
    declare_parameter("collision_rmp_gds_clearance_smoothing", 1e-4);
    declare_parameter("collision_rmp_gds_metric_velocity_floor", 0.5);
    declare_parameter("collision_rmp_gds_metric_velocity_scale", 0.03);
    declare_parameter("collision_rmp_gds_damping_velocity_scale", 0.03);
    declare_parameter("escape_stability_guard_enabled", false);
    declare_parameter("escape_stability_tank_capacity", 0.25);
    declare_parameter("escape_stability_scale_search_iterations", 16);
    declare_parameter("escape_stability_power_tolerance", 1e-9);
    declare_parameter("publish_stability_certificate_data", false);
    declare_parameter(
      "stability_certificate_data_topic",
      "/rmp_stability_certificate");
    declare_parameter("run_production_stability_analysis", false);
    declare_parameter("shutdown_after_production_stability_analysis", false);
    declare_parameter("production_stability_wait_for_external_goal", false);
    declare_parameter(
      "production_stability_output_directory",
      "~/ros2_ws/log/rmpflow_production_stability");
    declare_parameter("production_stability_equilibrium_max_iterations", 80);
    declare_parameter("production_stability_equilibrium_tolerance", 1e-7);
    declare_parameter("production_stability_settle_max_steps", 20000);
    declare_parameter("production_stability_settle_velocity_tolerance", 1e-6);
    declare_parameter("production_stability_settle_acceleration_tolerance", 1e-5);
    declare_parameter("production_stability_position_difference_step", 1e-5);
    declare_parameter("production_stability_velocity_difference_step", 1e-5);
    declare_parameter("production_stability_random_directions_per_radius", 64);
    declare_parameter(
      "production_stability_sample_radii",
      std::vector<double>{1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2});
    declare_parameter("damping_rmp_accel_d_gain", 120.0);
    declare_parameter("damping_rmp_metric_scalar", 0.0);
    declare_parameter("damping_rmp_inertia", 300.0);

    const auto initial_q = get_parameter("initial_q").as_double_array();
    state_.q = JointVector::Zero();
    state_.qd = JointVector::Zero();
    for (std::size_t index = 0; index < std::min<std::size_t>(initial_q.size(), 6); ++index) {
      state_.q[static_cast<int>(index)] = initial_q[index];
    }
    const auto initial_context = RB10Model::forward_context(
      state_.q, ControlPointJacobianMode::kNone);
    GoalTarget initial_goal_target;
    initial_goal_target.position = Eigen::Vector3d(
      get_parameter("goal_x").as_double(),
      get_parameter("goal_y").as_double(),
      get_parameter("goal_z").as_double());
    initialize_goal_from_first_state_ =
      get_parameter("initialize_goal_from_first_state").as_bool();
    min_goal_z_ = get_parameter("min_goal_z").as_double();
    safety_stop_on_min_z_ = get_parameter("safety_stop_on_min_z").as_bool();
    workspace_floor_z_ = get_parameter("workspace_floor_z").as_double();
    min_link6_z_ = get_parameter("min_link6_z").as_double();
    min_tcp_z_ = get_parameter("min_tcp_z").as_double();
    control_rate_hz_ = get_parameter("control_rate").as_double();
    if (!std::isfinite(control_rate_hz_) || control_rate_hz_ <= 0.0) {
      throw std::invalid_argument("control_rate must be finite and greater than zero");
    }
    publish_control_timing_debug_enabled_ =
      get_parameter("publish_control_timing_debug").as_bool();
    control_timing_publish_rate_hz_ =
      get_parameter("control_timing_publish_rate").as_double();
    if (
      !std::isfinite(control_timing_publish_rate_hz_) ||
      control_timing_publish_rate_hz_ <= 0.0)
    {
      throw std::invalid_argument(
              "control_timing_publish_rate must be finite and greater than zero");
    }
    enable_realtime_ = get_parameter("enable_realtime").as_bool();
    realtime_priority_ = get_parameter("realtime_priority").as_int();
    lock_memory_ = get_parameter("lock_memory").as_bool();
    const bool use_synced_input_velocity_filter =
      get_parameter("use_synced_input_velocity_filter").as_bool();
    estimate_velocity_in_controller_ =
      get_parameter("estimate_velocity_in_controller").as_bool();
    measured_position_feedback_blend_ = std::clamp(
      get_parameter("measured_position_feedback_blend").as_double(), 0.0, 1.0);
    measured_velocity_feedback_blend_ = std::clamp(
      get_parameter("measured_velocity_feedback_blend").as_double(), 0.0, 1.0);
    use_velocity_feedback_in_solver_ =
      get_parameter("use_velocity_feedback_in_solver").as_bool();
    controller_velocity_filter_alpha_ = std::clamp(
      get_parameter("controller_velocity_filter_alpha").as_double(), 0.0, 1.0);
    max_joint_accel_ = get_parameter("max_joint_accel").as_double();
    preserve_joint_accel_direction_ =
      get_parameter("preserve_joint_accel_direction").as_bool();
    command_mode_ = parse_command_mode(get_parameter("command_mode").as_string());
    hardware_command_enabled_ = get_parameter("hardware_command_enabled").as_bool();
    command_guard_max_step_rad_ = std::max(
      get_parameter("command_guard_max_step_rad").as_double(),
      1e-4);
    const double configured_max_velocity =
      get_parameter("command_guard_max_velocity_rad_s").as_double();
    command_guard_max_velocity_rad_s_ = std::max(
      configured_max_velocity,
      command_mode_ == CommandMode::kVelocity ? 1e-4 : command_guard_max_step_rad_ * control_rate_hz_);
    command_guard_velocity_slew_enabled_ =
      get_parameter("command_guard_velocity_slew_enabled").as_bool();
    command_guard_max_accel_rad_s2_ = std::max(
      get_parameter("command_guard_max_accel_rad_s2").as_double(),
      1e-4);
    predictive_joint_limit_guard_ = get_parameter("predictive_joint_limit_guard").as_bool();
    joint_lower_limits_ = parse_joint_limit_degrees(
      "joint_lower_limits_deg",
      RB10Model::joint_lower_limits);
    joint_upper_limits_ = parse_joint_limit_degrees(
      "joint_upper_limits_deg",
      RB10Model::joint_upper_limits);
    for (std::size_t index = 0; index < joint_lower_limits_.size(); ++index) {
      if (joint_lower_limits_[index] >= joint_upper_limits_[index]) {
        throw std::runtime_error(
                "joint_lower_limits_deg must be less than joint_upper_limits_deg for every joint");
      }
    }
    const auto joint_limit_buffers = get_parameter("joint_limit_buffers").as_double_array();
    for (std::size_t index = 0; index < std::min<std::size_t>(joint_limit_buffers.size(), 6); ++index) {
      joint_limit_buffers_[index] = std::max(0.0, joint_limit_buffers[index]);
    }
    if (use_synced_input_velocity_filter && estimate_velocity_in_controller_) {
      RCLCPP_WARN(
        get_logger(),
        "use_synced_input_velocity_filter is enabled but estimate_velocity_in_controller is also true; the controller-side finite-difference estimator will override the synced backend velocity unless you set estimate_velocity_in_controller:=false");
    }
    body_goal_ = parse_vector3_parameter("body_goal", Eigen::Vector3d(0.45, 0.0, 0.9));
    initial_goal_target.orientation =
      Eigen::Quaterniond(initial_context.link_rotations[RB10Model::TCP_RMP]);
    initial_goal_target.orientation.normalize();
    goal_target_box_.set(initial_goal_target);
    declare_graph_parameters();
    declare_body_obstacle_parameters();
    obstacles_box_.set(std::vector<ObstacleSphere>{ObstacleSphere{}});
    proximity_sensor_obstacles_box_.set(
      std::vector<std::optional<ObstacleSphere>>(RB10Model::sensor_control_points.size()));
    tcp_accel_visualization_box_.set(TcpAccelerationSample{});
    tangent_escape_visualization_box_.set(std::vector<double>{0.0});
    tangent_escape_dual_visualization_box_.set(std::vector<double>{0.0});
    const auto solver_config = build_solver_config();
    target_metric_params_ = solver_config.target;
    collision_metric_params_ = solver_config.collision;
    configure_external_rmp_inputs(solver_config.graph_nodes);
    solver_ = build_solver(solver_config);
    body_obstacles_visual_ = solver_config.body_obstacles;
    tangent_escape_rmp_enabled_ = solver_config.tangent_escape.enabled;
    production_stability_analysis_enabled_ =
      get_parameter("run_production_stability_analysis").as_bool();

    publish_joint_states_enabled_ = get_parameter("publish_joint_states").as_bool();
    publish_visualization_enabled_ = get_parameter("publish_visualization").as_bool();
    publish_goal_marker_enabled_ =
      get_parameter("publish_goal_marker").as_bool();
    publish_control_point_markers_enabled_ =
      get_parameter("publish_control_point_markers").as_bool();
    publish_body_obstacle_markers_enabled_ =
      get_parameter("publish_body_obstacle_markers").as_bool();
    publish_end_effector_pose_enabled_ =
      get_parameter("publish_end_effector_pose").as_bool();
    publish_debug_state_enabled_ =
      get_parameter("publish_debug_state").as_bool();
    publish_repulsion_metric_markers_enabled_ =
      get_parameter("publish_repulsion_metric_markers").as_bool();
    control_point_marker_diameter_ =
      get_parameter("control_point_marker_diameter").as_double();
    repulsion_metric_marker_min_norm_ = std::clamp(
      get_parameter("repulsion_metric_marker_min_norm").as_double(), 0.0, 1.0);
    repulsion_metric_marker_dot_diameter_ = std::max(
      get_parameter("repulsion_metric_marker_dot_diameter").as_double(), 0.005);
    publish_tcp_accel_marker_enabled_ =
      get_parameter("publish_tcp_accel_marker").as_bool();
    tcp_accel_marker_max_length_ = std::max(
      get_parameter("tcp_accel_marker_max_length").as_double(), 0.01);
    tcp_accel_marker_norm_for_max_length_ = std::max(
      get_parameter("tcp_accel_marker_norm_for_max_length").as_double(), 1e-9);
    tcp_accel_marker_min_norm_ = std::max(
      get_parameter("tcp_accel_marker_min_norm").as_double(), 0.0);
    publish_tangent_escape_rmp_data_enabled_ =
      get_parameter("publish_tangent_escape_rmp_data").as_bool();
    publish_tangent_escape_dual_solve_data_enabled_ =
      get_parameter("publish_tangent_escape_dual_solve_data").as_bool();
    publish_leaf_ablation_data_enabled_ =
      get_parameter("publish_leaf_ablation_data").as_bool();
    publish_stability_certificate_data_enabled_ =
      get_parameter("publish_stability_certificate_data").as_bool();
    publish_tangent_escape_geometry_debug_enabled_ =
      get_parameter("publish_tangent_escape_geometry_debug").as_bool();
    tangent_escape_geometry_marker_length_ = std::max(
      get_parameter("tangent_escape_geometry_marker_length").as_double(), 0.01);
    publish_rmp_ee_pose_enabled_ = get_parameter("publish_rmp_ee_pose").as_bool();
    publish_goal_tf_enabled_ = get_parameter("publish_goal_tf").as_bool();
    goal_tf_parent_frame_ = get_parameter("goal_tf_parent_frame").as_string();
    goal_tf_child_frame_ = get_parameter("goal_tf_child_frame").as_string();
    joint_state_topic_ = get_parameter("joint_state_topic").as_string();
    rmp_flag_gate_enabled_ = get_parameter("rmp_flag_gate_enabled").as_bool();
    rmp_active_flag_value_ = static_cast<int>(get_parameter("rmp_active_flag_value").as_int());
    rmp_active_.store(!rmp_flag_gate_enabled_);
    if (publish_joint_states_enabled_) {
      joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>(
        get_parameter("joint_state_publish_topic").as_string(),
        10);
      rt_joint_state_pub_ = std::make_shared<JointStateRtPublisher>(joint_state_pub_);
    }
    const auto configured_backend_mode = get_parameter("backend_mode").as_string();
    if (configured_backend_mode == "rb10_direct_api" && hardware_command_enabled_) {
      direct_command_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("position_command_topic").as_string(), 10);
      rt_direct_command_pub_ = std::make_shared<Float64ArrayRtPublisher>(direct_command_pub_);
    } else if (!hardware_command_enabled_) {
      direct_command_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("dry_run_command_topic").as_string(), 10);
      rt_direct_command_pub_ = std::make_shared<Float64ArrayRtPublisher>(direct_command_pub_);
      if (configured_backend_mode == "simulation") {
        RCLCPP_INFO(
          get_logger(),
          "LOCAL SIMULATION ONLY: calculated %s commands cannot reach RB10 hardware",
          command_mode_to_string(command_mode_));
      } else {
        RCLCPP_WARN(
          get_logger(),
          "HARDWARE COMMANDS DISABLED: calculated %s commands are published only to %s",
          command_mode_to_string(command_mode_),
          get_parameter("dry_run_command_topic").as_string().c_str());
      }
    }
    if (get_parameter("publish_position_command").as_bool()) {
      position_command_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("position_command_state_topic").as_string(), 10);
      rt_position_command_pub_ =
        std::make_shared<Float64ArrayRtPublisher>(position_command_pub_);
    }
    if (get_parameter("publish_target_q").as_bool()) {
      target_q_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("target_q_topic").as_string(), 10);
      rt_target_q_pub_ = std::make_shared<Float64ArrayRtPublisher>(target_q_pub_);
    }
    if (get_parameter("publish_target_metric").as_bool()) {
      target_metric_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("target_metric_topic").as_string(), 10);
      rt_target_metric_pub_ = std::make_shared<Float64ArrayRtPublisher>(target_metric_pub_);
    }
    if (get_parameter("publish_rmp_accel_debug").as_bool()) {
      rmp_joint_accel_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("rmp_joint_accel_topic").as_string(), 10);
      rt_rmp_joint_accel_pub_ = std::make_shared<Float64ArrayRtPublisher>(rmp_joint_accel_pub_);
      rmp_tcp_accel_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("rmp_tcp_accel_topic").as_string(), 10);
      rt_rmp_tcp_accel_pub_ = std::make_shared<Float64ArrayRtPublisher>(rmp_tcp_accel_pub_);
    }
    if (publish_control_timing_debug_enabled_) {
      control_timing_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("control_timing_debug_topic").as_string(), 10);
      rt_control_timing_pub_ =
        std::make_shared<Float64ArrayRtPublisher>(control_timing_pub_);
    }
    if (publish_tangent_escape_rmp_data_enabled_) {
      tangent_escape_rmp_data_pub_ =
        create_publisher<std_msgs::msg::Float64MultiArray>(
          get_parameter("tangent_escape_rmp_data_topic").as_string(), 10);
      rt_tangent_escape_rmp_data_pub_ =
        std::make_shared<Float64ArrayRtPublisher>(tangent_escape_rmp_data_pub_);
    }
    if (publish_tangent_escape_dual_solve_data_enabled_) {
      tangent_escape_dual_solve_data_pub_ =
        create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("tangent_escape_dual_solve_topic").as_string(), 10);
      rt_tangent_escape_dual_solve_data_pub_ =
        std::make_shared<Float64ArrayRtPublisher>(
        tangent_escape_dual_solve_data_pub_);
    }
    if (publish_leaf_ablation_data_enabled_) {
      leaf_ablation_data_pub_ =
        create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("leaf_ablation_topic").as_string(), 10);
      rt_leaf_ablation_data_pub_ =
        std::make_shared<Float64ArrayRtPublisher>(leaf_ablation_data_pub_);
    }
    if (publish_stability_certificate_data_enabled_) {
      stability_certificate_data_pub_ =
        create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("stability_certificate_data_topic").as_string(), 10);
      rt_stability_certificate_data_pub_ =
        std::make_shared<Float64ArrayRtPublisher>(
        stability_certificate_data_pub_);
    }
    if (publish_tangent_escape_geometry_debug_enabled_) {
      tangent_escape_geometry_data_pub_ =
        create_publisher<std_msgs::msg::Float64MultiArray>(
          get_parameter("tangent_escape_geometry_data_topic").as_string(), 10);
      rt_tangent_escape_geometry_data_pub_ =
        std::make_shared<Float64ArrayRtPublisher>(tangent_escape_geometry_data_pub_);
    }
    if (publish_visualization_enabled_) {
      if (publish_goal_marker_enabled_) {
        goal_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("goal_marker", 10);
      }
      if (publish_control_point_markers_enabled_) {
        control_point_pub_ =
          create_publisher<visualization_msgs::msg::MarkerArray>("control_points", 10);
      }
      if (publish_body_obstacle_markers_enabled_) {
        body_obstacle_pub_ =
          create_publisher<visualization_msgs::msg::MarkerArray>("body_obstacle_markers", 10);
      }
      if (publish_repulsion_metric_markers_enabled_) {
        repulsion_metric_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
          get_parameter("repulsion_metric_marker_topic").as_string(), 10);
      }
      if (publish_tcp_accel_marker_enabled_) {
        tcp_accel_pub_ = create_publisher<visualization_msgs::msg::Marker>(
          get_parameter("tcp_accel_marker_topic").as_string(), 10);
      }
      if (publish_tangent_escape_geometry_debug_enabled_) {
        tangent_escape_geometry_marker_pub_ =
          create_publisher<visualization_msgs::msg::MarkerArray>(
            get_parameter("tangent_escape_geometry_marker_topic").as_string(), 10);
      }
      if (publish_end_effector_pose_enabled_) {
        eef_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
          "end_effector_pose",
          10);
      }
      if (publish_debug_state_enabled_) {
        debug_state_pub_ =
          create_publisher<std_msgs::msg::Float64MultiArray>("rmp_debug_state", 10);
      }
    }
    if (publish_rmp_ee_pose_enabled_) {
      rmp_eef_pose_pub_ = create_publisher<geometry_msgs::msg::Pose>("rmp_ee_pose", 10);
      rt_rmp_eef_pose_pub_ = std::make_shared<PoseRtPublisher>(rmp_eef_pose_pub_);
    }
    if (publish_goal_tf_enabled_) {
      goal_tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    goal_sub_ = create_subscription<geometry_msgs::msg::Point>(
      "goal_position",
      10,
      std::bind(&RmpflowControllerNode::on_goal, this, std::placeholders::_1));
    goal_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "goal_pose",
      10,
      std::bind(&RmpflowControllerNode::on_goal_pose, this, std::placeholders::_1));
    if (rmp_flag_gate_enabled_) {
      rmp_flag_sub_ = create_subscription<std_msgs::msg::UInt8>(
        get_parameter("rmp_flag_topic").as_string(),
        10,
        std::bind(&RmpflowControllerNode::on_rmp_flag, this, std::placeholders::_1));
    }
    obstacle_sub_ = create_subscription<visualization_msgs::msg::MarkerArray>(
      "obstacles",
      10,
      std::bind(&RmpflowControllerNode::on_obstacles, this, std::placeholders::_1));

    const auto backend_mode = get_parameter("backend_mode").as_string();
    const bool controller_box_simulation = get_parameter("simulation_mode").as_bool();
    if (backend_mode == "rb10_direct_api" && controller_box_simulation) {
      RCLCPP_WARN(
        get_logger(),
        "CONTROLLER BOX SIMULATION via direct C++ backend: RB10 network sockets are open, "
        "but initialization explicitly requests PG_MODE.SIMULATION");
    }
    backend_command_application_enabled_ =
      hardware_command_enabled_ || backend_mode == "simulation";
    if (backend_mode == "hardware_bridge") {
      backend_ = std::make_unique<HardwareBridgeBackend>(
        this,
        state_.q,
        get_parameter("hardware_state_topic").as_string(),
        get_parameter("hardware_command_topic").as_string());
      RCLCPP_INFO(get_logger(), "Using hardware bridge backend");
    } else if (backend_mode == "rb10_direct_api") {
      backend_ = std::make_unique<DirectRb10Backend>(
        this,
        state_.q,
        get_parameter("robot_ip").as_string(),
        get_parameter("simulation_mode").as_bool(),
        get_parameter("real_joint_state_source").as_string(),
        get_parameter("publish_debug_joint_state_sources").as_bool(),
        command_mode_,
        hardware_command_enabled_,
        publish_control_timing_debug_enabled_,
        get_parameter("hardware_data_request_rate").as_double(),
        get_parameter("control_rate").as_double(),
        get_parameter("servo_t1").as_double(),
        get_parameter("servo_t2").as_double(),
        get_parameter("servo_gain").as_double(),
        get_parameter("servo_alpha").as_double(),
        get_parameter("speedj_t1").as_double(),
        get_parameter("speedj_t2").as_double(),
        get_parameter("speedj_gain").as_double(),
        get_parameter("speedj_alpha").as_double(),
        get_parameter("use_synced_input_velocity_filter").as_bool(),
        get_parameter("synced_input_velocity_filter_alpha").as_double(),
        get_parameter("synced_input_velocity_filter_beta").as_double(),
        get_parameter("synced_input_velocity_filter_type").as_string(),
        get_parameter("synced_input_velocity_ratio_tolerance").as_double(),
        get_parameter("synced_input_velocity_max_raw_rad_s").as_double(),
        get_parameter("synced_input_velocity_position_step_margin_rad").as_double(),
        get_parameter("synced_input_velocity_max_output_rad_s").as_double(),
        get_parameter("startup_move_to_default_pose").as_bool(),
        get_parameter("startup_home_joints_deg").as_double_array(),
        get_parameter("startup_movej_speed").as_double(),
        get_parameter("startup_movej_accel").as_double(),
        get_parameter("startup_release_tolerance_deg").as_double(),
        get_parameter("startup_release_timeout_sec").as_double(),
        get_parameter("enable_socket_realtime").as_bool(),
        get_parameter("socket_realtime_priority").as_int(),
        get_parameter("stop_on_shutdown").as_bool(),
        get_parameter("shutdown_action").as_string());
      RCLCPP_INFO(
        get_logger(),
        "Using direct RB10 API backend (%s, state source: %s, command_mode=%s, commands=%s)",
        get_parameter("robot_ip").as_string().c_str(),
        get_parameter("real_joint_state_source").as_string().c_str(),
        command_mode_to_string(command_mode_),
        hardware_command_enabled_ ? "enabled" : "DISABLED (read-only dry-run)");
      if (command_mode_ == CommandMode::kVelocity) {
        RCLCPP_INFO(
          get_logger(),
          "Direct RB10 SpeedJ profile: t1=%.4f s, t2=%.4f s, gain=%.4f, alpha=%.4f",
          get_parameter("speedj_t1").as_double(),
          get_parameter("speedj_t2").as_double(),
          get_parameter("speedj_gain").as_double(),
          get_parameter("speedj_alpha").as_double());
      }
    } else if (backend_mode == "joint_command_topics") {
      backend_ = std::make_unique<JointCommandTopicsBackend>(
        this,
        state_.q,
        get_parameter("joint_state_topic").as_string(),
        get_parameter("position_command_topic").as_string(),
        command_mode_,
        publish_control_timing_debug_enabled_,
        get_parameter("hardware_data_request_rate").as_double(),
        get_parameter("control_rate").as_double(),
        get_parameter("use_synced_input_velocity_filter").as_bool(),
        get_parameter("synced_input_velocity_filter_alpha").as_double(),
        get_parameter("synced_input_velocity_filter_beta").as_double(),
        get_parameter("synced_input_velocity_filter_type").as_string(),
        get_parameter("synced_input_velocity_ratio_tolerance").as_double(),
        get_parameter("synced_input_velocity_max_raw_rad_s").as_double(),
        get_parameter("synced_input_velocity_position_step_margin_rad").as_double(),
        get_parameter("synced_input_velocity_max_output_rad_s").as_double());
      RCLCPP_INFO(
        get_logger(),
        "Using joint command topic backend (%s -> %s, command_mode=%s, velocity_filter=%s)",
        get_parameter("joint_state_topic").as_string().c_str(),
        get_parameter("position_command_topic").as_string().c_str(),
        command_mode_to_string(command_mode_),
        get_parameter("use_synced_input_velocity_filter").as_bool() ? "enabled" : "disabled");
    } else if (backend_mode == "simulation") {
      backend_ = std::make_unique<SimulationBackend>(state_.q);
      RCLCPP_INFO(get_logger(), "Using simulation backend");
    } else {
      throw std::runtime_error("Unsupported backend_mode: " + backend_mode);
    }

    if (publish_joint_states_enabled_ && backend_->ready()) {
      publish_joint_states(backend_->read_state());
    }

    if (publish_visualization_enabled_ || publish_goal_tf_enabled_) {
      const double visualization_rate = get_parameter("visualization_rate").as_double();
      if (visualization_rate > std::numeric_limits<double>::epsilon()) {
        const auto visualization_period = std::chrono::duration<double>(1.0 / visualization_rate);
        visualization_timer_ = create_wall_timer(
          std::chrono::duration_cast<std::chrono::milliseconds>(visualization_period),
          std::bind(&RmpflowControllerNode::publish_visualization, this));
      } else {
        RCLCPP_WARN(
          get_logger(),
          "publish_visualization is enabled but visualization_rate <= 0.0; skipping visualization timer");
      }
    }

    control_thread_ = std::thread(&RmpflowControllerNode::control_loop, this);

    RCLCPP_INFO(
      get_logger(),
      "C++ RMPflow controller started at %.1f Hz",
      get_parameter("control_rate").as_double());
  }

  ~RmpflowControllerNode() override
  {
    running_.store(false);
    if (control_thread_.joinable()) {
      control_thread_.join();
    }
  }

private:
  void on_rmp_flag(const std_msgs::msg::UInt8::SharedPtr msg)
  {
    const bool requested_active = static_cast<int>(msg->data) == rmp_active_flag_value_;
    const bool was_active = rmp_active_.exchange(requested_active);
    if (was_active == requested_active) {
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "Received controller /RMP_flag: %d -> active=%s",
      static_cast<int>(msg->data),
      requested_active ? "true" : "false");
    if (!requested_active) {
      virtual_velocity_state_initialized_ = false;
      last_safe_command_state_initialized_ = false;
      last_min_z_safety_triggered_.store(false);
      visualization_cleared_for_inactive_ = false;
    }
  }

  void on_goal(const geometry_msgs::msg::Point::SharedPtr msg)
  {
    GoalTarget next_goal;
    goal_target_box_.get(next_goal);
    next_goal.position = Eigen::Vector3d(
      msg->x,
      msg->y,
      std::max(msg->z, min_goal_z_));
    goal_target_box_.set(next_goal);
    external_goal_received_.store(true);
    startup_goal_synced_.store(true);
  }

  void on_goal_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    GoalTarget next_goal;
    next_goal.position = Eigen::Vector3d(
      msg->pose.position.x,
      msg->pose.position.y,
      std::max(msg->pose.position.z, min_goal_z_));
    Eigen::Quaterniond goal_orientation(
      msg->pose.orientation.w,
      msg->pose.orientation.x,
      msg->pose.orientation.y,
      msg->pose.orientation.z);
    if (!goal_orientation.coeffs().allFinite() || goal_orientation.norm() < 1e-9) {
      next_goal.orientation = Eigen::Quaterniond::Identity();
    } else {
      next_goal.orientation = goal_orientation.normalized();
    }
    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      1000,
      "Received controller /goal_pose: position=(%.4f, %.4f, %.4f), orientation=(%.4f, %.4f, %.4f, %.4f)",
      msg->pose.position.x,
      msg->pose.position.y,
      msg->pose.position.z,
      msg->pose.orientation.x,
      msg->pose.orientation.y,
      msg->pose.orientation.z,
      msg->pose.orientation.w);
    goal_target_box_.set(next_goal);
    external_goal_received_.store(true);
    startup_goal_synced_.store(true);
  }

  void on_obstacles(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
  {
    std::vector<ObstacleSphere> obstacles;
    std::vector<std::optional<ObstacleSphere>> proximity_sensor_obstacles(
      RB10Model::sensor_control_points.size());
    for (const auto & marker : msg->markers) {
      const auto proximity_control_point_index = proximity_marker_control_point_index(marker);
      if (marker.action != visualization_msgs::msg::Marker::ADD) {
        continue;
      }
	      const Eigen::Vector3d marker_center(
	        marker.pose.position.x,
	        marker.pose.position.y,
	        marker.pose.position.z);
	      const double obstacle_radius = 0.5 * std::max(marker.scale.x, marker.scale.y);
	      ObstacleSphere obstacle{marker_center, obstacle_radius};
      obstacle.source_id = obstacle_source_id(marker);
      if (proximity_control_point_index.has_value()) {
        obstacle.proximity_control_point_index =
          static_cast<int>(proximity_control_point_index.value());
      }

	      if (
	        marker.type == visualization_msgs::msg::Marker::CYLINDER &&
	        marker.scale.z > obstacle_radius * 2.0)
	      {
        Eigen::Quaterniond orientation(
          marker.pose.orientation.w,
          marker.pose.orientation.x,
          marker.pose.orientation.y,
          marker.pose.orientation.z);
        if (orientation.norm() <= 1e-9) {
          orientation = Eigen::Quaterniond::Identity();
        }
        const Eigen::Vector3d cylinder_axis =
          orientation.normalized().toRotationMatrix() * Eigen::Vector3d::UnitZ();
        const double height = std::max(marker.scale.z, 0.0);
        const double spacing = std::max(obstacle_radius, 0.02);
        const int sphere_count =
          std::clamp(static_cast<int>(std::ceil(height / spacing)) + 1, 2, 64);
	        const double step = height / static_cast<double>(sphere_count - 1);
	        for (int index = 0; index < sphere_count; ++index) {
	          const double offset = -0.5 * height + step * static_cast<double>(index);
          ObstacleSphere cylinder_obstacle{
            marker_center + offset * cylinder_axis,
            obstacle_radius};
          cylinder_obstacle.proximity_control_point_index =
            obstacle.proximity_control_point_index;
          cylinder_obstacle.source_id = obstacle.source_id;
	          obstacles.push_back(cylinder_obstacle);
	        }
	      } else {
	        obstacles.push_back(obstacle);
	      }
      if (proximity_control_point_index.has_value()) {
        proximity_sensor_obstacles[proximity_control_point_index.value()] = obstacle;
      }
    }
    if (obstacles.empty()) {
      obstacles.push_back(ObstacleSphere{});
    }
    obstacles_box_.set(obstacles);
    proximity_sensor_obstacles_box_.set(proximity_sensor_obstacles);
  }

  std::optional<std::size_t> proximity_marker_control_point_index(
    const visualization_msgs::msg::Marker & marker) const
  {
    if (marker.ns != "proximity_obstacles") {
      return std::nullopt;
    }

    if (!marker.text.empty()) {
      for (std::size_t index = 0; index < RB10Model::sensor_control_points.size(); ++index) {
        if (marker.text == RB10Model::sensor_control_points[index].frame_name) {
          return index;
        }
      }
      return std::nullopt;
    }

    return std::nullopt;
  }

  EigenRmpConfig build_solver_config() const
  {
    EigenRmpConfig config;
    config.graph_nodes = parse_graph_nodes();
    config.max_joint_accel = max_joint_accel_;
    config.preserve_joint_accel_direction = preserve_joint_accel_direction_;
    config.enable_tangent_escape_dual_solve_diagnostics =
      get_parameter("publish_tangent_escape_dual_solve_data").as_bool();
    config.enable_tangent_escape_diagnostics =
      get_parameter("publish_tangent_escape_rmp_data").as_bool() ||
      config.enable_tangent_escape_dual_solve_diagnostics ||
      get_parameter("publish_tangent_escape_geometry_debug").as_bool();
    config.enable_leaf_ablation_diagnostics =
      get_parameter("publish_leaf_ablation_data").as_bool();
    config.enable_stability_certificate_diagnostics =
      get_parameter("publish_stability_certificate_data").as_bool() ||
      get_parameter("escape_stability_guard_enabled").as_bool();
    config.solve_offset = get_parameter("root_solve_offset").as_double();
    config.solve_method = get_parameter("solve_method").as_string();
    config.rmp_type = get_parameter("rmp_type").as_string();
    config.casadi_rmp2_max_obstacles = static_cast<int>(
      get_parameter("casadi_rmp2_max_obstacles").as_int());
    config.casadi_rmp2_jit = get_parameter("casadi_rmp2_jit").as_bool();
    config.body_obstacles = parse_body_obstacles();

    const auto default_q = get_parameter("default_q").as_double_array();
    for (std::size_t index = 0; index < std::min<std::size_t>(default_q.size(), 6); ++index) {
      config.default_q[index] = default_q[index];
    }
    config.joint_lower_limits = joint_lower_limits_;
    config.joint_upper_limits = joint_upper_limits_;

    const auto joint_limit_buffers = get_parameter("joint_limit_buffers").as_double_array();
    for (std::size_t index = 0; index < std::min<std::size_t>(joint_limit_buffers.size(), 6); ++index) {
      config.joint_limit_buffers[index] = joint_limit_buffers[index];
    }

    config.cspace_target.metric_scalar = get_parameter("cspace_target_metric_scalar").as_double();
    config.cspace_target.position_gain = get_parameter("cspace_target_position_gain").as_double();
    config.cspace_target.damping_gain = get_parameter("cspace_target_damping_gain").as_double();
    config.cspace_target.robust_position_term_thresh =
      get_parameter("cspace_target_robust_position_term_thresh").as_double();
    config.cspace_target.inertia = get_parameter("cspace_target_inertia").as_double();

    config.joint_limit.policy =
      get_parameter("joint_limit_policy").as_string();
    config.joint_limit.metric_scalar = get_parameter("joint_limit_metric_scalar").as_double();
    config.joint_limit.metric_length_scale =
      get_parameter("joint_limit_metric_length_scale").as_double();
    config.joint_limit.metric_exploder_eps =
      get_parameter("joint_limit_metric_exploder_eps").as_double();
    config.joint_limit.metric_velocity_gate_length_scale =
      get_parameter("joint_limit_metric_velocity_gate_length_scale").as_double();
    config.joint_limit.accel_damper_gain =
      get_parameter("joint_limit_accel_damper_gain").as_double();
    config.joint_limit.accel_potential_gain =
      get_parameter("joint_limit_accel_potential_gain").as_double();
    config.joint_limit.accel_potential_exploder_eps =
      get_parameter("joint_limit_accel_potential_exploder_eps").as_double();
    config.joint_limit.accel_potential_exploder_length_scale =
      get_parameter("joint_limit_accel_potential_exploder_length_scale").as_double();
    config.joint_limit.gds_center_fraction =
      get_parameter("joint_limit_gds_center_fraction").as_double();
    config.joint_limit.gds_domain_epsilon =
      get_parameter("joint_limit_gds_domain_epsilon").as_double();

    config.joint_velocity_cap.max_velocity =
      get_parameter("joint_velocity_cap_max_velocity").as_double();
    config.joint_velocity_cap.velocity_damping_region =
      get_parameter("joint_velocity_cap_velocity_damping_region").as_double();
    config.joint_velocity_cap.damping_gain =
      get_parameter("joint_velocity_cap_damping_gain").as_double();
    config.joint_velocity_cap.metric_weight =
      get_parameter("joint_velocity_cap_metric_weight").as_double();

    config.target.accel_p_gain = get_parameter("target_rmp_accel_p_gain").as_double();
    config.target.accel_d_gain = get_parameter("target_rmp_accel_d_gain").as_double();
    config.target.accel_norm_eps = get_parameter("target_rmp_accel_norm_eps").as_double();
    config.target.metric_alpha_length_scale =
      get_parameter("target_rmp_metric_alpha_length_scale").as_double();
    config.target.min_metric_alpha =
      get_parameter("target_rmp_min_metric_alpha").as_double();
    config.target.max_metric_scalar =
      get_parameter("target_rmp_max_metric_scalar").as_double();
    config.target.min_metric_scalar =
      get_parameter("target_rmp_min_metric_scalar").as_double();
    config.target.proximity_metric_boost_scalar =
      get_parameter("target_rmp_proximity_metric_boost_scalar").as_double();
    config.target.proximity_metric_boost_length_scale =
      get_parameter("target_rmp_proximity_metric_boost_length_scale").as_double();
    config.axis_target.accel_p_gain =
      get_parameter("axis_target_rmp_accel_p_gain").as_double();
    config.axis_target.accel_d_gain =
      get_parameter("axis_target_rmp_accel_d_gain").as_double();
    config.axis_target.metric_scalar =
      get_parameter("axis_target_rmp_metric_scalar").as_double();
    config.axis_target.proximity_metric_boost_scalar =
      get_parameter("axis_target_rmp_proximity_metric_boost_scalar").as_double();
    config.axis_target.proximity_metric_boost_length_scale =
      get_parameter("axis_target_rmp_proximity_metric_boost_length_scale").as_double();
    config.wrist_axis_target.accel_p_gain =
      get_parameter("wrist_axis_target_rmp_accel_p_gain").as_double();
    config.wrist_axis_target.accel_d_gain =
      get_parameter("wrist_axis_target_rmp_accel_d_gain").as_double();
    config.wrist_axis_target.metric_scalar =
      get_parameter("wrist_axis_target_rmp_metric_scalar").as_double();
    config.wrist_axis_target.proximity_metric_boost_scalar =
      get_parameter("wrist_axis_target_rmp_proximity_metric_boost_scalar").as_double();
    config.wrist_axis_target.proximity_metric_boost_length_scale =
      get_parameter("wrist_axis_target_rmp_proximity_metric_boost_length_scale").as_double();

    config.collision.policy = get_parameter("collision_policy").as_string();
    config.collision.margin = get_parameter("collision_rmp_margin").as_double();
    config.collision.damping_gain = get_parameter("collision_rmp_damping_gain").as_double();
    config.collision.damping_std_dev = get_parameter("collision_rmp_damping_std_dev").as_double();
    config.collision.damping_robustness_eps =
      get_parameter("collision_rmp_damping_robustness_eps").as_double();
    config.collision.damping_velocity_gate_length_scale =
      get_parameter("collision_rmp_damping_velocity_gate_length_scale").as_double();
    config.collision.repulsion_gain =
      get_parameter("collision_rmp_repulsion_gain").as_double();
    config.collision.repulsion_std_dev =
      get_parameter("collision_rmp_repulsion_std_dev").as_double();
    config.collision.metric_modulation_radius =
      get_parameter("collision_rmp_metric_modulation_radius").as_double();
    config.collision.metric_scalar =
      get_parameter("collision_rmp_metric_scalar").as_double();
    config.collision.metric_exploder_std_dev =
      get_parameter("collision_rmp_metric_exploder_std_dev").as_double();
	    config.collision.metric_exploder_eps =
	      get_parameter("collision_rmp_metric_exploder_eps").as_double();
    config.collision.gds_clearance_smoothing =
      get_parameter("collision_rmp_gds_clearance_smoothing").as_double();
    config.collision.gds_metric_velocity_floor =
      get_parameter("collision_rmp_gds_metric_velocity_floor").as_double();
    config.collision.gds_metric_velocity_scale =
      get_parameter("collision_rmp_gds_metric_velocity_scale").as_double();
    config.collision.gds_damping_velocity_scale =
      get_parameter("collision_rmp_gds_damping_velocity_scale").as_double();

    config.tangent_escape.enabled = get_parameter("enable_tangent_escape_rmp").as_bool();
    config.tangent_escape.acceleration_model =
      get_parameter("tangent_escape_rmp_acceleration_model").as_string();
    if (
      config.tangent_escape.acceleration_model != "canonical_velocity" &&
      config.tangent_escape.acceleration_model != "risk_damped" &&
      config.tangent_escape.acceleration_model != "tangent_plane" &&
      config.tangent_escape.acceleration_model != "tangent_ode")
    {
      throw std::runtime_error(
              "tangent_escape_rmp_acceleration_model must be "
              "'canonical_velocity', 'risk_damped', 'tangent_plane', or "
              "'tangent_ode'");
    }
    config.tangent_escape.metric_scalar =
      get_parameter("tangent_escape_rmp_metric_scalar").as_double();
    config.tangent_escape.clearance_margin =
      get_parameter("tangent_escape_rmp_clearance_margin").as_double();
    config.tangent_escape.velocity_gain =
      get_parameter("tangent_escape_rmp_velocity_gain").as_double();
    config.tangent_escape.max_accel =
      get_parameter("tangent_escape_rmp_max_accel").as_double();
    config.tangent_escape.goal_block_beta_on =
      get_parameter("tangent_escape_rmp_goal_block_beta_on").as_double();
    config.tangent_escape.goal_block_beta_full =
      get_parameter("tangent_escape_rmp_goal_block_beta_full").as_double();
    config.tangent_escape.nominal_prediction_dt =
      get_parameter("tangent_escape_rmp_nominal_prediction_dt").as_double();
    config.tangent_escape.normal_tolerance =
      get_parameter("tangent_escape_rmp_normal_tolerance").as_double();
    config.tangent_escape.control_dt =
      control_rate_hz_ > 1e-9 ? 1.0 / control_rate_hz_ : 0.01;
    config.tangent_escape.blocked_memory_penalty_weight =
      get_parameter("tangent_escape_rmp_blocked_memory_penalty_weight").as_double();
    config.tangent_escape.prevent_weight =
      get_parameter("tangent_escape_rmp_prevent_weight").as_double();
    config.tangent_escape.activation_time_constant =
      get_parameter("tangent_escape_rmp_activation_time_constant").as_double();
    config.tangent_escape.activation_rise_rate =
      get_parameter("tangent_escape_rmp_activation_rise_rate").as_double();
    config.tangent_escape.activation_fall_rate =
      get_parameter("tangent_escape_rmp_activation_fall_rate").as_double();
    config.tangent_escape.progress_filter_time_constant =
      get_parameter("tangent_escape_rmp_progress_filter_time_constant").as_double();
    config.tangent_escape.progress_low_threshold =
      get_parameter("tangent_escape_rmp_progress_low_threshold").as_double();
    config.tangent_escape.progress_ok_threshold =
      get_parameter("tangent_escape_rmp_progress_ok_threshold").as_double();
    config.tangent_escape.still_speed_threshold =
      get_parameter("tangent_escape_rmp_still_speed_threshold").as_double();
    config.tangent_escape.moving_speed_threshold =
      get_parameter("tangent_escape_rmp_moving_speed_threshold").as_double();
    config.tangent_escape.intent_on_speed =
      get_parameter("tangent_escape_rmp_intent_on_speed").as_double();
    config.tangent_escape.intent_full_speed =
      get_parameter("tangent_escape_rmp_intent_full_speed").as_double();
    config.tangent_escape.prevent_speed =
      get_parameter("tangent_escape_rmp_prevent_speed").as_double();
    config.tangent_escape.recovery_speed =
      get_parameter("tangent_escape_rmp_recovery_speed").as_double();
    config.tangent_escape.max_speed =
      get_parameter("tangent_escape_rmp_max_speed").as_double();
    config.tangent_escape.desired_velocity_time_constant =
      get_parameter("tangent_escape_rmp_desired_velocity_time_constant").as_double();
    config.tangent_escape.release_stop_speed =
      get_parameter("tangent_escape_rmp_release_stop_speed").as_double();
    config.tangent_escape.release_hold_speed =
      get_parameter("tangent_escape_rmp_release_hold_speed").as_double();
    config.tangent_escape.drive_ramp_duration =
      get_parameter("tangent_escape_rmp_drive_ramp_duration").as_double();
    config.tangent_escape.handoff_duration =
      get_parameter("tangent_escape_rmp_handoff_duration").as_double();
    config.tangent_escape.minimum_drive_duration =
      get_parameter("tangent_escape_rmp_minimum_drive_duration").as_double();
    config.tangent_escape.maximum_active_duration =
      get_parameter("tangent_escape_rmp_maximum_active_duration").as_double();
    config.tangent_escape.timeout_reentry_cooldown =
      get_parameter("tangent_escape_rmp_timeout_reentry_cooldown").as_double();
    config.tangent_escape.release_blockage_threshold =
      get_parameter("tangent_escape_rmp_release_blockage_threshold").as_double();
    config.tangent_escape.pair_switch_margin =
      get_parameter("tangent_escape_rmp_pair_switch_margin").as_double();
    config.tangent_escape.direction_switch_margin =
      get_parameter("tangent_escape_rmp_direction_switch_margin").as_double();
    config.tangent_escape.goal_score_scale =
      get_parameter("tangent_escape_rmp_goal_score_scale").as_double();
    config.tangent_escape.sector_risk_scale =
      get_parameter("tangent_escape_rmp_sector_risk_scale").as_double();
    config.tangent_escape.sector_risk_hard_limit =
      get_parameter("tangent_escape_rmp_sector_risk_hard_limit").as_double();
    config.tangent_escape.sector_risk_change_threshold =
      get_parameter("tangent_escape_rmp_sector_risk_change_threshold").as_double();
    config.tangent_escape.accel_jump_weight =
      get_parameter("tangent_escape_rmp_accel_jump_weight").as_double();
    config.tangent_escape.accel_jump_scale =
      get_parameter("tangent_escape_rmp_accel_jump_scale").as_double();
    config.tangent_escape.blocked_memory_sigma =
      get_parameter("tangent_escape_rmp_blocked_memory_sigma").as_double();
    config.tangent_escape.candidate_min_displacement =
      get_parameter("tangent_escape_rmp_candidate_min_displacement").as_double();
    config.tangent_escape.refinement_iterations =
      static_cast<int>(get_parameter("tangent_escape_rmp_refinement_iterations").as_int());
    config.tangent_escape.command_test_distance =
      get_parameter("tangent_escape_rmp_command_test_distance").as_double();
    config.tangent_escape.minimum_move_ratio =
      get_parameter("tangent_escape_rmp_minimum_move_ratio").as_double();
    config.tangent_escape.goal_change_reset_distance =
      get_parameter("tangent_escape_rmp_goal_change_reset_distance").as_double();
    config.tangent_escape.risk_distance_gain =
      get_parameter("tangent_escape_rmp_risk_distance_gain").as_double();
    config.tangent_escape.risk_distance_scale =
      get_parameter("tangent_escape_rmp_risk_distance_scale").as_double();
    config.tangent_escape.risk_approach_gain =
      get_parameter("tangent_escape_rmp_risk_approach_gain").as_double();
    config.tangent_escape.risk_approach_distance_scale =
      get_parameter("tangent_escape_rmp_risk_approach_distance_scale").as_double();
    config.tangent_escape.risk_approach_epsilon =
      get_parameter("tangent_escape_rmp_risk_approach_epsilon").as_double();
    config.tangent_escape.risk_velocity_gate_scale =
      get_parameter("tangent_escape_rmp_risk_velocity_gate_scale").as_double();
    config.tangent_escape.risk_clearance_rate_filter_time_constant =
      get_parameter(
      "tangent_escape_rmp_risk_clearance_rate_filter_time_constant").as_double();
    config.tangent_escape.risk_tangent_damping_gain =
      get_parameter("tangent_escape_rmp_risk_tangent_damping_gain").as_double();
    config.tangent_escape.risk_stuck_accel_floor =
      get_parameter("tangent_escape_rmp_risk_stuck_accel_floor").as_double();
    config.tangent_escape.risk_stuck_dwell_time =
      get_parameter("tangent_escape_rmp_risk_stuck_dwell_time").as_double();
    config.tangent_escape.risk_stuck_ramp_time =
      get_parameter("tangent_escape_rmp_risk_stuck_ramp_time").as_double();
    config.tangent_escape_ode.metric_max =
      get_parameter("tangent_escape_ode_metric_max").as_double();
    config.tangent_escape_ode.metric_sigma =
      get_parameter("tangent_escape_ode_metric_sigma").as_double();
    config.tangent_escape_ode.accel_max =
      get_parameter("tangent_escape_ode_accel_max").as_double();
    config.tangent_escape_ode.accel_sigma =
      get_parameter("tangent_escape_ode_accel_sigma").as_double();
    config.tangent_escape_ode.budget_scale =
      get_parameter("tangent_escape_ode_budget_scale").as_double();
    config.tangent_escape_ode.budget_recovery_time =
      get_parameter("tangent_escape_ode_budget_recovery_time").as_double();
    config.tangent_escape_ode.tangent_damping =
      get_parameter("tangent_escape_ode_tangent_damping").as_double();
    config.tangent_escape_ode.direction_rate =
      get_parameter("tangent_escape_ode_direction_rate").as_double();
    config.tangent_escape_ode.risk_sigma =
      get_parameter("tangent_escape_ode_risk_sigma").as_double();
    config.tangent_escape_ode.weight_goal =
      get_parameter("tangent_escape_ode_weight_goal").as_double();
    config.tangent_escape_ode.weight_velocity =
      get_parameter("tangent_escape_ode_weight_velocity").as_double();
    config.tangent_escape_ode.weight_risk =
      get_parameter("tangent_escape_ode_weight_risk").as_double();
    config.tangent_escape_ode.weight_curvature =
      get_parameter("tangent_escape_ode_weight_curvature").as_double();
    config.tangent_escape_ode.velocity_epsilon =
      get_parameter("tangent_escape_ode_velocity_epsilon").as_double();
    config.tangent_escape_ode.axial_preference =
      get_parameter("tangent_escape_ode_axial_preference").as_double();
    config.tangent_escape_ode.pocket_threshold =
      get_parameter("tangent_escape_ode_pocket_threshold").as_double();
    config.tangent_escape_ode.pocket_epsilon =
      get_parameter("tangent_escape_ode_pocket_epsilon").as_double();
    config.tangent_escape_ode.activation_distance =
      get_parameter("tangent_escape_ode_activation_distance").as_double();
    config.tangent_escape_ode.release_distance =
      get_parameter("tangent_escape_ode_release_distance").as_double();
    config.tangent_escape_ode.switch_hysteresis =
      get_parameter("tangent_escape_ode_switch_hysteresis").as_double();
    config.tangent_escape_ode.weight_stall =
      get_parameter("tangent_escape_ode_weight_stall").as_double();
    config.tangent_escape_ode.stall_time_constant =
      get_parameter("tangent_escape_ode_stall_time_constant").as_double();
    config.tangent_escape_ode.stall_speed =
      get_parameter("tangent_escape_ode_stall_speed").as_double();
    config.tangent_escape.metric_isotropy =
      get_parameter("tangent_escape_rmp_metric_isotropy").as_double();
    config.tangent_escape.push_safe_distance =
      get_parameter("tangent_escape_rmp_push_safe_distance").as_double();
    config.tangent_escape.push_influence_distance =
      get_parameter("tangent_escape_rmp_push_influence_distance").as_double();
    config.tangent_escape.metric_safe_distance =
      get_parameter("tangent_escape_rmp_metric_safe_distance").as_double();
    config.tangent_escape.metric_influence_distance =
      get_parameter("tangent_escape_rmp_metric_influence_distance").as_double();

    config.escape_energy_certificate.enabled =
      get_parameter("escape_stability_guard_enabled").as_bool();
    config.escape_energy_certificate.tank_capacity =
      get_parameter("escape_stability_tank_capacity").as_double();
    // Start every controller episode with a full finite-energy tank. This
    // removes a redundant public parameter while retaining direct-config
    // initial_energy support in solver unit tests.
    config.escape_energy_certificate.initial_energy =
      config.escape_energy_certificate.tank_capacity;
    config.escape_energy_certificate.control_dt =
      control_rate_hz_ > 1e-9 ? 1.0 / control_rate_hz_ : 0.01;
    config.escape_energy_certificate.scale_search_iterations =
      static_cast<int>(
      get_parameter("escape_stability_scale_search_iterations").as_int());
    config.escape_energy_certificate.power_tolerance =
      get_parameter("escape_stability_power_tolerance").as_double();
    config.tangent_escape.safe_distance =
      get_parameter("tangent_escape_rmp_safe_distance").as_double();
    config.tangent_escape.influence_distance =
      get_parameter("tangent_escape_rmp_influence_distance").as_double();
    config.tangent_escape.candidate_lookahead =
      get_parameter("tangent_escape_rmp_candidate_lookahead").as_double();
    config.tangent_escape.goal_weight =
      get_parameter("tangent_escape_rmp_goal_weight").as_double();
    config.tangent_escape.continuity_weight =
      get_parameter("tangent_escape_rmp_continuity_weight").as_double();
    config.tangent_escape.sector_risk_weight =
      get_parameter("tangent_escape_rmp_sector_risk_weight").as_double();
    config.tangent_escape.min_activation =
      get_parameter("tangent_escape_rmp_min_activation").as_double();
    config.tangent_escape.min_tangent_norm =
      get_parameter("tangent_escape_rmp_min_tangent_norm").as_double();

    if (!config.tangent_escape.enabled) {
      for (auto & node : config.graph_nodes) {
        if (
          node.leaf_rmp_type == "tangent_escape" ||
          node.handcrafted_leaf_rmp_type == "tangent_escape")
        {
          node.enabled = false;
        }
      }
    }

	    config.damping.accel_d_gain = get_parameter("damping_rmp_accel_d_gain").as_double();
    config.damping.metric_scalar = get_parameter("damping_rmp_metric_scalar").as_double();
    config.damping.inertia = get_parameter("damping_rmp_inertia").as_double();
    return config;
  }

  std::unique_ptr<RmpSolverInterface> build_solver(const EigenRmpConfig & config) const
  {
    const std::string backend = get_parameter("solver_backend").as_string();
    const std::string urdf_path = resolve_robot_urdf_path();
    if (backend == "pinocchio_direct") {
      RCLCPP_INFO(get_logger(), "Using Pinocchio explicit RMP2 solver backend");
      return std::make_unique<PinocchioDirectRmpSolver>(config, urdf_path);
    }
    if (backend == "casadi_rmp2") {
      RCLCPP_INFO(
        get_logger(),
        "Using CasADi Algorithm-3 RMP2 solver backend (jit=%s)",
        config.casadi_rmp2_jit ? "true" : "false");
      return std::make_unique<CasadiRmp2Solver>(config, urdf_path);
    }
    throw std::invalid_argument(
            "solver_backend must be pinocchio_direct or casadi_rmp2, got: " + backend);
  }

  std::string resolve_robot_urdf_path() const
  {
    const auto configured = get_parameter("robot_urdf_path").as_string();
    if (!configured.empty()) {
      return configured;
    }
    return ament_index_cpp::get_package_share_directory("rb10_rmpflow_rviz") +
           "/urdf/rb10_1300e.urdf";
  }

  void declare_graph_parameters()
  {
    const auto node_names = get_parameter("graph.node_names").as_string_array();
    const auto defaults = default_rmp_graph_nodes();
    for (const auto & name : node_names) {
      auto match_it = std::find_if(
        defaults.begin(),
        defaults.end(),
        [&name](const RmpNodeConfig & node) {return node.name == name;});
      const RmpNodeConfig fallback =
        match_it != defaults.end() ? *match_it : make_rmp_node_config(name, "root", "identity", "none", true);
      const std::string prefix = "graph." + name + ".";
      declare_parameter(prefix + "parent", fallback.parents.front());
      declare_parameter(prefix + "parents", fallback.parents);
      declare_parameter(prefix + "task_map", fallback.task_map_type);
      declare_parameter(prefix + "leaf", fallback.leaf_rmp_type);
      declare_parameter(prefix + "enabled", fallback.enabled);
      declare_parameter(prefix + "target_key", fallback.target_key);
      declare_parameter(prefix + "link_name", fallback.link_name);
      declare_parameter(prefix + "axis", fallback.axis);
      declare_parameter(prefix + "handcrafted_leaf", fallback.handcrafted_leaf_rmp_type);
      declare_parameter(prefix + "parent_weights", fallback.parent_weights);
      declare_parameter(prefix + "bias", fallback.bias);
      declare_parameter(prefix + "matrix", fallback.matrix);
      declare_parameter(prefix + "slice_start", fallback.slice_start);
      declare_parameter(prefix + "slice_length", fallback.slice_length);
      declare_parameter(prefix + "scale", fallback.scale);
      declare_parameter(prefix + "identity_multiplier", fallback.identity_multiplier);
      declare_parameter(prefix + "epsilon", fallback.epsilon);
    }
  }

  std::vector<RmpNodeConfig> parse_graph_nodes() const
  {
    const auto node_names = get_parameter("graph.node_names").as_string_array();
    std::vector<RmpNodeConfig> nodes;
    nodes.reserve(node_names.size());

    for (const auto & name : node_names) {
      const std::string prefix = "graph." + name + ".";
      auto parents = get_parameter(prefix + "parents").as_string_array();
      if (parents.empty()) {
        parents.push_back(get_parameter(prefix + "parent").as_string());
      }
      nodes.push_back(make_rmp_node_config(
        name,
        parents,
        get_parameter(prefix + "task_map").as_string(),
        get_parameter(prefix + "leaf").as_string(),
        get_parameter(prefix + "enabled").as_bool(),
        get_parameter(prefix + "target_key").as_string(),
        get_parameter(prefix + "link_name").as_string(),
        get_parameter(prefix + "axis").as_string(),
        get_parameter(prefix + "handcrafted_leaf").as_string(),
        get_parameter(prefix + "parent_weights").as_double_array(),
        get_parameter(prefix + "bias").as_double_array(),
        get_parameter(prefix + "matrix").as_double_array(),
        static_cast<int>(get_parameter(prefix + "slice_start").as_int()),
        static_cast<int>(get_parameter(prefix + "slice_length").as_int()),
        get_parameter(prefix + "scale").as_double(),
        get_parameter(prefix + "identity_multiplier").as_double(),
        get_parameter(prefix + "epsilon").as_double()));
    }

    return nodes;
  }

  std::vector<BodyObstacle> parse_body_obstacles() const
  {
    std::vector<std::string> names;
    get_parameter_or("body_obstacles.names", names, std::vector<std::string>{});
    std::vector<BodyObstacle> obstacles;
    obstacles.reserve(names.size());
    for (const auto & name : names) {
      const std::string prefix = "body_obstacles." + name + ".";
      BodyObstacle obstacle;
      obstacle.type = get_parameter(prefix + "type").as_string();
      obstacle.link_name = get_parameter(prefix + "link_name").as_string();
      obstacle.mins = parse_vector3_parameter(prefix + "mins", Eigen::Vector3d::Zero());
      obstacle.maxs = parse_vector3_parameter(prefix + "maxs", Eigen::Vector3d::Zero());
      obstacle.center = parse_vector3_parameter(prefix + "center", Eigen::Vector3d::Zero());
      obstacle.radius = get_parameter(prefix + "radius").as_double();
      obstacles.push_back(obstacle);
    }
    return obstacles;
  }

  int control_point_count() const
  {
    return static_cast<int>(RB10Model::sensor_control_points.size());
  }

  std::vector<std::size_t> build_topological_order(
    const std::vector<RmpNodeConfig> & nodes) const
  {
    std::unordered_map<std::string, std::size_t> enabled_nodes;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
      if (!nodes[index].enabled) {
        continue;
      }
      enabled_nodes.emplace(nodes[index].name, index);
    }

    std::unordered_map<std::string, int> indegree;
    std::unordered_map<std::string, std::vector<std::string>> outgoing;
    for (const auto & entry : enabled_nodes) {
      indegree.emplace(entry.first, 0);
    }

    for (const auto & entry : enabled_nodes) {
      const auto & node = nodes[entry.second];
      for (const auto & parent : node.parents) {
        if (parent == "root") {
          continue;
        }
        if (!enabled_nodes.count(parent)) {
          throw std::runtime_error(
                  "Graph node " + node.name + " references missing parent " + parent);
        }
        ++indegree[node.name];
        outgoing[parent].push_back(node.name);
      }
    }

    std::vector<std::string> ready;
    for (const auto & entry : indegree) {
      if (entry.second == 0) {
        ready.push_back(entry.first);
      }
    }
    std::sort(ready.begin(), ready.end());

    std::vector<std::size_t> order;
    while (!ready.empty()) {
      const auto name = ready.front();
      ready.erase(ready.begin());
      order.push_back(enabled_nodes.at(name));
      for (const auto & child : outgoing[name]) {
        auto & child_indegree = indegree.at(child);
        --child_indegree;
        if (child_indegree == 0) {
          ready.push_back(child);
        }
      }
      std::sort(ready.begin(), ready.end());
    }

    if (order.size() != enabled_nodes.size()) {
      throw std::runtime_error("Cycle detected in graph configuration");
    }
    return order;
  }

  std::unordered_map<std::string, int> infer_node_dims(
    const std::vector<RmpNodeConfig> & nodes) const
  {
    std::unordered_map<std::string, int> dims;
    dims.emplace("root", 6);
    for (const auto index : build_topological_order(nodes)) {
      const auto & node = nodes[index];
      if (node.task_map_type == "tcp_position" ||
        node.task_map_type == "link_position" ||
        node.task_map_type == "link_orientation_axis")
      {
        dims[node.name] = 3;
      } else if (node.task_map_type == "joint_limit") {
        dims[node.name] = 12;
      } else if (node.task_map_type == "control_points") {
        dims[node.name] = 3 * control_point_count();
      } else if (node.task_map_type == "collision_distance") {
        dims[node.name] = control_point_count();
      } else if (node.task_map_type == "norm") {
        dims[node.name] = 1;
      } else if (node.task_map_type == "affine") {
        if (!node.bias.empty()) {
          dims[node.name] = static_cast<int>(node.bias.size());
        } else if (!node.matrix.empty()) {
          int input_dim = 0;
          for (const auto & parent : node.parents) {
            input_dim += dims.at(parent);
          }
          if (input_dim == 0 || static_cast<int>(node.matrix.size()) % input_dim != 0) {
            throw std::runtime_error("Invalid affine matrix size for node " + node.name);
          }
          dims[node.name] = static_cast<int>(node.matrix.size()) / input_dim;
        } else {
          dims[node.name] = dims.at(node.parents.front());
        }
      } else if (node.task_map_type == "concat") {
        int dim = 0;
        for (const auto & parent : node.parents) {
          dim += dims.at(parent);
        }
        dims[node.name] = dim;
      } else if (
        node.task_map_type == "elem_multiply" ||
        node.task_map_type == "elem_divide" ||
        node.task_map_type == "sin" ||
        node.task_map_type == "cos" ||
        node.task_map_type == "tanh" ||
        node.task_map_type == "square" ||
        node.task_map_type == "abs")
      {
        dims[node.name] = dims.at(node.parents.front());
      } else if (node.task_map_type == "slice") {
        dims[node.name] = node.slice_length > 0 ? node.slice_length :
          dims.at(node.parents.front()) - node.slice_start;
      } else {
        dims[node.name] = dims.at(node.parents.front());
      }
    }
    return dims;
  }

  void declare_external_rmp_feature_parameters(const std::vector<RmpNodeConfig> & nodes)
  {
    std::vector<std::string> feature_keys;
    for (const auto & node : nodes) {
      if (node.leaf_rmp_type == "external" || node.handcrafted_leaf_rmp_type == "external") {
        feature_keys.push_back(node.target_key);
      }
    }
    std::sort(feature_keys.begin(), feature_keys.end());
    feature_keys.erase(std::unique(feature_keys.begin(), feature_keys.end()), feature_keys.end());
    for (const auto & key : feature_keys) {
      const std::string prefix = "external_rmp." + key + ".";
      declare_parameter(prefix + "enabled", true);
      declare_parameter(prefix + "topic_prefix", key);
    }
  }

  void configure_external_rmp_inputs(const std::vector<RmpNodeConfig> & nodes)
  {
    declare_external_rmp_feature_parameters(nodes);
    const auto dims = infer_node_dims(nodes);
    const std::string root_prefix = get_parameter("external_rmp.topic_prefix").as_string();
    for (const auto & node : nodes) {
      const bool uses_external =
        node.leaf_rmp_type == "external" || node.handcrafted_leaf_rmp_type == "external";
      if (!uses_external) {
        continue;
      }
      const std::string key = node.target_key;
      const std::string prefix = "external_rmp." + key + ".";
      if (!get_parameter(prefix + "enabled").as_bool()) {
        continue;
      }
      if (external_rmp_buffers_.count(key)) {
        continue;
      }
      const int dim = dims.at(node.name);
      external_rmp_buffers_.emplace(
        key,
        ExternalRmpBuffer{
          dim,
          Eigen::MatrixXd::Zero(dim, dim),
          Eigen::VectorXd::Zero(dim),
          false,
          false
        });
      const std::string topic_prefix =
        root_prefix + "/" + get_parameter(prefix + "topic_prefix").as_string();
      external_metric_subs_.push_back(create_subscription<std_msgs::msg::Float64MultiArray>(
          topic_prefix + "/metric_sqrt",
          10,
          [this, key, dim](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
            if (static_cast<int>(msg->data.size()) != dim * dim) {
              RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "External RMP metric_sqrt size mismatch for %s: expected %d, got %zu",
                key.c_str(),
                dim * dim,
                msg->data.size());
              return;
            }
            Eigen::MatrixXd matrix(dim, dim);
            for (int row = 0; row < dim; ++row) {
              for (int col = 0; col < dim; ++col) {
                matrix(row, col) = msg->data[static_cast<std::size_t>(row * dim + col)];
              }
            }
            std::scoped_lock lock(external_rmp_mutex_);
            auto & buffer = external_rmp_buffers_.at(key);
            buffer.metric_sqrt = matrix;
            buffer.has_metric = true;
          }));
      external_accel_subs_.push_back(create_subscription<std_msgs::msg::Float64MultiArray>(
          topic_prefix + "/acceleration",
          10,
          [this, key, dim](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
            if (static_cast<int>(msg->data.size()) != dim) {
              RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "External RMP acceleration size mismatch for %s: expected %d, got %zu",
                key.c_str(),
                dim,
                msg->data.size());
              return;
            }
            Eigen::VectorXd acceleration(dim);
            for (int index = 0; index < dim; ++index) {
              acceleration[index] = msg->data[static_cast<std::size_t>(index)];
            }
            std::scoped_lock lock(external_rmp_mutex_);
            auto & buffer = external_rmp_buffers_.at(key);
            buffer.acceleration = acceleration;
            buffer.has_acceleration = true;
          }));
      RCLCPP_INFO(
        get_logger(),
        "External RMP input enabled for key '%s' on topics %s/metric_sqrt and %s/acceleration",
        key.c_str(),
        topic_prefix.c_str(),
        topic_prefix.c_str());
    }
  }

  void declare_body_obstacle_parameters()
  {
    std::vector<std::string> names;
    get_parameter_or("body_obstacles.names", names, std::vector<std::string>{});
    for (const auto & name : names) {
      const std::string prefix = "body_obstacles." + name + ".";
      declare_parameter(prefix + "type", "ball");
      declare_parameter(prefix + "link_name", "");
      declare_parameter(prefix + "mins", std::vector<double>{0.0, 0.0, 0.0});
      declare_parameter(prefix + "maxs", std::vector<double>{0.0, 0.0, 0.0});
      declare_parameter(prefix + "center", std::vector<double>{0.0, 0.0, 0.0});
      declare_parameter(prefix + "radius", 0.0);
    }
  }

  Eigen::Vector3d parse_vector3_parameter(
    const std::string & name,
    const Eigen::Vector3d & fallback) const
  {
    const auto values = get_parameter(name).as_double_array();
    Eigen::Vector3d out = fallback;
    for (std::size_t index = 0; index < std::min<std::size_t>(values.size(), 3); ++index) {
      out[static_cast<int>(index)] = values[index];
    }
    return out;
  }

  std::size_t link_index_from_name(const std::string & link_name) const
  {
    for (std::size_t index = 0; index < RB10Model::link_names.size(); ++index) {
      if (link_name == RB10Model::link_names[index]) {
        return index;
      }
    }
    throw std::runtime_error("Unknown RB10 link name: " + link_name);
  }

  Eigen::Vector3d body_obstacle_center_world(
    const BodyObstacle & obstacle,
    const KinematicsContext & context) const
  {
    if (!obstacle.link_name.empty()) {
      const auto link_index = link_index_from_name(obstacle.link_name);
      const auto & rotation = context.link_rotations[link_index];
      const auto & origin = context.link_positions[link_index];
      if (obstacle.type == "box") {
        return origin + rotation * (0.5 * (obstacle.mins + obstacle.maxs));
      }
      return origin + rotation * obstacle.center;
    }

    if (obstacle.type == "box") {
      return 0.5 * (obstacle.mins + obstacle.maxs);
    }
    return obstacle.center;
  }

  double body_obstacle_min_z_world(
    const BodyObstacle & obstacle,
    const KinematicsContext & context) const
  {
    if (obstacle.type == "ball") {
      return body_obstacle_center_world(obstacle, context).z() - obstacle.radius;
    }

    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    if (!obstacle.link_name.empty()) {
      const auto link_index = link_index_from_name(obstacle.link_name);
      rotation = context.link_rotations[link_index];
      origin = context.link_positions[link_index];
    }

    double min_z = std::numeric_limits<double>::infinity();
    for (int x_sel = 0; x_sel < 2; ++x_sel) {
      for (int y_sel = 0; y_sel < 2; ++y_sel) {
        for (int z_sel = 0; z_sel < 2; ++z_sel) {
          const Eigen::Vector3d local_corner(
            x_sel == 0 ? obstacle.mins.x() : obstacle.maxs.x(),
            y_sel == 0 ? obstacle.mins.y() : obstacle.maxs.y(),
            z_sel == 0 ? obstacle.mins.z() : obstacle.maxs.z());
          min_z = std::min(min_z, (origin + rotation * local_corner).z());
        }
      }
    }
    return min_z;
  }

  bool body_obstacle_interacts_with_sensor_control_point(
    std::size_t /*point_index*/,
    const BodyObstacle & obstacle) const
  {
    (void)obstacle;
    return false;
  }

  struct FloorSafetyMetrics
  {
    double min_link_z{std::numeric_limits<double>::infinity()};
    double min_joint_z{std::numeric_limits<double>::infinity()};
    double min_control_point_z{std::numeric_limits<double>::infinity()};
    double min_body_obstacle_z{std::numeric_limits<double>::infinity()};
  };

  FloorSafetyMetrics compute_floor_safety_metrics(const KinematicsContext & context) const
  {
    FloorSafetyMetrics metrics;
    for (const auto & position : context.link_positions) {
      metrics.min_link_z = std::min(metrics.min_link_z, position.z());
    }
    for (const auto & origin : context.joint_origins) {
      metrics.min_joint_z = std::min(metrics.min_joint_z, origin.z());
    }
    for (const auto & control_point : context.control_points) {
      metrics.min_control_point_z = std::min(
        metrics.min_control_point_z,
        control_point.position.z() - control_point.radius);
    }
    for (const auto & obstacle : body_obstacles_visual_) {
      metrics.min_body_obstacle_z = std::min(
        metrics.min_body_obstacle_z,
        body_obstacle_min_z_world(obstacle, context));
    }
    return metrics;
  }

  void publish_debug_state(const RobotState & current_state, const KinematicsContext & context)
  {
    if (!debug_state_pub_) {
      return;
    }

    Eigen::Vector3d goal;
    std::vector<ObstacleSphere> obstacles;
    GoalTarget goal_target;
    goal_target_box_.get(goal_target);
    goal = goal_target.position;
    obstacles_box_.get(obstacles);

    double min_external_clearance = std::numeric_limits<double>::infinity();
    for (const auto & control_point : context.control_points) {
      for (const auto & obstacle : obstacles) {
        const double clearance =
          (control_point.position - obstacle.center).norm() -
          (control_point.radius + obstacle.radius);
        min_external_clearance = std::min(min_external_clearance, clearance);
      }
    }

    double min_body_clearance = std::numeric_limits<double>::infinity();
    for (std::size_t point_index = 0; point_index < context.control_points.size(); ++point_index) {
      const auto & control_point = context.control_points[point_index];
      for (const auto & obstacle : body_obstacles_visual_) {
        if (!body_obstacle_interacts_with_sensor_control_point(point_index, obstacle)) {
          continue;
        }
        if (obstacle.type != "ball") {
          continue;
        }
        const auto center = body_obstacle_center_world(obstacle, context);
        const double clearance =
          (control_point.position - center).norm() -
          (control_point.radius + obstacle.radius);
        min_body_clearance = std::min(min_body_clearance, clearance);
      }
    }

    if (!std::isfinite(min_external_clearance)) {
      min_external_clearance = 1e6;
    }
    if (!std::isfinite(min_body_clearance)) {
      min_body_clearance = 1e6;
    }

    const auto floor_metrics = compute_floor_safety_metrics(context);

    std_msgs::msg::Float64MultiArray debug;
    debug.data = {
      (context.tcp_position - goal).norm(),
      context.tcp_position.z(),
      context.link_positions[RB10Model::LINK6].z(),
      min_external_clearance,
      min_body_clearance,
      current_state.qd.norm(),
      last_min_z_safety_triggered_.load() ? 1.0 : 0.0,
      floor_metrics.min_link_z,
      floor_metrics.min_joint_z,
      floor_metrics.min_control_point_z,
      floor_metrics.min_body_obstacle_z,
    };
    debug_state_pub_->publish(debug);
  }

  struct RepulsionMetricSample
  {
    Eigen::Vector3d control_point{Eigen::Vector3d::Zero()};
    Eigen::Vector3d direction{Eigen::Vector3d::UnitZ()};
    double clearance{0.0};
    double metric_norm{0.0};
    double effective_metric_norm{0.0};
    double repulsion_norm{0.0};
    double intensity_norm{0.0};
  };

  static double sigmoid(double value)
  {
    if (value >= 0.0) {
      const double exp_value = std::exp(-value);
      return 1.0 / (1.0 + exp_value);
    }
    const double exp_value = std::exp(value);
    return exp_value / (1.0 + exp_value);
  }

  static geometry_msgs::msg::Point to_point(const Eigen::Vector3d & value)
  {
    geometry_msgs::msg::Point point;
    point.x = value.x();
    point.y = value.y();
    point.z = value.z();
    return point;
  }

  static void append_vector3(std::vector<double> & values, const Eigen::Vector3d & vector)
  {
    values.push_back(vector.x());
    values.push_back(vector.y());
    values.push_back(vector.z());
  }

  static Eigen::Vector3d normalized_or(
    const Eigen::Vector3d & vector,
    const Eigen::Vector3d & fallback)
  {
    const double norm = vector.norm();
    if (!vector.allFinite() || norm <= 1e-9) {
      return fallback;
    }
    return vector / norm;
  }

  TangentEscapeGeometrySample make_tangent_escape_geometry_sample(
    const KinematicsContext & context,
    std::size_t sensor_index,
    const ObstacleSphere & obstacle) const
  {
    TangentEscapeGeometrySample sample;
    sample.sensor_index = sensor_index;
    if (
      sensor_index >= RB10Model::sensor_control_points.size() ||
      sensor_index >= context.control_points.size() ||
      obstacle.radius <= 0.0 ||
      !obstacle.center.allFinite())
    {
      return sample;
    }

    const auto & sensor = RB10Model::sensor_control_points[sensor_index];
    const auto & control_point = context.control_points[sensor_index];
    sample.control_point = control_point.position;
    sample.obstacle_center = obstacle.center;

    const Eigen::Vector3d delta = sample.control_point - sample.obstacle_center;
    sample.center_distance = delta.norm();
    if (sample.center_distance <= 1e-9) {
      return sample;
    }

    const Eigen::Matrix3d & rotation = context.link_rotations[sensor.parent_link];
    sample.sensor_normal =
      normalized_or(rotation * sensor.local_normal, Eigen::Vector3d::UnitZ());
    sample.tangent_bias =
      normalized_or(rotation * sensor.local_tangent_bias, Eigen::Vector3d::UnitX());
    sample.collision_normal = delta / sample.center_distance;
    sample.obstacle_direction = -sample.collision_normal;
    sample.clearance =
      sample.center_distance - (control_point.radius + obstacle.radius) -
      collision_metric_params_.margin;
    sample.sensor_obstacle_dot = sample.sensor_normal.dot(sample.obstacle_direction);
    sample.collision_obstacle_dot = sample.collision_normal.dot(sample.obstacle_direction);
    sample.bias_sensor_dot = sample.tangent_bias.dot(sample.sensor_normal);
    sample.bias_obstacle_dot = sample.tangent_bias.dot(sample.obstacle_direction);
    if (sensor_index < context.control_point_velocities.size()) {
      sample.control_point_velocity = context.control_point_velocities[sensor_index];
      sample.velocity_obstacle_dot =
        sample.control_point_velocity.dot(sample.obstacle_direction);
      sample.velocity_tangent_dot =
        sample.control_point_velocity.dot(sample.tangent_bias);
    }
    if (sensor_index < context.control_point_jacobians.size()) {
      const auto & jacobian = context.control_point_jacobians[sensor_index];
      sample.jacobian_frobenius_norm = jacobian.norm();
      sample.sensor_normal_jacobian_norm =
        (sample.sensor_normal.transpose() * jacobian).norm();
      sample.obstacle_direction_jacobian_norm =
        (sample.obstacle_direction.transpose() * jacobian).norm();
      sample.tangent_bias_jacobian_norm =
        (sample.tangent_bias.transpose() * jacobian).norm();
    }
    sample.active = true;
    return sample;
  }

  std::optional<TangentEscapeGeometrySample> make_closest_tangent_escape_geometry_sample(
    const KinematicsContext & context,
    const std::vector<std::optional<ObstacleSphere>> & proximity_sensor_obstacles) const
  {
    std::optional<TangentEscapeGeometrySample> best_sample;
    for (std::size_t index = 0; index < proximity_sensor_obstacles.size(); ++index) {
      if (!proximity_sensor_obstacles[index].has_value()) {
        continue;
      }
      auto sample = make_tangent_escape_geometry_sample(
        context,
        index,
        proximity_sensor_obstacles[index].value());
      if (!sample.active) {
        continue;
      }
      if (
        !best_sample.has_value() ||
        sample.clearance < best_sample->clearance)
      {
        best_sample = sample;
      }
    }
    return best_sample;
  }

  std::vector<TangentEscapeGeometrySample> make_tangent_escape_geometry_samples(
    const KinematicsContext & context)
  {
    std::vector<std::optional<ObstacleSphere>> proximity_sensor_obstacles;
    proximity_sensor_obstacles_box_.get(proximity_sensor_obstacles);
    std::vector<TangentEscapeGeometrySample> samples;
    samples.reserve(proximity_sensor_obstacles.size());
    for (std::size_t index = 0; index < proximity_sensor_obstacles.size(); ++index) {
      if (!proximity_sensor_obstacles[index].has_value()) {
        continue;
      }
      auto sample = make_tangent_escape_geometry_sample(
        context,
        index,
        proximity_sensor_obstacles[index].value());
      if (sample.active) {
        samples.push_back(sample);
      }
    }
    return samples;
  }

  static std_msgs::msg::ColorRGBA repulsion_metric_color(double metric_norm)
  {
    const double clamped_metric = std::clamp(metric_norm, 0.0, 1.0);
    std_msgs::msg::ColorRGBA color;
    color.r = static_cast<float>(std::min(1.0, 2.0 * clamped_metric));
    color.g = static_cast<float>(std::min(1.0, 2.0 * (1.0 - clamped_metric)));
    color.b = 0.05F;
    color.a = static_cast<float>(0.35 + 0.65 * clamped_metric);
    return color;
  }

  std::optional<RepulsionMetricSample> make_repulsion_metric_sample(
    const ControlPoint & control_point,
    const Eigen::Vector3d & control_point_velocity,
    const ObstacleSphere & obstacle) const
  {
    if (obstacle.radius <= 0.0) {
      return std::nullopt;
    }

    const Eigen::Vector3d delta = control_point.position - obstacle.center;
    const double center_distance = delta.norm();
    Eigen::Vector3d direction = Eigen::Vector3d::UnitZ();
    if (center_distance > 1e-9) {
      direction = delta / center_distance;
    }
    const double clearance = std::max(
      center_distance - (control_point.radius + obstacle.radius) -
      collision_metric_params_.margin,
      0.0);
    const double clearance_velocity = direction.dot(control_point_velocity);

    const double metric_radius =
      std::max(collision_metric_params_.metric_modulation_radius, 1e-9);
    const double metric_exploder_std_dev =
      std::max(collision_metric_params_.metric_exploder_std_dev, 1e-9);
    const double metric_exploder_eps =
      std::max(collision_metric_params_.metric_exploder_eps, 1e-9);
    const double metric_normalizer =
      std::max(std::abs(collision_metric_params_.metric_scalar) / metric_exploder_eps, 1e-9);

    double metric_gate =
      clearance * clearance / (metric_radius * metric_radius) -
      2.0 * clearance / metric_radius + 1.0;
    if (clearance > metric_radius) {
      metric_gate = 0.0;
    }

    const double distance_metric =
      collision_metric_params_.metric_scalar /
      (clearance / metric_exploder_std_dev + metric_exploder_eps) *
      metric_gate;
    const double velocity_gate_scale =
      std::max(collision_metric_params_.damping_velocity_gate_length_scale, 1e-9);
    const double sigma = sigmoid(clearance_velocity / velocity_gate_scale);
    const double effective_metric = clearance > metric_radius ?
      0.0 :
      distance_metric * (1.0 - sigma);

    const double repulsion_std_dev =
      std::max(collision_metric_params_.repulsion_std_dev, 1e-9);
    const double repulsion_norm =
      std::clamp(std::exp(-(clearance / repulsion_std_dev)), 0.0, 1.0);
    const double repel = collision_metric_params_.repulsion_gain * repulsion_norm;
    const double damping =
      -(1.0 - sigma) * collision_metric_params_.damping_gain * clearance_velocity /
      (clearance / std::max(collision_metric_params_.damping_std_dev, 1e-9) +
      std::max(collision_metric_params_.damping_robustness_eps, 1e-9));
    const double accel_normalizer =
      std::max(std::abs(collision_metric_params_.repulsion_gain), 1e-9);
    const double accel_norm =
      std::clamp(std::max(repel + damping, 0.0) / accel_normalizer, 0.0, 1.0);
    const double metric_norm =
      std::clamp(distance_metric / metric_normalizer, 0.0, 1.0);
    const double effective_metric_norm =
      std::clamp(effective_metric / metric_normalizer, 0.0, 1.0);
    const double intensity_norm =
      std::clamp(std::sqrt(effective_metric_norm * accel_norm), 0.0, 1.0);

    return RepulsionMetricSample{
      control_point.position,
      direction,
      clearance,
      metric_norm,
      effective_metric_norm,
      repulsion_norm,
      intensity_norm
    };
  }

  void publish_repulsion_metric_markers(
    const RobotState & state,
    const KinematicsContext & context)
  {
    if (!repulsion_metric_pub_) {
      return;
    }

    std::vector<std::optional<ObstacleSphere>> proximity_sensor_obstacles;
    proximity_sensor_obstacles_box_.get(proximity_sensor_obstacles);

    visualization_msgs::msg::MarkerArray markers;
    const auto stamp = now();
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header.frame_id = "base_link";
    clear_marker.header.stamp = stamp;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear_marker);

    int marker_id = 0;
    for (std::size_t point_index = 0; point_index < context.control_points.size(); ++point_index) {
      if (
        point_index >= proximity_sensor_obstacles.size() ||
        !proximity_sensor_obstacles[point_index].has_value())
      {
        continue;
      }

      const auto & control_point = context.control_points[point_index];
      Eigen::Vector3d control_point_velocity = Eigen::Vector3d::Zero();
      if (point_index < context.control_point_jacobians.size()) {
        control_point_velocity = context.control_point_jacobians[point_index] * state.qd;
      }

      auto sample = make_repulsion_metric_sample(
        control_point,
        control_point_velocity,
        proximity_sensor_obstacles[point_index].value());
      if (
        !sample.has_value() ||
        sample->metric_norm < repulsion_metric_marker_min_norm_)
      {
        continue;
      }

      const double metric_norm = std::max(sample->metric_norm, sample->effective_metric_norm);

      visualization_msgs::msg::Marker sensor_marker;
      sensor_marker.header.frame_id = "base_link";
      sensor_marker.header.stamp = stamp;
      sensor_marker.ns = "repulsion_metric_dots";
      sensor_marker.id = marker_id++;
      sensor_marker.type = visualization_msgs::msg::Marker::SPHERE;
      sensor_marker.action = visualization_msgs::msg::Marker::ADD;
      sensor_marker.pose.position = to_point(sample->control_point);
      sensor_marker.pose.orientation.w = 1.0;
      sensor_marker.scale.x = repulsion_metric_marker_dot_diameter_;
      sensor_marker.scale.y = repulsion_metric_marker_dot_diameter_;
      sensor_marker.scale.z = repulsion_metric_marker_dot_diameter_;
      sensor_marker.color = repulsion_metric_color(metric_norm);
      markers.markers.push_back(sensor_marker);
    }

    repulsion_metric_pub_->publish(markers);
  }

  static Eigen::Vector3d compute_tcp_acceleration(
    const KinematicsContext & context,
    const JointVector & qdd)
  {
    return context.tcp_jacobian * qdd + context.tcp_curvature;
  }

  void publish_tangent_escape_rmp_data(const std::vector<double> & values)
  {
    if (!tangent_escape_rmp_data_pub_) {
      return;
    }

    std_msgs::msg::Float64MultiArray msg;
    msg.data = values.empty() ? std::vector<double>{0.0} : values;
    if (rt_tangent_escape_rmp_data_pub_) {
      rt_tangent_escape_rmp_data_pub_->tryPublish(msg);
    } else {
      tangent_escape_rmp_data_pub_->publish(msg);
    }
  }

  void publish_tangent_escape_dual_solve_data(
    const std::vector<double> & values)
  {
    if (!tangent_escape_dual_solve_data_pub_) {
      return;
    }

    std_msgs::msg::Float64MultiArray msg;
    msg.data = values.empty() ? std::vector<double>{0.0} : values;
    if (rt_tangent_escape_dual_solve_data_pub_) {
      rt_tangent_escape_dual_solve_data_pub_->tryPublish(msg);
    } else {
      tangent_escape_dual_solve_data_pub_->publish(msg);
    }
  }

  void publish_leaf_ablation_data(const std::vector<double> & values)
  {
    if (!leaf_ablation_data_pub_) {
      return;
    }

    std_msgs::msg::Float64MultiArray msg;
    msg.data = values.empty() ? std::vector<double>{0.0} : values;
    if (rt_leaf_ablation_data_pub_) {
      rt_leaf_ablation_data_pub_->tryPublish(msg);
    } else {
      leaf_ablation_data_pub_->publish(msg);
    }
  }

  void publish_stability_certificate_data(const std::vector<double> & values)
  {
    if (!stability_certificate_data_pub_) {
      return;
    }

    std_msgs::msg::Float64MultiArray msg;
    msg.data = values.empty() ? std::vector<double>{0.0} : values;
    if (rt_stability_certificate_data_pub_) {
      rt_stability_certificate_data_pub_->tryPublish(msg);
    } else {
      stability_certificate_data_pub_->publish(msg);
    }
  }

  void publish_tangent_escape_geometry_data(
    const KinematicsContext & context,
    const std::vector<std::optional<ObstacleSphere>> & proximity_sensor_obstacles)
  {
    if (!tangent_escape_geometry_data_pub_) {
      return;
    }

    const auto sample =
      make_closest_tangent_escape_geometry_sample(context, proximity_sensor_obstacles);
    std_msgs::msg::Float64MultiArray msg;
    if (!sample.has_value()) {
      msg.data = {0.0};
    } else {
      msg.data.reserve(36);
      msg.data.push_back(1.0);
      msg.data.push_back(static_cast<double>(sample->sensor_index));
      msg.data.push_back(sample->clearance);
      msg.data.push_back(sample->center_distance);
      append_vector3(msg.data, sample->control_point);
      append_vector3(msg.data, sample->obstacle_center);
      append_vector3(msg.data, sample->sensor_normal);
      append_vector3(msg.data, sample->obstacle_direction);
      append_vector3(msg.data, sample->collision_normal);
      append_vector3(msg.data, sample->tangent_bias);
      append_vector3(msg.data, sample->control_point_velocity);
      msg.data.push_back(sample->sensor_obstacle_dot);
      msg.data.push_back(sample->collision_obstacle_dot);
      msg.data.push_back(sample->bias_sensor_dot);
      msg.data.push_back(sample->bias_obstacle_dot);
      msg.data.push_back(sample->jacobian_frobenius_norm);
      msg.data.push_back(sample->sensor_normal_jacobian_norm);
      msg.data.push_back(sample->obstacle_direction_jacobian_norm);
      msg.data.push_back(sample->tangent_bias_jacobian_norm);
      msg.data.push_back(sample->velocity_obstacle_dot);
      msg.data.push_back(sample->velocity_tangent_dot);
    }

    if (rt_tangent_escape_geometry_data_pub_) {
      rt_tangent_escape_geometry_data_pub_->tryPublish(msg);
    } else {
      tangent_escape_geometry_data_pub_->publish(msg);
    }
  }

  void update_tcp_accel_visualization(const Eigen::Vector3d & tcp_accel)
  {
    TcpAccelerationSample sample;
    sample.acceleration = {tcp_accel.x(), tcp_accel.y(), tcp_accel.z()};
    sample.norm = tcp_accel.norm();
    sample.valid = true;
    tcp_accel_visualization_box_.set(sample);
  }

  void publish_rmp_accel_debug(
    const JointVector & qdd,
    const Eigen::Vector3d * tcp_accel)
  {
    if (rmp_joint_accel_pub_) {
      std_msgs::msg::Float64MultiArray msg;
      msg.data.assign(qdd.data(), qdd.data() + qdd.size());
      if (rt_rmp_joint_accel_pub_) {
        rt_rmp_joint_accel_pub_->tryPublish(msg);
      } else {
        rmp_joint_accel_pub_->publish(msg);
      }
    }

    if (rmp_tcp_accel_pub_ && tcp_accel) {
      std_msgs::msg::Float64MultiArray msg;
      msg.data = {tcp_accel->x(), tcp_accel->y(), tcp_accel->z()};
      if (rt_rmp_tcp_accel_pub_) {
        rt_rmp_tcp_accel_pub_->tryPublish(msg);
      } else {
        rmp_tcp_accel_pub_->publish(msg);
      }
    }
  }

  void clear_tcp_accel_visualization_sample()
  {
    tcp_accel_visualization_box_.set(TcpAccelerationSample{});
  }

  void publish_tcp_accel_marker(const KinematicsContext & context)
  {
    if (!tcp_accel_pub_) {
      return;
    }

    TcpAccelerationSample sample;
    tcp_accel_visualization_box_.get(sample);

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "base_link";
    marker.header.stamp = now();
    marker.ns = "tcp_accel";
    marker.id = 0;

    if (!sample.valid || sample.norm < tcp_accel_marker_min_norm_) {
      marker.action = visualization_msgs::msg::Marker::DELETE;
      tcp_accel_pub_->publish(marker);
      return;
    }

    Eigen::Vector3d acceleration(
      sample.acceleration[0],
      sample.acceleration[1],
      sample.acceleration[2]);
    const double acceleration_norm = acceleration.norm();
    if (acceleration_norm < std::numeric_limits<double>::epsilon()) {
      marker.action = visualization_msgs::msg::Marker::DELETE;
      tcp_accel_pub_->publish(marker);
      return;
    }

    const double normalized_length =
      std::clamp(acceleration_norm / tcp_accel_marker_norm_for_max_length_, 0.0, 1.0);
    const double visible_length =
      tcp_accel_marker_max_length_ * std::max(normalized_length, 0.15);
    const Eigen::Vector3d start = context.tcp_position;
    const Eigen::Vector3d end = start + acceleration.normalized() * visible_length;

    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.points.push_back(to_point(start));
    marker.points.push_back(to_point(end));
    marker.scale.x = 0.01;
    marker.scale.y = 0.03;
    marker.scale.z = std::clamp(visible_length * 0.35, 0.015, 0.045);
    marker.color.r = 0.05F;
    marker.color.g = 0.85F;
    marker.color.b = 1.0F;
    marker.color.a = 0.9F;
    tcp_accel_pub_->publish(marker);
  }

  void publish_tangent_escape_geometry_markers(const KinematicsContext & context)
  {
    if (!tangent_escape_geometry_marker_pub_) {
      return;
    }

    visualization_msgs::msg::MarkerArray markers;
    const auto stamp = now();
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header.frame_id = "base_link";
    clear_marker.header.stamp = stamp;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear_marker);

    auto color = [](float r, float g, float b, float a) {
        std_msgs::msg::ColorRGBA value;
        value.r = r;
        value.g = g;
        value.b = b;
        value.a = a;
        return value;
      };

    int marker_id = 0;
    auto append_sphere =
      [&](const std::string & name,
        const Eigen::Vector3d & position,
        double diameter,
        const std_msgs::msg::ColorRGBA & rgba)
      {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = stamp;
        marker.ns = name;
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position = to_point(position);
        marker.pose.orientation.w = 1.0;
        marker.scale.x = diameter;
        marker.scale.y = diameter;
        marker.scale.z = diameter;
        marker.color = rgba;
        markers.markers.push_back(marker);
      };

    auto append_arrow =
      [&](const std::string & name,
        const Eigen::Vector3d & start,
        const Eigen::Vector3d & direction,
        double length,
        const std_msgs::msg::ColorRGBA & rgba)
      {
        const double direction_norm = direction.norm();
        if (direction_norm <= 1e-9) {
          return;
        }
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = stamp;
        marker.ns = name;
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::ARROW;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.points.push_back(to_point(start));
        marker.points.push_back(to_point(start + direction / direction_norm * length));
        marker.scale.x = 0.008;
        marker.scale.y = 0.024;
        marker.scale.z = 0.032;
        marker.color = rgba;
        markers.markers.push_back(marker);
      };

    auto append_line =
      [&](const std::string & name,
        const Eigen::Vector3d & start,
        const Eigen::Vector3d & end,
        const std_msgs::msg::ColorRGBA & rgba)
      {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = stamp;
        marker.ns = name;
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.points.push_back(to_point(start));
        marker.points.push_back(to_point(end));
        marker.scale.x = 0.004;
        marker.color = rgba;
        markers.markers.push_back(marker);
      };

    auto append_label =
      [&](const TangentEscapeGeometrySample & sample)
      {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = stamp;
        marker.ns = "tangent_escape_geometry_labels";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position = to_point(sample.control_point + Eigen::Vector3d(0.0, 0.0, 0.04));
        marker.pose.orientation.w = 1.0;
        marker.scale.z = 0.035;
        marker.color = color(1.0F, 1.0F, 1.0F, 0.95F);
        const auto & sensor = RB10Model::sensor_control_points[sample.sensor_index];
        marker.text =
          std::string(sensor.frame_name) +
          " dot=" + std::to_string(sample.sensor_obstacle_dot).substr(0, 5);
        markers.markers.push_back(marker);
      };

    const auto samples = make_tangent_escape_geometry_samples(context);
    for (const auto & sample : samples) {
      const double length = tangent_escape_geometry_marker_length_;
      append_sphere(
        "tangent_escape_geometry_cp",
        sample.control_point,
        0.025,
        color(1.0F, 1.0F, 1.0F, 0.9F));
      append_sphere(
        "tangent_escape_geometry_obstacle",
        sample.obstacle_center,
        0.025,
        color(1.0F, 0.15F, 0.15F, 0.85F));
      append_line(
        "tangent_escape_geometry_cp_obstacle",
        sample.control_point,
        sample.obstacle_center,
        color(0.9F, 0.9F, 0.9F, 0.35F));
      append_arrow(
        "tangent_escape_geometry_sensor_normal",
        sample.control_point,
        sample.sensor_normal,
        length,
        color(0.05F, 0.35F, 1.0F, 0.9F));
      append_arrow(
        "tangent_escape_geometry_obstacle_direction",
        sample.control_point,
        sample.obstacle_direction,
        length,
        color(1.0F, 0.05F, 0.05F, 0.9F));
      append_arrow(
        "tangent_escape_geometry_collision_normal",
        sample.control_point,
        sample.collision_normal,
        length,
        color(1.0F, 0.55F, 0.0F, 0.85F));
      append_arrow(
        "tangent_escape_geometry_tangent_bias",
        sample.control_point,
        sample.tangent_bias,
        length,
        color(0.05F, 0.9F, 0.2F, 0.9F));
      append_arrow(
        "tangent_escape_geometry_cp_velocity",
        sample.control_point,
        sample.control_point_velocity,
        length,
        color(0.75F, 0.45F, 1.0F, 0.9F));
      append_label(sample);
    }

    std::vector<double> canonical;
    tangent_escape_visualization_box_.get(canonical);
    std::vector<double> dual_solve;
    tangent_escape_dual_visualization_box_.get(dual_solve);
    constexpr std::size_t kCanonicalMinimumSize = 56;
    if (
      canonical.size() >= kCanonicalMinimumSize &&
      std::isfinite(canonical[0]) &&
      canonical[0] > 0.5)
    {
      const Eigen::Vector3d selected_cp(canonical[11], canonical[12], canonical[13]);
      const Eigen::Vector3d selected_normal(canonical[17], canonical[18], canonical[19]);
      const Eigen::Vector3d selected_tangent(canonical[20], canonical[21], canonical[22]);
      const int escape_schema =
        std::isfinite(canonical[26]) ?
        static_cast<int>(std::llround(canonical[26])) :
        0;
      const bool risk_damped_schema = escape_schema == 7;
      const double metric = canonical[10];
      const double scalar_accel = canonical[9];
      const double z = canonical[54];
      const double lambda = canonical[45];
      const double velocity_reference = canonical[28];
      const double scalar_velocity = canonical[29];
      Eigen::Vector3d cp_acceleration_with_escape = Eigen::Vector3d::Zero();
      Eigen::Vector3d cp_acceleration_without_escape = Eigen::Vector3d::Zero();
      Eigen::Vector3d delta_cp_acceleration = Eigen::Vector3d::Zero();
      double delta_cp_tangent_acceleration = 0.0;
      double cp_acceleration_direction_change_deg =
        std::numeric_limits<double>::quiet_NaN();
      bool dual_solve_active = false;
      constexpr std::size_t kDualSolveSize = 43;
      if (
        dual_solve.size() >= kDualSolveSize &&
        std::isfinite(dual_solve[0]) &&
        dual_solve[0] > 0.5)
      {
        cp_acceleration_with_escape = Eigen::Vector3d(
          dual_solve[32], dual_solve[33], dual_solve[34]);
        cp_acceleration_without_escape = Eigen::Vector3d(
          dual_solve[35], dual_solve[36], dual_solve[37]);
        delta_cp_acceleration = Eigen::Vector3d(
          dual_solve[38], dual_solve[39], dual_solve[40]);
        delta_cp_tangent_acceleration = dual_solve[41];
        dual_solve_active =
          cp_acceleration_with_escape.allFinite() &&
          cp_acceleration_without_escape.allFinite() &&
          delta_cp_acceleration.allFinite() &&
          std::isfinite(delta_cp_tangent_acceleration);
        const double with_norm = cp_acceleration_with_escape.norm();
        const double without_norm = cp_acceleration_without_escape.norm();
        if (dual_solve_active && with_norm > 1e-6 && without_norm > 1e-6) {
          const double direction_cosine = std::clamp(
            cp_acceleration_with_escape.dot(cp_acceleration_without_escape) /
            (with_norm * without_norm),
            -1.0,
            1.0);
          cp_acceleration_direction_change_deg =
            radians_to_degrees(std::acos(direction_cosine));
        }
      }
      if (
        selected_cp.allFinite() &&
        selected_normal.allFinite() &&
        selected_tangent.allFinite() &&
        std::isfinite(metric) &&
        std::isfinite(scalar_accel) &&
        std::isfinite(z) &&
        std::isfinite(lambda) &&
        std::isfinite(velocity_reference) &&
        std::isfinite(scalar_velocity) &&
        std::isfinite(canonical[47]) &&
        std::isfinite(canonical[33]))
      {
        const int state_id = static_cast<int>(std::llround(canonical[47]));
        const int phase_id = static_cast<int>(std::llround(canonical[33]));
        const double metric_ratio =
          std::clamp(std::max(metric, 0.0) / (std::max(metric, 0.0) + 25.0), 0.0, 1.0);
        append_sphere(
          "tangent_escape_selected_cp",
          selected_cp,
          0.04 + 0.025 * metric_ratio,
          color(1.0F, 0.0F, 0.85F, 0.95F));
        append_arrow(
          "tangent_escape_selected_normal",
          selected_cp,
          selected_normal,
          0.13,
          color(1.0F, 0.55F, 0.0F, 0.95F));

        if (selected_tangent.norm() > 1e-9) {
          visualization_msgs::msg::Marker tangent_marker;
          tangent_marker.header.frame_id = "base_link";
          tangent_marker.header.stamp = stamp;
          tangent_marker.ns = "tangent_escape_selected_tangent";
          tangent_marker.id = marker_id++;
          tangent_marker.type = visualization_msgs::msg::Marker::ARROW;
          tangent_marker.action = visualization_msgs::msg::Marker::ADD;
          tangent_marker.points.push_back(to_point(selected_cp));
          tangent_marker.points.push_back(
            to_point(selected_cp + selected_tangent.normalized() * 0.18));
          tangent_marker.scale.x = 0.012 + 0.018 * metric_ratio;
          tangent_marker.scale.y = 0.035 + 0.020 * metric_ratio;
          tangent_marker.scale.z = 0.045 + 0.020 * metric_ratio;
          tangent_marker.color = color(1.0F, 0.0F, 0.85F, 0.98F);
          markers.markers.push_back(tangent_marker);

          append_arrow(
            "tangent_escape_scalar_acceleration",
            selected_cp,
            std::copysign(1.0, scalar_accel) * selected_tangent,
            0.04 + 0.12 * std::clamp(std::abs(scalar_accel) / 0.6, 0.0, 1.0),
            color(1.0F, 0.95F, 0.05F, 0.95F));
        }

        if (dual_solve_active) {
          constexpr double kComparisonArrowLength = 0.18;
          const Eigen::Vector3d offset_direction =
            selected_normal.norm() > 1e-9 ?
            selected_normal.normalized() : Eigen::Vector3d::UnitZ();
          const Eigen::Vector3d without_start =
            selected_cp - 0.012 * offset_direction;
          const Eigen::Vector3d with_start =
            selected_cp + 0.012 * offset_direction;

          append_arrow(
            "tangent_escape_cp_acceleration_without",
            without_start,
            cp_acceleration_without_escape,
            kComparisonArrowLength,
            color(0.05F, 0.40F, 1.0F, 0.98F));
          append_arrow(
            "tangent_escape_cp_acceleration_with",
            with_start,
            cp_acceleration_with_escape,
            kComparisonArrowLength,
            color(0.05F, 1.0F, 0.25F, 0.98F));

          if (
            cp_acceleration_without_escape.norm() > 1e-9 &&
            cp_acceleration_with_escape.norm() > 1e-9)
          {
            const Eigen::Vector3d without_tip =
              without_start +
              kComparisonArrowLength * cp_acceleration_without_escape.normalized();
            const Eigen::Vector3d with_tip =
              with_start +
              kComparisonArrowLength * cp_acceleration_with_escape.normalized();
            append_line(
              "tangent_escape_cp_acceleration_angle",
              without_tip,
              with_tip,
              color(1.0F, 0.95F, 0.05F, 0.90F));
          }

          if (delta_cp_acceleration.norm() > 1e-9) {
            append_arrow(
              "tangent_escape_delta_cp_acceleration",
              selected_cp,
              delta_cp_acceleration,
              std::clamp(0.12 * delta_cp_acceleration.norm(), 0.0, 0.18),
              color(0.0F, 0.9F, 1.0F, 0.98F));
          }
        }

        const std::array<const char *, 5> canonical_state_names{
          "OFF", "PREVENT", "RECOVERY", "RELEASE", "RESELECT"};
        const std::array<const char *, 5> risk_state_names{
          "OFF", "DRIVE", "MONITOR", "RELEASE", "RESELECT"};
        const auto & state_names =
          risk_damped_schema ? risk_state_names : canonical_state_names;
        const std::string state_name =
          state_id >= 0 && state_id < static_cast<int>(state_names.size()) ?
          state_names[static_cast<std::size_t>(state_id)] :
          "UNKNOWN";
        visualization_msgs::msg::Marker label;
        label.header.frame_id = "base_link";
        label.header.stamp = stamp;
        label.ns = "tangent_escape_selected_status";
        label.id = marker_id++;
        label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        label.action = visualization_msgs::msg::Marker::ADD;
        label.pose.position = to_point(selected_cp + Eigen::Vector3d(0.0, 0.0, 0.10));
        label.pose.orientation.w = 1.0;
        label.scale.z = 0.035;
        label.color = color(1.0F, 1.0F, 1.0F, 0.98F);
        // Escape v2 has no activation filter, no handoff ramp, and no state
        // machine, so the canonical/risk labels would be pure fiction here.
        const bool ode_schema = escape_schema == 9;
        if (ode_schema && canonical.size() >= 56) {
          label.text =
            "ESC ODE cp=" + std::to_string(static_cast<int>(canonical[1])) +
            " d=" + std::to_string(canonical[2]).substr(0, 6) +
            " M=" + std::to_string(metric).substr(0, 7) +
            " a=" + std::to_string(scalar_accel).substr(0, 7) +
            " aPush=" + std::to_string(canonical[28]).substr(0, 7) +
            " th=" + std::to_string(canonical[3]).substr(0, 6) +
            " bud=" + std::to_string(canonical[6]).substr(0, 5) +
            " free=" + std::to_string(canonical[15]).substr(0, 5) +
            " v=" + std::to_string(scalar_velocity).substr(0, 6);
        } else if (risk_damped_schema && canonical.size() >= 68) {
          label.text =
            "ESC RISK " + state_name +
            " phase=" + std::to_string(phase_id) +
            " M=" + std::to_string(metric).substr(0, 6) +
            " B=" + std::to_string(z).substr(0, 5) +
            " lambda=" + std::to_string(lambda).substr(0, 5) +
            " d_dot=" + std::to_string(canonical[61]).substr(0, 7) +
            " aRisk=" + std::to_string(canonical[65]).substr(0, 6) +
            " a=" + std::to_string(scalar_accel).substr(0, 6) +
            " v=" + std::to_string(scalar_velocity).substr(0, 6);
        } else {
          label.text =
            "ESC VEL " + state_name +
            " phase=" + std::to_string(phase_id) +
            " M=" + std::to_string(metric).substr(0, 6) +
            " z=" + std::to_string(z).substr(0, 5) +
            " lambda=" + std::to_string(lambda).substr(0, 5) +
            " v=" + std::to_string(scalar_velocity).substr(0, 6) +
            "/" + std::to_string(velocity_reference).substr(0, 6);
        }
        if (dual_solve_active) {
          label.text +=
            "\nCP blue(no)=" +
            std::to_string(cp_acceleration_without_escape.norm()).substr(0, 6) +
            " green(with)=" +
            std::to_string(cp_acceleration_with_escape.norm()).substr(0, 6) +
            " angle=" +
            (
              std::isfinite(cp_acceleration_direction_change_deg) ?
              std::to_string(cp_acceleration_direction_change_deg).substr(0, 6) + "deg" :
              std::string("n/a")) +
            "\ndCP=" +
            std::to_string(delta_cp_acceleration.norm()).substr(0, 6) +
            " dT=" +
            std::to_string(delta_cp_tangent_acceleration).substr(0, 6);
        }
        markers.markers.push_back(label);
      }
    }

    tangent_escape_geometry_marker_pub_->publish(markers);
  }

  std::array<double, 6> parse_joint_limit_degrees(
    const std::string & parameter_name,
    const std::array<double, 6> & fallback) const
  {
    const auto values = get_parameter(parameter_name).as_double_array();
    if (values.empty()) {
      return fallback;
    }
    if (values.size() != fallback.size()) {
      throw std::runtime_error(parameter_name + " must contain exactly 6 values");
    }

    std::array<double, 6> limits{};
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (!std::isfinite(values[index])) {
        throw std::runtime_error(parameter_name + " must contain finite values");
      }
      limits[index] = degrees_to_radians(values[index]);
    }
    return limits;
  }

  JointVector clamp_joint_positions(const JointVector & q) const
  {
    JointVector clamped = q;
    for (int index = 0; index < clamped.size(); ++index) {
      const auto limit_index = static_cast<std::size_t>(index);
      clamped[index] = std::clamp(
        clamped[index],
        joint_lower_limits_[limit_index],
        joint_upper_limits_[limit_index]);
    }
    return clamped;
  }

  ProductionCommandMapConfig production_command_map_config(double dt) const
  {
    ProductionCommandMapConfig config;
    config.dt = dt;
    config.max_joint_accel = max_joint_accel_;
    config.preserve_joint_accel_direction = preserve_joint_accel_direction_;
    config.command_mode =
      command_mode_ == CommandMode::kVelocity ?
      ProductionCommandMode::kVelocity :
      ProductionCommandMode::kPosition;
    config.command_guard_max_step_rad = command_guard_max_step_rad_;
    config.command_guard_max_velocity_rad_s = command_guard_max_velocity_rad_s_;
    config.command_guard_velocity_slew_enabled =
      command_guard_velocity_slew_enabled_;
    config.command_guard_max_accel_rad_s2 = command_guard_max_accel_rad_s2_;
    config.predictive_joint_limit_guard = predictive_joint_limit_guard_;
    config.joint_lower_limits = joint_lower_limits_;
    config.joint_upper_limits = joint_upper_limits_;
    config.joint_limit_buffers = joint_limit_buffers_;
    return config;
  }

  void run_production_stability_analysis_once(
    const RobotState & seed_state,
    const GoalTarget & goal_target)
  {
    if (
      !production_stability_analysis_enabled_ ||
      production_stability_analysis_completed_)
    {
      return;
    }
    if (
      get_parameter("production_stability_wait_for_external_goal").as_bool() &&
      !external_goal_received_.load())
    {
      return;
    }
    production_stability_analysis_completed_ = true;

    if (get_parameter("backend_mode").as_string() != "simulation") {
      RCLCPP_ERROR(
        get_logger(),
        "Production stability analysis is restricted to the socket-free simulation backend");
      return;
    }

    const double dt = 1.0 / std::max(control_rate_hz_, 1e-9);
    const ProductionCommandMapConfig command_config =
      production_command_map_config(dt);
    std::unordered_map<std::string, Eigen::Vector3d> vector_targets;
    vector_targets.emplace("goal", goal_target.position);
    vector_targets.emplace("body_goal", body_goal_);
    const Eigen::Matrix3d goal_rotation =
      goal_target.orientation.normalized().toRotationMatrix();
    vector_targets.emplace("orientation_goal_x", goal_rotation.col(0));
    vector_targets.emplace("orientation_goal_y", goal_rotation.col(1));
    vector_targets.emplace("orientation_goal_z", goal_rotation.col(2));

    // NORMAL-mode certification uses a fixed obstacle-free environment. The
    // Escape leaf remains present in the configured graph, but its metric and
    // force are exactly zero because there is no active proximity pair.
    const std::vector<ObstacleSphere> normal_mode_obstacles;
    const std::unordered_map<std::string, ExternalRmpFeature> no_external_rmps;

    const auto raw_acceleration =
      [this, &vector_targets, &normal_mode_obstacles, &no_external_rmps](
      const ProductionJointState & state)
      {
        RmpBatchInput input;
        input.q = state.q;
        input.qd = state.qd;
        input.vector_targets = vector_targets;
        input.obstacles = normal_mode_obstacles;
        input.external_rmps = no_external_rmps;
        const auto results = solver_->solve_batch({input});
        if (results.size() != 1U) {
          throw std::runtime_error(
                  "Production stability analysis expected one batch result");
        }
        return results.front().qdd;
      };

    const auto static_acceleration =
      [&raw_acceleration](const ProductionJointVector & q)
      {
        ProductionJointState state;
        state.q = q;
        state.qd.setZero();
        // Search the smooth root equation before the production saturation.
        // A valid equilibrium has qdd=0, so the limiter is inactive there;
        // including it here can flatten the Newton Jacobian when the seed is
        // far from the goal. The certificate below still linearizes the exact
        // limited and guarded production one-cycle map.
        return raw_acceleration(state);
      };

    EquilibriumSearchOptions equilibrium_options;
    equilibrium_options.maximum_iterations = static_cast<int>(
      get_parameter("production_stability_equilibrium_max_iterations").as_int());
    equilibrium_options.acceleration_tolerance =
      get_parameter("production_stability_equilibrium_tolerance").as_double();
    const double position_difference_step =
      get_parameter("production_stability_position_difference_step").as_double();
    const double velocity_difference_step =
      get_parameter("production_stability_velocity_difference_step").as_double();
    equilibrium_options.finite_difference_step = position_difference_step;

    RCLCPP_WARN(
      get_logger(),
      "Running one-shot production-map stability analysis in local simulation; "
      "the control loop will pause while finite-difference solves are evaluated");

    ProductionJointState settled_seed;
    settled_seed.q = seed_state.q;
    settled_seed.qd.setZero();
    const int settle_max_steps = std::max(
      static_cast<int>(
        get_parameter("production_stability_settle_max_steps").as_int()),
      0);
    const double settle_velocity_tolerance = std::max(
      get_parameter(
        "production_stability_settle_velocity_tolerance").as_double(),
      0.0);
    const double settle_acceleration_tolerance = std::max(
      get_parameter(
        "production_stability_settle_acceleration_tolerance").as_double(),
      0.0);
    int settle_steps = 0;
    bool settle_converged = false;
    std::size_t settle_nonsmooth_steps = 0U;
    double settle_final_raw_acceleration_norm =
      std::numeric_limits<double>::infinity();
    for (int step_index = 0; step_index < settle_max_steps; ++step_index) {
      ProductionCommandMemory memory;
      memory.previous_velocity_command = settled_seed.qd;
      memory.previous_velocity_command_initialized = true;
      const ProductionJointVector raw_qdd = raw_acceleration(settled_seed);
      const ProductionCommandStep step = production_command_step(
        settled_seed,
        raw_qdd,
        command_config,
        memory);
      ++settle_steps;
      if (step.diagnostics.any_nonsmooth_branch_active()) {
        ++settle_nonsmooth_steps;
      }
      settled_seed = step.state;
      settle_final_raw_acceleration_norm = raw_acceleration(settled_seed).norm();
      if (
        settle_steps >= 100 &&
        settled_seed.qd.norm() <= settle_velocity_tolerance &&
        settle_final_raw_acceleration_norm <= settle_acceleration_tolerance)
      {
        settle_converged = true;
        break;
      }
    }

    const EquilibriumSearchResult equilibrium = find_production_equilibrium(
      static_acceleration,
      settled_seed.q,
      joint_lower_limits_,
      joint_upper_limits_,
      equilibrium_options);

    ProductionJointState equilibrium_state;
    equilibrium_state.q = equilibrium.q;
    equilibrium_state.qd.setZero();
    RmpBatchInput equilibrium_input;
    equilibrium_input.q = equilibrium_state.q;
    equilibrium_input.qd = equilibrium_state.qd;
    equilibrium_input.vector_targets = vector_targets;
    equilibrium_input.obstacles = normal_mode_obstacles;
    equilibrium_input.external_rmps = no_external_rmps;
    const auto equilibrium_solve_results =
      solver_->solve_batch({equilibrium_input});
    if (equilibrium_solve_results.size() != 1U) {
      throw std::runtime_error(
              "Production stability analysis could not evaluate equilibrium root form");
    }
    const auto & equilibrium_solve = equilibrium_solve_results.front();
    const auto symmetric_root_metric =
      0.5 * (equilibrium_solve.metric + equilibrium_solve.metric.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>>
    root_metric_eigen_solver(symmetric_root_metric);
    const double root_metric_min_eigenvalue =
      root_metric_eigen_solver.info() == Eigen::Success ?
      root_metric_eigen_solver.eigenvalues().minCoeff() :
      -std::numeric_limits<double>::infinity();
    const double root_metric_symmetry_residual =
      (equilibrium_solve.metric - equilibrium_solve.metric.transpose()).norm();
    const double root_resolve_residual =
      (
      equilibrium_solve.metric * equilibrium_solve.qdd -
      equilibrium_solve.force).norm();
    const bool equilibrium_natural_form_valid =
      equilibrium_solve.metric.allFinite() &&
      equilibrium_solve.force.allFinite() &&
      equilibrium_solve.qdd.allFinite() &&
      root_metric_min_eigenvalue > 0.0 &&
      root_metric_symmetry_residual <= 1e-9 &&
      root_resolve_residual <=
      1e-7 * (1.0 + equilibrium_solve.force.norm());

    struct EscapeNaturalFormAggregate
    {
      bool available{false};
      bool inactive_zero{false};
      bool all_feasible_sensors_activated{false};
      bool all_active_samples_valid{false};
      std::size_t sensor_count{0U};
      std::size_t feasible_sensor_count{0U};
      std::size_t activated_sensor_count{0U};
      std::size_t active_sample_count{0U};
      std::size_t failed_active_sample_count{0U};
      double inactive_metric_norm{std::numeric_limits<double>::infinity()};
      double inactive_force_norm{std::numeric_limits<double>::infinity()};
      double minimum_contribution_eigenvalue{
        std::numeric_limits<double>::infinity()};
      double maximum_contribution_symmetry_residual{0.0};
      double maximum_tangent_normal_dot_abs{0.0};
      double minimum_root_metric_eigenvalue{
        std::numeric_limits<double>::infinity()};
      double maximum_regularized_resolve_residual{0.0};
      int maximum_contribution_rank{0};
      std::string error;
      YAML::Node sensor_probes{YAML::NodeType::Sequence};
    };
    EscapeNaturalFormAggregate escape_natural_form_audit;

    try {
      if (get_parameter("solver_backend").as_string() != "pinocchio_direct") {
        throw std::runtime_error(
                "Active Escape natural-form audit requires solver_backend=pinocchio_direct");
      }

      EigenRmpConfig enabled_audit_config = build_solver_config();
      enabled_audit_config.tangent_escape.enabled = true;
      enabled_audit_config.enable_tangent_escape_diagnostics = true;
      enabled_audit_config.enable_tangent_escape_dual_solve_diagnostics = false;
      enabled_audit_config.enable_leaf_ablation_diagnostics = false;
      enabled_audit_config.enable_stability_certificate_diagnostics = false;
      enabled_audit_config.escape_energy_certificate.enabled = false;
      EigenRmpConfig base_audit_config = enabled_audit_config;
      base_audit_config.tangent_escape.enabled = false;
      base_audit_config.enable_tangent_escape_diagnostics = false;

      const std::string urdf_path = resolve_robot_urdf_path();
      const PinocchioModel audit_model(urdf_path);
      const JointVector audit_q = seed_state.q;
      const JointVector audit_qd = JointVector::Zero();
      const auto audit_context = audit_model.forward_context(audit_q, audit_qd);
      PinocchioDirectRmpSolver base_audit_solver(base_audit_config, urdf_path);

      const auto audit_targets =
        [this, &goal_target, &audit_context](const Eigen::Vector3d & position)
        {
          std::unordered_map<std::string, Eigen::Vector3d> targets;
          targets.emplace("goal", position);
          targets.emplace("body_goal", body_goal_);
          const Eigen::Matrix3d orientation =
            goal_target.orientation.normalized().toRotationMatrix();
          targets.emplace("orientation_goal_x", orientation.col(0));
          targets.emplace("orientation_goal_y", orientation.col(1));
          targets.emplace("orientation_goal_z", orientation.col(2));
          return targets;
        };

      const auto fixed_audit_targets = audit_targets(goal_target.position);
      PinocchioDirectRmpSolver inactive_enabled_solver(
        enabled_audit_config, urdf_path);
      const auto inactive_base = base_audit_solver.solve(
        audit_q, audit_qd, fixed_audit_targets, {});
      const auto inactive_enabled = inactive_enabled_solver.solve(
        audit_q, audit_qd, fixed_audit_targets, {});
      escape_natural_form_audit.inactive_metric_norm =
        (inactive_enabled.metric - inactive_base.metric).norm();
      escape_natural_form_audit.inactive_force_norm =
        (inactive_enabled.force - inactive_base.force).norm();
      escape_natural_form_audit.inactive_zero =
        escape_natural_form_audit.inactive_metric_norm <= 1e-12 &&
        escape_natural_form_audit.inactive_force_norm <= 1e-12;

      const std::array<Eigen::Vector3d, 6> probe_directions{
        Eigen::Vector3d::UnitX(),
        -Eigen::Vector3d::UnitX(),
        Eigen::Vector3d::UnitY(),
        -Eigen::Vector3d::UnitY(),
        Eigen::Vector3d::UnitZ(),
        -Eigen::Vector3d::UnitZ()};
      const double obstacle_radius = 0.04;
      const double probe_clearance =
        enabled_audit_config.tangent_escape.safe_distance +
        0.25 *
        (
        enabled_audit_config.tangent_escape.influence_distance -
        enabled_audit_config.tangent_escape.safe_distance);
      const int audit_cycles = std::clamp(
        static_cast<int>(std::ceil(
            enabled_audit_config.tangent_escape.handoff_duration /
            std::max(enabled_audit_config.tangent_escape.control_dt, 1e-4))) + 3,
        4,
        256);

      escape_natural_form_audit.sensor_count =
        audit_context.control_points.size();
      for (std::size_t sensor_index = 0;
        sensor_index < audit_context.control_points.size();
        ++sensor_index)
      {
        double best_predicted_speed = -1.0;
        Eigen::Vector3d best_obstacle_direction = Eigen::Vector3d::UnitX();
        std::unordered_map<std::string, Eigen::Vector3d> best_targets;
        for (const auto & direction : probe_directions) {
          const auto targets = audit_targets(
            audit_context.tcp_position + 0.20 * direction);
          const auto nominal_result = base_audit_solver.solve(
            audit_q, audit_qd, targets, {});
          const Eigen::Vector3d predicted_motion =
            audit_context.control_point_jacobians[sensor_index] *
            (
            audit_qd +
            enabled_audit_config.tangent_escape.nominal_prediction_dt *
            nominal_result.qdd);
          if (predicted_motion.norm() > best_predicted_speed) {
            best_predicted_speed = predicted_motion.norm();
            if (predicted_motion.norm() > 1e-12) {
              best_obstacle_direction = predicted_motion.normalized();
            }
            best_targets = targets;
          }
        }

        YAML::Node sensor_probe;
        sensor_probe["index"] = sensor_index;
        sensor_probe["name"] =
          RB10Model::sensor_control_points[sensor_index].frame_name;
        sensor_probe["best_nominal_speed_m_s"] = best_predicted_speed;
        const bool feasible =
          best_predicted_speed >=
          std::max(enabled_audit_config.tangent_escape.intent_on_speed, 1e-9);
        sensor_probe["feasible_intent"] = feasible;
        if (!feasible || best_targets.empty()) {
          sensor_probe["activated"] = false;
          sensor_probe["passed"] = false;
          sensor_probe["reason"] = "nominal_intent_below_entry_threshold";
          escape_natural_form_audit.sensor_probes.push_back(sensor_probe);
          continue;
        }
        ++escape_natural_form_audit.feasible_sensor_count;

        ObstacleSphere obstacle;
        const auto & point = audit_context.control_points[sensor_index];
        obstacle.center =
          point.position +
          (
          point.radius +
          obstacle_radius +
          probe_clearance +
          enabled_audit_config.tangent_escape.clearance_margin) *
          best_obstacle_direction;
        obstacle.radius = obstacle_radius;
        obstacle.proximity_control_point_index =
          static_cast<int>(sensor_index);
        obstacle.source_id =
          static_cast<std::int64_t>(70000U + sensor_index);

        const auto base_result = base_audit_solver.solve(
          audit_q, audit_qd, best_targets, {obstacle});
        PinocchioDirectRmpSolver enabled_solver(
          enabled_audit_config, urdf_path);
        bool sensor_activated = false;
        bool sensor_passed = true;
        TangentEscapeNaturalFormAudit last_audit;
        for (int cycle = 0; cycle < audit_cycles; ++cycle) {
          const auto enabled_result = enabled_solver.solve(
            audit_q, audit_qd, best_targets, {obstacle});
          const auto & debug = enabled_result.tangent_escape_rmp_data;
          if (
            debug.size() < 23U ||
            debug[0] <= 0.5 ||
            debug[10] <= 1e-10)
          {
            continue;
          }

          sensor_activated = true;
          Eigen::Vector3d normal(debug[17], debug[18], debug[19]);
          Eigen::Vector3d tangent(debug[20], debug[21], debug[22]);
          last_audit = audit_tangent_escape_natural_form(
            base_result.metric,
            base_result.force,
            enabled_result.metric,
            enabled_result.force,
            enabled_result.qdd,
            enabled_audit_config.solve_offset,
            tangent,
            normal);
          ++escape_natural_form_audit.active_sample_count;
          if (!last_audit.natural_form_compatible) {
            ++escape_natural_form_audit.failed_active_sample_count;
            sensor_passed = false;
          }
          escape_natural_form_audit.minimum_contribution_eigenvalue =
            std::min(
            escape_natural_form_audit.minimum_contribution_eigenvalue,
            last_audit.contribution_min_eigenvalue);
          escape_natural_form_audit.maximum_contribution_symmetry_residual =
            std::max(
            escape_natural_form_audit.maximum_contribution_symmetry_residual,
            last_audit.contribution_symmetry_residual);
          escape_natural_form_audit.maximum_tangent_normal_dot_abs =
            std::max(
            escape_natural_form_audit.maximum_tangent_normal_dot_abs,
            last_audit.tangent_normal_dot_abs);
          escape_natural_form_audit.minimum_root_metric_eigenvalue =
            std::min(
            escape_natural_form_audit.minimum_root_metric_eigenvalue,
            last_audit.root_metric_min_eigenvalue);
          escape_natural_form_audit.maximum_regularized_resolve_residual =
            std::max(
            escape_natural_form_audit.maximum_regularized_resolve_residual,
            last_audit.regularized_root_resolve_residual);
          escape_natural_form_audit.maximum_contribution_rank =
            std::max(
            escape_natural_form_audit.maximum_contribution_rank,
            last_audit.contribution_numerical_rank);
        }

        if (sensor_activated) {
          ++escape_natural_form_audit.activated_sensor_count;
        }
        sensor_probe["activated"] = sensor_activated;
        sensor_probe["passed"] = sensor_activated && sensor_passed;
        sensor_probe["contribution_rank"] =
          last_audit.contribution_numerical_rank;
        sensor_probe["contribution_min_eigenvalue"] =
          last_audit.contribution_min_eigenvalue;
        sensor_probe["contribution_symmetry_residual"] =
          last_audit.contribution_symmetry_residual;
        sensor_probe["tangent_normal_dot_abs"] =
          last_audit.tangent_normal_dot_abs;
        sensor_probe["root_metric_min_eigenvalue"] =
          last_audit.root_metric_min_eigenvalue;
        sensor_probe["regularized_resolve_residual"] =
          last_audit.regularized_root_resolve_residual;
        if (!sensor_activated) {
          sensor_probe["reason"] = "no_active_leaf_sample";
        } else if (!sensor_passed) {
          sensor_probe["reason"] = "natural_form_check_failed";
        } else {
          sensor_probe["reason"] = "passed";
        }
        escape_natural_form_audit.sensor_probes.push_back(sensor_probe);
      }

      escape_natural_form_audit.all_feasible_sensors_activated =
        escape_natural_form_audit.feasible_sensor_count > 0U &&
        escape_natural_form_audit.activated_sensor_count ==
        escape_natural_form_audit.feasible_sensor_count;
      escape_natural_form_audit.all_active_samples_valid =
        escape_natural_form_audit.active_sample_count > 0U &&
        escape_natural_form_audit.failed_active_sample_count == 0U;
      escape_natural_form_audit.available = true;
    } catch (const std::exception & error) {
      escape_natural_form_audit.error = error.what();
      RCLCPP_ERROR(
        get_logger(),
        "Active Escape natural-form audit failed: %s",
        error.what());
    }

    const auto production_step =
      [&raw_acceleration, &command_config](const ProductionJointState & state)
      {
        ProductionCommandMemory memory;
        // Under the ideal inner-loop tracking assumption, the velocity applied
        // in the previous cycle is the current velocity state.
        memory.previous_velocity_command = state.qd;
        memory.previous_velocity_command_initialized = true;
        return production_command_step(
          state,
          raw_acceleration(state),
          command_config,
          memory);
      };

    const DiscreteLyapunovResult certificate =
      analyze_production_local_stability(
      production_step,
      equilibrium_state,
      position_difference_step,
      velocity_difference_step);

    std::vector<double> sample_radii =
      get_parameter("production_stability_sample_radii").as_double_array();
    sample_radii.erase(
      std::remove_if(
        sample_radii.begin(),
        sample_radii.end(),
        [](double radius) {
          return !std::isfinite(radius) || radius <= 0.0;
        }),
      sample_radii.end());
    std::sort(sample_radii.begin(), sample_radii.end());
    const int random_direction_count = static_cast<int>(
      get_parameter(
        "production_stability_random_directions_per_radius").as_int());
    std::vector<NonlinearLyapunovSample> samples;
    if (
      certificate.lyapunov_solved &&
      certificate.lyapunov_positive_definite)
    {
      samples = sample_production_lyapunov_region(
        production_step,
        equilibrium_state,
        certificate.lyapunov_matrix,
        sample_radii,
        random_direction_count);
    }

    const auto equilibrium_context = RB10Model::forward_context(
      equilibrium.q,
      ControlPointJacobianMode::kNone);
    const auto floor_metrics = compute_floor_safety_metrics(equilibrium_context);
    const bool equilibrium_floor_guard_inactive =
      floor_metrics.min_link_z >= workspace_floor_z_ &&
      floor_metrics.min_joint_z >= workspace_floor_z_ &&
      floor_metrics.min_control_point_z >= workspace_floor_z_ &&
      floor_metrics.min_body_obstacle_z >= workspace_floor_z_ &&
      equilibrium_context.link_positions[RB10Model::LINK6].z() >= min_link6_z_ &&
      equilibrium_context.link_positions[RB10Model::TCP_GRIPPER].z() >= min_tcp_z_;

    double largest_sampled_decrease_radius = 0.0;
    YAML::Node radius_summaries(YAML::NodeType::Sequence);
    for (double radius : sample_radii) {
      std::size_t count = 0U;
      std::size_t positive_count = 0U;
      std::size_t nonsmooth_count = 0U;
      double maximum_delta = -std::numeric_limits<double>::infinity();
      double maximum_relative_delta = -std::numeric_limits<double>::infinity();
      for (const auto & sample : samples) {
        if (std::abs(sample.radius - radius) > 1e-15) {
          continue;
        }
        ++count;
        maximum_delta = std::max(maximum_delta, sample.delta_value);
        maximum_relative_delta =
          std::max(maximum_relative_delta, sample.relative_delta);
        if (sample.delta_value >= 0.0) {
          ++positive_count;
        }
        if (sample.nonsmooth_branch_active) {
          ++nonsmooth_count;
        }
      }
      const bool radius_passed =
        count > 0U &&
        positive_count == 0U &&
        nonsmooth_count == 0U &&
        std::isfinite(maximum_delta);
      if (radius_passed) {
        largest_sampled_decrease_radius = radius;
      }
      YAML::Node summary;
      summary["radius"] = radius;
      summary["sample_count"] = count;
      summary["strict_decrease_count"] = count - positive_count;
      summary["nondecrease_count"] = positive_count;
      summary["nonsmooth_branch_count"] = nonsmooth_count;
      summary["maximum_delta_v"] = maximum_delta;
      summary["maximum_relative_delta_v"] = maximum_relative_delta;
      summary["passed"] = radius_passed;
      radius_summaries.push_back(summary);
    }

    const double equilibrium_map_tolerance =
      std::max(10.0 * equilibrium_options.acceleration_tolerance * dt, 1e-9);
    const bool formal_normal_certificate =
      equilibrium.converged &&
      certificate.linearization_finite &&
      certificate.schur_stable &&
      certificate.lyapunov_solved &&
      certificate.lyapunov_positive_definite &&
      certificate.lyapunov_decrease_strict &&
      certificate.residual_norm <= 1e-6 &&
      certificate.equilibrium_map_residual_norm <= equilibrium_map_tolerance &&
      !certificate.equilibrium_nonsmooth_branch_active &&
      !certificate.perturbation_nonsmooth_branch_active &&
      equilibrium_floor_guard_inactive;

    YAML::Node report;
    report["schema_version"] = 1;
    report["analysis_type"] =
      "exact_production_discrete_map_local_normal_mode";
    report["timestamp"] = file_timestamp();
    report["scope"]["mode"] = "NORMAL";
    report["scope"]["environment"] = "fixed_goal_static_obstacle_free";
    report["scope"]["external_rmps_included"] = false;
    report["scope"]["escape_graph_configured"] = tangent_escape_rmp_enabled_;
    report["scope"]["escape_active_at_equilibrium"] = false;
    report["assumptions"].push_back(
      "The RB10 inner velocity loop tracks each guarded velocity command within one sample.");
    report["assumptions"].push_back(
      "Goal, orientation target, body target, and environment remain fixed.");
    report["assumptions"].push_back(
      "The floor safety hold, acceleration limiter, and command guard stay inactive in the certified neighborhood.");
    report["assumptions"].push_back(
      "Sensor and model uncertainty are outside this local nominal certificate.");

    report["configuration"]["control_rate_hz"] = control_rate_hz_;
    report["configuration"]["dt_s"] = dt;
    report["configuration"]["solver_backend"] =
      get_parameter("solver_backend").as_string();
    report["configuration"]["command_mode"] =
      command_mode_to_string(command_mode_);
    report["configuration"]["max_joint_accel_rad_s2"] = max_joint_accel_;
    report["configuration"]["preserve_joint_accel_direction"] =
      preserve_joint_accel_direction_;
    report["configuration"]["max_joint_velocity_rad_s"] =
      command_guard_max_velocity_rad_s_;
    report["configuration"]["velocity_slew_enabled"] =
      command_guard_velocity_slew_enabled_;
    report["configuration"]["predictive_joint_limit_guard"] =
      predictive_joint_limit_guard_;
    report["configuration"]["root_solve_offset"] =
      get_parameter("root_solve_offset").as_double();
    report["configuration"]["collision_policy"] =
      get_parameter("collision_policy").as_string();
    report["configuration"]["joint_limit_policy"] =
      get_parameter("joint_limit_policy").as_string();
    report["configuration"]["tangent_escape_enabled"] =
      tangent_escape_rmp_enabled_;

    YAML::Node goal_position(YAML::NodeType::Sequence);
    goal_position.push_back(goal_target.position.x());
    goal_position.push_back(goal_target.position.y());
    goal_position.push_back(goal_target.position.z());
    report["fixed_inputs"]["goal_position"] = goal_position;
    YAML::Node goal_quaternion(YAML::NodeType::Sequence);
    goal_quaternion.push_back(goal_target.orientation.x());
    goal_quaternion.push_back(goal_target.orientation.y());
    goal_quaternion.push_back(goal_target.orientation.z());
    goal_quaternion.push_back(goal_target.orientation.w());
    report["fixed_inputs"]["goal_quaternion_xyzw"] = goal_quaternion;
    YAML::Node body_goal_node(YAML::NodeType::Sequence);
    body_goal_node.push_back(body_goal_.x());
    body_goal_node.push_back(body_goal_.y());
    body_goal_node.push_back(body_goal_.z());
    report["fixed_inputs"]["body_goal"] = body_goal_node;

    report["equilibrium"]["converged"] = equilibrium.converged;
    report["equilibrium"]["iterations"] = equilibrium.iterations;
    report["equilibrium"]["settle_initialization"]["converged"] =
      settle_converged;
    report["equilibrium"]["settle_initialization"]["steps"] = settle_steps;
    report["equilibrium"]["settle_initialization"]["final_qd_norm_rad_s"] =
      settled_seed.qd.norm();
    report["equilibrium"]["settle_initialization"]["final_raw_qdd_norm_rad_s2"] =
      settle_final_raw_acceleration_norm;
    report["equilibrium"]["settle_initialization"]["nonsmooth_steps"] =
      settle_nonsmooth_steps;
    report["equilibrium"]["q_rad"] = yaml_joint_vector(equilibrium.q);
    report["equilibrium"]["qdd_rad_s2"] =
      yaml_joint_vector(equilibrium.acceleration);
    report["equilibrium"]["qdd_norm_rad_s2"] =
      equilibrium.acceleration_norm;
    report["equilibrium"]["map_residual_norm"] =
      certificate.equilibrium_map_residual_norm;
    report["equilibrium"]["floor_guard_inactive"] =
      equilibrium_floor_guard_inactive;
    report["equilibrium"]["min_link_z_m"] = floor_metrics.min_link_z;
    report["equilibrium"]["min_joint_z_m"] = floor_metrics.min_joint_z;
    report["equilibrium"]["min_control_point_z_m"] =
      floor_metrics.min_control_point_z;
    report["equilibrium"]["min_body_obstacle_z_m"] =
      floor_metrics.min_body_obstacle_z;

    report["linearization"]["jacobian"] =
      yaml_matrix(certificate.jacobian);
    YAML::Node eigenvalues(YAML::NodeType::Sequence);
    for (const auto & eigenvalue : certificate.eigenvalues) {
      YAML::Node value;
      value["real"] = eigenvalue.x();
      value["imag"] = eigenvalue.y();
      value["magnitude"] = eigenvalue.norm();
      eigenvalues.push_back(value);
    }
    report["linearization"]["eigenvalues"] = eigenvalues;
    report["linearization"]["spectral_radius"] =
      certificate.spectral_radius;
    report["linearization"]["schur_stable"] = certificate.schur_stable;
    report["linearization"]["equilibrium_nonsmooth_branch_active"] =
      certificate.equilibrium_nonsmooth_branch_active;
    report["linearization"]["perturbation_nonsmooth_branch_active"] =
      certificate.perturbation_nonsmooth_branch_active;

    report["rmp_natural_form"]["root_metric_min_eigenvalue"] =
      root_metric_min_eigenvalue;
    report["rmp_natural_form"]["root_metric_symmetry_residual"] =
      root_metric_symmetry_residual;
    report["rmp_natural_form"]["root_resolve_residual"] =
      root_resolve_residual;
    report["rmp_natural_form"]["equilibrium_natural_form_valid"] =
      equilibrium_natural_form_valid;
    report["rmp_natural_form"]["escape_leaf_formula"]["task_metric"] =
      "m_y = m_t * lambda * B >= 0";
    report["rmp_natural_form"]["escape_leaf_formula"]["task_force"] =
      "f_y = m_y * (a_y - Jdot_y * qdot)";
    report["rmp_natural_form"]["escape_leaf_formula"]["root_pullback"] =
      "DeltaM = J_y^T * m_y * J_y, Deltaf = J_y^T * f_y";
    report["rmp_natural_form"]["active_escape_audit"]["available"] =
      escape_natural_form_audit.available;
    report["rmp_natural_form"]["active_escape_audit"]["error"] =
      escape_natural_form_audit.error;
    report["rmp_natural_form"]["active_escape_audit"]["inactive_metric_norm"] =
      escape_natural_form_audit.inactive_metric_norm;
    report["rmp_natural_form"]["active_escape_audit"]["inactive_force_norm"] =
      escape_natural_form_audit.inactive_force_norm;
    report["rmp_natural_form"]["active_escape_audit"]["inactive_exact_zero"] =
      escape_natural_form_audit.inactive_zero;
    report["rmp_natural_form"]["active_escape_audit"]["sensor_count"] =
      escape_natural_form_audit.sensor_count;
    report["rmp_natural_form"]["active_escape_audit"]["feasible_sensor_count"] =
      escape_natural_form_audit.feasible_sensor_count;
    report["rmp_natural_form"]["active_escape_audit"]["activated_sensor_count"] =
      escape_natural_form_audit.activated_sensor_count;
    report["rmp_natural_form"]["active_escape_audit"]["active_sample_count"] =
      escape_natural_form_audit.active_sample_count;
    report["rmp_natural_form"]["active_escape_audit"]["failed_active_sample_count"] =
      escape_natural_form_audit.failed_active_sample_count;
    report["rmp_natural_form"]["active_escape_audit"]["all_feasible_sensors_activated"] =
      escape_natural_form_audit.all_feasible_sensors_activated;
    report["rmp_natural_form"]["active_escape_audit"]["all_active_samples_valid"] =
      escape_natural_form_audit.all_active_samples_valid;
    report["rmp_natural_form"]["active_escape_audit"]["minimum_contribution_eigenvalue"] =
      escape_natural_form_audit.minimum_contribution_eigenvalue;
    report["rmp_natural_form"]["active_escape_audit"]["maximum_contribution_rank"] =
      escape_natural_form_audit.maximum_contribution_rank;
    report["rmp_natural_form"]["active_escape_audit"]["maximum_contribution_symmetry_residual"] =
      escape_natural_form_audit.maximum_contribution_symmetry_residual;
    report["rmp_natural_form"]["active_escape_audit"]["maximum_tangent_normal_dot_abs"] =
      escape_natural_form_audit.maximum_tangent_normal_dot_abs;
    report["rmp_natural_form"]["active_escape_audit"]["minimum_root_metric_eigenvalue"] =
      escape_natural_form_audit.minimum_root_metric_eigenvalue;
    report["rmp_natural_form"]["active_escape_audit"]["maximum_regularized_resolve_residual"] =
      escape_natural_form_audit.maximum_regularized_resolve_residual;
    report["rmp_natural_form"]["active_escape_audit"]["sensor_probes"] =
      escape_natural_form_audit.sensor_probes;
    const bool active_escape_natural_form_valid =
      escape_natural_form_audit.available &&
      escape_natural_form_audit.inactive_zero &&
      escape_natural_form_audit.all_feasible_sensors_activated &&
      escape_natural_form_audit.all_active_samples_valid;
    report["rmp_natural_form"]["active_escape_natural_form_valid"] =
      active_escape_natural_form_valid;
    report["rmp_natural_form"]["structured_gds_closure_claimed"] = false;
    report["rmp_natural_form"]["interpretation"] =
      "The active Escape contribution was checked as a symmetric PSD rank-one "
      "pullback and the complete root remained SPD and consistently resolved. "
      "This is natural-form compatibility, not a structured-GDS closure claim "
      "for the stateful hybrid Escape policy.";

    const std::string escape_acceleration_model =
      get_parameter("tangent_escape_rmp_acceleration_model").as_string();
    // risk_damped and tangent_plane both drive entry from raw blockage rather
    // than the filtered min_activation of the canonical model.
    const double escape_entry_threshold =
      (escape_acceleration_model == "risk_damped" ||
      escape_acceleration_model == "tangent_plane") ?
      1e-12 :
      std::max(get_parameter("tangent_escape_rmp_min_activation").as_double(), 0.0);
    const double escape_safe_distance =
      get_parameter("tangent_escape_rmp_safe_distance").as_double();
    const double escape_influence_distance =
      get_parameter("tangent_escape_rmp_influence_distance").as_double();
    const bool clear_guard_separation =
      std::isfinite(escape_entry_threshold) &&
      escape_entry_threshold > 0.0 &&
      std::isfinite(escape_safe_distance) &&
      std::isfinite(escape_influence_distance) &&
      escape_influence_distance > escape_safe_distance;
    report["normal_escape_guard_separation"]["normal_set"] =
      "Omega_N_sample = {x: ||x-x*||_2 <= sampled decrease radius}";
    report["normal_escape_guard_separation"]["clear_environment_set"] =
      "C_clear = {all sensor-obstacle clearances d_i >= d_off}";
    report["normal_escape_guard_separation"]["combined_set"] =
      "Omega_N_clear = Omega_N_sample x C_clear";
    report["normal_escape_guard_separation"]["distance_activation"] =
      "alpha_d = quintic01((d_off-d)/(d_off-d_safe))";
    report["normal_escape_guard_separation"]["safe_distance_m"] =
      escape_safe_distance;
    report["normal_escape_guard_separation"]["distance_off_m"] =
      escape_influence_distance;
    report["normal_escape_guard_separation"]["escape_entry_threshold"] =
      escape_entry_threshold;
    report["normal_escape_guard_separation"]["entry_guard_upper_bound_in_clear_set"] =
      0.0;
    report["normal_escape_guard_separation"]["strict_guard_exclusion"] =
      clear_guard_separation;
    report["normal_escape_guard_separation"]["scope"] =
      "Analytical guard exclusion is conditional on every clearance remaining "
      "at or above d_off; it is not a collision-free global ROA certificate.";

    report["lyapunov"]["matrix_p"] =
      yaml_matrix(certificate.lyapunov_matrix);
    report["lyapunov"]["equation"] = "A^T P A - P = -I";
    report["lyapunov"]["solved"] = certificate.lyapunov_solved;
    report["lyapunov"]["p_min_eigenvalue"] =
      certificate.lyapunov_min_eigenvalue;
    report["lyapunov"]["p_max_eigenvalue"] =
      certificate.lyapunov_max_eigenvalue;
    report["lyapunov"]["decrease_max_eigenvalue"] =
      certificate.decrease_max_eigenvalue;
    report["lyapunov"]["residual_norm"] = certificate.residual_norm;
    report["lyapunov"]["positive_definite"] =
      certificate.lyapunov_positive_definite;
    report["lyapunov"]["strict_linear_decrease"] =
      certificate.lyapunov_decrease_strict;

    report["nonlinear_sampling"]["radius_summaries"] = radius_summaries;
    report["nonlinear_sampling"]["largest_strict_decrease_radius"] =
      largest_sampled_decrease_radius;
    YAML::Node sample_values(YAML::NodeType::Sequence);
    for (const auto & sample : samples) {
      YAML::Node value;
      value["radius"] = sample.radius;
      value["direction_index"] = sample.direction_index;
      value["state_norm"] = sample.state_norm;
      value["v"] = sample.value;
      value["v_next"] = sample.next_value;
      value["delta_v"] = sample.delta_value;
      value["relative_delta_v"] = sample.relative_delta;
      value["nonsmooth_branch_active"] =
        sample.nonsmooth_branch_active;
      sample_values.push_back(value);
    }
    report["nonlinear_sampling"]["samples"] = sample_values;

    report["verdict"]["normal_local_exponential_stability_certified"] =
      formal_normal_certificate;
    report["verdict"]["software_command_box_forward_invariant_certified"] =
      command_config.predictive_joint_limit_guard;
    report["verdict"]["software_command_box"]["q_lower_rad"] =
      yaml_joint_vector(
      Eigen::Map<const JointVector>(command_config.joint_lower_limits.data()));
    report["verdict"]["software_command_box"]["q_upper_rad"] =
      yaml_joint_vector(
      Eigen::Map<const JointVector>(command_config.joint_upper_limits.data()));
    report["verdict"]["software_command_box"]["max_abs_qd_rad_s"] =
      command_config.command_guard_max_velocity_rad_s;
    report["verdict"]["software_command_box"]["max_abs_qdd_rad_s2"] =
      command_config.max_joint_accel;
    report["verdict"]["sampled_nonlinear_decrease_observed"] =
      largest_sampled_decrease_radius > 0.0;
    report["verdict"]["escape_leaf_natural_form_compatible"] =
      active_escape_natural_form_valid;
    report["verdict"]["normal_escape_guard_separated_in_clear_set"] =
      clear_guard_separation;
    report["verdict"]["global_stability_certified"] = false;
    report["verdict"]["hybrid_escape_stability_certified"] = false;
    report["verdict"]["collision_safe_set_invariance_certified"] = false;

    try {
      const std::filesystem::path output_directory = expand_home_path(
        get_parameter("production_stability_output_directory").as_string());
      std::filesystem::create_directories(output_directory);
      const std::filesystem::path output_path =
        output_directory /
        ("production_stability_" + file_timestamp() + ".yaml");
      std::ofstream output(output_path);
      if (!output) {
        throw std::runtime_error(
                "Cannot open production stability report: " +
                output_path.string());
      }
      output << report;
      output.close();
      production_stability_report_path_ = output_path.string();
      RCLCPP_INFO(
        get_logger(),
        "Production stability report saved: %s",
        production_stability_report_path_.c_str());
      RCLCPP_INFO(
        get_logger(),
        "NORMAL local certificate: %s (rho=%.9g, min_eig(P)=%.9g, "
        "max_eig(dV)=%.9g, sampled_radius=%.9g)",
        formal_normal_certificate ? "PASS" : "FAIL",
        certificate.spectral_radius,
        certificate.lyapunov_min_eigenvalue,
        certificate.decrease_max_eigenvalue,
        largest_sampled_decrease_radius);
    } catch (const std::exception & error) {
      RCLCPP_ERROR(
        get_logger(),
        "Failed to write production stability analysis: %s",
        error.what());
    }
    if (get_parameter("shutdown_after_production_stability_analysis").as_bool()) {
      running_.store(false);
      rclcpp::shutdown();
    }
  }

  JointVector compute_acceleration(
    const RobotState & state,
    const Eigen::Vector3d & goal,
    const Eigen::Quaterniond & goal_orientation,
    const std::vector<ObstacleSphere> & obstacles,
    std::vector<double> * tangent_escape_rmp_data = nullptr,
    std::vector<double> * tangent_escape_dual_solve_data = nullptr,
    std::vector<double> * leaf_ablation_data = nullptr,
    std::vector<double> * stability_certificate_data = nullptr) const
  {
    std::unordered_map<std::string, Eigen::Vector3d> vector_targets;
    vector_targets.emplace("goal", goal);
    vector_targets.emplace("body_goal", body_goal_);
    const Eigen::Matrix3d goal_rotation = goal_orientation.normalized().toRotationMatrix();
    vector_targets.emplace("orientation_goal_x", goal_rotation.col(0));
    vector_targets.emplace("orientation_goal_y", goal_rotation.col(1));
    vector_targets.emplace("orientation_goal_z", goal_rotation.col(2));
    std::unordered_map<std::string, ExternalRmpFeature> external_rmps;
    {
      std::scoped_lock lock(external_rmp_mutex_);
      for (const auto & entry : external_rmp_buffers_) {
        const auto & buffer = entry.second;
        if (!buffer.has_metric || !buffer.has_acceleration) {
          continue;
        }
        external_rmps.emplace(
          entry.first,
          ExternalRmpFeature{buffer.metric_sqrt, buffer.acceleration});
      }
    }
    const auto solution = solver_->solve(
      state.q,
      state.qd,
      vector_targets,
      obstacles,
      external_rmps);
    if (tangent_escape_rmp_data != nullptr) {
      *tangent_escape_rmp_data = solution.tangent_escape_rmp_data;
    }
    if (tangent_escape_dual_solve_data != nullptr) {
      *tangent_escape_dual_solve_data =
        solution.tangent_escape_dual_solve_data;
    }
    if (leaf_ablation_data != nullptr) {
      *leaf_ablation_data = solution.leaf_ablation_data;
    }
    if (stability_certificate_data != nullptr) {
      *stability_certificate_data = solution.stability_certificate_data;
    }
    return limit_joint_acceleration(
      solution.qdd,
      max_joint_accel_,
      preserve_joint_accel_direction_);
  }

  RobotState integrate_command(
    const RobotState & state,
    const JointVector & qdd,
    double dt) const
  {
    const ProductionJointState production_state{state.q, state.qd};
    const auto integrated = integrate_production_command(
      production_state,
      qdd,
      production_command_map_config(dt));
    return RobotState{integrated.q, integrated.qd};
  }

  void send_velocity_hold_command(const RobotState & measured_state)
  {
    if (command_mode_ != CommandMode::kVelocity || !backend_) {
      return;
    }

    RobotState hold_state = measured_state;
    hold_state.qd.setZero();
    previous_velocity_command_.setZero();
    previous_velocity_command_initialized_ = true;
    if (backend_command_application_enabled_) {
      backend_->apply_command(hold_state, RB10Model::joint_names);
    }
    publish_position_command(hold_state);
    publish_direct_command(hold_state);
    if (backend_command_application_enabled_) {
      virtual_velocity_state_ = hold_state;
      virtual_velocity_state_initialized_ = true;
    } else {
      virtual_velocity_state_initialized_ = false;
    }
  }

  void reset_controller_velocity_estimator()
  {
    controller_velocity_estimator_initialized_ = false;
    controller_velocity_previous_q_.setZero();
    controller_velocity_estimate_.setZero();
    controller_velocity_previous_time_ = std::chrono::steady_clock::time_point{};
  }

  JointVector estimate_controller_velocity(
    const JointVector & q,
    const std::chrono::steady_clock::time_point & sample_time)
  {
    if (!controller_velocity_estimator_initialized_) {
      controller_velocity_previous_q_ = q;
      controller_velocity_estimate_.setZero();
      controller_velocity_previous_time_ = sample_time;
      controller_velocity_estimator_initialized_ = true;
      return controller_velocity_estimate_;
    }

    const double dt = std::chrono::duration<double>(
      sample_time - controller_velocity_previous_time_).count();
    controller_velocity_previous_time_ = sample_time;
    if (dt <= std::numeric_limits<double>::epsilon() || dt > 0.25) {
      controller_velocity_previous_q_ = q;
      controller_velocity_estimate_.setZero();
      return controller_velocity_estimate_;
    }
    const JointVector raw_qd = (q - controller_velocity_previous_q_) / dt;
    controller_velocity_previous_q_ = q;
    controller_velocity_estimate_ =
      controller_velocity_filter_alpha_ * raw_qd +
      (1.0 - controller_velocity_filter_alpha_) * controller_velocity_estimate_;
    return controller_velocity_estimate_;
  }

  void publish_rmp_ee_pose(const KinematicsContext & context)
  {
    if (!rmp_eef_pose_pub_) {
      return;
    }

    geometry_msgs::msg::Pose pose;
    pose.position.x = context.tcp_position.x();
    pose.position.y = context.tcp_position.y();
    pose.position.z = context.tcp_position.z();

    Eigen::Quaterniond tcp_orientation(context.link_rotations[RB10Model::TCP_RMP]);
    tcp_orientation.normalize();
    pose.orientation.x = tcp_orientation.x();
    pose.orientation.y = tcp_orientation.y();
    pose.orientation.z = tcp_orientation.z();
    pose.orientation.w = tcp_orientation.w();

    if (rt_rmp_eef_pose_pub_) {
      rt_rmp_eef_pose_pub_->tryPublish(pose);
      return;
    }
    rmp_eef_pose_pub_->publish(pose);
  }

  void publish_target_metric(
    const KinematicsContext & context,
    const Eigen::Vector3d & goal)
  {
    if (!target_metric_pub_) {
      return;
    }

    const Eigen::Vector3d delta = goal - context.tcp_position;
    const double delta_norm = delta.norm();
    const double soft_delta_norm =
      std::max(delta_norm, target_metric_params_.accel_norm_eps / 10.0);
    const Eigen::Vector3d delta_hat = delta / soft_delta_norm;

    const Eigen::Matrix3d eye = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d shape = delta_hat * delta_hat.transpose();
    const double scaled_dist = delta_norm / target_metric_params_.metric_alpha_length_scale;
    const double alpha =
      (1.0 - target_metric_params_.min_metric_alpha) *
      std::exp(-0.5 * scaled_dist * scaled_dist) +
      target_metric_params_.min_metric_alpha;
    Eigen::Matrix3d leaf_metric =
      alpha * target_metric_params_.max_metric_scalar * eye +
      (1.0 - alpha) * target_metric_params_.min_metric_scalar * shape;

    const double boost_scaled_dist =
      delta_norm / target_metric_params_.proximity_metric_boost_length_scale;
    const double boost_alpha = std::exp(-0.5 * boost_scaled_dist * boost_scaled_dist);
    const double metric_boost_scalar =
      boost_alpha * target_metric_params_.proximity_metric_boost_scalar +
      (1.0 - boost_alpha);
    leaf_metric *= metric_boost_scalar;

    std_msgs::msg::Float64MultiArray metric_msg;
    metric_msg.data.reserve(9);
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        metric_msg.data.push_back(leaf_metric(row, col));
      }
    }

    if (rt_target_metric_pub_) {
      rt_target_metric_pub_->tryPublish(metric_msg);
      return;
    }
    target_metric_pub_->publish(metric_msg);
  }

  void publish_goal_tf(const GoalTarget & goal_target)
  {
    if (!goal_tf_broadcaster_ || goal_tf_parent_frame_.empty() || goal_tf_child_frame_.empty()) {
      return;
    }

    Eigen::Quaterniond goal_orientation = goal_target.orientation;
    if (!goal_orientation.coeffs().allFinite() || goal_orientation.norm() < 1e-9) {
      goal_orientation = Eigen::Quaterniond::Identity();
    } else {
      goal_orientation.normalize();
    }

    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = now();
    transform.header.frame_id = goal_tf_parent_frame_;
    transform.child_frame_id = goal_tf_child_frame_;
    transform.transform.translation.x = goal_target.position.x();
    transform.transform.translation.y = goal_target.position.y();
    transform.transform.translation.z = goal_target.position.z();
    transform.transform.rotation.x = goal_orientation.x();
    transform.transform.rotation.y = goal_orientation.y();
    transform.transform.rotation.z = goal_orientation.z();
    transform.transform.rotation.w = goal_orientation.w();
    goal_tf_broadcaster_->sendTransform(transform);
  }

  bool violates_min_z_safety(const KinematicsContext & context) const
  {
    if (rmp_flag_gate_enabled_ && !rmp_active_.load()) {
      return false;
    }

    if (!safety_stop_on_min_z_) {
      return false;
    }

    const auto floor_metrics = compute_floor_safety_metrics(context);
    return
      floor_metrics.min_link_z < workspace_floor_z_ ||
      floor_metrics.min_joint_z < workspace_floor_z_ ||
      floor_metrics.min_control_point_z < workspace_floor_z_ ||
      floor_metrics.min_body_obstacle_z < workspace_floor_z_ ||
      context.link_positions[RB10Model::LINK6].z() < min_link6_z_ ||
      context.link_positions[RB10Model::TCP_GRIPPER].z() < min_tcp_z_;
  }

  void reset_velocity_command_guard()
  {
    previous_velocity_command_.setZero();
    previous_velocity_command_initialized_ = false;
  }

  CommandGuardDiagnostics apply_command_guard(
    const RobotState & measured_state,
    RobotState & command_state,
    double dt)
  {
    ProductionJointState production_measured{measured_state.q, measured_state.qd};
    ProductionJointState production_command{command_state.q, command_state.qd};
    ProductionCommandMemory memory;
    memory.previous_velocity_command = previous_velocity_command_;
    memory.previous_velocity_command_initialized =
      previous_velocity_command_initialized_;
    ProductionCommandDiagnostics production_diagnostics;
    apply_production_command_guard(
      production_measured,
      production_command,
      production_command_map_config(dt),
      memory,
      &production_diagnostics);
    command_state.q = production_command.q;
    command_state.qd = production_command.qd;
    previous_velocity_command_ = memory.previous_velocity_command;
    previous_velocity_command_initialized_ =
      memory.previous_velocity_command_initialized;

    CommandGuardDiagnostics diagnostics;
    diagnostics.clamped = production_diagnostics.command_guard_clamped;
    diagnostics.velocity_slew_clamped =
      production_diagnostics.velocity_slew_clamped;
    diagnostics.pre_guard_command = production_diagnostics.pre_guard_command;
    diagnostics.command_step = production_diagnostics.command_step;
    if (diagnostics.velocity_slew_clamped) {
      ++velocity_slew_clamp_count_;
    }
    return diagnostics;
  }

  void record_and_publish_control_timing(
    const std::chrono::steady_clock::time_point & cycle_start,
    const std::chrono::steady_clock::time_point & scheduled_start,
    const std::chrono::steady_clock::time_point & deadline,
    const std::chrono::steady_clock::time_point & work_end,
    double pre_solve_duration_sec,
    double solve_duration_sec,
    double post_solve_duration_sec,
    const RobotState & measured_state,
    const JointVector & qdd,
    const RobotState & command_state,
    const CommandGuardDiagnostics & command_guard_diagnostics)
  {
    auto & timing = control_timing_accumulator_;
    ++timing.active_cycle_count;

    if (timing.previous_active_cycle_valid) {
      timing.cycle_interval_latest_sec =
        std::chrono::duration<double>(
        cycle_start - timing.previous_active_cycle_start).count();
      timing.cycle_interval_total_sec += timing.cycle_interval_latest_sec;
      timing.cycle_interval_max_sec = std::max(
        timing.cycle_interval_max_sec, timing.cycle_interval_latest_sec);
      timing.cycle_interval_window_max_sec = std::max(
        timing.cycle_interval_window_max_sec, timing.cycle_interval_latest_sec);
    }
    timing.previous_active_cycle_start = cycle_start;
    timing.previous_active_cycle_valid = true;

    timing.solve_latest_sec = std::max(solve_duration_sec, 0.0);
    timing.solve_total_sec += timing.solve_latest_sec;
    timing.solve_max_sec = std::max(timing.solve_max_sec, timing.solve_latest_sec);
    timing.solve_window_max_sec = std::max(
      timing.solve_window_max_sec, timing.solve_latest_sec);
    const double pre_solve_sec = std::max(pre_solve_duration_sec, 0.0);
    const double post_solve_sec = std::max(post_solve_duration_sec, 0.0);
    timing.pre_solve_total_sec += pre_solve_sec;
    timing.pre_solve_window_max_sec = std::max(
      timing.pre_solve_window_max_sec, pre_solve_sec);
    timing.post_solve_total_sec += post_solve_sec;
    timing.post_solve_window_max_sec = std::max(
      timing.post_solve_window_max_sec, post_solve_sec);

    timing.work_latest_sec = std::max(
      std::chrono::duration<double>(work_end - cycle_start).count(), 0.0);
    timing.work_total_sec += timing.work_latest_sec;
    timing.work_max_sec = std::max(timing.work_max_sec, timing.work_latest_sec);
    timing.work_window_max_sec = std::max(
      timing.work_window_max_sec, timing.work_latest_sec);

    timing.start_lateness_latest_sec = std::max(
      std::chrono::duration<double>(cycle_start - scheduled_start).count(), 0.0);
    timing.start_lateness_total_sec += timing.start_lateness_latest_sec;
    timing.start_lateness_max_sec = std::max(
      timing.start_lateness_max_sec, timing.start_lateness_latest_sec);
    timing.start_lateness_window_max_sec = std::max(
      timing.start_lateness_window_max_sec, timing.start_lateness_latest_sec);
    if (work_end > deadline) {
      ++timing.deadline_miss_count;
    }

    const double measured_qd_norm = measured_state.qd.norm();
    const double measured_qd_max_abs = measured_state.qd.cwiseAbs().maxCoeff();
    const double qdd_norm = qdd.norm();
    const double qdd_max_abs = qdd.cwiseAbs().maxCoeff();
    const JointVector & command_vector =
      command_mode_ == CommandMode::kVelocity ? command_state.qd : command_state.q;
    const double command_norm = command_vector.norm();
    const double command_max_abs = command_vector.cwiseAbs().maxCoeff();
    timing.measured_qd_norm_max = std::max(timing.measured_qd_norm_max, measured_qd_norm);
    timing.measured_qd_max_abs_max = std::max(
      timing.measured_qd_max_abs_max, measured_qd_max_abs);
    timing.qdd_norm_max = std::max(timing.qdd_norm_max, qdd_norm);
    timing.qdd_max_abs_max = std::max(timing.qdd_max_abs_max, qdd_max_abs);
    timing.command_norm_max = std::max(timing.command_norm_max, command_norm);
    timing.command_max_abs_max = std::max(
      timing.command_max_abs_max, command_max_abs);

    const double publish_period_sec = 1.0 / control_timing_publish_rate_hz_;
    if (
      timing.last_publish_time.time_since_epoch().count() != 0 &&
      std::chrono::duration<double>(work_end - timing.last_publish_time).count() <
      publish_period_sec)
    {
      return;
    }
    timing.last_publish_time = work_end;

    const BackendTimingSnapshot backend_timing = backend_->take_timing_snapshot();
    std_msgs::msg::Float64MultiArray message;
    // Schema 1 contains cumulative counters/totals so a recorder can calculate
    // statistics for only its own observation window without high-rate logging.
    // The vectors appended after the original 45 fields preserve schema-1
    // compatibility while allowing event-level q/qd/qdd/command correlation.
    message.data.reserve(116U);
    message.data = {
      1.0,
      control_rate_hz_,
      1.0 / control_rate_hz_,
      static_cast<double>(timing.active_cycle_count),
      timing.cycle_interval_latest_sec,
      timing.cycle_interval_total_sec,
      timing.cycle_interval_max_sec,
      timing.solve_latest_sec,
      timing.solve_total_sec,
      timing.solve_max_sec,
      timing.work_latest_sec,
      timing.work_total_sec,
      timing.work_max_sec,
      timing.start_lateness_latest_sec,
      timing.start_lateness_total_sec,
      timing.start_lateness_max_sec,
      static_cast<double>(timing.deadline_miss_count),
      static_cast<double>(backend_timing.state_interval_count),
      backend_timing.state_interval_latest_sec,
      backend_timing.state_interval_total_sec,
      backend_timing.state_interval_max_sec,
      static_cast<double>(backend_timing.state_gap_over_5ms_count),
      static_cast<double>(backend_timing.state_gap_over_20ms_count),
      static_cast<double>(backend_timing.state_gap_over_100ms_count),
      measured_qd_norm,
      timing.measured_qd_norm_max,
      measured_qd_max_abs,
      timing.measured_qd_max_abs_max,
      qdd_norm,
      timing.qdd_norm_max,
      qdd_max_abs,
      timing.qdd_max_abs_max,
      command_norm,
      timing.command_norm_max,
      command_max_abs,
      timing.command_max_abs_max,
      timing.cycle_interval_window_max_sec,
      timing.solve_window_max_sec,
      timing.work_window_max_sec,
      timing.start_lateness_window_max_sec,
      backend_timing.state_interval_window_max_sec,
      timing.pre_solve_total_sec,
      timing.pre_solve_window_max_sec,
      timing.post_solve_total_sec,
      timing.post_solve_window_max_sec,
    };
    for (int joint = 0; joint < 6; ++joint) {
      message.data.push_back(measured_state.q[joint]);
    }
    for (int joint = 0; joint < 6; ++joint) {
      message.data.push_back(measured_state.qd[joint]);
    }
    for (int joint = 0; joint < 6; ++joint) {
      message.data.push_back(qdd[joint]);
    }
    for (int joint = 0; joint < 6; ++joint) {
      message.data.push_back(command_vector[joint]);
    }
    message.data.push_back(static_cast<double>(backend_timing.velocity_filter_update_count));
    message.data.push_back(backend_timing.velocity_filter_synced_dt_window_max_sec);
    for (int joint = 0; joint < 6; ++joint) {
      message.data.push_back(backend_timing.raw_position_step_abs_window_max[joint]);
    }
    for (int joint = 0; joint < 6; ++joint) {
      message.data.push_back(backend_timing.raw_velocity_abs_window_max[joint]);
    }
    for (int joint = 0; joint < 6; ++joint) {
      message.data.push_back(
        backend_timing.velocity_filter_residual_abs_window_max[joint]);
    }
    for (int joint = 0; joint < 6; ++joint) {
      message.data.push_back(backend_timing.filtered_velocity_abs_window_max[joint]);
    }
    message.data.push_back(
      static_cast<double>(backend_timing.velocity_filter_input_reject_count));
    message.data.push_back(
      static_cast<double>(backend_timing.velocity_filter_output_reject_count));
    for (int joint = 0; joint < 6; ++joint) {
      message.data.push_back(command_guard_diagnostics.pre_guard_command[joint]);
    }
    for (int joint = 0; joint < 6; ++joint) {
      message.data.push_back(command_guard_diagnostics.command_step[joint]);
    }
    message.data.push_back(static_cast<double>(velocity_slew_clamp_count_));
    message.data.push_back(
      command_guard_diagnostics.velocity_slew_clamped ? 1.0 : 0.0);
    message.data.push_back(
      static_cast<double>(backend_timing.controller_packet_time_used_count));
    message.data.push_back(
      static_cast<double>(backend_timing.controller_packet_time_fallback_count));
    message.data.push_back(
      static_cast<double>(backend_timing.controller_packet_time_duplicate_count));
    message.data.push_back(
      static_cast<double>(backend_timing.controller_packet_time_reset_count));
    message.data.push_back(backend_timing.controller_packet_time_dt_window_max_sec);
    timing.cycle_interval_window_max_sec = 0.0;
    timing.solve_window_max_sec = 0.0;
    timing.work_window_max_sec = 0.0;
    timing.start_lateness_window_max_sec = 0.0;
    timing.pre_solve_window_max_sec = 0.0;
    timing.post_solve_window_max_sec = 0.0;
    if (rt_control_timing_pub_) {
      rt_control_timing_pub_->tryPublish(message);
      return;
    }
    if (control_timing_pub_) {
      control_timing_pub_->publish(message);
    }
  }

  void control_loop()
  {
    enable_realtime(
      get_logger(),
      enable_realtime_,
      realtime_priority_,
      lock_memory_);

    const auto period = std::chrono::duration<double>(1.0 / control_rate_hz_);
    const auto period_duration =
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
    auto next_tick = std::chrono::steady_clock::now();

    while (rclcpp::ok() && running_.load()) {
      const auto scheduled_start = next_tick;
      next_tick += period_duration;
      const auto cycle_start = std::chrono::steady_clock::now();

      if (!backend_->ready()) {
        virtual_velocity_state_initialized_ = false;
        last_safe_command_state_initialized_ = false;
        reset_velocity_command_guard();
        clear_tcp_accel_visualization_sample();
        reset_controller_velocity_estimator();
        RCLCPP_INFO_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "Waiting for initial robot state on %s before enabling RMP commands",
          joint_state_topic_.c_str());
        std::this_thread::sleep_until(next_tick);
        continue;
      }

      RobotState measured_state = backend_->read_state();
      if (estimate_velocity_in_controller_) {
        // Estimate qd on the controller's configured control timeline so the
        // solver sees a velocity signal aligned with the solve loop instead
        // of an asynchronously sampled derivative from the bridge.
        measured_state.qd = estimate_controller_velocity(measured_state.q, cycle_start);
      }
      const auto measured_context = RB10Model::forward_context(
        measured_state.q, ControlPointJacobianMode::kNone);
      if (rmp_flag_gate_enabled_ && !rmp_active_.load()) {
        virtual_velocity_state_initialized_ = false;
        last_safe_command_state_initialized_ = false;
        reset_velocity_command_guard();
        clear_tcp_accel_visualization_sample();
        state_ = measured_state;
        if (estimate_velocity_in_controller_) {
          state_.qd = measured_state.qd;
        }
        if (publish_joint_states_enabled_) {
          publish_joint_states(state_);
        }
        publish_rmp_ee_pose(measured_context);
        GoalTarget goal_target;
        goal_target_box_.get(goal_target);
        publish_target_metric(measured_context, goal_target.position);
        send_velocity_hold_command(measured_state);
        RCLCPP_INFO_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "RMP standby: waiting for /RMP_flag == %d before running the solve loop",
          rmp_active_flag_value_);
        std::this_thread::sleep_until(next_tick);
        continue;
      }
      if (rmp_flag_gate_enabled_ && !external_goal_received_.load()) {
        virtual_velocity_state_initialized_ = false;
        last_safe_command_state_initialized_ = false;
        reset_velocity_command_guard();
        clear_tcp_accel_visualization_sample();
        state_ = measured_state;
        if (estimate_velocity_in_controller_) {
          state_.qd = measured_state.qd;
        }
        if (publish_joint_states_enabled_) {
          publish_joint_states(state_);
        }
        publish_rmp_ee_pose(measured_context);
        GoalTarget goal_target;
        goal_target_box_.get(goal_target);
        publish_target_metric(measured_context, goal_target.position);
        send_velocity_hold_command(measured_state);
        RCLCPP_INFO_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "RMP active but waiting for /goal_pose before running the solve loop");
        std::this_thread::sleep_until(next_tick);
        continue;
      }
      if (
        initialize_goal_from_first_state_ &&
        !startup_goal_synced_.load() &&
        !external_goal_received_.load())
      {
        const Eigen::Vector3d startup_goal =
          measured_context.link_positions[RB10Model::TCP_RMP];
        Eigen::Quaterniond startup_orientation(
          measured_context.link_rotations[RB10Model::TCP_RMP]);
        startup_orientation.normalize();
        goal_target_box_.set(GoalTarget{startup_goal, startup_orientation});
        startup_goal_synced_.store(true);
        RCLCPP_INFO(
          get_logger(),
          "Initialized startup goal from the current tcp_rmp pose to avoid a jump at launch");
      }
      const bool measured_state_violates_min_z = violates_min_z_safety(measured_context);
      if (!last_safe_command_state_initialized_ && !measured_state_violates_min_z) {
        last_safe_command_state_ = measured_state;
        last_safe_command_state_.qd.setZero();
        last_safe_command_state_initialized_ = true;
      }

      RobotState control_state = measured_state;
      if (virtual_velocity_state_initialized_) {
        // Blending measured q with the previous commanded q lets the outer
        // loop stay responsive when the robot's internal servo is still
        // catching up, while preserving an easy path back to pure measured
        // feedback for safety-oriented testing.
        control_state.q =
          measured_position_feedback_blend_ * measured_state.q +
          (1.0 - measured_position_feedback_blend_) * virtual_velocity_state_.q;

        // Blend measured and commanded joint velocity. Pure commanded velocity
        // is responsive but can hunt on real hardware, while pure measured
        // velocity tends to become too sluggish because it reflects the robot's
        // lagging servo response and filtered state estimate.
        control_state.qd =
          measured_velocity_feedback_blend_ * measured_state.qd +
          (1.0 - measured_velocity_feedback_blend_) * virtual_velocity_state_.qd;
      }
      if (!use_velocity_feedback_in_solver_) {
        // Allow testing the target attractor with position-only feedback by
        // removing the joint velocity term from the solver input.
        control_state.qd.setZero();
      }
      Eigen::Vector3d goal;
      Eigen::Quaterniond goal_orientation;
      std::vector<ObstacleSphere> obstacles;
      std::vector<std::optional<ObstacleSphere>> proximity_sensor_obstacles;
      GoalTarget goal_target;
      goal_target_box_.get(goal_target);
      goal = goal_target.position;
      goal_orientation = goal_target.orientation;
      obstacles_box_.get(obstacles);
      proximity_sensor_obstacles_box_.get(proximity_sensor_obstacles);
      try {
        run_production_stability_analysis_once(control_state, goal_target);
      } catch (const std::exception & error) {
        production_stability_analysis_completed_ = true;
        RCLCPP_ERROR(
          get_logger(),
          "Production stability analysis failed: %s",
          error.what());
      }

      JointVector qdd = JointVector::Zero();
      std::vector<double> tangent_escape_rmp_data;
      std::vector<double> tangent_escape_dual_solve_data;
      std::vector<double> leaf_ablation_data;
      std::vector<double> stability_certificate_data;
      last_min_z_safety_triggered_.store(false);
      bool solve_ok = false;
      const auto solve_start = std::chrono::steady_clock::now();
      try {
        qdd = compute_acceleration(
          control_state,
          goal,
          goal_orientation,
          obstacles,
          (publish_tangent_escape_rmp_data_enabled_ ||
          publish_tangent_escape_geometry_debug_enabled_) ?
          &tangent_escape_rmp_data : nullptr,
          publish_tangent_escape_dual_solve_data_enabled_ ?
          &tangent_escape_dual_solve_data : nullptr,
          publish_leaf_ablation_data_enabled_ ? &leaf_ablation_data : nullptr,
          publish_stability_certificate_data_enabled_ ?
          &stability_certificate_data : nullptr);
        solve_ok = true;
      } catch (const std::exception & error) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "RMP solve failed, holding command at zero acceleration: %s",
          error.what());
      }
      const auto solve_end = std::chrono::steady_clock::now();
      const double solve_duration_sec = std::chrono::duration<double>(
        solve_end - solve_start).count();
      if (publish_tangent_escape_geometry_debug_enabled_) {
        tangent_escape_visualization_box_.set(
          solve_ok ? tangent_escape_rmp_data : std::vector<double>{});
        if (publish_tangent_escape_dual_solve_data_enabled_) {
          tangent_escape_dual_visualization_box_.set(
            solve_ok ? tangent_escape_dual_solve_data : std::vector<double>{});
        }
      }
      if (solve_ok) {
        if (publish_tangent_escape_rmp_data_enabled_) {
          publish_tangent_escape_rmp_data(tangent_escape_rmp_data);
        }
        if (publish_tangent_escape_dual_solve_data_enabled_) {
          publish_tangent_escape_dual_solve_data(tangent_escape_dual_solve_data);
        }
        if (publish_leaf_ablation_data_enabled_) {
          publish_leaf_ablation_data(leaf_ablation_data);
        }
        if (publish_stability_certificate_data_enabled_) {
          publish_stability_certificate_data(stability_certificate_data);
        }
      }

      std::optional<KinematicsContext> control_context_storage;
      const auto control_context_jacobian_mode =
        publish_tangent_escape_geometry_debug_enabled_ ?
        ControlPointJacobianMode::kAnalytic :
        ControlPointJacobianMode::kNone;
      auto control_context = [&]() -> const KinematicsContext & {
          if (
            control_context_jacobian_mode == ControlPointJacobianMode::kNone &&
            (control_state.q - measured_state.q).cwiseAbs().maxCoeff() <= 1e-12)
          {
            return measured_context;
          }
          if (!control_context_storage.has_value()) {
            control_context_storage.emplace(RB10Model::forward_context(
                control_state.q, control_context_jacobian_mode));
          }
          return control_context_storage.value();
        };

      if (publish_tangent_escape_geometry_debug_enabled_) {
        publish_tangent_escape_geometry_data(
          control_context(),
          proximity_sensor_obstacles);
      }

      std::optional<Eigen::Vector3d> tcp_accel;
      if (tcp_accel_pub_ || rmp_tcp_accel_pub_) {
        tcp_accel = compute_tcp_acceleration(control_context(), qdd);
      }
      const Eigen::Vector3d * tcp_accel_ptr = tcp_accel ? &(*tcp_accel) : nullptr;
      if (tcp_accel_pub_ && tcp_accel_ptr) {
        update_tcp_accel_visualization(*tcp_accel_ptr);
      }
      publish_rmp_accel_debug(qdd, tcp_accel_ptr);
      RobotState predicted_state = integrate_command(control_state, qdd, period.count());
      const auto predicted_context = RB10Model::forward_context(
        predicted_state.q, ControlPointJacobianMode::kNone);
      RobotState command_state = predicted_state;
      const bool predicted_state_violates_min_z = violates_min_z_safety(predicted_context);
      const KinematicsContext * command_context = &predicted_context;
      std::optional<KinematicsContext> safe_command_context;
      const bool previous_min_z_safety_triggered = last_min_z_safety_triggered_.load();
      const bool min_z_safety_triggered =
        measured_state_violates_min_z || predicted_state_violates_min_z;
      if (min_z_safety_triggered) {
        reset_velocity_command_guard();
        if (last_safe_command_state_initialized_) {
          command_state = last_safe_command_state_;
          safe_command_context.emplace(RB10Model::forward_context(
              command_state.q, ControlPointJacobianMode::kNone));
          command_context = &safe_command_context.value();
        } else {
          command_state = measured_state;
          command_context = &measured_context;
        }
        command_state.qd.setZero();
        last_min_z_safety_triggered_.store(true);
        if (!previous_min_z_safety_triggered) {
          RCLCPP_ERROR(
            get_logger(),
            "Min-Z safety triggered. Holding the last safe command to avoid driving below workspace.");
        }
      }

      const CommandGuardDiagnostics command_guard_diagnostics =
        apply_command_guard(measured_state, command_state, period.count());
      if (command_guard_diagnostics.clamped) {
        safe_command_context.emplace(RB10Model::forward_context(
            command_state.q, ControlPointJacobianMode::kNone));
        command_context = &safe_command_context.value();
        RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          1000,
          command_mode_ == CommandMode::kVelocity ?
          "Command guard limited the joint velocity command to avoid triggering robot safety." :
          "Command guard limited the per-cycle joint step/velocity to avoid triggering robot safety.");
      }

      if (!min_z_safety_triggered) {
        if (previous_min_z_safety_triggered) {
          RCLCPP_INFO(get_logger(), "Min-Z safety cleared.");
        }
        last_safe_command_state_ = command_state;
        last_safe_command_state_initialized_ = true;
      }
      publish_rmp_ee_pose(*command_context);
      if (target_metric_pub_) {
        publish_target_metric(control_context(), goal);
      }
      publish_target_q(command_state);
      publish_position_command(command_state);
      if (backend_command_application_enabled_) {
        backend_->apply_command(command_state, RB10Model::joint_names);
      }
      publish_direct_command(command_state);
      if (backend_command_application_enabled_) {
        virtual_velocity_state_ = command_state;
        virtual_velocity_state_initialized_ = true;
      } else {
        virtual_velocity_state_initialized_ = false;
      }
      state_ = backend_->read_state();
      if (estimate_velocity_in_controller_) {
        state_.qd = measured_state.qd;
      }
      if (publish_joint_states_enabled_) {
        publish_joint_states(state_);
      }

      if (publish_control_timing_debug_enabled_) {
        const auto work_end = std::chrono::steady_clock::now();
        record_and_publish_control_timing(
          cycle_start,
          scheduled_start,
          next_tick,
          work_end,
          std::chrono::duration<double>(solve_start - cycle_start).count(),
          solve_duration_sec,
          std::chrono::duration<double>(work_end - solve_end).count(),
          measured_state,
          qdd,
          command_state,
          command_guard_diagnostics);
      }

      std::this_thread::sleep_until(next_tick);
    }
  }

  void publish_joint_states(const RobotState & state)
  {
    if (!joint_state_pub_) {
      return;
    }
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = now();
    msg.name.assign(RB10Model::joint_names.begin(), RB10Model::joint_names.end());
    msg.position.assign(state.q.data(), state.q.data() + state.q.size());
    msg.velocity.assign(state.qd.data(), state.qd.data() + state.qd.size());
    if (rt_joint_state_pub_) {
      rt_joint_state_pub_->tryPublish(msg);
      return;
    }
    joint_state_pub_->publish(msg);
  }

  void publish_direct_command(const RobotState & command_state)
  {
    if (!direct_command_pub_) {
      return;
    }

    std_msgs::msg::Float64MultiArray command;
    const JointVector & command_vector =
      command_mode_ == CommandMode::kVelocity ? command_state.qd : command_state.q;
    command.data.assign(command_vector.data(), command_vector.data() + command_vector.size());
    if (rt_direct_command_pub_) {
      rt_direct_command_pub_->tryPublish(command);
      return;
    }
    direct_command_pub_->publish(command);
  }

  void publish_position_command(const RobotState & command_state)
  {
    if (!position_command_pub_) {
      return;
    }

    std_msgs::msg::Float64MultiArray command;
    command.data.assign(command_state.q.data(), command_state.q.data() + command_state.q.size());
    if (rt_position_command_pub_) {
      rt_position_command_pub_->tryPublish(command);
      return;
    }
    position_command_pub_->publish(command);
  }

  void publish_target_q(const RobotState & command_state)
  {
    if (!target_q_pub_) {
      return;
    }

    std_msgs::msg::Float64MultiArray command;
    command.data.assign(command_state.q.data(), command_state.q.data() + command_state.q.size());
    if (rt_target_q_pub_) {
      rt_target_q_pub_->tryPublish(command);
      return;
    }
    target_q_pub_->publish(command);
  }

  void publish_visualization()
  {
    const bool publish_goal_tf_visualization =
      publish_goal_tf_enabled_ && static_cast<bool>(goal_tf_broadcaster_);
    const bool publish_goal_marker_visualization = static_cast<bool>(goal_marker_pub_);
    const bool publish_context_visualization =
      static_cast<bool>(control_point_pub_) ||
      static_cast<bool>(body_obstacle_pub_) ||
      static_cast<bool>(debug_state_pub_) ||
      static_cast<bool>(eef_pose_pub_) ||
      static_cast<bool>(repulsion_metric_pub_) ||
      static_cast<bool>(tcp_accel_pub_) ||
      static_cast<bool>(tangent_escape_geometry_marker_pub_);

    if (
      !publish_goal_marker_visualization &&
      !publish_context_visualization &&
      !publish_goal_tf_visualization)
    {
      return;
    }

    if (rmp_flag_gate_enabled_ && (!rmp_active_.load() || !external_goal_received_.load())) {
      if (publish_goal_marker_visualization || publish_context_visualization) {
        clear_rmp_visualization();
      }
      return;
    }

    GoalTarget goal_target;
    goal_target_box_.get(goal_target);
    if (publish_goal_tf_visualization) {
      publish_goal_tf(goal_target);
    }

    const Eigen::Vector3d goal = goal_target.position;
    const auto stamp = now();

    if (goal_marker_pub_) {
      visualization_msgs::msg::Marker goal_marker;
      goal_marker.header.frame_id = "base_link";
      goal_marker.header.stamp = stamp;
      goal_marker.ns = "goal";
      goal_marker.id = 0;
      goal_marker.type = visualization_msgs::msg::Marker::SPHERE;
      goal_marker.action = visualization_msgs::msg::Marker::ADD;
      goal_marker.pose.position.x = goal.x();
      goal_marker.pose.position.y = goal.y();
      goal_marker.pose.position.z = goal.z();
      goal_marker.pose.orientation.w = 1.0;
      goal_marker.scale.x = 0.04;
      goal_marker.scale.y = 0.04;
      goal_marker.scale.z = 0.04;
      goal_marker.color.r = 0.0F;
      goal_marker.color.g = 1.0F;
      goal_marker.color.b = 0.0F;
      goal_marker.color.a = 0.8F;
      goal_marker_pub_->publish(goal_marker);
    }

    if (!publish_context_visualization) {
      visualization_cleared_for_inactive_ = false;
      return;
    }

    const auto state = backend_->read_state();
    const auto context = RB10Model::forward_context(
      state.q, ControlPointJacobianMode::kAnalytic);

    if (debug_state_pub_) {
      publish_debug_state(state, context);
    }

    if (control_point_pub_) {
      visualization_msgs::msg::MarkerArray points;
      for (std::size_t index = 0; index < context.control_points.size(); ++index) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = stamp;
        marker.ns = "control_points";
        marker.id = static_cast<int>(index);
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = context.control_points[index].position.x();
        marker.pose.position.y = context.control_points[index].position.y();
        marker.pose.position.z = context.control_points[index].position.z();
        marker.pose.orientation.w = 1.0;
        const double marker_diameter = control_point_marker_diameter_ > 0.0 ?
          control_point_marker_diameter_ :
          context.control_points[index].radius * 2.0;
        marker.scale.x = marker_diameter;
        marker.scale.y = marker_diameter;
        marker.scale.z = marker_diameter;
        marker.color.r = 0.0F;
        marker.color.g = 0.5F;
        marker.color.b = 1.0F;
        marker.color.a = 0.3F;
        points.markers.push_back(marker);
      }
      control_point_pub_->publish(points);
    }

    if (body_obstacle_pub_) {
      visualization_msgs::msg::MarkerArray body_obstacles;
      for (std::size_t index = 0; index < body_obstacles_visual_.size(); ++index) {
        const auto & obstacle = body_obstacles_visual_[index];
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = stamp;
        marker.ns = "body_obstacles";
        marker.id = static_cast<int>(index);
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.color.r = 1.0F;
        marker.color.g = 0.6F;
        marker.color.b = 0.0F;
        marker.color.a = 0.22F;

        Eigen::Vector3d center = obstacle.center;
        Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
        if (!obstacle.link_name.empty()) {
          const auto link_index = link_index_from_name(obstacle.link_name);
          rotation = context.link_rotations[link_index];
          if (obstacle.type == "box") {
            center =
              context.link_positions[link_index] +
              rotation * (0.5 * (obstacle.mins + obstacle.maxs));
          } else {
            center =
              context.link_positions[link_index] +
              rotation * obstacle.center;
          }
        } else if (obstacle.type == "box") {
          center = 0.5 * (obstacle.mins + obstacle.maxs);
        }

        marker.pose.position.x = center.x();
        marker.pose.position.y = center.y();
        marker.pose.position.z = center.z();

        if (obstacle.type == "ball") {
          marker.type = visualization_msgs::msg::Marker::SPHERE;
          marker.pose.orientation.w = 1.0;
          marker.scale.x = obstacle.radius * 2.0;
          marker.scale.y = obstacle.radius * 2.0;
          marker.scale.z = obstacle.radius * 2.0;
        } else if (obstacle.type == "box") {
          marker.type = visualization_msgs::msg::Marker::CUBE;
          Eigen::Quaterniond q(rotation);
          q.normalize();
          marker.pose.orientation.x = q.x();
          marker.pose.orientation.y = q.y();
          marker.pose.orientation.z = q.z();
          marker.pose.orientation.w = q.w();
          const Eigen::Vector3d size = obstacle.maxs - obstacle.mins;
          marker.scale.x = size.x();
          marker.scale.y = size.y();
          marker.scale.z = size.z();
        } else {
          continue;
        }

        body_obstacles.markers.push_back(marker);
      }
      body_obstacle_pub_->publish(body_obstacles);
    }

    publish_repulsion_metric_markers(state, context);
    publish_tcp_accel_marker(context);
    publish_tangent_escape_geometry_markers(context);

    if (eef_pose_pub_) {
      geometry_msgs::msg::PoseStamped eef_pose;
      const Eigen::Vector3d & eef = context.tcp_position;
      eef_pose.header.frame_id = "base_link";
      eef_pose.header.stamp = stamp;
      eef_pose.pose.position.x = eef.x();
      eef_pose.pose.position.y = eef.y();
      eef_pose.pose.position.z = eef.z();
      Eigen::Quaterniond tcp_orientation(context.link_rotations[RB10Model::TCP_RMP]);
      tcp_orientation.normalize();
      eef_pose.pose.orientation.x = tcp_orientation.x();
      eef_pose.pose.orientation.y = tcp_orientation.y();
      eef_pose.pose.orientation.z = tcp_orientation.z();
      eef_pose.pose.orientation.w = tcp_orientation.w();
      eef_pose_pub_->publish(eef_pose);
    }
    visualization_cleared_for_inactive_ = false;
  }

  void clear_rmp_visualization()
  {
    if (visualization_cleared_for_inactive_) {
      return;
    }

    visualization_msgs::msg::Marker clear_goal;
    clear_goal.header.frame_id = "base_link";
    clear_goal.header.stamp = now();
    clear_goal.action = visualization_msgs::msg::Marker::DELETEALL;
    if (goal_marker_pub_) {
      goal_marker_pub_->publish(clear_goal);
    }

    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header.frame_id = "base_link";
    clear_marker.header.stamp = now();
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;

    visualization_msgs::msg::MarkerArray clear_array;
    clear_array.markers.push_back(clear_marker);
    if (control_point_pub_) {
      control_point_pub_->publish(clear_array);
    }
    if (body_obstacle_pub_) {
      body_obstacle_pub_->publish(clear_array);
    }
    if (repulsion_metric_pub_) {
      repulsion_metric_pub_->publish(clear_array);
    }
    if (tcp_accel_pub_) {
      tcp_accel_pub_->publish(clear_marker);
    }
    if (tangent_escape_geometry_marker_pub_) {
      tangent_escape_geometry_marker_pub_->publish(clear_array);
    }
    visualization_cleared_for_inactive_ = true;
  }

  std::atomic<bool> running_;
  std::thread control_thread_;
  std::unique_ptr<ControllerBackend> backend_;
  std::unique_ptr<RmpSolverInterface> solver_;
  RobotState state_;
  RobotState virtual_velocity_state_;
  RobotState last_safe_command_state_;
  JointVector controller_velocity_previous_q_{JointVector::Zero()};
  JointVector controller_velocity_estimate_{JointVector::Zero()};
  std::chrono::steady_clock::time_point controller_velocity_previous_time_{};
  bool virtual_velocity_state_initialized_{false};
  bool last_safe_command_state_initialized_{false};
  bool controller_velocity_estimator_initialized_{false};
  Eigen::Vector3d body_goal_;
  CommandMode command_mode_{CommandMode::kPosition};
  bool hardware_command_enabled_{true};
  bool backend_command_application_enabled_{true};
  TargetRmpParams target_metric_params_;
  CollisionRmpParams collision_metric_params_;
  realtime_tools::RealtimeBox<GoalTarget> goal_target_box_;
  realtime_tools::RealtimeBox<std::vector<ObstacleSphere>> obstacles_box_;
  realtime_tools::RealtimeBox<std::vector<std::optional<ObstacleSphere>>>
  proximity_sensor_obstacles_box_;
  realtime_tools::RealtimeBox<TcpAccelerationSample> tcp_accel_visualization_box_;
  realtime_tools::RealtimeBox<std::vector<double>> tangent_escape_visualization_box_;
  realtime_tools::RealtimeBox<std::vector<double>> tangent_escape_dual_visualization_box_;
  std::vector<BodyObstacle> body_obstacles_visual_;
  std::atomic<bool> last_min_z_safety_triggered_{false};
  bool initialize_goal_from_first_state_{true};
  std::atomic<bool> startup_goal_synced_{false};
  std::atomic<bool> external_goal_received_{false};
  mutable std::mutex external_rmp_mutex_;
  std::unordered_map<std::string, ExternalRmpBuffer> external_rmp_buffers_;
  bool publish_joint_states_enabled_{true};
  bool publish_visualization_enabled_{true};
  bool publish_goal_marker_enabled_{true};
  bool publish_control_point_markers_enabled_{true};
  bool publish_body_obstacle_markers_enabled_{true};
  bool publish_end_effector_pose_enabled_{true};
  bool publish_debug_state_enabled_{true};
  bool publish_repulsion_metric_markers_enabled_{true};
  bool publish_tcp_accel_marker_enabled_{true};
  bool publish_rmp_ee_pose_enabled_{true};
  bool publish_goal_tf_enabled_{true};
  double control_point_marker_diameter_{0.10};
  std::string joint_state_topic_{"joint_states"};
  std::string goal_tf_parent_frame_{"base_link"};
  std::string goal_tf_child_frame_{"rmp_goal_target"};
  bool rmp_flag_gate_enabled_{false};
  bool visualization_cleared_for_inactive_{false};
  double min_goal_z_{0.05};
  bool safety_stop_on_min_z_{true};
  double workspace_floor_z_{0.0};
  double min_link6_z_{0.03};
  double min_tcp_z_{0.05};
  double control_rate_hz_{100.0};
  bool publish_control_timing_debug_enabled_{false};
  double control_timing_publish_rate_hz_{10.0};
  ControlTimingAccumulator control_timing_accumulator_;
  bool enable_realtime_{false};
  int realtime_priority_{80};
  bool lock_memory_{false};
  bool estimate_velocity_in_controller_{false};
  double measured_position_feedback_blend_{1.0};
  double measured_velocity_feedback_blend_{0.35};
  bool use_velocity_feedback_in_solver_{true};
  double controller_velocity_filter_alpha_{0.25};
  double max_joint_accel_{20.0};
  bool preserve_joint_accel_direction_{true};
  double command_guard_max_step_rad_{0.00436332313};
  double command_guard_max_velocity_rad_s_{0.436332313};
  bool command_guard_velocity_slew_enabled_{true};
  double command_guard_max_accel_rad_s2_{10.0};
  JointVector previous_velocity_command_{JointVector::Zero()};
  bool previous_velocity_command_initialized_{false};
  std::uint64_t velocity_slew_clamp_count_{0U};
  bool predictive_joint_limit_guard_{true};
  std::array<double, 6> joint_lower_limits_ = RB10Model::joint_lower_limits;
  std::array<double, 6> joint_upper_limits_ = RB10Model::joint_upper_limits;
  std::array<double, 6> joint_limit_buffers_{0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
  double repulsion_metric_marker_min_norm_{0.01};
  double repulsion_metric_marker_dot_diameter_{0.04};
  double tcp_accel_marker_max_length_{0.15};
  double tcp_accel_marker_norm_for_max_length_{2.0};
  double tcp_accel_marker_min_norm_{0.001};
  bool publish_tangent_escape_rmp_data_enabled_{false};
  bool publish_tangent_escape_dual_solve_data_enabled_{false};
  bool publish_leaf_ablation_data_enabled_{false};
  bool publish_stability_certificate_data_enabled_{false};
  bool publish_tangent_escape_geometry_debug_enabled_{false};
  bool tangent_escape_rmp_enabled_{false};
  bool production_stability_analysis_enabled_{false};
  bool production_stability_analysis_completed_{false};
  std::string production_stability_report_path_;
  double tangent_escape_geometry_marker_length_{0.12};
  int rmp_active_flag_value_{1};
  std::atomic<bool> rmp_active_{true};

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr direct_command_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr position_command_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_q_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_metric_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr rmp_joint_accel_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr rmp_tcp_accel_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr control_timing_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr tangent_escape_rmp_data_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
    tangent_escape_dual_solve_data_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
    leaf_ablation_data_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
    stability_certificate_data_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
    tangent_escape_geometry_data_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr control_point_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr body_obstacle_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr repulsion_metric_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    tangent_escape_geometry_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr tcp_accel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr eef_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr rmp_eef_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr debug_state_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> goal_tf_broadcaster_;
  std::shared_ptr<JointStateRtPublisher> rt_joint_state_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_direct_command_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_position_command_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_target_q_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_target_metric_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_rmp_joint_accel_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_rmp_tcp_accel_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_control_timing_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_tangent_escape_rmp_data_pub_;
  std::shared_ptr<Float64ArrayRtPublisher>
    rt_tangent_escape_dual_solve_data_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_leaf_ablation_data_pub_;
  std::shared_ptr<Float64ArrayRtPublisher>
    rt_stability_certificate_data_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_tangent_escape_geometry_data_pub_;
  std::shared_ptr<PoseRtPublisher> rt_rmp_eef_pose_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr goal_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_sub_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr rmp_flag_sub_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr obstacle_sub_;
  std::vector<rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr> external_metric_subs_;
  std::vector<rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr> external_accel_subs_;
  rclcpp::TimerBase::SharedPtr visualization_timer_;
};

}  // namespace

}  // namespace rb10_rmpflow_rviz

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rb10_rmpflow_rviz::RmpflowControllerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

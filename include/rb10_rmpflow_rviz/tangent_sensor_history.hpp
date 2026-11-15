#pragma once

#include <array>
#include <deque>
#include <limits>

#include "rb10_rmpflow_rviz/rb10_model.hpp"

namespace rb10_rmpflow_rviz::tangent_sensor_history
{
using JointVector = RB10Model::JointVector;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
constexpr std::size_t kSensorCount = RB10Model::sensor_control_points.size();

// Keep id 2 for recordings of the former fixed-angle baseline.
enum class DirectionMode { kLegacy = 0, kSensorHistory = 1, kFixed = 2, kSimple = 3 };

struct Params
{
  double window{1.0};              // local regression history, seconds
  double horizon{0.12};            // frozen-policy lookahead, seconds
  double clearance{0.08};          // calibrated range, NOT geometric clearance
  double risk_length{0.05};
  double weight_goal{0.5};
  double weight_escape{0.2};
  double escape_length{0.01};  // common displacement scale, metres
  double reselect_interval{0.1};
  double weight_neighbor_risk{0.0}; // observed, supported non-owner closing only
  double weight_risk{0.0};  // range extrapolation is diagnostic by default
  double weight_continuity{0.1};
  double unknown_motion_weight{0.1};
  double max_direction_rate{1.5};  // rad/s inside the owner tangent plane
  double direction_filter_tau{0.1};
  double handoff_time{0.15};
  double initial_angle{3.14159265358979323846};
  // Estimator safeguards, shared by both experimental leaves.
  double fresh_time{0.15};
  double expiry{0.6};
  double sample_interval{0.02};
  double ridge{1e-5};              // rad^2, limits unsupported joint directions
  double noise_floor{0.002};       // equivalent-distance noise, metres
  double max_joint_step{0.05};     // local model trust radius, radians
};

struct Sample
{
  double stamp{-1.0};
  double range{std::numeric_limits<double>::quiet_NaN()};
};

// q follows the configured backend source (possibly reference), before blending.
// encoder_q/encoder_stamp are independent physical feedback for direction trials.
// Timestamps share the controller clock. Each sensor retains its acquisition
// stamp; republishing a sample never creates another training observation.
struct Input
{
  // Last post-guard command submitted to the backend, not predicted motion.
  double command_stamp{-1.0};
  JointVector command_qd{JointVector::Zero()};
  bool command_available{false};
  double encoder_stamp{-1.0};
  JointVector encoder_q{JointVector::Zero()};
  bool source_available{true};
  double stamp{-1.0};
  JointVector q{JointVector::Zero()};
  JointVector qd{JointVector::Zero()};
  bool velocity_available{false};  // measured/encoder-estimated, before command blending
  std::array<Sample, kSensorCount> samples{};
};

struct Model
{
  bool valid{false};
  JointVector gradient{JointVector::Zero()};
  Matrix6 information_inverse{Matrix6::Identity()};
  JointVector anchor_q{JointVector::Zero()};
  double range{0.0};
  double stamp{-1.0};
  double noise{0.002};
  double rate{0.0};
  std::size_t count{0};
};

struct Risk
{
  // Diagnostic only: retain the original conservative cost used by control.
  double nominal_cost{0}, uncertainty_cost{0}, shadow_support{0};
  double critical_change{0}, critical_sigma{0};
  int shadow_sensor{-1};
  JointVector predicted_displacement{JointVector::Zero()};
  double unclipped_displacement_norm{0.0};
  double value{0.0};
  double gradient{0.0};           // supplied by the direction objective
  double escape{0.0};
  double escape_gradient{0.0};
  double measured{0.0};
  double unknown_motion{0.0};
  double min_predicted_range{std::numeric_limits<double>::infinity()};
  double confidence{0.0};
  std::size_t model_count{0};
  std::size_t unknown_count{0};
  int critical_sensor{-1};
};

struct PredictionCheck
{
  double stamp{-1}, model_stamp{-1}, predicted{0}, observed{0}, error{0}, sigma{0}, support{0};
  bool valid{false};
};

class History
{
public:
  explicit History(Params params = {}) : params_(params) {}
  void update(const Input & input);
  void reset();
  const std::array<Model, kSensorCount> & models() const { return models_; }
  const Input & current() const { return current_; }
  const auto & prediction_checks() const { return prediction_checks_; }
  const std::array<int, kSensorCount> & update_reasons() const { return update_reasons_; }
  const std::array<double, kSensorCount> & update_stamps() const { return update_stamps_; }
  // Fresh scalar samples only: no virtual spheres or synthetic hold ramps.
  std::array<double, kSensorCount> ranges() const;
  std::array<double, kSensorCount> rates() const;
  Risk evaluate(const JointVector & displacement, const KinematicsContext & context) const;
  // Soft cost in [0,1], not a free-space certificate. Unknown/stale/unexcited
  // directions carry no invented bearing or directional risk.
  Risk evaluate_neighbors(const JointVector & displacement, std::size_t owner) const;

private:
  std::array<PredictionCheck,kSensorCount> prediction_checks_{};
  // 1 accepted, 2 invalid range cleared, 3 expired, 4 discontinuity reset,
  // 5 pose interpolation unavailable, 6 out of range/stale, 7 interval skip,
  // 8 clock reset. Timestamp is controller observation time of that decision.
  std::array<int, kSensorCount> update_reasons_{};
  std::array<double, kSensorCount> update_stamps_{};
  struct Observation { double stamp; JointVector q; double range; };
  struct Pose { double stamp; JointVector q; };
  bool interpolate(double stamp, JointVector & q) const;
  void fit(std::size_t index);
  Params params_;
  Input current_{};
  std::deque<Pose> poses_;
  std::array<std::deque<Observation>, kSensorCount> observations_;
  std::array<Model, kSensorCount> models_{};
  std::array<double, kSensorCount> processed_stamps_ = [] {
      std::array<double, kSensorCount> result{}; result.fill(-1.0); return result;
    }();
};
}  // namespace rb10_rmpflow_rviz::tangent_sensor_history

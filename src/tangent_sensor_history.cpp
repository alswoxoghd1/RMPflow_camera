#include "rb10_rmpflow_rviz/tangent_sensor_history.hpp"

#include <algorithm>
#include <cmath>

namespace rb10_rmpflow_rviz::tangent_sensor_history
{
namespace
{
double penalty(double range, const Params & params)
{
  const double x = (params.clearance - range) / params.risk_length;
  const double softplus = std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
  return softplus * softplus;
}
}

void History::reset()
{
  current_ = Input{};
  poses_.clear();
  for (auto & observations : observations_) { observations.clear(); }
  models_ = {};
  prediction_checks_ = {};
  processed_stamps_.fill(-1.0);
}

bool History::interpolate(double stamp, JointVector & q) const
{
  if (poses_.empty() || stamp < poses_.front().stamp || stamp > poses_.back().stamp) {
    return false;
  }
  for (std::size_t k = 0; k < poses_.size(); ++k) {
    if (poses_[k].stamp < stamp) { continue; }
    if (k == 0 || poses_[k].stamp == stamp) { q = poses_[k].q; return true; }
    const auto & before = poses_[k - 1];
    const auto & after = poses_[k];
    if (after.stamp - before.stamp > 0.05) { return false; }
    const double fraction = (stamp - before.stamp) / (after.stamp - before.stamp);
    q = before.q + fraction * (after.q - before.q);
    return true;
  }
  return false;
}

void History::update(const Input & input)
{
  if (!std::isfinite(input.stamp) || input.stamp < 0.0 || !input.q.allFinite()) { return; }
  if (input.stamp < current_.stamp) {
    reset(); update_reasons_.fill(8); update_stamps_.fill(input.stamp);
  }
  if (input.stamp == current_.stamp) { return; }
  current_ = input;
  poses_.push_back({input.stamp, input.q});
  while (poses_.size() > 2 && poses_[1].stamp < input.stamp - params_.window) {
    poses_.pop_front();
  }
  for (std::size_t i = 0; i < kSensorCount; ++i) {
    const Sample & sample = input.samples[i];
    auto & observations = observations_[i];
    auto & model = models_[i];
    if (model.stamp >= 0.0 && input.stamp - model.stamp > params_.expiry) {
      observations.clear(); model = Model{};
      update_reasons_[i] = 3; update_stamps_[i] = input.stamp;
    }
    if (!std::isfinite(sample.stamp) || sample.stamp < 0.0 ||
      sample.stamp <= processed_stamps_[i] || sample.stamp > input.stamp)
    {
      continue;
    }
    processed_stamps_[i] = sample.stamp;
    prediction_checks_[i] = PredictionCheck{};
    prediction_checks_[i].stamp = sample.stamp;
    update_stamps_[i] = input.stamp;
    update_reasons_[i] = 1;
    // NaN is invalid/disabled; +infinity is out of sensing range. Retain a
    // recently observed model for a bounded time only in the latter case.
    if (std::isnan(sample.range) || sample.range < 0.0) {
      update_reasons_[i] = 2;
      observations.clear(); model = Model{}; continue;
    }
    if (!std::isfinite(sample.range) || input.stamp - sample.stamp > params_.fresh_time) {
      update_reasons_[i] = 6;
      continue;
    }
    JointVector sample_q;
    if (!interpolate(sample.stamp, sample_q)) { update_reasons_[i] = 5; continue; }
    if (!observations.empty() &&
      sample.stamp - observations.back().stamp < params_.sample_interval)
    {
      update_reasons_[i] = 7;
      continue;
    }
    // Validate the PREVIOUS model on a newly acquired sample, before this
    // sample enters fitting (including abrupt changes, which must not disappear
    // from validation just because fitting subsequently resets).
    const JointVector validation_step=sample_q-model.anchor_q;
    if(model.valid && sample.stamp>model.stamp && sample.stamp-model.stamp<=params_.expiry &&
       validation_step.norm()<=params_.max_joint_step){
      auto & check=prediction_checks_[i];
      check.valid=true;check.model_stamp=model.stamp;
      check.predicted=model.range+model.gradient.dot(validation_step);
      check.observed=sample.range;check.error=sample.range-check.predicted;
      const double variance=std::max(0.,validation_step.dot(model.information_inverse*validation_step));
      check.sigma=model.noise*std::sqrt(1.+variance);
      check.support=validation_step.squaredNorm()>1e-16 ?
        std::clamp(1.-params_.ridge*variance/validation_step.squaredNorm(),0.,1.) : 0.;
    }
    if (!observations.empty()) {
      const auto & last = observations.back();
      const double dt = sample.stamp - last.stamp;
      const double rate = (sample.range - last.range) / dt;
      // Abrupt target/fusion/calibration changes invalidate a local fit.
      if (dt > params_.fresh_time || std::abs(rate) > 2.0 ||
        (sample_q - last.q).norm() > 0.15)
      {
        observations.clear(); model = Model{};
        update_reasons_[i] = 4;
      } else {
        model.rate += dt / (0.08 + dt) * (rate - model.rate);
      }
    }
    observations.push_back({sample.stamp, sample_q, sample.range});
    while (observations.size() > 64 ||
      (!observations.empty() && observations.front().stamp < sample.stamp - params_.window))
    {
      observations.pop_front();
    }
    fit(i);
  }
}

void History::fit(std::size_t index)
{
  const auto & observations = observations_[index];
  auto & model = models_[index];
  model.anchor_q = observations.back().q;
  model.range = observations.back().range;
  model.stamp = observations.back().stamp;
  model.count = observations.size();
  model.valid = false;
  if (observations.size() < 6) { return; }
  JointVector mean_q = JointVector::Zero();
  double mean_range = 0.0;
  for (const auto & observation : observations) {
    mean_q += observation.q; mean_range += observation.range;
  }
  mean_q /= static_cast<double>(observations.size());
  mean_range /= static_cast<double>(observations.size());
  Matrix6 gram = Matrix6::Zero();
  JointVector rhs = JointVector::Zero();
  for (const auto & observation : observations) {
    const JointVector dq = observation.q - mean_q;
    gram.noalias() += dq * dq.transpose();
    rhs.noalias() += dq * (observation.range - mean_range);
  }
  const Eigen::LDLT<Matrix6> factor(gram + params_.ridge * Matrix6::Identity());
  model.gradient = factor.solve(rhs);
  model.information_inverse = factor.solve(Matrix6::Identity());
  double variance = 0.0;
  for (const auto & observation : observations) {
    const double error = observation.range - mean_range -
      model.gradient.dot(observation.q - mean_q);
    variance += error * error;
  }
  model.noise = std::max(params_.noise_floor,
    std::sqrt(variance / static_cast<double>(observations.size())));
  model.valid = gram.trace() > params_.ridge && model.gradient.allFinite() &&
    model.information_inverse.allFinite();
}

std::array<double, kSensorCount> History::ranges() const
{
  std::array<double, kSensorCount> result;
  result.fill(std::numeric_limits<double>::infinity());
  for (std::size_t i = 0; i < kSensorCount; ++i) {
    const auto & sample = current_.samples[i];
    const double age = current_.stamp - sample.stamp;
    if (sample.stamp >= 0.0 && age >= 0.0 && age <= params_.fresh_time &&
      std::isfinite(sample.range) && sample.range >= 0.0)
    {
      result[i] = sample.range;
    }
  }
  return result;
}

std::array<double, kSensorCount> History::rates() const
{
  std::array<double, kSensorCount> result{};
  const auto fresh = ranges();
  for (std::size_t i = 0; i < kSensorCount; ++i) {
    if (std::isfinite(fresh[i]) && models_[i].stamp >= 0.0 &&
      current_.stamp - models_[i].stamp <= params_.fresh_time)
    {
      result[i] = models_[i].rate;
    }
  }
  return result;
}

Risk History::evaluate(const JointVector & displacement, const KinematicsContext & context) const
{
  Risk risk;
  if (!displacement.allFinite() || current_.stamp < 0.0) { return risk; }
  const auto fresh = ranges();
  double nearby = 0.0;
  for (double distance : fresh) {
    if (std::isfinite(distance)) {
      nearby = std::max(nearby, std::exp(-std::max(distance - params_.clearance, 0.0) /
        params_.risk_length));
    }
  }
  const std::size_t count = std::min(context.control_point_jacobians.size(), kSensorCount);
  double critical_cost = 0.0;
  double min_support=1.,largest_closing=-1.;
  std::size_t observed_count=0;
  for (std::size_t i = 0; i < count; ++i) {
    const auto & model = models_[i];
    const JointVector from_anchor = current_.q - model.anchor_q;
    const double age = current_.stamp - model.stamp;
    const bool supported = model.valid && age >= 0.0 && age <= params_.expiry &&
      from_anchor.norm() <= params_.max_joint_step;
    const bool observed=std::isfinite(fresh[i]);
    if(observed)++observed_count;
    if (!supported) {
      if(observed)min_support=0.;
      ++risk.unknown_count;
      // Motion exposure only. This is NOT a fabricated obstacle distance or
      // a prediction of first contact. Symmetric for opposite motions.
      risk.unknown_motion += nearby *
        (context.control_point_jacobians[i] * displacement).squaredNorm() / (0.02 * 0.02);
      continue;
    }
    ++risk.model_count;
    const double freshness = std::clamp(1.0 - age / params_.expiry, 0.0, 1.0);
    const double variance0 = std::max(0.0,
      from_anchor.dot(model.information_inverse * from_anchor));
    const JointVector future = from_anchor + displacement;
    const double variance1 = std::max(0.0, future.dot(model.information_inverse * future));
    const double uncertainty_increase = 2.0 * model.noise * std::max(0.0,
      std::sqrt(1.0 + variance1) - std::sqrt(1.0 + variance0));
    const double current_range = model.range + model.gradient.dot(from_anchor);
    const double change = model.gradient.dot(displacement);
    const double predicted = current_range + std::min(change, 0.0) - uncertainty_increase;
    const double cost = freshness * std::max(0.0,
      penalty(predicted, params_) - penalty(current_range, params_));
    const double nominal_cost=freshness*std::max(0.,
      penalty(current_range+std::min(change,0.),params_)-penalty(current_range,params_));
    risk.nominal_cost+=nominal_cost;
    risk.uncertainty_cost+=std::max(0.,cost-nominal_cost);
    risk.measured += cost;
    risk.min_predicted_range = std::min(risk.min_predicted_range, predicted);
    const double support = displacement.squaredNorm() > 1e-16 ?
      std::clamp(1.0 - params_.ridge *
      displacement.dot(model.information_inverse * displacement) /
      displacement.squaredNorm(), 0.0, 1.0) : 0.0;
    risk.confidence += freshness * support;
    if(observed)min_support=std::min(min_support,
      future.norm()<=params_.max_joint_step?freshness*support:0.);
    if(observed && std::max(-change,0.)>largest_closing){
      largest_closing=std::max(-change,0.);risk.shadow_sensor=static_cast<int>(i);
      risk.critical_change=change;
      risk.critical_sigma=model.noise*std::sqrt(std::max(0.,
        displacement.dot(model.information_inverse*displacement))+2.);
    }
    if (cost > critical_cost) { critical_cost = cost; risk.critical_sensor = static_cast<int>(i); }
  }
  // Fixed normalization avoids a score jump when a sensor first detects the
  // obstacle. A newly detected, sufficiently distant target is not a hazard.
  risk.unknown_motion *= params_.unknown_motion_weight / std::max<std::size_t>(count, 1);
  risk.value = risk.measured + risk.unknown_motion;
  risk.shadow_support=observed_count?min_support:0.;
  if (risk.model_count > 0) { risk.confidence /= static_cast<double>(risk.model_count); }
  return risk;
}
NeighborRisk History::evaluate_neighbors(
  const Eigen::Vector3d & tangent, std::size_t owner, const KinematicsContext & context) const
{
  NeighborRisk result;
  if (!tangent.allFinite() || tangent.norm()<=1e-12) { return result; }
  const Eigen::Vector3d direction=tangent.normalized();
  const auto fresh=ranges();
  for (std::size_t k=0;k<kSensorCount;++k) {
    const double limit=params_.neighbor_distances[k];
    if (k==owner || !std::isfinite(fresh[k]) || !std::isfinite(limit) || limit<=0.0) continue;
    const double closeness=std::clamp(1.0-fresh[k]/limit,0.0,1.0);
    if (closeness<=0.0) continue;
    const auto & sensor=RB10Model::sensor_control_points[k];
    const Eigen::Vector3d axis=context.link_rotations[sensor.parent_link]*sensor.local_normal;
    if (!axis.allFinite() || axis.norm()<=1e-12) continue;
    ++result.detection_count;
    const double alignment=std::clamp(axis.normalized().dot(direction),0.0,1.0);
    const double cost=closeness*alignment;
    if (cost>result.value) {
      result.value=cost; result.alignment=alignment; result.closeness=closeness;
      result.critical_sensor=static_cast<int>(k); result.range=fresh[k]; result.distance_limit=limit;
    }
  }
  return result;
}
}  // namespace rb10_rmpflow_rviz::tangent_sensor_history

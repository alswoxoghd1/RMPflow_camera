#include <cmath>
#include <limits>
#include <gtest/gtest.h>
#include "rb10_rmpflow_rviz/tg_initial_event.hpp"

#include "rb10_rmpflow_rviz/tangent_sensor_history.hpp"
#include "rb10_rmpflow_rviz/tangent_escape_supervisor.hpp"
#include "rb10_rmpflow_rviz/simple_tangent_leaf.hpp"

namespace rb10_rmpflow_rviz
{
namespace
{
namespace history = tangent_sensor_history;
namespace ode = tangent_escape_ode;
using JointVector = RB10Model::JointVector;
constexpr double pi = 3.14159265358979323846;

KinematicsContext history_context()
{
  KinematicsContext context;
  for (std::size_t k = 0; k < context.link_rotations.size(); ++k) {
    context.link_rotations[k].setIdentity();
    context.link_angular_jacobians[k].setZero();
  }
  for (std::size_t i = 0; i < history::kSensorCount; ++i) {
    context.control_points.push_back({Eigen::Vector3d(0.2 * i, 0.0, 0.0), 0.01});
    RB10Model::Jacobian jacobian = RB10Model::Jacobian::Zero();
    jacobian.leftCols<3>().setIdentity();
    context.control_point_jacobians.push_back(jacobian);
    context.control_point_velocities.push_back(Eigen::Vector3d::Zero());
    context.control_point_curvatures.push_back(Eigen::Vector3d::Zero());
  }
  return context;
}

history::Input training_input(int k, bool second = false, double baseline = 0.09)
{
  history::Input input;
  input.stamp = 100.0 + 0.025 * k;
  input.q[0] = 0.025 * std::sin(2.0 * pi * k / 60.0);
  input.samples[0] = {input.stamp, baseline - 0.7 * input.q[0]};
  if (second) { input.samples[1] = {input.stamp, baseline + 0.7 * input.q[0]}; }
  return input;
}

TEST(TangentSensorHistory, LearnsScalarResponseWithoutAssumingObstacleBearing)
{
  history::History estimator;
  for (int k = 0; k <= 60; ++k) { estimator.update(training_input(k)); }
  ASSERT_TRUE(estimator.models()[0].valid);
  EXPECT_NEAR(estimator.models()[0].gradient[0], -0.7, 0.01);
  EXPECT_NEAR(estimator.models()[0].gradient.tail<5>().norm(), 0.0, 1e-12);
  JointVector delta = JointVector::Zero(); delta[0] = 0.01;
  const auto toward = estimator.evaluate(delta, history_context());
  const auto away = estimator.evaluate(-delta, history_context());
  EXPECT_GT(toward.measured, away.measured);
  EXPECT_LT(toward.min_predicted_range, 0.09);
  EXPECT_GT(toward.confidence, 0.9);
}

TEST(TangentSensorHistory, ShadowDecompositionDoesNotChangeConservativeCost)
{
  history::History estimator;
  for(int k=0;k<=60;++k)estimator.update(training_input(k));
  JointVector step=JointVector::Zero();step[0]=.003;
  auto r=estimator.evaluate(step,history_context());
  EXPECT_NEAR(r.nominal_cost+r.uncertainty_cost,r.measured,1e-12);
  EXPECT_NEAR(r.nominal_cost+r.uncertainty_cost+r.unknown_motion,r.value,1e-12);
  EXPECT_GE(r.uncertainty_cost,0.);EXPECT_EQ(r.shadow_sensor,0);
  step.setZero();step[5]=.003;
  r=estimator.evaluate(step,history_context());
  EXPECT_LT(r.shadow_support,.1); // unknown motion must not be called supported
  EXPECT_GT(r.critical_sigma,0.);
}

TEST(TangentSensorHistory, ValidationUsesPreviousModelAndDoesNotReuseSampleForAccuracy)
{
  history::History estimator;
  for(int k=0;k<=60;++k)estimator.update(training_input(k));
  const auto model=estimator.models()[0];ASSERT_TRUE(model.valid);
  auto input=training_input(61);input.samples[0].range+=.01;
  const double predicted=model.range+model.gradient.dot(input.q-model.anchor_q);
  estimator.update(input);
  const auto check=estimator.prediction_checks()[0];ASSERT_TRUE(check.valid);
  EXPECT_DOUBLE_EQ(check.model_stamp,model.stamp);
  EXPECT_NEAR(check.predicted,predicted,1e-12);
  EXPECT_NEAR(check.error,input.samples[0].range-predicted,1e-12);
  EXPECT_GT(std::abs(check.error),.005);
  input.stamp+=.005;estimator.update(input);
  EXPECT_DOUBLE_EQ(estimator.prediction_checks()[0].stamp,check.stamp);
  estimator.reset();EXPECT_FALSE(estimator.prediction_checks()[0].valid);
}

TEST(TangentSensorHistory, UnexcitedDirectionIsNotReportedAsConfident)
{
  history::History estimator;
  for (int k = 0; k <= 60; ++k) { estimator.update(training_input(k)); }
  JointVector delta = JointVector::Zero(); delta[5] = 0.01;
  const auto result = estimator.evaluate(delta, history_context());
  EXPECT_NEAR(result.confidence, 0.0, 1e-12);
  EXPECT_GT(result.measured, 0.0);  // uncertain response, not known-safe zero
}

TEST(TangentSensorHistory, RepeatedAcquisitionStampNeverCreatesTrainingMotion)
{
  history::History estimator;
  auto input = training_input(0);
  for (int k = 0; k < 16; ++k) {
    input.stamp = 100.0 + 0.025 * k;
    input.q[0] += 0.001;
    estimator.update(input);
  }
  EXPECT_FALSE(estimator.models()[0].valid);
  EXPECT_LE(estimator.models()[0].count, 1U);
  EXPECT_FALSE(std::isfinite(estimator.ranges()[0]));
}

TEST(TangentSensorHistory, InterpolatesMeasuredPoseAtAcquisitionTime)
{
  history::History estimator;
  for (int k = 0; k < 120; ++k) {
    history::Input input;
    input.stamp = 100.0 + 0.01 * k;
    input.q[0] = 0.05 * (input.stamp - 100.0);
    const double sample_stamp = input.stamp - 0.015;
    input.samples[0] = {sample_stamp, 0.2 - (sample_stamp - 100.0) * 0.05};
    estimator.update(input);
  }
  const auto & model = estimator.models()[0];
  ASSERT_TRUE(model.valid);
  EXPECT_NEAR(model.anchor_q[0], 0.05 * (model.stamp - 100.0), 1e-10);
  EXPECT_NEAR(model.gradient[0], -1.0, 0.015);
}

TEST(TangentSensorHistory, StationaryHistoryDoesNotIdentifyAResponseGradient)
{
  history::History estimator;
  for (int k = 0; k < 30; ++k) {
    auto input = training_input(k); input.q.setZero();
    estimator.update(input);
  }
  EXPECT_FALSE(estimator.models()[0].valid);
}

TEST(TangentSensorHistory, InvalidDataExpiryAndClockResetDiscardTheModel)
{
  history::History estimator;
  for (int k = 0; k <= 60; ++k) { estimator.update(training_input(k)); }
  auto input = training_input(61);
  input.samples[0].range = std::numeric_limits<double>::infinity();
  estimator.update(input);
  EXPECT_TRUE(estimator.models()[0].valid);  // bounded last-seen prediction
  input.stamp += 0.7;
  estimator.update(input);
  EXPECT_FALSE(estimator.models()[0].valid);
  for (int k = 0; k <= 60; ++k) { estimator.update(training_input(k)); }
  EXPECT_TRUE(estimator.models()[0].valid);
  input = training_input(61);
  input.samples[0].range = std::numeric_limits<double>::quiet_NaN();
  estimator.update(input);
  EXPECT_FALSE(estimator.models()[0].valid);
  input = training_input(0); input.stamp = 1.0; input.samples[0].stamp = 1.0;
  estimator.update(input);
  EXPECT_LE(estimator.models()[0].count, 1U);
}

TEST(TangentSensorHistory, RetreatCannotCancelAnotherSensorsApproach)
{
  history::Params params; params.unknown_motion_weight = 0.0;
  history::History first(params), both(params);
  for (int k = 0; k <= 60; ++k) {
    first.update(training_input(k)); both.update(training_input(k, true));
  }
  JointVector delta = JointVector::Zero(); delta[0] = 0.01;
  EXPECT_GE(both.evaluate(delta, history_context()).measured,
    first.evaluate(delta, history_context()).measured);
}

TEST(TangentSensorHistory, NewDetectionAloneIsNotPenalizedAndUnknownMotionIsSymmetric)
{
  history::History estimator;
  estimator.update(training_input(0));
  const auto context = history_context();
  EXPECT_DOUBLE_EQ(estimator.evaluate(JointVector::Zero(), context).value, 0.0);
  JointVector delta = JointVector::Zero(); delta[0] = 0.01;
  const auto plus = estimator.evaluate(delta, context);
  const auto minus = estimator.evaluate(-delta, context);
  EXPECT_EQ(plus.model_count, 0U);
  EXPECT_EQ(plus.unknown_count, history::kSensorCount);
  EXPECT_GT(plus.unknown_motion, 0.0);
  EXPECT_DOUBLE_EQ(plus.unknown_motion, minus.unknown_motion);
  EXPECT_FALSE(std::isfinite(plus.min_predicted_range));
}

ode::LeafParams experimental_params()
{
  ode::LeafParams params;
  params.enabled = true;
  params.history.initial_angle = 0.6;
  params.history.weight_goal = 0.0;
  params.history.weight_continuity = 0.0;
  return params;
}

TEST(TangentSensorHistoryLeaf, RecedingFloorPreservesAuthorityWithoutBypassingRelease)
{
  auto p=experimental_params();p.direction_mode=history::DirectionMode::kSensorHistory;
  p.history_metric_rate_floor=.5;p.history.handoff_time=0.;
  auto c=history_context();c.tcp_jacobian.leftCols<3>().setIdentity();
  std::vector<double> distances(history::kSensorCount,INFINITY),rates(history::kSensorCount,0.);
  distances[0]=.12;rates[0]=.2;
  const auto run=[&](ode::LeafParams params,const Eigen::Vector3d & goal){
    ode::LeafState state;return ode::evaluate(state,params,c,JointVector::Zero(),
      JointVector::Zero(),goal,distances,rates);
  };
  auto with_floor=run(p,{-1,0,0});p.history_metric_rate_floor=0.;
  auto without=run(p,{-1,0,0});
  ASSERT_GT(with_floor.metric,0.);EXPECT_GT(with_floor.metric,without.metric);
  EXPECT_DOUBLE_EQ(with_floor.metric_gate,.5);
  p.history_metric_rate_floor=.5;
  EXPECT_DOUBLE_EQ(run(p,{1,0,0}).metric,0.); // unblocked goal
  distances[0]=.6;EXPECT_DOUBLE_EQ(run(p,{-1,0,0}).metric,0.); // outside range
}

TEST(FixedTangentLeaf, DirectionIsScoreFreeAndNaturalFormRemainsTangential)
{
  auto params = experimental_params();
  params.weight_goal = params.weight_risk = params.weight_i1 = 1e6;
  params.weight_stall = params.weight_predicted_risk = 1e6;
  params.e1_bias = -1e6;
  const auto context = history_context();
  std::vector<double> ranges(history::kSensorCount, INFINITY); ranges[0] = 0.1;
  ode::LeafState state;
  for (int k = 0; k < 30; ++k) {
    const auto leaf = simple_tangent_leaf::evaluate_fixed(state, params, context,
      JointVector::Zero(), JointVector::Zero(), {0.2, 0.1 + k, 0.5}, ranges, {});
    ASSERT_TRUE(leaf.active);
    EXPECT_NEAR(leaf.theta, 0.6, 1e-12);
    EXPECT_DOUBLE_EQ(leaf.theta_rate, 0.0);
    EXPECT_NEAR(leaf.tangent.dot(leaf.obstacle_direction), 0.0, 1e-12);
    EXPECT_TRUE(leaf.metric_contribution.allFinite());
    EXPECT_TRUE(leaf.force_contribution.allFinite());
    EXPECT_DOUBLE_EQ(leaf.total_direction_score, 0.0);
  }
}

TEST(TangentSensorHistoryLeaf, DecisionTraceCapturesUpdateAndDoesNotChangePolicy)
{
  auto plain = experimental_params();
  plain.direction_mode = history::DirectionMode::kSensorHistory;
  auto logged = plain; logged.capture_history_decision = true;
  ode::LeafState a, b;
  auto context = history_context();
  context.tcp_jacobian.leftCols<3>().setIdentity();
  std::vector<double> ranges(history::kSensorCount, INFINITY); ranges[0] = 0.1;
  const ode::DirectionObjective objective = [](double angle, std::size_t) {
      history::Risk risk; risk.value = 10.0 * (1.0 - std::cos(angle - 0.9));
      risk.gradient = 10.0 * std::sin(angle - 0.9); return risk;
    };
  for (int k = 0; k < 50; ++k) {
    const auto first = ode::evaluate(a, plain, context, JointVector::Zero(),
      JointVector::Zero(), {-0.4,0.0,0.0}, ranges, {}, {}, objective);
    const auto second = ode::evaluate(b, logged, context, JointVector::Zero(),
      JointVector::Zero(), {-0.4,0.0,0.0}, ranges, {}, {}, objective);
    EXPECT_TRUE(first.metric_contribution.isApprox(second.metric_contribution, 0.0));
    EXPECT_TRUE(first.force_contribution.isApprox(second.force_contribution, 0.0));
    EXPECT_DOUBLE_EQ(first.theta, second.theta);
    const auto & d = second.decision;
    EXPECT_DOUBLE_EQ(d.risk_before.gradient, objective(d.theta_evaluated,0).gradient);
    const double expected_rate = d.filtered_rate_before + logged.control_dt /
      (logged.history.direction_filter_tau + logged.control_dt) *
      (d.clamped_rate - d.filtered_rate_before);
    EXPECT_NEAR(second.theta_rate, expected_rate, 1e-14);
    EXPECT_NEAR(std::remainder(d.theta_evaluated + second.theta_rate * logged.control_dt,2*pi), second.theta,1e-14);
    if (k == 0) {
      EXPECT_EQ(d.seeds.size(),8U); EXPECT_TRUE(d.events & 256);
    } else { EXPECT_TRUE(d.seeds.empty()); }
  }
  ranges[0] = INFINITY;
  const auto reset = ode::evaluate(b, logged, context, JointVector::Zero(),
    JointVector::Zero(), {-0.4,0.0,0.0}, ranges, {}, {}, objective);
  EXPECT_TRUE(reset.decision.events & 16);
  EXPECT_TRUE(reset.decision.active_before);
  EXPECT_FALSE(b.active);
}

TEST(TangentSensorHistorySupervisor, RealInitializationTraceEntersRetainedQueue)
{
  EigenRmpConfig config;
  config.tangent_escape.enabled = true;
  config.enable_tangent_escape_diagnostics = true;
  config.record_history_diagnostics = true;
  config.tangent_escape_ode.direction_mode = history::DirectionMode::kSensorHistory;
  TangentEscapeSupervisor supervisor(config);
  auto input = training_input(0);supervisor.set_observations(input);
  auto context=history_context();context.tcp_jacobian.leftCols<3>().setIdentity();
  const auto command=supervisor.evaluate(input.q,context,JointVector::Zero(),
    {-0.4,0.03,0.02},{},JointVector::Zero(),history::Matrix6::Identity(),JointVector::Zero());
  ASSERT_GT(command.debug_data[10],0.0);
  TgInitialEventQueue queue;TgInitialEvent event;
  std::array<double,6> q{},qd{};std::array<double,3> goal{-.4,.03,.02};
  ASSERT_TRUE(queue.push(command.debug_data,123,1000,100,q,qd,goal));
  ASSERT_TRUE(queue.pop(event));
  EXPECT_DOUBLE_EQ(event.data[1],2.0);
  for (int i=0;i<8;++i) EXPECT_DOUBLE_EQ(event.data[77+i],command.debug_data[command.debug_data.size()-505-14-17-484-static_cast<std::size_t>(command.debug_data[command.debug_data.size()-537])+i]);
  const auto payload=tangent_escape_debug::kSize+3;
  EXPECT_DOUBLE_EQ(event.data[6],command.debug_data[payload+10]);
  for (int i=0;i<40;++i) EXPECT_DOUBLE_EQ(event.data[37+i],command.debug_data[payload+67+i]);
}

TEST(TangentSensorHistorySupervisor, WideSearchRecordsFiniteSameStateCandidates)
{
  EigenRmpConfig config;config.tangent_escape.enabled=true;
  config.enable_tangent_escape_diagnostics=true;config.record_history_diagnostics=true;
  config.tangent_escape_ode.direction_mode=history::DirectionMode::kSensorHistory;
  TangentEscapeSupervisor supervisor(config);
  auto context=history_context();context.tcp_jacobian.leftCols<3>().setIdentity();
  bool found=false;double max_us=0;
  for(int k=0;k<300;++k){
    auto input=training_input(0);input.stamp=100+k*.005;
    input.encoder_stamp=input.stamp;input.encoder_q.setZero();input.q.setZero();
    input.samples[0]={input.stamp,.09};supervisor.set_observations(input);
    const auto command=supervisor.evaluate(input.q,context,JointVector::Zero(),
      {-.4,.03,.02},{},JointVector::Zero(),history::Matrix6::Identity(),JointVector::Zero());
    const auto & trace=command.debug_data;
    ASSERT_GT(trace.size(),476U);const auto extra=trace.size()-505-14-17-466-static_cast<std::size_t>(trace[trace.size()-537]);
    if(trace[extra+3]>0){
      found=true;max_us=std::max(max_us,trace[extra+7]);
      for(std::size_t i=extra+11;i<extra+11+72;++i)EXPECT_TRUE(std::isfinite(trace[i]));
      EXPECT_GE(trace[extra+6],0.);EXPECT_LT(trace[extra+6],2*M_PI);
    }
  }
  EXPECT_TRUE(found);std::cout << "Wide search maximum duration (synthetic context): " << max_us << " us\n";
}

TEST(TangentSensorHistorySupervisor, ExtraTraceIsHistoryOnlyAndAppendOnly)
{
  for (auto mode : {history::DirectionMode::kSensorHistory, history::DirectionMode::kSimple}) {
    EigenRmpConfig config;
    config.tangent_escape.enabled = true;
    config.enable_tangent_escape_diagnostics = true;
    config.record_history_diagnostics = true;
    config.tangent_escape_ode.direction_mode = mode;
    TangentEscapeSupervisor supervisor(config);
    auto input = training_input(0); supervisor.set_observations(input);
    const auto command = supervisor.evaluate(input.q, history_context(), JointVector::Zero(),
      {0.4,0.3,0.2}, {}, JointVector::Zero(), history::Matrix6::Identity(), JointVector::Zero());
    const auto offset = tangent_escape_debug::kSize;
    if (mode == history::DirectionMode::kSimple) {
      EXPECT_EQ(command.debug_data.size(),offset);
    } else {
      ASSERT_GT(command.debug_data.size(),offset+3);
      EXPECT_DOUBLE_EQ(command.debug_data[offset],45045.0);
      EXPECT_DOUBLE_EQ(command.debug_data[offset+1],11.0);
      EXPECT_DOUBLE_EQ(command.debug_data[offset+2],command.debug_data.size()-offset-3);
      const auto seeds = static_cast<std::size_t>(command.debug_data[offset+3+66]);
      const auto tail = command.debug_data.size() - 505 - 14 - 17 - 12 - 7 - seeds - 476 - static_cast<std::size_t>(command.debug_data[command.debug_data.size()-537]);
      EXPECT_DOUBLE_EQ(command.debug_data[tail],2.0);  // angle-linear goal-progress reference
      EXPECT_DOUBLE_EQ(command.debug_data[tail+2],config.tangent_escape_ode.nominal_motion_epsilon);
    }
  }
}

TEST(TangentSensorHistoryLeaf, SeedsOnceThenTurnsContinuouslyWithBoundedRate)
{
  auto params = experimental_params();
  params.direction_mode = history::DirectionMode::kSensorHistory;
  params.history.initial_angle = 0.0;
  params.history.weight_risk = 1.0; // explicit legacy-risk opt-in for this objective test
  auto context = history_context();
  context.tcp_jacobian.leftCols<3>().setIdentity();
  std::vector<double> ranges(history::kSensorCount, INFINITY); ranges[0] = 0.1;
  ode::LeafState state;
  double preferred = pi;
  const ode::DirectionObjective objective = [&preferred](double theta, std::size_t) {
      history::Risk risk;
      risk.value = 10.0 * (1.0 - std::cos(theta - preferred));
      risk.gradient = 10.0 * std::sin(theta - preferred);
      return risk;
    };
  auto leaf = ode::evaluate(state, params, context, JointVector::Zero(),
    JointVector::Zero(), {-0.4, 0.0, 0.0}, ranges, {}, {}, objective);
  EXPECT_NEAR(std::abs(leaf.theta), pi, 1e-9);
  preferred = pi / 2.0;
  double previous = leaf.theta;
  for (int k = 0; k < 80; ++k) {
    leaf = ode::evaluate(state, params, context, JointVector::Zero(),
      JointVector::Zero(), {-0.4, 0.0, 0.0}, ranges, {}, {}, objective);
    EXPECT_LE(std::abs(leaf.theta_rate), params.history.max_direction_rate + 1e-12);
    EXPECT_LE(std::abs(std::remainder(leaf.theta - previous, 2.0 * pi)),
      params.history.max_direction_rate * params.control_dt + 1e-12);
    previous = leaf.theta;
  }
  EXPECT_NEAR(std::abs(std::remainder(leaf.theta - preferred, 2.0 * pi)), pi / 2.0, 1e-9); // no encoder: hold
}

TEST(TangentSensorHistoryLeaf, EscapeRewardSeedsButMissingEncoderPreventsSpeculativeTurn)
{
  auto params = experimental_params();
  params.direction_mode = history::DirectionMode::kSensorHistory;
  params.history.initial_angle = 0.0;
  params.history.weight_risk = 0.0;
  params.history.weight_escape = 1.0;
  auto context = history_context(); context.tcp_jacobian.leftCols<3>().setIdentity();
  std::vector<double> ranges(history::kSensorCount, INFINITY); ranges[0] = .1;
  ode::LeafState state;
  double target = pi / 2;
  const ode::DirectionObjective objective = [&](double theta, std::size_t) {
    history::Risk r; r.escape = std::cos(theta-target);
    r.escape_gradient = -std::sin(theta-target); return r;
  };
  auto leaf = ode::evaluate(state, params, context, JointVector::Zero(),
    JointVector::Zero(), {-0.4,0,0}, ranges, {}, {}, objective);
  EXPECT_NEAR(leaf.theta, pi/2, 1e-9);
  // Current angle becomes a stationary minimum; the global check must escape.
  target = -pi/2;
  for (int k=0;k<300;++k) {
    const double before=state.theta;
    leaf=ode::evaluate(state, params, context, JointVector::Zero(),
      JointVector::Zero(), {-0.4,0,0}, ranges, {}, {}, objective);
    EXPECT_LE(std::abs(std::remainder(state.theta-before,2*pi)),
      params.history.max_direction_rate*params.control_dt+1e-12);
  }
  EXPECT_NEAR(std::abs(std::remainder(state.theta-pi/2,2*pi)), 0., 1e-9); // no encoder: no speculative turn
}

struct ResponseHarness {
  TgResponseDirection state; history::Input in;
  RB10Model::Jacobian J=RB10Model::Jacobian::Zero();
  double time=1,x=0,theta=0,range=.1,metric=100,wanted=M_PI/2;
  int calls=0;
  bool missing=false,flat=false;
  TgAuthorityBoostParams boost;
  double command_speed=.03;bool command_available=true;
  Eigen::Vector3d goal_axis{1,0,0};
  ResponseHarness(){J.leftCols<3>().setIdentity();}
  double tick(double dx=0,double block=1){
    for(int k=0;k<26;++k)state.authority.advance(.005,boost);
    time+=.13;x+=dx;in.stamp=in.encoder_stamp=time;in.encoder_q[0]=x;
    in.samples[0]={time,range};
    in.command_stamp=time;in.command_available=command_available;in.command_qd[0]=command_speed;
    return state.update(theta,0,0,{1,0,0},{1,0,0},J,in,block,
      [&](double a){++calls;return missing?NAN:(flat?0.:std::cos(a-wanted));},metric,J,goal_axis,.1,.24,1.5,boost);
  }
  void start(){for(int i=0;i<8 && state.phase!=1;++i)tick();ASSERT_EQ(state.phase,1);}
  void windows(double dx){tick(dx);tick(dx);tick(dx);}
};
TEST(TgResponseDirection, InitialExecutionGetsAnObservationIntervalBeforeFailure)
{
  ResponseHarness h;h.tick();h.tick();h.tick();
  EXPECT_FALSE(h.state.turning);EXPECT_DOUBLE_EQ(h.state.poor_evidence,0.);
  for(double cost:h.state.sweep_failure)EXPECT_DOUBLE_EQ(cost,0.);
}
TEST(TgResponseDirection, UnexecutedHeadingDoesNotAcquireFailureCostButSearchCanResume)
{
  ResponseHarness h;h.command_speed=-.03;
  for(int i=0;i<15;++i){h.theta=h.state.target;h.tick();}
  EXPECT_FALSE(h.state.execution_supported);
  EXPECT_EQ(h.state.reason,TgResponseDirection::UnexecutedSearch);
  EXPECT_TRUE(h.state.turning);
  EXPECT_GT(std::abs(h.state.target),.1);
  EXPECT_DOUBLE_EQ(h.state.poor_evidence,0.);
  for(double cost:h.state.sweep_failure)EXPECT_DOUBLE_EQ(cost,0.);
}
TEST(TgResponseDirection, AlternatingMotionIsNotSustainedEscape)
{
  ResponseHarness h;h.start();
  for(int i=0;i<20;++i)h.tick(i%2?.003:-.003);
  EXPECT_FALSE(h.state.has_accepted);
  EXPECT_TRUE(h.state.turning);
  EXPECT_LT(h.state.progress_ratio,.5);
}
TEST(TgResponseDirection, NoCommandFeedbackCannotPenalizeDirection)
{
  ResponseHarness h;h.command_available=false;
  for(int i=0;i<12;++i)h.tick();
  EXPECT_FALSE(h.state.execution_supported);
  EXPECT_DOUBLE_EQ(h.state.poor_evidence,0.);
  for(double cost:h.state.sweep_failure)EXPECT_DOUBLE_EQ(cost,0.);
}
TEST(TgResponseDirection, OldFailuresExpireAndNewRemoteDetectionIsNotDanger)
{
  ResponseHarness h;h.start();h.command_speed=-.03;
  const double cost=h.state.sweep_failure[0];
  for(int i=0;i<12;++i){h.in.samples[1]={h.time+.13,.15};h.tick();}
  EXPECT_LT(h.state.sweep_failure[0],cost*.1);
  EXPECT_EQ(h.state.adverse_evidence,0.);
}
TEST(TgResponseDirection, FullCirclePriorChoosesSignButNotAnUnobservedDestination)
{
  ResponseHarness h;h.wanted=220*M_PI/180;h.start();
  EXPECT_LT(h.state.target,0);EXPECT_GT(h.state.target,-M_PI/9);
  EXPECT_EQ(h.state.sweep_scores.size(),72U);EXPECT_EQ(h.calls,72*static_cast<int>(h.state.sweep_sequence));
  EXPECT_EQ(h.state.sign,-1);
}
TEST(TgResponseDirection, SustainedMeasuredMotionRetainsWorkingAngleInsteadOfChasingPrior)
{
  ResponseHarness h;h.start();
  h.theta=.3;h.windows(.003);h.windows(.003);
  EXPECT_EQ(h.state.phase,0);EXPECT_TRUE(h.state.has_accepted);
  EXPECT_NEAR(h.state.accepted,.3,1e-9);EXPECT_NEAR(h.state.target,.3,1e-9);
  h.wanted=-M_PI/2;
  for(int i=0;i<16;++i)h.tick(.003);
  EXPECT_NEAR(h.state.target,.3,1e-9);EXPECT_FALSE(h.state.turning);
}
TEST(TgResponseDirection, MetricFluctuationLowersConfidenceWithoutStoppingTurn)
{
  ResponseHarness h;h.start();
  h.theta=.2;h.metric=1;h.tick(-.003);
  EXPECT_LT(h.state.observation_confidence,.5);EXPECT_EQ(h.state.phase,1);
  EXPECT_GT(h.state.target,h.theta);EXPECT_EQ(h.state.sign,1);
}
TEST(TgResponseDirection, ReversalNeedsPersistentAlternativeAndMeasuredPoorProgress)
{
  ResponseHarness h;h.start();h.theta=.6;
  h.wanted=-M_PI/2;
  h.tick(-.003);EXPECT_EQ(h.state.sign,1);
  for(int i=0;i<16;++i)h.tick(-.003);
  EXPECT_LT(h.state.target,h.theta);EXPECT_EQ(h.state.phase,1);
  EXPECT_EQ(h.state.reverse_count,1);
}
TEST(TgResponseDirection, OneBadWindowDoesNotReverseDirection)
{
  ResponseHarness h;h.start();
  h.wanted=-M_PI/2;h.tick(-.003);h.wanted=M_PI/2;h.windows(.003);
  EXPECT_EQ(h.state.sign,1);EXPECT_EQ(h.state.reverse_count,0);
}
TEST(TgResponseDirection, NewlyRecoveredMotionPreventsPendingOppositeSideReversal)
{
  ResponseHarness h;h.start();h.theta=.6;h.wanted=-M_PI/2;
  for(int i=0;i<30 && h.state.pending_count==0;++i)h.tick();
  ASSERT_EQ(h.state.pending_count,1);ASSERT_EQ(h.state.sign,1);
  h.windows(.003);
  EXPECT_EQ(h.state.reverse_count,0);EXPECT_FALSE(h.state.turning);
}
TEST(TgResponseDirection, RangeDropAndNoiseDoNotFreezeCandidateEvaluationOrSteering)
{
  ResponseHarness h;h.flat=true;h.start();h.theta=.3;h.range=.065;h.tick();
  EXPECT_EQ(h.state.phase,4);EXPECT_GT(h.state.target,h.theta);
  const auto scans=h.state.sweep_sequence;
  // Regression for the 31--39s failure: neither stable-close values nor >2mm
  // fluctuations may leave direction search waiting for three quiet windows.
  for(int i=0;i<24;++i){
    h.theta=h.state.target;h.range=i<6?.065:(i%2?.111:.116);h.tick();
    EXPECT_NE(h.state.phase,5);
  }
  EXPECT_GT(h.state.sweep_sequence,scans+8);EXPECT_GT(h.theta,1.);
}
TEST(TgResponseDirection, StaleFeedbackAndUnavailableCandidatesNeverInventMotion)
{
  ResponseHarness h;h.start();h.in.stamp+=1;
  h.state.update(h.theta,0,0,{1,0,0},{1,0,0},h.J,h.in,1,[](double){return 0;});
  EXPECT_FALSE(h.state.seeded);
  ResponseHarness missing;missing.missing=true;
  for(int i=0;i<8;++i)missing.tick();
  EXPECT_DOUBLE_EQ(missing.state.target,0);EXPECT_EQ(missing.state.result,-5);
  EXPECT_EQ(missing.state.phase,5);
}
TEST(TgResponseDirection, ContinuousTrackingStillUsesBoundedSearchFrequency)
{
  ResponseHarness h;h.start();const int before=h.calls;
  h.tick();h.tick();EXPECT_EQ(h.calls,before+72);
}
TEST(TgResponseDirection, StalledFlatPriorAccumulatesPastNinetyAndRemembersOnlyTriedAngles)
{
  ResponseHarness h;h.flat=true;h.start();
  EXPECT_GT(h.state.sweep_failure[0],0);
  EXPECT_EQ(h.state.sweep_failure[36],0); // untried is unknown, not failed
  for(int i=0;i<20;++i){h.theta=h.state.target;h.tick();}
  EXPECT_GT(h.theta,M_PI/2);EXPECT_EQ(h.state.sign,1);
  EXPECT_LE(h.state.target-h.theta,M_PI/9+1e-9);
}
TEST(TgResponseDirection, OneGoodWindowDoesNotAbandonCumulativeTurn)
{
  ResponseHarness h;h.start();h.theta=h.state.target;
  h.tick(.003);EXPECT_TRUE(h.state.turning);
  h.tick();EXPECT_GT(h.state.target,h.theta);
}
TEST(TgResponseDirection, TemporaryGoalRetreatIsAllowedButPersistentUnimprovedRetreatIsNot)
{
  ResponseHarness h;h.goal_axis={-1,0,0};h.start();h.windows(.003);
  EXPECT_FALSE(h.state.turning); // useful tangential detour, despite goal retreat
  for(int i=0;i<16;++i)h.tick(.003);
  EXPECT_GE(h.state.retreat_evidence,.6);EXPECT_TRUE(h.state.turning);
}
TEST(TgResponseDirection, RepeatedSensorPacketCannotCountAsSuccessfulRecovery)
{
  ResponseHarness h;h.start();const double stamp=h.in.samples[0].stamp;
  h.in.encoder_stamp+=.13;h.in.stamp=h.in.encoder_stamp;h.in.samples[0].stamp=stamp;
  h.state.update(h.theta,0,0,{1,0,0},{1,0,0},h.J,h.in,1,[](double){return 1.;});
  EXPECT_EQ(h.state.phase,5);EXPECT_EQ(h.state.paired_samples,0);
  EXPECT_DOUBLE_EQ(h.state.target,h.theta);
}
TEST(TgResponseDirection, BlockInactiveDoesNotAccumulateFailureEvidence)
{
  ResponseHarness h;h.start();h.tick(0,0);
  EXPECT_FALSE(h.state.turning);EXPECT_EQ(h.state.poor_evidence,0);
  EXPECT_EQ(h.state.reason,TgResponseDirection::BlockInactive);
}

TEST(TangentSensorHistoryLeaf, WorseningRangeStillPassesSteeringThroughTheRateLimitedOde)
{
  auto params=experimental_params();
  params.direction_mode=history::DirectionMode::kSensorHistory;
  params.response_mobility=[](double,std::size_t){return TgDirectionCandidate(0.);};
  auto context=history_context();context.tcp_jacobian.leftCols<3>().setIdentity();
  std::vector<double> ranges(history::kSensorCount,INFINITY);ranges[0]=.119;
  ode::LeafState state;
  bool risk_turn=false;double movement=0;
  for(int k=0;k<400;++k){
    const double stamp=100+k*params.control_dt;
    ranges[0]=k<100?.119:.065;
    params.response_input.stamp=params.response_input.encoder_stamp=stamp;
    params.response_input.samples[0]={stamp,ranges[0]};
    const double before=state.theta;
    auto leaf=ode::evaluate(state,params,context,JointVector::Zero(),
      JointVector::Zero(),{-.4,0,0},ranges,{});
    ASSERT_TRUE(leaf.active);
    EXPECT_LE(std::abs(leaf.theta_rate),params.history.max_direction_rate+1e-9);
    if(state.response.phase==4 && std::abs(leaf.theta_rate)>1e-4)risk_turn=true;
    if(k>1)movement+=std::abs(std::remainder(state.theta-before,2*M_PI));
  }
  EXPECT_TRUE(risk_turn);EXPECT_GT(movement,.2);
}

TEST(TangentSensorHistoryLeaf, OwnerSwitchHasNoZeroAuthorityFade)
{
  auto params=experimental_params();params.block_on_dot=-2.;params.block_off_dot=-1.;
  const auto context=history_context();
  std::vector<double> ranges(history::kSensorCount,INFINITY);ranges[0]=.1;ranges[1]=.2;
  ode::LeafState state;
  const auto run=[&](){return simple_tangent_leaf::evaluate_fixed(state,params,context,
    JointVector::Zero(),JointVector::Zero(),{.4,.3,.2},ranges,{});};
  ASSERT_EQ(run().sensor_index,0U);
  ranges[1]=.098;EXPECT_EQ(run().sensor_index,0U); // hysteresis retained
  ranges[1]=.05;const auto changed=run();
  EXPECT_EQ(changed.sensor_index,1U);EXPECT_DOUBLE_EQ(changed.handoff_gain,1.);
  EXPECT_GT(changed.metric,0.);EXPECT_GT(changed.force_contribution.norm(),0.);
}

TEST(TangentSensorHistoryLeaf, CloserNonblockingSensorCannotStealOwnership)
{
  auto p=experimental_params();p.direction_mode=history::DirectionMode::kSensorHistory;
  auto c=history_context();c.tcp_jacobian.leftCols<3>().setIdentity();
  std::vector<double> ranges(history::kSensorCount,INFINITY);ranges[0]=.12;ranges[2]=.02;
  ode::LeafState state;
  const auto run=[&](const Eigen::Vector3d & goal){return ode::evaluate(state,p,c,
    JointVector::Zero(),JointVector::Zero(),goal,ranges,{});};
  auto a=run({-1,0,0});ASSERT_TRUE(a.active);EXPECT_EQ(a.sensor_index,0U);EXPECT_GT(a.metric,0.);
  EXPECT_DOUBLE_EQ(a.alpha_free,1.);EXPECT_DOUBLE_EQ(a.handoff_gain,1.);
  EXPECT_NEAR(a.alpha_total,a.alpha_distance*a.alpha_block,1e-12);
  a=run({1,0,0});EXPECT_EQ(a.sensor_index,2U);EXPECT_GT(a.metric,0.);
  ranges[2]=INFINITY;a=run({1,0,0});EXPECT_DOUBLE_EQ(a.metric,0.);
}

TEST(TangentSensorHistory, NeighborCostIsDirectionalAndExcludesOwner)
{
  history::History estimator;
  for(int k=0;k<=60;++k)estimator.update(training_input(k,true));
  JointVector delta=JointVector::Zero();delta[0]=.02;
  // Sensor 0 closes on +q0; sensor 1 closes on -q0. Owner 0 is excluded.
  const auto toward=estimator.evaluate_neighbors(-delta,0);
  const auto away=estimator.evaluate_neighbors(delta,0);
  EXPECT_GT(toward.value,0.);EXPECT_DOUBLE_EQ(away.value,0.);
  EXPECT_EQ(toward.critical_sensor,1);EXPECT_GE(toward.confidence,.7);
  EXPECT_LE(toward.value,1.);
  // Same pair with reversed owner reverses which movement is penalized.
  EXPECT_GT(estimator.evaluate_neighbors(delta,1).value,0.);
  EXPECT_DOUBLE_EQ(estimator.evaluate_neighbors(-delta,1).value,0.);
}

TEST(TangentSensorHistory, UnknownOrLostNeighborCannotInventDirectionalRisk)
{
  history::History estimator;estimator.update(training_input(0,true));
  JointVector delta=JointVector::Zero();delta[0]=.02;
  EXPECT_DOUBLE_EQ(estimator.evaluate_neighbors(-delta,0).value,0.);
  for(int k=1;k<=60;++k)estimator.update(training_input(k,true));
  JointVector unsupported=JointVector::Zero();unsupported[5]=.02;
  auto r=estimator.evaluate_neighbors(unsupported,0);EXPECT_DOUBLE_EQ(r.value,0.);
  EXPECT_GT(r.unknown_count,0U);
  auto input=training_input(61,true);input.samples[1].range=INFINITY;estimator.update(input);
  EXPECT_DOUBLE_EQ(estimator.evaluate_neighbors(-delta,0).value,0.);
}

TEST(TangentSensorHistory, NeighborRiskRejectsContradictionAndOutsideLocalTrustRegion)
{
  history::History estimator;
  for(int k=0;k<=60;++k)estimator.update(training_input(k,true));
  JointVector delta=JointVector::Zero();delta[0]=-.02;
  ASSERT_GT(estimator.evaluate_neighbors(delta,0).value,0.);
  JointVector far=delta*5.;EXPECT_DOUBLE_EQ(estimator.evaluate_neighbors(far,0).value,0.);
  auto input=training_input(61,true);input.samples[1].range+=.025;estimator.update(input);
  EXPECT_DOUBLE_EQ(estimator.evaluate_neighbors(delta,0).value,0.);
}

TEST(TangentSensorHistorySupervisor, UsesScalarInputInsteadOfMovingVirtualSphereGeometry)
{
  EigenRmpConfig config;
  config.tangent_escape.enabled = true;
  config.enable_tangent_escape_diagnostics = true;
  config.tangent_escape_ode.direction_mode = history::DirectionMode::kSimple;
  config.tangent_escape_ode.alpha_off_distance = 0.45;
  TangentEscapeSupervisor supervisor(config);
  const auto context = history_context();
  auto input = training_input(0);
  supervisor.set_observations(input);
  const Eigen::Matrix<double, 6, 6> metric = Eigen::Matrix<double, 6, 6>::Identity();
  for (int k = 0; k < 5; ++k) {
    input.stamp += 0.025;
    supervisor.set_observations(input); // acquisition stamp deliberately unchanged
    ObstacleSphere sphere{{100.0 + k, 100.0, 100.0}, 0.01, 0, 0};
    const auto command = supervisor.evaluate(input.q, context, JointVector::Zero(),
      {0.4, 0.3, 0.2}, {sphere}, JointVector::Zero(), metric, JointVector::Zero());
    ASSERT_TRUE(command.active);
    EXPECT_NEAR(command.debug_data[2], 0.09, 1e-12);
    EXPECT_DOUBLE_EQ(command.debug_data[tangent_escape_debug::kAccel], 0.0);
    EXPECT_LE(command.debug_data[tangent_escape_debug::kHistorySensors], 1.0);
    EXPECT_DOUBLE_EQ(command.debug_data[tangent_escape_debug::kHistory], 3.0);
  }
}

TEST(TangentSensorHistorySupervisor, InactiveLogPreservesGoalErrorAndUnknownRanges)
{
  EigenRmpConfig config;
  config.tangent_escape.enabled = true;
  config.enable_tangent_escape_diagnostics = true;
  config.tangent_escape_ode.direction_mode = history::DirectionMode::kSensorHistory;
  TangentEscapeSupervisor supervisor(config);
  auto context = history_context();
  context.tcp_position = {0.1, 0.2, 0.3};
  history::Input input; input.stamp = 1.0;
  supervisor.set_observations(input);
  const Eigen::Vector3d goal{0.4, 0.2, 0.3};
  const auto command = supervisor.evaluate(input.q, context, JointVector::Zero(),
    goal, {}, JointVector::Zero(), history::Matrix6::Identity(), JointVector::Zero());
  EXPECT_FALSE(command.active);
  ASSERT_EQ(command.debug_data.size(), tangent_escape_debug::kSize);
  EXPECT_NEAR(command.debug_data[tangent_escape_debug::kProgress + 3], 0.3, 1e-12);
  EXPECT_DOUBLE_EQ(command.debug_data[tangent_escape_debug::kHistory], 1.0);
  for (std::size_t i = 0; i < history::kSensorCount; ++i) {
    EXPECT_TRUE(std::isinf(command.debug_data[56 + 4 * i]));
  }
}

KinematicsContext moving_context(const Eigen::Vector3d & velocity)
{
  auto context = history_context();
  for (auto & value : context.control_point_velocities) { value = velocity; }
  return context;
}

TEST(SimpleVelocitySteering, ChoosesVelocityNormalForceWithoutScoreSearch)
{
  auto params = experimental_params();
  params.weight_goal = params.weight_risk = params.weight_i1 = 1e6;
  params.weight_stall = params.weight_predicted_risk = 1e6;
  const auto & sensor = RB10Model::sensor_control_points[0];
  const Eigen::Vector3d n = sensor.local_normal.normalized();
  const Eigen::Vector3d velocity = 0.1 * (n + sensor.local_tangent_bias).normalized();
  const auto context = moving_context(velocity);
  JointVector qd = JointVector::Zero(); qd.head<3>() = velocity;
  std::vector<double> ranges(history::kSensorCount, INFINITY); ranges[0] = 0.1;
  ode::LeafState state;
  const auto leaf = simple_tangent_leaf::evaluate(state, params, context,
    qd, JointVector::Zero(), {0.3, 0.5, 0.8}, ranges, {});
  ASSERT_TRUE(leaf.active);
  const Eigen::Vector3d u = velocity.normalized();
  const Eigen::Vector3d expected = (-n + u * n.dot(u)).normalized();
  EXPECT_NEAR((leaf.tangent - expected).norm(), 0.0, 1e-12);
  EXPECT_NEAR(leaf.tangent.dot(velocity), 0.0, 1e-12);
  EXPECT_LT(leaf.tangent.dot(n), -0.1);  // not restricted to sensor-normal tangent plane
  EXPECT_NEAR(leaf.force_contribution.dot(qd), 0.0, 1e-12);
  EXPECT_GT(leaf.force_contribution.head<3>().dot(leaf.tangent), 0.0);
  EXPECT_DOUBLE_EQ(leaf.total_direction_score, 0.0);
  EXPECT_NEAR(leaf.injected_power, 0.0, 1e-10);
  EXPECT_TRUE(leaf.metric_contribution.isApprox(
    leaf.metric_contribution.transpose(), 1e-12));

  // Changing target and old score weights cannot select another steering side.
  ode::LeafState other_state;
  params.weight_goal = params.weight_risk = params.weight_i1 = 0.0;
  const auto other = simple_tangent_leaf::evaluate(other_state, params, context,
    qd, JointVector::Zero(), {-3.0, -2.0, -1.0}, ranges, {});
  EXPECT_NEAR((other.tangent - leaf.tangent).norm(), 0.0, 1e-12);
}

TEST(SimpleVelocitySteering, UsesMeasuredVelocityEvenWhenSolverVelocityIsDifferent)
{
  auto params = experimental_params();
  const Eigen::Vector3d n = RB10Model::sensor_control_points[0].local_normal.normalized();
  const Eigen::Vector3d side = RB10Model::sensor_control_points[0].local_tangent_bias.normalized();
  const Eigen::Vector3d velocity = 0.1 * (n + side).normalized();
  const auto context = moving_context(0.1 * n);
  std::vector<Eigen::Vector3d> measured(history::kSensorCount, velocity);
  std::vector<double> ranges(history::kSensorCount, INFINITY); ranges[0] = 0.1;
  ode::LeafState state;
  const auto leaf = simple_tangent_leaf::evaluate(state, params, context,
    JointVector::Zero(), JointVector::Zero(), {0.3, 0.5, 0.8}, ranges, {}, measured);
  ASSERT_TRUE(leaf.active);
  EXPECT_EQ(leaf.steering_velocity_source, 1);
  EXPECT_NEAR(leaf.tangent.dot(velocity), 0.0, 1e-12);
  EXPECT_GT(std::abs(leaf.tangent.dot(context.control_point_velocities[0])), 0.01);
}

TEST(SimpleVelocitySteering, AtRestAndHeadOnCasesHaveFiniteConsistentDirections)
{
  auto params = experimental_params();
  const Eigen::Vector3d n = RB10Model::sensor_control_points[0].local_normal.normalized();
  const Eigen::Vector3d side = RB10Model::sensor_control_points[0].local_tangent_bias.normalized();
  std::vector<double> ranges(history::kSensorCount, INFINITY); ranges[0] = 0.1;
  ode::LeafState state;
  JointVector nominal = JointVector::Zero(); nominal.head<3>() = n;
  auto leaf = simple_tangent_leaf::evaluate(state, params, moving_context(Eigen::Vector3d::Zero()),
    JointVector::Zero(), nominal, {0.3, 0.5, 0.8}, ranges, {});
  ASSERT_TRUE(leaf.active);
  EXPECT_EQ(leaf.steering_fallback & 1, 1);
  EXPECT_EQ(leaf.steering_fallback & 4, 4);
  EXPECT_NEAR(leaf.tangent.dot(n), 0.0, 1e-12);
  EXPECT_GT(leaf.force_contribution.norm(), 0.0);  // startup need not wait for a stall
  Eigen::Vector3d previous = leaf.tangent;
  for (int k = 0; k < 30; ++k) {
    const Eigen::Vector3d velocity = 0.1 * n + (k % 2 ? 1e-7 : -1e-7) * side;
    JointVector qd = JointVector::Zero(); qd.head<3>() = velocity;
    leaf = simple_tangent_leaf::evaluate(state, params, moving_context(velocity),
      qd, nominal, {0.3, 0.5, 0.8}, ranges, {});
    EXPECT_TRUE(leaf.force_contribution.allFinite());
    EXPECT_NEAR(leaf.tangent.dot(velocity), 0.0, 1e-12);
    EXPECT_GT(leaf.tangent.dot(previous), 0.999);
    previous = leaf.tangent;
  }
  ode::LeafState cold;
  leaf = simple_tangent_leaf::evaluate(cold, params, moving_context(Eigen::Vector3d::Zero()),
    JointVector::Zero(), JointVector::Zero(), {0.3, 0.5, 0.8}, ranges, {});
  EXPECT_TRUE(leaf.active);
  EXPECT_EQ(leaf.steering_fallback & 2, 2);
  EXPECT_TRUE(leaf.force_contribution.allFinite());
}

TEST(SimpleVelocitySteering, RateLimitAndCartesianCurvatureRemainConsistent)
{
  auto params = experimental_params();
  params.history.max_direction_rate = 0.8;
  params.block_on_dot = -2.0;
  params.block_off_dot = -1.0;
  const Eigen::Vector3d n = RB10Model::sensor_control_points[0].local_normal.normalized();
  const Eigen::Vector3d side = RB10Model::sensor_control_points[0].local_tangent_bias.normalized();
  const Eigen::Vector3d velocity = 0.1 * (n + side).normalized();
  const auto context = moving_context(velocity);
  std::vector<double> ranges(history::kSensorCount, INFINITY); ranges[0] = 0.1;
  JointVector qd = JointVector::Zero(); qd.head<3>() = velocity;
  ode::LeafState state;
  auto first = simple_tangent_leaf::evaluate(state, params, context,
    qd, JointVector::Zero(), {0.3, 0.5, 0.8}, ranges, {});
  Eigen::Vector3d previous = first.tangent;
  auto changed = context;
  for (auto & rotation : changed.link_rotations) {
    rotation = Eigen::AngleAxisd(1.0, velocity.normalized()).toRotationMatrix();
  }
  changed.control_point_curvatures[0] = {0.1, -0.2, 0.3};
  for (int k = 0; k < 30; ++k) {
    const auto leaf = simple_tangent_leaf::evaluate(state, params, changed,
      qd, JointVector::Zero(), {0.3, 0.5, 0.8}, ranges, {});
    ASSERT_TRUE(leaf.active);
    const double angle = std::acos(std::clamp(previous.dot(leaf.tangent), -1.0, 1.0));
    EXPECT_LE(angle, params.history.max_direction_rate * params.control_dt + 1e-9);
    EXPECT_NEAR(leaf.tangent.dot(velocity), 0.0, 1e-12);
    const Eigen::Matrix3d cartesian_metric = leaf.metric * leaf.tangent * leaf.tangent.transpose();
    const auto & j = changed.control_point_jacobians[0];
    const JointVector expected = j.transpose() *
      (leaf.natural_force * leaf.tangent - cartesian_metric * changed.control_point_curvatures[0]);
    EXPECT_NEAR((leaf.force_contribution - expected).norm(), 0.0, 1e-9);
    EXPECT_DOUBLE_EQ(leaf.jdot_theta_term, 0.0);
    previous = leaf.tangent;
  }
}

TEST(SimpleVelocitySteering, StartupReferenceJoinsTheVelocityPlaneContinuously)
{
  auto params = experimental_params();
  params.velocity_epsilon = 0.005;
  params.block_on_dot = -2.0; params.block_off_dot = -1.0;
  const Eigen::Vector3d n = RB10Model::sensor_control_points[0].local_normal.normalized();
  const Eigen::Vector3d side = RB10Model::sensor_control_points[0].local_tangent_bias.normalized();
  const Eigen::Vector3d heading = (n + 0.8 * side).normalized();
  std::vector<double> ranges(history::kSensorCount, INFINITY); ranges[0] = 0.1;
  JointVector nominal = JointVector::Zero(); nominal.head<3>() = n;
  ode::LeafState state;
  Eigen::Vector3d previous = Eigen::Vector3d::Zero();
  for (int k = 0; k <= 200; ++k) {
    const Eigen::Vector3d velocity = params.velocity_epsilon * k / 100.0 * heading;
    JointVector qd = JointVector::Zero(); qd.head<3>() = velocity;
    const auto leaf = simple_tangent_leaf::evaluate(state, params, moving_context(velocity),
      qd, nominal, {0.3, 0.5, 0.8}, ranges, {});
    ASSERT_TRUE(leaf.active);
    EXPECT_TRUE(leaf.force_contribution.allFinite());
    if (k > 0) { EXPECT_GT(previous.dot(leaf.tangent), 0.999); }
    if (k >= 100) { EXPECT_NEAR(leaf.tangent.dot(velocity), 0.0, 1e-12); }
    previous = leaf.tangent;
  }
}
}
}  // namespace rb10_rmpflow_rviz

TEST(TgInitialEventQueue, PreservesInitializationAndDetectsBoundedQueueOverflow)
{
  rb10_rmpflow_rviz::TgInitialEventQueue queue;
  rb10_rmpflow_rviz::TgInitialEvent out;
  std::vector<double> tg(600,0.0);
  tg[26]=9;tg[50]=20;tg[274]=45045;tg[275]=2;
  const std::size_t payload=277;
  tg[payload+2]=256;tg[payload+66]=8;tg[payload+10]=0.7;
  tg[10]=100;tg[46]=0.4;tg[11]=.11;tg[160]=1;tg[164]=1;
  tg[payload+67]=0.7;tg[payload+68]=.25;
  std::array<double,6> q{.42,0,0,0,0,0},qd{};
  std::array<double,3> goal{.8,.2,.5};
  EXPECT_FALSE(queue.pop(out));
  for (std::size_t i=0;i<queue.capacity;++i) {
    ASSERT_TRUE(queue.push(tg,i,1000+i,100+i,q,qd,goal));
  }
  EXPECT_FALSE(queue.push(tg,100,1100,200,q,qd,goal));
  for (std::size_t i=0;i<queue.capacity;++i) {
    ASSERT_TRUE(queue.pop(out));EXPECT_EQ(out.data[2],i);
    EXPECT_DOUBLE_EQ(out.data[6],.7);EXPECT_DOUBLE_EQ(out.data[20],.42);
    EXPECT_DOUBLE_EQ(out.data[36],1);EXPECT_DOUBLE_EQ(out.data[38],.25);
  }
  EXPECT_FALSE(queue.pop(out));
  tg[44]=1;EXPECT_FALSE(queue.push(tg,101,1101,201,q,qd,goal));tg[44]=0;
  tg[payload+2]=0;
  EXPECT_FALSE(queue.push(tg,101,1101,201,q,qd,goal));
  tg.resize(10);EXPECT_FALSE(queue.push(tg,102,1102,202,q,qd,goal));
}

TEST(TgInitialEventQueue, VersionSevenSuffixPreservesInitialExecutionScores)
{
  rb10_rmpflow_rviz::TgInitialEventQueue queue;
  rb10_rmpflow_rviz::TgInitialEvent out;
  constexpr std::size_t payload=277,suffix=476;
  std::vector<double> tg(payload+107+9+suffix,0.);
  tg[26]=9;tg[50]=20;tg[274]=45045;tg[275]=7;tg[10]=100;
  tg[payload+2]=256;tg[payload+66]=8;
  tg[tg.size()-suffix-9]=8;
  for(int i=0;i<8;++i)tg[tg.size()-suffix-8+i]=.1*i;
  ASSERT_TRUE(queue.push(tg,1,100,10,{},{},{}));ASSERT_TRUE(queue.pop(out));
  EXPECT_EQ(out.data[1],2);
  for(int i=0;i<8;++i)EXPECT_DOUBLE_EQ(out.data[77+i],.1*i);
}

TEST(TgInitialEventQueue, VersionNinePreservesSeedExecutionScores)
{
  rb10_rmpflow_rviz::TgInitialEventQueue queue;
  rb10_rmpflow_rviz::TgInitialEvent out;
  constexpr std::size_t payload=277,shadow=509+8*20,suffix=476+shadow+17;
  std::vector<double> tg(payload+107+9+suffix,0.);
  tg[26]=9;tg[50]=20;tg[274]=45045;tg[275]=9;tg[10]=100;
  tg[payload+2]=256;tg[payload+66]=8;tg[tg.size()-18]=shadow;
  tg[tg.size()-suffix-9]=8;
  for(int i=0;i<8;++i)tg[tg.size()-suffix-8+i]=.1*i;
  ASSERT_TRUE(queue.push(tg,1,100,10,{},{},{}));ASSERT_TRUE(queue.pop(out));
  EXPECT_EQ(out.data[1],2);
  for(int i=0;i<8;++i)EXPECT_DOUBLE_EQ(out.data[77+i],.1*i);
}

namespace rb10_rmpflow_rviz {
namespace {
TEST(TgAuthorityBoost, RotationStartsBoostWithoutDelayingSearch)
{
  ResponseHarness baseline,h;h.boost.enabled=true;h.boost.maximum=1.2;
  baseline.command_speed=h.command_speed=0.;
  bool started=false;
  for(int i=0;i<16;++i){
    const bool was_turning=h.state.turning;
    baseline.tick();h.tick();
    EXPECT_EQ(h.state.turning,baseline.state.turning);
    if(!was_turning){EXPECT_DOUBLE_EQ(h.state.authority.gain,1.);}
    if(h.state.turning){started=true;EXPECT_EQ(h.state.authority.phase,TgAuthorityBoost::Rising);}
  }
  EXPECT_TRUE(started);EXPECT_NEAR(h.state.authority.gain,1.2,1e-12);
  for(double cost:h.state.sweep_failure)EXPECT_DOUBLE_EQ(cost,0.);
}
TEST(TgAuthorityBoost, AdequateCommandsStillBoostWhenStallStartsRotation)
{
  ResponseHarness h;h.boost.enabled=true;
  for(int i=0;i<12;++i)h.tick();
  EXPECT_TRUE(h.state.turning);EXPECT_GT(h.state.authority.gain,1.);
  h.state.pause();EXPECT_DOUBLE_EQ(h.state.authority.gain,1.);
}
TEST(TgAuthorityBoost, CommandDeficitAloneNeverBoosts)
{
  TgAuthorityBoost b;TgAuthorityBoostParams p;p.enabled=true;
  for(int i=0;i<100;++i){
    b.observe(.13,p,true,false,false,false,false,1.);b.advance(.005,p);
  }
  EXPECT_DOUBLE_EQ(b.gain,1.);EXPECT_EQ(b.phase,TgAuthorityBoost::Idle);
}
TEST(TgAuthorityBoost, DangerDecaysEvenWhileTurning)
{
  TgAuthorityBoost b;TgAuthorityBoostParams p;p.enabled=true;b.gain=1.5;
  b.observe(.13,p,true,true,false,true,true,1.);b.advance(.005,p);
  EXPECT_EQ(b.phase,TgAuthorityBoost::Danger);EXPECT_NEAR(b.gain,1.495,1e-12);
}
TEST(TgAuthorityBoost, GainRampsAndRecoversAtBoundedControlRate)
{
  TgAuthorityBoost b;TgAuthorityBoostParams p;p.enabled=true;
  b.observe(.13,p,true,true,false,false,true,1.);
  ASSERT_EQ(b.phase,TgAuthorityBoost::Rising);
  b.advance(.005,p);EXPECT_NEAR(b.gain,1.005,1e-12);
  b.observe(.13,p,true,true,true,false,true,1.);
  b.advance(.005,p);EXPECT_NEAR(b.gain,1.,1e-12);
  b.gain=2.;b.observe(.13,p,false,false,false,false,false,1.);
  EXPECT_DOUBLE_EQ(b.gain,1.);
}
TEST(TgAuthorityBoost, NaturalFormScalesBothMetricAndForceButNotDesiredAcceleration)
{
  auto p=experimental_params();p.direction_mode=history::DirectionMode::kSensorHistory;
  p.authority_boost.enabled=true;
  auto c=history_context();c.tcp_jacobian.leftCols<3>().setIdentity();
  std::vector<double> d(history::kSensorCount,INFINITY),rates(history::kSensorCount,0.);d[0]=.12;
  auto run=[&](double gain,const Eigen::Vector3d & goal){
    ode::LeafState state;state.active=true;state.sensor_index=0;state.theta_valid=true;
    state.theta=.6;state.response.authority.gain=gain;
    return simple_tangent_leaf::evaluate_fixed(state,p,c,JointVector::Zero(),JointVector::Zero(),goal,d,rates);
  };
  auto base=run(1.,{-1,0,0}),boosted=run(2.,{-1,0,0});
  ASSERT_GT(base.metric,0.);
  EXPECT_NEAR(boosted.metric,2*base.metric,1e-8);
  EXPECT_NEAR((boosted.metric_contribution-2*base.metric_contribution).norm(),0.,1e-8);
  EXPECT_NEAR((boosted.force_contribution-2*base.force_contribution).norm(),0.,1e-8);
  EXPECT_DOUBLE_EQ(boosted.desired_acceleration,base.desired_acceleration);
  EXPECT_DOUBLE_EQ(run(2.,{1,0,0}).metric,0.);
  d[0]=.6;EXPECT_DOUBLE_EQ(run(2.,{-1,0,0}).metric,0.);
}
TEST(TgInitialEventQueue, VersionTenPreservesSeedExecutionScores)
{
  rb10_rmpflow_rviz::TgInitialEventQueue queue;rb10_rmpflow_rviz::TgInitialEvent out;
  constexpr std::size_t payload=277,shadow=509+8*20,suffix=476+shadow+17+14;
  std::vector<double> tg(payload+107+9+suffix,0.);
  tg[26]=9;tg[50]=20;tg[274]=45045;tg[275]=10;tg[10]=100;
  tg[payload+2]=256;tg[payload+66]=8;tg[tg.size()-32]=shadow;
  tg[tg.size()-suffix-9]=8;
  for(int i=0;i<8;++i)tg[tg.size()-suffix-8+i]=.1*i;
  ASSERT_TRUE(queue.push(tg,1,100,10,{},{},{}));ASSERT_TRUE(queue.pop(out));
  EXPECT_EQ(out.data[1],2);
  for(int i=0;i<8;++i)EXPECT_DOUBLE_EQ(out.data[77+i],.1*i);
}

}  // namespace
}  // namespace rb10_rmpflow_rviz

TEST(TgInitialEventQueue, VersionElevenPreservesSeedExecutionScores)
{
  rb10_rmpflow_rviz::TgInitialEventQueue queue;rb10_rmpflow_rviz::TgInitialEvent out;
  constexpr std::size_t payload=277,shadow=509+8*20,suffix=476+shadow+17+14+505;
  std::vector<double> tg(payload+107+9+suffix,0.);
  tg[26]=9;tg[50]=20;tg[274]=45045;tg[275]=11;tg[10]=100;
  tg[payload+2]=256;tg[payload+66]=8;tg[tg.size()-537]=shadow;
  tg[tg.size()-suffix-9]=8;
  for(int i=0;i<8;++i)tg[tg.size()-suffix-8+i]=.1*i;
  ASSERT_TRUE(queue.push(tg,1,100,10,{},{},{}));ASSERT_TRUE(queue.pop(out));
  EXPECT_EQ(out.data[1],2);
  for(int i=0;i<8;++i)EXPECT_DOUBLE_EQ(out.data[77+i],.1*i);
}

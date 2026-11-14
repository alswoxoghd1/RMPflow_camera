#pragma once
#include "rb10_rmpflow_rviz/tg_authority_boost.hpp"
#include "rb10_rmpflow_rviz/tangent_sensor_history.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
namespace rb10_rmpflow_rviz {
struct TgDirectionCandidate {
  // nominal cost, uncertainty cost, unknown exposure, support, sensor, delta, sigma
  std::array<double,7> shadow{};
  double goal{0},execution{0},risk{0};
  bool valid{false};
  TgDirectionCandidate()=default;
  TgDirectionCandidate(double value):execution(value),valid(std::isfinite(value)){}
  double total() const {return valid?goal+execution-risk:std::numeric_limits<double>::quiet_NaN();}
};
// Model scores choose a provisional rotation sign, NOT an unobserved destination.
// Fresh encoder/range responses drive cumulative steering and failure memory.
// None of these scalar observations certify free space or an obstacle normal.
struct TgResponseDirection {
  static constexpr std::size_t candidate_count=72;
  static constexpr double window_seconds=.12, failure_weight=.2;
  enum Reason { Observing=0, Moving=1, StallStart=2, StallTurn=3,
    RiskStart=4, RiskTurn=5, Reverse=6, FullCircle=7, BlockInactive=8,
    MissingFeedback=9, InvalidCandidates=10, ExecutionLimited=11, UnexecutedSearch=12, AuthorityTrial=13 };
  int phase{0},failures{0},sign{1},result{0},pending_count{0},safe_windows{0};
  unsigned tried{0},sweep_sequence{0}; // tried is reserved for old trace readers
  bool bound_clamped{false},seeded{false},has_accepted{false},search_valid{false};
  double target{0},accepted{0},anchor{0},start{-1},range_start{-1},last{-1},last_scan{-1};
  double baseline{0},speed{0},best_speed{0},comparison_block{0},scan_metric{0};
  double current_score{0},best_score{0},chosen_angle{0},sweep_us{0};
  double observation_confidence{1},goal_speed{0},poor_evidence{0},progress_evidence{0},pending_angle{0};
  std::array<double,candidate_count> sweep_scores{},sweep_goal{},sweep_execution{},sweep_risk{},sweep_continuity{};
  std::array<std::array<double,candidate_count>,7> shadow{};
  double shadow_window_stamp{-1},shadow_stalled{0},shadow_worsening{0};
  // Weighted costs, updated only around directions actually applied to the robot.
  std::array<double,candidate_count> sweep_failure{};
  std::array<bool,candidate_count> score_valid{};
  // Independent command opportunity and rolling measured progress diagnostics.
  double command_last{-1}, command_time{0}, supported_time{0}, command_integral{0};
  double command_support{0}, command_speed{0}, unsupported_elapsed{0};
  double net_travel{0}, absolute_travel{0}, progress_duration{0}, progress_ratio{0};
  std::array<double,5> travel_window{}, duration_window{};
  std::size_t progress_cursor{0};
  double progress_theta{0};
  bool execution_supported{false};
  TgAuthorityBoost authority;
  bool turning{false};
  int reason{Observing},reverse_count{0},paired_samples{0};
  double turn_travel{0},turn_elapsed{0},range_closing{0},min_range{0},
    adverse_evidence{0},retreat_evidence{0},positive_score{0},negative_score{0},
    sign_margin{0},window_dt{0},turn_rate_limit{0},previous_theta{0},window_theta{0},evaluated_angle{0};
  std::size_t owner{static_cast<std::size_t>(-1)};
  Eigen::Vector3d goal{Eigen::Vector3d::Zero()},tangent{Eigen::Vector3d::Zero()},tcp_goal{Eigen::Vector3d::Zero()};
  RB10Model::Jacobian J{RB10Model::Jacobian::Zero()},tcpJ{RB10Model::Jacobian::Zero()};
  RB10Model::JointVector q0{RB10Model::JointVector::Zero()};
  std::array<tangent_sensor_history::Sample,tangent_sensor_history::kSensorCount> ranges{},risk_anchor{};
  double risk_stamp{-1};
  void pause(){authority.reset();seeded=false;phase=0;start=last=-1;search_valid=false;turning=false;}
  template<class Score>
  double update(double theta,double /*rate*/,std::size_t id,const Eigen::Vector3d & g,
    const Eigen::Vector3d & t,const RB10Model::Jacobian & jac,
    const tangent_sensor_history::Input & input,double block,Score score,double metric=1.,
    const RB10Model::Jacobian & tcp_jac=RB10Model::Jacobian::Zero(),
    const Eigen::Vector3d & goal_unit=Eigen::Vector3d::Zero(),double continuity_weight=.1,
    double scan_interval=.24,double max_turn_rate=1.5,
    const TgAuthorityBoostParams & boost_params=TgAuthorityBoostParams{}) {
    constexpr double H=window_seconds,minimum_speed=.0005/H;
    // At most 60 deg/s of target advance, still subject to the existing ODE limit.
    turn_rate_limit=std::min(std::max(0.,max_turn_rate),M_PI/3);
    const double now=input.encoder_stamp;
    if(!input.encoder_q.allFinite() || now<0 || input.stamp-now<0 || input.stamp-now>.05){
      pause();target=theta;result=-3;reason=MissingFeedback;return theta;
    }
    if(!seeded || owner!=id || (goal-g).norm()>1e-6 || now<last){
      *this=TgResponseDirection{};seeded=true;owner=id;goal=g;target=accepted=theta;
      previous_theta=theta;turn_rate_limit=std::min(std::max(0.,max_turn_rate),M_PI/3);
    }
    // Integrate post-guard backend commands even if encoder packets repeat.
    if(!input.command_available || input.stamp<input.command_stamp ||
        input.stamp-input.command_stamp>.05)command_last=-1;
    if(input.command_available && input.command_stamp>command_last){
      const double dt=input.command_stamp-command_last;
      if(command_last>=0 && dt>0 && dt<=.17 && input.command_available &&
          input.command_qd.allFinite() && input.stamp>=input.command_stamp &&
          input.stamp-input.command_stamp<=.05){
        const double v=t.dot(jac*input.command_qd);
        command_time+=dt;command_integral+=v*dt;
        if(v>minimum_speed)supported_time+=dt;
      }
      command_last=input.command_stamp;
    }
    if(now<=last)return target;
    last=now;
    const auto begin=[&](){start=now;range_start=input.stamp;q0=input.encoder_q;
      J=jac;tangent=t;tcpJ=tcp_jac;tcp_goal=goal_unit;ranges=input.samples;
      comparison_block=block;scan_metric=metric;window_theta=theta;
      command_time=supported_time=command_integral=0;};
    const auto fresh=[](const auto & s,double stamp){return std::isfinite(s.range)&&s.range>=0 &&
      stamp-s.stamp>=0 && stamp-s.stamp<=.15;};
    if(start<0){risk_anchor=input.samples;risk_stamp=input.stamp;begin();return target;}
    // Evaluate only independent observation windows; do not label one repeated
    // sensor sample as multiple pieces of evidence.
    if(now-start<H)return target;
    bool danger=false;int checked=0;double max_change=0,max_closing=0;
    double nearest=std::numeric_limits<double>::infinity(),max_opening=0;
    for(std::size_t i=0;i<ranges.size();++i){
      const auto a=ranges[i],b=input.samples[i];
      if(!fresh(b,input.stamp))continue;
      nearest=std::min(nearest,b.range);
      if(fresh(a,range_start) && b.stamp>a.stamp){
        ++checked;max_change=std::max(max_change,std::abs(a.range-b.range));
        if(std::min(a.range,b.range)<.08){
          max_closing=std::max(max_closing,a.range-b.range);
          max_opening=std::max(max_opening,b.range-a.range);
        }
        danger=danger || (b.range<.08 && b.range<a.range-3*std::sqrt(2.)*.002);
      }
      if(fresh(risk_anchor[i],risk_stamp) && b.stamp>risk_anchor[i].stamp)
        danger=danger || (b.range<.08 && b.range<risk_anchor[i].range-3*std::sqrt(2.)*.002);
    }
    window_dt=now-start;evaluated_angle=window_theta;paired_samples=checked;range_closing=max_closing;
    min_range=std::isfinite(nearest)?nearest:0.;
    if(now-start>H+.05 || checked==0 || metric<=1e-9){
      authority.reset();observation_confidence=0;phase=5;result=-3;reason=MissingFeedback;
      travel_window.fill(0);duration_window.fill(0);progress_evidence=0;
      command_support=0;execution_supported=false;
      target=theta;previous_theta=theta;begin();return target;
    }
    const double metric_ratio=std::max(metric,scan_metric)/std::max(1e-9,std::min(metric,scan_metric));
    observation_confidence=std::clamp(1./(1.+max_change/.02+std::abs(std::log(metric_ratio))+
      5*std::abs(block-comparison_block)),.1,1.);
    speed=tangent.dot(J*(input.encoder_q-q0))/(now-start);
    goal_speed=tcp_goal.dot(tcpJ*(input.encoder_q-q0))/(now-start);
    baseline=best_speed;best_speed=std::max(speed,.98*best_speed);
    const double evidence=observation_confidence;
    // Permit temporary goal retreat during a useful detour. Sustained retreat
    // without improving near-sensor response is not evidence of successful escape.
    const bool opening=max_opening>.002 && max_closing<=.002;
    retreat_evidence=std::clamp(retreat_evidence+
      (goal_speed < -minimum_speed && !opening ? window_dt*evidence : -window_dt),0.,1.2);
    adverse_evidence=std::clamp(adverse_evidence+(danger?1.:-evidence),0.,3.);
    const bool sustained_retreat=goal_speed < -minimum_speed && !opening && retreat_evidence>=.6;
    command_support=std::clamp(supported_time/window_dt,0.,1.);
    command_speed=command_time>0?command_integral/command_time:0.;
    execution_supported=command_support>=.8 && command_speed>minimum_speed;
    unsupported_elapsed=execution_supported?0.:std::min(2.,unsupported_elapsed+window_dt);
    // A direction change or abrupt range change breaks the local comparison.
    // No obstacle identity is inferred. Old directional failures also expire.
    if(std::abs(std::remainder(theta-progress_theta,2*M_PI))>M_PI/6 || max_change>.03){
      travel_window.fill(0);duration_window.fill(0);progress_cursor=0;
      progress_evidence=0;progress_theta=theta;
      if(max_change>.03){authority.reset();sweep_failure.fill(0);poor_evidence=0;}
    }
    travel_window[progress_cursor]=speed*window_dt;
    duration_window[progress_cursor]=window_dt;
    progress_cursor=(progress_cursor+1)%travel_window.size();
    net_travel=absolute_travel=progress_duration=0;
    for(std::size_t i=0;i<travel_window.size();++i){
      net_travel+=travel_window[i];absolute_travel+=std::abs(travel_window[i]);
      progress_duration+=duration_window[i];
    }
    progress_ratio=absolute_travel>1e-9?net_travel/absolute_travel:0.;
    const bool good=progress_duration>=.36 && net_travel>minimum_speed*progress_duration &&
      progress_ratio>=.5 && speed>minimum_speed && !danger && !sustained_retreat;
    bool trial_failed=execution_supported && progress_duration>=.36 && !good;
    // Command cancellation is not evidence that the obstacle blocks this heading.
    // A prolonged inability to execute can still trigger an unpenalized search.
    authority.observe(window_dt,boost_params,block>.1 && metric>1e-9,
      execution_supported,good,danger,turning,std::clamp(command_time/window_dt,0.,1.));
    const bool exploratory=!execution_supported && unsupported_elapsed>=.6 && !good;
    shadow_window_stamp=now;
    shadow_stalled=speed<=minimum_speed?1.:0.;shadow_worsening=danger?1.:0.;
    poor_evidence=std::clamp(poor_evidence+(good?-evidence:(danger?1.:(trial_failed?evidence:0.))),0.,3.);
    progress_evidence=std::clamp(progress_evidence+(good?evidence:-evidence),0.,3.);
    if(input.stamp-risk_stamp>.5){risk_anchor=input.samples;risk_stamp=input.stamp;}
    if(block<=.1){
      target=theta;turning=false;phase=0;result=0;reason=BlockInactive;
      poor_evidence=progress_evidence=adverse_evidence=retreat_evidence=0;
      previous_theta=theta;begin();return target;
    }
    if(turning){
      // Accumulate actual applied angle, not a target the ODE has not reached.
      turn_travel+=std::max(0.,sign*std::remainder(theta-previous_theta,2*M_PI));
      turn_elapsed+=window_dt;
    }
    previous_theta=theta;
    for(std::size_t k=0;k<candidate_count;++k){
      const double distance=std::abs(std::remainder(k*2*M_PI/candidate_count-evaluated_angle,2*M_PI));
      const double local=std::max(0.,1.-distance/(M_PI/6));
      sweep_failure[k]*=std::exp(-window_dt/.6);
      const double change=good ? -.5*evidence : (trial_failed?evidence:0.);
      sweep_failure[k]=std::clamp(sweep_failure[k]+failure_weight*local*change,0.,3*failure_weight);
    }
    const double reference=theta;
    const double inertia=std::max(0.,continuity_weight)*(1.+progress_evidence/3.);
    const bool scan_due=last_scan<0 || now-last_scan>=std::max(.24,scan_interval);
    if(scan_due){
      last_scan=now;
      const auto started=std::chrono::steady_clock::now();
      const double alpha=.35*observation_confidence;
      for(std::size_t k=0;k<candidate_count;++k){
        const TgDirectionCandidate c=score(k*2*M_PI/candidate_count);const double raw=c.total();
        sweep_goal[k]=c.goal;sweep_execution[k]=c.execution;sweep_risk[k]=c.risk;
        for(std::size_t j=0;j<shadow.size();++j)shadow[j][k]=c.shadow[j];
        if(!std::isfinite(raw)){score_valid[k]=false;sweep_scores[k]=NAN;continue;}
        sweep_scores[k]=score_valid[k]?(1-alpha)*sweep_scores[k]+alpha*raw:raw;
        score_valid[k]=true;
      }
      ++sweep_sequence;
      sweep_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
    }
    best_score=-std::numeric_limits<double>::infinity();chosen_angle=reference;
    positive_score=negative_score=-std::numeric_limits<double>::infinity();
    bool chosen_valid=false;
    for(std::size_t k=0;k<candidate_count;++k){
      const double angle=k*2*M_PI/candidate_count;
      sweep_continuity[k]=inertia*std::cos(angle-reference);
      if(!score_valid[k])continue;
      const double value=sweep_scores[k]+sweep_continuity[k]-sweep_failure[k];
      const double offset=std::remainder(angle-reference,2*M_PI);
      // The ambiguous antipode and current +/-15 degrees do not vote for a side.
      if(offset>=M_PI/12 && offset<M_PI-1e-6)positive_score=std::max(positive_score,value);
      if(offset<=-M_PI/12 && offset>-M_PI+1e-6)negative_score=std::max(negative_score,value);
      if(value>best_score+1e-9 || (std::abs(value-best_score)<=1e-9 &&
        std::abs(std::remainder(angle-reference,2*M_PI))<std::abs(std::remainder(chosen_angle-reference,2*M_PI)))){
        best_score=value;chosen_angle=angle;chosen_valid=true;
      }
    }
    // Keep full-circle diagnostics; a winning model angle is only a proposal.
    auto normalized=std::fmod(reference+2*M_PI,2*M_PI);if(normalized<0)normalized+=2*M_PI;
    const double bin=normalized*candidate_count/(2*M_PI);const auto k=static_cast<std::size_t>(bin)%candidate_count;
    const auto next=(k+1)%candidate_count;const double fraction=bin-std::floor(bin);
    current_score=score_valid[k]&&score_valid[next] ?
      (1-fraction)*(sweep_scores[k]-sweep_failure[k])+
      fraction*(sweep_scores[next]-sweep_failure[next])+inertia : NAN;
    if(!chosen_valid || !std::isfinite(positive_score) || !std::isfinite(negative_score)){
      target=theta;phase=5;result=-5;reason=InvalidCandidates;begin();return target;
    }
    sign_margin=positive_score-negative_score;
    reason=Observing;result=0;
    if(good && progress_evidence>=2.){
      if(turning)target=theta;
      turning=false;accepted=theta;has_accepted=true;pending_count=0;
      phase=0;reason=Moving;result=1;begin();return target;
    }
    bool just_started=false;
    if(!turning && (poor_evidence>=1.5 || exploratory)){
      // Ties retain the prior sign; no unsupported claim that one side is safer.
      if(std::abs(sign_margin)>.03)sign=sign_margin>0?1:-1;
      turning=true;just_started=true;anchor=theta;target=theta;
      // Start the metric ramp in the same observation that starts rotation.
      authority.observe(window_dt,boost_params,metric>1e-9,
        execution_supported,good,danger,true,std::clamp(command_time/window_dt,0.,1.));
      turn_travel=turn_elapsed=0;pending_count=0;if(trial_failed || danger)++failures;
      reason=danger?RiskStart:StallStart;
    }
    if(turning){
      // Reverse only after observing a failed angular sector and a persistent
      // opposite-side advantage. Never reverse just because the best bin changed.
      const double alternate=sign>0?negative_score:positive_score;
      const double incumbent=sign>0?positive_score:negative_score;
      const bool failed_sector=!good && turn_travel>=M_PI/6 && turn_elapsed>=.6 && poor_evidence>=1.5;
      if(scan_due){
        pending_count=failed_sector && alternate>incumbent+.08 ? pending_count+1 : 0;
        pending_angle=theta-sign*M_PI/6;
      }
      const bool loop=!good && turn_travel>=2*M_PI && poor_evidence>=1.5;
      if(loop || pending_count>=2){
        sign=-sign;++reverse_count;target=theta;anchor=theta;
        turn_travel=turn_elapsed=0;pending_count=0;reason=loop?FullCircle:Reverse;
      }else if(!just_started)reason=danger?RiskTurn:StallTurn;
      // Advance from the retained target, capped 20 degrees ahead of the actual
      // angle: cumulative steering cannot skip a sector the ODE never applied.
      if(!good){
        const double ahead=std::clamp(sign*std::remainder(target-theta,2*M_PI),0.,M_PI/9);
        const double advance=turn_rate_limit*window_dt*(.5+.5*poor_evidence/3.);
        target=theta+sign*std::min(M_PI/9,ahead+advance);
      }
      search_valid=true;phase=danger?4:1;result=2;
      if(!execution_supported && !danger)reason=UnexecutedSearch;
    }else {phase=0;if(!execution_supported && !good && !danger)
      reason=ExecutionLimited;}
    begin();return target;
  }
};
}

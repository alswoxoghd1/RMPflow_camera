#pragma once
#include <algorithm>
#include <cmath>

namespace rb10_rmpflow_rviz {
struct TgAuthorityBoostParams {
  bool enabled{false};
  double maximum{2.0}, rise_rate{1.0}, decay_rate{1.0};
  // Reserved v10 trace fields; no time-based boost trigger or settlement gate.
  double trigger_time{0.0}, settle_time{0.0};
};

// Bounded authority trial, not an obstacle-failure score. Decisions use fresh
// measured windows; the multiplier itself advances smoothly at control rate.
struct TgAuthorityBoost {
  enum Phase { Idle=0, Waiting=1, Rising=2, Settling=3, Exhausted=4,
    Supported=5, Recovering=6, Danger=7, Searching=8 };
  double gain{1.0}, deficit_time{0.0}, cap_time{0.0}, command_coverage{0.0};
  int phase{Idle};
  bool exhausted{false};
  void reset() { *this=TgAuthorityBoost{}; }
  void advance(double dt,const TgAuthorityBoostParams & p) {
    if (!p.enabled) { reset();return; }
    if (!std::isfinite(dt) || dt<=0 || dt>.1) { reset();return; }
    if (phase==Rising) gain=std::min(p.maximum,gain+p.rise_rate*dt);
    if (phase==Recovering || phase==Danger || phase==Searching)
      gain=std::max(1.0,gain-p.decay_rate*dt);
    gain=std::clamp(gain,1.0,p.maximum);
  }
  void observe(double dt,const TgAuthorityBoostParams & p,bool valid,
    bool supported,bool good,bool danger,bool turning,double coverage) {
    if (!p.enabled || !valid || !std::isfinite(dt) || dt<=0) { reset();return; }
    (void)supported;
    command_coverage=coverage;
    deficit_time=cap_time=0;exhausted=false;
    if (danger) { phase=Danger;return; }
    // Rotation is the trigger. Command-deficit duration never starts a boost,
    // and the booster never delays failure evidence or the direction search.
    if (turning && !good) { phase=Rising;return; }
    phase=gain>1.0+1e-9?Recovering:Idle;
  }
};
}

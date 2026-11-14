#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <vector>

namespace rb10_rmpflow_rviz {
// SPSC: controller produces, ROS executor consumes. No allocation in push/pop.
// Wire v1: magic,version,cycle,ros,steady,owner,theta,metric,block,models,
// confidence,cp[3],e1[3],e2[3],q[6],qd[6],goal[3],seed_count,dropped,seeds[8*5].
struct TgInitialEvent { std::array<double,85> data{}; };
class TgInitialEventQueue {
public:
  static constexpr std::size_t capacity=64;
  bool push(const std::vector<double> & tg, double cycle, double ros, double steady,
    const std::array<double,6> & q, const std::array<double,6> & qd,
    const std::array<double,3> & goal)
  {
    if (tg.size()<56 || tg[26]!=9 || !std::isfinite(tg[50]) || tg[50]<0 || tg[50]>1000) return false;
    if (tg[44]>.5 || !(tg[10]>1e-9)) return false;  // actual TG contribution only
    const auto n=static_cast<std::size_t>(tg[50]);
    const auto tail=114+8*n, payload=tail+3;
    if (tg.size()<payload+67 || tg[tail]!=45045 || tg[tail+1]<1 || tg[tail+1]>10 || std::floor(tg[tail+1])!=tg[tail+1]) return false;
    if (!(static_cast<unsigned>(tg[payload+2]) & 256U)) return false;
    if (tg[payload+66]!=8 || tg.size()<payload+107) return false;
    const auto w=write_.load(std::memory_order_relaxed),r=read_.load(std::memory_order_acquire);
    if (w-r>=capacity) { dropped_.fetch_add(1,std::memory_order_relaxed); return false; }
    auto & v=events_[w%capacity].data;
    v[0]=46046;v[1]=1;v[2]=cycle;v[3]=ros;v[4]=steady;v[5]=tg[1];
    v[6]=tg[payload+10];v[7]=tg[10];v[8]=tg[46];v[9]=tg[payload+28];v[10]=tg[payload+27];
    std::copy_n(tg.begin()+11,3,v.begin()+11);
    std::copy_n(tg.begin()+80+4*n,6,v.begin()+14);
    std::copy(q.begin(),q.end(),v.begin()+20);std::copy(qd.begin(),qd.end(),v.begin()+26);
    std::copy(goal.begin(),goal.end(),v.begin()+32);v[35]=8;v[36]=0;
    std::copy_n(tg.begin()+payload+67,40,v.begin()+37);
    std::fill(v.begin()+77,v.end(),0.0);
    const std::size_t extra=(tg[tail+1]>=9?17:0)+(tg[tail+1]>=10?14:0);
    if(tg[tail+1]>=8 && (tg.size()<=extra || tg[tg.size()-extra-1]!=509+8*n))return false;
    const std::size_t suffix=tg[tail+1]>=8 ? 476+509+8*n+extra : (tg[tail+1]==7 ? 476 : (tg[tail+1]==6 ? 387 : (tg[tail+1]==5 ? 58 : (tg[tail+1]==4 ? 10 : 0))));
    if (tg[tail+1]>=3 && tg.size()>=15+suffix && tg[tg.size()-suffix-9]==8) {
      v[1]=2; std::copy_n(tg.end()-suffix-8,8,v.begin()+77);
    }
    write_.store(w+1,std::memory_order_release);return true;
  }
  bool pop(TgInitialEvent & event) {
    const auto r=read_.load(std::memory_order_relaxed),w=write_.load(std::memory_order_acquire);
    if (r==w)return false;
    event=events_[r%capacity];event.data[36]=static_cast<double>(dropped_.load(std::memory_order_relaxed));
    read_.store(r+1,std::memory_order_release);return true;
  }
private:
  std::array<TgInitialEvent,capacity> events_{};
  std::atomic<std::size_t> write_{0},read_{0},dropped_{0};
};
}

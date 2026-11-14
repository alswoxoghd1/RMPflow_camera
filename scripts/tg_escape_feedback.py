"""Encoder-based, local-linear 0.12s TG outcome audit; never commands the robot.

The world tangent and sensor Jacobian are frozen at the beginning of each
rolling interval. Missing encoders/ranges remain unknown, never successful.
"""
from collections import deque
import math
import numpy as np

class EscapeFeedback:
    def __init__(self, horizon=.12, position_epsilon=.0005, noise_floor=.002):
        self.horizon=horizon
        self.position_epsilon=position_epsilon
        self.noise_floor=noise_floor
        self.encoders=deque(maxlen=256)
        self.frames=deque(maxlen=64)
        self.last_output=-math.inf
        self.models={}
        self.tcp=None

    def encoder(self, stamp, q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(6,) or not np.isfinite(q).all(): return
        if self.encoders and stamp<=self.encoders[-1][0]:
            if stamp<self.encoders[-1][0]: self.encoders.clear();self.frames.clear()
            else: return
        self.encoders.append((stamp,q.copy()))

    def observe(self, snapshot):
        stamp=snapshot['ros_time_s'];tg=snapshot['tg'];h=snapshot.get('history')
        if stamp-self.last_output < .02: return None
        self.last_output=stamp
        if not snapshot.get('solve_ok',True) or not snapshot.get('backend_application_enabled',True):
            self.frames.clear();return dict(status='command_inactive')
        if not h or len(tg)<56 or tg[44]>.5 or tg[10]<=1e-9:
            self.frames.clear();return dict(status='tg_off')
        if h.get('model_keyframe'): self.models.clear()
        for m in h.get('models',[]): self.models[int(m['sensor'])]=m
        tcp_data=snapshot.get('solver_tcp_position_J_curvature')
        if tcp_data is not None:
            rows=np.asarray(tcp_data).reshape(3,8)
            self.tcp=(stamp,rows[:,0].copy(),rows[:,1:7].copy(),np.asarray(snapshot['backend_q']))
        owner=int(tg[1]);points=snapshot['visualization_points']
        available=[v for v in self.encoders if 0<=stamp-v[0]<=.05]
        if not available or owner<0 or owner>=len(points):
            self.frames.clear();return dict(status='encoder_unavailable')
        encoder_stamp,q=available[-1]
        frame=dict(t=stamp,et=encoder_stamp,q=q,owner=owner,goal=np.asarray(snapshot['goal_pose'][:3]),
                   tangent=np.asarray(tg[20:23]),J=np.asarray(points[owner][4:22]).reshape(3,6),
                   predicted=np.asarray(h['predicted_displacement']),samples=h['samples'])
        if self.frames and (owner!=self.frames[-1]['owner'] or
                np.linalg.norm(frame['goal']-self.frames[-1]['goal'])>1e-6 or stamp<=self.frames[-1]['t']):
            self.frames.clear()
        frame['models']=dict(self.models)
        frame['tcp']=self.tcp
        self.frames.append(frame)
        while len(self.frames)>1 and self.frames[1]['t']<=stamp-self.horizon: self.frames.popleft()
        old=self.frames[0];elapsed=stamp-old['t']
        if elapsed<self.horizon or elapsed>self.horizon+.05:
            return dict(status='collecting',elapsed_s=elapsed)
        delta=old['J']@(q-old['q'])
        actual=float(old['tangent']@delta)
        predicted=float(old['tangent']@old['J']@old['predicted'])
        # Same one-sided softplus pressure as the controller. Compare sensors
        # individually: clearance recovery cannot mask another sensor worsening.
        p=h['effective_parameters'];clearance=p['clearance'];length=p['risk_length']
        def pressure(d):
            x=(clearance-d)/length
            return (max(x,0)+math.log1p(math.exp(-abs(x))))**2
        checked=0;worsened=[];unknown=0;residuals=[]
        for j,(a,b) in enumerate(zip(old['samples'],frame['samples'])):
            da,db=a['raw_equivalent_range'],b['raw_equivalent_range']
            fresh=lambda s,now: 0<=now-s['acquisition_stamp']<=p['fresh_time']
            if not (math.isfinite(da) and math.isfinite(db) and fresh(a,old['t']) and fresh(b,stamp)):
                unknown+=1;continue
            checked+=1
            epsilon_d=3*math.sqrt(2)*self.noise_floor
            threshold=max(abs(pressure(da+epsilon_d)-pressure(da)),
                          abs(pressure(da-epsilon_d)-pressure(da)))
            change=pressure(db)-pressure(da)
            if change>threshold: worsened.append(j)
            residuals.append(dict(sensor=j,range_change_m=db-da,pressure_change=change,
                                  pressure_threshold=threshold))
        range_errors=[]
        for j,(a,b) in enumerate(zip(old['samples'],frame['samples'])):
            m=old['models'].get(j)
            if not m or not m['valid'] or old['t']-m['stamp']>p['expiry']: continue
            da,db=a['raw_equivalent_range'],b['raw_equivalent_range']
            if not (math.isfinite(da) and math.isfinite(db) and fresh(a,old['t']) and fresh(b,stamp)): continue
            expected=float(np.asarray(m['gradient'])@(q-old['q']))
            range_errors.append(dict(sensor=j,error_m=db-da-expected,predicted_change_m=expected))
        goal_progress=None
        if old['tcp'] is not None and old['t']-old['tcp'][0]<.2:
            _,pos,jac,anchor=old['tcp']
            before=pos+jac@(old['q']-anchor);after=pos+jac@(q-anchor)
            goal_progress=float(np.linalg.norm(old['goal']-before)-np.linalg.norm(old['goal']-after))
        status=('range_worsening' if worsened else 'reverse' if actual < -self.position_epsilon
                else 'no_directional_progress' if actual<=self.position_epsilon
                else 'progress_observed' if checked else 'range_unknown')
        return dict(status=status,elapsed_s=elapsed,owner=owner,actual_progress_m=actual,
                    predicted_progress_m=predicted,motion_prediction_error_m=actual-predicted,
                    range_prediction_errors=range_errors,goal_progress_m=goal_progress,
                    goal_progress_confirmed=(goal_progress>self.position_epsilon) if goal_progress is not None else None,
                    encoder_q_start=old['q'].tolist(),encoder_q_end=q.tolist(),
                    checked_sensors=checked,unknown_sensors=unknown,worsened_sensors=worsened,
                    range_changes=residuals,encoder_age_s=stamp-encoder_stamp,
                    position_epsilon_m=self.position_epsilon,range_noise_floor_m=self.noise_floor,
                    source='encoder_joint_states_with_frozen_sensor_J',
                    note='Local movement evidence; not global bypass completion or collision safety.')

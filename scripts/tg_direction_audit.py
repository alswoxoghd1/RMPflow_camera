#!/usr/bin/env python3
"""Offline evidence for TG seed selection and continuous direction updates.

Candidate scores are recorded initialization decisions only. No missing model
or candidate score is interpolated or reconstructed from a different cycle.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from rmp_detail_capture import decode_snapshot
from rmp_history_record import RecordingQuality
from tg_initial_events import TOPIC as INITIAL_TOPIC, decode_initial_event, event_key

SCORE_FIELDS = ('goal_score', 'continuity_score', 'negative_history_cost', 'total_score')
COLORS = ('#24a269', '#3584e4', '#e66151')


def direction(tg, angle):
    k = 76 + 4 * int(tg[50])
    return np.cos(angle)*np.asarray(tg[k+4:k+7]) + np.sin(angle)*np.asarray(tg[k+7:k+10])


def seed_choice(seeds):
    # Match the controller's deterministic tie tolerance and iteration order.
    best, score = None, -math.inf
    for i, s in enumerate(seeds):
        if s['total_score'] > score + 1e-9:
            best, score = i, s['total_score']
    return best


def load_audit(path):
    frames, episodes, quality = [], [], RecordingQuality()
    start, previous, episode = None, None, None
    initial_events={};event_errors=0
    with Path(path).open() as stream:
        for line in stream:
            if INITIAL_TOPIC in line[:256]:
                try:
                    item=json.loads(line);data=item['message']['data']
                    initial_events[event_key(data)]=decode_initial_event(data)
                except (ValueError,IndexError,KeyError,TypeError,OverflowError):event_errors+=1
                continue
            if '/rmp_recording_snapshot' not in line[:256]:
                continue
            item = json.loads(line)
            if item.get('topic') != '/rmp_recording_snapshot':
                continue
            s = decode_snapshot(item['message']['data'])
            if start is None:
                start = s['steady_time_s']
            quality.observe(s)
            tg, h = s['tg'], s['history']
            if h is None:
                continue
            t = s['steady_time_s']-start
            active, owner = tg[0] > .5, int(tg[1])
            params = h['effective_parameters']
            dt, tau = h['dt'], params['direction_filter_tau']
            rate = h['filtered_rate_after']
            expected_rate = (h['filtered_rate_before'] + dt/(max(tau, 0)+dt)*
                             (h['clamped_rate']-h['filtered_rate_before'])) if dt > 0 else math.nan
            k = 68 + 4*int(tg[50])  # weighted goal/velocity dipole coefficients
            angle = h['theta_evaluated']
            goal_score = tg[k]*math.cos(angle)+tg[k+1]*math.sin(angle)
            motion_score = tg[k+2]*math.cos(angle)+tg[k+3]*math.sin(angle)
            row = dict(time_s=t, sequence=int(s['control_sequence']), owner=owner,
                       active=active, contributing=active and tg[10] > 1e-9 and tg[44] < .5,
                       block=tg[46], metric=tg[10], theta_rad=tg[3],
                       theta_evaluated=angle, clearance_m=tg[2] if active else math.nan,
                       goal_score=goal_score, motion_score=motion_score,
                       negative_history_cost=-params['weight_risk']*h['risk_before'],
                       goal_gradient=h['goal_gradient'], motion_gradient=h['continuity_gradient'],
                       history_gradient=h['history_gradient'], escape_gradient=h.get('escape_gradient',0), response=h.get('response'),
                       escape_score=h.get('weight_escape',0)*tg[46]*h.get('escape_before',0), total_gradient=h['total_gradient'],
                       raw_rate=h['raw_rate'], clamped_rate=h['clamped_rate'], filtered_rate=rate,
                       model_count=h['model_count_before'], confidence=h['confidence_before'],
                       measured_risk=h['measured_risk_before'], unknown_risk=h['unknown_risk_before'],
                       predicted_min_range_m=h['min_predicted_range_before'] if math.isfinite(h['min_predicted_range_before']) else math.nan,
                       critical_sensor=int(h['critical_sensor_before']),
                       predicted_joint_step_norm=float(np.linalg.norm(h['predicted_displacement'])),
                       models_complete=quality.models_complete,
                       rate_error=abs(expected_rate-rate) if active else math.nan,
                       angle_error=abs(math.remainder(h['theta_after']-angle-rate*dt, 2*math.pi)) if active else math.nan,
                       gradient_error=abs(h['goal_gradient']+h['continuity_gradient']+h['history_gradient']+h.get('escape_gradient',0)-h['total_gradient']) if active else math.nan,
                       direction_error=float(np.linalg.norm(direction(tg,tg[3])-tg[20:23])) if active and tg[10]>1e-9 and tg[44]<.5 else math.nan,
                       world_turn_deg_s=math.nan, missing_cycles=0,
                       events=';'.join(h['event_names']))
            if previous is not None:
                row['missing_cycles'] = max(0, row['sequence']-previous['sequence']-1)
                gap = t-previous['time_s']
                if row['contributing'] and previous['contributing'] and owner == previous['owner'] and 0 < gap <= .1:
                    v, u = np.asarray(tg[20:23]), np.asarray(previous['tg'][20:23])
                    row['world_turn_deg_s'] = math.degrees(math.acos(np.clip(
                        v@u/max(np.linalg.norm(v)*np.linalg.norm(u), 1e-12), -1, 1)))/gap
            if h.get('response',{}).get('bound_clamped',False):
                row['rate_error']=row['angle_error']=0.0  # explicit constrained integration
            row.update(tg=tg[:114+8*int(tg[50])], seeds=h['seeds'],
                       seed_applied='initial_seed_applied' in h['event_names'],
                       q=s['solver_q'], params=params, goal=s['goal_pose'][:3])
            frames.append(row)
            if active and (previous is None or not previous['active'] or row['seed_applied']):
                episode = dict(index=len(episodes), start=t, first=len(frames)-1,
                               owner=owner, first_contribution=None)
                episodes.append(episode)
            if active and episode is not None and row['contributing'] and episode['first_contribution'] is None:
                episode['first_contribution'] = t
            previous = row
    if not frames:
        raise ValueError('No sensor_history decision traces in this recording')
    events=sorted(initial_events.values(),key=lambda e:e['steady_time_s'])
    for event in events:event['time_s']=event['steady_time_s']-start
    frames[0]['initial_events']=events
    index=0;reference=None;last_inactive=-1
    for row in frames:
        while index<len(events) and events[index]['steady_time_s']<=start+row['time_s']+1e-8:
            reference=events[index];reference['time_s']=reference['steady_time_s']-start;index+=1
        if not row['active']:last_inactive=row['sequence']
        if (reference is not None and row['active'] and reference['owner']==row['owner']
                and reference['sequence']>last_inactive):
            row['initial_reference']=reference
    summary=quality.summary()
    summary.update(retained_initial_events=len(events),initial_event_decode_errors=event_errors,
                   initial_event_queue_dropped=max((e['queue_dropped'] for e in events),default=0))
    return frames, episodes, summary


def contribution_episodes(frames):
    """Use retained source events even if the entire ON burst missed telemetry."""
    events=frames[0].get('initial_events',[]) if frames else []
    result=[]
    for event in events:
        if event['contributing'] and event['time_s']>=0:
            first=next((i for i,r in enumerate(frames) if r['time_s']>=event['time_s']),len(frames)-1)
            result.append(dict(start=event['time_s'],first=first,owner=event['owner'],
                               first_contribution=event['time_s'],trigger='retained_initial_event',
                               initial_event=event))
    for i,row in enumerate(frames):
        if row['contributing'] and (i == 0 or not frames[i-1]['contributing']):
            previous=frames[i-1]['time_s'] if i else -float('inf')
            source=next((e for e in reversed(events) if previous<e['time_s']<=row['time_s']+1e-8
                         and e['owner']==row['owner']),None)
            if source is not None and source['time_s']>=0:continue
            result.append(dict(start=row['time_s'],first=i,owner=row['owner'],
                               first_contribution=row['time_s'],
                               trigger='metric_positive_and_not_observe_only'))
    result.sort(key=lambda e:e['start'])
    for i,episode in enumerate(result):episode['index']=i
    return result


def contextual_seed(frames, now):
    """Last recorded seed in the current observed detection, never a future seed."""
    selected=None
    for row in frames:
        if row['time_s'] > now+1e-8: break
        if not row['active']: selected=None
        elif row.get('initial_reference') is not None: selected=row['initial_reference']
        elif row['seed_applied']: selected=row if row['seeds'] else None
    return selected


def window(frames, start, duration=5.):
    return [r for r in frames if start <= r['time_s'] <= start+duration]


def seed_frame(rows, now):
    choices = [r for r in rows if r['time_s'] <= now+1e-8 and r['seed_applied'] and r['seeds']]
    return choices[-1] if choices else None


def seed_rows(frame):
    if frame is None:
        return []
    winner = seed_choice(frame['seeds'])
    rows = []
    for i, seed in enumerate(frame['seeds']):
        rows.append(dict(time_s=frame['time_s'], sequence=frame['sequence'], owner=frame['owner'],
                         candidate=i, angle_deg=math.degrees(seed['theta']) % 360,
                         **{k:seed[k] for k in SCORE_FIELDS}, escape_score=seed.get('escape_score',0.), selected=i == winner,
                         sum_error=abs(sum(seed[k] for k in SCORE_FIELDS[:3])+seed.get('escape_score',0)-seed['total_score'])))
    return rows


def summarize(rows, contributing_only=False):
    active = [r for r in rows if r['contributing' if contributing_only else 'active']]
    seed_map={r['sequence']:r for r in rows if r['seed_applied'] and r['seeds']}
    for r in rows:
        ref=r.get('initial_reference')
        if ref is not None and rows[0]['time_s']<=ref['time_s']<=rows[-1]['time_s']:
            seed_map[ref['sequence']]=ref
    for ref in (rows[0].get('initial_events',[]) if rows else []):
        if rows[0]['time_s']<=ref['time_s']<=rows[-1]['time_s']:seed_map[ref['sequence']]=ref
    seeds=sorted(seed_map.values(),key=lambda r:r['time_s'])
    margins, mismatches = [], 0
    for r in seeds:
        winner = seed_choice(r['seeds'])
        scores = sorted((s['total_score'] for s in r['seeds']), reverse=True)
        margins.append(scores[0]-scores[1] if len(scores)>1 else math.nan)
        mismatches += abs(math.remainder(r['seeds'][winner]['theta']-r['theta_evaluated'], 2*math.pi)) > 1e-8
    finite_max = lambda key: max((r[key] for r in active if math.isfinite(r[key])), default=None)
    return dict(saved_frames=len(rows), active_frames=len(active),
                contributing_frames=sum(r['contributing'] for r in rows), recorded_initializations=len(seeds),
                initial_selection_mismatches=mismatches, initial_score_margins=margins,
                initial_history_cost_spreads=[max(s['negative_history_cost'] for s in r['seeds'])-min(s['negative_history_cost'] for s in r['seeds']) for r in seeds],
                max_candidate_sum_error=max((s['sum_error'] for r in seeds for s in seed_rows(r)),default=None),
                max_filter_error=finite_max('rate_error'), max_integrator_error=finite_max('angle_error'),
                max_gradient_sum_error=finite_max('gradient_error'), max_direction_vector_error=finite_max('direction_error'),
                max_abs_filtered_rate_rad_s=max((abs(r['filtered_rate']) for r in active),default=None),
                max_world_turn_deg_s=finite_max('world_turn_deg_s'),
                frames_with_complete_models=sum(r['models_complete'] for r in rows),
                missing_control_cycles_between_saved_frames=sum(r['missing_cycles'] for r in rows[1:]),
                scope='Saved samples only; candidate optimality is within eight recorded initial seeds. No full model replay or actual obstacle bearing validation.')


def write_csv(path, rows, fields=None):
    if not rows and fields is None:
        path.write_text('')
        return
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else fields)
        writer.writeheader(); writer.writerows(rows)


def draw_figure(fig, rows, seed=None, now=None, initial_preview=False):
    fig.clear()
    axes = fig.subplots(3, 2)
    a = axes[0,0]
    if seed is not None:
        sr = seed_rows(seed); x = np.arange(len(sr)); width=.18
        for i,(key,label,color) in enumerate(zip((*SCORE_FIELDS[:3],'escape_score'), ['Goal (weighted)','Motion (weighted)','History cost (negative)','Escape (weighted)'],(*COLORS,'#a27bd3'))):
            a.bar(x+(i-1.5)*width,[r[key] for r in sr],width,color=color,label=label)
        a.plot(x,[r['total_score'] for r in sr],'kD-',label='Total',markersize=4)
        winner = seed_choice(seed['seeds']);a.axvspan(winner-.45,winner+.45,color='#f6d32d',alpha=.22)
        a.set_xticks(x,[f'C{i}' for i in x]);a.set_title(f'REFERENCE: initial choice at {seed["time_s"]:.3f}s; C{winner}',fontsize=10)
        a.legend(fontsize=7,loc='best'); a.set_ylabel('Weighted score (not probability)')
    else:
        a.text(.5,.5,'Initial candidate scores not recorded\nin this window before the cursor',ha='center',va='center',transform=a.transAxes)
    if not rows:
        return axes
    t = np.array([r['time_s'] for r in rows]); active=np.array([r['contributing'] for r in rows])
    current = min(rows,key=lambda r:abs(r['time_s']-(now if now is not None else t[0])))
    response=current.get('response') or {}
    full_circle='sweep_goal' in response and response.get('sweep_sequence',0)>0
    a=axes[0,1]
    keys=['goal_gradient','motion_gradient','history_gradient','escape_gradient','total_gradient']
    if initial_preview:
        a.text(.5,.5,'INITIAL EVENT VIEW\nCandidate scores and recorded pose only',ha='center',va='center',transform=a.transAxes)
    elif current['contributing'] and full_circle:
        deg=np.arange(72)*5
        for key,label in [('sweep_goal','Goal (raw)'),('sweep_execution','Execution (raw)'),('sweep_risk','Risk cost (raw)'),('sweep_continuity','Continuity')]:
            values=np.asarray(response[key]);a.plot(deg,-values if key=='sweep_risk' else values,label=label,lw=.8)
        failure=np.asarray(response.get('sweep_failure',[0.]*72))
        if 'sweep_failure' in response:
            a.plot(deg,-failure,label='Observed failure cost',lw=.8)
        a.plot(deg,np.asarray(response['sweep_scores'])+response['sweep_continuity']-failure,color='black',label='Guidance score')
        a.axvline(math.degrees(response['target'])%360,color='#e5a50a',label='Steering target')
        if 'reason_name' in response:
            a.text(.02,.97,response['reason_name'],transform=a.transAxes,va='top',fontsize=7)
        a.legend(fontsize=6);a.set_xlim(0,360);a.set_xlabel('Sensor-plane angle (deg)')
    elif current['contributing']:
        a.bar(['Goal','Motion','History','Direction correction','Sum'],[current[k] for k in keys],color=[*COLORS,'#a27bd3','#777777'])
    else:
        a.text(.5,.5,'TG OFF: direction updates hidden',ha='center',transform=a.transAxes)
    a.axhline(0,color='black',lw=.5);a.set_ylabel('d score / d theta (per rad)')
    a.set_title(f'Initial event at {seed["time_s"]:.6f}s' if initial_preview and seed else f'LOCAL turning preference at {current["time_s"]:.3f}s',fontsize=10)
    if full_circle and not initial_preview:
        a.set_ylabel('Dimensionless score');a.set_title('Full-circle update: raw components / filtered total',fontsize=9)
    if initial_preview:a.set_axis_off()
    if full_circle and not initial_preview:
        for key,label in [('observation_confidence','Observation confidence'),('poor_evidence','Poor progress evidence'),('progress_evidence','Good progress evidence')]:
            axes[1,0].plot(t,[(r.get('response') or {}).get(key,np.nan) if r['contributing'] else np.nan for r in rows],label=label)
        axes[1,0].set_ylabel('Confidence / accumulated evidence');axes[1,0].legend(fontsize=7)
    else:
        for key,color,label in zip(keys[:4],(*COLORS,'#a27bd3'),['Goal','Motion','History','Direction correction']):
            vals=np.array([r[key] for r in rows],dtype=float);vals[~active]=np.nan
            axes[1,0].plot(t,vals,label=label,color=color)
        axes[1,0].set_ylabel('d score / d theta');axes[1,0].legend(fontsize=7)
    for key,label,style in [('raw_rate','Before limit',':'),('clamped_rate','After limit','--'),('filtered_rate','Applied filter','-')]:
        vals=np.array([r[key] for r in rows],dtype=float);vals[~active]=np.nan
        axes[1,1].plot(t,vals,style,label=label)
    axes[1,1].set_ylabel('Local theta rate (rad/s)');axes[1,1].legend(fontsize=7)
    angles=np.full(len(rows),np.nan);begin=0
    for j in range(len(rows)+1):
        split=(j==len(rows) or not rows[j]['contributing'] or (j>0 and (
            rows[j]['owner']!=rows[j-1]['owner'] or rows[j]['seed_applied'] or t[j]-t[j-1]>.1)))
        if split:
            if j>begin:
                angles[begin:j]=np.degrees(np.unwrap([r['theta_rad'] for r in rows[begin:j]]))
                if begin>0: angles[begin]=np.nan
            begin=j+1 if j<len(rows) and not rows[j]['contributing'] else j
    axes[2,0].plot(t,angles,color='#e5a50a',label='Selected local angle (segments)')
    axes[2,0].set_ylabel('theta (deg, sensor basis)');axes[2,0].legend(fontsize=7)
    for j in range(1,len(rows)):
        if rows[j]['contributing'] and rows[j-1]['contributing'] and rows[j]['owner']!=rows[j-1]['owner']:
            axes[2,0].axvline(t[j],color='#888888',ls=':',lw=.7)
            axes[2,0].text(t[j],.95,f'S{rows[j-1]["owner"]} -> S{rows[j]["owner"]}',transform=axes[2,0].get_xaxis_transform(),rotation=90,va='top',fontsize=7)
    axes[2,1].plot(t,[r['block'] for r in rows],label='Block',color='#9141ac')
    axes[2,1].step(t,[float(r['contributing']) for r in rows],where='post',label='TG contributes (0/1)',color='#3584e4')
    axes[2,1].set(ylim=(-.05,1.05),ylabel='Activation');axes[2,1].legend(fontsize=7)
    for a in axes.flat:
        a.grid(alpha=.2)
    for a in axes[1:,:].flat:
        a.set(xlim=(t[0],max(t[-1],t[0]+.01)),xlabel='Recording time (s)')
        if now is not None:a.axvline(now,color='black',lw=.7)
        for r in rows:
            if r['seed_applied'] or 'owner_changed' in r['events']:
                a.axvline(r['time_s'],color='#999999',alpha=.4,ls=':')
    fig.suptitle('TG direction audit | metric-positive intervals only; gaps = TG OFF',fontsize=11)
    return axes


def export_window(frames, episode, folder, duration=5.):
    import matplotlib
    matplotlib.use('Agg') if 'matplotlib.pyplot' not in __import__('sys').modules else None
    from matplotlib.figure import Figure
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rows=window(frames,episode['start'],duration)
    stem=f"tg_on_{episode['index']:02d}_{episode['start']:.3f}s"
    excluded={'tg','q','seeds','params','goal','initial_reference','initial_events'}
    write_csv(folder/(stem+'_updates.csv'),[{k:v for k,v in r.items() if k not in excluded} for r in rows])
    sweeps=[];previous=None
    for r in rows:
        response=r.get('response') or {};key=(r['owner'],response.get('sweep_sequence',0))
        if r['contributing'] and 'sweep_goal' in response and key[1]>0 and key!=previous:
            for k in range(72):
                sweeps.append(dict(observed_time_s=r['time_s'],owner=r['owner'],sweep_sequence=key[1],angle_deg=5*k,
                    goal=response['sweep_goal'][k],execution=response['sweep_execution'][k],risk_cost=response['sweep_risk'][k],
                    continuity=response['sweep_continuity'][k],filtered_score=response['sweep_scores'][k],
                    observed_failure_cost=response.get('sweep_failure',[0.]*72)[k],
                    reason=response.get('reason_name','model_destination'),
                    steering_target_deg=math.degrees(response['target'])%360,
                    decision_score=response['sweep_scores'][k]+response['sweep_continuity'][k]-response.get('sweep_failure',[0.]*72)[k]))
        previous=key
    if sweeps:write_csv(folder/(stem+'_full_circle_candidates.csv'),sweeps)
    sources={}
    if episode.get('initial_event') is not None:
        initial=episode['initial_event'];sources[initial['sequence']]=initial
    for r in rows:
        seed=contextual_seed(frames,r['time_s']) if r['contributing'] else None
        if seed is not None: sources[seed['sequence']]=seed
    write_csv(folder/(stem+'_candidates.csv'),[{**s,'reference_only':True,'selected_with_TG_on':r['contributing']} for r in sources.values() for s in seed_rows(r)],
              ['time_s','sequence','owner','candidate','angle_deg',*SCORE_FIELDS,'escape_score','selected','sum_error','reference_only','selected_with_TG_on'])
    report=summarize(rows,contributing_only=True);report.update(episode=episode,duration_requested_s=duration,
        direction_metric_scope='TG contribution-positive samples only',
        candidate_reference_times=[r['time_s'] for r in sources.values()],
        duration_available_s=rows[-1]['time_s']-episode['start'] if rows else 0,
        parameters=rows[0]['params'] if rows else {})
    (folder/(stem+'_summary.json')).write_text(json.dumps(report,indent=2,allow_nan=False))
    initial=episode.get('initial_event') or contextual_seed(frames,episode['start'])
    fig=Figure(figsize=(12,9),layout='constrained');draw_figure(fig,rows,initial,episode['start'])
    for ext in ('pdf','svg','png'):fig.savefig(folder/(stem+'.'+ext),dpi=160)
    return folder/stem


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset');parser.add_argument('--output',required=True)
    args=parser.parse_args();frames,_,quality=load_audit(args.dataset)
    episodes=contribution_episodes(frames)
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    (folder/'recording_quality.json').write_text(json.dumps(quality,indent=2))
    (folder/'whole_recording_summary.json').write_text(json.dumps(summarize(frames),indent=2))
    for episode in episodes:print(export_window(frames,episode,folder))


if __name__=='__main__':main()

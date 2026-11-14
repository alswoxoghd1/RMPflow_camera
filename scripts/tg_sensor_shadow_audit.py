#!/usr/bin/env python3
"""Stream v8 recordings into independent sensor-evidence CSVs and a PNG.

Read-only. No direction/force commands. Comparisons concern supported model
candidates, not certified free paths. Validation is pre-fit on observed motion;
it is NOT proof that an unexecuted candidate would have produced its prediction.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from rmp_detail_capture import decode_snapshot
from rmp_history_record import SHADOW_FIELDS


def compare_sides(shadow, theta, support_threshold=.6):
    """Conservative diagnostic comparison; threshold is experimental, not probability."""
    sides={1:[], -1:[]}
    for k in range(72):
        offset=math.remainder(math.radians(5*k)-theta,2*math.pi)
        if abs(offset)<math.pi/12 or abs(offset)>=math.pi-1e-6: continue
        if shadow['support'][k]<support_threshold: continue
        nominal,extra=shadow['nominal_cost'][k],shadow['uncertainty_cost'][k]
        if not all(math.isfinite(v) for v in (nominal,extra)): continue
        sides[1 if offset>0 else -1].append((nominal,extra,k))
    if not sides[1] or not sides[-1]:return 'insufficient_support',0
    pos=min(sides[1]);neg=min(sides[-1])
    # Non-overlap of conservative cost envelopes, not a statistical test.
    if pos[0]+pos[1]+1e-9<neg[0]: return 'positive_supported_candidates',1
    if neg[0]+neg[1]+1e-9<pos[0]: return 'negative_supported_candidates',-1
    return 'no_resolved_difference',0


def audit(path, output, start=0., duration=math.inf):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    sweeps=[];checks=[];outcomes=[];seen_checks={};previous=None;origin=None;frames=0;last_window=None
    with Path(path).open() as stream:
        for line in stream:
            if '/rmp_recording_snapshot' not in line[:256]:continue
            item=json.loads(line)
            if item.get('topic')!='/rmp_recording_snapshot':continue
            s=decode_snapshot(item['message']['data'])
            if origin is None:origin=s['steady_time_s']
            t=s['steady_time_s']-origin
            if t<start:continue
            if t>start+duration:break
            h=s.get('history') or {};r=h.get('response') or {}
            if 'shadow' not in r:continue
            frames+=1
            window=(int(s['tg'][1]),r['shadow_window_stamp'])
            if s['tg'][10]>1e-9 and s['tg'][44]<.5 and window[1]>=0 and window!=last_window:
                last_window=window
                outcomes.append(dict(time_s=t,owner=window[0],window_stamp=window[1],
                    evaluated_angle_deg=math.degrees(r['evaluated_angle']),
                    stalled=r['shadow_stalled'],range_worsening=r['shadow_worsening'],
                    tangent_speed=r['speed'],goal_speed=r['goal_speed'],reason=r['reason_name']))
            for c in h['prediction_checks']:
                if not c['valid'] or seen_checks.get(c['sensor'])==c['stamp']:continue
                seen_checks[c['sensor']]=c['stamp']
                sample_time=t+c['stamp']-s['ros_time_s']
                if start<=sample_time<=start+duration:checks.append(dict(time_s=sample_time,**c))
            tg=s['tg'];owner=int(tg[1]);key=(owner,r['sweep_sequence'])
            if tg[10]<=1e-9 or tg[44]>.5:
                previous=None
                continue
            if key[1]<=0 or key==previous:continue
            previous=key
            status,sign=compare_sides(r['shadow'],h['theta_evaluated'])
            sweeps.append(dict(time_s=t,owner=owner,sequence=r['sweep_sequence'],
                theta=h['theta_evaluated'],target=r['target'],shadow=r['shadow'],
                status=status,provisional_sign=sign))
    if not frames:raise ValueError('No v8 shadow data in this interval; record a new sensor_history run with record_rmp_details:=true.')
    fields=['time_s','owner','sequence','angle_deg',*SHADOW_FIELDS,'status','provisional_sign']
    with (output/'candidates.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for sweep in sweeps:
            for k in range(72):
                row={key:sweep[key] for key in ('time_s','owner','sequence','status','provisional_sign')}
                row.update(angle_deg=5*k,**{key:sweep['shadow'][key][k] for key in SHADOW_FIELDS})
                w.writerow(row)
    check_fields=['time_s','sensor','stamp','model_stamp','predicted','observed','error','sigma','support','valid']
    with (output/'prediction_checks.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=check_fields);w.writeheader();w.writerows(checks)
    with (output/'observed_outcomes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['time_s','owner','window_stamp','evaluated_angle_deg',
            'stalled','range_worsening','tangent_speed','goal_speed','reason'])
        w.writeheader();w.writerows(outcomes)
    from collections import Counter
    report=dict(frames=frames,scans=len(sweeps),validation_samples=len(checks),
                comparisons=dict(Counter(x['status'] for x in sweeps)),
                note='Shadow only. Cost envelopes are heuristic; support is not a safety probability. Pre-fit errors use configured model pose source; not counterfactual execution validation.')
    report['validation_by_sensor']={}
    for sensor in sorted({c['sensor'] for c in checks}):
        values=[c for c in checks if c['sensor']==sensor]
        report['validation_by_sensor'][str(sensor)]=dict(samples=len(values),
            mae_mm=1000*sum(abs(c['error']) for c in values)/len(values),
            rmse_mm=1000*math.sqrt(sum(c['error']**2 for c in values)/len(values)),
            mean_support=sum(c['support'] for c in values)/len(values))
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    if sweeps:
        # Show the scan with the strongest minimum support; its timestamp is explicit.
        best=max(sweeps,key=lambda x:min(x['shadow']['support']))
        sh=best['shadow'];angles=list(range(0,360,5))
        for key in ('nominal_cost','uncertainty_cost','unknown_exposure'):
            axes[0,0].plot(angles,sh[key],label=key)
        axes[0,0].set_title(f"Model costs at {best['time_s']:.3f}s, sensor {best['owner']}")
        axes[0,0].set_xlabel('Candidate angle (deg)');axes[0,0].legend(fontsize=8)
        axes[0,1].plot(angles,sh['support'],label='direction support')
        axes[0,1].axhline(.6,linestyle=':',color='grey',label='diagnostic threshold')
        axes[0,1].set_ylim(-.05,1.05);axes[0,1].set_title(best['status'])
        axes[0,1].set_xlabel('Candidate angle (deg)');axes[0,1].legend(fontsize=8)
    if checks:
        sensor=Counter(c['sensor'] for c in checks).most_common(1)[0][0]
        values=[c for c in checks if c['sensor']==sensor]
        for key in ('predicted','observed'):
            axes[1,0].plot([c['time_s'] for c in values],[1000*c[key] for c in values],label=key)
        axes[1,0].set_title(f'Pre-fit validation, sensor {sensor}');axes[1,0].set_ylabel('Equivalent range (mm)');axes[1,0].legend()
        axes[1,1].plot([c['time_s'] for c in values],[1000*c['error'] for c in values],label='observed - predicted')
        axes[1,1].set_ylabel('Prediction error (mm)');axes[1,1].legend()
    for a in axes[1]:a.set_xlabel('Recording time (s)')
    fig.suptitle('Sensor evidence audit — observation only, no control change')
    fig.tight_layout();fig.savefig(output/'sensor_evidence.png',dpi=160);plt.close(fig)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('recording');p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=math.inf)
    args=p.parse_args()
    print(json.dumps(audit(args.recording,args.output,args.start,args.duration),indent=2))

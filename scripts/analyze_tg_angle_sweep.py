#!/usr/bin/env python3
"""Frozen-state TG angle counterfactual. No robot commands or path prediction.
Input: decoded around_*.json produced from recording snapshots, including root
matrices/forces and solver kinematics. Artifacts preserve source recording paths.
"""
import argparse,csv,json,math,zipfile,re
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

JOINTS=['base','shoulder','elbow','wrist1','wrist2','wrist3']
def rot(axis,a):
    axis=np.asarray(axis,float);axis/=np.linalg.norm(axis)
    x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(a)*K+(1-math.cos(a))*(K@K)

def angular_velocity(urdf,q,qd,link):
    states={'base_link':(np.eye(3),np.zeros(3))};pending=list(urdf.findall('joint'))
    while pending:
        progressed=False
        for j in pending[:]:
            parent=j.find('parent').get('link');child=j.find('child').get('link')
            if parent not in states:continue
            R,w=states[parent];o=j.find('origin')
            roll,pitch,yaw=map(float,o.get('rpy','0 0 0').split()) if o is not None else (0,0,0)
            R=R@rot([0,0,1],yaw)@rot([0,1,0],pitch)@rot([1,0,0],roll)
            if j.get('name') in JOINTS:
                i=JOINTS.index(j.get('name'));axis=np.fromstring(j.find('axis').get('xyz'),sep=' ')
                w=w+R@axis*qd[i];R=R@rot(axis,q[i])
            states[child]=(R,w);pending.remove(j);progressed=True
        if not progressed:raise ValueError('URDF tree unresolved')
    return states[link][1]

def solve(M,f,offset,limit):
    scale=max(np.abs(M).max()*.01,1.)
    raw=np.linalg.solve(M/scale+offset*np.eye(6),f/scale)
    cap=min(1.,limit/max(np.abs(raw).max(),1e-15))
    return raw,raw*cap,cap

def analyze(path,lo,hi,out):
    cache=json.loads(path.read_text());record=Path(cache['recording']);stem=str(record).removesuffix('.jsonl')
    params=yaml.safe_load(Path(stem+'_params.yaml').read_text())['rmpflow_controller']['ros__parameters']
    urdf=ET.parse(stem+'_rb10_1300e.urdf').getroot()
    with zipfile.ZipFile(stem+'_sources.zip') as z:
        model=z.read('include/rb10_rmpflow_rviz/rb10_model.hpp').decode()
    specs=re.findall(r'\{"([^"]+)", (LINK\w+), Eigen::Vector3d',model)
    assert len(specs)==20,'standard sensor mapping required'
    offset=params['root_solve_offset'];limit=params['max_joint_accel'];damping=params['tg_rmp_tangent_damping']
    angles=np.arange(-90,91,1);records=[];validation=[]
    for row in cache['rows']:
        s=row['snapshot'];tg=s['tg'];time=row['t']
        if not lo<=time<=hi or not s['groups'] or tg[10]<=1e-9:continue
        assert int(tg[50])==20,'only standard sensor configuration supported'
        groups={g['group_id']:g for g in s['groups']};g=groups[6]
        M=np.array(s['root_metric']).reshape(6,6);f=np.array(s['root_force'])
        oldM=np.array(g['metric']).reshape(6,6);oldf=np.array(g['force'])
        baseM=M-oldM;basef=f-oldf
        owner=int(tg[1]);kin=np.array(s['solver_point_J_curvature'][owner]).reshape(3,7);J=kin[:,:6];curv=kin[:,6]
        q=np.array(s['solver_q']);qd=np.array(s['solver_qd']);v=J@qd
        link=specs[owner][1].lower();w=angular_velocity(urdf,q,qd,link)
        k=80+4*int(tg[50]);e1=np.array(tg[k:k+3]);e2=np.array(tg[k+3:k+6]);theta=tg[3];m=tg[10];push=tg[28]
        def leaf(angle,rate=0):
            t=np.cos(angle)*e1+np.sin(angle)*e2;perp=-np.sin(angle)*e1+np.cos(angle)*e2
            j=t@J;jdot=(rate*perp+np.cross(w,t))@v+t@curv
            a=push-damping*(t@v)
            return t,m*np.outer(j,j),j*m*(a-jdot)
        t,reM,ref=leaf(theta,tg[4]);raw,capped,_=solve(M,f,offset,limit)
        errs=[np.linalg.norm(reM-oldM)/max(1,np.linalg.norm(oldM)),np.linalg.norm(ref-oldf)/max(1,np.linalg.norm(oldf)),np.max(abs(raw-s['raw_qdd'])),np.max(abs(capped-s['limited_qdd']))]
        validation.append(errs)
        if max(errs)>1e-6:raise ValueError(f'reconstruction mismatch at {time}: {errs}')
        tcp=np.array(s['solver_tcp_position_J_curvature']).reshape(3,8);goal=np.array(s['goal_pose'][:3])-tcp[:,0];goal/=max(np.linalg.norm(goal),1e-12)
        for deg in angles:
            t,Mc,fc=leaf(theta+np.deg2rad(deg)) # settled angle, theta_dot=0 for all candidates
            _,acc,cap=solve(baseM+Mc,basef+fc,offset,limit)
            pointacc=J@acc+curv
            records.append(dict(time_s=time,offset_deg=int(deg),owner=owner,metric=m,block=tg[46],range_m=tg[2],tangent_accel=float(t@pointacc),tangent_velocity=float(t@v),displacement_120ms=float(t@(.12*v+.5*.12**2*pointacc)),goal_accel=float(goal@(tcp[:,1:7]@acc+tcp[:,7])),accel_scale=cap))
    times=sorted(set(r['time_s'] for r in records));assert times
    A=np.array([r['tangent_accel'] for r in records]).reshape(len(times),len(angles))
    disp=np.array([r['displacement_120ms'] for r in records]).reshape(A.shape)
    summary={'recording':str(record),'window_s':[lo,hi],'frames':len(times),'validation_max_errors':np.max(validation,axis=0).tolist(),'ranges':{}}
    base=A[:,90];best5=A[:,85:96].max(axis=1)
    for bound in [5,30,60,90]:
        sub=A[:,90-bound:91+bound];best=sub.max(axis=1);arg=angles[90-bound:91+bound][sub.argmax(axis=1)]
        summary['ranges'][str(bound)]={'median_best_accel':float(np.median(best)),'median_gain_over_current':float(np.median(best-base)),'median_gain_over_best_5':float(np.median(best-best5)),'median_abs_best_angle':float(np.median(abs(arg))),'positive_best_fraction':float(np.mean(best>0))}
    local=A[:,95]>A[:,85];wide=A[:,91:].max(axis=1)>A[:,:90].max(axis=1)
    summary['local_5deg_side_agrees_with_wide_best_fraction']=float(np.mean(local==wide))
    summary['recorded_angle_frozen_rate_median_accel']=float(np.median(base))
    summary['side_preference']={'positive_fraction':float(np.mean(A[:,91:].max(axis=1)>A[:,:90].max(axis=1))),'negative_fraction':float(np.mean(A[:,91:].max(axis=1)<A[:,:90].max(axis=1)))}
    tag=record.name.split('_real_')[1].split('.details')[0];dest=out/tag;dest.mkdir(parents=True,exist_ok=True)
    with (dest/'sweep.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    im=axes[0,0].pcolormesh(angles,times,A,shading='nearest',cmap='coolwarm');fig.colorbar(im,ax=axes[0,0],label='Projected acceleration (m/s²)')
    axes[0,0].set(xlabel='Offset from recorded TG direction (deg)',ylabel='Recording time (s)',title='Same-state candidate comparison; direction rate frozen to zero')
    for target in np.linspace(lo,hi,4):
        i=int(np.argmin(abs(np.array(times)-target)));axes[0,1].plot(angles,A[i],label=f'{times[i]:.3f} s')
    axes[0,1].axhline(0,color='gray',lw=.7);axes[0,1].legend();axes[0,1].set(xlabel='Offset (deg)',ylabel='Projected acceleration (m/s²)',title='Individual frames: preferred side may change')
    for bound in [5,30,60,90]:axes[1,0].plot(times,A[:,90-bound:91+bound].max(axis=1),label=f'Best within ±{bound}°')
    axes[1,0].legend();axes[1,0].set(xlabel='Time (s)',ylabel='Projected acceleration (m/s²)',title='Instantaneous best is an upper envelope, not a trajectory')
    axes[1,1].plot(angles,np.median(A,axis=0),label='Median');axes[1,1].fill_between(angles,np.quantile(A,.1,axis=0),np.quantile(A,.9,axis=0),alpha=.2,label='10–90% across frames')
    axes[1,1].legend();axes[1,1].set(xlabel='Offset (deg)',ylabel='Projected acceleration (m/s²)',title='Time variation; not a confidence interval')
    fig.suptitle(f'{tag} | {lo}–{hi} s | Frozen-state counterfactual, NOT observed escape')
    fig.savefig(dest/'angle_sweep.png',dpi=180);fig.savefig(dest/'angle_sweep.pdf');plt.close(fig)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--end',type=float,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(analyze(a.cache,a.start,a.end,a.out),indent=2))

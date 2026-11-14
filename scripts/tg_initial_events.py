#!/usr/bin/env python3
"""Versioned, rare TG initialization events, independent of sampled telemetry."""
import math
from rmp_history_record import SEED_FIELDS
TOPIC='/tg_rmp/initial_direction_events'
FIELDS=['magic','version','control_sequence','ros_time_s','steady_time_s','owner',
        'theta_evaluated','metric','block','model_count','confidence',
        'cp[3]','e1[3]','e2[3]','q[6]','qd[6]','goal[3]','seed_count','queue_dropped','seeds[8*5]']


def decode_initial_event(data):
    if len(data) not in (77,85) or data[0]!=46046 or data[1] not in (1,2) or data[35]!=8:
        raise ValueError('invalid initial-direction event')
    if not all(math.isfinite(float(v)) for v in data):
        raise ValueError('nonfinite initial-direction event')
    n=20  # internal display prefix only; all geometry is explicit in this event
    tg=[0.]*(114+8*n);tg[0]=1;tg[1]=data[5];tg[3]=data[6];tg[10]=data[7];tg[46]=data[8];tg[50]=n
    tg[11:14]=data[11:14];k=80+4*n;tg[k:k+6]=data[14:20]
    import numpy as np
    e1=np.asarray(data[14:17]);e2=np.asarray(data[17:20]);normal=np.cross(e1,e2)
    tg[17:20]=(-normal).tolist();tg[20:23]=(math.cos(data[6])*e1+math.sin(data[6])*e2).tolist()
    seeds=[dict(zip(SEED_FIELDS,data[37+i*5:42+i*5])) for i in range(8)]
    if data[1] == 2:
        if len(data) != 85: raise ValueError("truncated escape scores")
        for seed, score in zip(seeds, data[77:85]): seed["escape_score"] = score
    return dict(sequence=int(data[2]),ros_time_s=data[3],steady_time_s=data[4],
                owner=int(data[5]),theta_evaluated=data[6],theta_rad=data[6],metric=data[7],block=data[8],
                model_count=data[9],confidence=data[10],q=list(data[20:26]),goal=list(data[32:35]),
                seeds=seeds,tg=tg,active=True,contributing=data[7]>1e-9,seed_applied=True,
                queue_dropped=int(data[36]),source='retained_initial_event')


def event_key(data):
    return (float(data[4]),int(data[2]))

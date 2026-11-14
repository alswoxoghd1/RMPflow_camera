import io
import json
import math
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tg_initial_events import TOPIC,decode_initial_event
from rmp_detail_capture import DetailCapture
from std_msgs.msg import Float64MultiArray


def event_packet():
    v=[0.]*77;v[:11]=[46046.,1.,2.,1000.,100.04,0.,0.,120.,.4,1.,.8]
    v[11:20]=[.1,.2,.3,1.,0.,0.,0.,1.,0.];v[20:26]=[.42]*6;v[32:35]=[.8,.2,.5];v[35]=8
    for i in range(8):
        angle=math.pi+i*math.pi/4;v[37+5*i:42+5*i]=[angle,math.cos(angle),.1*math.sin(angle),-.01,math.cos(angle)+.1*math.sin(angle)-.01]
    return v


class InitialEventTests(unittest.TestCase):
    def test_v2_keeps_escape_scores(self):
        from tg_initial_events import decode_initial_event
        v=[0.]*85;v[0]=46046;v[1]=2;v[35]=8
        v[77:85]=list(range(8))
        decoded=decode_initial_event(v)
        self.assertEqual([s['escape_score'] for s in decoded['seeds']],list(range(8)))

    def test_schema_retains_original_time_pose_and_both_basis_axes(self):
        r=decode_initial_event(event_packet())
        self.assertEqual(r['sequence'],2);self.assertEqual(r['steady_time_s'],100.04)
        self.assertEqual(r['q'],[.42]*6);self.assertEqual(len(r['seeds']),8)
        self.assertAlmostEqual(r['seeds'][1]['total_score'],math.cos(5*math.pi/4)+.1*math.sin(5*math.pi/4)-.01)
        with self.assertRaises(ValueError):decode_initial_event(event_packet()[:-1])
        v=event_packet();v[37]=float('nan')
        with self.assertRaises(ValueError):decode_initial_event(v)

    def node(self,raw=False):
        node=MagicMock();node.range_topics=[f'/raw_distance{i}' for i in range(1,21)]
        node.file_lock=threading.Lock();node.is_recording=False
        params={'record_detail_raw_topics':raw}
        node.declare_parameter.side_effect=lambda k,v:params.setdefault(k,v)
        node.get_parameter.side_effect=lambda k:SimpleNamespace(value=params[k])
        node.get_clock.return_value.now.return_value.nanoseconds=1000000000
        return node

    def test_minimal_profile_does_not_subscribe_duplicate_streams(self):
        node=self.node();capture=DetailCapture(node)
        topics=[call.args[1] for call in node.create_subscription.call_args_list]
        self.assertEqual(set(topics),{TOPIC,'/rmp_recording_snapshot','/RMP_flag','/parameter_events','/rb10/measured_joint_states'})
        node=self.node(raw=True);DetailCapture(node)
        self.assertIn('/raw_distance20',[c.args[1] for c in node.create_subscription.call_args_list])

    def test_pre_recording_cache_and_dedup_keep_event_once(self):
        node=self.node();capture=DetailCapture(node);msg=Float64MultiArray(data=event_packet())
        capture.receive_initial(msg)
        with tempfile.TemporaryDirectory() as folder,patch.object(capture,'capture_provenance'):
            capture.start(Path(folder)/'record.csv');node.is_recording=True
            capture.receive_initial(msg);capture.stop()
            entries=[json.loads(line) for line in capture.path.read_text().splitlines()]
            self.assertEqual(len(entries),1);self.assertEqual(entries[0]['topic'],TOPIC)
            self.assertEqual(entries[0]['message']['data'][4],100.04)
            self.assertEqual(capture.metadata['initial_events_saved'],1)

    def test_late_event_recovers_initial_scores_when_selection_snapshot_missing(self):
        from test_rmp_history_record import packet
        from test_rmp_rviz_replay import snapshot
        from tg_direction_audit import load_audit,contextual_seed
        def snap(seq,active):
            original=snapshot(seq,False);tg=packet(sensor_count=2);offset=114+8*2+3
            tg[0]=float(active);tg[10]=120. if active else 0.;tg[1]=0
            tg[offset+1]=.005;tg[offset+2]=0.;tg[offset+11]=tg[offset+10]
            return original[:-123]+[len(tg)]+tg
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.details.jsonl'
            entries=[dict(topic='/rmp_recording_snapshot',message=dict(data=snap(1,False))),
                     dict(topic='/rmp_recording_snapshot',message=dict(data=snap(3,True))),
                     dict(topic=TOPIC,message=dict(data=event_packet()))]
            path.write_text(''.join(json.dumps(e)+'\n' for e in entries))
            frames,_,quality=load_audit(path)
        reference=contextual_seed(frames,frames[-1]['time_s'])
        self.assertIsNotNone(reference)
        self.assertEqual(reference['source'],'retained_initial_event')
        self.assertAlmostEqual(reference['time_s'],.02)
        self.assertEqual(reference['sequence'],2);self.assertEqual(reference['q'],[.42]*6)
        self.assertEqual(quality['retained_initial_events'],1)
        self.assertEqual(sum(r['seed_applied'] for r in frames),0)
        from tg_direction_audit import contribution_episodes
        episodes=contribution_episodes(frames)
        self.assertEqual(len(episodes),1)
        from tg_direction_audit import summarize
        self.assertEqual(summarize(frames)['recorded_initializations'],1)
        self.assertAlmostEqual(episodes[0]['start'],.02)
        frames[-1]['contributing']=False
        self.assertEqual(len(contribution_episodes(frames)),1)  # event survives even a completely missed ON burst

if __name__=='__main__':unittest.main()

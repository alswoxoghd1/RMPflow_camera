import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tg_sensor_shadow_audit import compare_sides

class ShadowTest(unittest.TestCase):
    def data(self):
        return dict(nominal_cost=[0.]*72,uncertainty_cost=[.1]*72,support=[.9]*72)

    def test_equal_responses_do_not_invent_preferred_side(self):
        self.assertEqual(compare_sides(self.data(),0),('no_resolved_difference',0))

    def test_unknown_direction_is_not_known_safe(self):
        s=self.data();s['support'][37:]=[0.]*35
        self.assertEqual(compare_sides(s,0),('insufficient_support',0))

    def test_comparison_requires_separation_beyond_uncertainty(self):
        s=self.data();s['nominal_cost'][37:]=[.3]*35
        self.assertEqual(compare_sides(s,0)[1],1)
        s['uncertainty_cost']=[.5]*72
        self.assertEqual(compare_sides(s,0)[1],0)

    def test_stream_export_deduplicates_scans_and_validation_samples(self):
        import csv,json,tempfile
        from unittest.mock import patch
        from tg_sensor_shadow_audit import audit
        from rmp_history_record import SHADOW_FIELDS
        snapshots=[]
        for i in range(3):
            sequence=1 if i<2 else 2
            shadow={name:[0.]*72 for name in SHADOW_FIELDS};shadow['support']=[.9]*72
            r=dict(shadow=shadow,sweep_sequence=sequence,target=.2,shadow_window_stamp=sequence,
                   shadow_stalled=1.,shadow_worsening=0.,evaluated_angle=0.,speed=0.,goal_speed=0.,reason_name='stall_accumulate')
            stamp=100. if i<2 else 100.2
            check=dict(sensor=0,stamp=stamp,model_stamp=stamp-.1,predicted=.1,
                       observed=.11,error=.01,sigma=.002,support=.9,valid=1.)
            tg=[0.]*56;tg[10]=1.
            snapshots.append(dict(steady_time_s=100+i*.1,ros_time_s=100+i*.1,tg=tg,
                                  history=dict(response=r,theta_evaluated=0.,prediction_checks=[check])))
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);record=folder/'data.jsonl'
            record.write_text('\n'.join(json.dumps(dict(topic='/rmp_recording_snapshot',message=dict(data=[]))) for _ in snapshots))
            with patch('tg_sensor_shadow_audit.decode_snapshot',side_effect=snapshots):report=audit(record,folder/'out')
            self.assertEqual(report['scans'],2);self.assertEqual(report['validation_samples'],2)
            self.assertAlmostEqual(report['validation_by_sensor']['0']['mae_mm'],10.)
            with (folder/'out/candidates.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),144)
            with (folder/'out/observed_outcomes.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),2)
            self.assertTrue((folder/'out/sensor_evidence.png').exists())

if __name__=='__main__':unittest.main()

"""Decoder compatibility and recovery behavior for the history sidecar."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rmp_history_record import DECISION_FIELDS, PARAMETER_FIELDS, SAMPLE_FIELDS, RESPONSE_V7_FIELDS, decode_history, RecordingQuality


def packet(sequence=1, revision=1, keyframe=True, sensor_count=2):
    tg = [0.] * (114 + 8 * sensor_count)
    tg[26] = 9.; tg[50] = sensor_count; tg[98 + 4 * sensor_count] = 1.
    fields = dict(decision_sequence=sequence, events=257, theta_evaluated=.3, theta_after=.31,
                  risk_gradient_before=4., history_gradient=-4.)
    payload = [fields.get(key, 0.) for key in DECISION_FIELDS] + [0.] * 31
    payload += [1., .3, .1, .2, -.4, -.1]  # one initialization candidate
    payload += [0.] * len(PARAMETER_FIELDS)
    payload += [sensor_count] + [0.] * (sensor_count * len(SAMPLE_FIELDS))
    payload += [int(keyframe), revision, sensor_count if keyframe else 0]
    if keyframe:
        for sensor in range(sensor_count):
            payload += [sensor, 1., 100., 10., .2, -.1, .002]
            payload += [0.] * 6 + [1., 2., 3., 4., 5., 6.] + [0.] * 36
    return tg + [45045., 1., len(payload)] + payload


class HistoryTraceTest(unittest.TestCase):
    def test_block_reference_v2_and_v1_compatibility(self):
        old = packet()
        self.assertEqual(decode_history(old)['trace_version'], 1)
        self.assertNotIn('block_reference_vector', decode_history(old))
        new = old.copy()
        offset = 114 + 8 * int(new[50])
        new[offset + 1] = 2
        new[offset + 2] += 12
        new += [1., 1., .001] + [.1, .2, .3, .4, .5, .6] + [-.2, .1, .05]
        decoded = decode_history(new)
        self.assertEqual(decoded['block_reference_source'], 1)
        self.assertTrue(decoded['block_reference_valid'])
        self.assertEqual(decoded['goal_progress_joint_motion'], [.1, .2, .3, .4, .5, .6])
        self.assertEqual(decoded['block_reference_vector'], [-.2, .1, .05])
        new[-12] = 2.0
        self.assertEqual(decode_history(new)["block_reference_source"], 2)
        with self.assertRaises(ValueError):
            decode_history(new[:-1])

    def test_escape_v3_preserves_old_seed_components(self):
        tg=packet();offset=114+8*int(tg[50])
        tg[offset+1]=3
        tg += [2.,1.,.001]+[0.]*9
        tg += [.2,.01,.1,.4,.5,.6,1.,.12]
        tg[offset+2]=len(tg)-offset-3
        decoded=decode_history(tg)
        self.assertEqual(decoded['escape_gradient'],.4)
        self.assertEqual(decoded['seeds'][0]['escape_score'],.12)
        self.assertEqual(decoded['seeds'][0]['negative_history_cost'],-.4)
        with self.assertRaises(ValueError): decode_history(tg[:-1])

    def test_response_v4_is_distinct_from_predicted_score(self):
        tg=packet();offset=114+8*int(tg[50]);tg[offset+1]=4
        tg += [2.,1.,.001]+[0.]*9
        tg += [.2,.01,.1,.4,.5,.6,1.,.12]
        tg += [2.,1.,.1,.05,.02,.01,0.,100.,.03,1.]
        tg[offset+2]=len(tg)-offset-3
        decoded=decode_history(tg)
        self.assertEqual(decoded['response']['phase'],2)
        self.assertEqual(decoded['response']['speed'],.02)
        self.assertEqual(decoded['seeds'][0]['escape_score'],.12)

    def test_wide_search_v5_roundtrip_and_truncation(self):
        tg=packet();offset=114+8*int(tg[50]);tg[offset+1]=5
        tg += [2.,1.,.001]+[0.]*9
        tg += [.2,.01,.1,.4,.5,.6,1.,.12]
        tg += [1.,2.,1.57,0.,-.02,-.01,0.,100.,.03,1.]
        tg += [0.,0.,1.,3.,-3.,.6,1.57,200.,100.,0.,0.]+list(range(37))
        tg[offset+2]=len(tg)-offset-3
        r=decode_history(tg)['response']
        self.assertEqual(r['sweep_sequence'],3)
        self.assertEqual(r['sweep_scores'],list(range(37)))
        with self.assertRaises(ValueError):decode_history(tg[:-1])

    def test_full_circle_v6_components_and_confidence(self):
        tg=packet();offset=114+8*int(tg[50]);tg[offset+1]=6
        tg += [2.,1.,.001]+[0.]*9
        tg += [.2,.01,.1,.4,.5,.6,1.,.12]
        tg += [1.,2.,1.57,0.,-.02,-.01,0.,100.,.03,1.]
        tg += [0.,0.,1.,3.,-.3,.6,1.57,200.,100.,0.,0.]+list(range(72))
        tg += [.25,-.01,2.,0.,1.57,2.]+[.1]*72+[.2]*72+[.3]*72+[.4]*72
        tg[offset+2]=len(tg)-offset-3
        r=decode_history(tg)['response']
        self.assertEqual(r['sweep_scores'],list(range(72)))
        self.assertEqual(r['observation_confidence'],.25)
        self.assertEqual(r['sweep_goal'],[.1]*72)
        self.assertEqual(r['sweep_continuity'],[.4]*72)
        with self.assertRaises(ValueError):decode_history(tg[:-1])

    def test_fields_and_full_gradient(self):
        decoded = decode_history(packet())
        self.assertEqual(decoded['risk_gradient_before'], 4.)
        self.assertEqual(decoded['event_names'], ['initial_search', 'initial_seed_applied'])
        self.assertEqual(decoded['models'][1]['gradient'], [1., 2., 3., 4., 5., 6.])
        self.assertEqual(decoded['seeds'][0]['total_score'], -.1)

    def test_v8_shadow_diagnostics_and_prefit_checks(self):
        tg=packet();offset=114+8*int(tg[50]);tg[offset+1]=8
        tg += [2.,1.,.001]+[0.]*9
        tg += [.2,.01,.1,.4,.5,.6,1.,.12]
        tg += [0.]*476
        for value in (.1,.2,.3,.9,0.,-.004,.002):tg += [value]*72
        tg += [10.,1.,0.]+[2.]+[10.,9.,.1,.11,.01,.002,.8,1.]*2+[525.]
        tg[offset+2]=len(tg)-offset-3
        h=decode_history(tg)
        self.assertEqual(h['response']['shadow']['nominal_cost'],[.1]*72)
        self.assertEqual(h['prediction_checks'][1]['error'],.01)
        self.assertLess(h['prediction_checks'][0]['model_stamp'],h['prediction_checks'][0]['stamp'])
        with self.assertRaises(ValueError):decode_history(tg[:-1])
        tg[-1]=521.
        with self.assertRaises(ValueError):decode_history(tg)

    def test_v9_command_opportunity_and_net_progress_roundtrip(self):
        tg=packet();offset=114+8*int(tg[50]);tg[offset+1]=9
        tg += [2.,1.,.001]+[0.]*9
        tg += [.2,.01,.1,.4,.5,.6,1.,.12]+[0.]*476
        tg += [0.]*(7*72)+[10.,1.,0.]+[2.]+[0.]*16+[525.]
        tg += [.9,.03,1.,0.,.01,.014,.65,.714,10.,1.,.5]+[.01]*6
        tg[offset+2]=len(tg)-offset-3
        h=decode_history(tg);r=h['response']
        self.assertEqual(h['trace_version'],9)
        self.assertEqual(r['command_support'],.9)
        self.assertEqual(r['net_travel'],.01)
        self.assertEqual(r['command_qd'],[.01]*6)
        with self.assertRaises(ValueError):decode_history(tg[:-1])

    def test_v10_authority_boost_and_v9_backward_compatibility(self):
        tg=packet();offset=114+8*int(tg[50]);tg[offset+1]=9
        tg += [2.,1.,.001]+[0.]*9
        tg += [.2,.01,.1,.4,.5,.6,1.,.12]+[0.]*476
        tg += [0.]*(7*72)+[10.,1.,0.]+[2.]+[0.]*16+[525.]
        tg += [.9,.03,1.,0.,.01,.014,.65,.714,10.,1.,.5]+[.01]*6
        tg[offset+2]=len(tg)-offset-3
        self.assertNotIn('authority_boost',decode_history(tg)['response'])
        tg[offset+1]=10
        tg += [100.,1.5,1.5,2.,.5,0.,1.,0.,1.,2.,1.,1.,.36,.36]
        tg[offset+2]=len(tg)-offset-3
        boost=decode_history(tg)['response']['authority_boost']
        self.assertEqual(boost['applied_gain'],1.5)
        self.assertEqual(boost['base_metric'],100.)
        self.assertEqual(boost['phase_name'],'rising')
        with self.assertRaises(ValueError):decode_history(tg[:-1])

    def test_v7_observed_failure_memory_and_reason_preserve_v6_layout(self):
        tg=packet();offset=114+8*int(tg[50]);tg[offset+1]=7
        tg += [2.,1.,.001]+[0.]*9
        tg += [.2,.01,.1,.4,.5,.6,1.,.12]
        tg += [4.,2.,.3,0.,-.02,-.01,0.,100.,.03,1.]
        tg += [0.,0.,1.,3.,-.3,.6,1.57,200.,100.,0.,0.]+[.5]*72
        tg += [.25,-.01,2.,0.,1.57,2.]+[.1]*72+[.2]*72+[.3]*72+[.4]*72
        fields=dict(reason=5.,turning=1.,turn_travel=.4,range_closing=.01,
                    paired_samples=2.,evaluated_angle=.2)
        tg += [fields.get(k,0.) for k in RESPONSE_V7_FIELDS]+[.15]*72
        tg[offset+2]=len(tg)-offset-3
        r=decode_history(tg)['response']
        self.assertEqual(r['reason_name'],'range_worsening_accumulate')
        self.assertEqual(r['phase'],4)  # v7 phase 4 does NOT freeze steering
        self.assertEqual(r['sweep_failure'],[.15]*72)
        self.assertEqual(r['sweep_scores'],[.5]*72)
        self.assertEqual(r['evaluated_angle'],.2)
        with self.assertRaises(ValueError):decode_history(tg[:-1])

    def test_old_tg_and_truncation(self):
        self.assertIsNone(decode_history(packet()[:130]))
        self.assertIsNone(decode_history([0.]))
        with self.assertRaises(ValueError):
            decode_history(packet()[:-1])

    def test_corrupt_direction_update_is_reported(self):
        tg=packet();tg[0]=1.
        trace=decode_history(tg)
        trace.update(dt=.005,filtered_rate_before=0.,clamped_rate=8.,filtered_rate_after=.4,
                     theta_evaluated=.3,theta_after=.302)
        trace['effective_parameters']['direction_filter_tau']=.095
        quality=RecordingQuality()
        quality.observe(dict(control_sequence=1,tg=tg,history=trace))
        self.assertEqual(quality.counts['direction_update_mismatches'],0)
        trace['theta_after']=.5
        quality.observe(dict(control_sequence=2,tg=tg,history=trace))
        self.assertEqual(quality.counts['direction_update_mismatches'],1)

    def test_loss_requires_keyframe_and_old_history_is_flagged(self):
        quality = RecordingQuality()
        def observe(sequence, revision, keyframe):
            tg = packet(sequence, revision, keyframe)
            quality.observe(dict(control_sequence=sequence, tg=tg, history=decode_history(tg)))
        observe(1, 1, True)
        observe(3, 2, False)
        observe(4, 3, True)
        quality.observe(dict(control_sequence=5,tg=packet()[:130],history=None))
        result = quality.summary()
        self.assertEqual(result['sequence_gaps'], 1)
        self.assertEqual(result['history_without_complete_models'], 1)
        self.assertEqual(result['history_missing_trace'], 1)
        self.assertTrue(quality.models_complete)


if __name__ == '__main__':
    unittest.main()

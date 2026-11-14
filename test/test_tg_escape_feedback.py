import unittest
import numpy as np
from tg_escape_feedback import EscapeFeedback

class FeedbackTest(unittest.TestCase):
    def snapshot(self,t,d=.04):
        tg=[0.]*56;tg[1]=0;tg[10]=100;tg[20]=1
        J=np.zeros((3,6));J[:3,:3]=np.eye(3)
        return dict(ros_time_s=t,tg=tg,visualization_points=[[0,0,0,.01]+J.ravel().tolist()],
                    goal_pose=[1,0,0,0,0,0,1],history=dict(predicted_displacement=[.004,0,0,0,0,0],
                    samples=[dict(acquisition_stamp=t,raw_equivalent_range=d)],
                    effective_parameters=dict(clearance=.08,risk_length=.05,fresh_time=.15)))
    def run_motion(self,dx,d=.04):
        monitor=EscapeFeedback()
        monitor.encoder(10,[0]*6);monitor.observe(self.snapshot(10))
        monitor.encoder(10.13,[dx,0,0,0,0,0])
        return monitor.observe(self.snapshot(10.13,d))
    def test_missing_encoder_is_unknown(self):
        self.assertEqual(EscapeFeedback().observe(self.snapshot(10))['status'],'encoder_unavailable')
    def test_stall_and_reverse_are_not_progress(self):
        self.assertEqual(self.run_motion(0)['status'],'no_directional_progress')
        self.assertEqual(self.run_motion(-.004)['status'],'reverse')
    def test_progress_requires_range_evidence(self):
        self.assertEqual(self.run_motion(.004)['status'],'progress_observed')
        self.assertEqual(self.run_motion(.004,.005)['status'],'range_worsening')
    def test_prediction_error_uses_encoder_motion(self):
        r=self.run_motion(.001)
        self.assertAlmostEqual(r['motion_prediction_error_m'],-.003)

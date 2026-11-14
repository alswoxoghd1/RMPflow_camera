import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tg_direction_audit import seed_choice, seed_frame, seed_rows, direction, load_audit, contribution_episodes, contextual_seed

class AuditTests(unittest.TestCase):
    def test_tie_matches_controller_order_and_tolerance(self):
        self.assertEqual(seed_choice([{'total_score':1.},{'total_score':1.+5e-10}]),0)
        self.assertEqual(seed_choice([{'total_score':1.},{'total_score':1.+2e-9}]),1)

    def test_no_future_candidate_scores_at_cursor(self):
        rows=[dict(time_s=2.,seed_applied=True,seeds=[{}])]
        self.assertIsNone(seed_frame(rows,1.))
        self.assertIs(seed_frame(rows,2.),rows[0])

    def test_weighted_scores_and_negative_cost_not_reweighted(self):
        seeds=[dict(theta=a,goal_score=g,continuity_score=v,negative_history_cost=-c,total_score=g+v-c)
               for a,g,v,c in [(0,.5,.1,.2),(math.pi,-.5,0,.1)]]
        rows=seed_rows(dict(time_s=2.,sequence=20,owner=5,seeds=seeds))
        self.assertTrue(rows[0]['selected']);self.assertAlmostEqual(rows[0]['total_score'],.4)
        self.assertAlmostEqual(rows[0]['sum_error'],0.)

    def test_sensor_basis_rotates_candidate_vectors(self):
        tg=[0.]*150;tg[50]=8;k=76+4*8
        tg[k+4:k+7]=[0,1,0];tg[k+7:k+10]=[0,0,1]
        np.testing.assert_allclose(direction(tg,math.pi/2),[0,0,1],atol=1e-12)

    def test_windows_start_at_contribution_not_detection(self):
        rows=[dict(time_s=float(i),active=True,contributing=on,owner=5)
              for i,on in enumerate([False,False,True,True,False,True])]
        self.assertEqual([e['start'] for e in contribution_episodes(rows)],[2.,5.])
        self.assertEqual(contribution_episodes([dict(time_s=0.,active=True,contributing=False,owner=5)]),[])

    def test_initial_score_reference_does_not_cross_reset(self):
        seed=dict(time_s=1.,active=True,seed_applied=True,seeds=[{}])
        off=dict(time_s=2.,active=False,seed_applied=False,seeds=[])
        on=dict(time_s=3.,active=True,seed_applied=False,seeds=[])
        self.assertIs(contextual_seed([seed,off,on],1.5),seed)
        self.assertIsNone(contextual_seed([seed,off,on],3.))

    def test_full_circle_plot_uses_recorded_components_and_filtered_scores(self):
        from matplotlib.figure import Figure
        from tg_direction_audit import draw_figure
        r=dict(time_s=1.,contributing=True,owner=6,seed_applied=False,events=[],
               theta_rad=.2,block=.8,raw_rate=.2,clamped_rate=.2,filtered_rate=.1)
        for k in ('goal_gradient','motion_gradient','history_gradient','escape_gradient','total_gradient'):r[k]=0.
        r['response']=dict(sweep_sequence=1,target=.5,sweep_goal=[.1]*72,
            sweep_execution=[.2]*72,sweep_risk=[.05]*72,sweep_continuity=[.03]*72,
            sweep_scores=[.25]*72,observation_confidence=.5,poor_evidence=2,progress_evidence=0)
        fig=Figure();axes=draw_figure(fig,[r],now=1.)
        self.assertEqual(axes[0,1].get_ylabel(),'Dimensionless score')
        total=next(line for line in axes[0,1].lines if line.get_label()=='Guidance score')
        np.testing.assert_allclose(total.get_ydata(),[.28]*72)
        r['response'].update(sweep_failure=[.2]*72,reason_name='stall_accumulate')
        axes=draw_figure(fig,[r],now=1.)
        total=next(line for line in axes[0,1].lines if line.get_label()=='Guidance score')
        np.testing.assert_allclose(total.get_ydata(),[.08]*72)
        fig.savefig('/tmp/tg360_plot_smoke.png')

    def test_latest_recording_uses_metric_and_observe_flag(self):
        # A source fixture is provided by the workspace integration test, not a fixed user path.
        import os
        path=os.environ.get('TG_AUDIT_TEST_RECORDING')
        if not path:self.skipTest('no integration recording supplied')
        frames,episodes,quality=load_audit(path)
        self.assertTrue(episodes)
        active=[r for r in frames if r['active']]
        self.assertTrue(any(r['contributing'] for r in active))
        for r in active:
            self.assertEqual(r['contributing'],r['metric']>1e-9 and r['tg'][44]<.5)
            if r['contributing']:self.assertLess(r['direction_error'],1e-8)
            self.assertLess(r['gradient_error'],1e-8)
        self.assertTrue(any(e['first_contribution'] is not None for e in episodes))

if __name__=='__main__':unittest.main()

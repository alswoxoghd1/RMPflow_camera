import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rmp_rviz_replay import load_frames, pose_at, group_contributions, Replay, rclpy


def snapshot(sequence, full):
    tg = [0.] * 122
    tg[26], tg[50], tg[102] = 9., 1., 1.
    ablation = []
    if full:
        ablation = [0.] * 25 + [42042., 0., 44044., 0.] + [0.] * 24
        ablation += [43043.] + np.eye(6).ravel().tolist() + [0.] * 12
    return ([1., sequence, 100., 100.+sequence*.02, 1., 0.] + [0.] * 42 + [0.] * 7 +
            [0.] + [0.] * 15 + [0., len(ablation)] + ablation + [len(tg)] + tg)


class ReplayTest(unittest.TestCase):
    def test_compact_samples_remain_on_pose_timeline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.details.jsonl'
            with path.open('w') as stream:
                for seq, full in [(1, True), (2, False), (3, False), (4, True)]:
                    stream.write(json.dumps(dict(topic='/rmp_recording_snapshot',
                                                  message=dict(data=snapshot(seq, full))))+'\n')
            frames, info = load_frames(str(path))
        self.assertEqual(len(frames), 4)
        self.assertEqual(info['detail_frames'], 2)
        self.assertEqual([f['detail_index'] for f in frames], [0, 0, 0, 3])

    def test_interpolation_does_not_bridge_long_recording_gaps(self):
        frames = [dict(solver_q=[q]*6) for q in (0., 1., 3.)]
        times = [0., .02, .4]
        self.assertAlmostEqual(pose_at(frames, times, .01)[1][0], .5)
        self.assertEqual(pose_at(frames, times, .2)[1][0], 1.)
        self.assertEqual(pose_at(frames, times, .01, False)[1][0], 0.)

    def test_cached_group_solve_matches_full_metric_decomposition(self):
        matrix = np.diag([2.,3.,4.,5.,6.,7.])
        groups = [dict(group_id=gid, force=[gid*1.]*6, metric=matrix.ravel().tolist()) for gid in (4,5,6)]
        frame = dict(root_metric=matrix.ravel().tolist(), groups=groups)
        result = group_contributions(frame, .001)
        for gid in (4,5,6):
            expected = np.linalg.lstsq(matrix+.001*np.eye(6), np.full(6,gid), rcond=None)[0]
            np.testing.assert_allclose(result[gid][0], expected, atol=1e-12)
        self.assertIs(group_contributions(frame, .001), result)

    def test_paused_scene_does_not_rebuild_and_toggled_arrows_are_deleted(self):
        import rmp_rviz_replay as module
        from unittest.mock import patch
        frames = [dict(time=0., solver_q=[0.]*6, detail_index=-1, tg_row=dict(
            active='1', cp_x='.1', cp_y='.2', cp_z='.3', t_x='1', t_y='0', t_z='0',
            metric='1', sensor_index='0', direction_mode='1'))]
        rclpy.init()
        try:
            with patch.object(module, 'load_frames', return_value=(frames,dict(dataset='test',detail_frames=0))):
                node = Replay()
            class Sink:
                def __init__(self): self.messages=[]
                def publish(self, msg): self.messages.append(msg)
            sink = Sink()
            node.markers = sink
            node.tick(); node.tick()
            self.assertEqual(len(sink.messages), 1)
            self.assertFalse(any(m.action==3 for m in sink.messages[0].markers))
            node.show['direction'] = False
            node.tick()
            removed = [m.ns for m in sink.messages[-1].markers if m.action==2]
            self.assertIn('direction/S00', removed)
            self.assertIn('direction/label/S00', removed)
            node.destroy_node()
        finally:
            rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()

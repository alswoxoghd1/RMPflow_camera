import importlib.util
from pathlib import Path
import csv
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'tg_rmp_html_monitor.py'
SPEC = importlib.util.spec_from_file_location('tg_html', SCRIPT)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def sample(**changes):
    row = dict(t='1', active='1', observe_only='0', metric='10', direction_mode='3',
               leaf_jacobian_norm='2', root_metric_norm='40', root_force_norm='5',
               steering_fallback='0', steering_speed='.1', steering_velocity_dot='0',
               t_x='1', t_y='0', t_z='0', sensor_index='0')
    row.update(changes)
    return row


class MonitorAudit(unittest.TestCase):
    def test_empty_does_not_report_success(self):
        result = monitor.audit([], {})
        self.assertEqual(result['samples'], 0)
        self.assertIsNone(result['max_abs_cos_velocity'])
        self.assertIsNone(result['max_metric_identity_relative_error'])

    def test_metric_pullback_and_velocity_checks_detect_bad_data(self):
        good = monitor.audit([sample()], {})
        self.assertEqual(good['applied_samples'], 1)
        self.assertEqual(good['max_metric_identity_relative_error'], 0)
        bad = monitor.audit([sample(root_metric_norm='20', steering_velocity_dot='.05')], {})
        self.assertEqual(bad['max_abs_cos_velocity'], .5)
        self.assertGreater(bad['max_metric_identity_relative_error'], .1)

    def test_observe_only_and_startup_are_excluded_from_claims(self):
        result = monitor.audit([sample(observe_only='1')], {})
        self.assertEqual(result['applied_samples'], 0)
        for update in (dict(steering_fallback='1'), dict(direction_mode='1')):
            result = monitor.audit([sample(**update)], {})
            self.assertEqual(result['orthogonality_samples'], 0)

    def test_gap_and_owner_changes_do_not_become_turn_jumps(self):
        result = monitor.audit([sample(), sample(t='1.01', sensor_index='1', t_x='-1'),
                                sample(t='2', sensor_index='1')], {})
        self.assertEqual(result['owner_switches'], 1)
        self.assertEqual(result['logging_gap_count'], 1)
        self.assertIsNone(result['max_world_turn_rate'])

    def test_offline_html_embeds_plotly_and_marks_missing_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'data.csv'
            path.touch()
            output = Path(folder)/'report.html'
            monitor.render(path, output, {})
            text = output.read_text()
            self.assertIn('TG 진단 샘플 대기', text)
            self.assertIn('plotly.js', text)
            self.assertNotIn('src="https://cdn.plot.ly', text)
            with path.open('w', newline='') as stream:
                row = sample()
                writer = csv.DictWriter(stream, fieldnames=row.keys())
                writer.writeheader(); writer.writerow(row)
            monitor.render(path, output, {'parameters': {'real_joint_state_source': 'reference'}})
            self.assertIn('reference', output.read_text())
            self.assertEqual(monitor.read_rows(path)[0]['metric'], '10')


if __name__ == '__main__':
    unittest.main()

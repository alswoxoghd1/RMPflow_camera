#!/usr/bin/env python3
"""Schema for the append-only TG history decision trace (45045, versions 1--12).

The enclosing control snapshot supplies the control sequence and timestamps.
Model updates are deltas; full keyframes permit recovery after packet loss.
"""
DECISION_FIELDS = (
    'decision_sequence dt events active_before theta_valid_before owner_before '
    'closest_sensor closest_clearance owner_clearance theta_before theta_evaluated theta_after '
    'filtered_rate_before raw_rate clamped_rate filtered_rate_after handoff_before handoff_after '
    'goal_gradient continuity_gradient history_gradient total_gradient '
    'risk_before risk_gradient_before measured_risk_before unknown_risk_before '
    'min_predicted_range_before confidence_before model_count_before unknown_count_before '
    'critical_sensor_before have_measurements observation_stamp source_available velocity_available'
).split()
PARAMETER_FIELDS = (
    'window horizon clearance risk_length weight_goal weight_risk weight_continuity '
    'unknown_motion_weight max_direction_rate direction_filter_tau handoff_time initial_angle '
    'fresh_time expiry sample_interval ridge noise_floor max_joint_step '
    'direction_rate owner_hysteresis sensor_hold_time root_solve_offset'
).split()
SAMPLE_FIELDS = ('acquisition_stamp raw_equivalent_range used_clearance used_rate held missing_elapsed '
                 'model_update_reason model_update_stamp').split()
SEED_FIELDS = 'theta goal_score continuity_score negative_history_cost total_score'.split()
EVENT_BITS = {
    1: 'initial_search', 2: 'owner_change_requested', 4: 'owner_changed',
    8: 'handoff_fade', 16: 'reset_no_observations', 32: 'reset_outside_release',
    64: 'disabled', 128: 'singular_handoff_projection', 256: 'initial_seed_applied',
}
RESPONSE_V7_FIELDS = ('reason turning turn_travel turn_elapsed reverse_count range_closing '
                     'min_range adverse_evidence retreat_evidence positive_score negative_score '
                     'sign_margin window_dt paired_samples turn_rate_limit failure_weight evaluated_angle').split()
RESPONSE_REASONS = {
    0:'observing', 1:'measured_motion', 2:'stall_start', 3:'stall_accumulate',
    4:'range_worsening_start', 5:'range_worsening_accumulate',
    6:'persistent_alternative_reverse', 7:'failed_full_circle_reverse',
    8:'block_inactive', 9:'missing_feedback', 10:'invalid_candidates', 11:'execution_limited', 12:'unexecuted_search', 13:'authority_trial',
}
SHADOW_FIELDS = ('nominal_cost uncertainty_cost unknown_exposure support '
                 'critical_sensor critical_change critical_sigma').split()
PREDICTION_CHECK_FIELDS = 'stamp model_stamp predicted observed error sigma support valid'.split()


def decode_history(tg):
    if len(tg) < 56 or int(tg[26]) != 9:
        return None
    offset = 114 + 8 * int(tg[50])
    if len(tg) <= offset:
        return None
    if len(tg) < offset + 3 or tg[offset] != 45045 or tg[offset + 1] not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
        raise ValueError('unsupported TG history trace')
    version = int(tg[offset + 1])
    count = int(tg[offset + 2])
    data = [float(v) for v in tg[offset + 3:]]
    if len(data) != count:
        raise ValueError('truncated TG history trace')
    cursor = 0

    def take(size):
        nonlocal cursor
        if size < 0 or cursor + size > len(data):
            raise ValueError('truncated TG history field')
        result = data[cursor:cursor + size]
        cursor += size
        return result

    result = dict(zip(DECISION_FIELDS, take(len(DECISION_FIELDS))))
    result['event_names'] = [name for bit, name in EVENT_BITS.items() if int(result['events']) & bit]
    for name in ('observation_q', 'observation_qd', 'nominal_qdd', 'filtered_nominal_qdd'):
        result[name] = take(6)
    result['predicted_displacement'] = take(6)
    result['unclipped_displacement_norm'] = take(1)[0]
    result['seeds'] = [dict(zip(SEED_FIELDS, take(5))) for _ in range(int(take(1)[0]))]
    result['effective_parameters'] = dict(zip(PARAMETER_FIELDS, take(len(PARAMETER_FIELDS))))
    result['samples'] = [dict(zip(SAMPLE_FIELDS, take(len(SAMPLE_FIELDS)))) for _ in range(int(take(1)[0]))]
    keyframe, revision, count = take(3)
    result.update(model_keyframe=bool(keyframe), model_revision=int(revision), models=[])
    for _ in range(int(count)):
        model = dict(zip(('sensor', 'valid', 'stamp', 'count', 'range', 'rate', 'noise'), take(7)))
        model['anchor_q'] = take(6)
        model['gradient'] = take(6)
        model['information_inverse'] = take(36)
        result['models'].append(model)
    result['trace_version'] = version
    if version >= 2:
        source, valid, epsilon = take(3)
        result['block_reference_source'] = int(source)
        result['block_reference_valid'] = bool(valid)
        result['block_reference_epsilon'] = epsilon
        result['goal_progress_joint_motion'] = take(6)
        result['block_reference_vector'] = take(3)
    if version >= 3:
        names = ("weight_escape", "escape_length", "reselect_interval", "escape_gradient",
                 "escape_before", "escape_after")
        result.update(zip(names, take(6)))
        scores = take(int(take(1)[0]))
        for seed, score in zip(result["seeds"], scores):
            seed["escape_score"] = score
    if version >= 4:
        result['response'] = dict(zip(('phase','result','target','accepted','speed','baseline',
                                      'failures','encoder_stamp','best_speed','seeded'),take(10)))
    if version >= 5:
        result['response'].update(zip(('anchor','has_accepted','sign','sweep_sequence',
            'current_score','best_score','chosen_angle','sweep_us','scan_metric','tried','bound_clamped'),take(11)))
        result['response']['sweep_scores']=take(72 if version>=6 else 37)
    if version >= 6:
        r=result['response']
        r.update(zip(('observation_confidence','goal_speed','poor_evidence',
            'progress_evidence','pending_angle','pending_count'),take(6)))
        for name in ('sweep_goal','sweep_execution','sweep_risk','sweep_continuity'):
            r[name]=take(72)
    if version >= 7:
        r=result['response']
        r.update(zip(RESPONSE_V7_FIELDS,take(len(RESPONSE_V7_FIELDS))))
        r['reason_name']=RESPONSE_REASONS.get(int(r['reason']),'unknown')
        r['sweep_failure']=take(72)
    if version >= 8:
        begin=cursor
        result['response']['shadow']={name:take(72) for name in SHADOW_FIELDS}
        result['response'].update(zip(('shadow_window_stamp','shadow_stalled','shadow_worsening'),take(3)))
        n=int(take(1)[0])
        if n!=int(tg[50]): raise ValueError('shadow sensor count mismatch')
        result['prediction_checks']=[dict(sensor=i,**dict(zip(PREDICTION_CHECK_FIELDS,take(8)))) for i in range(n)]
        length=take(1)[0]
        if length!=cursor-begin: raise ValueError('shadow length mismatch')
    if version >= 9:
        result['response'].update(zip(('command_support','command_speed','execution_supported',
            'unsupported_elapsed','net_travel','absolute_travel','progress_duration','progress_ratio',
            'command_stamp','command_available','metric_rate_floor'),take(11)))
        result['response']['command_qd']=take(6)
    if version >= 10:
        result['response']['authority_boost']=dict(zip((
            'base_metric','applied_gain','state_gain','phase','deficit_time','cap_time',
            'command_coverage','exhausted','enabled','maximum','rise_rate','decay_rate',
            'trigger_time','settle_time'),take(14)))
        boost=result['response']['authority_boost']
        boost['phase_name']={0:'idle',1:'waiting',2:'rising',3:'settling',4:'exhausted',
            5:'supported',6:'recovering',7:'danger',8:'searching'}.get(int(boost['phase']),'unknown')
    if version >= 11:
        result['weight_neighbor_risk'] = take(1)[0]
        names = ('weighted_cost', 'alignment', 'closeness', 'detection_count',
                 'critical_sensor', 'range', 'distance_limit') if version >= 12 else (
                 'weighted_cost', 'support', 'model_count', 'unknown_count',
                 'critical_sensor', 'predicted_change', 'sigma')
        result['response']['neighbor_risk'] = {name: take(72) for name in names}
        result['response']['neighbor_risk']['method'] = (
            'current_range_mounting_alignment' if version >= 12 else 'validated_local_regression')
    if cursor != len(data):
        raise ValueError('unexpected TG history fields')
    return result


class RecordingQuality:
    """Track actual received data, including unrecoverable model intervals."""
    def __init__(self):
        self.last_sequence = None
        self.last_decision = None
        self.last_revision = None
        self.models_complete = False
        self.counts = dict(snapshots=0, sequence_gaps=0, sequence_resets=0,
                           history_frames=0, history_missing_trace=0, model_keyframes=0,
                           history_without_complete_models=0, decode_errors=0,
                           direction_update_mismatches=0)
        self.events = {}
        self.modes = {}

    def observe(self, snapshot):
        sequence = int(snapshot['control_sequence'])
        if self.last_sequence is not None:
            gap = sequence - self.last_sequence
            if gap <= 0:
                self.counts['sequence_resets'] += 1
                self.models_complete = False
                self.last_decision = self.last_revision = None
            elif gap > 1:
                self.counts['sequence_gaps'] += gap - 1
        self.last_sequence = sequence
        self.counts['snapshots'] += 1
        tg = snapshot.get('tg', [])
        if len(tg) < 56:
            return
        index = 98 + 4 * int(tg[50])
        if index >= len(tg):
            return
        mode = int(tg[index])
        self.modes[str(mode)] = self.modes.get(str(mode), 0) + 1
        if mode != 1:
            self.models_complete = False
            self.last_decision = self.last_revision = None
            return
        trace = snapshot.get('history')
        if trace is None:
            self.counts['history_missing_trace'] += 1
            return
        self.counts['history_frames'] += 1
        if tg[0] > .5:
            import math
            dt=trace['dt']; tau=trace['effective_parameters']['direction_filter_tau']
            if not math.isfinite(dt) or dt<=0 or not math.isfinite(tau):
                raise ValueError('invalid history integration timing')
            expected_rate=trace['filtered_rate_before'] + dt/(max(tau,0.)+dt) * (
                trace['clamped_rate']-trace['filtered_rate_before'])
            angle_error=math.remainder(trace['theta_after']-trace['theta_evaluated']-
                                       trace['filtered_rate_after']*dt,2*math.pi)
            bounded=bool(trace.get('response',{}).get('bound_clamped',False))
            if not bounded and (not math.isfinite(expected_rate) or not math.isfinite(angle_error) or
                    abs(expected_rate-trace['filtered_rate_after'])>1e-9 or abs(angle_error)>1e-9):
                self.counts['direction_update_mismatches']+=1
        decision, revision = int(trace['decision_sequence']), trace['model_revision']
        if self.last_decision is not None and decision != self.last_decision + 1:
            self.models_complete = False
        if self.last_revision is not None and revision not in (self.last_revision, self.last_revision + 1):
            self.models_complete = False
        if trace['model_keyframe']:
            expected = set(range(int(tg[50])))
            self.models_complete = {int(m['sensor']) for m in trace['models']} == expected
            self.counts['model_keyframes'] += 1
        if not self.models_complete:
            self.counts['history_without_complete_models'] += 1
        self.last_decision, self.last_revision = decision, revision
        for event in trace['event_names']:
            self.events[event] = self.events.get(event, 0) + 1

    def summary(self):
        return dict(**self.counts, modes=dict(self.modes), event_counts=dict(self.events))


SCHEMA = dict(v12_note="Neighbor cost uses current ranges and world-frame mounting-axis alignment only; seven arrays: weighted cost, alignment, closeness, detection count, critical sensor, range, distance limit. Historical regression shadow fields are diagnostic only.", v11_note="No pocket/ownership fades; goal-blocking owner eligibility. Append neighbor-risk weight then seven 72-candidate arrays: weighted cost, support, model count, unknown count, sensor, change, sigma. Existing shadow fields keep their old diagnostic meaning.", v10_note="Bounded authority boost (trigger semantics depend on archived source; current trigger is rotation); base/applied metric and observation state appended after v9.", magic=45045, version=12, supported_versions=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              v9_note="Post-guard submitted command coverage and rolling encoder net progress. Unexecuted search does not penalize headings. Forecast costs are diagnostic by default; compare effective weight_risk. History-only metric-rate floor is logged.", decision_fields=DECISION_FIELDS,
              v8_note='Observation-only diagnostics; controller scores and direction decisions unchanged. Seven arrays of72 raw (unweighted) candidate diagnostics, window stamp/stall/range-worsening, sensor count, eight fields per pre-fit sample validation, extension length. Sigma is a model/noise scale, NOT a calibrated confidence interval. Validation uses the configured model pose source, not an unexecuted candidate trajectory.',
              v8_shadow_fields=SHADOW_FIELDS, v8_prediction_check_fields=PREDICTION_CHECK_FIELDS,
              v7_note='Model scores select provisional rotation sign only. Observed failure advances angle cumulatively; sustained motion retains the applied heading. Phase 4 steers during worsening range response (no frozen search). Candidate guidance = filtered physical + continuity - observed failure cost, not a destination or safety certificate.',
              v7_response_fields=RESPONSE_V7_FIELDS, v7_response_reasons=RESPONSE_REASONS,
              v6_note='72 candidates at absolute sensor-plane angles 0:5:355 degrees. Dimensionless filtered goal+execution-risk scores; continuity added separately. Measured confidence and progress govern update persistence, not stop-and-retry gates.',
              v5_note='Same-state +/-90 degree acceleration search at 5 degree spacing. 37 sweep scores in m/s^2, offsets relative to anchor. Accepted angle validity is explicit; sweep_us measures search cost.',
              v4_note='Encoder response state machine replaces continuous predicted-score ascent. escape_gradient records the correction from score gradient to feedback angular command.',
              v2_tail_fields=['block_reference_source', 'block_reference_valid',
                              'block_reference_epsilon', 'goal_progress_joint_motion[6]',
                              'block_reference_vector[3]'],
              block_reference_note='source 2 uses goal-progress angle-linear activation (start angle in controller parameters); source 1 uses goal-progress dot smoothstep; source 0 uses nominal acceleration. Legacy prefix nominal fields retain their original meaning.',
              event_count_note='event_counts counts frames carrying each flag; fade/request flags can persist across multiple frames',
              vector_fields=['observation_q[6]','observation_qd[6]','nominal_qdd[6]',
                             'filtered_nominal_qdd[6]','predicted_displacement[6]',
                             'unclipped_displacement_norm'],
              timing_note='dt is the configured integrator step; use enclosing snapshot steady_time_s for actual cycle spacing',
              layout='decision fields; vector fields; seed count and seeds; effective parameters; sensor count and samples; keyframe,model_revision,model_count; model records',
              effective_parameters=PARAMETER_FIELDS, sample_fields=SAMPLE_FIELDS,
              seed_fields=SEED_FIELDS, event_bits=EVENT_BITS,
              model_update_reasons={1:'accepted',2:'invalid_range_cleared',3:'expired',4:'discontinuity_reset',
                                    5:'pose_interpolation_unavailable',6:'out_of_range_or_stale',7:'interval_skip',8:'clock_reset'},
              model_fields=['sensor', 'valid', 'stamp', 'count', 'range', 'rate', 'noise',
                            'anchor_q[6]', 'gradient[6]', 'information_inverse_row_major[36]'])


def audit_file(path):
    import json
    from rmp_detail_capture import decode_snapshot
    quality=RecordingQuality()
    with open(path) as stream:
        for line in stream:
            item=json.loads(line)
            if item.get('topic')!='/rmp_recording_snapshot': continue
            try:
                quality.observe(decode_snapshot(item['message']['data']))
            except (ValueError,IndexError,TypeError,OverflowError,ZeroDivisionError):
                quality.counts['decode_errors']+=1
    return quality.summary()


if __name__ == '__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description='Audit a saved history recording without running ROS control.')
    parser.add_argument('dataset',help='Path to .details.jsonl')
    print(json.dumps(audit_file(parser.parse_args().dataset),ensure_ascii=False,indent=2))

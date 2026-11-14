#!/usr/bin/env python3
"""Read-only TG recorder and standalone interactive HTML report.

No controller settings, flags, goals or robot commands are published.
"""
import argparse
import csv
from datetime import datetime
import html
import json
import math
from pathlib import Path
import time
import statistics


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def read_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline='') as stream:
        reader = csv.DictReader(line for line in stream if not line.startswith('#'))
        if not {'t', 'active', 'metric'}.issubset(reader.fieldnames or []):
            raise ValueError('TG observation CSV required (t, active, metric columns)')
        return [row for row in reader if None not in row and all(v is not None for v in row.values())]


def audit(rows, metadata):
    def num(row, key):
        return finite(row.get(key))
    valid = [r for r in rows if num(r, 't') is not None]
    active = [r for r in valid if (num(r, 'active') or 0) > .5]
    applied = [r for r in active if num(r, 'observe_only') == 0
               and (num(r, 'metric') or 0) > 1e-12]
    modes = sorted({int(num(r, 'direction_mode')) for r in valid
                    if num(r, 'direction_mode') is not None})
    # In simple mode t is perpendicular to velocity only outside its startup branch.
    moving = [r for r in applied if num(r, 'direction_mode') == 3
              and num(r, 'steering_fallback') is not None
              and not (int(num(r, 'steering_fallback')) & 3)
              and (num(r, 'steering_speed') or 0) > 1e-8]
    cosines = [abs(num(r, 'steering_velocity_dot')) / num(r, 'steering_speed')
               for r in moving if num(r, 'steering_velocity_dot') is not None]
    identities = []
    invalid_metric = 0
    roundoff_negative = 0
    for r in active:
        m, j, root = (num(r, key) for key in ('metric', 'leaf_jacobian_norm', 'root_metric_norm'))
        if m is not None and m < -1e-9:
            invalid_metric += 1
        elif m is not None and m < 0:
            roundoff_negative += 1
        if m is not None and j is not None and root is not None:
            expected = m * j * j
            identities.append(abs(root - expected) / max(1., abs(root), abs(expected)))
    turns, steps, arrival_intervals, switches, gaps = [], [], [], 0, 0
    for a, b in zip(valid, valid[1:]):
        dt = num(b, 't') - num(a, 't')
        if dt > 0: arrival_intervals.append(dt)
        if not 0 < dt <= .1:
            gaps += 1
            continue
        if (num(a, 'active') or 0) <= .5 or (num(b, 'active') or 0) <= .5:
            continue
        if a.get('sensor_index') != b.get('sensor_index'):
            switches += 1
            continue
        vectors = [[num(r, 't_' + axis) for axis in 'xyz'] for r in (a, b)]
        if all(x is not None for v in vectors for x in v):
            norms = [math.sqrt(sum(x*x for x in v)) for v in vectors]
            if min(norms) > 1e-8:
                cosine = sum(x*y for x, y in zip(*vectors)) / (norms[0]*norms[1])
                angle = math.acos(max(-1., min(1., cosine)))
                turns.append(angle / dt)
                if all(num(r, 'observe_only') == 0 and (num(r, 'metric') or 0) > 1e-9 for r in (a,b)):
                    steps.append(math.degrees(angle))
    def maximum(key, records):
        values = [num(r, key) for r in records if num(r, key) is not None]
        return max(values) if values else None
    return dict(samples=len(valid), active_samples=len(active), applied_samples=len(applied),
                modes=modes, orthogonality_samples=len(cosines),
                max_abs_cos_velocity=max(cosines) if cosines else None,
                metric_identity_samples=len(identities),
                max_metric_identity_relative_error=max(identities) if identities else None,
                negative_metric_samples=invalid_metric, max_root_metric_norm=maximum('root_metric_norm', applied),
                roundoff_negative_metric_samples=roundoff_negative,
                max_applied_direction_step_deg=max(steps) if steps else None,
                p95_applied_direction_step_deg=sorted(steps)[int(.95*(len(steps)-1))] if steps else None,
                median_arrival_interval_s=statistics.median(arrival_intervals) if arrival_intervals else None,
                minimum_arrival_interval_s=min(arrival_intervals) if arrival_intervals else None,
                max_root_force_norm=maximum('root_force_norm', applied),
                max_world_turn_rate=max(turns) if turns else None,
                owner_switches=switches, logging_gap_count=gaps,
                latest_sample_time=num(valid[-1], 't') if valid else None,
                controller_parameters=metadata.get('parameters', {}))


def render(csv_path, output, metadata):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    rows = read_rows(csv_path)
    result = audit(rows, metadata)
    if len(rows) > 5000:
        step = math.ceil(len(rows)/5000)
        rows = rows[::step] + [rows[-1]]
    def fmt(value):
        return '미확인' if value is None else f'{value:.4g}' if isinstance(value, float) else str(value)
    notes = []
    if not rows:
        notes.append('TG 진단 샘플 대기 중입니다. 아직 방향이나 metric의 동작을 판정할 데이터가 없습니다.')
    elif not result['applied_samples']:
        notes.append('관측 기간에 TG metric이 제어에 기여한 샘플이 없습니다. 비활성·거리 gate·관측 모드를 그래프에서 확인하세요.')
    if result['max_abs_cos_velocity'] is None:
        notes.append('속도 수직성 검증에 필요한 simple 모드의 이동 구간이 아직 없습니다.')
    elif result['max_abs_cos_velocity'] > 1e-5:
        notes.append('이동 구간에서 속도 수직성 잔차가 큽니다. 속도 출처와 fallback 상태를 확인해야 합니다.')
    if result['metric_identity_samples'] == 0:
        notes.append('root metric 진단값이 없어 pullback 크기의 일치 여부는 미확인입니다.')
    elif result['max_metric_identity_relative_error'] > 1e-6:
        notes.append('m · ||J_t||²와 root metric norm 사이에 불일치가 관측됐습니다.')
    if result['negative_metric_samples']:
        notes.append('음수 metric이 관측됐습니다. 해당 구간의 원본 진단값을 확인해야 합니다.')
    if result['roundoff_negative_metric_samples']:
        notes.append(f"거의 0인 음수 metric {result['roundoff_negative_metric_samples']}건은 절댓값 1e-9 이하의 수치 오차 구간으로 별도 집계합니다.")
    if result['max_applied_direction_step_deg'] is not None:
        notes.append(f"TG 기여 구간의 수신 샘플 사이 방향 변화: 최대 {result['max_applied_direction_step_deg']:.2f}°, 95백분위 {result['p95_applied_direction_step_deg']:.2f}°. 수신 지연·버스트·샘플 누락이 포함될 수 있어 한 제어 주기의 변화량으로 단정하지 않습니다.")
    if result['minimum_arrival_interval_s'] is not None:
        notes.append('세계 방향 각속도는 메시지 수신 시간으로 계산한 추정값입니다. 제어 시각이 포함되지 않은 토픽이므로 실제 로봇 각속도 판정에는 사용할 수 없습니다.')
    params = metadata.get('parameters', {})
    if params.get('real_joint_state_source') == 'reference':
        notes.append('현재 관절 상태 출처는 reference입니다. 속도 수직성도 controller reference 관절 상태를 기준으로 해석합니다.')
    if result['latest_sample_time'] is not None and time.time() - result['latest_sample_time'] > 3:
        notes.append('최근 3초 동안 새 TG 진단이 없습니다. 마지막 저장된 데이터를 표시합니다.')
    fig = make_subplots(rows=4, cols=2, subplot_titles=(
        'TG scalar metric / root metric norm', '거리·blockage·pocket·handoff 활성화',
        '세계 좌표의 회피 방향', '방향·속도 수직성 |t·v| / |v|',
        '담당 센서 / 모드 / fallback', '목표 오차와 센서 등가 거리',
        'root force norm / 원하는 가속도', '속도 출처와 기록된 속도'))
    first = finite(rows[0].get('t')) if rows else 0
    def series(key):
        return [finite(r.get(key)) for r in rows]
    x = [(finite(r.get('t')) or first) - first for r in rows]
    def line(key, row, col, label=None):
        fig.add_trace(go.Scattergl(x=x, y=series(key), name=label or key, mode='lines'), row=row, col=col)
    for key in ('metric', 'root_metric_norm'): line(key, 1, 1)
    for key in ('alpha_d', 'alpha_block', 'alpha_free', 'handoff_gain', 'alpha_tg'): line(key, 1, 2)
    for key in ('t_x', 't_y', 't_z'): line(key, 2, 1)
    cos = []
    for r in rows:
        speed, dot, fallback, mode = [finite(r.get(k)) for k in
                                      ('steering_speed', 'steering_velocity_dot', 'steering_fallback', 'direction_mode')]
        eligible = mode == 3 and fallback is not None and not (int(fallback) & 3) and speed is not None and speed > 1e-8
        cos.append(abs(dot)/speed if eligible and dot is not None else None)
    fig.add_trace(go.Scattergl(x=x, y=cos, name='|cos(t,v)|', mode='lines'), row=2, col=2)
    for key in ('sensor_index', 'direction_mode', 'steering_fallback'): line(key, 3, 1)
    for key in ('goal_distance', 'clearance'): line(key, 3, 2)
    for key in ('root_force_norm', 'desired_accel'): line(key, 4, 1)
    for key in ('steering_speed', 'steering_velocity_source'): line(key, 4, 2)
    fig.update_layout(template='plotly_dark', height=1080, hovermode='x unified',
                      margin=dict(t=65, b=50), legend=dict(orientation='h', y=-.09))
    fig.update_xaxes(title_text='수집 시작 후 시간 [s]')
    trajectory = go.Figure()
    if rows:
        trajectory.add_trace(go.Scatter3d(x=series('tcp_x'), y=series('tcp_y'), z=series('tcp_z'),
                                          mode='lines', name='TCP 경로', line=dict(color='#38bdf8', width=4)))
        for r in rows[::max(1, len(rows)//60)]:
            if (finite(r.get('metric')) or 0) <= 0 or (finite(r.get('active')) or 0) <= .5: continue
            p = [finite(r.get('cp_'+a)) for a in 'xyz']
            t = [finite(r.get('t_'+a)) for a in 'xyz']
            if any(v is None for v in p+t): continue
            segment = [[value, value+.06*direction] for value, direction in zip(p,t)]
            trajectory.add_trace(go.Scatter3d(x=segment[0], y=segment[1], z=segment[2], mode='lines',
                                              line=dict(color='#fb923c', width=4), showlegend=False))
    trajectory.update_layout(template='plotly_dark', height=520, title='TCP 경로와 센서 위치의 TG 방향 (주황, 표시 길이 6 cm)',
                             scene=dict(aspectmode='data', xaxis_title='X [m]', yaxis_title='Y [m]', zaxis_title='Z [m]'))
    cards = [('진단 샘플', result['samples']), ('TG 기여 샘플', result['applied_samples']),
             ('이동 중 최대 |cos(t,v)|', result['max_abs_cos_velocity']),
             ('metric 식 상대 오차 최대', result['max_metric_identity_relative_error']),
             ('최대 root metric norm', result['max_root_metric_norm']),
             ('기여 구간 최대 샘플 방향 변화 [°]', result['max_applied_direction_step_deg'])]
    body = ''.join(f'<div class="card"><small>{html.escape(name)}</small><strong>{fmt(value)}</strong></div>' for name,value in cards)
    text = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>TG-RMP 관측 보고서</title>'
    text += '<style>body{background:#0b1120;color:#e2e8f0;font:16px system-ui;margin:24px}h1{margin-bottom:8px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card,aside{background:#182235;padding:18px;border-radius:12px}small{display:block;color:#94a3b8}strong{display:block;font-size:26px;margin-top:8px}li{margin:9px 0}pre{white-space:pre-wrap;font-size:12px}button{padding:8px 18px;cursor:pointer}</style>'
    text += '<h1>TG-RMP 방향·metric 관측</h1><p>기록 중에도 CSV와 이 HTML을 갱신합니다. 확대·이동·범례 클릭이 가능하며 외부 인터넷 연결이 필요 없습니다. 수치 검사는 전체 샘플, 그래프는 최대 약 5천 점을 사용합니다.</p>'
    text += f'<p>갱신: {datetime.now().isoformat(timespec="seconds")} · <button onclick="location.reload()">새로고침</button></p>'
    text += '<div class="cards">'+body+'</div><aside><ul>'+''.join('<li>'+html.escape(note)+'</li>' for note in notes)+'</ul></aside>'
    text += fig.to_html(full_html=False, include_plotlyjs=True)
    text += trajectory.to_html(full_html=False, include_plotlyjs=False)
    text += '<aside><b>해석 범위</b><p>root metric norm은 TG leaf의 joint 공간 기여량입니다. active이고 observe_only=0일 때 root 합산 경로에 들어갑니다. m·||J_t||²와의 일치를 검사하지만, 전체 root 행렬 대비 비중이나 최종 명령 가속도 차이·장애물 회피 성공을 확정하지는 않습니다. 수직성 검사는 simple=3 이동 구간에만 적용하며 저속 fallback은 제외합니다. 세계 방향 변화율은 owner 교체와 0.1초 초과 기록 공백을 제외한 값이며, 평면 내부 각속도 제한과 다릅니다. 센서 거리는 등가 거리이며 실제 표면 간격이 아닙니다.</p></aside>'
    text += '<details><summary>원본 경로·런타임 설정·수치 요약</summary><pre>'+html.escape(json.dumps(dict(csv=str(csv_path), metadata=metadata, audit=result), ensure_ascii=False, indent=2))+'</pre></details></html>'
    temp = output.with_suffix('.tmp')
    temp.write_text(text); temp.replace(output)
    output.with_suffix('.summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def lower_priority():
    import os
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.nice(10)


def monitor(args):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    import rclpy
    from rcl_interfaces.srv import GetParameters
    from log_tg_rmp_observation import Logger
    folder = args.output_dir.expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    csv_path, output = folder/'observation.csv', folder/'report.html'
    metadata = dict(started_at=datetime.now().isoformat(), topic=args.topic,
                    duration_s=args.duration, controller='/rmpflow_controller', parameters={})
    rclpy.init()
    node = Logger(args.topic, str(csv_path))
    client = node.create_client(GetParameters, '/rmpflow_controller/get_parameters')
    names = ['enable_tangent_escape_rmp', 'tg_rmp_direction_mode', 'tg_rmp_observe_only',
             'real_joint_state_source', 'tg_rmp_velocity_epsilon', 'publish_tangent_escape_rmp_data',
             'rmp_flag_gate_enabled', 'publish_tangent_escape_dual_solve_data']
    future, last_query = None, -100
    started, last_report = time.monotonic(), -100
    pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                               initializer=lower_priority)
    pending_report = None
    print(f'CSV: {csv_path}\nHTML: {output}', flush=True)
    try:
        while rclpy.ok() and (args.duration == 0 or time.monotonic()-started < args.duration):
            rclpy.spin_once(node, timeout_sec=.1)
            elapsed = time.monotonic()-started
            if future is not None and future.done():
                response = future.result()
                if response:
                    values = {}
                    for key, value in zip(names, response.values):
                        values[key] = {1:value.bool_value, 2:value.integer_value,
                                       3:value.double_value, 4:value.string_value}.get(value.type)
                    metadata['parameters'] = values
                    metadata['parameters_received_at'] = datetime.now().isoformat()
                future = None
            if elapsed-last_query >= 5 and future is None and client.service_is_ready():
                future = client.call_async(GetParameters.Request(names=names)); last_query = elapsed
            if pending_report is not None and pending_report.done():
                pending_report.result()
                pending_report = None
            if elapsed-last_report >= args.refresh and pending_report is None:
                node.file.flush()
                metadata.update(rows=node.rows, skipped=node.skipped, recording=True)
                (folder/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
                pending_report = pool.submit(render, csv_path, output, dict(metadata))
                last_report = elapsed
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        metadata.update(rows=node.rows, skipped=node.skipped, recording=False, finished_at=datetime.now().isoformat())
        (folder/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        if pending_report is not None:
            pending_report.result()
        pool.submit(render, csv_path, output, metadata).result()
        pool.shutdown()
        node.destroy_node(); rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, help='Render an existing dedicated TG observation CSV without ROS')
    parser.add_argument('--output', type=Path, help='HTML destination for --csv')
    parser.add_argument('--output-dir', type=Path, default=Path('~/ros2_ws/data/tg_rmp_monitor')/datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--topic', default='/tangent_escape_rmp_data')
    parser.add_argument('--duration', type=float, default=600, help='Recording seconds; 0 until Ctrl-C')
    parser.add_argument('--refresh', type=float, default=5, help='HTML update interval in seconds')
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 0 or not math.isfinite(args.refresh) or args.refresh < 1:
        parser.error('duration must be nonnegative; refresh must be at least 1 second')
    if args.csv:
        if not args.csv.is_file(): parser.error('CSV does not exist')
        output = args.output or args.csv.with_suffix('.html')
        output.parent.mkdir(parents=True, exist_ok=True)
        render(args.csv, output, {})
        print(output)
    else:
        monitor(args)


if __name__ == '__main__':
    main()

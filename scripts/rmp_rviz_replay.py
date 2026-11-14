#!/usr/bin/env python3
"""Read-only RViz replay with separate pose and policy timelines."""
import bisect
import csv
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Header
from std_srvs.srv import SetBool, Trigger
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from rmp_detail_capture import decode_snapshot

NAMES = ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']
MODES = {0: 'legacy', 1: 'sensor_history', 2: 'fixed', 3: 'simple'}
LAYERS = {
    'direction': ((1., .50, .10, 1.), 'TG direction', '주황 · TG가 선택한 접선 방향 — 길이 고정', True),
    'target': ((.15, .85, .35, 1.), 'Target', '초록 · Target의 가속도 기여 — TCP에서 표시', True),
    'collision': ((1., .22, .18, 1.), 'Collision', '빨강 · Collision 전체의 가속도 기여 — 감지 센서에서 표시', True),
    'tg': ((.28, .52, 1., 1.), 'TG contribution', '파랑 · TG의 가속도 기여 — 담당 센서에서 표시', True),
    'requested': ((.10, .85, .95, 1.), 'TG requested', '하늘 · TG leaf가 요청한 가속도 — 다른 정책과 합성 전', False),
    'total': ((.9, .9, .9, 1.), 'Total', '흰색 · 전체 RMP의 합성 가속도 — 아래 기준점에서 표시', False),
    'limited': ((1., .85, .15, 1.), 'Limited total', '노랑 · 관절 가속도 제한 후 합성 가속도 — 아래 기준점', False),
}


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(line for line in stream if not line.startswith('#')))


def number(row, key, default=0.):
    try:
        value = float(row[key])
        return value if math.isfinite(value) else default
    except (ValueError, TypeError, KeyError):
        return default


def load_frames(dataset, observation=''):
    root = Path.home() / 'ros2_ws/data'
    path = Path(dataset).expanduser() if dataset else None
    if path is None:
        files = sorted((root / 'rmp_datasets').glob('*.details.jsonl'),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        path = next((p for p in files if p.stat().st_size > 0), None)
        if path is None:
            files = list((root / 'rmp_datasets').glob('*.csv'))
            if not files:
                raise ValueError('No recorded CSV or details JSONL found')
            path = max(files, key=lambda p: p.stat().st_mtime)
    if path.suffix == '.jsonl':
        frames, detail_index = [], -1
        with path.open() as stream:
            for line in stream:
                # Recorder writes topic first. Skip other streams before JSON decoding.
                if '/rmp_recording_snapshot' not in line[:256]:
                    continue
                item = json.loads(line)
                if item.get('topic') != '/rmp_recording_snapshot':
                    continue
                frame = decode_snapshot(item['message']['data'], include_history=False)
                # The replay needs the fixed diagnostic prefix, not regression history.
                tg = frame['tg']
                if len(tg) >= 56:
                    tg = tg[:114 + 8 * int(tg[50])]
                kept = {'steady_time_s': frame['steady_time_s'], 'solver_q': frame['solver_q'],
                        'tg': tg, 'control_sequence': frame['control_sequence']}
                if frame.get('root_metric') is not None:
                    for key in ('root_metric', 'groups', 'solver_point_J_curvature',
                                'solver_tcp_position_J_curvature', 'visualization_points',
                                'raw_qdd', 'limited_qdd', 'obstacle_proxies'):
                        kept[key] = frame[key]
                    detail_index = len(frames)
                kept['detail_index'] = detail_index
                frames.append(kept)
        if not frames:
            raise ValueError('No coherent snapshots in ' + str(path))
        start = frames[0]['steady_time_s']
        for frame in frames:
            frame['time'] = frame.pop('steady_time_s') - start
        meta_path = path.with_suffix('.metadata.json')
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return frames, dict(kind='details', dataset=str(path), metadata=metadata,
                            detail_frames=sum('root_metric' in f for f in frames))
    raw = [r for r in read_csv(path) if all(r.get('q' + str(i)) for i in range(1, 7))]
    if not raw:
        raise ValueError('No valid joint states')
    jt = np.asarray([float(r['timestamp_unix']) for r in raw])
    q = np.array([[float(r['q' + str(i)]) for i in range(1, 7)] for r in raw])
    candidates = [Path(observation).expanduser()] if observation else list(
        (root / 'tg_rmp_monitor').glob('*/observation.csv'))
    selected, rows = None, None
    for obs in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        if not obs.stat().st_size:
            continue
        values = [r for r in read_csv(obs) if r.get('t') and jt[0] <= float(r['t']) <= jt[-1]]
        if values:
            selected, rows = obs, values
            break
    if not rows:
        raise ValueError('No overlapping TG observation CSV; specify observation_file')
    frames, start = [], float(rows[0]['t'])
    for row in rows:
        stamp = float(row['t'])
        index = bisect.bisect_left(jt, stamp)
        if min(abs(jt[min(index, len(jt)-1)]-stamp), abs(jt[max(index-1, 0)]-stamp)) > .05:
            continue
        frames.append(dict(time=stamp-start,
                           solver_q=[float(np.interp(stamp, jt, q[:,i])) for i in range(6)],
                           tg_row=row, detail_index=-1))
    return frames, dict(kind='legacy_csv', dataset=str(path), observation=str(selected),
                        detail_frames=0)


def tg_values(frame):
    row = frame.get('tg_row')
    tg = frame.get('tg', [])
    if row:
        return dict(cp=[number(row, 'cp_'+a) for a in 'xyz'],
                    direction=[number(row, 't_'+a) for a in 'xyz'],
                    metric=number(row, 'metric'), accel=number(row, 'desired_accel'),
                    active=number(row, 'active') > .5, observing=number(row, 'observe_only') > .5,
                    owner=int(number(row, 'sensor_index')), mode=int(number(row, 'direction_mode', -1)))
    if len(tg) > 51:
        offset = 98 + 4 * int(tg[50])
        return dict(cp=tg[11:14], direction=tg[20:23], metric=tg[10], accel=tg[9],
                    active=tg[0] > .5, observing=tg[44] > .5, owner=int(tg[1]),
                    mode=int(tg[offset]) if len(tg) > offset else -1)
    return dict(cp=[0.]*3, direction=[0.]*3, metric=0., accel=0.,
                active=False, observing=True, owner=0, mode=-1)


def pose_at(frames, times, elapsed, interpolate=True, max_gap=.1):
    index = max(0, bisect.bisect_right(times, elapsed)-1)
    q = frames[index]['solver_q']
    gap = 0.
    if index+1 < len(frames):
        gap = times[index+1]-times[index]
        if interpolate and 0 < gap <= max_gap:
            weight = max(0., min(1., (elapsed-times[index])/gap))
            q = ((1-weight)*np.asarray(q)+weight*np.asarray(frames[index+1]['solver_q'])).tolist()
    return index, q, gap


def group_contributions(frame, solve_offset):
    """One factorization for three RHSs, cached per recorded detailed frame."""
    if '_contributions' in frame:
        return frame['_contributions']
    forces, metrics = {}, {}
    for group in frame['groups']:
        gid = group['group_id']
        if gid not in (4, 5, 6):
            continue
        forces[gid] = forces.get(gid, np.zeros(6)) + np.asarray(group['force'])
        metrics[gid] = metrics.get(gid, np.zeros((6, 6))) + np.asarray(group['metric']).reshape(6, 6)
    result = {}
    if forces:
        matrix = np.asarray(frame['root_metric']).reshape(6, 6)
        scale = max(np.abs(matrix).max()*.01, 1.)
        matrix = matrix/scale + float(solve_offset)*np.eye(6)
        ids = sorted(forces)
        rhs = np.column_stack([forces[gid]/scale for gid in ids])
        try:
            accelerations = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            accelerations = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        result = {gid: (accelerations[:,i], float(np.linalg.norm(metrics[gid])))
                  for i, gid in enumerate(ids)}
    frame['_contributions'] = result
    return result


class Replay(Node):
    def __init__(self):
        super().__init__('replay', namespace='/rmp_replay')
        for key, value in [('dataset',''), ('observation_file',''), ('speed',1.),
                           ('paused',True), ('loop',False), ('frame_id','rmp_replay/base_link'),
                           ('control_panel',True), ('acceleration_scale',.015),
                           ('solve_offset',.001), ('detail_max_age',.15)]:
            self.declare_parameter(key, value)
        begin = time.monotonic()
        self.frames, self.info = load_frames(self.get_parameter('dataset').value,
                                            self.get_parameter('observation_file').value)
        params = self.info.get('metadata', {}).get('controller_parameters') or {}
        self.solve_offset = params.get('root_solve_offset', {}).get(
            'double_value', self.get_parameter('solve_offset').value)
        self.times = [f['time'] for f in self.frames]
        self.elapsed = 0.
        self.paused = self.get_parameter('paused').value
        self.speed = self.get_parameter('speed').value
        self.last = time.monotonic()
        self.seek_requested = None
        self.selected_point = -1
        self.show = {key: value[3] for key, value in LAYERS.items()}
        self.show_labels, self.interpolate = True, True
        self.last_visual_key = None
        self.last_publish = 0.
        self.marker_keys = set()
        self.panel_status = ''
        self.metric_status = ''
        self.detail_status = ''
        self.joints = self.create_publisher(JointState, 'joint_states', 1)
        self.markers = self.create_publisher(
            MarkerArray, 'markers', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_service(SetBool, 'pause', self.pause)
        self.create_service(Trigger, 'restart', self.restart)
        self.create_subscription(Float64, 'seek', lambda msg: setattr(self, 'seek_requested', msg.data), 1)
        self.add_on_set_parameters_callback(self.parameters_changed)
        self.timer = self.create_timer(1/30, self.tick)
        self.get_logger().info(
            f"{Path(self.info['dataset']).name}: {len(self.frames)} pose frames, "
            f"{self.info['detail_frames']} policy frames, {self.times[-1]:.2f}s; "
            f"loaded in {time.monotonic()-begin:.2f}s. Initially paused.")

    def parameters_changed(self, params):
        for param in params:
            if param.name == 'speed':
                if not math.isfinite(param.value) or param.value <= 0:
                    return SetParametersResult(successful=False, reason='speed must be positive')
                self.speed = param.value
            if param.name == 'paused':
                self.paused = param.value
        return SetParametersResult(successful=True)

    def pause(self, request, response):
        self.paused = request.data
        response.success = True
        response.message = str(self.paused)
        return response

    def restart(self, request, response):
        self.seek_requested = 0.
        response.success = True
        return response

    def tick(self):
        now = time.monotonic()
        dt, self.last = now-self.last, now
        if self.seek_requested is not None:
            self.elapsed = max(0., min(self.times[-1], self.seek_requested))
            self.seek_requested = None
        elif not self.paused:
            self.elapsed += dt*self.speed
        if self.elapsed > self.times[-1]:
            if self.get_parameter('loop').value:
                self.elapsed = 0.
            else:
                self.elapsed, self.paused = self.times[-1], True
        index, q, gap = pose_at(self.frames, self.times, self.elapsed, self.interpolate)
        frame = self.frames[index]
        tg = tg_values(frame)
        detail_index = frame.get('detail_index', -1)
        detail = self.frames[detail_index] if detail_index >= 0 else None
        age = self.elapsed-detail['time'] if detail is not None else math.inf
        fresh = age <= self.get_parameter('detail_max_age').value
        visual_key = (index, self.selected_point, tuple(self.show.items()), self.show_labels, fresh,
                      self.get_parameter('acceleration_scale').value)
        stamp = self.get_clock().now().to_msg()
        # A paused view only needs a heartbeat. Stable marker IDs avoid Ogre object churn.
        if not self.paused or now-self.last_publish >= 1. or visual_key != self.last_visual_key:
            self.joints.publish(JointState(header=Header(stamp=stamp), name=NAMES, position=q))
            self.last_publish = now
        if visual_key != self.last_visual_key:
            self.publish_markers(frame, tg, detail if fresh else None)
            self.last_visual_key = visual_key
        self.panel_status = (
            f"{MODES.get(tg['mode'],'unknown')} · TG 센서 S{tg['owner']:02d} · "
            f"TG {'활성' if tg['active'] and tg['metric'] > 1e-9 and not tg['observing'] else '기여 없음'}")
        if detail is None:
            self.detail_status = 'Target / Collision 가속도: 이 시점에 상세 기록 없음'
        elif not fresh:
            self.detail_status = f'상세 기록이 {age*1000:.0f}ms 전: 오래된 합성 가속도 화살표 숨김'
        else:
            self.detail_status = f'합성 가속도: {age*1000:.0f}ms 전 기록 · TG 선택 방향은 현재 표본'
        if gap > .1:
            self.detail_status += f' | 자세 기록 공백 {gap*1000:.0f}ms: 보간하지 않음'

    def publish_markers(self, frame, tg, detail):
        output = MarkerArray()
        keys = set()

        def marker(kind, namespace, color):
            m = Marker()
            m.header.frame_id = self.get_parameter('frame_id').value
            # Zero means latest TF. The points are already in the fixed base frame.
            m.ns, m.id, m.type, m.action = namespace, 0, kind, Marker.ADD
            m.pose.orientation.w = 1.
            m.color.r, m.color.g, m.color.b, m.color.a = color
            output.markers.append(m)
            keys.add((namespace, 0))
            return m

        def arrow(start, vector, layer, suffix='', scale=None, width=.008):
            if not self.show[layer]:
                return
            vec = np.asarray(vector, dtype=float)
            if not np.isfinite(vec).all() or np.linalg.norm(vec) < 1e-10:
                return
            scale = self.get_parameter('acceleration_scale').value if scale is None else scale
            delta = vec*scale
            length = np.linalg.norm(delta)
            if length > .6:
                delta *= .6/length
            tip = np.asarray(start)+delta
            color, label, _, _ = LAYERS[layer]
            m = marker(Marker.ARROW, layer+'/'+suffix, color)
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in (start, tip)]
            m.scale.x, m.scale.y, m.scale.z = width, width*2, width*3
            if self.show_labels:
                text = marker(Marker.TEXT_VIEW_FACING, layer+'/label/'+suffix, color)
                text.pose.position = Point(x=float(tip[0]), y=float(tip[1]), z=float(tip[2]+.025))
                text.scale.z = .022
                text.text = label + (' '+suffix if suffix else '')

        detected = {int(p[4]) for p in detail.get('obstacle_proxies', []) if int(p[4]) >= 0} if detail else set()
        if tg['active']:
            detected.add(tg['owner'])
        for sensor in sorted(detected):
            if tg['active'] and sensor == tg['owner']:
                position = tg['cp']
            elif detail and 0 <= sensor < len(detail['visualization_points']):
                position = detail['visualization_points'][sensor][:3]
            else:
                continue
            m = marker(Marker.SPHERE, f'detected/{sensor}', (1., .6, .05, .9))
            m.pose.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
            m.scale.x = m.scale.y = m.scale.z = .026
        if tg['active']:
            suffix = f"S{tg['owner']:02d}"
            arrow(tg['cp'], tg['direction'], 'direction', suffix, scale=.18, width=.005)
            if tg['metric'] > 1e-9 and not tg['observing']:
                arrow(tg['cp'], np.asarray(tg['direction'])*tg['accel'], 'requested', suffix)
        norms = []
        if detail is not None:
            tcp = np.asarray(detail['solver_tcp_position_J_curvature']).reshape(3, 8)

            def at_sensor(sensor):
                if 0 <= sensor < len(detail['solver_point_J_curvature']):
                    values = np.asarray(detail['solver_point_J_curvature'][sensor]).reshape(3, 7)
                    return np.asarray(detail['visualization_points'][sensor][:3]), values[:,:6], values[:,6]
                return None

            contributions = group_contributions(detail, self.solve_offset)
            for gid, (qdd, norm) in contributions.items():
                if gid == 4:
                    arrow(tcp[:,0], tcp[:,1:7]@qdd, 'target', 'TCP')
                elif gid == 5:
                    for sensor in sorted(detected):
                        local = at_sensor(sensor)
                        if local is not None:
                            arrow(local[0], local[1]@qdd, 'collision', f'S{sensor:02d}')
                elif gid == 6:
                    # Root TG data belongs to its recorded owner, not a newer owner.
                    old_tg = tg_values(detail)
                    if old_tg['active'] and old_tg['owner'] == tg['owner']:
                        local = at_sensor(old_tg['owner'])
                        if local is not None:
                            arrow(local[0], local[1]@qdd, 'tg', f"S{old_tg['owner']:02d}")
                norms.append(f"{ {4:'Target',5:'Collision',6:'TG'}[gid]} ||M||={norm:.3g}")
            local = at_sensor(self.selected_point) if self.selected_point >= 0 else None
            point, jacobian, curvature = local if local is not None else (tcp[:,0], tcp[:,1:7], tcp[:,7])
            where = f'S{self.selected_point:02d}' if local is not None else 'TCP'
            arrow(point, jacobian@np.asarray(detail['raw_qdd'])+curvature, 'total', where, width=.012)
            arrow(point, jacobian@np.asarray(detail['limited_qdd'])+curvature, 'limited', where, width=.005)
        self.metric_status = f"TG leaf metric={tg['metric']:.3g}   " + ' · '.join(norms)
        for namespace, ident in self.marker_keys-keys:
            removed = Marker()
            removed.ns, removed.id, removed.action = namespace, ident, Marker.DELETE
            output.markers.append(removed)
        self.marker_keys = keys
        self.markers.publish(output)

    def panel(self):
        import tkinter as tk
        from tkinter import ttk
        root = tk.Tk()
        root.title('RMP 재생 · 화살표 범례')
        root.geometry('870x650')
        root.minsize(820, 620)
        clock = tk.StringVar()
        status, details, metrics = tk.StringVar(), tk.StringVar(), tk.StringVar()
        ttk.Label(root, textvariable=clock, font=('', 13, 'bold')).pack(pady=(10, 5))
        controls = ttk.Frame(root)
        controls.pack()
        ttk.Button(controls, text='재생 / 일시정지', command=lambda: setattr(self, 'paused', not self.paused)).pack(side='left')
        ttk.Button(controls, text='처음', command=lambda: setattr(self, 'seek_requested', 0.)).pack(side='left')
        peak = max(range(len(self.frames)), key=lambda i: tg_values(self.frames[i])['metric'])
        ttk.Button(controls, text='TG 최대 시점', command=lambda: setattr(self, 'seek_requested', self.times[peak])).pack(side='left')
        speed = ttk.Combobox(controls, values=['0.25','0.5','1','2'], state='readonly', width=6)
        speed.set(str(self.speed))
        speed.pack(side='left', padx=8)
        speed.bind('<<ComboboxSelected>>', lambda e: setattr(self, 'speed', float(speed.get())))
        slider = tk.Scale(root, from_=0, to=self.times[-1], resolution=.01, orient='horizontal', length=810)
        slider.pack()
        drag = {'value':False}
        slider.bind('<ButtonPress-1>', lambda e: drag.update(value=True))

        def release(event):
            self.seek_requested = slider.get()
            drag['value'] = False
        slider.bind('<ButtonRelease-1>', release)
        ttk.Label(root, text=Path(self.info['dataset']).name).pack()
        ttk.Label(root, textvariable=status).pack(pady=(5, 0))
        ttk.Label(root, textvariable=details).pack()
        legend = ttk.LabelFrame(root, text='화살표 이름과 의미 — 체크해서 표시 / 숨김')
        legend.pack(fill='x', padx=15, pady=10)
        variables = []
        for layer, (color, _, meaning, enabled) in LAYERS.items():
            line = ttk.Frame(legend)
            line.pack(fill='x', pady=2)
            variable = tk.BooleanVar(value=enabled)
            variables.append(variable)
            ttk.Checkbutton(line, variable=variable,
                            command=lambda k=layer,v=variable: self.show.update({k:v.get()})).pack(side='left')
            hex_color = '#'+''.join(f'{int(c*255):02x}' for c in color[:3])
            tk.Label(line, text='━━▶', fg=hex_color, bg='#29313e', width=5).pack(side='left', padx=(0,8))
            ttk.Label(line, text=meaning).pack(side='left')
        options = ttk.Frame(root)
        options.pack(fill='x', padx=20)
        names = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text='화살표에 이름 표시', variable=names,
                        command=lambda: setattr(self, 'show_labels', names.get())).pack(side='left')
        interpolation = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text='짧은 간격의 자세 보간 (100ms 이하)', variable=interpolation,
                        command=lambda: setattr(self, 'interpolate', interpolation.get())).pack(side='left', padx=10)
        ttk.Label(options, text='흰색/노랑 기준점').pack(side='left')
        count = max((len(f.get('visualization_points', [])) for f in self.frames), default=0)
        point = ttk.Combobox(options, values=['TCP']+[f'S{i:02d}' for i in range(count)],
                             state='readonly', width=6)
        point.set('TCP')
        point.pack(side='left')
        point.bind('<<ComboboxSelected>>', lambda e: setattr(
            self, 'selected_point', -1 if point.get() == 'TCP' else int(point.get()[1:])))
        ttk.Label(root, textvariable=metrics).pack(pady=(10, 3))
        ttk.Label(root, text='초록·빨강·파랑 = 공통 root metric을 거친 정책별 가속도 기여. 빨강은 센서별 독립 반발력이 아님.').pack()
        ttk.Label(root, text='주황 길이는 18cm 고정. 나머지는 가속도 크기에 비례하며 화면 길이 60cm에서 제한.').pack()
        ttk.Label(root, text='값이 0이거나 미감지·미기록이면 화살표가 없습니다. 오래된 상세 가속도는 150ms 후 숨깁니다.').pack()
        quality = self.info.get('metadata', {}).get('recording_quality', {})
        gaps = quality.get('sequence_gaps', 0)
        ttk.Label(root, text=f"기록: 자세 {len(self.frames):,}개 / 상세 {self.info['detail_frames']:,}개"
                  f" · 원본 중간 누락 {gaps:,}주기").pack(pady=(8,0))
        ttk.Label(root, text='RViz 회전: Move Camera 선택 → 왼쪽 드래그 / 이동: 휠 버튼 드래그 / 확대: 휠').pack()

        def update():
            clock.set(f"{self.elapsed:.2f} / {self.times[-1]:.2f} s  ·  "
                      f"{'일시정지' if self.paused else '재생 중'}  ·  {self.speed:g}배")
            status.set(self.panel_status)
            details.set(self.detail_status)
            metrics.set(self.metric_status)
            if not drag['value']:
                slider.set(self.elapsed)
            if rclpy.ok():
                root.after(100, update)
            else:
                root.destroy()
        root.after(100, update)
        root.mainloop()


def main():
    rclpy.init()
    node = Replay()
    thread = None
    def spin():
        try:
            rclpy.spin(node)
        except (ExternalShutdownException, KeyboardInterrupt):
            pass
    try:
        if node.get_parameter('control_panel').value and os.environ.get('DISPLAY'):
            thread = threading.Thread(target=spin, daemon=True)
            thread.start()
            try:
                node.panel()
            except ImportError:
                node.get_logger().warning('tkinter unavailable; use pause/restart services and seek topic')
                thread.join()
        else:
            spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        if thread is not None:
            thread.join(timeout=2.)
        node.destroy_node()


if __name__ == '__main__':
    main()

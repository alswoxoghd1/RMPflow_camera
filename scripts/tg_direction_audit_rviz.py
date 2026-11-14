#!/usr/bin/env python3
"""Separate, offline RViz viewer for TG direction-selection evidence."""
import bisect
import math
import os
import threading
import signal
from rclpy.signals import SignalHandlerOptions
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from rmp_rviz_replay import Replay, NAMES
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from tg_direction_audit import load_audit, direction, seed_choice, seed_rows, seed_frame, window, draw_figure, export_window, contribution_episodes, contextual_seed


class DirectionAudit(Replay):
    def __init__(self):
        super().__init__()
        self.audit, _, self.audit_quality = load_audit(self.info['dataset'])
        self.episodes=contribution_episodes(self.audit)
        if not self.episodes:
            raise ValueError('No positive TG contribution in this recording')
        self.audit_times=[r['time_s'] for r in self.audit]
        self.output=Path.home()/'ros2_ws/data/tg_direction_audit'/Path(self.info['dataset']).name.removesuffix('.details.jsonl')/'metric_on'
        self.episode_index=next((i for i,e in enumerate(self.episodes) if e['first_contribution'] is not None),0)
        self.choose_episode(self.episode_index)

    def choose_episode(self,index):
        self.initial_preview=None
        self.episode_index=index
        self.episode=self.episodes[index]
        self.initial_preview=self.episode.get('initial_event')
        self.window_rows=window(self.audit,self.episode['start'])
        self.window_end=max(self.episode['start'],min(self.episode['start']+5,self.times[-1]))
        self.paused=True
        self.seek_requested=self.episode['start']
        if self.initial_preview is not None:
            self.elapsed=self.episode['start'];self.seek_requested=None
        self.last_visual_key=None

    def tick(self):
        if getattr(self,'initial_preview',None) is not None:
            row=self.initial_preview
            now=time.monotonic()
            if self.last_visual_key!=('initial_event',row['sequence']) or now-self.last_publish>=1.:
                self.joints.publish(JointState(header=Header(stamp=self.get_clock().now().to_msg()),name=NAMES,position=row['q']))
                self.publish_markers(None,None,None)
                self.last_visual_key=('initial_event',row['sequence']);self.last_publish=now
            self.last=now
            return
        if hasattr(self,'window_end') and not self.paused:
            if self.elapsed+(time.monotonic()-self.last)*self.speed >= self.window_end:
                self.seek_requested=self.window_end
                self.paused=True
        super().tick()

    def current_row(self):
        i=max(0,bisect.bisect_right(self.audit_times,self.elapsed)-1)
        return self.audit[i]

    def publish_markers(self,frame,tg,detail):
        if not hasattr(self,'audit_times'):
            return
        row=self.initial_preview or self.current_row();out=MarkerArray();keys=set()
        def base(kind,name,color):
            m=Marker();m.header.frame_id=self.get_parameter('frame_id').value
            m.ns=name;m.id=0;m.type=kind;m.action=Marker.ADD;m.pose.orientation.w=1.
            m.color.r,m.color.g,m.color.b,m.color.a=color;out.markers.append(m);keys.add((name,0))
            return m
        def point(v):return Point(x=float(v[0]),y=float(v[1]),z=float(v[2]))
        def label(pos,text,name,color=(1.,1.,1.,1.),size=.022):
            m=base(Marker.TEXT_VIEW_FACING,name,color);m.pose.position=point(pos);m.scale.z=size;m.text=text
        def arrow(origin,vector,name,color,width=.005,length=.18):
            v=np.asarray(vector);norm=np.linalg.norm(v)
            if norm<1e-10:return
            m=base(Marker.ARROW,name,color);m.points=[point(origin),point(origin+length*v/norm)]
            m.scale.x=width;m.scale.y=2*width;m.scale.z=3*width
        if row['contributing']:
            data=row['tg'];cp=np.asarray(data[11:14]);normal=-np.asarray(data[17:20])
            ring=base(Marker.LINE_STRIP,'audit/plane',(.65,.65,.7,.6));ring.scale.x=.002
            ring.points=[point(cp+.15*direction(data,a)) for a in np.linspace(0,2*math.pi,49)]
            arrow(cp,normal,'audit/mount_axis',(.65,.65,.7,.8),length=.12)
            label(cp+.13*normal,'mount axis','audit/mount_label',size=.018)
            if row['seed_applied'] and row['seeds']:
                winner=seed_choice(row['seeds'])
                for i,s in enumerate(row['seeds']):
                    v=direction(data,s['theta']);selected=i==winner
                    color=(1.,.85,.15,1.) if selected else (.45,.6,.8,.65)
                    arrow(cp,v,f'audit/C{i}',color,width=.009 if selected else .004)
                    label(cp+.22*v,f'C{i}: {s["total_score"]:.3f}',f'audit/label{i}',color,size=.018)
            color=(1.,.5,.1,1.) if row['contributing'] else (.7,.7,.7,.8)
            arrow(cp,data[20:23],'audit/selected',color,width=.009,length=.24)
            label(cp+np.array([0,0,.3]),f'S{row["owner"]:02d}  TG {"ON" if row["contributing"] else "OFF"}\nblock={row["block"]:.3f}  M={row["metric"]:.3g}',
                  'audit/status',color)
            k=76+4*int(data[50]);tcp=np.asarray(data[k:k+3]);goal=np.asarray(row['goal'])
            arrow(tcp,goal-tcp,'audit/goal',(.2,.85,.4,1.),length=.18)
            label(tcp+np.array([0,0,.08]),'TCP goal','audit/goal_label',(.2,.85,.4,1.))
        else:
            label(np.array([.3,0,.9]),'TG OFF: direction hidden','audit/status')
        for name,ident in self.marker_keys-keys:
            m=Marker();m.ns=name;m.id=ident;m.action=Marker.DELETE;out.markers.append(m)
        self.marker_keys=keys;self.markers.publish(out)

    def panel(self):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        root=tk.Tk();root.title('TG 방향 선택 근거 · TG 메트릭 ON 이후 5초');root.geometry('1240x970')
        head=ttk.Frame(root);head.pack(fill='x',padx=10,pady=5)
        names=[f'{i+1}. TG 메트릭 ON {e["start"]:.3f}s / S{e["owner"]:02d}' for i,e in enumerate(self.episodes)]
        episodes=ttk.Combobox(head,values=names,state='readonly',width=70);episodes.current(self.episode_index);episodes.pack(side='left')
        def select(_):
            self.choose_episode(episodes.current());slider.configure(from_=self.episode['start'],to=self.window_end)
        episodes.bind('<<ComboboxSelected>>',select)
        def play_pause():
            self.initial_preview=None;self.last_visual_key=None;self.paused=not self.paused
        ttk.Button(head,text='재생 / 정지',command=play_pause).pack(side='left',padx=5)
        speed=ttk.Combobox(head,values=['0.1','0.25','0.5','1'],state='readonly',width=5);speed.set('.25');speed.pack(side='left')
        speed.bind('<<ComboboxSelected>>',lambda _:setattr(self,'speed',float(speed.get())))
        controls=ttk.Frame(root);controls.pack(fill='x',padx=10)
        def step(delta):
            self.initial_preview=None;self.last_visual_key=None
            i=max(0,min(len(self.audit)-1,bisect.bisect_right(self.audit_times,self.elapsed)-1+delta))
            self.paused=True;self.seek_requested=float(np.clip(self.audit_times[i],self.episode['start'],self.window_end))
        ttk.Button(controls,text='이전 표본',command=lambda:step(-1)).pack(side='left')
        ttk.Button(controls,text='다음 표본',command=lambda:step(1)).pack(side='left')
        def initial():
            self.initial_preview=self.episode.get('initial_event');self.last_visual_key=None
            self.paused=True;self.elapsed=self.episode['start']
            self.seek_requested=None if self.initial_preview is not None else self.episode['start']
        def preview():
            seed=contextual_seed(self.audit,self.elapsed) or self.episode.get('initial_event')
            if seed is not None:
                self.paused=True;self.initial_preview=seed;self.last_visual_key=None
        preview_button=ttk.Button(controls,text='초기 8후보 정지화면',command=preview)
        preview_button.pack(side='left',padx=5)
        ttk.Button(controls,text='TG 켜진 순간으로',command=initial).pack(side='left',padx=5)
        message=tk.StringVar();status=tk.StringVar();evidence=tk.StringVar();seed_status=tk.StringVar()
        def export():
            path=export_window(self.audit,self.episode,self.output)
            message.set(f'저장: {path} (CSV / JSON / PDF / SVG / PNG)')
        ttk.Button(controls,text='이 5초를 논문용 파일로 저장',command=export).pack(side='left',padx=5)
        ttk.Label(root,text=Path(self.info['dataset']).name).pack()
        ttk.Label(root,textvariable=status).pack()
        slider=tk.Scale(root,from_=self.episode['start'],to=self.window_end,resolution=.001,orient='horizontal',length=1170)
        slider.pack(fill='x',padx=10);drag={'value':False}
        slider.bind('<ButtonPress-1>',lambda _:drag.update(value=True))
        def seek(_):
            self.initial_preview=None;self.last_visual_key=None
            self.paused=True;self.seek_requested=slider.get();drag['value']=False
        slider.bind('<ButtonRelease-1>',seek)
        ttk.Label(root,textvariable=seed_status).pack()
        columns=('id','angle','goal','motion','risk','escape','total','selected')
        table=ttk.Treeview(root,columns=columns,show='headings',height=8)
        labels=['후보','각도 [deg]','목표 (가중)','현재 움직임 (가중)','history (음의 비용)','우회 이동 (가중)','총점','선택']
        for col,title in zip(columns,labels):table.heading(col,text=title);table.column(col,width=145,anchor='center')
        table.tag_configure('selected',background='#fff1ab');table.pack(fill='x',padx=15)
        footer=ttk.Frame(root);footer.pack(side='bottom',fill='x')
        fig=Figure(figsize=(12,5),layout='constrained');canvas=FigureCanvasTkAgg(fig,master=root);canvas.get_tk_widget().pack(fill='both',expand=True)
        ttk.Label(footer,textvariable=evidence).pack()
        ttk.Label(footer,text='상단 표는 초기 8후보 기록입니다. v6 이후 갱신은 그래프의 전체 72후보·관측 근거를 확인하세요. +는 theta 증가, −는 감소.').pack()
        ttk.Label(footer,text='주황 = 선택 방향·TG 기여 있음 / TG OFF = 방향 화살표 숨김 / 노랑 = 초기 최고 점수 후보 / 초록 = TCP 목표 방향').pack()
        ttk.Label(footer,textvariable=message).pack()
        last={'seed':None,'draw':None,'wall':0.}
        def refresh():
            if not rclpy.ok():root.destroy();return
            row=self.current_row();now=self.elapsed
            if not drag['value']:slider.set(now)
            status.set(f'{now:.3f}s  |  TG ON 후 {now-self.episode["start"]:.3f}s  |  S{row["owner"]:02d}  |  TG {"기여 있음" if row["contributing"] else "기여 없음"}  |  block {row["block"]:.3f} / M {row["metric"]:.3g}')
            seed=self.initial_preview or (contextual_seed(self.audit,now) if row['contributing'] else None);sid=seed['sequence'] if seed else None
            preview_button.configure(state='normal' if seed is not None or self.episode.get('initial_event') is not None else 'disabled')
            if self.initial_preview is not None:
                status.set(f'초기 선택 정지화면: {seed["time_s"]:.6f}s / S{seed["owner"]:02d} / 실제 선택 당시 관절 자세·기저축·8후보 (재생 버튼으로 복귀)')
            if sid!=last['seed'] or last['draw'] is None:
                table.delete(*table.get_children())
                for s in seed_rows(seed):
                    table.insert('', 'end', values=(f'C{s["candidate"]}',f'{s["angle_deg"]:.1f}',*[f'{s[k]:.5f}' for k in ('goal_score','continuity_score','negative_history_cost','escape_score','total_score')],'선택' if s['selected'] else ''),tags=('selected',) if s['selected'] else ())
                if seed:
                    scores=sorted((s['total_score'] for s in seed['seeds']),reverse=True)
                    seed_status.set(f'초기 8후보 점수: 선택 시각 {seed["time_s"]:.6f}s / S{seed["owner"]:02d} / 1–2등 점수차 {scores[0]-scores[1]:.6g} / {"별도 보존 이벤트" if seed.get("source")=="retained_initial_event" else "일반 표본에 남은 기록"}')
                else:seed_status.set('TG OFF: 방향 점수 표시 숨김' if not row['contributing'] else '현재 방향의 초기 후보 점수가 기록되지 않았습니다.')
                last['seed']=sid
            predicted_text=f'{row["predicted_min_range_m"]:.4g} m' if math.isfinite(row['predicted_min_range_m']) else '미예측'
            critical_text=f'S{row["critical_sensor"]:02d}' if row['critical_sensor']>=0 else '없음'
            evidence.set(f'위험 비용: 이력 기반 {row["measured_risk"]:.4g} / 미학습 움직임 비용 {row["unknown_risk"]:.4g} / 최소 예측 환산거리 {predicted_text} / 주 위험 센서 {critical_text}\n현재 history 모델 {int(row["model_count"])}개 / 신뢰도 지표 {row["confidence"]:.3f} / 로그의 전체 모델 복원 {"가능" if row["models_complete"] else "불가"} / 직전 누락 {row["missing_cycles"]}주기 | theta 갱신오차 {row["angle_error"]:.2g} rad / {row["events"]}')
            response=row.get('response')
            if response:
                phase={0:'현재 방향 관찰',1:'선택 방향으로 연속 회전',2:'조정 결과 평가',3:'검증된 방향 복귀',4:'근접 위험으로 회전 보류',5:'필수 관측 대기'}.get(int(response['phase']),'미확인')
                verdict={1:'개선 확인',2:'선택 방향 갱신',0:'기존 선택 유지',-2:'근접 악화',-3:'실측 부족',-4:'관측 신뢰도 낮음',-5:'추가 개선 후보 없음'}.get(int(response['result']),'미확인')
                if 'reason' in response:
                    phase={0:'방향 관찰·유지',1:'실측 실패에 따른 누적 회전',4:'근접 악화에 대응하며 회전',5:'유효 관측·후보 대기'}.get(int(response['phase']),'미확인')
                    verdict={0:'관찰 중',1:'실제 이동 회복',2:'정체로 회전 시작',3:'정체가 이어져 각도 누적',
                        4:'근접 악화로 회전 시작',5:'근접 악화 중 각도 누적',6:'실패·대안 근거로 회전 반전',
                        7:'한 바퀴 실패 후 반전',8:'block 비활성',9:'신선한 관측 부족',10:'후보 계산 불가'}.get(int(response['reason']),'미확인')
                evidence.set(evidence.get()+f"\n실측 방향 제어: {phase} / 진행 {response['speed']*1000:.2f} mm/s / 비교 기준 {response['baseline']*1000:.2f} mm/s / 판정 {verdict} / 실패 {int(response['failures'])}")
                if "sweep_sequence" in response:
                    evidence.set(evidence.get()+f"\n넓은 탐색 #{int(response['sweep_sequence'])}: 유지 {response['current_score']:.2f} → 후보 {response['best_score']:.2f} / 계산 {response['sweep_us']:.0f} µs")
                if "observation_confidence" in response:
                    evidence.set(evidence.get()+f"\n360° 점수: 무차원 / 관측 신뢰도 {response['observation_confidence']:.2f} / 목표 진행 {response['goal_speed']*1000:.1f} mm/s / 변경 확인 {int(response['pending_count'])}회")
                if 'turn_travel' in response:
                    evidence.set(evidence.get()+f"\n회전 {'+' if response['sign']>0 else '−'} / 실제 누적각 {math.degrees(response['turn_travel']):.1f}° / 반전 {int(response['reverse_count'])}회 / 양·음 방향 점수차 {response['sign_margin']:.3f} (미관측 방향의 안전 보장 아님)")
                if 'shadow' in response:
                    from tg_sensor_shadow_audit import compare_sides
                    comparison,_=compare_sides(response['shadow'],row['theta_rad'])
                    label={'insufficient_support':'방향별 이력 부족',
                           'no_resolved_difference':'센서 반응 차이 불명확',
                           'positive_supported_candidates':'+방향의 관측 근거 있는 후보 우세',
                           'negative_supported_candidates':'−방향의 관측 근거 있는 후보 우세'}[comparison]
                    evidence.set(evidence.get()+f'\n센서 근거 비교(기록 전용, 제어 미반영): {label}')

            if self.initial_preview is not None:
                evidence.set(f'초기 선택 원본 이벤트 / 기록 제어 주기 {seed["sequence"]} / 모델 {int(seed.get("model_count",0))}개 / 신뢰도 {seed.get("confidence",0):.3f} / 이벤트 큐 누락 {seed.get("queue_dropped",0)}')
            key=(self.episode_index,row['sequence'],sid,self.initial_preview is not None)
            if key!=last['draw'] and time.monotonic()-last['wall']>.5:
                draw_figure(fig,self.window_rows,seed,now,initial_preview=self.initial_preview is not None);canvas.draw_idle();last.update(draw=key,wall=time.monotonic())
            root.after(100,refresh)
        root.after(50,refresh);root.mainloop()


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    def shutdown_signal(_number,_frame):
        if rclpy.ok():rclpy.shutdown()
    old_handlers={sig:signal.signal(sig,shutdown_signal) for sig in (signal.SIGINT,signal.SIGTERM)}
    node=DirectionAudit();thread=None
    def spin():
        try:rclpy.spin(node)
        except (ExternalShutdownException,KeyboardInterrupt):pass
        except Exception:
            if rclpy.ok():raise  # invalid wait-set during shutdown is expected
    try:
        if node.get_parameter('control_panel').value and os.environ.get('DISPLAY'):
            thread=threading.Thread(target=spin,daemon=True);thread.start();node.panel()
        else:spin()
    except KeyboardInterrupt:pass
    finally:
        if rclpy.ok():rclpy.shutdown()
        if thread is not None:thread.join(timeout=2)
        node.destroy_node()
        for sig,handler in old_handlers.items():signal.signal(sig,handler)


if __name__=='__main__':main()

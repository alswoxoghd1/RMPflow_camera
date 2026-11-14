#!/usr/bin/env python3
"""Build an offline URDF/TG replay from the existing observation and joint CSVs."""
import argparse,csv,json,math,struct,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from plotly.offline import get_plotlyjs

ROOT=Path(__file__).resolve().parents[1]

def numbers(text,default):return np.array([float(x) for x in (text or default).split()])
def transform(origin):
 t=np.eye(4)
 if origin is None:return t
 r,p,y=numbers(origin.get('rpy'),'0 0 0');c=np.cos([r,p,y]);s=np.sin([r,p,y])
 rx=np.array([[1,0,0],[0,c[0],-s[0]],[0,s[0],c[0]]]);ry=np.array([[c[1],0,s[1]],[0,1,0],[-s[1],0,c[1]]]);rz=np.array([[c[2],-s[2],0],[s[2],c[2],0],[0,0,1]])
 t[:3,:3]=rz@ry@rx;t[:3,3]=numbers(origin.get('xyz'),'0 0 0');return t

def mesh(path,scale):
 b=path.read_bytes();n=struct.unpack_from('<I',b,80)[0]
 if 84+50*n!=len(b):raise ValueError('Expected binary STL: '+str(path))
 dtype=np.dtype([('normal','<f4',3),('v','<f4',(3,3)),('attr','<u2')])
 vertices=np.frombuffer(b,offset=84,count=n,dtype=dtype)['v'].reshape(-1,3).astype(float)*scale
 # Vertex clustering bounds file size and replay cost, retaining closed surfaces.
 quantized=np.round(vertices/.012).astype(int)
 _,indices,inverse=np.unique(quantized,axis=0,return_index=True,return_inverse=True)
 v=vertices[indices];f=inverse.reshape(-1,3);f=f[(f[:,0]!=f[:,1])&(f[:,0]!=f[:,2])&(f[:,1]!=f[:,2])]
 return np.round(v,6).tolist(),f.tolist()

def build(obs,joints,urdf,out):
 robot=ET.parse(urdf).getroot();meshes=[]
 for link in robot.findall('link'):
  geometry=link.find('collision')
  if geometry is None:continue
  m=geometry.find('geometry/mesh')
  if m is None:continue
  filename=m.get('filename')
  path=ROOT/filename.split('rb10_rmpflow_rviz/',1)[-1] if filename.startswith('package://') else urdf.parent/filename
  if not path.is_file() or path.suffix.lower()!='.stl':continue
  v,f=mesh(path,numbers(m.get('scale'),'1 1 1'))
  meshes.append(dict(link=link.get('name'),v=v,f=f,origin=transform(geometry.find('origin')).tolist()))
 jointdefs=[]
 for j in robot.findall('joint'):
  jointdefs.append(dict(name=j.get('name'),kind=j.get('type'),parent=j.find('parent').get('link'),child=j.find('child').get('link'),origin=transform(j.find('origin')),axis=numbers(j.find('axis').get('xyz') if j.find('axis') is not None else None,'0 0 1')))
 with joints.open() as stream:
  raw=list(csv.DictReader(line for line in stream if not line.startswith('#')))
 valid=[r for r in raw if all(r.get('q'+str(i)) for i in range(1,7))]
 times=np.array([float(r['timestamp_unix']) for r in valid]);qs=np.array([[float(r['q'+str(i)]) for i in range(1,7)] for r in valid])
 with obs.open() as stream: rows=[r for r in csv.DictReader(stream) if r.get('t')]
 if not rows or not valid:raise ValueError('No recorded states to replay')
 frames=[];last=-math.inf;base=float(rows[0]['t']);names=['base','shoulder','elbow','wrist1','wrist2','wrist3']
 for row in rows:
  stamp=float(row['t'])
  if stamp-last<.099 or stamp<times[0] or stamp>times[-1]:continue
  index=np.searchsorted(times,stamp)
  if min(abs(times[min(index,len(times)-1)]-stamp),abs(times[max(index-1,0)]-stamp))>.05:continue
  last=stamp
  q={name:float(np.interp(stamp,times,qs[:,i])) for i,name in enumerate(names)}
  world={'base_link':np.eye(4)};remaining=list(jointdefs)
  while remaining:
   progress=False
   for j in list(remaining):
    if j['parent'] not in world:continue
    angle=q.get(j['name'],0);a=j['axis'];a=a/np.linalg.norm(a)
    k=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    motion=np.eye(4)
    if j['kind'] in ('revolute','continuous'):motion[:3,:3]=np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)
    world[j['child']]=world[j['parent']]@j['origin']@motion
    remaining.remove(j);progress=True
   if not progress:raise ValueError('Broken URDF tree')
  matrices=[np.round(world[m['link']]@np.array(m['origin']),8).reshape(-1).tolist() for m in meshes]
  def value(key):
   try:
    v=float(row[key]);return v if math.isfinite(v) else None
   except:return None
  frames.append(dict(t=stamp-base,matrices=matrices,cp=[value('cp_'+a) for a in 'xyz'],direction=[value('t_'+a) for a in 'xyz'],tcp=[value('tcp_'+a) for a in 'xyz'],metric=value('metric'),rootMetric=value('root_metric_norm'),accel=value('desired_accel'),active=value('active'),owner=value('sensor_index'),alpha=value('alpha_tg'),goalDistance=value('goal_distance'),observe=value('observe_only')))
 if not frames:raise ValueError('No overlapping timestamps')
 payload=json.dumps(dict(meshes=meshes,frames=frames,sources=[str(obs),str(joints),str(urdf)]),allow_nan=False,separators=(',',':')).replace('</','<\\/')
 template=r'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>TG 로봇 재생</title><style>body{margin:0;background:#0c1424;color:#e2e8f0;font:15px system-ui}header{padding:18px 25px;background:#152238}h1{margin:0;font-size:25px}.warning{padding:10px;background:#3b3021;color:#fde68a;margin-top:10px}.layout{display:grid;grid-template-columns:3fr 1fr}#scene{height:69vh;min-height:550px}.panel{padding:18px;background:#142033}button,select{padding:8px;background:#263854;color:white;border:1px solid #54647b;border-radius:5px;cursor:pointer}input[type=range]{width:55%}.controls{padding:14px 25px}table{width:100%;line-height:2.1}td:last-child{text-align:right;color:#7dd3fc}.muted{color:#9aa8bb}#chart{height:230px}label{margin-right:12px}a{color:#7dd3fc}</style>
<header><h1>로봇 위에서 보는 TG-RMP · 실제 기록 재생</h1><div class="warning">이번 기록에는 Target·Collision·최종 합성 벡터가 없습니다. 파랑은 기록된 TG 요구 가속도, 주황은 선택 방향입니다. 누락된 벡터는 만들지 않았습니다.</div></header>
<div class="controls"><button id="play">▶ 재생</button> <button id="start">처음</button> <button id="peak">TG 최대 metric</button> <select id="speed"><option value=".25">0.25×</option><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select> <input id="time" type="range" min="0" value="0" step="1"><b id="stamp"></b></div>
<div class="layout"><div id="scene"></div><section class="panel"><h2>현재 순간</h2><table id="stats"></table><hr><p style="color:#38bdf8">━ 파랑: TG 요구 가속도<br>길이 = |a| × 0.008 m/(m/s²), 최대 0.5 m 표시</p><p style="color:#fb923c">━ 주황: TG 선택 방향<br>고정 길이 0.18 m</p><p>파랑 선 굵기 = metric 크기의 로그 표현</p><label><input id="accel" type="checkbox" checked>TG 가속도</label><br><label><input id="dir" type="checkbox" checked>선택 방향</label><br><label><input id="path" type="checkbox" checked>TCP 궤적</label><hr><p class="muted">● Target: 미기록<br>● Collision: 미기록<br>● 최종 합성: 미기록</p><p class="muted">로봇은 reference 관절각으로 재생합니다. TG 시작점은 solver가 기록한 위치여서 혼합 상태·시간 보간 차이로 형상과 어긋날 수 있습니다. URDF collision 형상을 간소화해서 사용합니다.</p><p id="coverage"></p></section></div><div id="chart"></div><details style="padding:20px"><summary>기록 출처와 해석</summary><p>화살표는 TG가 요구한 가속도이며, 최종 합성 가속도 또는 실측 로봇 가속도와 다릅니다. metric이 0이거나 observe_only이면 파랑 가속도 화살표는 숨깁니다. 주황 방향만 보이는 경우 실제 TG 기여가 없을 수 있습니다. 원본 관측을 약 10 Hz로 표시하며 두 CSV를 수신 시각으로 정렬했습니다. 인터넷 연결 없이 열 수 있습니다.</p><pre id="sources"></pre></details>
<script>__PLOTLY__</script><script>const DATA=__DATA__; (async()=>{
const frames=DATA.frames, meshes=DATA.meshes, scene=document.getElementById('scene');let index=0,playing=false,lastTime=0,playbackTime=0,busy=false;
function xyz(m,v){return [m[0]*v[0]+m[1]*v[1]+m[2]*v[2]+m[3],m[4]*v[0]+m[5]*v[1]+m[6]*v[2]+m[7],m[8]*v[0]+m[9]*v[1]+m[10]*v[2]+m[11]]}
function verts(frame,n){let p=meshes[n].v.map(v=>xyz(frame.matrices[n],v));return {x:p.map(p=>p[0]),y:p.map(p=>p[1]),z:p.map(p=>p[2])}}
let traces=meshes.map((m,n)=>({...verts(frames[0],n),type:'mesh3d',i:m.f.map(f=>f[0]),j:m.f.map(f=>f[1]),k:m.f.map(f=>f[2]),color:n%2?'#a7b5c5':'#c7d3df',opacity:1,name:m.link,hoverinfo:'name',lighting:{ambient:.65,diffuse:.8},showlegend:false}));
const count=traces.length;traces.push({type:'scatter3d',mode:'lines',x:frames.map(f=>f.tcp[0]),y:frames.map(f=>f.tcp[1]),z:frames.map(f=>f.tcp[2]),line:{color:'#64748b',width:3},name:'TCP 궤적'});
function arrow(color,name){return {type:'scatter3d',mode:'lines+markers',x:[],y:[],z:[],line:{color,width:7},marker:{size:[3,5],color},name}}
traces.push(arrow('#38bdf8','TG 요구 가속도'),arrow('#fb923c','TG 선택 방향'));
traces.push({type:'cone',x:[],y:[],z:[],u:[],v:[],w:[],showscale:false,colorscale:[[0,'#38bdf8'],[1,'#38bdf8']],sizemode:'absolute',sizeref:.028,anchor:'tip',name:'가속도 화살촉',showlegend:false});
const layout={template:'plotly_dark',paper_bgcolor:'#0c1424',plot_bgcolor:'#0c1424',margin:{l:0,r:0,t:10,b:0},uirevision:'camera',scene:{aspectmode:'data',xaxis:{title:'X [m]',range:[-.9,.9]},yaxis:{title:'Y [m]',range:[-.9,.9]},zaxis:{title:'Z [m]',range:[-.1,1.25]},camera:{eye:{x:.85,y:.85,z:.65},center:{x:0,y:0,z:-.12}}},legend:{x:0,y:1}};
await Plotly.newPlot(scene,traces,layout,{responsive:true});document.getElementById('time').max=frames.length-1;
function fmt(v){return v===null?'미기록':Number(v).toFixed(3)}
async function draw(i){if(busy)return;busy=true;try{index=Math.max(0,Math.min(frames.length-1,i));let f=frames[index];const changes=meshes.map((m,n)=>verts(f,n));for(let n=0;n<count;n++)Object.assign(traces[n],changes[n]);
 let valid=f.direction.every(v=>v!==null)&&f.cp.every(v=>v!==null)&&f.active>.5;let active=valid&&f.metric>1e-9&&f.observe===0;
 for(let j=0;j<2;j++){let show=valid&&(j?document.getElementById('dir').checked:active&&document.getElementById('accel').checked);let length=j?.18:Math.min(.5,Math.abs(f.accel||0)*.008);let sign=j?1:Math.sign(f.accel||0);let end=f.cp.map((v,a)=>v+length*sign*f.direction[a]);let tr=traces[count+1+j];tr.x=show?[f.cp[0],end[0]]:[];tr.y=show?[f.cp[1],end[1]]:[];tr.z=show?[f.cp[2],end[2]]:[];tr.line.width=j?4:Math.min(14,2+2*Math.log10(1+Math.max(0,f.metric||0)));}
 let cone=traces[count+3],a=traces[count+1];cone.x=a.x.length?[a.x[1]]:[];cone.y=a.y.length?[a.y[1]]:[];cone.z=a.z.length?[a.z[1]]:[];cone.u=a.x.length?[f.direction[0]*Math.sign(f.accel||0)]:[];cone.v=a.x.length?[f.direction[1]*Math.sign(f.accel||0)]:[];cone.w=a.x.length?[f.direction[2]*Math.sign(f.accel||0)]:[];
 traces[count].visible=document.getElementById('path').checked;await Plotly.react(scene,traces.map(t=>({...t})),{...layout,datarevision:index});document.getElementById('time').value=index;document.getElementById('stamp').textContent=f.t.toFixed(2)+' / '+frames.at(-1).t.toFixed(2)+' s';document.getElementById('stats').innerHTML=[['TG 기여',active?'활성':'비활성'],['담당 센서 index',f.owner],['TG metric',fmt(f.metric)],['Root metric norm',fmt(f.rootMetric)],['요구 가속도 [m/s²]',fmt(f.accel)],['활성화 gate',fmt(f.alpha)],['목표 거리 [m]',fmt(f.goalDistance)]].map(r=>'<tr><td>'+r[0]+'</td><td>'+r[1]+'</td></tr>').join('');}finally{busy=false}}
function seek(i){playbackTime=frames[i].t;lastTime=performance.now();draw(i)}document.getElementById('time').oninput=e=>seek(+e.target.value);document.getElementById('play').onclick=()=>{playing=!playing;lastTime=performance.now();playbackTime=frames[index].t;document.getElementById('play').textContent=playing?'❚❚ 일시정지':'▶ 재생'};document.getElementById('start').onclick=()=>seek(0);const peak=frames.reduce((best,f,i)=>f.metric>frames[best].metric?i:best,0);document.getElementById('peak').onclick=()=>seek(peak);for(const id of ['accel','dir','path'])document.getElementById(id).onchange=()=>draw(index);
setInterval(()=>{if(!playing||busy)return;let now=performance.now(),target=playbackTime+(now-lastTime)/1000*+document.getElementById('speed').value;lastTime=now;playbackTime=target;let next=index;while(next<frames.length-1&&frames[next+1].t<=target)next++;if(next===index&&next<frames.length-1)return;draw(next);if(next===frames.length-1){playing=false;document.getElementById('play').textContent='▶ 재생'}},50);
await Plotly.newPlot('chart',[{x:frames.map(f=>f.t),y:frames.map(f=>f.metric),name:'TG metric',line:{color:'#38bdf8'}},{x:frames.map(f=>f.t),y:frames.map(f=>f.accel),name:'요구 가속도',yaxis:'y2',line:{color:'#fb923c'}}],{template:'plotly_dark',margin:{l:60,r:60,t:20,b:45},xaxis:{title:'실험 시간 [s]'},yaxis:{title:'metric'},yaxis2:{title:'가속도 [m/s²]',overlaying:'y',side:'right'},hovermode:'x unified'});document.getElementById('chart').on('plotly_click',e=>{let t=e.points[0].x;draw(frames.reduce((best,f,i)=>Math.abs(f.t-t)<Math.abs(frames[best].t-t)?i:best,0))});document.getElementById('sources').textContent=DATA.sources.join('\n');document.getElementById('coverage').textContent=frames.length+'개 재생 프레임 · 약 10 Hz';await draw(peak);
})().catch(error=>{document.getElementById('stats').textContent='재생 오류: '+error.message;console.error(error)});
</script></html>'''
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',payload))
 print(json.dumps(dict(html=str(out),frames=len(frames),meshes=len(meshes),vertices=sum(len(m['v']) for m in meshes),bytes=out.stat().st_size)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--observation',type=Path,required=True);p.add_argument('--joints',type=Path,required=True);p.add_argument('--urdf',type=Path,default=ROOT/'urdf/rb10_1300e.urdf');p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.observation,a.joints,a.urdf,a.output)

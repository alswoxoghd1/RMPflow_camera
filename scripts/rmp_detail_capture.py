"""Structured sidecar capture used by RmpDataRecorder; no control publications."""
import json
import math
import hashlib
from pathlib import Path
from collections import OrderedDict
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tg_initial_events import TOPIC as INITIAL_TOPIC, decode_initial_event, event_key, FIELDS as INITIAL_FIELDS
from rosidl_runtime_py.convert import message_to_ordereddict
from rcl_interfaces.srv import ListParameters, GetParameters
from rcl_interfaces.msg import ParameterEvent
from sensor_msgs.msg import JointState, Range
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, UInt8
from rmp_history_record import decode_history, RecordingQuality, SCHEMA as HISTORY_SCHEMA


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else '+Infinity' if value > 0 else '-Infinity'
    if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    return value


class DetailCapture:
    def __init__(self, node):
        self.node=node
        self.stream=None
        self.path=None
        self.count=0
        self.topic_counts={}
        self.parameters=None
        self.pending=False
        self.quality=RecordingQuality()
        self.generation=0
        self.deduplicated_history_messages=0
        node.declare_parameter('snapshot_files', [''])
        node.declare_parameter('snapshot_urdf_file', '')
        node.declare_parameter('snapshot_params_file', '')
        self.initial_cache=OrderedDict()
        self.initial_written=set()
        self.initial_event_errors=0
        self.initial_queue_dropped=0
        node.declare_parameter('record_detail_raw_topics', False)
        self.raw_topics=bool(node.get_parameter('record_detail_raw_topics').value)
        from tg_escape_feedback import EscapeFeedback
        self.escape_feedback=EscapeFeedback()
        self.subscriptions=[]
        self.subscriptions.append(node.create_subscription(JointState,
            "/rb10/measured_joint_states", self.receive_encoder, 50))
        topics={
            '/rmp_recording_snapshot':Float64MultiArray,
            '/tangent_escape_rmp_data':Float64MultiArray,
            '/rmp_leaf_ablation':Float64MultiArray,
            '/rmp_joint_accel':Float64MultiArray,
            '/target_q':Float64MultiArray,
            '/rmp_position_command':Float64MultiArray,
            '/joint_states':JointState,
            '/rb10/reference_joint_states':JointState,
            '/rb10/measured_joint_states':JointState,
            '/goal_pose':PoseStamped,
            '/RMP_flag':UInt8,
            '/tg_rmp/sensor_ranges':Range,
            '/parameter_events':ParameterEvent,
        }
        if self.raw_topics:
            for topic in node.range_topics: topics[topic]=Range
        else:
            topics={k:v for k,v in topics.items() if k in
                    ('/rmp_recording_snapshot','/RMP_flag','/parameter_events')}
        self.subscriptions.append(node.create_subscription(Float64MultiArray,INITIAL_TOPIC,
            self.receive_initial,QoSProfile(depth=64,reliability=ReliabilityPolicy.RELIABLE,
                                          durability=DurabilityPolicy.TRANSIENT_LOCAL)))
        for topic,kind in topics.items():
            self.subscriptions.append(node.create_subscription(kind,topic,
                lambda msg,t=topic:self.receive(t,msg),100))
        self.list_client=node.create_client(ListParameters,'/rmpflow_controller/list_parameters')
        self.get_client=node.create_client(GetParameters,'/rmpflow_controller/get_parameters')
        self.timer=node.create_timer(2.,self.poll_parameters)

    def start(self, csv_path):
        self.escape_feedback.frames.clear()
        self.generation+=1
        self.quality=RecordingQuality()
        self.initial_written=set()
        self.initial_event_errors=0
        self.deduplicated_history_messages=0
        self.parameters=None  # obtain a fresh effective configuration for every recording
        self.path=Path(csv_path).with_suffix('.details.jsonl')
        self.stream=self.path.open('x',buffering=65536)
        self.count=0
        self.topic_counts={}
        import uuid
        self.metadata={'schema':1,'run_id':str(uuid.uuid4()),'history_trace_schema':HISTORY_SCHEMA,
            'nonfinite_encoding':['NaN','+Infinity','-Infinity'],
            'clock_note':'received_ros_ns/received_steady_ns are arrival clocks; snapshot[2] is controller publish ROS time; snapshot[3] is cycle-start steady time; sequence counts command cycles',
            'root_group_names':{1:'cspace_target',2:'joint_limits',3:'joint_velocity_cap',4:'target',5:'collision',6:'tg',7:'damping',8:'body_target',9:'axis_x',10:'axis_y',11:'axis_z',12:'wrist_axis',13:'external',99:'other'},
            'units':{'q':'rad','qd':'rad/s','qdd':'rad/s^2','position':'m','clock':'s; arrival fields ns'},
            'snapshot_schema':{'prefix':['version','control_sequence','ros_time_s','steady_time_s','solve_ok','backend_application_enabled'],
                'vectors_6':['backend_q','backend_qd','solver_q','solver_qd','limited_qdd','command_q','command_qd'],
                'goal_7':'xyz quaternion_xyzw', 'geometry':'count then point xyz radius J_row_major[18]',
                'guards':'clamped,velocity_slew_clamped,min_z_triggered,pre_guard_command[6],command_step[6]',
                'obstacles':'count then center_xyz radius sensor_index (proxies for capacitive detections)',
                'diagnostics':'length then leaf_ablation data (zero on compact cycles); length then TG data',
                'ablation_extension':'after 25+17*group_count: 42042,count,(group_id,M_row_major[36],f[6])*count; then 44044,count,point(J_row[6],curvature_row)*3 per point,TCP(position_row,J_row[6],curvature_row)*3; then 43043,root_M[36],root_f[6],raw_qdd[6]'},
            'limitations':['grouped root contributions, not individual obstacle-pair task policies',
                'controller_parameters are ROS service values at the stamped query time; history.effective_parameters are values actually used by the constructed solver',
                'command is the state sent to backend apply_command, before any backend-internal conversion/guard',
                '44044 contains solver-native Jacobians and curvature; top-level geometry uses the RB10 visualization model',
                'backend_q follows real_joint_state_source; measured/reference debug streams may be disabled',
                'compact state/command/TG packets are emitted each command cycle; root matrices/geometry only on detailed cycles'],
            'files':[], 'controller_parameters':self.parameters,
            'detail_profile':'raw_topics' if self.raw_topics else 'coherent_snapshot',
            'escape_feedback_schema':{'topic':'/tg_rmp/escape_feedback','source':'independent encoder increments, frozen sensor Jacobian','horizon_s':.12,'position_deadband_m':.0005,'equivalent_range_noise_floor_m':.002,'threshold_note':'initial floors, not calibrated stationary noise; unknown sensors remain unknown'},
            'initial_event_schema':{'topic':INITIAL_TOPIC,'magic':46046,'version':2,'supported_versions':[1,2],'fields':INITIAL_FIELDS,'v2_appendix':'weighted escape scores[8]'},
            'omitted_duplicate_topics':[] if self.raw_topics else [
                '/joint_states','/rb10/reference_joint_states','/rb10/measured_joint_states',
                '/target_q','/rmp_position_command','/rmp_joint_accel','/rmp_leaf_ablation',
                '/tangent_escape_rmp_data','/goal_pose','/tg_rmp/sensor_ranges','/raw_distance*']}
        # Transient-local events received before recording started are retained
        # with their original source timestamps, not relabelled as current data.
        for entry in self.initial_cache.values():
            self.write_initial(entry)
        for name in list(self.node.get_parameter('snapshot_files').value) + [self.node.get_parameter(key).value for key in ('snapshot_urdf_file','snapshot_params_file')]:
            source=Path(name)
            if source.is_file():
                content=source.read_bytes()
                destination=self.path.with_name(self.path.stem+'_'+source.name)
                destination.write_bytes(content)
                self.metadata['files'].append({'source':str(source),'saved':str(destination),'sha256':hashlib.sha256(content).hexdigest()})
        self.save_metadata()
        self.capture_provenance()

    def save_metadata(self):
        if self.path:
            self.metadata['recording_quality']=self.quality.summary()
            self.metadata['parameter_snapshot_complete']=self.parameters is not None
            parameters=self.parameters or {}
            expected=(parameters.get('tg_rmp_direction_mode',{}).get('string_value')=='sensor_history'
                      and parameters.get('enable_tangent_escape_rmp',{}).get('bool_value',False))
            self.metadata['history_trace_expected']=expected if self.parameters is not None else None
            self.metadata['history_trace_received']=self.quality.counts['history_frames']>0
            self.metadata['deduplicated_history_messages']=self.deduplicated_history_messages
            self.metadata['initial_events_saved']=len(self.initial_written)
            self.metadata['initial_event_decode_errors']=self.initial_event_errors
            self.metadata['initial_event_queue_dropped']=self.initial_queue_dropped
            self.metadata['initial_event_note']='Reliable transient-local events; bounded 64-slot controller queue and DDS history. Source clocks/cycle identify the original selection; cached pre-recording events can precede the first snapshot.'
            self.metadata['history_topic_note']='Extended /tangent_escape_rmp_data duplicates are omitted; the identical complete TG packet is stored inside /rmp_recording_snapshot.'
            path=self.path.with_suffix('.metadata.json')
            temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(clean(self.metadata),ensure_ascii=False,indent=2,allow_nan=False))
            temporary.replace(path)

    def write_initial(self,entry):
        key=event_key(entry['message']['data'])
        if key in self.initial_written:return
        self.stream.write(json.dumps(entry,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
        self.initial_written.add(key)
        self.count+=1
        self.topic_counts[INITIAL_TOPIC]=self.topic_counts.get(INITIAL_TOPIC,0)+1

    def receive_initial(self,msg):
        import time
        data=list(msg.data)
        try:
            event=decode_initial_event(data)
        except (ValueError,IndexError,TypeError,OverflowError):
            self.initial_event_errors+=1;return
        entry={'topic':INITIAL_TOPIC,'received_ros_ns':self.node.get_clock().now().nanoseconds,
               'received_steady_ns':time.monotonic_ns(),'message':{'data':data}}
        with self.node.file_lock:
            self.initial_queue_dropped=max(self.initial_queue_dropped,event['queue_dropped'])
            self.initial_cache[event_key(data)]=entry
            while len(self.initial_cache)>64:self.initial_cache.popitem(last=False)
            if self.stream and self.node.is_recording:self.write_initial(entry)

    def receive_encoder(self,msg):
        names=("base","shoulder","elbow","wrist1","wrist2","wrist3")
        if not all(n in msg.name for n in names): return
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        self.escape_feedback.encoder(stamp,[msg.position[msg.name.index(n)] for n in names])

    def receive(self,topic,msg):
        import time
        if not self.stream or not self.node.is_recording:return
        if topic=='/tangent_escape_rmp_data' and len(msg.data)>=56:
            offset=114+8*int(msg.data[50])
            if len(msg.data)>offset and msg.data[offset]==45045.0:
                if self.stream and self.node.is_recording:self.deduplicated_history_messages+=1
                return
        # ROS's generic recursive conversion is costly for large numeric arrays.
        # Keep the same on-disk message shape while copying the numeric buffer once.
        message=({'layout':message_to_ordereddict(msg.layout),'data':list(msg.data)}
                 if isinstance(msg,Float64MultiArray) else message_to_ordereddict(msg))
        entry={'topic':topic,'received_ros_ns':self.node.get_clock().now().nanoseconds,
               'received_steady_ns':time.monotonic_ns(),'message':clean(message)}
        with self.node.file_lock:
            if self.stream and self.node.is_recording:
                self.stream.write(json.dumps(entry,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
                self.count+=1
                self.topic_counts[topic]=self.topic_counts.get(topic,0)+1
                if topic == '/rmp_recording_snapshot':
                    try:
                        snapshot=decode_snapshot(msg.data)
                        self.quality.observe(snapshot)
                        feedback=self.escape_feedback.observe(snapshot)
                        if feedback is not None:
                            derived=dict(entry,topic='/tg_rmp/escape_feedback',message=feedback)
                            self.stream.write(json.dumps(clean(derived),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
                            self.count+=1
                            self.topic_counts['/tg_rmp/escape_feedback']=self.topic_counts.get('/tg_rmp/escape_feedback',0)+1
                    except (ValueError, IndexError, TypeError, OverflowError):
                        self.quality.counts['decode_errors']+=1

    def poll_parameters(self):
        with self.node.file_lock:
            if self.stream:
                self.stream.flush()
                self.metadata['messages']=self.count
                self.metadata['topic_counts']=dict(self.topic_counts)
                self.metadata['recording_quality']=self.quality.summary()
                self.metadata['parameter_snapshot_complete']=self.parameters is not None
                self.save_metadata()
        if self.pending or self.parameters is not None or not self.list_client.service_is_ready() or not self.get_client.service_is_ready(): return
        self.pending=True
        generation=self.generation
        future=self.list_client.call_async(ListParameters.Request(depth=0))
        future.add_done_callback(lambda response:self.listed(response,generation))

    def listed(self,future,generation):
        try:
            names=future.result().result.names
            request=GetParameters.Request(names=names)
            result=self.get_client.call_async(request)
            result.add_done_callback(lambda response:self.got(names,response,generation))
        except Exception:
            self.pending=False

    def got(self,names,future,generation):
        try:
            if generation != self.generation: return
            self.parameters={key:message_to_ordereddict(value) for key,value in zip(names,future.result().values)}
            with self.node.file_lock:
                if self.stream:
                    self.metadata['controller_parameters']=self.parameters
                    self.metadata['parameters_received_ros_ns']=self.node.get_clock().now().nanoseconds
                    self.metadata['parameter_snapshot_complete']=True
                    build_id=self.parameters.get('diagnostic_build_source_id',{}).get('string_value')
                    self.metadata['running_build_source_id']=build_id
                    manifest_id=self.metadata.get('build_source_manifest',{}).get('source_id')
                    self.metadata['runtime_build_matches_manifest']=(build_id==manifest_id) if build_id and manifest_id else None
                    self.save_metadata()
        except Exception as error:
            self.parameters=None
            self.node.get_logger().warning('Parameter snapshot will retry: '+str(error))
        finally:
            self.pending=False

    def stop(self):
        if self.stream:
            self.stream.flush();self.stream.close();self.stream=None
            self.metadata['messages']=self.count
            self.metadata['topic_counts']=dict(self.topic_counts)
            self.metadata['recording_quality']=self.quality.summary()
            self.metadata['parameter_snapshot_complete']=self.parameters is not None
            self.save_metadata()
            self.node.get_logger().info('Recording quality: '+json.dumps(self.metadata['recording_quality']))
            if (self.quality.counts['snapshots']==0 or self.quality.counts['history_missing_trace'] or
                    self.quality.counts['decode_errors'] or self.quality.counts['direction_update_mismatches'] or
                    self.initial_event_errors or self.initial_queue_dropped or
                    (self.metadata['history_trace_expected'] and not self.metadata['history_trace_received']) or
                    self.parameters is None):
                self.node.get_logger().warning('Detailed recording incomplete; inspect .details.metadata.json')

    def capture_provenance(self):
        """Filesystem work runs in the recorder, never in the control loop."""
        import subprocess
        import zipfile
        from ament_index_python.packages import get_package_share_directory, get_package_prefix
        try:
            package='rb10_rmpflow_rviz'
            share=Path(get_package_share_directory(package))
            repo=Path(__file__).resolve().parents[1]
            manifest=share/'rmp_build_sources.sha256'
            if manifest.is_file():
                data=manifest.read_bytes()
                saved=self.path.with_name(self.path.stem+'_build_sources.sha256')
                saved.write_bytes(data)
                self.metadata['build_source_manifest']={'saved':str(saved),'source_id':hashlib.sha256(data).hexdigest()}
                archive=self.path.with_name(self.path.stem+'_sources.zip')
                mismatches=[]
                with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
                    bundle.writestr('build_sources.sha256',data)
                    for line in data.decode().splitlines():
                        expected,name=line.split('  ',1)
                        source=repo/name
                        if not source.is_file():
                            mismatches.append(name);continue
                        content=source.read_bytes()
                        if hashlib.sha256(content).hexdigest()!=expected:mismatches.append(name)
                        bundle.writestr(name,content)
                    for name in ('rmp_detail_capture.py','rmp_history_record.py','tg_initial_events.py','tg_escape_feedback.py'):
                        source=Path(__file__).resolve().with_name(name)
                        if source.is_file():bundle.writestr('scripts/'+name,source.read_bytes())
                self.metadata['source_archive']={'saved':str(archive),
                    'matches_build_manifest':not mismatches,'mismatched_files':mismatches,
                    'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
            folder=Path(get_package_prefix(package))/'lib'/package
            self.metadata['installed_binary_sha256']={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                for name in ('rmpflow_controller','SoC_rmpflow_controller') if (folder/name).is_file()}
            result=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True,timeout=3)
            self.metadata['git_head_at_record_start']=result.stdout.strip() if result.returncode==0 else None
            self.metadata['provenance_note']='Runtime diagnostic_build_source_id identifies compiled source; installed binary hashes are file snapshots, not proof of the running process image.'
            self.save_metadata()
        except Exception as error:
            self.metadata['provenance_error']=str(error)
            self.save_metadata()


def decode_snapshot(data, *, include_history=True):
    """Decode a version-1 coherent control snapshot; reject truncated packets."""
    cursor=0
    def take(count):
        nonlocal cursor
        if cursor+count>len(data): raise ValueError('truncated snapshot')
        result=list(data[cursor:cursor+count]);cursor+=count;return result
    prefix=take(6)
    if prefix[0]!=1: raise ValueError('unsupported snapshot version')
    result=dict(zip(('version','control_sequence','ros_time_s','steady_time_s','solve_ok','backend_application_enabled'),prefix))
    for name in ('backend_q','backend_qd','solver_q','solver_qd','limited_qdd','command_q','command_qd'):
        result[name]=take(6)
    result['goal_pose']=take(7)
    result['visualization_points']=[take(22) for _ in range(int(take(1)[0]))]
    result['guard_flags']=take(3)
    result['pre_guard_command']=take(6);result['command_step']=take(6)
    result['obstacle_proxies']=[take(5) for _ in range(int(take(1)[0]))]
    ablation=take(int(take(1)[0])); result['tg']=take(int(take(1)[0]))
    result['history']=decode_history(result['tg']) if include_history else None
    if cursor!=len(data): raise ValueError('unexpected trailing snapshot data')
    if not ablation:
        result.update(details_decimated=bool(result['solve_ok']), solve_diagnostics_missing=not bool(result['solve_ok']), groups=[], root_metric=None, root_force=None)
        return result
    if len(ablation)<25: raise ValueError('missing solve diagnostics')
    count=int(ablation[1]); offset=25+17*count
    if ablation[offset:offset+1]!=[42042.]: raise ValueError('missing root group matrices')
    offset+=2;groups=[]
    for _ in range(count):
        groups.append(dict(group_id=int(ablation[offset]),metric=ablation[offset+1:offset+37],force=ablation[offset+37:offset+43]))
        offset+=43
    if ablation[offset:offset+1]!=[44044.]: raise ValueError('missing solver kinematics')
    points=int(ablation[offset+1]);offset+=2
    result['solver_point_J_curvature']=[ablation[offset+i*21:offset+(i+1)*21] for i in range(points)]
    offset+=points*21
    result['solver_tcp_position_J_curvature']=ablation[offset:offset+24];offset+=24
    if ablation[offset:offset+1]!=[43043.]: raise ValueError('missing complete root')
    result['root_metric']=ablation[offset+1:offset+37]
    result['root_force']=ablation[offset+37:offset+43]
    result['raw_qdd']=ablation[offset+43:offset+49]
    if len(result['raw_qdd'])!=6: raise ValueError('truncated root solve')
    result['groups']=groups
    return result

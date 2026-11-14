"""Visualization only: no controller or hardware bridge is launched."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction,RegisterEventHandler,Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def actions(context):
 package=Path(get_package_share_directory('rb10_rmpflow_rviz'))
 override=LaunchConfiguration('urdf').perform(context)
 urdf=Path(override).expanduser() if override else package/'urdf/rb10_1300e.urdf'
 dataset=LaunchConfiguration('dataset').perform(context)
 if not dataset:
  folder=Path.home()/'ros2_ws/data/rmp_datasets'
  files=sorted(folder.glob('*.details.jsonl'),key=lambda p:p.stat().st_mtime,reverse=True)
  selected=next((p for p in files if p.stat().st_size),None)
  if selected is None:
   files=list(folder.glob('*.csv'))
   selected=max(files,key=lambda p:p.stat().st_mtime) if files else None
  if selected is not None:dataset=str(selected)
 else:dataset=str(Path(dataset).expanduser())
 if not override and dataset.endswith('.jsonl'):
  meta=Path(dataset).with_suffix('.metadata.json')
  if meta.exists():
   import json
   for entry in json.loads(meta.read_text()).get('files',[]):
    if str(entry.get('saved','')).endswith('.urdf') and Path(entry['saved']).is_file():urdf=Path(entry['saved']);break
 env={'ROS_DOMAIN_ID':LaunchConfiguration('replay_domain').perform(context),
      'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
 replay=Node(package='rb10_rmpflow_rviz',executable='rmp_rviz_replay.py',namespace='rmp_replay',name='replay',output='screen',additional_env=env,parameters=[{
  'dataset':dataset,'observation_file':LaunchConfiguration('observation_file'),
  'control_panel':ParameterValue(LaunchConfiguration('control_panel'),value_type=bool),
  'paused':True,'speed':ParameterValue(LaunchConfiguration('speed'),value_type=float)}])
 state=Node(package='robot_state_publisher',executable='robot_state_publisher',namespace='rmp_replay',additional_env=env,parameters=[{'robot_description':urdf.read_text(),'frame_prefix':'rmp_replay/'}])
 view=Node(package='rviz2',executable='rviz2',additional_env=env,arguments=['-d',str(package/'config/rmp_recording_replay.rviz')])
 return [state,replay,view,RegisterEventHandler(OnProcessExit(target_action=view,on_exit=[Shutdown(reason='Replay window closed')])),RegisterEventHandler(OnProcessExit(target_action=replay,on_exit=[Shutdown(reason='Replay controls closed')]))]


def generate_launch_description():
 package=Path(get_package_share_directory('rb10_rmpflow_rviz'))
 return LaunchDescription([
 DeclareLaunchArgument('dataset',default_value='',description='Recorded CSV or detailed JSONL; empty selects newest available recording.'),
 DeclareLaunchArgument('observation_file',default_value='',description='Matching TG observation CSV for legacy joint CSV replay.'),
 DeclareLaunchArgument('urdf',default_value='',description='Optional override; otherwise use the recording URDF.'),
 DeclareLaunchArgument('speed',default_value='1.0'),
 DeclareLaunchArgument('control_panel',default_value='true'),
 DeclareLaunchArgument('replay_domain',default_value='213',description='Separate ROS domain used by all visualization processes.'),
 OpaqueFunction(function=actions)])

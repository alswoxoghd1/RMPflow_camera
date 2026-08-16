#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _validate_launch(context):
    params_path = LaunchConfiguration("params_file").perform(context)
    if not os.path.isfile(params_path):
        raise RuntimeError(
            f'Launch argument "params_file" must point to an existing YAML file: {params_path}'
        )

    controller_box_simulation = _as_bool(
        LaunchConfiguration("controller_box_simulation").perform(context)
    )
    if controller_box_simulation:
        if _as_bool(LaunchConfiguration("cb_simulation").perform(context)):
            raise RuntimeError(
                "controller_box_simulation:=true cannot be combined with "
                "cb_simulation:=true. Select exactly one simulation backend."
            )
        if not _as_bool(
            LaunchConfiguration("hardware_command_enabled").perform(context)
        ):
            raise RuntimeError(
                "controller_box_simulation:=true requires "
                "hardware_command_enabled:=true because Controller Box simulation "
                "uses the RB10 command socket."
            )
    return []


def generate_launch_description():
    rmp_pkg = get_package_share_directory("rb10_rmpflow_rviz")
    config_dir = os.path.join(rmp_pkg, "config")
    default_params_path = os.path.join(config_dir, "params.yaml")

    urdf_path = os.path.join(rmp_pkg, "urdf", "rb10_1300e.urdf")
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    rviz_config_path = os.path.join(config_dir, "rb10_rmpflow.rviz")

    robot_ip = LaunchConfiguration("robot_ip")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    control_rate = LaunchConfiguration("control_rate")
    solver_backend = LaunchConfiguration("solver_backend")
    casadi_rmp2_max_obstacles = LaunchConfiguration("casadi_rmp2_max_obstacles")
    casadi_rmp2_jit = LaunchConfiguration("casadi_rmp2_jit")
    publish_control_timing_debug = LaunchConfiguration(
        "publish_control_timing_debug"
    )
    control_timing_debug_topic = LaunchConfiguration(
        "control_timing_debug_topic"
    )
    control_timing_publish_rate = LaunchConfiguration(
        "control_timing_publish_rate"
    )
    cb_simulation = LaunchConfiguration("cb_simulation")
    controller_box_simulation = LaunchConfiguration("controller_box_simulation")
    use_direct_hardware_backend = LaunchConfiguration("use_direct_hardware_backend")
    hardware_command_enabled = LaunchConfiguration("hardware_command_enabled")
    dry_run_command_topic = LaunchConfiguration("dry_run_command_topic")
    real_joint_state_source = LaunchConfiguration("real_joint_state_source")
    publish_debug_joint_state_sources = LaunchConfiguration("publish_debug_joint_state_sources")
    publish_rmp_ee_pose = LaunchConfiguration("publish_rmp_ee_pose")
    use_interactive_goal = LaunchConfiguration("use_interactive_goal")
    use_obstacles = LaunchConfiguration("use_obstacles")
    use_proximity_bridge = LaunchConfiguration("use_proximity_bridge")
    proximity_surface_visualization = LaunchConfiguration("proximity_surface_visualization")
    proximity_collision_obstacle_topic = LaunchConfiguration("proximity_collision_obstacle_topic")
    proximity_visualization_obstacle_topic = LaunchConfiguration("proximity_visualization_obstacle_topic")
    record_data = LaunchConfiguration("record_data")
    auto_start_recording = LaunchConfiguration("auto_start_recording")
    recording_rate = LaunchConfiguration("recording_rate")
    recording_output_directory = LaunchConfiguration("recording_output_directory")
    recording_output_prefix = LaunchConfiguration("recording_output_prefix")
    recorder_range_topics = [f"/raw_distance{index}" for index in range(1, 21)]
    use_rmpflow_trace_logger = LaunchConfiguration("use_rmpflow_trace_logger")
    rmpflow_trace_log_rate = LaunchConfiguration("rmpflow_trace_log_rate")
    rmpflow_trace_log_directory = LaunchConfiguration("rmpflow_trace_log_directory")
    rmpflow_trace_console_summary = LaunchConfiguration("rmpflow_trace_console_summary")
    servo_t1 = LaunchConfiguration("servo_t1")
    servo_t2 = LaunchConfiguration("servo_t2")
    servo_gain = LaunchConfiguration("servo_gain")
    servo_alpha = LaunchConfiguration("servo_alpha")
    command_mode = LaunchConfiguration("command_mode")
    startup_move_to_default_pose = LaunchConfiguration("startup_move_to_default_pose")
    startup_movej_speed = LaunchConfiguration("startup_movej_speed")
    startup_movej_accel = LaunchConfiguration("startup_movej_accel")
    startup_release_timeout_sec = LaunchConfiguration("startup_release_timeout_sec")
    stop_on_shutdown = LaunchConfiguration("stop_on_shutdown")
    shutdown_action = LaunchConfiguration("shutdown_action")
    enable_realtime = LaunchConfiguration("enable_realtime")
    realtime_priority = LaunchConfiguration("realtime_priority")
    lock_memory = LaunchConfiguration("lock_memory")
    enable_socket_realtime = LaunchConfiguration("enable_socket_realtime")
    socket_realtime_priority = LaunchConfiguration("socket_realtime_priority")
    bridge_max_command_step_deg = LaunchConfiguration("bridge_max_command_step_deg")
    bridge_max_command_velocity_deg_s = LaunchConfiguration("bridge_max_command_velocity_deg_s")
    bridge_large_command_jump_warn_deg = LaunchConfiguration("bridge_large_command_jump_warn_deg")
    bridge_publish_only_new_state_packets = LaunchConfiguration(
        "bridge_publish_only_new_state_packets"
    )
    bridge_state_max_raw_velocity_rad_s = LaunchConfiguration(
        "bridge_state_max_raw_velocity_rad_s"
    )
    bridge_state_position_step_margin_rad = LaunchConfiguration(
        "bridge_state_position_step_margin_rad"
    )
    bridge_publish_rate = LaunchConfiguration("bridge_publish_rate")
    record_joint_velocity = LaunchConfiguration("record_joint_velocity")
    joint_velocity_log_directory = LaunchConfiguration("joint_velocity_log_directory")
    joint_velocity_log_prefix = LaunchConfiguration("joint_velocity_log_prefix")
    params_file = LaunchConfiguration("params_file")
    controller_parameters = {
        "robot_ip": robot_ip,
        "solver_backend": solver_backend,
        "control_rate": ParameterValue(control_rate, value_type=float),
        "casadi_rmp2_max_obstacles": ParameterValue(
            casadi_rmp2_max_obstacles,
            value_type=int,
        ),
        "casadi_rmp2_jit": ParameterValue(casadi_rmp2_jit, value_type=bool),
        "publish_control_timing_debug": ParameterValue(
            publish_control_timing_debug,
            value_type=bool,
        ),
        "control_timing_debug_topic": control_timing_debug_topic,
        "control_timing_publish_rate": ParameterValue(
            control_timing_publish_rate,
            value_type=float,
        ),
        # controller_box_simulation is an explicit networked Controller Box
        # simulation session. cb_simulation remains the socket-free local mode.
        "simulation_mode": ParameterValue(
            controller_box_simulation,
            value_type=bool,
        ),
        "real_joint_state_source": real_joint_state_source,
        "hardware_data_request_rate": bridge_publish_rate,
        "joint_state_topic": "/joint_states",
        "position_command_topic": "/position_controllers/commands",
        "hardware_command_enabled": ParameterValue(
            PythonExpression([
                'False if "', cb_simulation,
                '" == "true" else (True if "', hardware_command_enabled,
                '" == "true" else False)'
            ]),
            value_type=bool,
        ),
        "dry_run_command_topic": dry_run_command_topic,
        "publish_position_command": True,
        "position_command_state_topic": "/rmp_position_command",
        "command_mode": command_mode,
        "publish_target_q": True,
        "target_q_topic": "/target_q",
        "publish_joint_states": ParameterValue(
            PythonExpression([
                'True if "', cb_simulation, '" == "true" or "',
                use_direct_hardware_backend, '" == "true" else False'
            ]),
            value_type=bool,
        ),
        "backend_mode": ParameterValue(
            PythonExpression([
                '"simulation" if "', cb_simulation,
                '" == "true" else ("rb10_direct_api" if "',
                use_direct_hardware_backend,
                '" == "true" else "joint_command_topics")'
            ]),
            value_type=str,
        ),
        "servo_t1": servo_t1,
        "servo_t2": servo_t2,
        "servo_gain": servo_gain,
        "servo_alpha": servo_alpha,
        "startup_move_to_default_pose": startup_move_to_default_pose,
        "startup_movej_speed": startup_movej_speed,
        "startup_movej_accel": startup_movej_accel,
        "startup_release_timeout_sec": startup_release_timeout_sec,
        "stop_on_shutdown": stop_on_shutdown,
        "shutdown_action": shutdown_action,
        "enable_realtime": enable_realtime,
        "realtime_priority": realtime_priority,
        "lock_memory": lock_memory,
        "enable_socket_realtime": enable_socket_realtime,
        "socket_realtime_priority": socket_realtime_priority,
        "publish_debug_joint_state_sources": publish_debug_joint_state_sources,
        "publish_rmp_ee_pose": ParameterValue(
            publish_rmp_ee_pose,
            value_type=bool,
        ),
    }

    api_bridge = Node(
        package="rb10_rmpflow_rviz",
        executable="rb10_api_bridge.py",
        name="rb10_api_bridge",
        output="screen",
        condition=IfCondition(PythonExpression([
            '"', cb_simulation, '" != "true" and "',
            use_direct_hardware_backend, '" != "true" and "',
            hardware_command_enabled, '" == "true"',
        ])),
        parameters=[
            params_file,
            {
                "robot_ip": robot_ip,
                "simulation_mode": ParameterValue(
                    controller_box_simulation,
                    value_type=bool,
                ),
                "command_mode": command_mode,
                "command_topic": "/position_controllers/commands",
                "joint_state_topic": "/joint_states",
                "real_joint_state_source": real_joint_state_source,
                "publish_rate": bridge_publish_rate,
                "servo_t1": servo_t1,
                "servo_t2": servo_t2,
                "servo_gain": servo_gain,
                "servo_alpha": servo_alpha,
                "max_command_step_deg": bridge_max_command_step_deg,
                "max_command_velocity_deg_s": bridge_max_command_velocity_deg_s,
                "large_command_jump_warn_deg": bridge_large_command_jump_warn_deg,
                "publish_only_new_state_packets": ParameterValue(
                    bridge_publish_only_new_state_packets,
                    value_type=bool,
                ),
                "state_max_raw_velocity_rad_s": ParameterValue(
                    bridge_state_max_raw_velocity_rad_s,
                    value_type=float,
                ),
                "state_position_step_margin_rad": ParameterValue(
                    bridge_state_position_step_margin_rad,
                    value_type=float,
                ),
                "stop_on_shutdown": stop_on_shutdown,
                "shutdown_action": shutdown_action,
            },
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "publish_frequency": 50.0,
        }],
        remappings=[
            ("robot_description", "/rb10/robot_description"),
        ],
    )

    rmpflow_controller = Node(
        package="rb10_rmpflow_rviz",
        executable="rmpflow_controller",
        name="rmpflow_controller",
        output="screen",
        parameters=[
            params_file,
            controller_parameters,
        ],
    )

    interactive_goal = Node(
        package="rb10_rmpflow_rviz",
        executable="interactive_goal",
        name="interactive_goal",
        output="screen",
        parameters=[
            params_file,
            {
                "lock_orientation_to_tcp": False,
            },
        ],
        condition=IfCondition(use_interactive_goal),
    )

    obstacle_manager = Node(
        package="rb10_rmpflow_rviz",
        executable="obstacle_manager",
        name="obstacle_manager",
        output="screen",
        condition=IfCondition(PythonExpression([
            '"', use_obstacles, '" == "true" and "',
            use_proximity_bridge, '" != "true"',
        ])),
        parameters=[params_file],
    )

    proximity_obstacle_bridge = Node(
        package="rb10_rmpflow_rviz",
        executable="proximity_obstacle_bridge",
        name="proximity_obstacle_bridge",
        output="screen",
        condition=IfCondition(PythonExpression([
            '"', use_proximity_bridge, '" == "true" or "',
            proximity_surface_visualization, '" == "true"',
        ])),
        parameters=[
            params_file,
            {
                "publish_collision_obstacles": ParameterValue(
                    PythonExpression([
                        'True if "', use_proximity_bridge, '" == "true" else False'
                    ]),
                    value_type=bool,
                ),
                "surface_patch_fixed_visualization": ParameterValue(
                    PythonExpression([
                        'True if "', proximity_surface_visualization,
                        '" == "true" else False'
                    ]),
                    value_type=bool,
                ),
                "obstacle_topic": proximity_collision_obstacle_topic,
                "visualization_obstacle_topic": proximity_visualization_obstacle_topic,
            },
        ],
    )

    data_recorder = Node(
        package="rb10_rmpflow_rviz",
        executable="rmp_data_recorder.py",
        name="rmp_data_recorder",
        output="screen",
        condition=IfCondition(record_data),
        parameters=[{
            "mode": ParameterValue(
                PythonExpression([
                    '"simulation" if "', cb_simulation,
                    '" == "true" else "real"'
                ]),
                value_type=str,
            ),
            "auto_start": auto_start_recording,
            "recording_rate": recording_rate,
            "output_directory": recording_output_directory,
            "output_prefix": recording_output_prefix,
            "joint_state_topic": "/joint_states",
            "range_topics": recorder_range_topics,
        }],
    )


    joint_velocity_logger = Node(
        package="rb10_rmpflow_rviz",
        executable="joint_velocity_logger.py",
        name="joint_velocity_logger",
        output="screen",
        condition=IfCondition(record_joint_velocity),
        parameters=[{
            "joint_state_topic": "/joint_states",
            "output_directory": joint_velocity_log_directory,
            "output_prefix": joint_velocity_log_prefix,
            "flush_every": 1,
        }],
    )

    rmpflow_trace_logger = Node(
        package="rb10_rmpflow_rviz",
        executable="rmpflow_trace_logger.py",
        name="rmpflow_trace_logger",
        output="screen",
        condition=IfCondition(use_rmpflow_trace_logger),
        parameters=[{
            "log_rate_hz": ParameterValue(rmpflow_trace_log_rate, value_type=float),
            "console_summary": ParameterValue(
                PythonExpression([
                    'True if "', rmpflow_trace_console_summary, '" == "true" else False'
                ]),
                value_type=bool,
            ),
            "output_directory": rmpflow_trace_log_directory,
            "rmp_flag_topic": "/RMP_flag",
            "external_goal_topic": "/RMP_goal",
            "controller_goal_topic": "/goal_pose",
            "joint_state_topic": "/joint_states",
            "command_topic": ParameterValue(
                PythonExpression([
                    '"/position_controllers/commands" if "', cb_simulation,
                    '" != "true" and "',
                    hardware_command_enabled,
                    '" == "true" else "',
                    dry_run_command_topic,
                    '"',
                ]),
                value_type=str,
            ),
            "target_q_topic": "/target_q",
            "target_metric_topic": "/target_metric",
            "debug_state_topic": "/rmp_debug_state",
            "rmp_ee_pose_topic": "/rmp_ee_pose",
            "rmp_joint_accel_topic": "/rmp_joint_accel",
            "rmp_tcp_accel_topic": "/rmp_tcp_accel",
            "tangent_escape_rmp_data_topic": "/tangent_escape_rmp_data",
            "tangent_escape_dual_solve_topic": "/tangent_escape_dual_solve",
            "tangent_escape_geometry_data_topic": "/tangent_escape_geometry_data",
            "obstacle_marker_topic": "/obstacles",
            "repulsion_metric_marker_topic": "/repulsion_metric_markers",
            "tcp_accel_marker_topic": "/tcp_accel_marker",
            "range_scale": 0.001,
            "minimum_hold_distance": 0.05,
            "trigger_distance": 0.29,
            "range_topics": recorder_range_topics,
        }],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(use_rviz),
        arguments=["-d", rviz_config],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.111.50",
            description="Hostname or IP address of the RB10 controller.",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params_path,
            description="Absolute path to the experiment-specific controller params YAML.",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Start RViz.",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=rviz_config_path,
            description="Absolute path to the RViz config file.",
        ),
        DeclareLaunchArgument(
            "control_rate",
            default_value="200.0",
            description="RMP controller loop rate in Hz.",
        ),
        DeclareLaunchArgument(
            "solver_backend",
            default_value="pinocchio_direct",
            description=(
                "Runtime RMP solver: pinocchio_direct (real-time default) or "
                "casadi_rmp2 (reference/equivalence backend)."
            ),
        ),
        DeclareLaunchArgument("casadi_rmp2_max_obstacles", default_value="24"),
        DeclareLaunchArgument("casadi_rmp2_jit", default_value="true"),
        DeclareLaunchArgument(
            "publish_control_timing_debug",
            default_value="false",
            description=(
                "Publish low-rate cumulative controller/socket timing telemetry."
            ),
        ),
        DeclareLaunchArgument(
            "control_timing_debug_topic",
            default_value="/rmp_control_timing",
            description="Topic for low-rate controller timing telemetry.",
        ),
        DeclareLaunchArgument(
            "control_timing_publish_rate",
            default_value="10.0",
            description="Timing telemetry publication rate in Hz.",
        ),
        DeclareLaunchArgument(
            "cb_simulation",
            default_value="false",
            description=(
                "Use the local in-process simulation backend. This mode never opens "
                "RB10 hardware sockets or sends hardware commands."
            ),
        ),
        DeclareLaunchArgument(
            "controller_box_simulation",
            default_value="false",
            description=(
                "Connect to the RB10 Controller Box and request PG_MODE.SIMULATION. "
                "This still opens the real data/command sockets and sends controller "
                "initialization, mode, motion, and shutdown commands. Requires "
                "cb_simulation:=false, use_direct_hardware_backend:=false, and "
                "hardware_command_enabled:=true."
            ),
        ),
        DeclareLaunchArgument(
            "use_direct_hardware_backend",
            default_value="true",
            description=(
                "Bypass the ROS command bridge and let rmpflow_controller send RB10 API servo "
                "commands directly while keeping RViz/debug publication alive."
            ),
        ),
        DeclareLaunchArgument(
            "hardware_command_enabled",
            default_value="true",
            description=(
                "Allow RB10 initialization and motion commands. Set false for read-only "
                "dry-run; the direct backend then opens only the data socket."
            ),
        ),
        DeclareLaunchArgument(
            "dry_run_command_topic",
            default_value="/rmp_dry_run_command",
            description="Safe diagnostics-only topic for commands calculated in dry-run mode.",
        ),
        DeclareLaunchArgument(
            "real_joint_state_source",
            default_value="measured",
            description=(
                "Joint position source for /joint_states in real mode: "
                "measured uses encoder angle, reference uses controller reference angle."
            ),
        ),
        DeclareLaunchArgument(
            "publish_debug_joint_state_sources",
            default_value="false",
            description=(
                "Publish reference/measured joint-state debug topics and their tracking error "
                "for vibration-source diagnosis."
            ),
        ),
        DeclareLaunchArgument(
            "publish_rmp_ee_pose",
            default_value="false",
            description=(
                "Publish the current TCP pose on /rmp_ee_pose for arrival-based "
                "experiment_goal_runner trajectories."
            ),
        ),
        DeclareLaunchArgument(
            "use_interactive_goal",
            default_value="true",
            description="Start the interactive goal publisher.",
        ),
        DeclareLaunchArgument(
            "use_obstacles",
            default_value="true",
            description="Enable the interactive obstacle manager.",
        ),
        DeclareLaunchArgument(
            "use_proximity_bridge",
            default_value="false",
            description="Use external proximity topics to build obstacle markers.",
        ),
        DeclareLaunchArgument(
            "proximity_surface_visualization",
            default_value="true",
            description=(
                "Publish RViz-only fixed surface patches from proximity sensor range hits. "
                "When use_proximity_bridge is false this does not publish collision obstacles."
            ),
        ),
        DeclareLaunchArgument(
            "proximity_collision_obstacle_topic",
            default_value="obstacles",
            description="MarkerArray topic used by proximity_obstacle_bridge for RMP collision obstacles.",
        ),
        DeclareLaunchArgument(
            "proximity_visualization_obstacle_topic",
            default_value="obstacle_markers",
            description="MarkerArray topic used by proximity_obstacle_bridge for RViz obstacle markers.",
        ),
        DeclareLaunchArgument(
            "bridge_publish_rate",
            default_value="500.0",
            description="RB10 state receive/publish rate in Hz before filtering.",
        ),
        DeclareLaunchArgument(
            "bridge_publish_only_new_state_packets",
            default_value="true",
            description="Publish one JointState per new RB10 Controller Box state packet.",
        ),
        DeclareLaunchArgument(
            "bridge_state_max_raw_velocity_rad_s",
            default_value="6.0",
            description="Reject bridge state steps implying a faster joint velocity.",
        ),
        DeclareLaunchArgument(
            "bridge_state_position_step_margin_rad",
            default_value="0.01",
            description="Additional bridge position-step allowance above the velocity bound.",
        ),
        DeclareLaunchArgument(
            "record_data",
            default_value="false",
            description="Record RMP topics to a dataset file.",
        ),
        DeclareLaunchArgument(
            "use_rmpflow_trace_logger",
            default_value="false",
            description="Start the tangent/RMPFlow CSV trace logger.",
        ),
        DeclareLaunchArgument(
            "rmpflow_trace_log_rate",
            default_value="200.0",
            description="Trace logger snapshot rate in Hz.",
        ),
        DeclareLaunchArgument(
            "rmpflow_trace_log_directory",
            default_value=os.path.expanduser("~/ros2_ws/log/rmpflow_trace"),
            description="Directory for RMPFlow trace CSV logs.",
        ),
        DeclareLaunchArgument(
            "rmpflow_trace_console_summary",
            default_value="false",
            description="Print one-line trace summaries to the launch terminal.",
        ),
        DeclareLaunchArgument(
            "record_joint_velocity",
            default_value="false",
            description="Log /joint_states velocity values to a txt file.",
        ),
        DeclareLaunchArgument(
            "joint_velocity_log_directory",
            default_value=os.path.expanduser("~/ros2_ws/data/joint_velocity_logs"),
            description="Directory for joint velocity txt logs.",
        ),
        DeclareLaunchArgument(
            "joint_velocity_log_prefix",
            default_value="joint_velocity",
            description="Prefix for joint velocity txt logs.",
        ),
        DeclareLaunchArgument(
            "auto_start_recording",
            default_value="false",
            description="Keep recorder idle until the experiment runner starts it.",
        ),
        DeclareLaunchArgument(
            "recording_rate",
            default_value="100.0",
            description="Dataset recording rate in Hz.",
        ),
        DeclareLaunchArgument(
            "recording_output_directory",
            default_value=os.path.expanduser("~/ros2_ws/data/rmp_datasets"),
            description="Directory for saved dataset files.",
        ),
        DeclareLaunchArgument(
            "recording_output_prefix",
            default_value="target_only_tuning",
            description="Prefix for saved dataset filenames.",
        ),
        DeclareLaunchArgument(
            "servo_t1",
            default_value="0.002",
            description="ServoJ interpolation time for the internal RB10 bridge.",
        ),
        DeclareLaunchArgument(
            "servo_t2",
            default_value="0.1",
            description="ServoJ smoothing time for the internal RB10 bridge.",
        ),
        DeclareLaunchArgument(
            "servo_gain",
            default_value="0.02",
            description="ServoJ gain for the internal RB10 bridge.",
        ),
        DeclareLaunchArgument(
            "servo_alpha",
            default_value="0.2",
            description="ServoJ low-pass filter gain for the internal RB10 bridge.",
        ),
        DeclareLaunchArgument(
            "command_mode",
            default_value="velocity",
            description="RB10 command output mode: position sends move_servo_j, velocity sends move_speed_j.",
        ),
        DeclareLaunchArgument(
            "bridge_max_command_step_deg",
            default_value="1.0",
            description="Hard per-cycle joint-step limit for the Python RB10 bridge to avoid hardware safety trips.",
        ),
        DeclareLaunchArgument(
            "bridge_max_command_velocity_deg_s",
            default_value="100.0",
            description="Hard joint-velocity limit used by the Python RB10 bridge command guard.",
        ),
        DeclareLaunchArgument(
            "bridge_large_command_jump_warn_deg",
            default_value="2.0",
            description="Warn when the requested bridge command jumps more than this far from the current joint reference.",
        ),
        DeclareLaunchArgument(
            "startup_move_to_default_pose",
            default_value="false",
            description="Move to the configured default joint pose before enabling joint-state publication.",
        ),
        DeclareLaunchArgument(
            "startup_movej_speed",
            default_value="20.0",
            description="Joint speed for the startup move_j to the default pose.",
        ),
        DeclareLaunchArgument(
            "startup_movej_accel",
            default_value="20.0",
            description="Joint acceleration for the startup move_j to the default pose.",
        ),
        DeclareLaunchArgument(
            "startup_release_timeout_sec",
            default_value="12.0",
            description="How long to wait for the startup move_j to settle before releasing joint-state publication.",
        ),
        DeclareLaunchArgument(
            "stop_on_shutdown",
            default_value="true",
            description="Send a stop command to the RB10 when the launch exits.",
        ),
        DeclareLaunchArgument(
            "shutdown_action",
            default_value="halt",
            description="Stop action to send on shutdown: halt or pause.",
        ),
        DeclareLaunchArgument(
            "enable_realtime",
            default_value="false",
            description="Run the controller loop as SCHED_FIFO on PREEMPT_RT systems.",
        ),
        DeclareLaunchArgument(
            "realtime_priority",
            default_value="80",
            description="SCHED_FIFO priority used when enable_realtime is true.",
        ),
        DeclareLaunchArgument(
            "lock_memory",
            default_value="false",
            description="Call mlockall for the controller process when realtime is enabled.",
        ),
        DeclareLaunchArgument(
            "enable_socket_realtime",
            default_value="false",
            description="Run the direct RB10 socket send/receive threads as SCHED_FIFO as well.",
        ),
        DeclareLaunchArgument(
            "socket_realtime_priority",
            default_value="60",
            description="SCHED_FIFO priority for the direct RB10 socket threads when enabled.",
        ),
        OpaqueFunction(function=_validate_launch),
        api_bridge,
        robot_state_publisher,
        rmpflow_controller,
        data_recorder,
        joint_velocity_logger,
        rmpflow_trace_logger,
        interactive_goal,
        obstacle_manager,
        proximity_obstacle_bridge,
        rviz,
    ])

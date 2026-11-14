#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _validate_params_file(context):
    params_path = LaunchConfiguration("params_file").perform(context)
    if not os.path.isfile(params_path):
        raise RuntimeError(
            f'Launch argument "params_file" must point to an existing YAML file: {params_path}'
        )
    tuning_params_path = LaunchConfiguration("tuning_params_file").perform(context).strip()
    if tuning_params_path and not os.path.isfile(tuning_params_path):
        raise RuntimeError(
            'Launch argument "tuning_params_file" must be empty or point to an '
            f'existing YAML file: {tuning_params_path}'
        )
    return []


# TG-RMP flags whose value lives in params.yaml. The launch argument stays available
# for one-off overrides, but an empty value (the default) leaves params.yaml untouched.
TG_RMP_PARAMS_YAML_ARGS = (
    "enable_tangent_escape_rmp",
    "publish_tangent_escape_rmp_data",
    "publish_tangent_escape_geometry_debug",
)

# Controller values whose persistent defaults live in params.yaml. Keeping the
# corresponding launch arguments empty means the YAML value is used; a non-empty
# launch argument remains available for an explicit one-off override.
CONTROLLER_BOOL_PARAMS_YAML_ARGS = (
    "enable_realtime",
    "lock_memory",
    "enable_socket_realtime",
    "use_synced_input_velocity_filter",
    "estimate_velocity_in_controller",
    "use_velocity_feedback_in_solver",
)

CONTROLLER_INT_PARAMS_YAML_ARGS = (
    "realtime_priority",
    "socket_realtime_priority",
)

CONTROLLER_FLOAT_PARAMS_YAML_ARGS = (
    "measured_velocity_feedback_blend",
    "synced_input_velocity_filter_alpha",
    "synced_input_velocity_filter_beta",
    "synced_input_velocity_ratio_tolerance",
)


def _tg_rmp_overrides(context):
    overrides = {}
    for name in TG_RMP_PARAMS_YAML_ARGS:
        value = LaunchConfiguration(name).perform(context).strip().lower()
        if not value:
            continue
        if value not in ("true", "false"):
            raise RuntimeError(
                f'Launch argument "{name}" must be "true", "false", or empty '
                f'(follow params.yaml), got: {value}'
            )
        overrides[name] = value == "true"
    return overrides


def _controller_params_yaml_overrides(context):
    overrides = {}
    for name in CONTROLLER_BOOL_PARAMS_YAML_ARGS:
        value = LaunchConfiguration(name).perform(context).strip().lower()
        if not value:
            continue
        if value not in ("true", "false"):
            raise RuntimeError(
                f'Launch argument "{name}" must be "true", "false", or empty '
                f'(follow params.yaml), got: {value}'
            )
        overrides[name] = value == "true"

    for name in CONTROLLER_INT_PARAMS_YAML_ARGS:
        value = LaunchConfiguration(name).perform(context).strip()
        if not value:
            continue
        try:
            overrides[name] = int(value)
        except ValueError as error:
            raise RuntimeError(
                f'Launch argument "{name}" must be an integer or empty '
                f'(follow params.yaml), got: {value}'
            ) from error

    for name in CONTROLLER_FLOAT_PARAMS_YAML_ARGS:
        value = LaunchConfiguration(name).perform(context).strip()
        if not value:
            continue
        try:
            overrides[name] = float(value)
        except ValueError as error:
            raise RuntimeError(
                f'Launch argument "{name}" must be a number or empty '
                f'(follow params.yaml), got: {value}'
            ) from error
    return overrides


def _rmpflow_controller_actions(
    context, params_file, controller_parameters, controller_executable="rmpflow_controller"
):
    parameters = [params_file, controller_parameters]
    tuning_params_path = LaunchConfiguration("tuning_params_file").perform(context).strip()
    if tuning_params_path:
        # Keep experiment-only values out of params.yaml.  This file is applied
        # after the launch defaults, so the effective tuning is exactly what is
        # captured in the profile.
        parameters.append(tuning_params_path)
    overrides = _controller_params_yaml_overrides(context)
    overrides.update(_tg_rmp_overrides(context))
    if overrides:
        parameters.append(overrides)
    return [
        Node(
            package="rb10_rmpflow_rviz",
            executable=controller_executable,
            name="rmpflow_controller",
            output="screen",
            parameters=parameters,
        )
    ]


def _proximity_obstacle_bridge_actions(context, params_file, use_proximity_bridge):
    parameters = [params_file]
    tuning_params_path = LaunchConfiguration("tuning_params_file").perform(context).strip()
    if tuning_params_path:
        parameters.append(tuning_params_path)
    parameters.append({
        "rmp_flag_gate_enabled": ParameterValue(
            PythonExpression([
                'True if "', use_proximity_bridge, '" == "true" else False'
            ]),
            value_type=bool,
        ),
        "rmp_flag_topic": "/RMP_flag",
        "rmp_active_flag_value": 1,
        "publish_collision_obstacles": ParameterValue(
            PythonExpression([
                'True if "', use_proximity_bridge, '" == "true" else False'
            ]),
            value_type=bool,
        ),
    })
    return [
        Node(
            package="rb10_rmpflow_rviz",
            executable="proximity_obstacle_bridge",
            name="proximity_obstacle_bridge",
            output="screen",
            condition=IfCondition(use_proximity_bridge),
            parameters=parameters,
        )
    ]


def _rmpflow_bridge_actions(context, start_rmpflow_bridge):
    parameters = [{
        "flag_topic": "/RMP_flag",
        "goal_topic": "/RMP_goal",
        "controller_goal_topic": "/goal_pose",
        "controller_command_topic": "/position_controllers/commands",
        "target_q_topic": "/target_q",
        "forward_target_q": False,
        "goal_frame_id": "base_link",
        "active_flag_value": 1,
        "command_forward_delay_ms": 100,
    }]
    tuning_params_path = LaunchConfiguration("tuning_params_file").perform(context).strip()
    if tuning_params_path:
        parameters.append(tuning_params_path)
    return [
        Node(
            package="rb10_rmpflow_rviz",
            executable="rmpflow_bridge",
            name="rmpflow_bridge",
            output="screen",
            condition=IfCondition(start_rmpflow_bridge),
            parameters=parameters,
        )
    ]


def generate_launch_description(
    *,
    params_filename="params.yaml",
    urdf_filename="rb10_1300e.urdf",
    controller_executable="rmpflow_controller",
    sensor_topic_ids=None,
    range_topics_from_params=False,
):
    rmp_pkg = get_package_share_directory("rb10_rmpflow_rviz")
    config_dir = os.path.join(rmp_pkg, "config")
    default_params_path = os.path.join(config_dir, params_filename)

    urdf_path = os.path.join(rmp_pkg, "urdf", urdf_filename)
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    rviz_config_path = os.path.join(config_dir, "rb10_rmpflow.rviz")

    robot_ip = LaunchConfiguration("robot_ip")
    params_file = LaunchConfiguration("params_file")
    use_rviz = LaunchConfiguration("use_rviz")
    solver_backend = LaunchConfiguration("solver_backend")
    cb_simulation = LaunchConfiguration("cb_simulation")
    use_direct_hardware_backend = LaunchConfiguration("use_direct_hardware_backend")
    real_joint_state_source = LaunchConfiguration("real_joint_state_source")
    raw_joint_state_topic = LaunchConfiguration("raw_joint_state_topic")
    normalized_joint_state_topic = LaunchConfiguration("normalized_joint_state_topic")
    robot_description_topic = LaunchConfiguration("robot_description_topic")
    start_local_robot_state_publisher = LaunchConfiguration("start_local_robot_state_publisher")
    publish_debug_joint_state_sources = LaunchConfiguration("publish_debug_joint_state_sources")
    start_api_bridge = LaunchConfiguration("start_api_bridge")
    start_rmpflow_bridge = LaunchConfiguration("start_rmpflow_bridge")
    use_interactive_goal = LaunchConfiguration("use_interactive_goal")
    use_obstacles = LaunchConfiguration("use_obstacles")
    use_proximity_bridge = LaunchConfiguration("use_proximity_bridge")
    record_data = LaunchConfiguration("record_data")
    auto_start_recording = LaunchConfiguration("auto_start_recording")
    recording_rate = LaunchConfiguration("recording_rate")
    recording_output_directory = LaunchConfiguration("recording_output_directory")
    recording_output_prefix = LaunchConfiguration("recording_output_prefix")
    recorder_range_topics = [
        f"/raw_distance{index}"
        for index in (range(1, 21) if sensor_topic_ids is None else sensor_topic_ids)
    ]
    trace_range_topics = [
        f"/proximity_distance{index}"
        for index in (range(1, 13) if sensor_topic_ids is None else sensor_topic_ids)
    ]
    recorder_topic_parameters = (
        {} if range_topics_from_params else {"range_topics": recorder_range_topics}
    )
    trace_topic_parameters = (
        {} if range_topics_from_params else {"range_topics": trace_range_topics}
    )
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
    use_velocity_filter = LaunchConfiguration("use_velocity_filter")
    velocity_filter_alpha = LaunchConfiguration("velocity_filter_alpha")
    velocity_filter_beta = LaunchConfiguration("velocity_filter_beta")
    bridge_max_command_step_deg = LaunchConfiguration("bridge_max_command_step_deg")
    bridge_max_command_velocity_deg_s = LaunchConfiguration("bridge_max_command_velocity_deg_s")
    bridge_large_command_jump_warn_deg = LaunchConfiguration("bridge_large_command_jump_warn_deg")
    bridge_publish_rate = LaunchConfiguration("bridge_publish_rate")
    publish_visualization = LaunchConfiguration("publish_visualization")
    publish_rmp_ee_pose = LaunchConfiguration("publish_rmp_ee_pose")
    publish_target_metric = LaunchConfiguration("publish_target_metric")
    target_metric_topic = LaunchConfiguration("target_metric_topic")
    visualize_rmp_goal_sequence = LaunchConfiguration("visualize_rmp_goal_sequence")
    rmp_goal_sequence_topic = LaunchConfiguration("rmp_goal_sequence_topic")
    rmp_goal_sequence_input_type = LaunchConfiguration("rmp_goal_sequence_input_type")
    rmp_goal_sequence_marker_topic = LaunchConfiguration("rmp_goal_sequence_marker_topic")
    rmp_goal_sequence_alpha = LaunchConfiguration("rmp_goal_sequence_alpha")
    rmp_goal_sequence_marker_scale = LaunchConfiguration("rmp_goal_sequence_marker_scale")
    use_rmp_goal_logger = LaunchConfiguration("use_rmp_goal_logger")
    rmp_goal_log_rate = LaunchConfiguration("rmp_goal_log_rate")
    rmp_goal_log_directory = LaunchConfiguration("rmp_goal_log_directory")
    rmp_goal_log_prefix = LaunchConfiguration("rmp_goal_log_prefix")
    use_rmpflow_trace_logger = LaunchConfiguration("use_rmpflow_trace_logger")
    rmpflow_trace_log_rate = LaunchConfiguration("rmpflow_trace_log_rate")
    rmpflow_trace_log_directory = LaunchConfiguration("rmpflow_trace_log_directory")
    rmpflow_trace_console_summary = LaunchConfiguration("rmpflow_trace_console_summary")
    record_joint_velocity = LaunchConfiguration("record_joint_velocity")
    joint_velocity_log_directory = LaunchConfiguration("joint_velocity_log_directory")
    joint_velocity_log_prefix = LaunchConfiguration("joint_velocity_log_prefix")
    controller_parameters = {
        "record_rmp_details": ParameterValue(LaunchConfiguration("record_rmp_details"), value_type=bool),
        "rmp_details_rate": ParameterValue(LaunchConfiguration("rmp_details_rate"), value_type=float),
        "tg_rmp_direction_mode": LaunchConfiguration("tg_direction_mode"),
        "tg_rmp_initial_angle": ParameterValue(LaunchConfiguration("tg_initial_angle"), value_type=float),
        "robot_ip": robot_ip,
        "solver_backend": solver_backend,
        # Local simulation must never be forwarded as RB controller-box mode.
        "simulation_mode": False,
        "hardware_command_enabled": ParameterValue(
            PythonExpression([
                'False if "', cb_simulation, '" == "true" else True'
            ]),
            value_type=bool,
        ),
        "real_joint_state_source": real_joint_state_source,
        "hardware_data_request_rate": bridge_publish_rate,
        "joint_state_topic": normalized_joint_state_topic,
        "rmp_flag_gate_enabled": True,
        "rmp_flag_topic": "/RMP_flag",
        "rmp_active_flag_value": 1,
        "position_command_topic": "/position_controllers/commands",
        "publish_position_command": True,
        "position_command_state_topic": "/rmp_position_command",
        "command_mode": command_mode,
        "publish_target_q": True,
        "target_q_topic": "/target_q",
        "publish_target_metric": publish_target_metric,
        "target_metric_topic": target_metric_topic,
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
        "publish_debug_joint_state_sources": ParameterValue(PythonExpression([
            "'", LaunchConfiguration("record_rmp_details"), "'.lower() == 'true' or '",
            publish_debug_joint_state_sources, "'.lower() == 'true'"
        ]), value_type=bool),
        "publish_visualization": publish_visualization,
        "publish_rmp_ee_pose": publish_rmp_ee_pose,
        "tangent_escape_geometry_data_topic": "/tangent_escape_geometry_data",
        "tangent_escape_geometry_marker_topic": "tangent_escape_geometry_markers",
    }
    if urdf_filename != "rb10_1300e.urdf":
        controller_parameters["robot_urdf_path"] = urdf_path

    api_bridge = Node(
        package="rb10_rmpflow_rviz",
        executable="rb10_api_bridge.py",
        name="rb10_api_bridge",
        output="screen",
        condition=IfCondition(PythonExpression([
            '"', cb_simulation, '" != "true" and "',
            use_direct_hardware_backend, '" != "true" and "',
            start_api_bridge, '" == "true"',
        ])),
        parameters=[
            params_file,
            {
                "robot_ip": robot_ip,
                "simulation_mode": False,
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
                "stop_on_shutdown": stop_on_shutdown,
                "shutdown_action": shutdown_action,
            },
        ],
    )

    joint_state_adapter = Node(
        package="rb10_rmpflow_rviz",
        executable="joint_state_adapter",
        name="joint_state_adapter",
        output="screen",
        condition=IfCondition(PythonExpression([
            '"', cb_simulation, '" != "true" and "',
            use_direct_hardware_backend, '" != "true" and "',
            raw_joint_state_topic, '" != "', normalized_joint_state_topic, '"'
        ])),
        parameters=[{
            "input_topic": raw_joint_state_topic,
            "output_topic": normalized_joint_state_topic,
            "source_joint_names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            "target_joint_names": ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"],
        }],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        condition=IfCondition(start_local_robot_state_publisher),
        parameters=[{
            "robot_description": robot_description,
            "publish_frequency": 100.0,
        }],
        remappings=[
            ("robot_description", robot_description_topic),
            ("/joint_states", normalized_joint_state_topic),
        ],
    )

    rmpflow_controller = OpaqueFunction(
        function=_rmpflow_controller_actions,
        kwargs={
            "params_file": params_file,
            "controller_parameters": controller_parameters,
            "controller_executable": controller_executable,
        },
    )

    rmpflow_bridge = OpaqueFunction(
        function=_rmpflow_bridge_actions,
        kwargs={
            "start_rmpflow_bridge": start_rmpflow_bridge,
        },
    )

    interactive_goal = Node(
        package="rb10_rmpflow_rviz",
        executable="interactive_goal",
        name="interactive_goal",
        output="screen",
        parameters=[
            params_file,
            {
                "joint_state_topic": normalized_joint_state_topic,
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

    proximity_obstacle_bridge = OpaqueFunction(
        function=_proximity_obstacle_bridge_actions,
        kwargs={
            "params_file": params_file,
            "use_proximity_bridge": use_proximity_bridge,
        },
    )

    data_recorder = Node(
        package="rb10_rmpflow_rviz",
        executable="rmp_data_recorder.py",
        name="rmp_data_recorder",
        output="screen",
        condition=IfCondition(record_data),
        parameters=[params_file, {
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
            "record_rmp_details": ParameterValue(LaunchConfiguration("record_rmp_details"), value_type=bool),
            "record_detail_raw_topics": ParameterValue(LaunchConfiguration("record_detail_raw_topics"), value_type=bool),
            "snapshot_urdf_file": ParameterValue(urdf_path, value_type=str),
            "snapshot_params_file": ParameterValue(params_file, value_type=str),
            "joint_state_topic": normalized_joint_state_topic,
            **recorder_topic_parameters,
        }],
    )


    joint_velocity_logger = Node(
        package="rb10_rmpflow_rviz",
        executable="joint_velocity_logger.py",
        name="joint_velocity_logger",
        output="screen",
        condition=IfCondition(record_joint_velocity),
        parameters=[{
            "joint_state_topic": normalized_joint_state_topic,
            "output_directory": joint_velocity_log_directory,
            "output_prefix": joint_velocity_log_prefix,
            "flush_every": 1,
        }],
    )

    rmp_goal_logger = Node(
        package="rb10_rmpflow_rviz",
        executable="rmp_goal_logger.py",
        name="rmp_goal_logger",
        output="screen",
        condition=IfCondition(use_rmp_goal_logger),
        parameters=[{
            "sample_rate_hz": rmp_goal_log_rate,
            "output_directory": rmp_goal_log_directory,
            "output_prefix": rmp_goal_log_prefix,
            "external_goal_topic": "/RMP_goal",
            "controller_goal_topic": "/goal_pose",
            "rmp_flag_topic": "/RMP_flag",
            "active_flag_value": 1,
            "skip_until_first_goal": True,
            "flush_every": 1,
        }],
    )

    rmpflow_trace_logger = Node(
        package="rb10_rmpflow_rviz",
        executable="rmpflow_trace_logger.py",
        name="rmpflow_trace_logger",
        output="screen",
        condition=IfCondition(use_rmpflow_trace_logger),
        parameters=[params_file, {
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
            "joint_state_topic": normalized_joint_state_topic,
            "command_topic": ParameterValue(
                PythonExpression([
                    '"/rmp_dry_run_command" if "', cb_simulation,
                    '" == "true" else "/position_controllers/commands"'
                ]),
                value_type=str,
            ),
            "target_q_topic": "/target_q",
            "target_metric_topic": target_metric_topic,
            "debug_state_topic": "/rmp_debug_state",
            "rmp_ee_pose_topic": "/rmp_ee_pose",
            "rmp_joint_accel_topic": "/rmp_joint_accel",
            "rmp_tcp_accel_topic": "/rmp_tcp_accel",
            "tangent_escape_geometry_data_topic": "/tangent_escape_geometry_data",
            "obstacle_marker_topic": "/obstacles",
            "repulsion_metric_marker_topic": "/repulsion_metric_markers",
            "tcp_accel_marker_topic": "/tcp_accel_marker",
            "range_scale": 0.001,
            "minimum_hold_distance": 0.05,
            "trigger_distance": 0.29,
            **trace_topic_parameters,
        }],
    )

    rmp_goal_sequence_visualizer = Node(
        package="rb10_rmpflow_rviz",
        executable="rmp_goal_sequence_visualizer.py",
        name="rmp_goal_sequence_visualizer",
        output="screen",
        condition=IfCondition(visualize_rmp_goal_sequence),
        parameters=[{
            "input_topic": rmp_goal_sequence_topic,
            "input_type": rmp_goal_sequence_input_type,
            "marker_topic": rmp_goal_sequence_marker_topic,
            "frame_id": "base_link",
            "alpha": rmp_goal_sequence_alpha,
            "marker_scale": rmp_goal_sequence_marker_scale,
            "line_width": 0.008,
            "max_points": 100,
            "float_stride": 0,
        }],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(use_rviz),
        arguments=["-d", rviz_config_path] if os.path.exists(rviz_config_path) else [],
    )

    return LaunchDescription([
        DeclareLaunchArgument("record_rmp_details", default_value=LaunchConfiguration("record_data", default="false")),
        DeclareLaunchArgument("record_detail_raw_topics", default_value="false", description="Save duplicate raw streams in addition to coherent snapshots"),
        DeclareLaunchArgument("rmp_details_rate", default_value="30.0"),
        DeclareLaunchArgument(
            "tg_direction_mode", default_value="simple",
            choices=["sensor_history", "simple", "fixed", "legacy"],
            description="Velocity-normal steering (default), sensor-history optimization, fixed tangent, or legacy score.",
        ),
        DeclareLaunchArgument(
            "tg_initial_angle", default_value="3.141592653589793",
            description="Initial angle for fixed/history; tie-breaking angle for head-on or stationary simple steering.",
        ),
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.111.50",
            description="Hostname or IP address of the RB10 controller.",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params_path,
            description="Absolute path to the controller params YAML.",
        ),
        DeclareLaunchArgument(
            "tuning_params_file",
            default_value="",
            description=(
                "Optional YAML applied after params_file to rmpflow_controller, "
                "rmpflow_bridge, and proximity_obstacle_bridge. Use it to save and "
                "replay tuning runs."
            ),
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Start RViz.",
        ),
        DeclareLaunchArgument(
            "solver_backend",
            default_value="pinocchio_direct",
            description=(
                "Runtime RMP solver: pinocchio_direct (real-time default) or "
                "casadi_rmp2 (reference/equivalence backend)."
            ),
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
            "use_direct_hardware_backend",
            default_value="true",
            description=(
                "Bypass the ROS command bridge and let rmpflow_controller send RB10 API servo "
                "commands directly while keeping RViz/debug publication alive."
            ),
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
            "raw_joint_state_topic",
            default_value="/joint_states",
            description="Incoming raw joint_states topic to normalize for RViz and the controller.",
        ),
        DeclareLaunchArgument(
            "normalized_joint_state_topic",
            default_value="/joint_states",
            description="Joint states topic used by RViz and the controller.",
        ),
        DeclareLaunchArgument(
            "robot_description_topic",
            default_value="/rb10/robot_description",
            description=(
                "Robot description topic used by this launch. Kept separate from /robot_description "
                "to avoid conflicts with hand or external robot_state_publisher nodes."
            ),
        ),
        DeclareLaunchArgument(
            "start_local_robot_state_publisher",
            default_value="true",
            description=(
                "Start the local robot_state_publisher. Keep false on the compute PC when another "
                "machine already publishes /robot_description, /tf, and /tf_static."
            ),
        ),
        DeclareLaunchArgument(
            "publish_debug_joint_state_sources",
            default_value="true",
            description=(
                "Publish reference/measured joint-state debug topics and their tracking error "
                "for vibration-source diagnosis."
            ),
        ),
        DeclareLaunchArgument(
            "start_api_bridge",
            default_value="false",
            description=(
                "Start the local RB10 API bridge when using the joint-command-topics backend. "
                "Set false on the compute PC when another machine will forward q commands to the robot."
            ),
        ),
        DeclareLaunchArgument(
            "start_rmpflow_bridge",
            default_value="true",
            description=(
                "Start the topic bridge that gates RMP execution with /RMP_flag, forwards "
                "/RMP_goal to the controller, and republishes /target_q."
            ),
        ),
        DeclareLaunchArgument(
            "use_interactive_goal",
            default_value="false",
            description="Start the interactive goal publisher.",
        ),
        DeclareLaunchArgument(
            "use_obstacles",
            default_value="true",
            description="Enable the interactive obstacle manager.",
        ),
        DeclareLaunchArgument(
            "use_proximity_bridge",
            default_value="true",
            description="Use external proximity topics to build obstacle markers.",
        ),
        DeclareLaunchArgument(
            "bridge_publish_rate",
            default_value="500.0",
            description="RB10 state receive/publish rate in Hz before filtering.",
        ),
        DeclareLaunchArgument(
            "publish_visualization",
            default_value="true",
            description=(
                "Publish controller visualization topics such as goal markers, control points, "
                "body obstacle markers, end-effector pose, and debug state."
            ),
        ),
        DeclareLaunchArgument(
            "publish_rmp_ee_pose",
            default_value="false",
            description="Publish the controller's internal RMP end-effector pose topic.",
        ),
        DeclareLaunchArgument(
            "publish_target_metric",
            default_value="false",
            description="Publish the controller target RMP 3x3 leaf metric as Float64MultiArray.",
        ),
        DeclareLaunchArgument(
            "target_metric_topic",
            default_value="/target_metric",
            description="Topic name for the target RMP metric.",
        ),
        DeclareLaunchArgument(
            "visualize_rmp_goal_sequence",
            default_value="true",
            description="Visualize /RMP_goal_sequence as transparent green RViz markers.",
        ),
        DeclareLaunchArgument(
            "rmp_goal_sequence_topic",
            default_value="/RMP_goal_sequence",
            description="Incoming goal sequence topic to visualize in RViz.",
        ),
        DeclareLaunchArgument(
            "rmp_goal_sequence_input_type",
            default_value="auto",
            description=(
                "Goal sequence topic type: auto, pose_array, path, pose_stamped, pose, "
                "or float64_multi_array."
            ),
        ),
        DeclareLaunchArgument(
            "rmp_goal_sequence_marker_topic",
            default_value="/rmp_goal_sequence_marker",
            description="RViz MarkerArray topic for the transparent goal sequence.",
        ),
        DeclareLaunchArgument(
            "rmp_goal_sequence_alpha",
            default_value="0.25",
            description="Transparency alpha for the goal sequence markers.",
        ),
        DeclareLaunchArgument(
            "rmp_goal_sequence_marker_scale",
            default_value="0.03",
            description="Sphere diameter for each transparent goal sequence point.",
        ),
        DeclareLaunchArgument(
            "use_rmp_goal_logger",
            default_value="true",
            description="Record the bridged /goal_pose target at a fixed rate.",
        ),
        DeclareLaunchArgument(
            "rmp_goal_log_rate",
            default_value="200.0",
            description="Goal logger sampling rate in Hz.",
        ),
        DeclareLaunchArgument(
            "rmp_goal_log_directory",
            default_value=os.path.expanduser("~/ros2_ws/log/rmp_goal"),
            description="Directory for 200 Hz RMP goal CSV logs.",
        ),
        DeclareLaunchArgument(
            "rmp_goal_log_prefix",
            default_value="rmp_goal",
            description="Prefix for RMP goal CSV logs.",
        ),
        DeclareLaunchArgument(
            "use_rmpflow_trace_logger",
            default_value="false",
            description="Start a trace logger that records RMP inputs, sensor ranges, metrics, and commands.",
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
            "enable_tangent_escape_rmp",
            default_value="",
            description=(
                "Enable the canonical tangent Escape leaf. Empty (default) follows "
                "params.yaml; set true/false to override it."
            ),
        ),
        DeclareLaunchArgument(
            "publish_tangent_escape_rmp_data",
            default_value="",
            description=(
                "Publish tangent escape RMP diagnostics for trace logging. Empty "
                "(default) follows params.yaml; set true/false to override it."
            ),
        ),
        DeclareLaunchArgument(
            "publish_tangent_escape_geometry_debug",
            default_value="",
            description=(
                "Publish Stage-0 tangent escape geometry data and RViz markers. Empty "
                "(default) follows params.yaml; set true/false to override it."
            ),
        ),
        DeclareLaunchArgument(
            "record_data",
            default_value="false",
            description="Record RMP topics to a dataset file.",
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
            default_value="",
            description=(
                "Run the controller loop as SCHED_FIFO on PREEMPT_RT systems. "
                "Empty (default) follows params.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "realtime_priority",
            default_value="",
            description=(
                "SCHED_FIFO priority used when enable_realtime is true. "
                "Empty (default) follows params.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "lock_memory",
            default_value="",
            description=(
                "Call mlockall for the controller process when realtime is enabled. "
                "Empty (default) follows params.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "enable_socket_realtime",
            default_value="",
            description=(
                "Run the direct RB10 socket send/receive threads as SCHED_FIFO as well. "
                "Empty (default) follows params.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "socket_realtime_priority",
            default_value="",
            description=(
                "SCHED_FIFO priority for the direct RB10 socket threads when enabled. "
                "Empty (default) follows params.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "use_velocity_filter",
            default_value="false",
            description="Use the alpha-beta filter when estimating /joint_states velocity from position.",
        ),
        DeclareLaunchArgument(
            "velocity_filter_alpha",
            default_value="0.5",
            description="Alpha parameter for the RB10 joint velocity alpha-beta filter.",
        ),
        DeclareLaunchArgument(
            "velocity_filter_beta",
            default_value="0.015",
            description="Beta parameter for the RB10 joint velocity alpha-beta filter.",
        ),
        DeclareLaunchArgument(
            "measured_velocity_feedback_blend",
            default_value="",
            description=(
                "Blend factor for solver joint velocity feedback. "
                "1.0 uses only measured velocity, 0.0 uses only previous command velocity. "
                "Empty (default) follows params.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "use_synced_input_velocity_filter",
            default_value="",
            description=(
                "Use the high-rate direct-backend state input to build a control-rate-synchronized "
                "joint velocity estimate before feeding qd to the solver. "
                "For direct backend use, pair this with estimate_velocity_in_controller:=false. "
                "Empty (default) follows params.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "synced_input_velocity_filter_alpha",
            default_value="",
            description=(
                "Low-pass blend for the synchronized high-rate input velocity filter. "
                "1.0 keeps the windowed raw velocity, smaller values smooth it. "
                "Empty (default) follows params.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "synced_input_velocity_filter_beta",
            default_value="",
            description=(
                "Beta parameter for the fixed alpha-beta synchronized high-rate input velocity "
                "filter. Empty (default) follows params.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "synced_input_velocity_ratio_tolerance",
            default_value="",
            description=(
                "Tolerance used when matching hardware_data_request_rate/control_rate to an integer "
                "sample multiple for the synchronized input velocity filter. "
                "Empty (default) follows params.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "estimate_velocity_in_controller",
            default_value="",
            description=(
                "Estimate joint velocity inside the controller loop from joint position "
                "instead of consuming the bridge-published /joint_states velocity directly. "
                "Empty (default) follows params.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "use_velocity_feedback_in_solver",
            default_value="",
            description=(
                "Pass joint velocity to the RMP solver. Set false to test position-only feedback "
                "by zeroing qd before the solve step. Empty (default) follows params.yaml."
            ),
        ),
        OpaqueFunction(function=_validate_params_file),
        api_bridge,
        joint_state_adapter,
        robot_state_publisher,
        rmpflow_controller,
        rmpflow_bridge,
        data_recorder,
        joint_velocity_logger,
        rmp_goal_logger,
        rmpflow_trace_logger,
        interactive_goal,
        obstacle_manager,
        proximity_obstacle_bridge,
        rmp_goal_sequence_visualizer,
        rviz,
    ])

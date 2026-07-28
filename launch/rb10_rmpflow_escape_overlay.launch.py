#!/usr/bin/env python3

import atexit
import copy
import os
import shutil
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


SENSOR_FRAMES = [
    "tof6_1_L",
    "tof6_1_F",
    "tof6_1_R",
    "tof6_1_U",
    "tof_S",
    "tof_E",
    "tof_N",
    "tof_W",
    "tof3_1_S",
    "tof3_1_W",
    "tof3_1_N",
    "tof3_1_E",
    "tof2_1_E",
    "tof2_1_S",
    "tof2_1_W",
    "tof2_1_N",
    "tof2_E",
    "tof2_S",
    "tof2_W",
    "tof2_N",
]

TOPIC_INDICES = [
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    5,
    12,
    9,
    10,
    11,
    17,
    18,
    19,
    20,
    16,
    15,
    14,
    13,
]

_PARAMETER_DIRECTORY = None


def _as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _flatten_parameters(values, prefix=""):
    flattened = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_parameters(value, name))
        else:
            flattened[name] = value
    return flattened


def _load_node_parameters(document, node_name):
    node_values = document.get(node_name, {})
    ros_parameters = node_values.get("ros__parameters", {})
    if not isinstance(ros_parameters, dict):
        raise RuntimeError(
            f"{node_name}.ros__parameters must be a mapping in the params file"
        )
    return _flatten_parameters(ros_parameters)


class _QuotedStringDumper(yaml.SafeDumper):
    pass


def _represent_quoted_string(dumper, value):
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str",
        value,
        style='"',
    )


_QuotedStringDumper.add_representer(str, _represent_quoted_string)


def _parameter_file(node_fqn, values):
    global _PARAMETER_DIRECTORY
    if _PARAMETER_DIRECTORY is None:
        _PARAMETER_DIRECTORY = tempfile.mkdtemp(
            prefix="rb10_escape_overlay_params_"
        )
        atexit.register(
            shutil.rmtree,
            _PARAMETER_DIRECTORY,
            ignore_errors=True,
        )

    filename = node_fqn.strip("/").replace("/", "_") + ".yaml"
    path = os.path.join(_PARAMETER_DIRECTORY, filename)
    document = {node_fqn: {"ros__parameters": values}}
    with open(path, "w", encoding="utf-8") as stream:
        yaml.dump(
            document,
            stream,
            Dumper=_QuotedStringDumper,
            default_flow_style=False,
            sort_keys=True,
        )
    return path


def _resolve_optional_override(context, name, current_value, value_type):
    value = LaunchConfiguration(name).perform(context).strip()
    if value == "params":
        return current_value
    return value_type(value)


def _launch_setup(context):
    package_share = get_package_share_directory("rb10_rmpflow_rviz")
    params_path = os.path.abspath(
        os.path.expanduser(LaunchConfiguration("params_file").perform(context))
    )
    if not os.path.isfile(params_path):
        raise RuntimeError(f"params_file does not exist: {params_path}")

    with open(params_path, "r", encoding="utf-8") as stream:
        params_document = yaml.safe_load(stream) or {}

    use_proximity_bridge = _as_bool(
        LaunchConfiguration("use_proximity_bridge").perform(context)
    )
    show_fixed_obstacle = _as_bool(
        LaunchConfiguration("show_fixed_obstacle").perform(context)
    )
    fixed_obstacle_collision_enabled = _as_bool(
        LaunchConfiguration("fixed_obstacle_collision_enabled").perform(context)
    )

    urdf_path = os.path.join(package_share, "urdf", "rb10_1300e.urdf")
    with open(urdf_path, "r", encoding="utf-8") as stream:
        robot_description = stream.read()

    controller_common = _load_node_parameters(
        params_document, "rmpflow_controller"
    )
    controller_common.update(
        {
            "backend_mode": "simulation",
            "simulation_mode": False,
            "hardware_command_enabled": False,
            "publish_joint_states": True,
            "joint_state_publish_topic": "joint_states",
            "joint_state_topic": "joint_states",
            "publish_position_command": False,
            "position_command_state_topic": "rmp_position_command",
            "position_command_topic": "position_controllers/commands",
            "dry_run_command_topic": "rmp_dry_run_command",
            "publish_target_q": False,
            "publish_target_metric": False,
            "publish_rmp_accel_debug": False,
            "publish_control_timing_debug": False,
            "publish_tangent_escape_dual_solve_data": False,
            "publish_leaf_ablation_data": False,
            "publish_stability_certificate_data": False,
            "rmp_flag_gate_enabled": False,
            "enable_realtime": False,
            "lock_memory": False,
        }
    )
    controller_common["control_rate"] = _resolve_optional_override(
        context,
        "control_rate",
        float(controller_common.get("control_rate", 100.0)),
        float,
    )
    controller_common["solver_backend"] = _resolve_optional_override(
        context,
        "solver_backend",
        str(controller_common.get("solver_backend", "pinocchio_direct")),
        str,
    )
    for launch_name, parameter_name in [
        ("escape_metric_scalar", "tangent_escape_rmp_metric_scalar"),
        ("escape_risk_distance_gain", "tangent_escape_rmp_risk_distance_gain"),
        ("escape_risk_distance_scale", "tangent_escape_rmp_risk_distance_scale"),
        ("escape_stuck_accel_floor", "tangent_escape_rmp_risk_stuck_accel_floor"),
        ("escape_stuck_dwell_time", "tangent_escape_rmp_risk_stuck_dwell_time"),
        ("escape_stuck_ramp_time", "tangent_escape_rmp_risk_stuck_ramp_time"),
    ]:
        controller_common[parameter_name] = _resolve_optional_override(
            context,
            launch_name,
            float(controller_common.get(parameter_name, 0.0)),
            float,
        )

    controller_on = copy.deepcopy(controller_common)
    controller_on.update(
        {
            "enable_tangent_escape_rmp": True,
            "publish_visualization": True,
            "publish_tangent_escape_rmp_data": True,
            "publish_tangent_escape_dual_solve_data": True,
            "publish_tangent_escape_geometry_debug": True,
            "tangent_escape_rmp_data_topic": "tangent_escape_rmp_data",
            "tangent_escape_dual_solve_topic": "tangent_escape_dual_solve",
            "tangent_escape_geometry_data_topic": "tangent_escape_geometry_data",
            "tangent_escape_geometry_marker_topic": (
                "tangent_escape_geometry_markers"
            ),
            "publish_rmp_ee_pose": True,
        }
    )

    controller_off = copy.deepcopy(controller_common)
    controller_off.update(
        {
            "enable_tangent_escape_rmp": False,
            "publish_visualization": False,
            "publish_tangent_escape_rmp_data": False,
            "publish_tangent_escape_geometry_debug": False,
            "publish_rmp_ee_pose": True,
        }
    )

    shared_input_remappings = [
        ("goal_pose", "/goal_pose"),
        ("goal_position", "/goal_position"),
        ("obstacles", "/obstacles"),
    ]
    controller_on_remappings = [
        *shared_input_remappings,
        ("rmp_ee_pose", "/rmp_ee_pose"),
    ]

    controller_on_node = Node(
        package="rb10_rmpflow_rviz",
        executable="rmpflow_controller",
        namespace="escape_on",
        name="rmpflow_controller",
        output="screen",
        parameters=[
            _parameter_file(
                "/escape_on/rmpflow_controller",
                controller_on,
            )
        ],
        remappings=controller_on_remappings,
    )
    controller_off_node = Node(
        package="rb10_rmpflow_rviz",
        executable="rmpflow_controller",
        namespace="escape_off",
        name="rmpflow_controller",
        output="screen",
        parameters=[
            _parameter_file(
                "/escape_off/rmpflow_controller",
                controller_off,
            )
        ],
        remappings=shared_input_remappings,
    )

    robot_state_on = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="escape_on",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "publish_frequency": 50.0,
                "frame_prefix": "",
            }
        ],
        remappings=[
            ("joint_states", "/escape_on/joint_states"),
            ("robot_description", "/escape_on/robot_description"),
        ],
    )
    robot_state_off = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="escape_off",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "publish_frequency": 50.0,
                "frame_prefix": "escape_off/",
            }
        ],
        remappings=[
            ("joint_states", "/escape_off/joint_states"),
            ("robot_description", "/escape_off/robot_description"),
        ],
    )

    off_base_anchor = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="escape_off_base_anchor",
        output="screen",
        arguments=[
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            "base_link",
            "--child-frame-id",
            "escape_off/base_link",
        ],
    )

    actions = [
        robot_state_on,
        robot_state_off,
        off_base_anchor,
        controller_on_node,
        controller_off_node,
    ]

    merged_obstacle_topics = []
    if show_fixed_obstacle:
        fixed_obstacle_parameters = _load_node_parameters(
            params_document, "obstacle_manager"
        )
        fixed_obstacle_topic = "/fixed_obstacle_markers"
        if fixed_obstacle_collision_enabled:
            merged_obstacle_topics.append(fixed_obstacle_topic)
        actions.append(
            Node(
                package="rb10_rmpflow_rviz",
                executable="obstacle_manager",
                name="escape_overlay_fixed_obstacle",
                output="screen",
                parameters=[
                    _parameter_file(
                        "/escape_overlay_fixed_obstacle",
                        fixed_obstacle_parameters,
                    )
                ],
                remappings=[
                    ("obstacles", fixed_obstacle_topic),
                    ("add_obstacle", "/escape_overlay_fixed_obstacle/add_obstacle"),
                ],
            )
        )

    if use_proximity_bridge:
        bridge_parameters = _load_node_parameters(
            params_document, "proximity_obstacle_bridge"
        )
        proximity_collision_topic = "/proximity_obstacles_raw"
        merged_obstacle_topics.append(proximity_collision_topic)
        bridge_parameters.update(
            {
                "fixed_frame": "base_link",
                "obstacle_topic": proximity_collision_topic,
                "visualization_obstacle_topic": "/obstacle_markers",
                "publish_collision_obstacles": True,
                "surface_patch_fixed_visualization": False,
                "range_topics": [
                    f"/fake_proximity_distance{index}"
                    for index in TOPIC_INDICES
                ],
                "sensor_frames": SENSOR_FRAMES,
                "sensor_enabled": [True] * len(SENSOR_FRAMES),
                "trigger_distances": [0.55] * len(SENSOR_FRAMES),
                "joint_state_topic": "/escape_on/joint_states",
                "elbow_tof3_1_w_ignore_enabled": False,
            }
        )
        actions.append(
            Node(
                package="rb10_rmpflow_rviz",
                executable="proximity_obstacle_bridge",
                name="escape_overlay_proximity_bridge",
                output="screen",
                parameters=[
                    _parameter_file(
                        "/escape_overlay_proximity_bridge",
                        bridge_parameters,
                    )
                ],
            )
        )

    # The controller replaces its complete obstacle set on every MarkerArray,
    # so multiple publishers must be merged before sharing /obstacles.
    actions.append(
        Node(
            package="rb10_rmpflow_rviz",
            executable="camera_body_sphere_cloud_bridge.py",
            name="escape_overlay_obstacle_merger",
            output="screen",
            parameters=[
                {
                    "cloud_topic": "/escape_overlay/unused_obstacle_cloud",
                    "camera_marker_topic": (
                        "/escape_overlay/unused_camera_obstacle_markers"
                    ),
                    "subscribe_camera_marker_topic": False,
                    "relay_camera_markers_to_rviz": False,
                    "additional_obstacle_topics": merged_obstacle_topics,
                    "collision_obstacle_topic": "/obstacles",
                    "rviz_marker_topic": (
                        "/escape_overlay/unused_merged_obstacle_markers"
                    ),
                    "output_frame": "base_link",
                    "stale_timeout_sec": 0.5,
                    "publish_rate": min(
                        max(float(controller_common["control_rate"]), 10.0),
                        100.0,
                    ),
                    "publish_empty_when_stale": True,
                    "log_rate_hz": 1.0,
                }
            ],
        )
    )

    if _as_bool(LaunchConfiguration("use_interactive_goal").perform(context)):
        goal_parameters = _load_node_parameters(
            params_document, "interactive_goal"
        )
        goal_parameters.update(
            {
                "joint_state_topic": "/escape_on/joint_states",
                "lock_orientation_to_tcp": False,
            }
        )
        actions.append(
            Node(
                package="rb10_rmpflow_rviz",
                executable="interactive_goal",
                name="escape_overlay_interactive_goal",
                output="screen",
                parameters=[
                    _parameter_file(
                        "/escape_overlay_interactive_goal",
                        goal_parameters,
                    )
                ],
            )
        )

    actions.append(
        Node(
            package="rviz2",
            executable="rviz2",
            name="escape_overlay_rviz",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_rviz")),
            arguments=["-d", LaunchConfiguration("rviz_config")],
        )
    )
    return actions


def generate_launch_description():
    package_share = get_package_share_directory("rb10_rmpflow_rviz")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(package_share, "config", "params.yaml"),
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=os.path.join(
                    package_share,
                    "config",
                    "rb10_rmpflow_escape_overlay.rviz",
                ),
            ),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("use_interactive_goal", default_value="true"),
            DeclareLaunchArgument("use_proximity_bridge", default_value="true"),
            DeclareLaunchArgument("show_fixed_obstacle", default_value="true"),
            DeclareLaunchArgument(
                "fixed_obstacle_collision_enabled",
                default_value="true",
                description=(
                    "Feed the visible ground-truth fixed obstacle to the Collision "
                    "RMP. Disable only when intentionally testing sparse ToF hit "
                    "points without solid obstacle geometry."
                ),
            ),
            DeclareLaunchArgument(
                "control_rate",
                default_value="params",
                description=(
                    "Use 'params' to retain params.yaml, or provide a rate in Hz."
                ),
            ),
            DeclareLaunchArgument(
                "solver_backend",
                default_value="params",
                description=(
                    "Use 'params' to retain params.yaml, or provide a solver name."
                ),
            ),
            DeclareLaunchArgument("escape_metric_scalar", default_value="params"),
            DeclareLaunchArgument("escape_risk_distance_gain", default_value="params"),
            DeclareLaunchArgument("escape_risk_distance_scale", default_value="params"),
            DeclareLaunchArgument("escape_stuck_accel_floor", default_value="params"),
            DeclareLaunchArgument("escape_stuck_dwell_time", default_value="params"),
            DeclareLaunchArgument("escape_stuck_ramp_time", default_value="params"),
            OpaqueFunction(function=_launch_setup),
        ]
    )

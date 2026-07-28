#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _validate_params_file(context):
    params_path = LaunchConfiguration("params_file").perform(context)
    if not os.path.isfile(params_path):
        raise RuntimeError(f"params_file does not exist: {params_path}")
    return []


def generate_launch_description():
    package_share = get_package_share_directory("rb10_rmpflow_rviz")
    urdf_path = os.path.join(package_share, "urdf", "rb10_1300e.urdf")
    with open(urdf_path, "r", encoding="utf-8") as urdf_file:
        robot_description = urdf_file.read()
    rviz_config_path = os.path.join(
        package_share,
        "config",
        "rb10_rmpflow.rviz",
    )
    use_rviz = LaunchConfiguration("use_rviz")

    controller = Node(
        package="rb10_rmpflow_rviz",
        executable="rmpflow_controller",
        name="rmpflow_controller",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "robot_ip": LaunchConfiguration("robot_ip"),
                "backend_mode": "simulation",
                "simulation_mode": True,
                "hardware_command_enabled": False,
                "solver_backend": LaunchConfiguration("solver_backend"),
                "control_rate": ParameterValue(
                    LaunchConfiguration("control_rate"),
                    value_type=float,
                ),
                "casadi_rmp2_max_obstacles": ParameterValue(
                    LaunchConfiguration("casadi_rmp2_max_obstacles"),
                    value_type=int,
                ),
                "casadi_rmp2_jit": ParameterValue(
                    LaunchConfiguration("casadi_rmp2_jit"),
                    value_type=bool,
                ),
                "startup_move_to_default_pose": False,
                "publish_control_timing_debug": True,
                "control_timing_debug_topic": LaunchConfiguration("timing_topic"),
                "control_timing_publish_rate": ParameterValue(
                    LaunchConfiguration("timing_publish_rate"),
                    value_type=float,
                ),
                "publish_joint_states": ParameterValue(use_rviz, value_type=bool),
                "publish_position_command": False,
                "publish_target_q": False,
                "publish_target_metric": False,
                "publish_rmp_accel_debug": False,
                "publish_visualization": ParameterValue(use_rviz, value_type=bool),
                "publish_goal_marker": ParameterValue(use_rviz, value_type=bool),
                "publish_control_point_markers": False,
                "publish_body_obstacle_markers": False,
                "publish_repulsion_metric_markers": False,
                "publish_tcp_accel_marker": False,
                "publish_end_effector_pose": False,
                "publish_debug_state": False,
                "publish_rmp_ee_pose": False,
                "publish_goal_tf": False,
                "publish_debug_joint_state_sources": False,
                "enable_tangent_escape_rmp": ParameterValue(
                    LaunchConfiguration("enable_tangent_escape_rmp"),
                    value_type=bool,
                ),
                "publish_tangent_escape_rmp_data": False,
                "publish_tangent_escape_dual_solve_data": False,
                "publish_tangent_escape_geometry_debug": False,
                "publish_leaf_ablation_data": False,
                "publish_stability_certificate_data": False,
                "enable_realtime": False,
                "enable_socket_realtime": False,
            },
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        condition=IfCondition(use_rviz),
        parameters=[{
            "robot_description": robot_description,
            "publish_frequency": 50.0,
        }],
        remappings=[("robot_description", "/rb10/robot_description")],
    )

    interactive_goal = Node(
        package="rb10_rmpflow_rviz",
        executable="interactive_goal",
        name="interactive_goal",
        output="screen",
        condition=IfCondition(use_rviz),
        parameters=[
            LaunchConfiguration("params_file"),
            {"lock_orientation_to_tcp": False},
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(use_rviz),
        arguments=["-d", LaunchConfiguration("rviz_config")],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(package_share, "config", "params.yaml"),
        ),
        DeclareLaunchArgument("robot_ip", default_value="192.168.111.50"),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Show the robot and interactive goal for the motion benchmark.",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=rviz_config_path,
        ),
        DeclareLaunchArgument(
            "control_rate",
            default_value="50.0",
            description="Rate under test: run separately at 50, 100, and 200 Hz.",
        ),
        DeclareLaunchArgument(
            "solver_backend",
            default_value="pinocchio_direct",
            description="Runtime solver backend under test.",
        ),
        DeclareLaunchArgument(
            "enable_tangent_escape_rmp",
            default_value="false",
            description="Include the Tangent Escape supervisor and leaf in the benchmark.",
        ),
        DeclareLaunchArgument("casadi_rmp2_max_obstacles", default_value="24"),
        DeclareLaunchArgument("casadi_rmp2_jit", default_value="true"),
        DeclareLaunchArgument(
            "timing_topic",
            default_value="/rmp_control_timing",
        ),
        DeclareLaunchArgument(
            "timing_publish_rate",
            default_value="10.0",
        ),
        OpaqueFunction(function=_validate_params_file),
        robot_state_publisher,
        controller,
        interactive_goal,
        rviz,
    ])

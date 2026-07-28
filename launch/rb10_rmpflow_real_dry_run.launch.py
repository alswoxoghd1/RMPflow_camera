#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package_share = get_package_share_directory("rb10_rmpflow_rviz")
    base_launch = os.path.join(package_share, "launch", "rb10_rmpflow_test.launch.py")

    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_launch),
        launch_arguments={
            "params_file": LaunchConfiguration("params_file"),
            "robot_ip": LaunchConfiguration("robot_ip"),
            "use_rviz": LaunchConfiguration("use_rviz"),
            "control_rate": LaunchConfiguration("control_rate"),
            "cb_simulation": "false",
            "use_direct_hardware_backend": "true",
            "hardware_command_enabled": "false",
            "dry_run_command_topic": LaunchConfiguration("dry_run_command_topic"),
            "startup_move_to_default_pose": "false",
            "stop_on_shutdown": "false",
            "use_interactive_goal": LaunchConfiguration("use_interactive_goal"),
            "use_obstacles": "false",
            "use_proximity_bridge": LaunchConfiguration("use_proximity_bridge"),
            "proximity_surface_visualization": "false",
            "publish_debug_joint_state_sources": "false",
            "record_joint_velocity": "false",
            "publish_control_timing_debug": "true",
            "control_timing_publish_rate": LaunchConfiguration(
                "control_timing_publish_rate"
            ),
            "estimate_velocity_in_controller": "false",
            "use_synced_input_velocity_filter": "true",
            "synced_input_velocity_filter_alpha": LaunchConfiguration(
                "synced_input_velocity_filter_alpha"
            ),
            "synced_input_velocity_filter_beta": LaunchConfiguration(
                "synced_input_velocity_filter_beta"
            ),
            "measured_velocity_feedback_blend": "1.0",
            "enable_tangent_escape_rmp": LaunchConfiguration(
                "enable_tangent_escape_rmp"
            ),
            "publish_tangent_escape_rmp_data": LaunchConfiguration(
                "publish_tangent_escape_rmp_data"
            ),
            "publish_tangent_escape_dual_solve_data": LaunchConfiguration(
                "publish_tangent_escape_dual_solve_data"
            ),
            "publish_tangent_escape_geometry_debug": LaunchConfiguration(
                "publish_tangent_escape_geometry_debug"
            ),
            "use_rmpflow_trace_logger": LaunchConfiguration(
                "use_rmpflow_trace_logger"
            ),
            "rmpflow_trace_log_rate": LaunchConfiguration(
                "rmpflow_trace_log_rate"
            ),
            "rmpflow_trace_log_directory": LaunchConfiguration(
                "rmpflow_trace_log_directory"
            ),
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(package_share, "config", "params.yaml"),
        ),
        DeclareLaunchArgument("robot_ip", default_value="192.168.111.50"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("control_rate", default_value="200.0"),
        DeclareLaunchArgument("control_timing_publish_rate", default_value="100.0"),
        DeclareLaunchArgument("synced_input_velocity_filter_alpha", default_value="0.35"),
        DeclareLaunchArgument("synced_input_velocity_filter_beta", default_value="0.02"),
        DeclareLaunchArgument("use_interactive_goal", default_value="false"),
        DeclareLaunchArgument("use_proximity_bridge", default_value="true"),
        DeclareLaunchArgument("enable_tangent_escape_rmp", default_value="false"),
        DeclareLaunchArgument("publish_tangent_escape_rmp_data", default_value="false"),
        DeclareLaunchArgument(
            "publish_tangent_escape_dual_solve_data",
            default_value="false",
        ),
        DeclareLaunchArgument(
            "publish_tangent_escape_geometry_debug",
            default_value="false",
        ),
        DeclareLaunchArgument("use_rmpflow_trace_logger", default_value="false"),
        DeclareLaunchArgument("rmpflow_trace_log_rate", default_value="100.0"),
        DeclareLaunchArgument(
            "rmpflow_trace_log_directory",
            default_value=os.path.expanduser("~/ros2_ws/log/rmpflow_trace"),
        ),
        DeclareLaunchArgument(
            "dry_run_command_topic",
            default_value="/rmp_dry_run_command",
        ),
        controller,
    ])

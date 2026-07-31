#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _validate(context):
    params_path = LaunchConfiguration("params_file").perform(context)
    if not os.path.isfile(params_path):
        raise RuntimeError(f"params_file does not exist: {params_path}")
    return []


def generate_launch_description():
    package_share = get_package_share_directory("rb10_rmpflow_rviz")
    default_params = os.path.join(package_share, "config", "params.yaml")

    params_file = LaunchConfiguration("params_file")
    output_directory = LaunchConfiguration("output_directory")
    control_rate = LaunchConfiguration("control_rate")
    wait_for_external_goal = LaunchConfiguration("wait_for_external_goal")
    shutdown_after_analysis = LaunchConfiguration("shutdown_after_analysis")
    enable_tangent_escape_rmp = LaunchConfiguration("enable_tangent_escape_rmp")
    goal_x = LaunchConfiguration("goal_x")
    goal_y = LaunchConfiguration("goal_y")
    goal_z = LaunchConfiguration("goal_z")

    controller = Node(
        package="rb10_rmpflow_rviz",
        executable="rmpflow_controller",
        name="rmpflow_controller",
        output="screen",
        parameters=[
            params_file,
            {
                "backend_mode": "simulation",
                "hardware_command_enabled": False,
                "solver_backend": "pinocchio_direct",
                "control_rate": ParameterValue(control_rate, value_type=float),
                "initialize_goal_from_first_state": False,
                "goal_x": ParameterValue(goal_x, value_type=float),
                "goal_y": ParameterValue(goal_y, value_type=float),
                "goal_z": ParameterValue(goal_z, value_type=float),
                "publish_visualization": False,
                "publish_goal_tf": False,
                "publish_joint_states": False,
                "publish_position_command": False,
                "publish_target_q": False,
                "publish_rmp_ee_pose": False,
                "publish_control_timing_debug": False,
                "publish_tangent_escape_rmp_data": False,
                "publish_tangent_escape_dual_solve_data": False,
                "publish_tangent_escape_geometry_debug": False,
                "enable_tangent_escape_rmp": ParameterValue(
                    enable_tangent_escape_rmp,
                    value_type=bool,
                ),
                "run_production_stability_analysis": True,
                "production_stability_wait_for_external_goal": ParameterValue(
                    wait_for_external_goal,
                    value_type=bool,
                ),
                "shutdown_after_production_stability_analysis": ParameterValue(
                    shutdown_after_analysis,
                    value_type=bool,
                ),
                "production_stability_output_directory": output_directory,
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument(
                "output_directory",
                default_value="~/ros2_ws/log/rmpflow_production_stability",
            ),
            DeclareLaunchArgument("control_rate", default_value="200.0"),
            DeclareLaunchArgument("goal_x", default_value="0.6"),
            DeclareLaunchArgument("goal_y", default_value="-0.4"),
            DeclareLaunchArgument("goal_z", default_value="0.6"),
            DeclareLaunchArgument("wait_for_external_goal", default_value="false"),
            DeclareLaunchArgument("shutdown_after_analysis", default_value="true"),
            DeclareLaunchArgument("enable_tangent_escape_rmp", default_value="true"),
            OpaqueFunction(function=_validate),
            controller,
        ]
    )

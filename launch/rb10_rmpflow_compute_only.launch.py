#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package_share = get_package_share_directory("rb10_rmpflow_rviz")
    base_launch = os.path.join(
        package_share,
        "launch",
        "rb10_rmpflow.launch.py",
    )

    compute_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_launch),
        launch_arguments={
            "params_file": LaunchConfiguration("params_file"),
            "robot_ip": LaunchConfiguration("robot_ip"),
            "solver_backend": LaunchConfiguration("solver_backend"),
            "cb_simulation": LaunchConfiguration("cb_simulation"),
            "use_direct_hardware_backend": LaunchConfiguration(
                "use_direct_hardware_backend"
            ),
            "real_joint_state_source": LaunchConfiguration(
                "real_joint_state_source"
            ),
            "raw_joint_state_topic": LaunchConfiguration("raw_joint_state_topic"),
            "normalized_joint_state_topic": LaunchConfiguration(
                "normalized_joint_state_topic"
            ),
            "robot_description_topic": LaunchConfiguration(
                "robot_description_topic"
            ),
            "start_api_bridge": LaunchConfiguration("start_api_bridge"),
            "start_rmpflow_bridge": LaunchConfiguration("start_rmpflow_bridge"),
            "use_interactive_goal": LaunchConfiguration("use_interactive_goal"),
            "use_obstacles": LaunchConfiguration("use_obstacles"),
            "use_proximity_bridge": LaunchConfiguration("use_proximity_bridge"),
            "proximity_surface_visualization": LaunchConfiguration(
                "proximity_surface_visualization"
            ),
            "visualize_rmp_goal_sequence": LaunchConfiguration(
                "visualize_rmp_goal_sequence"
            ),
            "publish_debug_joint_state_sources": LaunchConfiguration(
                "publish_debug_joint_state_sources"
            ),
            # The compute PC owns the model and TF topics consumed by remote RViz.
            "start_local_robot_state_publisher": "true",
            "publish_visualization": "true",
            # Keep GUI and non-control disk/diagnostic work out of this profile.
            "use_rviz": "false",
            "publish_rmp_ee_pose": "false",
            "publish_target_metric": "false",
            "publish_tangent_escape_rmp_data": "false",
            "use_rmp_goal_logger": "false",
            "use_rmpflow_trace_logger": "false",
            "record_data": "false",
            "record_joint_velocity": "false",
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(package_share, "config", "params.yaml"),
            description="Absolute path to the RMP controller parameter file.",
        ),
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.111.50",
            description="Hostname or IP address of the RB10 controller.",
        ),
        DeclareLaunchArgument(
            "solver_backend",
            default_value="pinocchio_direct",
            description="RMP solver backend used on the compute PC.",
        ),
        DeclareLaunchArgument(
            "cb_simulation",
            default_value="false",
            description="Use the local in-process simulation backend.",
        ),
        DeclareLaunchArgument(
            "use_direct_hardware_backend",
            default_value="true",
            description="Let rmpflow_controller communicate with the RB10 directly.",
        ),
        DeclareLaunchArgument(
            "real_joint_state_source",
            default_value="measured",
            description="Real robot joint source: measured or reference.",
        ),
        DeclareLaunchArgument(
            "raw_joint_state_topic",
            default_value="/joint_states",
            description="Raw joint-state input topic when an adapter is needed.",
        ),
        DeclareLaunchArgument(
            "normalized_joint_state_topic",
            default_value="/joint_states",
            description="Joint-state topic used by RMP and robot_state_publisher.",
        ),
        DeclareLaunchArgument(
            "robot_description_topic",
            default_value="/rb10/robot_description",
            description="RobotModel description topic consumed by remote RViz.",
        ),
        DeclareLaunchArgument(
            "start_api_bridge",
            default_value="false",
            description="Start the Python API bridge for the topic-command backend.",
        ),
        DeclareLaunchArgument(
            "start_rmpflow_bridge",
            default_value="true",
            description="Bridge /RMP_flag and /RMP_goal into the RMP controller.",
        ),
        DeclareLaunchArgument(
            "use_interactive_goal",
            default_value="false",
            description="Run the interactive-marker goal server on the compute PC.",
        ),
        DeclareLaunchArgument(
            "use_obstacles",
            default_value="false",
            description="Run the interactive obstacle manager when proximity input is disabled.",
        ),
        DeclareLaunchArgument(
            "use_proximity_bridge",
            default_value="true",
            description="Convert proximity sensor topics into RMP obstacles.",
        ),
        DeclareLaunchArgument(
            "proximity_surface_visualization",
            default_value="true",
            description="Publish proximity surface markers for remote RViz.",
        ),
        DeclareLaunchArgument(
            "visualize_rmp_goal_sequence",
            default_value="true",
            description="Publish the RMP goal sequence as RViz markers.",
        ),
        DeclareLaunchArgument(
            "publish_debug_joint_state_sources",
            default_value="false",
            description="Publish reference/measured joint-state diagnostic topics.",
        ),
        compute_stack,
    ])

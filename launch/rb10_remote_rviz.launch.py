#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _validate_rviz_config(context):
    config_path = LaunchConfiguration("rviz_config").perform(context)
    if not os.path.isfile(config_path):
        raise RuntimeError(
            f'Launch argument "rviz_config" must point to an existing file: {config_path}'
        )
    return []


def generate_launch_description():
    package_share = get_package_share_directory("rb10_rmpflow_rviz")
    default_config = os.path.join(
        package_share,
        "config",
        "rb10_rmpflow.rviz",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "rviz_config",
            default_value=default_config,
            description="RViz config used on the visualization-only PC.",
        ),
        OpaqueFunction(function=_validate_rviz_config),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rb10_remote_rviz",
            output="screen",
            arguments=["-d", LaunchConfiguration("rviz_config")],
        ),
    ])

#!/usr/bin/env python3
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _launch_setup(context):
    params_path = os.path.abspath(
        os.path.expanduser(LaunchConfiguration("params_file").perform(context))
    )
    if not os.path.isfile(params_path):
        raise RuntimeError(f"params_file does not exist: {params_path}")

    with open(params_path, "r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    obstacle_parameters = (
        document.get("obstacle_manager", {})
        .get("ros__parameters", {})
    )
    positions = obstacle_parameters.get("default_positions", [0.0, 0.0, 0.0])
    fallback_position = [float(value) for value in positions[:3]]
    if len(fallback_position) != 3:
        raise RuntimeError(
            "obstacle_manager.default_positions must contain at least 3 values"
        )

    return [
        Node(
            package="rb10_rmpflow_rviz",
            executable="fixed_obstacle_tof_raycast.py",
            name="fixed_obstacle_tof_raycast",
            output="screen",
            parameters=[
                {
                    "publish_rate_hz": ParameterValue(
                        LaunchConfiguration("publish_rate"),
                        value_type=float,
                    ),
                    "fixed_frame": "base_link",
                    "obstacle_topic": "/fixed_obstacle_markers",
                    "inactive_range_m": ParameterValue(
                        LaunchConfiguration("inactive_range_m"),
                        value_type=float,
                    ),
                    "maximum_range_m": ParameterValue(
                        LaunchConfiguration("maximum_range_m"),
                        value_type=float,
                    ),
                    "range_scale": 0.001,
                    "proximity_topic_prefix": "/fake_proximity_distance",
                    "raw_topic_prefix": "/fake_raw_distance",
                    "publish_raw_topics": True,
                    "publish_rmp_flag": True,
                    "fallback_position": fallback_position,
                    "fallback_radius": float(
                        obstacle_parameters.get("default_radius", 0.06)
                    ),
                    "fallback_height": float(
                        obstacle_parameters.get("default_height", 0.8)
                    ),
                }
            ],
        )
    ]


def generate_launch_description():
    package_share = get_package_share_directory("rb10_rmpflow_rviz")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(package_share, "config", "params.yaml"),
            ),
            DeclareLaunchArgument("publish_rate", default_value="50.0"),
            DeclareLaunchArgument("inactive_range_m", default_value="0.90"),
            DeclareLaunchArgument("maximum_range_m", default_value="0.90"),
            OpaqueFunction(function=_launch_setup),
        ]
    )

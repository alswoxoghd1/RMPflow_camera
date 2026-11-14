#!/usr/bin/env python3
"""Test RMPflow workflow using the SoC model and nine enabled sensor inputs.

All launch arguments are inherited from rb10_rmpflow_test.launch.py.
The proximity bridge is enabled by default for the centered arm sensors.
"""
import importlib.util
from pathlib import Path


def generate_launch_description():
    base_path = Path(__file__).with_name("rb10_rmpflow_test.launch.py")
    spec = importlib.util.spec_from_file_location("soc_rmpflow_test_base", base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base.generate_launch_description(
        params_filename="SoC_params.yaml",
        urdf_filename="SoC_rb10_1300e.urdf",
        controller_executable="SoC_rmpflow_controller",
        range_topics_from_params=True,
        use_proximity_bridge_default="true",
    )

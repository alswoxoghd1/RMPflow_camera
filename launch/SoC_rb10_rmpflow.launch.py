#!/usr/bin/env python3
"""Original RMPflow workflow with two centered arm rings and 12 sensors.

All launch arguments are inherited from rb10_rmpflow.launch.py.
The nine enabled sensor inputs and recording topics come from SoC_params.yaml.
"""
import importlib.util
from pathlib import Path


def generate_launch_description():
    base_path = Path(__file__).with_name("rb10_rmpflow.launch.py")
    spec = importlib.util.spec_from_file_location("soc_rmpflow_base", base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base.generate_launch_description(
        params_filename="SoC_params.yaml",
        urdf_filename="SoC_rb10_1300e.urdf",
        controller_executable="SoC_rmpflow_controller",
        range_topics_from_params=True,
    )

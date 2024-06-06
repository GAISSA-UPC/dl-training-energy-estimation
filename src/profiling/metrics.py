"""
This module contains functions to collect metrics of a process, such as CPU usage, RAM usage and temperature.
It also contains a function to compute the number of multiply accumulate operations (MACCS) of a TensorFlow model.
"""

import csv
import time
from datetime import datetime
from os import PathLike
from typing import Union

import psutil
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

from src.profiling import MEASURING_INTERVAL

MILISECONDS_TO_SECONDS = 1e-3
BYTES_TO_MEGABYTES = 1e-6
SHARED_METRICS = ["timestamp", "cpu usage (%)", "memory usage (MB)"]


def get_basic_metrics(n_cpus: int, process: psutil.Process) -> list:
    """
    Returns the basic metrics of a process.

    Parameters
    ----------
    n_cpus : int
        Number of available CPUs.
    process : psutil.Process
        Process to profile.

    Returns
    -------
    list
        List containing the timestamp, CPU usage and RAM usage.
    """

    # Divide by the number of available CPUs to obtain the average CPU usage.
    return [
        datetime.now().isoformat(),
        round(process.cpu_percent() / n_cpus, 3),
        round(process.memory_info().rss * BYTES_TO_MEGABYTES, 3),
    ]


def collect_linux_metrics(pid, out_file):
    """
    Collects the CPU usage and temperature, and the RAM usage of a process in Linux.

    Parameters
    ----------
    pid : int
        Process ID.
    out_file : str
        Output file to write the profiling results to.
    """

    n_cpus = psutil.cpu_count()
    process = psutil.Process(pid)
    process.cpu_percent()  # Call to avoid a 0% in the first registered entry
    with open(out_file, "w", newline="", encoding="utf8") as file:
        print("CPU and RAM profiling started...")
        writer = csv.writer(file)
        writer.writerow(
            SHARED_METRICS + ["temperature (Celsius)"],
        )
        while process.is_running():
            with process.oneshot():
                data = get_basic_metrics(n_cpus, process)
                temps = psutil.sensors_temperatures()
                # Key for AMD CPUs in Linux
                if not temps:
                    temp = None
                elif "k10temp" in temps.keys():
                    temp = temps["k10temp"][-1].current
                elif "coretemp" in temps.keys():
                    temp = temps["coretemp"][-1].current
                data.append(temp)
            writer.writerow(data)
            file.flush()
            time.sleep(MEASURING_INTERVAL * MILISECONDS_TO_SECONDS)


def collect_windows_metrics(pid: int, out_file: str):
    """
    Collects the CPU usage and the RAM usage of a process in Windows.

    Parameters
    ----------
    pid : int
        Process ID.
    out_file : str
        Output file to write the profiling results to.
    """

    n_cpus = psutil.cpu_count()
    process = psutil.Process(pid)
    process.cpu_percent()  # Call to avoid a 0% in the first registered entry
    with open(out_file, "w", newline="", encoding="utf8") as file:
        print("CPU and RAM profiling started...")
        writer = csv.writer(file)
        writer.writerow(SHARED_METRICS)
        while True:
            with process.oneshot():
                data = get_basic_metrics(n_cpus, process)
            writer.writerow(data)
            file.flush()
            time.sleep(MEASURING_INTERVAL * MILISECONDS_TO_SECONDS)


def compute_maccs(model: Model, outfile: Union[str, PathLike, None] = None) -> int:
    """
    Computes the number of multiply accumulate operations (MACCS) of a model.

    Parameters
    ----------
    model : Model
        The model to profile.
    outfile : Union[str, PathLike, None], optional
        The file to write the profiling results to, by default None

    Returns
    -------
    int
        The number of MACCS of the model.
    """

    tf.keras.backend.clear_session()
    input_signature = [
        tf.TensorSpec(shape=(1, *params.shape[1:]), dtype=params.dtype, name=params.name) for params in model.inputs
    ]
    forward_pass = tf.function(model.call, input_signature=input_signature)
    if outfile is not None:
        options = ProfileOptionBuilder(ProfileOptionBuilder.float_operation()).with_file_output(outfile=outfile).build()
    else:
        options = ProfileOptionBuilder(ProfileOptionBuilder.float_operation()).build()

    graph_info = profile(
        forward_pass.get_concrete_function().graph,
        options=options,
    )
    # The //2 is necessary since `profile` counts multiply and accumulate
    # as two flops, here we report the total number of multiply accumulate ops
    maccs = graph_info.total_float_ops // 2
    print(f"MACCS: {maccs:,}")

    if outfile is not None:
        with open(outfile, "a", encoding="utf8") as file:
            file.write(f"\nMACCS: {maccs:,}\n")
    return maccs

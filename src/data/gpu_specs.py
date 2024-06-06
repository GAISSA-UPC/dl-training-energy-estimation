"""
This module defines data classes for different GPU models.

Classes:
    GPU: Abstract base class for GPU models.
    RTX3090: RTX 3090 GPU model.
    RTX3080: RTX 3080 GPU model.
    RTX3070: RTX 3070 GPU model.
    RTX2080: RTX 2080 GPU model.
    GTX1660: GTX 1660 GPU model.
    GTX1050: GTX 1050 GPU model.
    GTX750Ti: GTX 750 Ti GPU model.

Each class has the following attributes:
    name (str): The name of the GPU model.
    architecture (str): The architecture of the GPU.
    cuda_cores (int): The number of CUDA cores in the GPU.
    memory (int): The memory size of the GPU in GB.
    memory_type (str): The type of memory used in the GPU.
    memory_bandwidth (int): The memory bandwidth of the GPU in GB/s.
    base_frequency (float): The base frequency of the GPU in GHz.
    power (int): The power consumption of the GPU in W.
    temperature (int): The maximum temperature of the GPU in °C.
"""

from abc import ABC
from dataclasses import dataclass


@dataclass
class GPU(ABC):
    name: str
    architecture: str
    cuda_cores: int
    memory: int
    memory_type: str
    memory_bandwidth: float
    base_frequency: float
    power: int
    temperature: int


@dataclass
class RTX3090(GPU):
    name: str = "RTX 3090"
    architecture: str = "Ampere"
    cuda_cores: int = 10496
    memory: int = 24
    memory_type: str = "GDDR6X"
    memory_bandwidth: float = 936
    base_frequency: float = 1.40
    power: int = 350
    temperature: int = 93


@dataclass
class RTX3080(GPU):
    name: str = "RTX 3080"
    architecture: str = "Ampere"
    cuda_cores: int = 8704
    memory: int = 10
    memory_type: str = "GDDR6X"
    memory_bandwidth: float = 760
    base_frequency: float = 1.44
    power: int = 320
    temperature: int = 93


@dataclass
class RTX3070(GPU):
    name: str = "RTX 3070"
    architecture: str = "Ampere"
    cuda_cores: int = 5888
    memory: int = 8
    memory_type: str = "GDDR6"
    memory_bandwidth: float = 448
    base_frequency: float = 1.50
    power: int = 220
    temperature: int = 93


@dataclass
class RTX2080(GPU):
    name: str = "RTX 2080"
    architecture: str = "Turing"
    cuda_cores: int = 2944
    memory: int = 8
    memory_type: str = "GDDR6"
    memory_bandwidth: float = 448
    base_frequency: float = 1.50
    power: int = 215
    temperature: int = 88


@dataclass
class GTX1660(GPU):
    name: str = "GTX 1660"
    architecture: str = "Turing"
    cuda_cores: int = 1408
    memory: int = 6
    memory_type: str = "GDDR5"
    memory_bandwidth: float = 192
    base_frequency: float = 1.53
    power: int = 120
    temperature: int = 95


@dataclass
class GTX1050(GPU):
    name: str = "GTX 1050"
    architecture: str = "Pascal"
    cuda_cores: int = 640
    memory: int = 2
    memory_type: str = "GDDR5"
    memory_bandwidth: float = 112
    base_frequency: float = 1.35
    power: int = 75
    temperature: int = 97


@dataclass
class GTX750Ti(GPU):
    name: str = "GTX 750 Ti"
    architecture: str = "Maxwell"
    cuda_cores: int = 640
    memory: int = 2
    memory_type: str = "GDDR5"
    memory_bandwidth: float = 86.4
    base_frequency: float = 1.02
    power: int = 60
    temperature: int = 95


GPU_MODELS = {
    "RTX 3090": RTX3090,
    "RTX 3080": RTX3080,
    "RTX 3070": RTX3070,
    "RTX 2080": RTX2080,
    "GTX 1660": GTX1660,
    "GTX 1050": GTX1050,
    "GTX 750 Ti": GTX750Ti,
}


def get_gpu_specs(gpu_name):
    """
    Returns the GPU specifications for the given GPU model.

    Parameters
    ----------
    gpu_name : str
        The name of the GPU model.

    Returns
    -------
    GPU
        The GPU specifications for the given GPU model.

    Raises
    ------
    ValueError
        If the GPU model is not found.
    """
    if gpu_name not in GPU_MODELS:
        raise ValueError(f"GPU model '{gpu_name}' not found.")
    return GPU_MODELS[gpu_name]()

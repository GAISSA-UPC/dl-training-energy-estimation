from typing import Union
from warnings import warn

import tensorflow as tf

from src.environment import GPU_MEM_LIMIT


def limit_gpu_memory(max_size: Union[int, None]):
    """
    Put a hard limit on the available GPU memory for the training.

    Parameters
    ----------
    max_size : Union[int, None]
        The maximum available GPU memory. If None the default TensorFlow configuration will be used.

    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        warn("No GPU(s) found. The training will be performed on the CPU.", UserWarning)
    if max_size is not None:
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                tf.config.set_logical_device_configuration(
                    gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=max_size)]
                )
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)


limit_gpu_memory(GPU_MEM_LIMIT)

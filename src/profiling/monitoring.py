import logging
import multiprocessing as mp
import os
import sys

from src.models.dl.model import BaseModel
from src.models.dl.train_model import run_main
from src.profiling.metrics import collect_linux_metrics, collect_windows_metrics

logger = logging.getLogger("profiler")


def monitor_training(model: BaseModel, out_file: str):
    """
    Monitor the training of a model.

    Parameters
    ----------
    model : BaseModel
        The model to train.
    out_file : str
        The file to write the metrics to.

    Returns
    -------
    run_id : str
        The ID of the run.
    acc_plot : str
        The path to the accuracy plot.
    model : BaseModel
        The trained model.
    """

    pid = os.getpid()
    arguments = (pid, out_file)
    if sys.platform == "linux":
        worker_process = mp.Process(target=collect_linux_metrics, args=arguments)
    elif sys.platform == "win32":
        worker_process = mp.Process(target=collect_windows_metrics, args=arguments)
    else:
        raise NotImplementedError("The monitoring function is not implemented for this OS.")
    worker_process.start()

    run_id = None
    acc_plot = None
    try:
        run_id, acc_plot, model = run_main(model, reproducible=True)
    except Exception as e:
        logger.error("Error while training model %s: %s", model.ARCHITECTURE, e)

    worker_process.terminate()

    return run_id, acc_plot, model

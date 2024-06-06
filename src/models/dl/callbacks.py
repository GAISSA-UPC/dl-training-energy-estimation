import csv
from datetime import datetime

from mlflow import log_text
from mlflow.tensorflow import MLflowCallback
from tensorflow.io.gfile import GFile
from tensorflow.keras.callbacks import Callback


class MetricsMonitor(MLflowCallback):
    """
    Callback to monitor the training process with MLflow.

    This callback logs model information at training start, and logs training metrics every epoch or
    every n steps (defined by the user) to MLflow.

    Parameters
    ----------
        run: `mlflow.entities.run.Run`
            The MLflow run.
        log_every_epoch: bool, optional
            If True, log metrics every epoch. If False, log metrics every n steps.
        log_every_n_steps: int, optional
            Log metrics every n steps. If None, log metrics every epoch. Must be `None` if `log_every_epoch=True`.
    """

    def on_train_begin(self, logs=None):
        """Log model architecture when training begins."""
        model_summary = []

        def print_fn(line, *args, **kwargs):
            model_summary.append(line)

        self.model.summary(print_fn=print_fn)
        summary = "\n".join(model_summary)
        log_text(summary, artifact_file="model_summary.txt")


class EpochEndCSVLogger(Callback):
    """
    Callback to log the end time of each epoch to a CSV file.

    Parameters
    ----------
    filename: str
        The name of the CSV file to write to.
    """

    def __init__(self, filename):
        self.filename = filename
        self.writer = None
        self.append_header = True
        super().__init__()

    def on_train_begin(self, logs=None):
        self.csv_file = GFile(self.filename, "w")

    def on_epoch_end(self, epoch, logs=None):
        """Log the end time of the epoch."""
        if not self.writer:
            self.writer = csv.DictWriter(self.csv_file, fieldnames=["epoch", "end_time"])
            if self.append_header:
                self.writer.writeheader()

        row_dict = {"epoch": epoch, "end_time": datetime.now()}
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None

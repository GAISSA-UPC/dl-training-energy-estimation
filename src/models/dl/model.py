"""
Module to define models using transfer learning for different datasets.
"""

import logging
import os
from abc import ABC
from datetime import datetime
from typing import Union

import mlflow
import tensorflow as tf
from mlflow.utils.autologging_utils import BatchMetricsLogger
from tensorflow.data import Dataset
from tensorflow.errors import ResourceExhaustedError
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

from src.data.datasets import BaseDatasetConfig
from src.environment import METRICS_DIR, MODELS_DIR
from src.models.dl.callbacks import EpochEndCSVLogger, MetricsMonitor
from src.profiling.metrics import compute_maccs


def configure_optimizer(
    optimizer: str = "adam", learning_rate: float = 1e-3, momentum: float = None
) -> tf.keras.optimizers.Optimizer:
    """
    Configure the optimizer for a TensorFlow model.

    Parameters
    ----------
    optimizer : str
        The name of the optimizer. Available options are: "adam", "sgd".
    learning_rate : float
        The learning rate to use for the optimizer.
    momentum : float
        The momentum to use for the optimizer if used at all.

    Returns
    -------
    optimizer
        The configured optimizer.
    """
    if optimizer == "adam":
        return Adam(learning_rate)
    if optimizer == "sgd":
        if momentum is None:
            raise ValueError("Momentum can not be None for SGD optimizer.")
        return SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)

    raise NotImplementedError(f"Configuration for {optimizer} optimizer is not implemented.")


class BaseModel(ABC):
    """
    Abstract class to define a base model.

    Attributes
    ----------
    ARCHITECTURE: str
        The name of the architecture.
    PREPROCESSING_LAYER
        The preprocessing layer of the model.
    BASE_MODEL: tf.keras.Model
        The base model.
    CLASSIFIER: tf.keras.layers.Layer
        The classifier layer of the model.
    INITIAL_EPOCHS: int
        The number of epochs to train the classifier.
    OPTIMIZER: str
        The optimizer to use for training the classifier.
    LEARNING_RATE: float
        The learning rate to use for training the classifier.
    MOMENTUM: float
        The momentum to use for training the classifier.
    FINE_TUNING_EPOCHS: int
        The number of epochs to fine-tune the model.
    FREEZE_AT: float
        Percentage of layers to freeze during fine-tuning. 1.0 = freeze all layers.
    FT_OPTIMIZER: str
        The optimizer to use for fine-tuning the model.
    FT_LEARNING_RATE: float
        The learning rate to use for fine-tuning the model.
    FT_MOMENTUM: float
        The momentum to use for fine-tuning the model.
    BATCH_SIZE: int
        The batch size to use for training the model.
    METRICS: list | str
        The metrics to use for training the model.
    LOSS: str
        The loss function to use for training the model.
    DATASET: BaseDatasetConfig
        The dataset to use for training the model.

    Methods
    -------
    build_model(input_shape=None)
        Builds the model.
    compile_model()
        Compiles the model.
    train()
        Trains the model.
    """

    ARCHITECTURE: str = None
    PREPROCESSING_LAYER = None
    BASE_MODEL: Model = None
    CLASSIFIER: Layer = None

    INITIAL_EPOCHS: int = 0
    OPTIMIZER: str = None
    LEARNING_RATE: float = None
    MOMENTUM: float = None

    FINE_TUNING_EPOCHS: int = 0
    FREEZE_AT: float = 1.0  # Percentage of layers to freeze during fine-tuning. 1.0 = freeze all layers.
    FT_OPTIMIZER: str = None
    FT_LEARNING_RATE: float = None
    FT_MOMENTUM: float = None

    INPUT_SHAPE: tuple = None
    BATCH_SIZE: int = None
    METRICS: Union[list, str] = None
    LOSS: str = None
    DATASET: BaseDatasetConfig = None

    LOGGER = logging.getLogger("BaseModel")

    def __init__(self):
        self._creation_time = datetime.now()
        self._model = None
        self._metrics_dir = (
            METRICS_DIR / "raw" / os.getenv("TRAINING_ENVIRONMENT", "unkown") / self.ARCHITECTURE / self.DATASET.NAME
        )
        os.makedirs(self._metrics_dir, exist_ok=True)

    @property
    def model(self) -> tf.keras.Model:
        """
        The model.

        Returns
        -------
        tf.keras.Model
            The model.
        """
        return self._model

    @property
    def creation_time(self):
        """
        The creation time of the model.

        Returns
        -------
        str
            The creation time of the model in the format YYYYMMDDTHHMMSS.
        """
        return self._creation_time.strftime("%Y%m%dT%H%M%S")

    def build_model(self) -> "BaseModel":
        """
        Builds the model.

        Parameters
        ----------
        input_shape: tuple
            The input shape of the model.

        Returns
        -------
        self
        """
        self.BASE_MODEL.trainable = False

        inputs = tf.keras.Input(shape=self.INPUT_SHAPE)
        x = tf.cast(inputs, tf.float32)
        x = self.PREPROCESSING_LAYER(x)
        x = self.BASE_MODEL(x, training=False)
        predictions = self.CLASSIFIER(x)
        self._model = tf.keras.Model(inputs=inputs, outputs=predictions)
        return self

    def compile_model(self) -> tf.keras.Model:
        """
        Compiles the model.

        Returns
        -------
        tf.keras.Model
            The compiled model.
        """
        if self.model is None:
            self.build_model()

        self.model.compile(
            optimizer=configure_optimizer(self.OPTIMIZER, self.LEARNING_RATE, self.MOMENTUM),
            loss=self.LOSS,
            metrics=self.METRICS,
        )

        return self.model

    def train(self, additional_callbacks=None, override_callbacks=False, mlflow_enabled=False) -> (dict, str):
        """
        Trains the model.

        Parameters
        ----------
        additional_callbacks: list | None
            Additional callbacks to use during training.
            By default, the callbacks are EarlyStopping and ReduceLROnPlateau.
        override_callbacks: bool
            Whether to override the default callbacks or not.
        mlflow_enabled: bool
            Whether to use MLflow or not.

        Returns
        -------
        dict
            The history of the training.
        str
            The MLflow run id.
        """
        tf.keras.backend.clear_session()

        if self.model is None:
            self.build_model().compile_model()

        maccs = compute_maccs(self.model)

        callbacks, early_stopping_callback = self._set_up_callbacks(additional_callbacks, override_callbacks)

        train_ds, val_ds = self.DATASET.load_dataset(self.INPUT_SHAPE, self.BATCH_SIZE)

        run_id = None
        try:
            if mlflow_enabled:
                history, history_fine, run_id = self._mlflow_training(
                    callbacks, early_stopping_callback, maccs, train_ds, val_ds
                )
            else:
                history, history_fine = self._regular_training(callbacks, early_stopping_callback, train_ds, val_ds)
        except ResourceExhaustedError:
            if self.DATASET.USE_CACHE:
                self.DATASET.USE_CACHE = False
                self.LOGGER.warning("Resource exhausted. Disabling cache and retrying...")
                mlflow.log_metric("resource_exhausted", 1)
                return self.train(additional_callbacks, override_callbacks, mlflow_enabled)
            return None, run_id

        history = self._join_histories(history, history_fine)

        return history, run_id

    def _regular_training(
        self,
        callbacks: list,
        early_stopping_callback: EarlyStopping,
        train_ds: Dataset,
        val_ds: Dataset,
    ):
        """
        Trains the model without logging to MLflow.

        Parameters
        ----------
        callbacks: list
            The callbacks to use during training.
        early_stopping_callback: tf.keras.callbacks.EarlyStopping
            The early stopping callback.
        train_ds: tf.data.Dataset
            The training dataset.
        val_ds: tf.data.Dataset
            The validation dataset.

        Returns
        -------
        tf.keras.callbacks.History
            The history of the training.
        tf.keras.callbacks.History
            The history of the fine-tuning.
        """
        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(self._metrics_dir, f"performance-{self.creation_time}.csv"),
            append=True,
        )
        callbacks.append(csv_logger)
        history = None
        history_fine = None
        if self.INITIAL_EPOCHS > 0:
            if early_stopping_callback is not None:
                early_stopping_callback.restore_best_weights = self.FINE_TUNING_EPOCHS == 0
            history = self._train_classifier(train_ds, val_ds, callbacks)
        if self.FINE_TUNING_EPOCHS > 0:
            if early_stopping_callback is not None:
                early_stopping_callback.restore_best_weights = True
            history_fine = self._fine_tune_classifier(
                train_ds,
                val_ds,
                callbacks,
                history,
            )
        self.model.save(os.path.join(MODELS_DIR, self.ARCHITECTURE))
        return history, history_fine

    def _mlflow_training(
        self,
        callbacks: list[Callback],
        early_stopping_callback: EarlyStopping,
        maccs: float,
        train_ds: Dataset,
        val_ds: Dataset,
    ):
        """
        Trains the model and logs the training to MLflow.

        Parameters
        ----------
        callbacks: list
            The callbacks to use during training.
        early_stopping_callback: tf.keras.callbacks.EarlyStopping
            The early stopping callback.
        maccs: float
            The MACCS of the model.
        train_ds: tf.data.Dataset
            The training dataset.
        val_ds: tf.data.Dataset
            The validation dataset.

        Returns
        -------
        tf.keras.callbacks.History
            The history of the training.
        tf.keras.callbacks.History
            The history of the fine-tuning.
        str
            The MLflow run id.
        """
        with mlflow.start_run(tags={"model": self.ARCHITECTURE, "dataset": self.DATASET.NAME}) as run:
            metrics_logger = BatchMetricsLogger(run.info.run_id)
            self._log_basic_params()

            self._log_dataset_info()

            callbacks.append(MetricsMonitor(run))

            self._log_callbacks_params(callbacks)
            history = None
            initial_epochs = 0
            if self.INITIAL_EPOCHS > 0:
                if early_stopping_callback is not None:
                    early_stopping_callback.restore_best_weights = self.FINE_TUNING_EPOCHS == 0
                history = self._train_classifier(train_ds, val_ds, callbacks)
                initial_epochs = len(history.history["loss"])
                self._log_early_stop_callback_metrics(early_stopping_callback, history, metrics_logger)

            history_fine = None
            fine_tuning_epochs = 0
            if self.FINE_TUNING_EPOCHS > 0:
                if early_stopping_callback is not None:
                    early_stopping_callback.restore_best_weights = True
                self._log_callbacks_params(callbacks, fine_tunning=True)
                history_fine = self._fine_tune_classifier(
                    train_ds,
                    val_ds,
                    callbacks,
                    history,
                )
                fine_tuning_epochs = len(history_fine.history["loss"])
                self._log_early_stop_callback_metrics(
                    early_stopping_callback, history_fine, metrics_logger, fine_tunning=True
                )

            mlflow.log_metric("MACCS", maccs)
            mlflow.log_metric("trained_epochs", initial_epochs + fine_tuning_epochs)

            run_id = run.info.run_id
        return history, history_fine, run_id

    def _set_up_callbacks(
        self, additional_callbacks: list[tf.keras.callbacks.Callback], override_callbacks: bool
    ) -> list[tf.keras.callbacks.Callback]:
        """
        Set up the callbacks for training the model.

        Parameters
        ----------
        additional_callbacks: list | None
            Additional callbacks to use during training.
            By default, the callbacks are EarlyStopping and ReduceLROnPlateau.
        override_callbacks: bool
            Whether to override the default callbacks or not.

        Returns
        -------
        callbacks: list
            The callbacks to use during training.
        """
        early_stopping_callback = None
        if not override_callbacks:
            early_stopping_callback = EarlyStopping(
                monitor="val_loss",
                patience=20,
                mode="auto",
                restore_best_weights=False,
                verbose=1,
                min_delta=1e-3,
                start_from_epoch=100,
            )
            callbacks = [
                early_stopping_callback,
                ReduceLROnPlateau(
                    monitor="val_loss",
                    mode="auto",
                    factor=0.1,
                    patience=15,
                    min_delta=1e-3,
                    min_lr=1e-6,
                ),
                EpochEndCSVLogger(
                    os.path.join(self._metrics_dir, f"epoch_end-{self.creation_time}.csv"),
                ),
            ]

            if additional_callbacks is not None:
                callbacks.extend(additional_callbacks)
        elif additional_callbacks is not None:
            callbacks = additional_callbacks
            for callback in callbacks:
                if isinstance(callback, EarlyStopping):
                    early_stopping_callback = callback
                    break
        else:
            callbacks = []
        return callbacks, early_stopping_callback

    def _train_classifier(self, train_ds, val_ds, callbacks):
        """Trains the classifier.

        Parameters
        ----------
        train_ds: tf.data.Dataset
            The training dataset.
        val_ds: tf.data.Dataset
            The validation dataset.
        callbacks: list
            The callbacks to use during training.

        Returns
        -------
        tf.keras.callbacks.History
            The history of the training.
        """
        try:
            history = self._model.fit(
                train_ds,
                steps_per_epoch=self.DATASET.TRAIN_SIZE // self.BATCH_SIZE,
                validation_data=val_ds,
                validation_steps=self.DATASET.VAL_SIZE // self.BATCH_SIZE,
                epochs=self.INITIAL_EPOCHS,
                callbacks=callbacks,
                verbose=1,
            )
        except ResourceExhaustedError as e:
            mlflow.log_metric("resource_exhausted", 1)
            raise e

        return history

    def _fine_tune_classifier(self, train_ds, val_ds, callbacks, history=None):
        """Fine-tunes the classifier.

        Parameters
        ----------
        train_ds: tf.data.Dataset
            The training dataset.
        val_ds: tf.data.Dataset
            The validation dataset.
        callbacks: list
            The callbacks to use during training.
        history: tf.keras.callbacks.History | None
            The history of the training of the classifier.

        Returns
        -------
        tf.keras.callbacks.History
            The history of the training.
        """
        self.BASE_MODEL.trainable = True
        start_layer = int(len(self.BASE_MODEL.layers) * self.FREEZE_AT)
        for layer in self.BASE_MODEL.layers[:start_layer]:
            layer.trainable = False

        optimizer = configure_optimizer(self.FT_OPTIMIZER, self.FT_LEARNING_RATE, self.FT_MOMENTUM)
        self.model.compile(
            loss=self.LOSS,
            optimizer=optimizer,
            metrics=self.METRICS,
        )

        total_epochs = self.INITIAL_EPOCHS + self.FINE_TUNING_EPOCHS
        initial_epoch = 0 if history is None else history.epoch[-1] + 1
        self.LOGGER.info("Starting fine-tunning...")

        try:
            history_fine = self.model.fit(
                train_ds,
                steps_per_epoch=self.DATASET.TRAIN_SIZE // self.BATCH_SIZE,
                epochs=total_epochs,
                initial_epoch=initial_epoch,
                validation_data=val_ds,
                validation_steps=self.DATASET.VAL_SIZE // self.BATCH_SIZE,
                callbacks=callbacks,
            )
        except ResourceExhaustedError as e:
            mlflow.log_metric("resource_exhausted", 1)
            raise e

        return history_fine

    def _join_histories(self, history, history_fine):
        """
        Join two histories from two different training sessions.

        Parameters
        ----------
        history: tf.keras.callbacks.History
            The history of the first training session.
        history_fine: tf.keras.callbacks.History
            The history of the second training session.

        Returns
        -------
        hist: dict
            The joined history.
        """
        if history is None and history_fine is None:
            return None
        if history is None and history_fine is not None:
            return history_fine.history
        if history_fine is None:
            return history.history

        hist = {
            "loss": history.history["loss"] + history_fine.history["loss"],
            "val_loss": history.history["val_loss"] + history_fine.history["val_loss"],
        }
        for metric in self.METRICS:
            metric = metric.lower()
            hist[metric] = history.history[metric] + history_fine.history[metric]
            hist[f"val_{metric}"] = history.history[f"val_{metric}"] + history_fine.history[f"val_{metric}"]

        return hist

    def _log_basic_params(self):
        params = {
            "batch_size": self.BATCH_SIZE,
            "epochs_cl": self.INITIAL_EPOCHS,
            "epochs_ft": self.FINE_TUNING_EPOCHS,
            "epochs": self.INITIAL_EPOCHS + self.FINE_TUNING_EPOCHS,
            "optimizer": self.OPTIMIZER,
            "opt_lr": self.LEARNING_RATE,
            "opt_momentum": self.MOMENTUM,
            "optimizer_ft": self.FT_OPTIMIZER,
            "opt_lr_ft": self.FT_LEARNING_RATE,
            "opt_momentum_ft": self.FT_MOMENTUM,
            "image_size": self.DATASET.IMAGE_DIM[:-1],
        }
        mlflow.log_params(params)

    def _log_dataset_info(self):
        params = {
            "train_size": self.DATASET.TRAIN_SIZE,
            "validation_size": self.DATASET.VAL_SIZE,
        }
        mlflow.log_params(params)

    def _log_callbacks_params(self, callbacks, fine_tunning=False):
        suffix = "_ft" if fine_tunning else ""
        for callback in callbacks:
            if isinstance(callback, tf.keras.callbacks.ReduceLROnPlateau):
                params = {
                    f"reducelr_monitor{suffix}": callback.monitor,
                    f"reducelr_min_delta{suffix}": callback.min_delta,
                    f"reducelr_patience{suffix}": callback.patience,
                    f"reducelr_factor{suffix}": callback.factor,
                }
                mlflow.log_params(params)
            if isinstance(callback, tf.keras.callbacks.EarlyStopping):
                params = {
                    f"earlystopping_monitor{suffix}": callback.monitor,
                    f"earlystopping_min_delta{suffix}": callback.min_delta,
                    f"earlystopping_patience{suffix}": callback.patience,
                    f"earlystopping_restore_best_weights{suffix}": callback.restore_best_weights,
                }
                mlflow.log_params(params)

    def _log_early_stop_callback_metrics(self, callback, history, metrics_logger, fine_tunning=False):
        if callback is None:
            return

        suffix = "_ft" if fine_tunning else "_cl"
        stopped_epoch = callback.stopped_epoch
        restore_best_weights = callback.restore_best_weights

        metrics_logger.record_metrics({f"stopped_epoch{suffix}": stopped_epoch})

        # Do not logg metrics if the callback did not stop the training
        if not restore_best_weights or callback.best_weights is None:
            return

        monitored_metric = history.history.get(callback.monitor)
        if not monitored_metric:
            return

        restored_epoch = callback.best_epoch
        metrics_logger.record_metrics({f"restored_epoch{suffix}": restored_epoch})
        restored_index = history.epoch.index(restored_epoch)
        restored_metrics = {key: metrics[restored_index] for key, metrics in history.history.items()}
        # Checking that a metric history exists
        metric_key = next(iter(history.history), None)
        if metric_key is not None:
            metrics_logger.record_metrics(restored_metrics, stopped_epoch + 1)

import logging
import os.path
import random as python_random

import numpy as np
import tensorflow as tf

from src.environment import FIGURES_DIR, SEED
from src.models.dl.model import BaseModel
from src.profiling import MLFLOW_ENABLED
from src.utils import plot_hist

logger = logging.getLogger("profiler")

# def create_model(
#     architecture: str, num_classes: int, weights: str = "imagenet"
# ) -> Tuple[tf.keras.models.Model, tf.keras.models.Model]:
#     """
#     Creates a ``tensorflow.keras.models.Model`` with the given architecture as the base model.
#
#     Parameters
#     ----------
#     architecture : str
#         The base architecture for the model.
#     num_classes : int
#         The number of classes for the model.
#     weights : str, default "imagenet"
#         The weights to use for the pretrained model.
#
#     Returns
#     -------
#     model : ``tensorflow.keras.models.Model``
#         The created model.
#     base_model : ``tensorflow.keras.models.Model``
#         The base model.
#     """
#     optimizer = configure_optimizer(OPTIMIZER, LEARNING_RATE, MOMENTUM)
#     if architecture == "vgg16":
#         return create_VGG16(optimizer, num_classes, weights)
#     elif architecture == "resnet50":
#         return create_ResNet50(optimizer, num_classes, weights)
#     elif architecture == "xception":
#         return create_Xception(optimizer, num_classes, weights)
#     elif architecture == "mobilenet_v2":
#         return create_MobileNetV2(optimizer, num_classes, weights)
#     elif architecture == "nasnet_mobile":
#         return create_NASNetMobile(optimizer, num_classes, weights)
#     else:
#         raise NotImplementedError("Unknown architecture type.")


# def train(
#     model: tf.keras.models.Model,
#     base_model: tf.keras.models.Model,
#     architecture: str,
#     train_ds: tf.data.Dataset,
#     train_size: int,
#     val_ds: tf.data.Dataset,
#     val_size: int,
#     crossentropy_type: str,
# ):
#     maccs = compute_maccs(model)
#
#     tf.keras.backend.clear_session()
#
#     early_stopping_callback = tf.keras.callbacks.EarlyStopping(
#         monitor=f"val_accuracy",
#         patience=20,
#         mode="auto",
#         restore_best_weights=False,
#         verbose=1,
#         min_delta=1e-3,
#         start_from_epoch=100,
#     )
#     callbacks = [
#         early_stopping_callback,
#         tf.keras.callbacks.ReduceLROnPlateau(
#             monitor=f"val_accuracy",
#             mode="auto",
#             factor=0.1,
#             patience=15,
#             min_delta=1e-3,
#             min_lr=1e-6,
#         ),
#     ]
#     if MLFLOW_ENABLED:
#         with mlflow.start_run(tags={"model": architecture}) as run, mlflow.tensorflow.batch_metrics_logger(
#             run.info.run_id
#         ) as metrics_logger:
#             _log_basic_params()
#
#             _log_dataset_info(train_size, val_size)
#
#             callbacks.append(MetricsMonitor(metrics_logger))
#
#             _log_callbacks_params(callbacks)
#             if INITIAL_EPOCHS > 0:
#                 history = train_classifier(model, train_ds, train_size, val_ds, val_size, callbacks)
#                 _log_early_stop_callback_metrics(early_stopping_callback, history, metrics_logger)
#             else:
#                 history = None
#
#             early_stopping_callback.restore_best_weights = True
#             _log_callbacks_params(callbacks, fine_tunning=True)
#             history_fine = fine_tune_classifier(
#                 base_model,
#                 model,
#                 train_ds,
#                 train_size,
#                 val_ds,
#                 val_size,
#                 callbacks,
#                 crossentropy_type,
#                 history,
#             )
#             _log_early_stop_callback_metrics(early_stopping_callback, history_fine, metrics_logger, fine_tunning=True)
#
#             mlflow.log_metric("MACCS", maccs)
#             # model.save(os.path.join(MODELS_DIR, f"{architecture}-{task}"))
#
#         run_id = run.info.run_id
#     else:
#         run_id = None
#         metrics_dir = METRICS_DIR / "raw" / architecture
#         os.makedirs(metrics_dir, exist_ok=True)
#         csv_logger = tf.keras.callbacks.CSVLogger(
#             os.path.join(metrics_dir, f"performance-{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv"),
#             append=True,
#         )
#         callbacks.append(csv_logger)
#         if INITIAL_EPOCHS > 0:
#             history = train_classifier(model, train_ds, train_size, val_ds, val_size, callbacks)
#         else:
#             history = None
#
#         early_stopping_callback.restore_best_weights = True
#         history_fine = fine_tune_classifier(
#             base_model,
#             model,
#             train_ds,
#             train_size,
#             val_ds,
#             val_size,
#             callbacks,
#             crossentropy_type,
#             history,
#         )
#         model.save(os.path.join(MODELS_DIR, architecture))
#
#     if INITIAL_EPOCHS > 0:
#         hist = join_histories(history, history_fine)
#         return model, hist, run_id
#     return model, history_fine, run_id
#
#
# @deprecated
# def join_histories(history, history_fine):
#     hist = {
#         "loss": history.history["loss"] + history_fine.history["loss"],
#         "val_loss": history.history["val_loss"] + history_fine.history["val_loss"],
#     }
#     for metric in METRICS:
#         metric = metric.lower()
#         hist[metric] = history.history[metric] + history_fine.history[metric]
#         hist[f"val_{metric}"] = history.history[f"val_{metric}"] + history_fine.history[f"val_{metric}"]
#
#     return hist
#
#
# def train_classifier(
#     model: tf.keras.models.Model,
#     train_ds: tf.data.Dataset,
#     train_size: int,
#     val_ds: tf.data.Dataset,
#     val_size: int,
#     callbacks: List[tf.keras.callbacks.Callback],
# ):
#     history = model.fit(
#         train_ds,
#         steps_per_epoch=train_size // BATCH_SIZE,
#         validation_data=val_ds,
#         validation_steps=val_size // BATCH_SIZE,
#         epochs=INITIAL_EPOCHS,
#         callbacks=callbacks,
#         verbose=1,
#     )
#     return history
#
#
# def fine_tune_classifier(
#     base_model, model, train_ds, train_size, val_ds, val_size, callbacks, crossentropy_type, history=None
# ):
#     base_model.trainable = True
#     start_layer = int(len(base_model.layers) * FINE_TUNE_AT)
#     for layer in base_model.layers[:start_layer]:
#         layer.trainable = False
#
#     optimizer = configure_optimizer(FT_OPTIMIZER, FT_LEARNING_RATE, FT_MOMENTUM)
#     model.compile(
#         loss=f"{crossentropy_type}_crossentropy",
#         optimizer=optimizer,
#         metrics=METRICS,
#     )
#
#     total_epochs = INITIAL_EPOCHS + FT_EPOCHS
#     initial_epoch = 0 if history is None else history.epoch[-1] + 1
#     print("Starting fine-tunning...")
#     history_fine = model.fit(
#         train_ds,
#         steps_per_epoch=train_size // BATCH_SIZE,
#         epochs=total_epochs,
#         initial_epoch=initial_epoch,
#         validation_data=val_ds,
#         validation_steps=val_size // BATCH_SIZE,
#         callbacks=callbacks,
#     )
#     return history_fine
#
#
# def create_VGG16(optimizer: tf.keras.optimizers.Optimizer, num_classes: int, weights: str = "imagenet"):
#     shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
#     base_model = VGG16(weights=weights, include_top=False, input_shape=shape)
#     # Freeze layers
#     base_model.trainable = False
#
#     # Establish new fully connected block
#     inputs = tf.keras.Input(shape=shape)
#     x = tf.cast(inputs, tf.float32)
#     x = tf.keras.applications.vgg16.preprocess_input(x)
#     x = base_model(x, training=False)
#     x = GlobalAveragePooling2D()(x)
#     model = build_model(inputs, x, optimizer, num_classes)
#     return model, base_model
#
#
# def create_ResNet50(optimizer: tf.keras.optimizers.Optimizer, num_classes: int, weights: str = "imagenet"):
#     shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
#     base_model = ResNet50(weights=weights, include_top=False, input_shape=shape)
#     # Freeze layers
#     base_model.trainable = False
#
#     # Establish new fully connected block
#     inputs = tf.keras.Input(shape=shape)
#     x = tf.cast(inputs, tf.float32)
#     x = tf.keras.applications.resnet50.preprocess_input(x)
#     x = base_model(x, training=False)
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation="relu")(x)
#     x = Dense(512, activation="relu")(x)
#     model = build_model(inputs, x, optimizer, num_classes)
#     return model, base_model
#
#
# def create_Xception(optimizer: tf.keras.optimizers.Optimizer, num_classes: int, weights: str = "imagenet"):
#     shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
#     base_model = Xception(weights=weights, include_top=False, input_shape=shape)
#     # Freeze layers
#     base_model.trainable = False
#
#     # Establish new fully connected block
#     inputs = tf.keras.Input(shape=shape)
#     x = tf.cast(inputs, tf.float32)
#     x = tf.keras.applications.xception.preprocess_input(x)
#     x = base_model(x, training=False)
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation="relu")(x)
#     model = build_model(inputs, x, optimizer, num_classes)
#     return model, base_model
#
#
# def create_MobileNetV2(optimizer: tf.keras.optimizers.Optimizer, num_classes: int, weights: str = "imagenet"):
#     shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
#     base_model = MobileNetV2(weights=weights, include_top=False, input_shape=shape, alpha=0.5)
#     # Freeze layers
#     base_model.trainable = False
#
#     # Establish new fully connected block
#     inputs = tf.keras.Input(shape=shape)
#     x = tf.cast(inputs, tf.float32)
#     x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
#     x = base_model(x, training=False)
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(512, activation="relu")(x)
#     model = build_model(inputs, x, optimizer, num_classes)
#     return model, base_model
#
#
# def create_NASNetMobile(optimizer: tf.keras.optimizers.Optimizer, num_classes: int, weights: str = "imagenet"):
#     shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
#     base_model = NASNetMobile(weights=weights, include_top=False, input_shape=shape)
#     # Freeze layers
#     base_model.trainable = False
#
#     inputs = tf.keras.Input(shape=shape)
#     x = tf.cast(inputs, tf.float32)
#     x = tf.keras.applications.nasnet.preprocess_input(x)
#     x = base_model(x, training=False)
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(512, activation="relu")(x)
#     model = build_model(inputs, x, optimizer, num_classes)
#     return model, base_model
#
#
# def build_model(inputs, x, optimizer, num_classes):
#     # This is the model we will train
#     if num_classes > 2:
#         predictions = Dense(num_classes, activation="softmax")(x)
#         model = Model(inputs=inputs, outputs=predictions)
#         model.compile(
#             optimizer=optimizer,
#             loss="categorical_crossentropy",
#             metrics=METRICS,
#         )
#     else:
#         predictions = Dense(1, activation="sigmoid")(x)
#         model = Model(inputs=inputs, outputs=predictions)
#         model.compile(
#             optimizer=optimizer,
#             loss="binary_crossentropy",
#             metrics=METRICS,
#         )
#     return model


def fix_seeds():
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(SEED)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    python_random.seed(SEED)

    # The below set_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    tf.random.set_seed(SEED)

    os.environ["PYTHONHASHSEED"] = "0"


def run_main(model: BaseModel, reproducible=True):
    if reproducible:
        fix_seeds()

    history, run_id = model.train(mlflow_enabled=MLFLOW_ENABLED)

    if history is None:
        logger.warning("The training failed.")
        return run_id, None, model

    figure_path = plot_hist(history, "accuracy", model.ARCHITECTURE, FIGURES_DIR)
    return run_id, figure_path, model

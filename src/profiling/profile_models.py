import gc
import logging
import os
import shlex
import subprocess
import time

import mlflow

from src.environment import DATASET_DIR, METRICS_DIR
from src.mlflow_utils.utils import upload_mlflow_run_metadata
from src.models.dl.model_factory import InceptionResNetV2Factory, InceptionV3Factory
from src.models.dl.time_stopping import TimeStopping
from src.profiling import (
    COOLDOWN,
    COOLDOWN_EVERY,
    MEASURING_INTERVAL,
    MINUTES_TO_SECONDS,
    MLFLOW_ENABLED,
    WARMUP_TIME,
)
from src.profiling.monitoring import monitor_training
from src.utils import create_folder_if_not_exists

logger = logging.getLogger("profiler")


def warmup():
    print("Starting warmup...")
    warmup_model = InceptionV3Factory().get_model("cifar10", (128, 128, 3), 32)
    warmup_model.INITIAL_EPOCHS = 1000
    warmup_model.train(
        additional_callbacks=[TimeStopping(seconds=int(WARMUP_TIME * MINUTES_TO_SECONDS), verbose=1)],
        override_callbacks=True,
    )


def profile_model(
    environment,
    arch,
    dataset,
    input_size,
    batch_size,
    user_defined_experiment_name=None,
    data_folder=DATASET_DIR,
    experiment_run=None,
    runs=0,
):
    os.environ["TRAINING_ENVIRONMENT"] = environment
    input_shape = (input_size, input_size, 3)
    if arch == "inception_v3":
        model = InceptionV3Factory().get_model(dataset, input_shape, batch_size)
    elif arch == "inception_resnet_v2":
        model = InceptionResNetV2Factory().get_model(dataset, input_shape, batch_size)
    else:
        raise NotImplementedError(f"The architecture {arch} is not implemented.")

    if experiment_run is None:
        logger.info("Start run for architecture %s with dataset %s", arch, dataset)
    else:
        logger.info("Start run %s for architecture %s with dataset %s", experiment_run, arch, dataset)

    if MLFLOW_ENABLED:
        if user_defined_experiment_name is None:
            experiment_name = f"{environment}-{dataset}-{arch}"
        else:
            experiment_name = user_defined_experiment_name

        mlflow.set_experiment(experiment_name)

    out_dir = METRICS_DIR / "raw" / environment / arch / dataset

    create_folder_if_not_exists(out_dir)

    if COOLDOWN_EVERY > 0 and runs != 0 and runs % COOLDOWN_EVERY == 0:
        logger.info("Waiting %s minutes to cooldown", COOLDOWN)
        time.sleep(COOLDOWN * MINUTES_TO_SECONDS)

    # Run garbage collector to free up memory
    gc.collect()

    creation_time = model.creation_time
    gpu_metrics = os.path.join(out_dir, f"gpu-power-{creation_time}.csv")
    cpu_metrics = os.path.join(out_dir, f"cpu-mem-usage-{creation_time}.csv")

    gpu_id = os.getenv("GPU_DEVICE_ORDINAL", "0")
    command = f"nvidia-smi -i {gpu_id} --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -lms {MEASURING_INTERVAL} -f {gpu_metrics}"
    with subprocess.Popen(shlex.split(command)) as gpu_process:
        time.sleep(3)
        run_id, acc_plot, model = monitor_training(model, cpu_metrics)
        time.sleep(3)
        gpu_process.terminate()

    if MLFLOW_ENABLED and run_id is not None:
        upload_mlflow_run_metadata(
            environment,
            model,
            run_id,
            acc_plot,
            cpu_metrics,
            gpu_metrics,
        )

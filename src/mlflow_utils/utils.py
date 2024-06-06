import os
import shutil
import time

import mlflow
import numpy as np
import tensorflow as tf
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from tqdm import tqdm

from src.environment import METRICS_DIR
from src.models.dl.model import BaseModel


def log_model(run_id, model_path, model_name):
    with mlflow.start_run(run_id=run_id):
        model = tf.keras.models.load_model(model_path)
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=model_name,
            await_registration_for=None,
        )


def upload_artifact(run_id, path, dst_folder):
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(path, dst_folder)


def upload_artifacts(run_id, path, dst_folder):
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifacts(path, dst_folder)


def compute_runs_f1score(runs):
    """
    Computes the validation F1-score for all the `runs`.
    Parameters
    ----------
    runs : `pandas.DataFrame`
        A dataframe with the precision and recall of the runs.

    Returns
    -------
    f1-score : `pandas.Series`
        The F1-score of the runs.
    """
    precision = runs.get("metrics.val_precision", runs.get("metrics.val_precision_1"))
    recall = runs.get("metrics.val_recall", runs.get("metrics.val_recall_1"))
    if precision is None or recall is None:
        return 0.0
    return (2 * (precision * recall) / (precision + recall + 1e-10)).fillna(0)


def search_runs_with_models(architecture, dataset):
    client = mlflow.client.MlflowClient()
    models = client.search_model_versions(filter_string=f"name = '{architecture}-{dataset}'")
    if models:
        run_ids = [f"'{model.run_id}'" for model in models]
        selection = ",".join(run_ids)
        runs = mlflow.search_runs(search_all_experiments=True, filter_string=f"run_id IN ({selection})")
        return runs
    return None


def search_best_f1score_and_loss(architecture, dataset):
    """
    Search the best F1-score and loss for a given architecture and dataset.

    Parameters
    ----------
    architecture : str
        The architecture to search for the F1-score.
    dataset : str
        The dataset with which the model was trained.

    Returns
    -------
    f1-score : float
        The best F1-score for the architecture and dataset.
    loss : float
        The best loss for the architecture and dataset.

    """
    runs = search_runs_with_models(architecture, dataset)
    if runs is None:
        return 0.0, 1e10
    return np.max(compute_runs_f1score(runs)), np.min(runs.get("metrics.val_loss", 1e10))


def download_best_model(architecture, dataset, out_path=None):
    runs = search_runs_with_models(architecture, dataset)
    if runs:
        runs["metrics.val_f1score"] = compute_runs_f1score(runs)
        best_run = runs.iloc[runs["metrics.val_f1score"].idxmax()].run_id
        print(f"Best run ID: {best_run}")
        mlflow.artifacts.download_artifacts(run_id=best_run, artifact_path="model/data/model", dst_path=out_path)
        shutil.move(
            os.path.join(out_path, "model/data/model"),
            os.path.join(out_path, f"best-{architecture}-{dataset}"),
        )
        shutil.rmtree(os.path.join(out_path, "model"))


def download_energy_metrics(strategy, architecture, dataset):
    runs = mlflow.search_runs(
        experiment_names=[f"ChessLive-{strategy}-{dataset}-{architecture}"],
        filter_string="attribute.status = 'FINISHED'",
    )

    out_path = os.path.join(METRICS_DIR, "raw", dataset, strategy)
    for run in tqdm(runs.run_id):
        try:
            mlflow.artifacts.download_artifacts(run_id=run, artifact_path="metrics", dst_path=out_path)
        except ConnectionResetError:
            # Suspend download for 5 seconds if connection is closed by the server and try again.
            print("Waiting 5 seconds")
            time.sleep(5)
            mlflow.artifacts.download_artifacts(run_id=run, artifact_path="metrics", dst_path=out_path)
        except Exception as e:
            print(e)

    os.rename(os.path.join(out_path, "metrics"), os.path.join(out_path, architecture))


def upload_mlflow_run_metadata(
    environment: str, model: BaseModel, run_id: str, acc_plot: str, cpu_metrics: str, gpu_metrics: str
):
    arch = model.ARCHITECTURE
    dataset = model.DATASET.NAME
    best_f1, best_loss = search_best_f1score_and_loss(arch, dataset)
    with mlflow.start_run(run_id=run_id) as run:
        mlflow.set_tag("environment", environment)
        mlflow.set_tag("dataset", dataset)
        if acc_plot is not None:
            mlflow.log_artifact(acc_plot, "figures")
        if cpu_metrics is not None:
            mlflow.log_artifact(cpu_metrics, "metrics")
        if gpu_metrics is not None:
            mlflow.log_artifact(gpu_metrics, "metrics")

        loss = run.data.metrics.get("val_loss", None)
        precision = run.data.metrics.get("val_precision", None)
        recall = run.data.metrics.get("val_recall", None)

        input_schema = Schema(
            [
                TensorSpec(np.dtype(np.float64), [-1] + list(model.INPUT_SHAPE)),
            ]
        )
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, model.DATASET.NUM_CLASSES))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        if precision is not None and recall is not None:
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            if f1 >= best_f1:
                mlflow.tensorflow.log_model(
                    model, "model", registered_model_name=f"{arch}-{dataset}", signature=signature
                )
        elif loss is not None and run.data.metrics.get("val_loss") <= best_loss:
            mlflow.tensorflow.log_model(
                model.model, "model", registered_model_name=f"{arch}-{dataset}", signature=signature
            )

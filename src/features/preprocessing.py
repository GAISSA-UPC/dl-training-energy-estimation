"""
This module contains the functions used to preprocess the collected data.
"""

import concurrent.futures
import glob
import os
import re
from pathlib import Path
from typing import List, Union

import mlflow
import numpy as np
import pandas as pd

from src.environment import METRICS_DIR, OUTPUT_DIR, logger

POWER_MEASUREMENT_ERROR = 5

WATTS_TO_KWATTS = 1e-3

JOULES_TO_KWH = 1 / 3.6e6

JOULES_TO_KJOULES = 1e-3
JOULES_TO_MJOULES = 1e-6
JOULES_TO_GJOULES = 1e-9

KJOULES_TO_JOULES = 1e3
KJOULES_TO_MJOULES = 1e-3

MJOULES_TO_JOULES = 1e6
MJOULES_TO_KJOULES = 1e3

GJOULES_TO_JOULES = 1e9
GJOULES_TO_KJOULES = 1e6
GJOULES_TO_MJOULES = 1e3

KWH_CO2e_SPA = 232
gCO2e_TO_TCO2e = 1e-6
FLOPS_TO_GFLOPS = 1e-9

HOURS_TO_SECONDS = 3.6e3
HOURS_TO_MILISECONDS = 3.6e6
SECONDS_TO_HOURS = 1 / 3.6e3

USED_MB_TO_WATTS = 3.75e-4


def measure2float(value: str):
    """
    Convert a string with a measure to a float.

    Parameters
    ----------
    value : str
        The string with the measure.

    Returns
    -------
    measure : float
        The measure as a float.
    """
    if value == "None" or not value:
        return None
    return float(re.findall(r"\d+", value)[0])


def measure2int(value: str):
    """
    Convert a string with a measure to an integer.

    Parameters
    ----------
    value : str
        The string with the measure.

    Returns
    -------
    measure : int
        The measure as an integer.
    """
    if value == "None" or not value:
        return None
    return np.int32(re.findall(r"\d+", value)[0])


def get_epoch_ends(run_data: pd.Series):
    """
    Get the epoch ends for the given run.

    Parameters
    ----------
    run_data : `pd.Series`
        A `pd.Series` with the data for the given run. It must contain the columns ``train_environment``,
        ``architecture``, ``dataset``, and ``creation_time``.

    Returns
    -------
    dataframe : `pd.DataFrame`
        A `pd.DataFrame` with the epoch ends for the given experiment.
    """
    training_environment = run_data["train_environment"]
    architecture = run_data["architecture"]
    dataset = run_data["dataset"]
    creation_time = run_data["creation_time"]
    try:
        return pd.read_csv(
            METRICS_DIR / "raw" / training_environment / architecture / dataset / f"epoch_end-{creation_time}.csv",
            delimiter=",",
            header=0,
            dtype={"epoch": int},
            parse_dates=["end_time"],
            date_format="%Y-%m-%d %H:%M:%S.%f",
        ).sort_values(by=["epoch"])
    except FileNotFoundError:
        return None


def build_metrics_dataset(input_folder=None, save_to_file=False, process_all_files=False):
    """
    Build the metrics dataset from the data in `input_folder`. If `output_folder` is specified it will save the dataset
    as a ``gzip`` file.

    Parameters
    ----------
    input_folder : `os.PathLike`, default None
        The folder with the collected data. If `None`, the default saving location will be used.
    save_to_file : boolean, default False
        Whether to save the resulting dataset to a parquet file or not.
    process_all_files : boolean, default False
        Whether to process all the files in the input folder or only the new ones.

    Returns
    -------
    dataframe : `pd.DataFrame`
        A `pd.DataFrame` with the collected data properly formatted.
    """
    if input_folder is None:
        input_dir = Path(METRICS_DIR) / "raw"
    elif isinstance(input_folder, str):
        input_dir = Path(input_folder)
    else:
        input_dir = input_folder

    df = pd.DataFrame()
    new_cpu_files = []
    new_gpu_files = []
    if process_all_files:
        cpu_files_processed = pd.DataFrame({"file": []})
        gpu_files_processed = pd.DataFrame({"file": []})
    else:
        cpu_files_processed, gpu_files_processed = _get_processed_files()
    for train_environment in os.listdir(input_dir):
        architectures_folder = input_dir / train_environment
        for architecture in os.listdir(architectures_folder):
            datasets_folder = architectures_folder / architecture
            for dataset in os.listdir(datasets_folder):
                if (datasets_folder / dataset).is_file():
                    continue
                cpu_files = (datasets_folder / dataset).glob(r"cpu*.csv")
                cpu_files = [
                    str(cpu_file) for cpu_file in cpu_files if str(cpu_file) not in cpu_files_processed["file"].values
                ]
                cpu_files = sorted(cpu_files, key=lambda x: os.path.basename(x).split("-")[-1])
                gpu_files = (datasets_folder / dataset).glob(r"gpu*.csv")
                gpu_files = [
                    str(gpu_file) for gpu_file in gpu_files if str(gpu_file) not in gpu_files_processed["file"].values
                ]
                gpu_files = sorted(gpu_files, key=lambda x: os.path.basename(x).split("-")[-1])
                if not cpu_files and not gpu_files:
                    continue
                experiment_name = f"{train_environment}-{dataset}-{architecture}"
                if experiment_name == "cloud-stanford_dogs-inception_v3":
                    experiment_names = [experiment_name, "memory-impact"]
                else:
                    experiment_names = [experiment_name]

                # Extract the start date of the files required to be processed and account for the timezone offset
                start_date = os.path.basename(cpu_files[0]).split("-")[-1].split(".")[0] + "+01:00"
                mlflow_runs = mlflow.search_runs(
                    experiment_names=experiment_names,
                    order_by=["start_time ASC"],
                )
                run_ids = mlflow_runs.loc[mlflow_runs["start_time"] >= start_date]["run_id"].values

                # results = []
                # for i, (cpu_file, gpu_file) in enumerate(zip(cpu_files, gpu_files)):
                #     result = _process_raw_files(train_environment, architecture, dataset, run_ids[i], cpu_file, gpu_file)
                #     results.append(result.dropna(axis="columns", how="all"))
                #     new_cpu_files.append(cpu_file)
                #     new_gpu_files.append(gpu_file)
                # dataframe = pd.concat(
                #     results, axis=0, ignore_index=True
                # )
                # if df is None:
                #     df = dataframe
                # else:
                #     df = pd.concat([df, dataframe], axis=0, ignore_index=True)

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        _process_raw_files,
                        [train_environment] * len(cpu_files),
                        [architecture] * len(cpu_files),
                        [dataset] * len(gpu_files),
                        run_ids,
                        cpu_files,
                        gpu_files,
                    )
                    new_cpu_files += cpu_files
                    new_gpu_files += gpu_files
                    results = [
                        result.dropna(axis="columns", how="all") for result in results if not empty_dataframe(result)
                    ]
                    if not results:
                        continue

                    dataframe = pd.concat(results, axis=0, ignore_index=True)
                    if df.empty:
                        df = dataframe
                    else:
                        df = pd.concat([df, dataframe], axis=0, ignore_index=True)

    if df.empty:
        logger.info("No new files to process.")
        return df

    os.makedirs(METRICS_DIR / "interim", exist_ok=True)
    out_file = METRICS_DIR / "interim" / "dl-training-profiling-dataset.gzip"
    if (
        not process_all_files
        and os.path.exists(out_file)
        and (not cpu_files_processed.empty or not gpu_files_processed.empty)
    ):
        old_data = pd.read_parquet(out_file)
        df = pd.concat([old_data, df])

    df = df.sort_values(by=["train_environment", "architecture", "dataset", "run_id", "timestamp"])

    if save_to_file:
        df.to_parquet(out_file, index=False, compression="gzip")

        cpu_files_processed = pd.concat(
            [cpu_files_processed, pd.DataFrame({"file": new_cpu_files})], axis=0, ignore_index=True
        )
        gpu_files_processed = pd.concat(
            [gpu_files_processed, pd.DataFrame({"file": new_gpu_files})], axis=0, ignore_index=True
        )
        os.makedirs(METRICS_DIR / "auxiliary", exist_ok=True)
        cpu_files_processed.to_parquet(METRICS_DIR / "auxiliary" / "cpu_files_processed.gzip", compression="gzip")
        gpu_files_processed.to_parquet(METRICS_DIR / "auxiliary" / "gpu_files_processed.gzip", compression="gzip")

    return df


def empty_dataframe(dataframe: Union[pd.DataFrame, None]):
    return dataframe is None or dataframe.empty or dataframe.isna().all().all()


def _process_raw_files(train_environment, architecture, dataset, run_id, cpu_file, gpu_file):
    """
    Process the raw files for a given experiment.

    Parameters
    ----------
    train_environment : str
        The training environment.
    architecture : str
        The architecture used for the experiment.
    dataset : str
        The dataset used for the experiment.
    run_id : str
        The run identifier.
    cpu_file : str
        The file with the CPU measurements.
    gpu_file : str
        The file with the GPU measurements.

    Returns
    -------
    dataframe : `pd.DataFrame`
        A `pd.DataFrame` with the raw CPU and GPU measurements.
    """
    cpu_data = _process_raw_cpu_file(cpu_file)
    gpu_data = _process_raw_gpu_data(gpu_file)

    if cpu_data.empty and gpu_data.empty:
        return None

    if cpu_data.empty:
        df = gpu_data
        for column in cpu_data.columns:
            if column == "timestamp":
                continue
            df[column] = None
    elif gpu_data.empty:
        df = cpu_data
        for column in gpu_data.columns:
            if column == "timestamp":
                continue
            df[column] = None
    else:
        # Merge CPU and GPU data
        tol = pd.Timedelta("1 second")
        df = pd.merge_asof(gpu_data, cpu_data, on="timestamp", direction="nearest", tolerance=tol)
        df = df.dropna(axis="index", how="any", ignore_index=True)

    # Compute the elapsed time to facilitate the analysis
    df["elapsed_time"] = df["timestamp"] - df["timestamp"].min()

    df["train_environment"] = train_environment
    df["architecture"] = architecture
    df["dataset"] = dataset
    df["run_id"] = run_id
    cols = df.columns.tolist()
    df = df.loc[:, cols[-4:] + cols[:-4]]
    df["creation_time"] = gpu_file.split("-")[-1].split(".")[0]

    df = _add_epoch_number(df)
    return df


def _add_epoch_number(df):
    """
    Add the epoch number to the given dataframe and get rid of the measurements that are beyond the last epoch.

    Parameters
    ----------
    df : `pd.DataFrame`
        A `pd.DataFrame` with the data.

    Returns
    -------
    dataframe : `pd.DataFrame`
        A `pd.DataFrame` with the epoch number added for each row.
    """
    epoch_ends = get_epoch_ends(df.iloc[0])
    if epoch_ends is None:
        return df

    # Get rid of the measurements that are beyord the last epoch
    df.query("`timestamp` <= @epoch_ends.iloc[-1]['end_time']", inplace=True)

    # Add the epoch number to the dataframe
    df["epoch"] = 0
    for i, row in epoch_ends.iloc[1:].iterrows():
        mask = (df["timestamp"] > epoch_ends.iloc[i - 1]["end_time"]) & (df["timestamp"] <= row["end_time"])
        df.loc[mask, "epoch"] = row["epoch"]

    return df


def _process_raw_gpu_data(gpu_file):
    gpu_data = pd.read_csv(
        gpu_file,
        header=0,
        sep=",",
        engine="python",
        names=[
            "timestamp",
            "gpu_name",
            "gpu_usage",
            "gpu_memory_usage",
            "gpu_total_memory",
            "gpu_memory_used",
            "gpu_power_draw",
            "gpu_max_power",
            "gpu_temperature",
        ],
        dtype={"gpu_temperature": int},
        parse_dates=["timestamp"],
        converters={
            "gpu_usage": lambda x: measure2float(x) / 100,
            "gpu_memory_usage": lambda x: measure2float(x) / 100,
            "gpu_total_memory": measure2int,
            "gpu_memory_used": measure2int,
            "gpu_power_draw": measure2float,
            "gpu_max_power": measure2float,
        },
        na_values="None",
        skipinitialspace=True,
        skipfooter=3,  # remove last 3 measurements to avoid errors
    )
    gpu_data = gpu_data.sort_values(by="timestamp", ascending=True)
    return gpu_data


def _process_raw_cpu_file(cpu_file):
    cpu_data = pd.read_csv(
        cpu_file,
        header=0,
        names=["timestamp", "cpu_usage", "memory_usage", "cpu_temperature", "total_memory"],
        dtype={
            "cpu_usage": np.float32,
            "memory_usage": np.float32,
            "cpu_temperature": np.float32,
            "total_memory": np.int8,
        },
        parse_dates=["timestamp"],
        date_format="ISO8601",
        na_values="None",
    )
    cpu_data.drop(columns=["cpu_temperature"], inplace=True)
    cpu_data = cpu_data.sort_values(by="timestamp", ascending=True)
    cpu_data["memory_power_draw"] = cpu_data["memory_usage"] * USED_MB_TO_WATTS
    return cpu_data


def _get_processed_files():
    """
    Get the list of files that have already been processed.

    Returns
    -------
    cpu_files_processed : `pd.DataFrame`
        A `pd.DataFrame` with the list of files that have already been processed.
    gpu_files_processed : `pd.DataFrame`
        A `pd.DataFrame` with the list of files that have already been processed.
    """
    input_dir = METRICS_DIR / "auxiliary"
    cpu_file = input_dir / "cpu_files_processed.gzip"
    if os.path.exists(cpu_file):
        cpu_files_processed = pd.read_parquet(cpu_file)
    else:
        cpu_files_processed = pd.DataFrame({"file": []})

    gpu_file = input_dir / "gpu_files_processed.gzip"
    if os.path.exists(gpu_file):
        gpu_files_processed = pd.read_parquet(gpu_file)
    else:
        gpu_files_processed = pd.DataFrame({"file": []})
    return cpu_files_processed, gpu_files_processed


def get_duration(series):
    """
    Get the duration of a series of timestamps.

    Parameters
    ----------
    series : `pd.Series`
        A `pd.Series` with timestamps.

    Returns
    -------
    duration : `np.float64`
        The duration of the series in hours.
    """
    tmp = series.sort_values(ascending=True)
    return (tmp.iloc[-1] - tmp.iloc[0]) / np.timedelta64(1, "h")


def build_analysis_dataset(metrics_file: Union[Path, None] = None, save_to_file: bool = False):
    """
    Build the analysis dataset from the data in `input_folder`. If `output_folder` is specified it will save the dataset
    as a ``gzip`` file.

    Parameters
    ----------
    metrics_file : Path, default None
        The file with the collected metrics. If `None`, the default saving location will be used.
    save_to_file : bool, default False
        Whether to save the resulting dataset to a parquet file or not.

    Returns
    -------
    dataframe : `pd.DataFrame`
        A `pd.DataFrame` with the experiment metrics processed and ready to be analyzed.
    """
    # TODO: Skip executions that already exist
    if metrics_file is None:
        df = pd.read_parquet(METRICS_DIR / "interim" / "dl-training-profiling-dataset.gzip")
    else:
        df = pd.read_parquet(metrics_file)
    grouping_features = ["train_environment", "architecture", "dataset", "run_id"]
    df_grouped = _aggregate_metrics(df, grouping_features)
    setting_variables = [
        (train_environment, dataset, architecture)
        for train_environment, dataset, architecture in (
            df[["train_environment", "dataset", "architecture"]].drop_duplicates().values
        )
    ]
    output_folder = METRICS_DIR / "processed"
    mlflow_df = _get_mlflow_metrics(setting_variables, grouping_features, output_folder)
    analysis_df = df_grouped.join(mlflow_df, how="inner")
    analysis_df = analysis_df.reset_index()
    analysis_df.rename(columns={"train_environment": "training environment"}, inplace=True)
    analysis_df.loc[
        (analysis_df["training environment"] == "local") & (analysis_df["gpu model"] == "NVIDIA GeForce RTX 3070"),
        "training environment",
    ] = "local-v2"
    analysis_df.replace(
        {
            "local": "Local Normal User",
            "local-v2": "Local ML Engineer",
            "cloud": "Cloud",
        },
        inplace=True,
    )

    # Compute new metrics
    training_size = analysis_df["training size"]
    validation_size = analysis_df["validation size"]
    batch_size = analysis_df["batch size"]
    measured_epochs = analysis_df["measured epochs"]
    analysis_df["total seen images"] = (
        training_size - (training_size % batch_size) + validation_size - (validation_size % batch_size)
    ) * measured_epochs

    oom_flags = _get_oom_metrics()
    oom_flags.sort_values(by=["start time"], inplace=True)
    analysis_df.sort_values(by=["start time"], inplace=True)
    analysis_df = pd.merge_asof(
        analysis_df,
        oom_flags,
        left_on="start time",
        right_on="start time",
        by=["architecture", "dataset", "batch size", "image size"],
        direction="backward",
        tolerance=pd.Timedelta("1 minute"),
    )
    analysis_df.fillna({"return code": 0}, inplace=True)
    analysis_df.loc[analysis_df["measured epochs"] == 0, "return code"] = -9

    # Ensure the rows are sorted by the start time
    analysis_df.sort_values(by=["start time"], inplace=True)

    if save_to_file:
        out_file = output_folder / "dl-training-energy-consumption-dataset.gzip"
        analysis_df.to_parquet(out_file, index=False, compression="gzip")

    return analysis_df


def _get_mlflow_metrics(training_configuration: List, grouping_features: List, output_folder: Path):
    """
    Get the metrics from the MLflow runs.

    Parameters
    ----------
    training_configuration : `list`
        A list with the training configuration.
    grouping_features : `list`
        A list with the features used to group the data.
    output_folder : `Path`, default None
        The folder where the dataset will be saved. If `None` the data will not be saved.

    Returns
    -------
    mlflow_df : `pd.DataFrame`
        A `pd.DataFrame` with the metrics from the MLflow runs.
    """
    mlflow_experiments = []
    for training_environment, dataset, architecture in training_configuration:
        experiment_name = f"{training_environment}-{dataset}-{architecture}"
        if experiment_name == "cloud-stanford_dogs-inception_v3":
            experiment_names = [experiment_name, "memory-impact"]
        else:
            experiment_names = [experiment_name]
        mlflow_runs = mlflow.search_runs(
            experiment_names=experiment_names,
            order_by=["start_time ASC"],
        )

        if dataset == "chesslive-occupancy":
            metrics = [
                "params.split_number",
                "metrics.val_binary_accuracy",
                "metrics.val_precision",
                "metrics.val_recall",
                "metrics.val_auc",
            ]
        else:
            metrics = [
                "metrics.val_accuracy",
                "metrics.val_auc",
            ]

        # Select relevant metrics
        relevant_metrics = mlflow_runs.loc[
            :,
            [
                "run_id",
                "params.train_size",
                "params.validation_size",
                "params.batch_size",
                "params.image_size",
                "metrics.MACCS",
            ]
            + metrics,
        ]

        # Turn count of multiply accumulate operations into count of FLOPs
        relevant_metrics.loc[:, "metrics.MACCS"] = relevant_metrics["metrics.MACCS"].apply(
            lambda x: x * 2 * FLOPS_TO_GFLOPS
        )

        relevant_metrics.rename(
            columns={
                "params.split_number": "split number",
                "params.train_size": "training size",
                "params.validation_size": "validation size",
                "params.batch_size": "batch size",
                "params.image_size": "image size",
                "metrics.MACCS": "GFLOPs",
                "metrics.val_binary_accuracy": "accuracy",
                "metrics.val_accuracy": "accuracy",
                "metrics.val_precision": "precision",
                "metrics.val_recall": "recall",
                "metrics.val_auc": "AUC",
            },
            inplace=True,
        )

        # Change training parameters from string to their numberic type
        relevant_metrics["training size"] = pd.to_numeric(
            relevant_metrics.loc[:, "training size"].fillna(0), errors="raise"
        ).astype(int)
        relevant_metrics["validation size"] = pd.to_numeric(
            relevant_metrics.loc[:, "validation size"].fillna(0), errors="raise"
        ).astype(int)
        relevant_metrics["batch size"] = relevant_metrics["batch size"].astype(int)
        relevant_metrics["split number"] = relevant_metrics.get(["split number"], 0)
        relevant_metrics["split number"] = relevant_metrics.loc[:, "split number"].astype(int)

        if dataset == "chesslive-occupancy":
            relevant_metrics["f1-score"] = (
                (2 * relevant_metrics.precision * relevant_metrics.recall)
                / (relevant_metrics.precision + relevant_metrics.recall)
            ).fillna(0)

        relevant_metrics["train_environment"] = training_environment
        relevant_metrics["architecture"] = architecture
        relevant_metrics["dataset"] = dataset
        mlflow_experiments.append(relevant_metrics)

    mlflow_df = pd.concat(mlflow_experiments, axis=0)
    mlflow_df = mlflow_df.set_index(grouping_features)
    if output_folder is not None:
        out_folder = os.path.join(METRICS_DIR, "interim")
        os.makedirs(out_folder, exist_ok=True)
        out_file = os.path.join(out_folder, "model_metrics.gzip")
        mlflow_df.to_parquet(out_file, compression="gzip")
    return mlflow_df


def _aggregate_metrics(df: pd.DataFrame, grouping_features: List):
    """
    Aggregate the metrics from the collected data.

    Parameters
    ----------
    df : `pd.DataFrame`
        A `pd.DataFrame` with the collected data.
    grouping_features : `list`
        A list with the features used to group the data.

    Returns
    -------
    df_grouped : `pd.DataFrame`
        A `pd.DataFrame` with the aggregated metrics.
    """
    df_grouped = df.groupby(grouping_features).agg(
        train_start=("timestamp", "first"),
        hours_training=("timestamp", get_duration),
        measured_epochs=("epoch", "max"),
        gpu_name=("gpu_name", "first"),
        gpu_working_time=("gpu_usage", lambda x: np.sum(x) * SECONDS_TO_HOURS),
        gpu_usage=("gpu_usage", lambda x: np.mean(x) * 100),
        gpu_memory_working_time=("gpu_memory_usage", lambda x: np.sum(x) * SECONDS_TO_HOURS),
        gpu_memory_usage=("gpu_memory_usage", "mean"),
        gpu_memory_used_avg=("gpu_memory_used", "mean"),
        gpu_memory_used_std=("gpu_memory_used", "std"),
        W=("gpu_power_draw", "sum"),
        W_avg=("gpu_power_draw", "mean"),
        W_std=("gpu_power_draw", "std"),
        gpu_max_power=("gpu_max_power", "first"),
        temperature_avg=("gpu_temperature", "mean"),
        temperature_std=("gpu_temperature", "std"),
        cpu_usage_avg=("cpu_usage", "mean"),
        cpu_usage_std=("cpu_usage", "std"),
        ram_used_avg=("memory_usage", "mean"),
        ram_used_std=("memory_usage", "std"),
        ram_W_avg=("memory_power_draw", "mean"),
        ram_W_std=("memory_power_draw", "std"),
        total_ram=("total_memory", "first"),
    )

    df_grouped.measured_epochs = df_grouped.measured_epochs.fillna(0).astype(int)
    df["elapsed_time"] = df["elapsed_time"].dt.total_seconds()
    df_grouped["trained epochs"] = df.groupby(grouping_features).apply(
        lambda d: get_epoch_ends(d.iloc[0]).shape[0] if not d.epoch.isna().any() else 0
    )

    df_grouped["gpu energy (MJ)"] = df.groupby(grouping_features).apply(
        lambda d: np.trapz(y=d.gpu_power_draw, x=d.elapsed_time) * JOULES_TO_MJOULES
    )
    df_grouped["ram energy (MJ)"] = df.groupby(grouping_features).apply(
        lambda d: np.trapz(y=d.memory_power_draw, x=d.elapsed_time) * JOULES_TO_MJOULES
    )
    df_grouped["energy (MJ)"] = df_grouped["gpu energy (MJ)"] + df_grouped["ram energy (MJ)"]
    # df_grouped["energy (GJ)"] = df_grouped.W * (df_grouped.hours_training * HOURS_TO_SECONDS) * JOULES_TO_GJOULES
    df_grouped["emissions (gCO2e)"] = df_grouped["energy (MJ)"] * MJOULES_TO_JOULES * JOULES_TO_KWH * KWH_CO2e_SPA

    df_grouped.rename(
        columns={
            "train_start": "start time",
            "hours_training": "training duration (h)",
            "measured_epochs": "measured epochs",
            "gpu_name": "gpu model",
            "gpu_working_time": "gpu working time (h)",
            "gpu_usage": "gpu usage (%)",
            "gpu_memory_working_time": "gpu memory working time (h)",
            "gpu_memory_usage": "gpu memory usage",
            "gpu_memory_used_avg": "gpu average memory used (MB)",
            "gpu_memory_used_std": "gpu memory used std (MB)",
            "W": "total power (W)",
            "W_avg": "average gpu power (W)",
            "W_std": "gpu power std (W)",
            "gpu_max_power": "max power limit (W)",
            "temperature_avg": "average temperature (Celsius)",
            "temperature_std": "temperature std (Celsius)",
            "cpu_usage_avg": "average cpu usage (%)",
            "cpu_usage_std": "cpu usage std (%)",
            "ram_used_avg": "average ram used (MB)",
            "ram_used_std": "ram used std (MB)",
            "ram_W_avg": "average ram power (W)",
            "ram_W_std": "ram power std (W)",
            "total_ram": "total ram (GB)",
        },
        inplace=True,
    )

    return df_grouped


def _get_oom_metrics():
    oom_flags = []
    for file in glob.glob(str(OUTPUT_DIR / "oom-experiments*.csv")):
        df = pd.read_csv(
            file,
            header=0,
            names=["start time", "architecture", "dataset", "image size", "batch size", "return code"],
            parse_dates=["start time"],
            dtype={"architecture": str, "dataset": str, "image size": str, "batch size": int, "return code": int},
        )
        df.loc[:, "image size"] = df["image size"].apply(lambda x: f"({x},{x})")
        oom_flags.append(df)

    return pd.concat(oom_flags)


def build_epoch_energy_dataset(save_to_file: bool = False):
    """
    Build the dataset with the energy consumption per epoch.

    Parameters
    ----------
    save_to_file : bool, default False
        Whether to save the resulting dataset to a parquet file or not.

    Returns
    -------
    df : `pd.DataFrame`
        A `pd.DataFrame` with the energy consumption per epoch.
    """
    # TODO: Skip executions that already exist
    df = pd.read_parquet(
        METRICS_DIR / "interim" / "dl-training-profiling-dataset.gzip",
        columns=[
            "train_environment",
            "architecture",
            "dataset",
            "run_id",
            "epoch",
            "timestamp",
            "total_memory",
            "gpu_power_draw",
            "memory_power_draw",
            "elapsed_time",
            "creation_time",
        ],
    )

    # Duplicate the last measurement of each epoch and add it to the next epoch in order to avoid loosing information
    # about energy and power consumption due to measurement frequency.
    last_measurement_per_epoch = df.groupby(
        ["train_environment", "architecture", "dataset", "run_id", "epoch"], as_index=False
    ).agg(
        timestamp=("timestamp", "max"),
    )

    # Remove the last epoch of each run since it does not need to be duplicated
    last_measurement_per_epoch = last_measurement_per_epoch.groupby(
        ["train_environment", "architecture", "dataset", "run_id"], as_index=False
    ).apply(lambda d: d.iloc[:-1])

    last_measurement_per_epoch = df.query("`timestamp` in @last_measurement_per_epoch['timestamp']")
    last_measurement_per_epoch.loc[:, "epoch"] += 1
    tmp = pd.concat([df, last_measurement_per_epoch], axis=0, ignore_index=True)
    # Sort the data to facilitate the computation of the energy consumption
    tmp = tmp.sort_values(
        by=["train_environment", "architecture", "dataset", "run_id", "timestamp"], ascending=True
    ).reset_index(drop=True)

    # Compute simple aggregation metrics
    df_grouped = tmp.groupby(["train_environment", "architecture", "dataset", "run_id", "epoch"]).agg(
        creation_time=("creation_time", "first"),
        total_memory=("total_memory", "first"),
        mean_gpu_power=("gpu_power_draw", "mean"),
        mean_ram_power=("memory_power_draw", "mean"),
    )

    # Compute the energy consumption
    tmp["elapsed_time"] = tmp["elapsed_time"].dt.total_seconds()

    df_grouped["gpu energy (kJ)"] = tmp.groupby(
        ["train_environment", "architecture", "dataset", "run_id", "epoch"]
    ).apply(lambda d: np.trapz(y=d.gpu_power_draw, x=d.elapsed_time) * JOULES_TO_KJOULES)
    df_grouped["ram energy (kJ)"] = tmp.groupby(
        ["train_environment", "architecture", "dataset", "run_id", "epoch"]
    ).apply(lambda d: np.trapz(y=d.memory_power_draw, x=d.elapsed_time) * JOULES_TO_KJOULES)
    df_grouped["total energy (kJ)"] = df_grouped["gpu energy (kJ)"] + df_grouped["ram energy (kJ)"]
    df_grouped.reset_index(inplace=True)

    # Compute the duration of each epoch using the timestamps collected during the training
    runs = df.loc[df.groupby(["train_environment", "architecture", "dataset", "run_id"])["timestamp"].idxmin()]
    epoch_data = []
    for _, run in runs.iterrows():
        epoch_ends = get_epoch_ends(run)
        if epoch_ends is None:
            continue
        initial_time = run.timestamp

        timestamps = pd.concat([pd.Series(initial_time), epoch_ends["end_time"]])
        timestamps = timestamps.sort_values()
        timestamps = timestamps.reset_index(drop=True)

        durations = pd.DataFrame(
            {"epoch": epoch_ends["epoch"].values, "duration (s)": timestamps.diff().dt.total_seconds().iloc[1:]}
        )
        durations["train_environment"] = run.train_environment
        durations["architecture"] = run.architecture
        durations["dataset"] = run.dataset
        durations["run_id"] = run.run_id
        epoch_data.append(durations)

    epoch_data = pd.concat(epoch_data, axis=0)
    df_grouped = pd.merge(
        df_grouped, epoch_data, on=["train_environment", "architecture", "dataset", "run_id", "epoch"]
    )

    df_grouped.rename(
        {
            "total_memory": "total ram (GB)",
            "start_time": "start time",
            "end_time": "end time",
            "mean_gpu_power": "mean gpu power (W)",
            "gpu_energy": "gpu energy (kJ)",
            "mean_ram_power": "mean ram power (W)",
            "ram_energy": "ram energy (kJ)",
        },
        inplace=True,
        axis=1,
    )

    run_settings = pd.read_parquet(
        METRICS_DIR / "processed" / "dl-training-energy-consumption-dataset.gzip",
        columns=["training environment", "run_id", "batch size", "image size"],
    )

    df_grouped = pd.merge(df_grouped, run_settings, on="run_id", how="inner")
    df_grouped.drop(columns=["train_environment"], inplace=True)
    df_grouped = df_grouped.loc[
        :,
        [
            "training environment",
            "architecture",
            "dataset",
            "run_id",
            "epoch",
            "duration (s)",
            "batch size",
            "image size",
            "total ram (GB)",
            "mean gpu power (W)",
            "gpu energy (kJ)",
            "mean ram power (W)",
            "ram energy (kJ)",
            "total energy (kJ)",
        ],
    ]
    df_grouped = df_grouped.sort_values(by=["run_id", "epoch"]).reset_index(drop=True)

    output_folder = METRICS_DIR / "interim"
    if save_to_file:
        out_file = output_folder / "dl-epoch-energy-consumption-dataset.gzip"
        df_grouped.to_parquet(out_file, index=False, compression="gzip")

    return df_grouped

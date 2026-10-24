"""
Data loading and splitting shared by the pre-training regression-based estimation (PRE) models.

All features are known before training starts. The runs are split at random into a training and a test set.
"""

import re

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.gpu_specs import get_gpu_specs
from src.environment import METRICS_DIR, SEED
from src.features.preprocessing import HOURS_TO_SECONDS, MJOULES_TO_JOULES, MJOULES_TO_KJOULES

FEATURES = [
    "batch size",
    "total ram (GB)",
    "training size",
    "validation size",
    "GFLOPs",
    "image height",
    "image width",
    "gpu cuda cores",
    "gpu total memory",
    "gpu max power",
]

STABILIZING_EPOCH = 10
TEST_SIZE = 0.30


def load_pre_dataset() -> pd.DataFrame:
    """
    Load one row per training run with the PRE features, the targets of each PRE model and the measured energy.

    Returns
    -------
    data : `pd.DataFrame`
        The PRE dataset.
    """
    data = pd.read_parquet(
        METRICS_DIR / "processed" / "clean-dl-training-energy-consumption-dataset.gzip",
        columns=[
            "run_id",
            "batch size",
            "image size",
            "gpu model",
            "total ram (GB)",
            "training size",
            "validation size",
            "training duration (h)",
            "GFLOPs",
            "measured epochs",
            "energy (MJ)",
        ],
    )
    epoch_data = pd.read_parquet(
        METRICS_DIR / "processed" / "clean-dl-epoch-energy-consumption-dataset.gzip",
        columns=["run_id", "epoch", "total energy (kJ)"],
    )

    data["image height"] = data["image size"].apply(lambda x: int(re.findall(r"\d+", x)[0]))
    data["image width"] = data["image size"].apply(lambda x: int(re.findall(r"\d+", x)[1]))

    specs = {model: get_gpu_specs(model.replace("NVIDIA GeForce ", "")) for model in data["gpu model"].unique()}
    data["gpu cuda cores"] = data["gpu model"].map(lambda m: specs[m].cuda_cores)
    data["gpu total memory"] = data["gpu model"].map(lambda m: specs[m].memory)
    data["gpu max power"] = data["gpu model"].map(lambda m: specs[m].power)

    # Targets. Power and energy include both the GPU and the RAM, as the measured energy does.
    training_seconds = data["training duration (h)"] * HOURS_TO_SECONDS
    data["average power (W)"] = data["energy (MJ)"] * MJOULES_TO_JOULES / training_seconds
    data["energy (kJ)"] = data["energy (MJ)"] * MJOULES_TO_KJOULES

    epoch_energy = epoch_data.groupby("run_id")["total energy (kJ)"]
    stable_epoch_energy = epoch_data.query("epoch >= @STABILIZING_EPOCH").groupby("run_id")["total energy (kJ)"]
    data["average epoch energy (kJ)"] = data["run_id"].map(epoch_energy.mean())
    data["average stable epoch energy (kJ)"] = data["run_id"].map(stable_epoch_energy.mean())

    # Order the runs by GPU model, as the original training scripts did, so that the random split is the same
    data = pd.concat([group for _, group in data.groupby("gpu model")])
    return data.reset_index(drop=True)


def split_runs(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the PRE dataset at random into a training set and a test set with `TEST_SIZE` of the runs.

    Parameters
    ----------
    data : `pd.DataFrame`
        The dataset returned by `load_pre_dataset`.

    Returns
    -------
    train, test : tuple of `pd.DataFrame`
        The training and test sets.
    """
    return train_test_split(data, test_size=TEST_SIZE, random_state=SEED)

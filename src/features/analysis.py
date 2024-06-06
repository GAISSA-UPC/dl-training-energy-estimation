import os
import pickle
from typing import Tuple

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import stumpy
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
from tqdm import tqdm

from src.environment import DATA_DIR, FIGURES_DIR, METRICS_DIR
from src.features.preprocessing import (
    HOURS_TO_SECONDS,
    JOULES_TO_KJOULES,
    KJOULES_TO_JOULES,
    MJOULES_TO_KJOULES,
    get_duration,
    get_epoch_ends,
)

ALPHA = 0.05


def test_normality(x, group=None, ax: matplotlib.axes.Axes = None, figsize: Tuple[int, int] = (5, 5)):
    """
    Test the normality of a group of samples.

    Parameters
    ----------
    x : array_like
        Array of sample data.
    group : str, optional
        Nambe of the group to test.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot the Q-Q plot.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    result : tuple
        Shapiro-Wilk test statistic and p-value.
    """
    result = stats.shapiro(x)
    if group is None:
        print(
            f"Shapiro test for normality: W = {result[0]} and p-value {result[1]}. Is normaly distributed? {result[1] > ALPHA}"
        )
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        sm.qqplot(x, line="q", ax=ax)
    else:
        print(
            f"Shapiro test for normality of group {group}: W = {result[0]} and p-value {result[1]}. Is normaly distributed? {result[1] > ALPHA}"
        )
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        sm.qqplot(x, line="q", ax=ax)
        ax.set_title(f"Q-Q plot for group {group}")
    return result


def test_assumptions(*args, nrows=1, ncols=1, figsize: Tuple[int, int] = (5, 5)):
    """
    Test the assumptions of normality and equal variance for different groups of samples.

    The Shapiro-Wilk test is used to test for normality and the Levene test is used to test for equal variance.
    Their results are printed to the console and a Q-Q plot is generated for each group of samples.

    If only one group of samples is provided, only the normality test is performed.

    Parameters
    ----------
    args : array_like
        Groups of sample data.
    nrows : int, optional
        Number of rows in the figure.
    ncols : int, optional
        Number of columns in the figure.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    shapiro_results : list
        Shapiro-Wilk test statistic and p-value for each group.
    levene_results : tuple
        Levene test statistic and p-value.
    """
    if len(args) < 2:
        return test_normality(np.asarray(args[0]), figsize=figsize), None
    args = [np.asarray(arg) for arg in args]
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()
    shapiro_results = []
    for i, arg in enumerate(args):
        if arg.ndim != 1:
            raise ValueError("Input samples must be one-dimensional.")
        if arg.size <= 1:
            raise ValueError("Input sample size must be greater than one.")
        if np.isinf(arg).any():
            raise ValueError("Input samples must be finite.")
        shapiro_results.append(test_normality(arg, i, axes[i]))

    levene_results = stats.levene(*args)
    print(
        f"Levene test for equal variances: W = {levene_results[0]} and p-value = {levene_results[1]}. Equal variances? {levene_results[1] > ALPHA}"
    )

    return shapiro_results, levene_results


def eta_squared(H, k, n):
    """
    Compute the eta-squared measure for the Kruskal-Wallis H-test.

    Parameters
    ----------
    H : float
        The value obtained in the Kruskal-Wallis test.
    k : int
        The number of groups.
    n : int
        The total number of samples.

    Returns
    -------
    eta_squared : float
        The eta-squared estimate.
    """
    return (H - k + 1) / (n - k)


def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points.

    Parameters
    ----------
    costs : array_like
        An (n_points, n_costs) array
    return_mask : bool, optional
        True to return a mask.

    Returns
    -------
    is_efficient : array_like
        An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array.
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = (costs[:, 0] <= costs[next_point_index, 0]) | (
            costs[:, 1] >= costs[next_point_index, 1]
        )
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def boxplot(
    data,
    x: str = None,
    y: str = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figname: str = None,
    figsize: tuple[int, int] = (5, 5),
):
    """
    Create a boxplot using seaborn.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing the data to plot.
    x : str, optional
        Column name to group the data by.
    y : str, optional
        Column name to plot.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label of the x-axis.
    ylabel : str, optional
        Label of the y-axis.
    figname : str, optional
        Name of the figure file to save.
    figsize : tuple, optional
        Figure size.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    sns.boxplot(data, x=x, y=y, ax=ax)
    ax.yaxis.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if figname is not None:
        plt.savefig(os.path.join(FIGURES_DIR, figname))


def barplot(
    data,
    x=None,
    y=None,
    xlabel=None,
    ylabel=None,
    title=None,
    hue=None,
    hue_order=None,
    errorbar=None,
    estimator="mean",
    figname=None,
    figsize=(5, 5),
    barlabel=False,
    ax=None,
):
    """
    Create a barplot using seaborn.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing the data to plot.
    x : str, optional
        Column name to group the data by.
    y : str, optional
        Column name to plot.
    xlabel : str, optional
        Label of the x-axis.
    ylabel : str, optional
        Label of the y-axis.
    title : str, optional
        Title of the plot.
    hue : str, optional
        Column name to group the data by.
    hue_order : list, optional
        Order of the hue levels.
    errorbar : str, optional
        Name of errorbar method (either “ci”, “pi”, “se”, or “sd”), or a tuple with a method name and a level parameter,
        or a function that maps from a vector to a (min, max) interval, or None to hide errorbar.
    estimator : str, optional
        Function to aggregate the data.
    figname : str, optional
        Name of the figure file to save.
    figsize : tuple, optional
        Figure size.
    barlabel : bool, optional
        True to add labels to the bars.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot the data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    if barlabel and errorbar is not None:
        raise Warning(
            "Setting using the errorbar and barlabel parameters will produce overlapping labels. Please use one or the other."
        )

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    ax = sns.barplot(
        data,
        x=x,
        y=y,
        ax=ax,
        hue=hue,
        hue_order=hue_order,
        errorbar=errorbar,
        # err_kws={"linewidth": 3},
        estimator=estimator,
    )
    ax.yaxis.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if barlabel:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f")

    if figname is not None:
        plt.savefig(os.path.join(FIGURES_DIR, figname.split(".")[-1], figname))
    return fig, ax


def print_improvement(dataframe, metric):
    median_values = dataframe.groupby(["architecture", "training environment"], as_index=False)[metric].median()
    median_values["combination"] = median_values.architecture + " - " + median_values["training environment"]
    tmp = pd.DataFrame(columns=median_values.combination.unique(), index=median_values.combination.unique())

    # Iterate over all combinations of architectures and training environments and calculate the relative difference in energy consumption. Then get the maximum and minimum values.
    for i, row in median_values.iterrows():
        for j, row2 in median_values.iterrows():
            if i != j:
                tmp.loc[row.combination, row2.combination] = (row[metric] - row2[metric]) / row[metric]
            else:
                tmp.loc[row.combination, row2.combination] = 0.0

    tmp = tmp.astype(float)

    # Get the rows and columns with maximum and minimum values of the relative difference in energy consumption.
    max_row = tmp.max(axis=1).idxmax()
    max_col = tmp.max(axis=0).idxmax()
    min_row = tmp[tmp > 0].min(axis=1).idxmin()
    min_col = tmp[tmp > 0].min(axis=0).idxmin()

    print(f"Maximum improvement: {tmp.loc[max_row, max_col]} ({max_row}, {max_col})")
    print(f"Minimum improvement: {tmp.loc[min_row, min_col]} ({min_row}, {min_col})")


def plot_gpu_power(run, epoch_ends):
    plt.figure(figsize=(15, 5))
    sns.lineplot(data=run, x="timestamp", y="gpu_power_draw")
    max_power = run["gpu_power_draw"].max()
    if epoch_ends is not None:
        for _, row in epoch_ends.iterrows():
            plt.axvline(row["end_time"], ymax=max_power, color="red", linestyle="--")


def plot_regime_change(run, epoch_ends, cac, regime_change, breaking_epoch):
    power = run.sort_values(by="timestamp")["gpu_power_draw"]
    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"hspace": 0})
    fig.suptitle(
        f"Power profile for run in {run['train_environment'].iloc[0]} with {run['architecture'].iloc[0]} on {run['dataset'].iloc[0]}"
    )

    axs[0].plot(run["timestamp"], power, color="red")
    max_power = power.max()
    for i, row in epoch_ends.iterrows():
        if i == breaking_epoch:
            if i == 0:
                width = row["end_time"] - run["timestamp"].min()
            else:
                width = row["end_time"] - epoch_ends.iloc[i - 1]["end_time"]
            rect = Rectangle((row["end_time"], 0), -width, max_power + 10, facecolor="green", alpha=0.2)
            axs[0].add_patch(rect)
            axs[0].axvline(row["end_time"], ymax=max_power, color="green", linestyle="--")
        else:
            axs[0].axvline(row["end_time"], ymax=max_power, color="grey", linestyle="--")

    axs[1].plot(run.iloc[: cac.shape[0]]["timestamp"], cac, color="orange")
    axs[1].axvline(x=run.iloc[regime_change].timestamp, linestyle="dashed", color="blue")


def find_stabilizing_point(runs, metrics: pd.DataFrame, m: int, L: int, save: bool = False):
    """
    Find the power consumption stabilizing point for a set of runs.

    Parameters
    ----------
    runs : array_like
        Array of run IDs.
    metrics : pandas.DataFrame
        Dataframe containing the metrics for all runs.
    m : int
        Window size.
    L : int
        The subsequence length that is set roughly to be one period length.
    save : bool, optional
        True to save the results to disk.

    Returns
    -------
    regimes_df : pandas.DataFrame
        Dataframe containing the run ID, elapsed time and stabilizing epoch for each run.
    profiles : dict
        Dictionary containing the matrix profile, cac and regime locations for each run.
    """
    regimes = np.zeros((runs.shape[0], 3), dtype=object)
    profiles = {}
    for i, (run_id, run) in tqdm(
        enumerate(metrics.loc[metrics["run_id"].isin(runs)].groupby("run_id")), total=len(runs)
    ):
        sorted_run = run.sort_values(by="timestamp")
        mp = stumpy.stump(sorted_run.gpu_power_draw.values, m)
        cac, regime_locations = stumpy.fluss(mp[:, 1], L, n_regimes=2, excl_factor=3)
        profiles[run_id] = {"mp": mp, "cac": cac, "regime_locations": regime_locations}
        row = run.iloc[0]
        epoch_ends = get_epoch_ends(row)
        stabilizing_point = sorted_run.iloc[regime_locations[0]].timestamp
        stabilizing_epoch = 0
        if epoch_ends is not None:
            for index, row in epoch_ends.iterrows():
                if row["end_time"] > stabilizing_point:
                    stabilizing_epoch = index
                    break
        else:
            stabilizing_epoch = -1
        elapsed_time = run.iloc[regime_locations[0]].elapsed_time
        regimes[i] = [run_id, elapsed_time, stabilizing_epoch]

    regimes_df = pd.DataFrame(regimes, columns=["run_id", "elapsed time", "stabilizing epoch"])
    regimes_df["elapsed time"] = pd.to_numeric(regimes_df["elapsed time"])
    if save:
        regimes_df.to_parquet(DATA_DIR / "analysis" / "processed" / "regimes.gzip", compression="gzip")
        with open(DATA_DIR / "analysis" / "processed" / "cloud_inception_mps.pkl", "wb") as f:
            pickle.dump(profiles, f)
    return regimes_df, profiles


def build_energy_estimation(mean_power_draw, stabilizing_epoch):
    aggregated_metrics = pd.read_parquet(
        METRICS_DIR / "processed" / "clean-dl-training-energy-consumption-dataset.gzip",
        columns=[
            "training environment",
            "architecture",
            "dataset",
            "batch size",
            "image size",
            "total ram (GB)",
            "run_id",
            "start time",
            "training duration (h)",
            "return code",
            "gpu usage (%)",
            "average gpu power (W)",
            "gpu energy (MJ)",
            "max power limit (W)",
            "average ram power (W)",
            "ram energy (MJ)",
            "energy (MJ)",
            "GFLOPs",
            "trained epochs",
            "measured epochs",
        ],
    ).sort_values(by=["start time"])

    metrics = pd.read_parquet(os.path.join(METRICS_DIR, "interim", "dl-training-profiling-dataset.gzip"))
    metrics.query("`run_id` in @aggregated_metrics['run_id'].values", inplace=True)
    metrics["elapsed_time"] = metrics["elapsed_time"] / np.timedelta64(1, "s")
    metrics["epoch"] = metrics["epoch"].astype("int")

    epoch_energy_df = pd.read_parquet(METRICS_DIR / "processed" / "clean-dl-epoch-energy-consumption-dataset.gzip")
    epoch_energy_df["epoch"] = epoch_energy_df["epoch"].astype("int")

    energy_estimation = pd.DataFrame.from_dict(mean_power_draw, orient="columns")
    stable_epochs = metrics.query("`epoch` >= @stabilizing_epoch")
    stable_epochs_energy = epoch_energy_df.query("`epoch` >= @stabilizing_epoch")

    grouping_features = ["run_id"]
    stable_energy = stable_epochs.groupby(grouping_features).agg(
        train_environment=("train_environment", "first"),
        architecture=("architecture", "first"),
        dataset=("dataset", "first"),
        training_duration=("timestamp", get_duration),
        mean_gpu_power=("gpu_power_draw", "mean"),
        gpu_max_power=("gpu_max_power", "first"),
        gpu_usage=("gpu_usage", lambda x: np.mean(x)),
        mean_ram_power=("memory_power_draw", "mean"),
        n_epochs=("epoch", "nunique"),
    )

    stable_energy["initial energy (J)"] = (
        epoch_energy_df.query("`epoch` < @stabilizing_epoch").groupby(grouping_features)["total energy (kJ)"].sum()
        * KJOULES_TO_JOULES
    )
    stable_energy["total epochs"] = metrics.groupby("run_id")["epoch"].nunique()
    stable_energy["energy (kJ)"] = stable_epochs_energy.groupby(grouping_features)["total energy (kJ)"].sum()

    stable_energy.reset_index(inplace=True)
    stable_energy.rename(
        columns={
            "train_environment": "training environment",
            "training_duration": "training duration (h)",
            "mean_gpu_power": "mean gpu power (W)",
            "gpu_max_power": "max power limit (W)",
            "gpu_usage": "gpu usage (%)",
            "mean_ram_power": "mean ram power (W)",
        },
        inplace=True,
    )

    energy_estimation = energy_estimation.merge(stable_energy, on="run_id", how="inner")
    energy_estimation = energy_estimation.merge(
        aggregated_metrics[
            [
                "run_id",
                "training duration (h)",
                "average gpu power (W)",
                "max power limit (W)",
                "gpu usage (%)",
                "average ram power (W)",
                "energy (MJ)",
                "total ram (GB)",
            ]
        ].rename(
            {
                "training duration (h)": "total training duration (h)",
                "average gpu power (W)": "total average gpu power (W)",
                "max power limit (W)": "total max power limit (W)",
                "gpu usage (%)": "total gpu usage (%)",
                "average ram power (W)": "total average ram power (W)",
                "energy (MJ)": "total energy (MJ)",
            },
            axis=1,
        ),
        on="run_id",
        how="inner",
    )
    energy_estimation["total energy (kJ)"] = energy_estimation["total energy (MJ)"] * MJOULES_TO_KJOULES
    energy_estimation["total gpu usage (%)"] = energy_estimation["total gpu usage (%)"] / 100

    # energy = window average power * training duration
    energy_estimation["estimated energy (kJ) (online power-based)"] = (
        (energy_estimation["mean gpu power draw"].fillna(0) + energy_estimation["mean ram power draw"].fillna(0))
        * energy_estimation["training duration (h)"]
        * HOURS_TO_SECONDS
        * JOULES_TO_KJOULES
    )
    energy_estimation["estimated total energy (kJ) (online power-based)"] = (
        (energy_estimation["mean gpu power draw"].fillna(0) + energy_estimation["mean ram power draw"].fillna(0))
        * energy_estimation["total training duration (h)"]
        * HOURS_TO_SECONDS
        * JOULES_TO_KJOULES
    )

    # energy = window total energy * #epochs/window size + unstable epochs energy
    energy_estimation["estimated energy (kJ) (online epoch-energy-based)"] = (
        energy_estimation["energy (J)"]
        * (energy_estimation["n_epochs"] / energy_estimation["window size"])
        * JOULES_TO_KJOULES
    )
    energy_estimation["estimated total energy (kJ) (online epoch-energy-based)"] = (
        energy_estimation["energy (J)"] * ((energy_estimation["n_epochs"]) / energy_estimation["window size"])
        + energy_estimation["initial energy (J)"]
    ) * JOULES_TO_KJOULES

    # energy = (TDP * gpu usage + (total ram * C)) * training duration
    energy_estimation["estimated energy (kJ) (GA)"] = (
        (
            energy_estimation["max power limit (W)"] * energy_estimation["gpu usage (%)"]
            + energy_estimation["total ram (GB)"] * 0.3725
        )
        * energy_estimation["training duration (h)"]
        * HOURS_TO_SECONDS
        * JOULES_TO_KJOULES
    )
    energy_estimation["estimated total energy (kJ) (GA)"] = (
        (
            energy_estimation["max power limit (W)"] * energy_estimation["total gpu usage (%)"]
            + energy_estimation["total ram (GB)"] * 0.3725
        )
        * energy_estimation["total training duration (h)"]
        * HOURS_TO_SECONDS
        * JOULES_TO_KJOULES
    )

    # energy = TDP * training duration
    energy_estimation["estimated energy (kJ) (MLCO2)"] = (
        energy_estimation["max power limit (W)"]
        * energy_estimation["training duration (h)"]
        * HOURS_TO_SECONDS
        * JOULES_TO_KJOULES
    )
    energy_estimation["estimated total energy (kJ) (MLCO2)"] = (
        energy_estimation["max power limit (W)"]
        * energy_estimation["total training duration (h)"]
        * HOURS_TO_SECONDS
        * JOULES_TO_KJOULES
    )

    return energy_estimation

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.environment import CONFIGS_DIR, FIGURES_DIR, METRICS_DIR
from src.features.preprocessing import HOURS_TO_MILISECONDS, MJOULES_TO_JOULES

FIGURES_FORMAT = "png"
SAVE_FIGS_DIR = FIGURES_DIR / FIGURES_FORMAT

if not SAVE_FIGS_DIR.exists():
    os.makedirs(SAVE_FIGS_DIR)

sns.set_theme(
    style="whitegrid",
    context="paper",
    palette="colorblind",
    color_codes=True,
    font_scale=1.5,
)
plt.style.use(CONFIGS_DIR / "figures.mplstyle")
plt.style.use("tableau-colorblind10")

analysis_df = pd.read_parquet(
    os.path.join(
        METRICS_DIR, "processed", "clean-dl-training-energy-consumption-dataset.gzip"
    )
)

TRAIN_STRATEGIES = ["Local N", "Local ML", "Cloud"]
MARKERS = {
    "MobileNet V2": "o",
    "NASNet Mobile": "v",
    "Xception": "^",
    "ResNet50": "X",
    "VGG16": "P",
}
COLORS = {"Local N": "b", "Local ML": "orange", "Cloud": "g"}
ARCHITECTURE_LABELS = [
    "MobileNet\nV2",
    "NASNet\nMobile",
    "ResNet50",
    "VGG16",
    "Xception",
]

analysis_df.query("architecture != 'inception_v3'", inplace=True)
analysis_df.rename(
    columns={
        "energy (MJ)": "energy",
        "gpu usage (%)": "gpu_usage",
        "average temperature (Celsius)": "temperature",
        "emissions (tCO2e)": "emissions",
    },
    inplace=True,
)

analysis_df.replace(
    {
        "Local Normal User": "Local N",
        "Local ML Engineer": "Local ML",
        "mobilenet_v2": "MobileNet V2",
        "nasnet_mobile": "NASNet Mobile",
        "xception": "Xception",
        "resnet50": "ResNet50",
        "vgg16": "VGG16",
    },
    inplace=True,
)

analysis_df["raw energy"] = analysis_df.energy.copy()
analysis_df["energy"] = (
    analysis_df["raw energy"] * MJOULES_TO_JOULES / analysis_df["total seen images"]
)
analysis_df["normalized duration"] = (
    analysis_df["training duration (h)"]
    * HOURS_TO_MILISECONDS
    / analysis_df["total seen images"]
)
analysis_df = analysis_df.sort_values(
    by=["GFLOPs", "training environment", "architecture"], ascending=True
).reset_index(drop=True)


# Create a bar plot for each architecture and training environment combination
architecture_order = {
    "MobileNet V2": 0,
    "NASNet Mobile": 1,
    "ResNet50": 2,
    "Xception": 3,
    "VGG16": 4,
}
architectures = analysis_df.architecture.sort_values(
    key=lambda x: x.map(architecture_order)
).unique()
energy_medians = (
    analysis_df.groupby(["architecture", "training environment"])["energy"]
    .median()
    .reset_index()
)

y = np.arange(len(architectures)) * 2
bar_width = 0.5
multiplier = 0
hatches = {"Local N": "//", "Local ML": "..", "Cloud": "|"}

fig, ax = plt.subplots(layout="tight", figsize=(15, 10))

ax.fill_between(
    [0, 1.02], y[0] - bar_width, y[-2] + bar_width * 3.2, color="lightgray", alpha=0.5
)

ax.fill_between(
    [0, 1.02],
    y[-2] + bar_width * 3.8,
    y[-1] + bar_width * 3.2,
    color="lightgray",
    alpha=0.5,
)

for training_environment in TRAIN_STRATEGIES:
    energy = energy_medians.query(
        f"`training environment` == '{training_environment}'"
    ).sort_values(by="architecture", key=lambda x: x.map(architecture_order))
    offset = (bar_width + 0.05) * multiplier
    if training_environment == "Local ML":
        best_local_ml = energy.query("`architecture` != 'VGG16'")["energy"].values
        rects = ax.barh(
            y=y[:-1] + offset,
            width=best_local_ml,
            height=bar_width,
            hatch=hatches[training_environment],
            color="green",
        )
        ax.bar_label(rects, fmt="%.2f", label_type="edge", fontsize=12)
        vgg16_energy = energy.query("`architecture` == 'VGG16'")["energy"].values[0]
        rects = ax.barh(
            y=y[-1] + offset,
            width=vgg16_energy,
            height=bar_width,
            hatch=hatches[training_environment],
            color="gray",
            label=training_environment,
        )
        ax.bar_label(rects, fmt="%.2f", label_type="edge", fontsize=12)
    elif training_environment == "Cloud":
        worst_cloud = energy.query("`architecture` != 'VGG16'")["energy"].values
        rects = ax.barh(
            y=y[:-1] + offset,
            width=worst_cloud,
            height=bar_width,
            hatch=hatches[training_environment],
            color="gray",
            label=training_environment,
        )
        ax.bar_label(rects, fmt="%.2f", label_type="edge", fontsize=12)
        vgg16_energy = energy.query("`architecture` == 'VGG16'")["energy"].values[0]
        rects = ax.barh(
            y=y[-1] + offset,
            width=vgg16_energy,
            height=bar_width,
            hatch=hatches[training_environment],
            color="green",
        )
        ax.bar_label(rects, fmt="%.2f", label_type="edge", fontsize=12)
    else:
        rects = ax.barh(
            y=y[: len(energy)] + offset,
            width=energy["energy"],
            height=bar_width,
            label=training_environment,
            hatch=hatches[training_environment],
            color="gray",
        )
        ax.bar_label(rects, fmt="%.2f", label_type="edge", fontsize=12)
    multiplier += 1

ax.set_xlabel("Median energy consumption (J/image)", fontsize=18)
ax.set_yticks(y + bar_width, architectures, fontsize=18)
ax.invert_yaxis()
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title="Training environment",
    loc="lower center",
    ncols=3,
    bbox_to_anchor=(0.5, 0.98),
)

plt.savefig(
    os.path.join(SAVE_FIGS_DIR, f"energy-comparison-slide-verion.{FIGURES_FORMAT}"),
    bbox_inches="tight",
)

fig, ax = plt.subplots(layout="tight", figsize=(15, 10))
multiplier = 0

ax.fill_between(
    [0, 1.02],
    y[0] - 0.8 * bar_width,
    y[0] + bar_width * 2.8,
    color="lightgray",
    alpha=0.5,
)

f1_scores = analysis_df.groupby("architecture")["f1-score"].median()

for training_environment in TRAIN_STRATEGIES:
    energy = energy_medians.query(
        f"`training environment` == '{training_environment}'"
    ).sort_values(by="architecture", key=lambda x: x.map(architecture_order))
    offset = (bar_width + 0.05) * multiplier
    rects = ax.barh(
        y=y[: len(energy)] + offset,
        width=energy["energy"],
        height=bar_width,
        label=training_environment,
        hatch=hatches[training_environment],
        color="gray",
    )
    ax.bar_label(rects, fmt="%.2f", label_type="edge", fontsize=12)
    multiplier += 1

ax.text(
    0.95,
    y[0] - bar_width,
    "F1-score",
    va="center",
    ha="center",
    fontsize=18,
)

ax.vlines(
    f1_scores, y - 0.5 * bar_width, y + 2.5 * bar_width, color="black", linewidth=3
)
for i, f1_score in enumerate(f1_scores):
    ax.text(
        f1_score + 0.01,
        y[i] + bar_width,
        f"{f1_score:.2f}",
        va="center",
        ha="left",
        fontsize=12,
    )

ax.set_xlabel("Median energy consumption (J/image)", fontsize=18)
ax.set_yticks(y + bar_width, architectures, fontsize=18)
ax.invert_yaxis()
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title="Training environment",
    loc="lower center",
    ncols=3,
    bbox_to_anchor=(0.5, 0.98),
)

plt.savefig(
    os.path.join(
        SAVE_FIGS_DIR, f"energy-f1-score-comparison-slide-verion.{FIGURES_FORMAT}"
    ),
    bbox_inches="tight",
)


fig, ax = plt.subplots(layout="tight", figsize=(15, 10))
multiplier = 0

for training_environment in TRAIN_STRATEGIES:
    energy = energy_medians.query(
        f"`training environment` == '{training_environment}'"
    ).sort_values(by="architecture", key=lambda x: x.map(architecture_order))
    offset = (bar_width + 0.05) * multiplier
    if training_environment == "Local ML":
        best_local_ml = energy.query("`architecture` != 'VGG16'")["energy"].values
        rects = ax.barh(
            y=y[:-1] + offset,
            width=best_local_ml,
            height=bar_width,
            hatch=hatches[training_environment],
            color="gray",
            label=training_environment,
        )
        ax.bar_label(rects, fmt="%.2f", label_type="edge", fontsize=12)
        vgg16_energy = energy.query("`architecture` == 'VGG16'")["energy"].values[0]
        rects = ax.barh(
            y=y[-1] + offset,
            width=vgg16_energy,
            height=bar_width,
            hatch=hatches[training_environment],
            color="red",
        )
        ax.bar_label(rects, fmt="%.2f", label_type="edge", fontsize=12)
    elif training_environment == "Cloud":
        worst_cloud = energy.query("`architecture` != 'VGG16'")["energy"].values
        rects = ax.barh(
            y=y[:-1] + offset,
            width=worst_cloud,
            height=bar_width,
            hatch=hatches[training_environment],
            color="gray",
            label=training_environment,
        )
        ax.bar_label(rects, fmt="%.2f", label_type="edge", fontsize=12)
        vgg16_energy = energy.query("`architecture` == 'VGG16'")["energy"].values[0]
        rects = ax.barh(
            y=y[-1] + offset,
            width=vgg16_energy,
            height=bar_width,
            hatch=hatches[training_environment],
            color="red",
        )
        ax.bar_label(rects, fmt="%.2f", label_type="edge", fontsize=12)
    else:
        rects = ax.barh(
            y=y[0] + offset,
            width=energy.query("architecture == 'MobileNet V2'")["energy"],
            height=bar_width,
            label=training_environment,
            hatch=hatches[training_environment],
            color="gray",
        )
        ax.bar_label(rects, fmt="%.2f", label_type="edge", fontsize=12)
        rects = ax.barh(
            y=y[1] + offset,
            width=energy.query("architecture == 'NASNet Mobile'")["energy"],
            height=bar_width,
            hatch=hatches[training_environment],
            color="red",
        )
        ax.bar_label(rects, fmt="%.2f", label_type="edge", fontsize=12)
    multiplier += 1


ax.set_xlabel("Median energy consumption (J/image)", fontsize=18)
ax.set_yticks(y + bar_width, architectures, fontsize=18)
ax.invert_yaxis()
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title="Training environment",
    loc="lower center",
    ncols=3,
    bbox_to_anchor=(0.5, 0.98),
)

plt.savefig(
    os.path.join(
        SAVE_FIGS_DIR, f"energy-greedy-comparison-slide-verion.{FIGURES_FORMAT}"
    ),
    bbox_inches="tight",
)

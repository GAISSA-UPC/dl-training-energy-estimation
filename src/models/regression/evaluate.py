import pickle
import re

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.data.gpu_specs import get_gpu_specs
from src.environment import (
    CONFIGS_DIR,
    DATA_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    SEED,
)
from src.features.analysis import build_energy_estimation
from src.features.preprocessing import (
    HOURS_TO_SECONDS,
    JOULES_TO_KJOULES,
    MJOULES_TO_KJOULES,
)

RELEVANT_FEATURES = [
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
    "energy (MJ)",  # Final target
]

print("Loading data...")
data = pd.read_parquet(
    METRICS_DIR / "processed" / "clean-dl-training-energy-consumption-dataset.gzip", columns=RELEVANT_FEATURES
)


with open(DATA_DIR / "analysis" / "processed" / "mean_power_draw_by_window.pkl", "rb") as f:
    mean_power_draw = pickle.load(f)

energy_estimation = build_energy_estimation(mean_power_draw, 10).query("`window size` == 5")

print("Preprocessing data...")
# Split image size into image height and image width
data["image height"] = data["image size"].apply(lambda x: int(re.findall(r"\d+", x)[0]))
data["image width"] = data["image size"].apply(lambda x: int(re.findall(r"\d+", x)[1]))

# Add GPU specifications specified in src.data.gpu_specs.py using the 'gpu model' feature
groups = []
for index, group in data.groupby("gpu model"):
    gpu_model = group["gpu model"].iloc[0]

    gpu_model = gpu_model.replace("NVIDIA GeForce ", "")
    specs = get_gpu_specs(gpu_model)
    # group["gpu architecture"] = specs.architecture
    group["gpu cuda cores"] = specs.cuda_cores
    group["gpu total memory"] = specs.memory
    group["gpu max power"] = specs.power

    groups.append(group)

data = pd.concat(groups)

# Standardize numerical features
X = data.drop(columns=["run_id", "image size", "gpu model", "training duration (h)", "measured epochs", "energy (MJ)"])
y = data["energy (MJ)"] * MJOULES_TO_KJOULES

X_numerical = X.select_dtypes(include=["int64", "float64"])
X_standardized = StandardScaler().fit_transform(X_numerical)
X_numerical = pd.DataFrame(X_standardized, index=X_numerical.index, columns=X_numerical.columns)
X_categorical = X.select_dtypes(include=["object"])
X_processed = pd.get_dummies(pd.concat([X_numerical, X_categorical], axis=1))

_, X_test, _, y_test = train_test_split(X_processed, y, test_size=0.30, random_state=SEED)
test_run_ids = data.loc[X_test.index, "run_id"]

with open(MODELS_DIR / "kernel-ridge-regression-power-estimator.joblib", "rb") as f:
    power_estimator = joblib.load(f)

y_pred = power_estimator.predict(X_test)
power_estimator_energy_pred = (
    y_pred * data.loc[X_test.index, "training duration (h)"] * HOURS_TO_SECONDS * JOULES_TO_KJOULES
)

with open(MODELS_DIR / "kernel-ridge-regression-full-energy-estimator.joblib", "rb") as f:
    energy_estimator = joblib.load(f)

y_pred = energy_estimator.predict(X_test)
energy_estimator_energy_pred = y_pred * data.loc[X_test.index, "measured epochs"]

results = pd.concat([test_run_ids, y_test, power_estimator_energy_pred, energy_estimator_energy_pred], axis=1)
results.columns = ["run_id", "true energy (kJ)", "power estimator energy (kJ)", "energy estimator energy (kJ)"]

energy_estimation = energy_estimation.query("run_id in @test_run_ids")

energy_estimation = results.merge(
    energy_estimation[
        [
            "run_id",
            "total energy (kJ)",
            "estimated total energy (kJ) (online power-based)",
            "estimated total energy (kJ) (online epoch-energy-based)",
            "estimated total energy (kJ) (GA)",
            "estimated total energy (kJ) (MLCO2)",
        ]
    ],
    on="run_id",
)

print(
    "RMSE offline power-based estimator:",
    root_mean_squared_error(energy_estimation["true energy (kJ)"], energy_estimation["power estimator energy (kJ)"]),
)
print(
    "RMSE offline epoch-energy-based estimator:",
    root_mean_squared_error(energy_estimation["true energy (kJ)"], energy_estimation["energy estimator energy (kJ)"]),
)
print(
    "RMSE online power-based estimator:",
    root_mean_squared_error(
        energy_estimation["true energy (kJ)"], energy_estimation["estimated total energy (kJ) (online power-based)"]
    ),
)
print(
    "RMSE online epoch-energy-based estimator:",
    root_mean_squared_error(
        energy_estimation["true energy (kJ)"],
        energy_estimation["estimated total energy (kJ) (online epoch-energy-based)"],
    ),
)
print(
    "RMSE GA estimator:",
    root_mean_squared_error(
        energy_estimation["true energy (kJ)"], energy_estimation["estimated total energy (kJ) (GA)"]
    ),
)
print(
    "RMSE MLCO2 estimator:",
    root_mean_squared_error(
        energy_estimation["true energy (kJ)"], energy_estimation["estimated total energy (kJ) (MLCO2)"]
    ),
)

plt.style.use([CONFIGS_DIR / "figures.mplstyle", "seaborn-v0_8-colorblind"])
alpha = 0.5

fig = plt.figure(figsize=(20, 5))

x = energy_estimation["true energy (kJ)"]

ax = fig.add_subplot(1, 6, 1)
y = energy_estimation["estimated total energy (kJ) (GA)"]
ax.scatter(x, y, alpha=alpha)
slope, intercept, r_value, _, _ = stats.linregress(x, y)
ax.plot(x, slope * x + intercept, color="blue", label="fitted line")
ax.plot(x, x, color="red", label="y=x")
ax.text(0.02, 0.95, f"$R^2$: {r_value**2:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top")
ax.legend(loc="lower right")
ax.set_xlabel("True energy (kJ)")
ax.set_ylabel("Predicted energy (kJ)")
ax.set_title("Green Algorithms method")

ax = fig.add_subplot(1, 6, 2, sharey=ax)
y = energy_estimation["estimated total energy (kJ) (MLCO2)"]
ax.scatter(x, y, alpha=alpha)
slope, intercept, r_value, _, _ = stats.linregress(x, y)
ax.plot(x, slope * x + intercept, color="blue", label="fitted line")
ax.plot(x, x, color="red", label="y=x")
ax.text(0.02, 0.95, f"$R^2$: {r_value**2:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top")
ax.set_xlabel("True energy (kJ)")
ax.set_title("MLCO2 Impact method")

ax = fig.add_subplot(1, 6, 3, sharey=ax)
y = energy_estimation["estimated total energy (kJ) (online power-based)"]
ax.scatter(x, y, alpha=alpha)
slope, intercept, r_value, _, _ = stats.linregress(x, y)
ax.plot(x, slope * x + intercept, color="blue", label="fitted line")
ax.plot(x, x, color="red", label="y=x")
ax.text(0.02, 0.95, f"$R^2$: {r_value**2:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top")
ax.set_xlabel("True energy (kJ)")
ax.set_title("Online power-based method")

ax = fig.add_subplot(1, 6, 4, sharey=ax)
y = energy_estimation["estimated total energy (kJ) (online epoch-energy-based)"]
ax.scatter(x, y, alpha=alpha)
slope, intercept, r_value, _, _ = stats.linregress(x, y)
ax.plot(x, slope * x + intercept, color="blue", label="fitted line")
ax.plot(x, x, color="red", label="y=x")
ax.text(0.02, 0.95, f"$R^2$: {r_value**2:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top")
ax.set_xlabel("True energy (kJ)")
ax.set_title("Online epoch-energy-based method")

ax = fig.add_subplot(1, 6, 5, sharey=ax)
y = energy_estimation["power estimator energy (kJ)"]
ax.scatter(x, y, alpha=alpha)
slope, intercept, r_value, _, _ = stats.linregress(x, y)
ax.plot(x, slope * x + intercept, color="blue", label="fitted line")
ax.plot(x, x, color="red", label="y=x")
ax.text(0.02, 0.95, f"$R^2$: {r_value**2:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top")
ax.set_xlabel("True energy (kJ)")
ax.set_title("Offline power-based method")

ax = fig.add_subplot(1, 6, 6, sharey=ax)
y = energy_estimation["energy estimator energy (kJ)"]
ax.scatter(x, y, alpha=alpha)
slope, intercept, r_value, _, _ = stats.linregress(x, y)
ax.plot(x, slope * x + intercept, color="blue", label="fitted line")
ax.plot(x, x, color="red", label="y=x")
ax.text(0.02, 0.95, f"$R^2$: {r_value**2:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top")
ax.set_xlabel("True energy (kJ)")
ax.set_title("Offline epoch-energy-based method")

for ax in fig.get_axes():
    ax.grid(False)

plt.tight_layout()

plt.savefig(FIGURES_DIR / "pdf" / "energy-estimation-methods-evaluation.pdf")

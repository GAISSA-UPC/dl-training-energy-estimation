import re

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.data.gpu_specs import get_gpu_specs
from src.environment import FIGURES_DIR, METRICS_DIR, REPORTS_DIR, SEED
from src.features.preprocessing import MJOULES_TO_KJOULES
from src.models.regression.utils import Models, ModelTrainer

RELEVANT_FEATURES = [
    # "training environment",
    # "architecture",
    "run_id",
    "batch size",
    "image size",
    "gpu model",
    "total ram (GB)",
    "training size",
    "validation size",
    "measured epochs",
    "GFLOPs",
    # "total seen images",
    "energy (MJ)",  # Final target
]
EPOCH_RELEVANT_FEATURES = [
    "run_id",
    "epoch",
    # "epoch duration (s)",
    # "epoch mean gpu power (W)",
    # "epoch gpu energy (kJ)",
    # "epoch ram energy (kJ)",
    "total energy (kJ)",  # Target
]

print("Loading data...")
data = pd.read_parquet(
    METRICS_DIR / "processed" / "clean-dl-training-energy-consumption-dataset.gzip", columns=RELEVANT_FEATURES
)

epoch_data = pd.read_parquet(
    METRICS_DIR / "processed" / "clean-dl-epoch-energy-consumption-dataset.gzip", columns=EPOCH_RELEVANT_FEATURES
)

# Add average epoch energy to the dataset
data.set_index("run_id", inplace=True)
data["average epoch energy (kJ)"] = epoch_data.groupby("run_id")["total energy (kJ)"].mean()
data["average stable epoch energy (kJ)"] = epoch_data.query("epoch >= 10").groupby("run_id")["total energy (kJ)"].mean()
data.reset_index(drop=True, inplace=True)

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
X = data.drop(
    columns=[
        "average epoch energy (kJ)",
        "average stable epoch energy (kJ)",
        "image size",
        "gpu model",
        "measured epochs",
        "energy (MJ)",
    ]
)
y = data["average epoch energy (kJ)"]
y_stable = data["average stable epoch energy (kJ)"]
gold_standard = data["energy (MJ)"] * MJOULES_TO_KJOULES

X_numerical = X.select_dtypes(include=["int64", "float64"])
X_categorical = X.select_dtypes(include=["object"])
X_standardized = StandardScaler().fit_transform(X_numerical)
X_numerical = pd.DataFrame(X_standardized, index=X_numerical.index, columns=X_numerical.columns)
X = pd.concat([X_numerical, X_categorical], axis=1)
X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.30, random_state=SEED)

print("Train models including initial epochs in the training set")
results_with_initial_epochs = ModelTrainer(X_train, y_train).run_training(
    "full-energy",
    [Models.LINEAR_REGRESSOR, Models.RIDGE_REGRESSOR, Models.KERNEL_RIDGE_REGRESSOR, Models.SVM_REGRESSOR],
)
print("")
results_with_initial_epochs_df = pd.DataFrame.from_dict(
    {key: values for key, values in results_with_initial_epochs.items() if key != "model"}
)
results_with_initial_epochs_df["has initial epochs"] = True

print("Train models without initial epochs in the training set")
results_without_initial_epochs = ModelTrainer(X_train, y_stable.loc[y_train.index]).run_training(
    "stable-energy",
    [Models.LINEAR_REGRESSOR, Models.RIDGE_REGRESSOR, Models.KERNEL_RIDGE_REGRESSOR, Models.SVM_REGRESSOR],
)
print("")
results_without_initial_epochs_df = pd.DataFrame.from_dict(
    {key: values for key, values in results_without_initial_epochs.items() if key != "model"}
)
results_without_initial_epochs_df["has initial epochs"] = False
results_df = pd.concat([results_with_initial_epochs_df, results_without_initial_epochs_df]).reset_index(drop=True)
results_df.to_latex(REPORTS_DIR / "energy-estimators-comparison.tex", encoding="utf8")
print(results_df)
print("")

models = results_with_initial_epochs["model"] + results_without_initial_epochs["model"]

best_model_idx = results_df["rmse"].idxmin()
best_model = models[results_df["rmse"].idxmin()]

if not results_df.loc[best_model_idx, "has initial epochs"]:
    X_test = X_test_without_initial_epochs
    y_test = y_test_without_initial_epochs

y_pred = best_model.predict(X_test)
energy_pred = y_pred * data.loc[X_test.index]["measured epochs"]
epoch_energy_rmse = root_mean_squared_error(y_test, y_pred)
final_target_rmse = root_mean_squared_error(gold_standard.loc[y_test.index], energy_pred)
print(f"Best model: {results_df.loc[best_model_idx, 'model_name']}")
print(f"RMSE for average epoch energy (kJ): {epoch_energy_rmse}")
print(f"RMSE for final target (kJ): {final_target_rmse}")

# Plot predictions vs true values
_, ax = plt.subplots(figsize=(10, 10))
x = y_test
y = y_pred
ax.scatter(x, y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
ax.plot(x, x, color="red", label="y=x")
ax.text(0.05, 0.9, f"$R^2$: {r_value**2:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top")
ax.set_xlabel("True epoch energy (kJ)")
ax.set_ylabel("Predicted epoch energy (kJ)")
ax.set_title("Epoch energy predictions vs true values")
ax.legend(loc="upper left")
plt.savefig(FIGURES_DIR / "epoch-energy-predictions-vs-true-values.png")

_, ax = plt.subplots(figsize=(10, 10))
x = gold_standard.loc[y_test.index]
y = energy_pred
ax.scatter(x, y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
ax.plot(x, x, color="red", label="y=x")
ax.text(0.05, 0.9, f"$R^2$: {r_value**2:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top")
ax.set_xlabel("True energy (kJ)")
ax.set_ylabel("Predicted energy (kJ)")
ax.set_title("Energy predictions vs true values")
ax.legend(loc="upper left")
plt.savefig(FIGURES_DIR / "method-2-energy-predictions-vs-true-values.png")

# Save results with test set
with open(REPORTS_DIR / "energy-estimator-test-set-results.txt", "w", encoding="utf8") as f:
    f.write(f"Best model: {results_df.loc[best_model_idx, 'model_name']}\n")
    f.write(f"RMSE for average epoch energy: {epoch_energy_rmse}\n")
    f.write(f"RMSE for final target: {final_target_rmse}\n")

import re

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.gpu_specs import get_gpu_specs
from src.environment import FIGURES_DIR, METRICS_DIR, REPORTS_DIR, SEED
from src.features.preprocessing import (
    HOURS_TO_SECONDS,
    JOULES_TO_KJOULES,
    MJOULES_TO_JOULES,
)
from src.models.regression.utils import Models, ModelTrainer

RELEVANT_FEATURES = [
    # "training environment",
    # "architecture",
    # "run_id",
    "batch size",
    "image size",
    "gpu model",
    "total ram (GB)",
    "training size",
    "validation size",
    "training duration (h)",
    "GFLOPs",
    # "total seen images",
    "average gpu power (W)",  # Target
    "energy (MJ)",  # Final target
]

print("Loading data...")
data = pd.read_parquet(
    METRICS_DIR / "processed" / "clean-dl-training-energy-consumption-dataset.gzip", columns=RELEVANT_FEATURES
)

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
X = data.drop(columns=["average gpu power (W)", "image size", "gpu model", "training duration (h)", "energy (MJ)"])
y = data["average gpu power (W)"]
gold_standard = data["energy (MJ)"] * MJOULES_TO_JOULES

X_numerical = X.select_dtypes(include=["int64", "float64"])
X_categorical = X.select_dtypes(include=["object"])
X_standardized = StandardScaler().fit_transform(X_numerical)
X_numerical = pd.DataFrame(X_standardized, index=X_numerical.index, columns=X_numerical.columns)
# X_encoded = TargetEncoder(target_type="continuous").fit_transform(X_categorical, y)
# X_categorical = pd.DataFrame(X_encoded, index=X_categorical.index, columns=X_categorical.columns)
# X_encoded = pd.concat([X_numerical, X_categorical], axis=1)
X = pd.concat([X_numerical, X_categorical], axis=1)
X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.30, random_state=SEED)

results = ModelTrainer(X_train, y_train).run_training(
    "power", [Models.LINEAR_REGRESSOR, Models.RIDGE_REGRESSOR, Models.KERNEL_RIDGE_REGRESSOR, Models.SVM_REGRESSOR]
)
results_df = pd.DataFrame.from_dict({key: values for key, values in results.items() if key != "model"})
results_df.to_latex(REPORTS_DIR / "power-estimators-comparison.tex", encoding="utf8")
print(results_df)
print("")

models = results["model"]

best_model_idx = results_df["rmse"].idxmin()
best_model = models[best_model_idx]

y_pred = best_model.predict(X_test)
energy_pred = y_pred * data.loc[X_test.index]["training duration (h)"] * HOURS_TO_SECONDS
average_power_rmse = root_mean_squared_error(y_test, y_pred)
final_target_rmse = root_mean_squared_error(gold_standard.loc[y_test.index], energy_pred) * JOULES_TO_KJOULES
print(f"Best model: {results_df.loc[best_model_idx, 'model_name']}")
print(f"RMSE for average power (W): {average_power_rmse}")
print(f"RMSE for final target (kJ): {final_target_rmse}")

# Plot predictions vs true values
_, ax = plt.subplots(figsize=(10, 10))
x = y_test
y = y_pred
ax.scatter(x, y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
ax.plot(x, x, color="red", label="y=x")
ax.text(0.02, 0.95, f"$R^2$: {r_value**2:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top")
ax.set_xlabel("True average power (W)")
ax.set_ylabel("Predicted average power (W)")
ax.set_title("Average power predictions vs true values")
ax.legend(loc="upper left")
plt.savefig(FIGURES_DIR / "average-power-predictions-vs-true-values.png")

_, ax = plt.subplots(figsize=(10, 10))
x = gold_standard.loc[y_test.index] * JOULES_TO_KJOULES
y = energy_pred * JOULES_TO_KJOULES
ax.scatter(x, y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
ax.plot(x, x, color="red", label="y=x")
ax.text(0.02, 0.95, f"$R^2$: {r_value**2:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top")
ax.set_xlabel("True energy (kJ)")
ax.set_ylabel("Predicted energy (kJ)")
ax.set_title("Energy predictions vs true values")
ax.legend(loc="upper left")
plt.savefig(FIGURES_DIR / "method-1-energy-predictions-vs-true-values.png")


# Save results with test set
with open(REPORTS_DIR / "power-estimator-test-set-results.txt", "w", encoding="utf8") as f:
    f.write(f"Best model: {results_df.loc[best_model_idx, 'model_name']}\n")
    f.write(f"RMSE for average power (W): {average_power_rmse}\n")
    f.write(f"RMSE for final target (kJ): {final_target_rmse}\n")

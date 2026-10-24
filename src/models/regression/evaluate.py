"""
Compare the energy estimation methods on the test runs (held-out training configurations).

PRE-P and PRE-E are evaluated a priori: the training time and the number of epochs are predicted, not measured. The
variants that use the measured duration or number of epochs are also saved, to separate the error of the power and
epoch-energy models from the error of predicting how long training lasts. STEP, GA and MLCO2 use measurements of the run
(they are runtime or post hoc methods). Run the three train_*_predictor scripts first.
"""

import pickle

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src.environment import DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.features.analysis import build_energy_estimation
from src.features.preprocessing import HOURS_TO_SECONDS, JOULES_TO_KJOULES
from src.models.regression.pre_data import FEATURES, STABILIZING_EPOCH, load_pre_dataset, split_by_configuration
from src.models.regression.utils import regression_metrics

WINDOW_SIZE = 5


CV_FOLDS = 5
TARGETS = {
    "power": "average power (W)",
    "full-energy": "average epoch energy (kJ)",
    "epochs": "measured epochs",
    "epoch-duration": "average epoch duration (s)",
}


def load_best_model(target_metric):
    with open(MODELS_DIR / f"pre-{target_metric}-best.joblib", "rb") as f:
        return joblib.load(f)


def pre_estimates(runs, predictions):
    """PRE energy estimates (kJ) from the predictions of each PRE model for the given runs."""
    measured_seconds = runs["training duration (h)"] * HOURS_TO_SECONDS
    return pd.DataFrame(
        {
            "run_id": runs["run_id"].values,
            "true energy (kJ)": runs["energy (kJ)"].values,
            "PRE-P energy (kJ)": predictions["power"] * predictions["epochs"] * predictions["epoch-duration"] * JOULES_TO_KJOULES,
            "PRE-E energy (kJ)": predictions["full-energy"] * predictions["epochs"],
            "PRE-P energy with measured duration (kJ)": predictions["power"] * measured_seconds.values * JOULES_TO_KJOULES,
            "PRE-E energy with measured epochs (kJ)": predictions["full-energy"] * runs["measured epochs"].values,
        }
    )


def out_of_fold_predictions(data, models):
    """Predict every run with models refitted (same hyperparameters) on folds that exclude its configuration."""
    predictions = {target: pd.Series(index=data.index, dtype=float) for target in models}
    for train_index, test_index in GroupKFold(n_splits=CV_FOLDS).split(data, groups=data["configuration"]):
        train, test = data.iloc[train_index], data.iloc[test_index]
        for target, model in models.items():
            fitted = clone(model).fit(train[FEATURES], train[TARGETS[target]])
            predictions[target].iloc[test_index] = fitted.predict(test[FEATURES])
    return {target: values.values for target, values in predictions.items()}


def add_runtime_methods(results, step_estimates):
    return results.merge(step_estimates, on="run_id")


print("Loading data...")
data = load_pre_dataset()
_, test = split_by_configuration(data)
models = {target: load_best_model(target) for target in TARGETS}

with open(DATA_DIR / "analysis" / "processed" / "mean_power_draw_by_window.pkl", "rb") as f:
    mean_power_draw = pickle.load(f)
step_estimates = build_energy_estimation(mean_power_draw, STABILIZING_EPOCH).query("`window size` == @WINDOW_SIZE")[
    [
        "run_id",
        "total energy (kJ)",
        "estimated total energy (kJ) (STEP-P)",
        "estimated total energy (kJ) (STEP-E)",
        "estimated total energy (kJ) (GA)",
        "estimated total energy (kJ) (MLCO2)",
    ]
]

# Held-out test configurations
test_predictions = {target: model.predict(test[FEATURES]) for target, model in models.items()}
energy_estimation = add_runtime_methods(pre_estimates(test, test_predictions), step_estimates)
energy_estimation.to_parquet(
    DATA_DIR / "analysis" / "processed" / "energy-estimation-results.gzip",
    index=False,
    compression="gzip",
)

# Every configuration held out once (out-of-fold predictions)
cv_estimation = add_runtime_methods(pre_estimates(data, out_of_fold_predictions(data, models)), step_estimates)

methods = {
    "GA": "estimated total energy (kJ) (GA)",
    "MLCO2": "estimated total energy (kJ) (MLCO2)",
    "STEP-P": "estimated total energy (kJ) (STEP-P)",
    "STEP-E": "estimated total energy (kJ) (STEP-E)",
    "PRE-P": "PRE-P energy (kJ)",
    "PRE-E": "PRE-E energy (kJ)",
    "PRE-P (measured duration)": "PRE-P energy with measured duration (kJ)",
    "PRE-E (measured epochs)": "PRE-E energy with measured epochs (kJ)",
}


def compare(estimation):
    return pd.DataFrame(
        {name: regression_metrics(estimation["true energy (kJ)"], estimation[column]) for name, column in methods.items()}
    ).T


comparison = compare(energy_estimation)
cv_comparison = compare(cv_estimation)
print(comparison)
print(cv_comparison)

with open(REPORTS_DIR / "energy-estimation-test-set-comparison.txt", "w", encoding="utf8") as f:
    f.write("R2 = 1 - SSE/SST; r2 = squared Pearson correlation; bias = mean signed error (kJ)\n")
    f.write(f"Only runs with STEP estimates (window size {WINDOW_SIZE}, stabilizing epoch {STABILIZING_EPOCH}).\n\n")
    f.write(
        f"Held-out test set: {len(energy_estimation)} runs from {test['configuration'].nunique()} configurations "
        f"({len(test)} runs before matching STEP estimates)\n"
    )
    f.write(comparison.to_string(float_format=lambda x: f"{x:.4f}"))
    f.write(
        f"\n\nOut-of-fold ({CV_FOLDS} folds grouped by configuration, hyperparameters from the training split): "
        f"{len(cv_estimation)} runs from {data['configuration'].nunique()} configurations\n"
    )
    f.write(cv_comparison.to_string(float_format=lambda x: f"{x:.4f}"))
    f.write("\n")

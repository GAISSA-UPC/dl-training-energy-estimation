"""
Compare the energy estimation methods on the test runs.

PRE-P predicts the average power and PRE-E the average energy per epoch. The user provides an estimate of the training
time (PRE-P) or of the number of epochs (PRE-E); here the measured values stand in for an accurate user estimate. STEP,
GA and MLCO2 use measurements of the run (they are runtime or post hoc methods). Run the train_*_predictor scripts first.
"""

import pickle

import joblib
import pandas as pd

from src.environment import DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.features.analysis import build_energy_estimation
from src.features.preprocessing import HOURS_TO_SECONDS, JOULES_TO_KJOULES
from src.models.regression.pre_data import FEATURES, STABILIZING_EPOCH, load_pre_dataset, split_runs
from src.models.regression.utils import regression_metrics

WINDOW_SIZE = 5


def load_best_model(target_metric):
    with open(MODELS_DIR / f"pre-{target_metric}-best.joblib", "rb") as f:
        return joblib.load(f)


print("Loading data...")
data = load_pre_dataset()
_, test = split_runs(data)

power = load_best_model("power").predict(test[FEATURES])
epoch_energy = load_best_model("full-energy").predict(test[FEATURES])

user_seconds = test["training duration (h)"] * HOURS_TO_SECONDS
results = pd.DataFrame(
    {
        "run_id": test["run_id"],
        "true energy (kJ)": test["energy (kJ)"],
        "PRE-P energy (kJ)": power * user_seconds * JOULES_TO_KJOULES,
        "PRE-E energy (kJ)": epoch_energy * test["measured epochs"],
    }
)

with open(DATA_DIR / "analysis" / "processed" / "mean_power_draw_by_window.pkl", "rb") as f:
    mean_power_draw = pickle.load(f)

energy_estimation = build_energy_estimation(mean_power_draw, STABILIZING_EPOCH).query("`window size` == @WINDOW_SIZE")
energy_estimation = results.merge(
    energy_estimation[
        [
            "run_id",
            "total energy (kJ)",
            "estimated total energy (kJ) (STEP-P)",
            "estimated total energy (kJ) (STEP-E)",
            "estimated total energy (kJ) (GA)",
            "estimated total energy (kJ) (MLCO2)",
        ]
    ],
    on="run_id",
)

energy_estimation.to_parquet(
    DATA_DIR / "analysis" / "processed" / "energy-estimation-results.gzip",
    index=False,
    compression="gzip",
)

methods = {
    "GA": "estimated total energy (kJ) (GA)",
    "MLCO2": "estimated total energy (kJ) (MLCO2)",
    "STEP-P": "estimated total energy (kJ) (STEP-P)",
    "STEP-E": "estimated total energy (kJ) (STEP-E)",
    "PRE-P": "PRE-P energy (kJ)",
    "PRE-E": "PRE-E energy (kJ)",
}
comparison = pd.DataFrame(
    {name: regression_metrics(energy_estimation["true energy (kJ)"], energy_estimation[column]) for name, column in methods.items()}
).T
print(comparison)

with open(REPORTS_DIR / "energy-estimation-test-set-comparison.txt", "w", encoding="utf8") as f:
    f.write("R2 = 1 - SSE/SST; r2 = squared Pearson correlation; bias = mean signed error (kJ)\n")
    f.write(
        f"Test set: {len(energy_estimation)} runs with STEP estimates (window size {WINDOW_SIZE}, stabilizing epoch "
        f"{STABILIZING_EPOCH}) out of {len(test)} test runs\n\n"
    )
    f.write(comparison.to_string(float_format=lambda x: f"{x:.4f}"))
    f.write("\n")

"""
Train the PRE-P models, which predict the average power (GPU and RAM) of a training run from features known before
training. The test set holds out whole training configurations (see `pre_data`).
"""

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src.environment import FIGURES_DIR, REPORTS_DIR
from src.features.preprocessing import HOURS_TO_SECONDS, JOULES_TO_KJOULES
from src.models.regression.pre_data import FEATURES, load_pre_dataset, split_by_configuration
from src.models.regression.utils import Models, regression_metrics, train_target

MODELS = [Models.LINEAR_REGRESSOR, Models.RIDGE_REGRESSOR, Models.KERNEL_RIDGE_REGRESSOR, Models.SVM_REGRESSOR]

logger.info("Loading data...")
data = load_pre_dataset()
train, test = split_by_configuration(data)
logger.info(f"{len(train)} training runs, {len(test)} test runs, {test['configuration'].nunique()} test configurations")

results_df, best_model = train_target(train, FEATURES, "average power (W)", "power", MODELS)
results_df.to_latex(REPORTS_DIR / "power-estimators-comparison.tex", encoding="utf8")
print(results_df)
print("")

y_test = test["average power (W)"]
y_pred = best_model.predict(test[FEATURES])
power_metrics = regression_metrics(y_test, y_pred)

# Energy obtained with the measured training duration. This isolates the error of the power model; the a priori
# estimate, which also predicts the duration, is computed in evaluate.py.
energy_pred = y_pred * test["training duration (h)"] * HOURS_TO_SECONDS * JOULES_TO_KJOULES
energy_metrics = regression_metrics(test["energy (kJ)"], energy_pred)

best_model_name = results_df.loc[results_df["rmse"].idxmin(), "model_name"]
print(f"Best model: {best_model_name}")
print(f"Average power (W): {power_metrics}")
print(f"Energy with measured duration (kJ): {energy_metrics}")

result = permutation_importance(best_model, test[FEATURES], y_test, n_repeats=10, random_state=42, n_jobs=2)
importances = pd.Series(result.importances_mean, index=FEATURES)
fig, ax = plt.subplots(figsize=(10, 5))
importances.plot.barh(xerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.savefig(FIGURES_DIR / "average-power-estimator-feature-importance-permutation.png")

for x, y, metrics, unit, title, file_name in [
    (y_test, y_pred, power_metrics, "average power (W)", "Average power", "average-power-predictions-vs-true-values"),
    (
        test["energy (kJ)"],
        energy_pred,
        energy_metrics,
        "energy (kJ)",
        "Energy (measured duration)",
        "method-1-energy-predictions-vs-true-values",
    ),
]:
    _, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x, y)
    ax.plot(x, x, color="red", label="y=x")
    ax.text(0.02, 0.95, f"$R^2$: {metrics['R2']:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top")
    ax.set_xlabel(f"True {unit}")
    ax.set_ylabel(f"Predicted {unit}")
    ax.set_title(f"{title} predictions vs true values")
    ax.legend(loc="upper left")
    plt.savefig(FIGURES_DIR / f"{file_name}.png")

with open(REPORTS_DIR / "power-estimator-test-set-results.txt", "w", encoding="utf8") as f:
    f.write(f"Best model: {best_model_name}\n")
    f.write(f"Test set: {len(test)} runs from {test['configuration'].nunique()} held-out configurations\n")
    f.write(f"Average power (W): {power_metrics}\n")
    f.write(f"Energy with measured training duration (kJ): {energy_metrics}\n")

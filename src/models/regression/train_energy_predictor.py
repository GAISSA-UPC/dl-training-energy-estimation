"""
Train the PRE-E models, which predict the average energy per epoch (GPU and RAM) of a training run from features known
before training. Two variants are trained: one on all epochs and one on the stable epochs only. The test set holds out
whole training configurations (see `pre_data`).
"""

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src.environment import FIGURES_DIR, REPORTS_DIR
from src.models.regression.pre_data import FEATURES, load_pre_dataset, split_by_configuration
from src.models.regression.utils import Models, regression_metrics, train_target

MODELS = [Models.LINEAR_REGRESSOR, Models.RIDGE_REGRESSOR, Models.KERNEL_RIDGE_REGRESSOR, Models.SVM_REGRESSOR]

logger.info("Loading data...")
data = load_pre_dataset()
train, test = split_by_configuration(data)
logger.info(f"{len(train)} training runs, {len(test)} test runs, {test['configuration'].nunique()} test configurations")

print("Train models including initial epochs in the training set")
results_full_df, best_full_model = train_target(train, FEATURES, "average epoch energy (kJ)", "full-energy", MODELS)
results_full_df["has initial epochs"] = True
print("")

print("Train models without initial epochs in the training set")
results_stable_df, _ = train_target(train, FEATURES, "average stable epoch energy (kJ)", "stable-energy", MODELS)
results_stable_df["has initial epochs"] = False
print("")

results_df = pd.concat([results_full_df, results_stable_df]).reset_index(drop=True)
results_df.to_latex(REPORTS_DIR / "energy-estimators-comparison.tex", encoding="utf8")
print(results_df)
print("")

# PRE-E multiplies the predicted average epoch energy by the number of epochs, so it uses the model trained on all
# epochs (the stable-epoch model would leave out the energy of the unstable initial epochs).
y_test = test["average epoch energy (kJ)"]
y_pred = best_full_model.predict(test[FEATURES])
epoch_energy_metrics = regression_metrics(y_test, y_pred)

# Energy obtained with the measured number of epochs. This isolates the error of the epoch-energy model; the a priori
# estimate, which also predicts the number of epochs, is computed in evaluate.py.
energy_pred = y_pred * test["measured epochs"]
energy_metrics = regression_metrics(test["energy (kJ)"], energy_pred)

best_model_name = results_full_df.loc[results_full_df["rmse"].idxmin(), "model_name"]
print(f"Best model: {best_model_name}")
print(f"Average epoch energy (kJ): {epoch_energy_metrics}")
print(f"Energy with measured epochs (kJ): {energy_metrics}")

result = permutation_importance(best_full_model, test[FEATURES], y_test, n_repeats=10, random_state=42, n_jobs=2)
importances = pd.Series(result.importances_mean, index=FEATURES)
fig, ax = plt.subplots(figsize=(10, 5))
importances.plot.barh(xerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.savefig(FIGURES_DIR / "energy-consumption-estimator-feature-importance-permutation.png")

for x, y, metrics, unit, title, file_name in [
    (y_test, y_pred, epoch_energy_metrics, "epoch energy (kJ)", "Epoch energy", "epoch-energy-predictions-vs-true-values"),
    (
        test["energy (kJ)"],
        energy_pred,
        energy_metrics,
        "energy (kJ)",
        "Energy (measured epochs)",
        "method-2-energy-predictions-vs-true-values",
    ),
]:
    _, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x, y)
    ax.plot(x, x, color="red", label="y=x")
    ax.text(0.05, 0.9, f"$R^2$: {metrics['R2']:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top")
    ax.set_xlabel(f"True {unit}")
    ax.set_ylabel(f"Predicted {unit}")
    ax.set_title(f"{title} predictions vs true values")
    ax.legend(loc="upper left")
    plt.savefig(FIGURES_DIR / f"{file_name}.png")

with open(REPORTS_DIR / "energy-estimator-test-set-results.txt", "w", encoding="utf8") as f:
    f.write(f"Best model: {best_model_name}\n")
    f.write(f"Test set: {len(test)} runs from {test['configuration'].nunique()} held-out configurations\n")
    f.write(f"Average epoch energy (kJ): {epoch_energy_metrics}\n")
    f.write(f"Energy with measured number of epochs (kJ): {energy_metrics}\n")

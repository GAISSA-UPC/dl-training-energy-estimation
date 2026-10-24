"""
Train the models that predict the number of epochs and the average epoch duration of a training run from features known
before training. PRE-P and PRE-E use them to estimate the training time and the number of epochs, so that the energy
estimate does not depend on measurements of the run itself. The test set holds out whole training configurations
(see `pre_data`).
"""

from loguru import logger

from src.environment import REPORTS_DIR
from src.models.regression.pre_data import FEATURES, load_pre_dataset, split_by_configuration
from src.models.regression.utils import Models, regression_metrics, train_target

MODELS = [Models.LINEAR_REGRESSOR, Models.RIDGE_REGRESSOR, Models.KERNEL_RIDGE_REGRESSOR, Models.SVM_REGRESSOR]

logger.info("Loading data...")
data = load_pre_dataset()
train, test = split_by_configuration(data)

lines = [f"Test set: {len(test)} runs from {test['configuration'].nunique()} held-out configurations"]
for target_column, target_metric in [("measured epochs", "epochs"), ("average epoch duration (s)", "epoch-duration")]:
    print(f"Train models for {target_column}")
    results_df, best_model = train_target(train, FEATURES, target_column, target_metric, MODELS)
    print(results_df)
    metrics = regression_metrics(test[target_column], best_model.predict(test[FEATURES]))
    best_model_name = results_df.loc[results_df["rmse"].idxmin(), "model_name"]
    print(f"Best model: {best_model_name}; test set: {metrics}\n")
    lines.append(f"{target_column}: best model {best_model_name}; test set {metrics}")

with open(REPORTS_DIR / "runtime-estimator-test-set-results.txt", "w", encoding="utf8") as f:
    f.write("\n".join(lines) + "\n")

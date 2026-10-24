from enum import Enum

import joblib
import numpy as np
import pandas as pd
from scipy.stats import expon, randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.environment import MODELS_DIR, SEED

SCORES = {
    "R2": "r2",
    "MAPE": "neg_mean_absolute_percentage_error",
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
}
FOLDS = 5


class Models(Enum):
    LINEAR_REGRESSOR = "linear-regression"
    RIDGE_REGRESSOR = "ridge-regression"
    KERNEL_RIDGE_REGRESSOR = "kernel-ridge-regression"
    SVM_REGRESSOR = "svm-regression"
    RANDOM_FOREST_REGRESSOR = "rf-regression"


def _with_scaler(regressor):
    """Standardize the features inside the model so that the scaler is fitted on the training folds only."""
    return Pipeline([("scaler", StandardScaler()), ("model", regressor)])


class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self._MODELS = {
            Models.LINEAR_REGRESSOR: self.train_LinearRegression,
            Models.RIDGE_REGRESSOR: self.train_RidgeRegression,
            Models.KERNEL_RIDGE_REGRESSOR: self.train_KernelRidge,
            Models.SVM_REGRESSOR: self.train_SVR,
            Models.RANDOM_FOREST_REGRESSOR: self.train_RandomForest,
        }

    def train_SVR(self, model_name):
        print("Start hyperparameter tunning for SVR")
        reg = _with_scaler(SVR())

        random_param_dist = {
            "model__C": expon(scale=300),
            "model__epsilon": expon(scale=1),
            "model__gamma": expon(scale=0.01),
            "model__kernel": ["rbf"],
        }

        random_search = RandomizedSearchCV(
            reg,
            param_distributions=random_param_dist,
            scoring=SCORES,
            refit="R2",
            n_iter=100,
            cv=FOLDS,
            random_state=SEED,
            n_jobs=-1,
        )

        random_search.fit(self.X, self.y)

        print("SVR RandomizedSearchCV results:")
        return _process_cv_results(random_search, model_name)

    def train_LinearRegression(self, model_name):
        print("Start training Linear Regressor")
        scores = {"R2": [], "MAPE": [], "RMSE": [], "MAE": []}
        model = None
        best_score = -1e10
        splits = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        for train_index, test_index in splits.split(np.zeros_like(self.y), self.y):
            reg = _with_scaler(LinearRegression()).fit(self.X.iloc[train_index], self.y.iloc[train_index])
            scores["R2"].append(reg.score(self.X.iloc[test_index], self.y.iloc[test_index]))
            y_true = self.y.iloc[test_index]
            y_pred = reg.predict(self.X.iloc[test_index])
            scores["MAPE"].append(mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred))
            scores["MAE"].append(mean_absolute_error(y_true=y_true, y_pred=y_pred))
            scores["RMSE"].append(root_mean_squared_error(y_true=y_true, y_pred=y_pred))

            if scores["R2"][-1] > best_score:
                best_score = scores["R2"][-1]
                model = reg

        print("Linear regression scores:")
        print(f"Mean validation r2: {np.mean(scores['R2']):.3f} (std: {np.std(scores['R2']):.3f})")
        print(f"Mean validation MAPE: {np.mean(scores['MAPE']):.3f} (std: {np.std(scores['MAPE']):.3f})")
        print(f"Mean validation RMSE: {np.mean(scores['RMSE']):.3f} (std: {np.std(scores['RMSE']):.3f})")
        print(f"Mean validation MAE: {np.mean(scores['MAE']):.3f} (std: {np.std(scores['MAE']):.3f})")
        print("")

        with open(MODELS_DIR / f"{model_name}.joblib", "wb") as f:
            joblib.dump(model, f)

        return {
            "model_name": model_name,
            "model": model,
            "r2": np.mean(scores["R2"]),
            "mape": np.mean(scores["MAPE"]),
            "rmse": np.mean(scores["RMSE"]),
            "mae": np.mean(scores["MAE"]),
        }

    def train_RidgeRegression(self, model_name):
        print("Start hyperparameter tunning for Ridge Regression")
        reg = _with_scaler(Ridge())
        random_param_dist = {
            "model__alpha": expon(scale=0.1),
        }
        random_search = RandomizedSearchCV(
            reg,
            param_distributions=random_param_dist,
            scoring=SCORES,
            refit="R2",
            n_iter=100,
            cv=FOLDS,
            random_state=SEED,
            n_jobs=-1,
        )
        random_search.fit(self.X, self.y)

        print("Ridge RandomizedSearchCV results:")
        return _process_cv_results(random_search, model_name)

    def train_KernelRidge(self, model_name):
        print("Start hyperparameter tunning for Kernel Ridge Regression")
        reg = _with_scaler(KernelRidge())
        random_param_dist = {
            "model__alpha": expon(scale=0.01),
            "model__gamma": expon(scale=0.01),
            "model__kernel": ["rbf"],
        }
        random_search = RandomizedSearchCV(
            reg,
            param_distributions=random_param_dist,
            scoring=SCORES,
            refit="R2",
            n_iter=100,
            cv=FOLDS,
            random_state=SEED,
            n_jobs=-1,
            error_score="raise",
        )
        random_search.fit(self.X, self.y)

        print("Kernel Ridge RandomizedSearchCV results:")
        return _process_cv_results(random_search, model_name)

    def train_RandomForest(self, model_name):
        print("Start hyperparameter tunning for Random Forest Regression")
        reg = _with_scaler(RandomForestRegressor())
        random_param_dist = {
            "model__n_estimators": randint(100, 1000),
            "model__max_depth": randint(10, 100),
        }

        random_search = RandomizedSearchCV(
            reg,
            param_distributions=random_param_dist,
            scoring=SCORES,
            refit="R2",
            n_iter=100,
            cv=FOLDS,
            random_state=SEED,
            n_jobs=-1,
        )
        random_search.fit(self.X, self.y)

        print("Random Forest RandomizedSearchCV results:")
        return _process_cv_results(random_search, model_name)

    def run_training(self, target_metric: str, models: list[Models]):
        valid_targets = ["full-energy", "stable-energy", "power"]
        if target_metric not in valid_targets:
            raise ValueError(f"Target metric {target_metric} is not valid. Please use one of {valid_targets}")

        print(f"Training data shape: {self.X.shape}")
        models_results = {"model_name": [], "model": [], "r2": [], "mape": [], "rmse": [], "mae": []}

        for model in models:
            results = self._MODELS[model](f"{model.value}-{target_metric}-estimator")
            for key, value in results.items():
                models_results[key].append(value)

        return models_results


def train_target(train: pd.DataFrame, features: list, target_column: str, target_metric: str, models: list[Models]):
    """
    Tune and cross-validate each model for one PRE target and keep the one with the lowest cross-validated RMSE.

    The best model is also saved as ``pre-{target_metric}-best.joblib`` so that the evaluation can load it.

    Returns
    -------
    results_df : `pd.DataFrame`
        Cross-validation scores of each model.
    best_model : estimator
        The model with the lowest cross-validated RMSE, refitted on the whole training set.
    """
    results = ModelTrainer(train[features], train[target_column]).run_training(target_metric, models)
    results_df = pd.DataFrame.from_dict({key: values for key, values in results.items() if key != "model"})
    best_model = results["model"][results_df["rmse"].idxmin()]
    with open(MODELS_DIR / f"pre-{target_metric}-best.joblib", "wb") as f:
        joblib.dump(best_model, f)
    return results_df, best_model


def regression_metrics(y_true, y_pred) -> dict:
    """
    Error metrics for a set of predictions.

    ``R2`` is the coefficient of determination, 1 - SSE/SST, which penalizes bias; ``r2`` is the squared Pearson
    correlation between predictions and true values, which does not. ``bias`` is the mean signed error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    return {
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAE": float(np.mean(np.abs(errors))),
        "R2": float(1 - np.sum(errors**2) / np.sum((y_true - y_true.mean()) ** 2)),
        "r2": float(np.corrcoef(y_true, y_pred)[0, 1] ** 2),
        "bias": float(np.mean(errors)),
    }


def report(results, n_top=3):
    """
    Utility function to report best scores.

    Parameters
    ----------
    results : dict
        Results of a randomized search over hyperparameters.
    n_top : int, optional
        Number of results to report. Default to 3.
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_R2"] == i)
        for candidate in candidates:
            print(f"Model with rank: {i}")
            print(
                f"Mean validation r2: {results['mean_test_R2'][candidate]:.3f} (std: {results['std_test_R2'][candidate]:.3f})"
            )
            print(
                f"Mean validation MAPE: {-results['mean_test_MAPE'][candidate]:.3f} (std: {-results['std_test_MAPE'][candidate]:.3f})"
            )
            print(
                f"Mean validation RMSE: {-results['mean_test_RMSE'][candidate]:.3f} (std: {-results['std_test_RMSE'][candidate]:.3f})"
            )
            print(
                f"Mean validation MAE: {-results['mean_test_MAE'][candidate]:.3f} (std: {-results['std_test_MAE'][candidate]:.3f})"
            )
            print(f"Parameters: {results['params'][candidate]}")
            print("")


def _process_cv_results(cv_results, model_name):
    report(cv_results.cv_results_)

    with open(MODELS_DIR / f"{model_name}.joblib", "wb") as f:
        joblib.dump(cv_results.best_estimator_, f)

    return {
        "model_name": model_name,
        "model": cv_results.best_estimator_,
        "r2": cv_results.best_score_,
        "mape": -cv_results.cv_results_["mean_test_MAPE"][cv_results.best_index_],
        "rmse": -cv_results.cv_results_["mean_test_RMSE"][cv_results.best_index_],
        "mae": -cv_results.cv_results_["mean_test_MAE"][cv_results.best_index_],
    }

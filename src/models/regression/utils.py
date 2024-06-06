from enum import Enum

import joblib
import numpy as np
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
        reg = SVR()

        random_param_dist = {
            "C": expon(scale=300),
            "epsilon": expon(scale=1),
            "gamma": expon(scale=0.01),
            "kernel": ["rbf"],
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
        splits = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

        scores = {"R2": [], "MAPE": [], "RMSE": [], "MAE": []}
        model = None
        best_score = -1e10
        for train_index, test_index in splits.split(np.zeros_like(self.y), self.y):
            reg = LinearRegression().fit(self.X.iloc[train_index], self.y.iloc[train_index])
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
        reg = Ridge()
        random_param_dist = {
            "alpha": expon(scale=0.1),
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
        reg = KernelRidge()
        random_param_dist = {
            "alpha": expon(scale=0.01),
            "gamma": expon(scale=0.01),
            "kernel": ["rbf"],
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
        reg = RandomForestRegressor()
        random_param_dist = {
            "n_estimators": randint(100, 1000),
            "max_depth": randint(10, 100),
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
        if target_metric not in ["full-energy", "stable-energy", "power"]:
            raise ValueError(
                f"Target metric {target_metric} is not valid. Please use 'full-energy', 'stable-energy' or 'power'"
            )

        print(f"Training data shape: {self.X.shape}")
        models_results = {"model_name": [], "model": [], "r2": [], "mape": [], "rmse": [], "mae": []}

        for model in models:
            results = self._MODELS[model](f"{model.value}-{target_metric}-estimator")
            for key, value in results.items():
                models_results[key].append(value)

        return models_results


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

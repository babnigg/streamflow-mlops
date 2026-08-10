"""
XGBoost hyperparameter tuning with MLflow tracking.
Compares grid search, random search, and Bayesian optimization (Optuna)
on the streamflow feature matrix (DVC-tracked).

Usage:
    python -m streamflow.train
"""

import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# We use the file store deliberately (no database to stand up), which mlflow 3
# only allows behind this flag. Set in code so no one has to export it.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
import mlflow
import mlflow.xgboost
import optuna
try:
    # Newer optuna versions split integrations into a separate package:
    # pip install "optuna-integration[mlflow]"
    from optuna_integration.mlflow import MLflowCallback
except ImportError:
    # Older optuna versions bundle it in optuna.integration
    from optuna.integration.mlflow import MLflowCallback
from sklearn.model_selection import (
    TimeSeriesSplit,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import mean_absolute_error

from .config import resolve, CONFIG, MLFLOW_TRACKING_URI as _TRACKING_URI
from .features import TARGET_COLUMN, _feature_columns
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    # scikit-learn < 1.4 doesn't have root_mean_squared_error or the
    # squared=False kwarg was later removed from mean_squared_error, so
    # fall back to computing it manually for compatibility either way.
    from sklearn.metrics import mean_squared_error as _mse

    def root_mean_squared_error(y_true, y_pred):
        return _mse(y_true, y_pred) ** 0.5


def nash_sutcliffe_efficiency(y_true, y_pred):
    """NSE = 1 - SS_res/SS_tot: standard hydrology goodness-of-fit metric.
    NSE=1 is a perfect model, NSE=0 means no better than predicting the
    observed mean, NSE<0 means worse than the mean. Numerically identical
    to sklearn's r2_score, computed explicitly here for the domain framing
    (config.yaml lists "nse" as this project's target metric)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


PERSISTENCE_COLUMN = "log_streamflow_t"   # today's flow, same space as the target


def persistence_index(y_true, y_pred, y_persistence):
    """PI = 1 - SS_res/SS_persistence (Kitanidis & Bras 1980).

    NSE scores against the observed mean, a bar nobody would forecast with, so
    on an autocorrelated series it flatters the model: here persistence alone
    scores NSE 0.86. PI scores against persistence itself, so 0 means "adds
    nothing over doing nothing" and negative means actively worse - a threshold
    with a decision attached, which is what monitoring needs.
    """
    y_true = np.asarray(y_true)
    ss_res = np.sum((y_true - np.asarray(y_pred)) ** 2)
    ss_per = np.sum((y_true - np.asarray(y_persistence)) ** 2)
    if ss_per == 0:                       # flat window: persistence is perfect
        return float("nan")
    return 1 - ss_res / ss_per


def persistence_index_from_features(y_true, y_pred, X):
    """PI using today's flow out of the feature frame, NaN if it isn't there.

    Returns NaN rather than raising so a feature set without the persistence
    column still trains - PI is a reporting/monitoring metric, not a gate on
    whether a model can be fit.
    """
    if PERSISTENCE_COLUMN not in getattr(X, "columns", []):
        return float("nan")
    return persistence_index(y_true, y_pred, X[PERSISTENCE_COLUMN])

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURE_PARQUET = CONFIG["data"]["features_path"]
N_TRIALS = 40              # trial budget — kept equal across methods for a fair comparison
CV_FOLDS = 5
RANDOM_STATE = 42
MLFLOW_EXPERIMENT_PARENT = CONFIG["mlflow"]["experiment_tune"]
# Pinned explicitly: mlflow's default-URI heuristic can silently pick an empty
# sqlite database over the file store that actually holds our runs.
MLFLOW_TRACKING_URI = _TRACKING_URI
TEST_DAYS = CONFIG["split"]["test_days"]      # most recent days held out as final test set
PURGE_DAYS = CONFIG["split"]["purge_days"]    # embargo between train/test
BEST_PARAMS_PATH = CONFIG["mlflow"]["best_params_path"]


# ---------------------------------------------------------------------------
# 0. Data
# ---------------------------------------------------------------------------
def ensure_data_pulled():
    """Make sure the feature parquet is materialized before we try to read it.

    There's no DVC remote configured for this repo, so `dvc pull` has
    nothing to pull from -- it always fails here. Instead, run the actual
    ingest + feature-engineering pipeline if the file doesn't exist yet.
    """
    if resolve(FEATURE_PARQUET).exists():
        return
    print("Feature matrix not found -- running ingest + feature engineering...")
    from . import ingest, features

    ingest.build_and_save()
    features.build_and_save()


def load_data(test_days: int = TEST_DAYS, purge_days: int = PURGE_DAYS):
    """Chronological train/test split.

    This is a daily time series with lag/rolling-window features, so a random
    split would scatter autocorrelated rows across both sides and inflate
    validation scores. Instead we sort by date, hold out the most recent
    `test_days` days as the test set, and drop a `purge_days` embargo
    immediately before it so no train row sits inside the longest rolling
    window's reach of the test period.

    The hold-out is a fixed number of days, not a fraction: 15% of an 82-year
    record is a 12-year hold-out, which would leave the deployed model blind to
    every recent year and make retraining unable to advance the training window.

    Note: `feature_columns` is recomputed via `_feature_columns` rather than
    read from `features.attrs["feature_columns"]` — DataFrame `.attrs` does
    not survive a parquet round-trip, so it's empty after `pd.read_parquet`.
    """
    features = pd.read_parquet(resolve(FEATURE_PARQUET))
    features = features.sort_values("date").reset_index(drop=True)
    feature_columns = _feature_columns(features)

    n_test = max(1, min(int(test_days), len(features) - 1))
    test_start = len(features) - n_test
    train_end = max(0, test_start - purge_days)

    train_df = features.iloc[:train_end]
    test_df = features.iloc[test_start:]

    X_train, y_train = train_df[feature_columns], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[feature_columns], test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test, feature_columns


# ---------------------------------------------------------------------------
# Pure helper — best-score-so-far, used for convergence curves
# ---------------------------------------------------------------------------
def compute_convergence(scores):
    """Given a list of per-trial scores (lower is better, e.g. RMSE), return
    the running best-so-far after each trial. Pure function, no I/O — this is
    what powers the convergence plots and is easy to unit test in isolation."""
    if not scores:
        return []
    convergence = []
    best_so_far = np.inf
    for s in scores:
        best_so_far = min(best_so_far, s)
        convergence.append(best_so_far)
    return convergence


# ---------------------------------------------------------------------------
# Shared eval helper
# ---------------------------------------------------------------------------
def eval_and_log_final_model(params, X_train, y_train, X_test, y_test, run_name, extra_tags=None):
    """Fit final model with best params on full train set, log to MLflow, return test RMSE/MAE/NSE/PI."""
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(params)
        if extra_tags:
            mlflow.set_tags(extra_tags)

        model = xgb.XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        nse = nash_sutcliffe_efficiency(y_test, preds)
        pi = persistence_index_from_features(y_test, preds, X_test)

        metrics = {"test_rmse": rmse, "test_mae": mae, "test_nse": nse}
        if np.isfinite(pi):
            metrics["test_pi"] = pi
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, name="model")

        return rmse, mae, nse, pi


# ---------------------------------------------------------------------------
# 1. Grid search
# ---------------------------------------------------------------------------
def run_grid_search(X_train, y_train, X_test, y_test, param_grid=None, base_params=None, cv_folds=None):
    # Restricted grid — full 6-param grid explodes combinatorially, so we
    # fix learning_rate/n_estimators and grid over the params that interact
    # most (max_depth + min_child_weight), per the redundancy discussion.
    # param_grid/base_params/cv_folds are overridable so tests can pass a
    # tiny grid instead of the full 108-combination default.
    if param_grid is None:
        param_grid = {
            "max_depth": [3, 5, 7, 9],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
        }
    if base_params is None:
        base_params = {"learning_rate": 0.05, "n_estimators": 300, "reg_lambda": 1.0}
    if cv_folds is None:
        cv_folds = CV_FOLDS

    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **base_params,
    )

    with mlflow.start_run(run_name="grid_search_parent") as parent_run:
        mlflow.set_tag("tuning_method", "grid_search")
        mlflow.log_param("grid_size", np.prod([len(v) for v in param_grid.values()]))

        start = time.time()
        search = GridSearchCV(
            base_model,
            param_grid,
            scoring="neg_root_mean_squared_error",
            cv=TimeSeriesSplit(n_splits=cv_folds),
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        elapsed = time.time() - start

        # Log every trial as a child run so it shows up in the MLflow UI
        results = search.cv_results_
        rmses = [-s for s in results["mean_test_score"]]
        convergence = compute_convergence(rmses)
        for i, (params, rmse) in enumerate(zip(results["params"], rmses)):
            with mlflow.start_run(run_name=f"grid_trial_{i}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_rmse", rmse)

        mlflow.log_metric("search_time_sec", elapsed)
        mlflow.log_metric("n_trials", len(results["params"]))
        mlflow.log_metric("best_cv_rmse", -search.best_score_)

        best_params = {**search.best_params_, **base_params}
        test_rmse, mae, nse, pi = eval_and_log_final_model(
            best_params, X_train, y_train, X_test, y_test,
            run_name="grid_search_best_model",
        )
        mlflow.log_metric("final_test_rmse", test_rmse)

        print(f"[Grid Search] best CV RMSE={-search.best_score_:.4f}, test RMSE={test_rmse:.4f}, time={elapsed:.1f}s")
        return {
            "method": "grid_search",
            "best_params": best_params,
            "best_cv_rmse": -search.best_score_,
            "test_rmse": test_rmse,
            "test_nse": nse,
            "test_pi": pi,
            "time_sec": elapsed,
            "n_trials": len(results["params"]),
            "convergence": convergence,
        }


# ---------------------------------------------------------------------------
# 2. Random search
# ---------------------------------------------------------------------------
def run_random_search(X_train, y_train, X_test, y_test, param_distributions=None, n_iter=None, cv_folds=None):
    if param_distributions is None:
        param_distributions = {
            "max_depth": np.arange(3, 11),
            "min_child_weight": np.arange(1, 8),
            "subsample": np.linspace(0.6, 1.0, 9),
            "colsample_bytree": np.linspace(0.6, 1.0, 9),
            "reg_lambda": np.logspace(-2, 2, 20),
            "learning_rate": np.logspace(-2.3, -0.7, 15),  # ~0.005 to ~0.2
        }
    if n_iter is None:
        n_iter = N_TRIALS
    if cv_folds is None:
        cv_folds = CV_FOLDS

    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="random_search_parent"):
        mlflow.set_tag("tuning_method", "random_search")
        mlflow.log_param("n_trials", n_iter)

        start = time.time()
        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=TimeSeriesSplit(n_splits=cv_folds),
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )
        search.fit(X_train, y_train)
        elapsed = time.time() - start

        results = search.cv_results_
        rmses = [-s for s in results["mean_test_score"]]
        convergence = compute_convergence(rmses)
        for i, (params, rmse) in enumerate(zip(results["params"], rmses)):
            with mlflow.start_run(run_name=f"random_trial_{i}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_rmse", rmse)

        mlflow.log_metric("search_time_sec", elapsed)
        mlflow.log_metric("best_cv_rmse", -search.best_score_)

        best_params = {**search.best_params_, "n_estimators": 300}
        test_rmse, mae, nse, pi = eval_and_log_final_model(
            best_params, X_train, y_train, X_test, y_test,
            run_name="random_search_best_model",
        )
        mlflow.log_metric("final_test_rmse", test_rmse)

        print(f"[Random Search] best CV RMSE={-search.best_score_:.4f}, test RMSE={test_rmse:.4f}, time={elapsed:.1f}s")
        return {
            "method": "random_search",
            "best_params": best_params,
            "best_cv_rmse": -search.best_score_,
            "test_rmse": test_rmse,
            "test_nse": nse,
            "test_pi": pi,
            "time_sec": elapsed,
            "n_trials": n_iter,
            "convergence": convergence,
        }


# ---------------------------------------------------------------------------
# 3. Bayesian optimization (Optuna)
# ---------------------------------------------------------------------------
def run_bayesian_search(X_train, y_train, X_test, y_test, n_trials=None, cv_folds=None):
    from sklearn.model_selection import cross_val_score

    if n_trials is None:
        n_trials = N_TRIALS
    if cv_folds is None:
        cv_folds = CV_FOLDS

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 1e2, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 2e-1, log=True),
        }
        model = xgb.XGBRegressor(
            **params,
            objective="reg:squarederror",
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        scores = cross_val_score(
            model, X_train, y_train,
            scoring="neg_root_mean_squared_error",
            cv=TimeSeriesSplit(n_splits=cv_folds),
            n_jobs=-1,
        )
        return -scores.mean()  # RMSE, minimize

    with mlflow.start_run(run_name="bayesian_search_parent"):
        mlflow.set_tag("tuning_method", "bayesian_optuna")
        mlflow.log_param("n_trials", n_trials)

        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name="cv_rmse",
            create_experiment=False,
            mlflow_kwargs={"nested": True},
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )

        start = time.time()
        study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_callback])
        elapsed = time.time() - start

        # Convergence: best-so-far RMSE after each trial
        convergence = compute_convergence([t.value for t in study.trials])

        mlflow.log_metric("search_time_sec", elapsed)
        mlflow.log_metric("best_cv_rmse", study.best_value)

        best_params = {**study.best_params, "n_estimators": 300}
        test_rmse, mae, nse, pi = eval_and_log_final_model(
            best_params, X_train, y_train, X_test, y_test,
            run_name="bayesian_search_best_model",
        )
        mlflow.log_metric("final_test_rmse", test_rmse)

        print(f"[Bayesian/Optuna] best CV RMSE={study.best_value:.4f}, test RMSE={test_rmse:.4f}, time={elapsed:.1f}s")
        return {
            "method": "bayesian_optuna",
            "best_params": best_params,
            "best_cv_rmse": study.best_value,
            "test_rmse": test_rmse,
            "test_nse": nse,
            "test_pi": pi,
            "time_sec": elapsed,
            "n_trials": n_trials,
            "convergence": convergence,
        }


# ---------------------------------------------------------------------------
# 4. Compare methods
# ---------------------------------------------------------------------------
def summarize(results):
    df = pd.DataFrame([
        {
            "method": r["method"],
            "best_cv_rmse": r["best_cv_rmse"],
            "test_rmse": r["test_rmse"],
            "test_nse": r["test_nse"],
            "test_pi": r.get("test_pi", float("nan")),
            "n_trials": r["n_trials"],
            "time_sec": r["time_sec"],
            "rmse_per_trial_efficiency": r["best_cv_rmse"] / r["n_trials"],
        }
        for r in results
    ])
    print("\n=== Tuning method comparison ===")
    print(df.to_string(index=False))

    with mlflow.start_run(run_name="method_comparison_summary"):
        for _, row in df.iterrows():
            mlflow.log_metric(f"{row['method']}_best_cv_rmse", row["best_cv_rmse"])
            mlflow.log_metric(f"{row['method']}_test_rmse", row["test_rmse"])
            mlflow.log_metric(f"{row['method']}_test_nse", row["test_nse"])
            if np.isfinite(row["test_pi"]):
                mlflow.log_metric(f"{row['method']}_test_pi", row["test_pi"])
            mlflow.log_metric(f"{row['method']}_time_sec", row["time_sec"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning_comparison.csv"
            df.to_csv(path, index=False)
            mlflow.log_artifact(str(path))

    return df


# ---------------------------------------------------------------------------
# 5. Save best hyperparameters for deployment
# ---------------------------------------------------------------------------
def _to_native(value):
    """yaml.safe_dump can't serialize numpy scalar types, which show up in
    best_params from the np.arange/np.linspace grids used in random search."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def save_best_params(results, path: str = BEST_PARAMS_PATH):
    """Pick the result with the lowest CV RMSE across methods and write its
    hyperparameters to a config file. streamflow.train reads this file to
    retrain the deploy model without re-running the search."""
    best = min(results, key=lambda r: r["best_cv_rmse"])

    output = resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": best["method"],
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "best_cv_rmse": float(best["best_cv_rmse"]),
        "test_rmse": float(best["test_rmse"]),
        "params": {k: _to_native(v) for k, v in best["best_params"].items()},
    }
    with open(output, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    print(f"\nSaved best hyperparameters ({best['method']}, cv_rmse={best['best_cv_rmse']:.4f}) -> {output}")
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_PARENT)

    ensure_data_pulled()
    X_train, X_test, y_train, y_test, feature_columns = load_data()
    print(f"Loaded {len(feature_columns)} features, train={len(X_train)}, test={len(X_test)}")

    grid_result = run_grid_search(X_train, y_train, X_test, y_test)
    random_result = run_random_search(X_train, y_train, X_test, y_test)
    bayes_result = run_bayesian_search(X_train, y_train, X_test, y_test)

    results = [grid_result, random_result, bayes_result]
    summarize(results)
    save_best_params(results)

    print("\nRun `mlflow ui` and open the streamflow_xgboost_tuning experiment to compare runs.")


if __name__ == "__main__":
    main()

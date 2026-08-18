"""Train the deployment XGBoost model.

Uses the hyperparameters streamflow.tune already found to be best
(config/best_params.yaml) -- no search here, so this runs fast. Run
streamflow.tune first (or whenever you want to re-tune); run this every time
you just want to retrain on fresh data with the existing hyperparameters.

Note: this imports shared helpers from .tune, so optuna must be installed
even though no search runs here.

Usage:
    python -m streamflow.train
"""
import argparse

import numpy as np
import yaml
import mlflow
import mlflow.xgboost
import xgboost as xgb
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error as _mse

    def root_mean_squared_error(y_true, y_pred):
        return _mse(y_true, y_pred) ** 0.5

from . import registry
from .config import resolve, EXPERIMENT_DEPLOY, MLFLOW_TRACKING_URI, REGISTERED_MODEL
from .tune import (
    BEST_PARAMS_PATH,
    RANDOM_STATE,
    load_data,
    nash_sutcliffe_efficiency,
    persistence_index_from_features,
)

MLFLOW_EXPERIMENT = EXPERIMENT_DEPLOY


def load_best_params(path: str = BEST_PARAMS_PATH) -> dict:
    with open(resolve(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(params_path: str = BEST_PARAMS_PATH):
    config = load_best_params(params_path)
    params = config["params"]
    print(f"Training with hyperparameters from {config['method']} (selected {config['selected_at']})")

    X_train, X_test, y_train, y_test, feature_columns = load_data()
    print(f"Loaded {len(feature_columns)} features, train={len(X_train)}, test={len(X_test)}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # scored before the candidate is trained, so both are measured on the same
    # held-out window - the champion's own recorded metrics describe whatever
    # split existed when it was fitted
    champion = registry.load_champion()
    champion_scores = registry.evaluate(champion, X_test, y_test) if champion else None

    with mlflow.start_run(run_name="deploy_train") as run:
        mlflow.log_params(params)
        mlflow.log_params({"n_train": len(X_train), "n_test": len(X_test),
                           "n_features": len(feature_columns)})
        mlflow.set_tag("tuning_method", config["method"])
        # which file the hyperparameters came from - two runs with different
        # params are otherwise indistinguishable in the run record
        mlflow.log_param("params_source", params_path)

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

        mlflow.log_metrics({"test_rmse": rmse, "test_mae": mae,
                            "test_nse": nse, "test_pi": pi})
        info = mlflow.xgboost.log_model(
            model, name="model",
            signature=infer_signature(X_test, preds),
            input_example=X_test.head(3),
            registered_model_name=REGISTERED_MODEL,
        )

        # PI is the one to watch: negative means persistence would beat us.
        print(f"test RMSE={rmse:.4f}, MAE={mae:.4f}, NSE={nse:.4f}, PI={pi:.4f}")

        candidate = {"rmse": rmse, "mae": mae, "nse": nse, "pi": pi}
        verdict = registry.compare(candidate, champion_scores)
        mlflow.set_tag("promoted", str(verdict["promote"]).lower())
        mlflow.set_tag("promotion_reason", verdict["reason"])
        if champion_scores:
            mlflow.log_metrics({f"champion_{k}": v for k, v in champion_scores.items()
                                if v is not None and np.isfinite(v)})

        version = info.registered_model_version
        if verdict["promote"]:
            registry.promote(version)

        print(f"{'PROMOTED' if verdict['promote'] else 'REJECTED'} v{version}: {verdict['reason']}")
        if not verdict["promote"]:
            print(f"  champion v{registry.champion_version()} stays deployed")

        run_id = run.info.run_id

    return model, run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=BEST_PARAMS_PATH,
                    help="hyperparameter file (default: the tuned best_params.yaml)")
    main(ap.parse_args().params)

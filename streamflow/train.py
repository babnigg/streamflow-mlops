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
import yaml
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error as _mse

    def root_mean_squared_error(y_true, y_pred):
        return _mse(y_true, y_pred) ** 0.5

from .config import resolve, EXPERIMENT_DEPLOY, MLFLOW_TRACKING_URI
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


def main():
    config = load_best_params()
    params = config["params"]
    print(f"Training with hyperparameters from {config['method']} (selected {config['selected_at']})")

    X_train, X_test, y_train, y_test, feature_columns = load_data()
    print(f"Loaded {len(feature_columns)} features, train={len(X_train)}, test={len(X_test)}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="deploy_train") as run:
        mlflow.log_params(params)
        mlflow.set_tag("tuning_method", config["method"])

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
        mlflow.xgboost.log_model(model, name="model")

        # PI is the one to watch: negative means persistence would beat us.
        print(f"test RMSE={rmse:.4f}, MAE={mae:.4f}, NSE={nse:.4f}, PI={pi:.4f}")

        run_id = run.info.run_id

    return model, run_id


if __name__ == "__main__":
    main()

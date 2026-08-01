"""Train the deployment XGBoost model.

Uses the hyperparameters streamflow.tune already found to be best
(config/best_params.yaml) -- no grid/random/Bayesian search here, so this
runs fast and doesn't depend on optuna. Run streamflow.tune first (or
whenever you want to re-tune); run this every time you just want to
retrain on fresh data with the existing hyperparameters.

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

from .config import resolve
from .tune import load_data, nash_sutcliffe_efficiency, RANDOM_STATE, BEST_PARAMS_PATH, MLFLOW_TRACKING_URI

MLFLOW_EXPERIMENT = "streamflow_xgboost_deploy"


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

        mlflow.log_metrics({"test_rmse": rmse, "test_mae": mae, "test_nse": nse})
        mlflow.xgboost.log_model(model, name="model")

        print(f"test RMSE={rmse:.4f}, MAE={mae:.4f}, NSE={nse:.4f}")

        run_id = run.info.run_id

    return model, run_id


if __name__ == "__main__":
    main()

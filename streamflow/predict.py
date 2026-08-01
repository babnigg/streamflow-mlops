"""Generate a next-day streamflow prediction using the current deployed model.

Loads the most recently trained model from the streamflow_xgboost_deploy
MLflow experiment (the one streamflow.train writes to) and scores it against
the most recent row of fresh feature data -- the row that has full lag/rolling
history but no next-day target yet, since tomorrow hasn't happened.

Does NOT retrain or re-tune. Run streamflow.train (on demand) to refit
weights on fresh data with existing hyperparameters, or streamflow.tune to
re-search hyperparameters.

Usage:
    python -m streamflow.predict
"""
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost

from .config import resolve, CONFIG
from .features import build_features, TARGET_COLUMN, _feature_columns
from .tune import MLFLOW_TRACKING_URI

MLFLOW_EXPERIMENT = "streamflow_xgboost_deploy"


def load_latest_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        raise RuntimeError(
            f"No MLflow experiment '{MLFLOW_EXPERIMENT}' found -- "
            "run `python -m streamflow.train` first"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"No runs found in experiment '{MLFLOW_EXPERIMENT}' -- "
            "run `python -m streamflow.train` first"
        )

    run = runs[0]
    model = mlflow.xgboost.load_model(f"runs:/{run.info.run_id}/model")
    return model, run.info.run_id


def latest_feature_row(raw_df: pd.DataFrame) -> pd.DataFrame:
    features = build_features(raw_df, drop_incomplete=False)
    undetermined = features[features[TARGET_COLUMN].isna()]
    if undetermined.empty:
        raise RuntimeError("No row with an undetermined target found -- check input data recency")
    return undetermined.iloc[[-1]]


def predict_next_day(raw_df: pd.DataFrame = None) -> dict:
    if raw_df is None:
        raw_df = pd.read_parquet(resolve(CONFIG["data"]["raw_path"]))

    row = latest_feature_row(raw_df)
    feature_columns = _feature_columns(row)

    model, run_id = load_latest_model()
    pred_log = model.predict(row[feature_columns])[0]
    pred_cfs = float(np.expm1(pred_log))

    result = {
        "as_of_date": str(row["date"].iloc[0].date()),
        "predicted_streamflow_cfs": pred_cfs,
        "model_run_id": run_id,
    }
    print(
        f"prediction as of {result['as_of_date']}: "
        f"{pred_cfs:.1f} cfs (model run {run_id})"
    )
    return result


def main():
    return predict_next_day()


if __name__ == "__main__":
    main()

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
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost

from .config import resolve, CONFIG, MLFLOW_TRACKING_URI, EXPERIMENT_DEPLOY
from .features import build_features, TARGET_COLUMN, _feature_columns

MLFLOW_EXPERIMENT = EXPERIMENT_DEPLOY


def load_latest_model():
    """Fetch the most recently logged model from the deploy experiment.

    Raises RuntimeError if no run exists yet -- streamflow.train must be run
    at least once before predictions are possible.
    """
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
    model = _load_model_for_run(client, experiment, run)
    return model, run.info.run_id


def _load_model_for_run(client, experiment, run):
    """Load the model logged by `run`, without trusting MLflow's recorded paths.

    MLflow records an absolute artifact path, so a store written on one machine
    is unreadable from another or from inside a container. For a local store we
    rebuild the path from the root we are pointed at; otherwise fall back to the
    usual `runs:/` lookup.
    """
    root = MLFLOW_TRACKING_URI
    if root.startswith("file:"):
        root = url2pathname(urlparse(root).path)
        try:
            logged = client.search_logged_models(experiment_ids=[experiment.experiment_id])
        except Exception:
            logged = []
        for m in logged:
            if getattr(m, "source_run_id", None) not in (None, run.info.run_id):
                continue
            local = Path(root) / experiment.experiment_id / "models" / m.model_id / "artifacts"
            if (local / "MLmodel").exists():
                return mlflow.xgboost.load_model(str(local))

    return mlflow.xgboost.load_model(f"runs:/{run.info.run_id}/model")


def latest_feature_row(raw_df: pd.DataFrame) -> pd.DataFrame:
    """The single most recent row with full lag/rolling history but no target yet.

    build_features with drop_incomplete=False keeps the final row (which has
    no next-day target, since it hasn't happened) instead of dropping it --
    that row is exactly what we want to score.
    """
    features = build_features(raw_df, drop_incomplete=False)
    undetermined = features[features[TARGET_COLUMN].isna()]
    if undetermined.empty:
        raise RuntimeError("No row with an undetermined target found -- check input data recency")
    return undetermined.iloc[[-1]]


def predict_next_day(raw_df: pd.DataFrame = None) -> dict:
    """Predict next-day streamflow (cfs) from the latest deployed model.

    raw_df: raw ingested table (as produced by streamflow.ingest). If None,
    reads the current raw parquet from disk via CONFIG.
    """
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
    result = predict_next_day()
    return result


if __name__ == "__main__":
    main()

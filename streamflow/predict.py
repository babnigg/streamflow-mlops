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

from . import registry
from .config import (resolve, CONFIG, MLFLOW_TRACKING_URI, EXPERIMENT_DEPLOY,
                     REGISTERED_MODEL, CHAMPION_ALIAS)
from .features import build_features, TARGET_COLUMN, _feature_columns

MLFLOW_EXPERIMENT = EXPERIMENT_DEPLOY


def load_latest_model():
    """Load the model carrying the champion alias.

    Serving follows the alias rather than the newest run: a retrain that loses
    its promotion comparison stays registered and unserved, and a bad promotion
    is undone by moving the alias back (streamflow.registry.rollback) instead of
    by retraining.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = registry.load_champion()
    if model is None:
        # An unreachable or unauthenticated tracking server looks identical to an
        # empty registry from here, and the two want opposite responses: retrain
        # vs fix the connection. Name both rather than send the operator to
        # retrain against a server that will refuse that too.
        raise RuntimeError(
            f"No '{CHAMPION_ALIAS}' model resolved for '{REGISTERED_MODEL}' at "
            f"{MLFLOW_TRACKING_URI} -- either nothing has been promoted yet "
            "(run `python -m streamflow.train`) or the tracking server is "
            "unreachable/unauthenticated"
        )
    version = registry.champion_version()
    run_id = mlflow.tracking.MlflowClient().get_model_version(
        REGISTERED_MODEL, version).run_id
    return model, run_id


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

    # XGBoost routes NaN down a default branch rather than failing, so an
    # incomplete history would return a plausible number instead of an error
    missing = row[feature_columns].columns[row[feature_columns].isna().any()].tolist()
    if missing:
        raise RuntimeError(f"incomplete feature history, missing values in: {missing}")

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

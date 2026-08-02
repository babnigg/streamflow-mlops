"""Inference API (FastAPI). Containerised by the repo Dockerfile.

Two ways to get a prediction:
  POST /predict         caller supplies the recent daily observations
  GET  /predict/latest  service reads the raw parquet it has on disk

The model is loaded once at startup from MLflow (not per request). Point
MLFLOW_TRACKING_URI at the mlruns directory; in the container that is a mount.

Run locally:  uvicorn streamflow.serve:app --reload
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import CONFIG, resolve
from .features import (
    DEFAULT_FLOW_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    TARGET_COLUMN,
    _feature_columns,
    build_features,
)

# Tracks features.py: the newest row needs the longest lag/window before it.
MIN_HISTORY_DAYS = max(*DEFAULT_FLOW_LAGS, *DEFAULT_ROLLING_WINDOWS) + 1

app = FastAPI(
    title="streamflow-mlops",
    version="0.1.0",
    description="Next-day streamflow forecast for USGS gauge 05532500 (Des Plaines River at Riverside, IL).",
)

_MODEL = None
_RUN_ID = None


def _load_model():
    """Resolve the deployed model once. Kept lazy so import never fails."""
    global _MODEL, _RUN_ID
    if _MODEL is None:
        from .predict import load_latest_model
        _MODEL, _RUN_ID = load_latest_model()
    return _MODEL, _RUN_ID


@app.on_event("startup")
def _warm():
    try:
        _load_model()
        print(f"model loaded: run {_RUN_ID}")
    except Exception as e:                       # keep serving /health either way
        print(f"model not loaded at startup: {e}")


class Observation(BaseModel):
    date: str
    streamflow_cfs: float = Field(ge=0, le=100_000)
    precip_mm: float = Field(ge=0, le=500)
    tmax_c: float = Field(ge=-60, le=60)
    tmin_c: float = Field(ge=-60, le=60)


class PredictRequest(BaseModel):
    observations: list[Observation] = Field(min_length=MIN_HISTORY_DAYS)


class Prediction(BaseModel):
    as_of_date: str
    predicted_streamflow_cfs: float
    model_run_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _MODEL is not None}


@app.get("/model")
def model_info():
    try:
        _, run_id = _load_model()
    except Exception as e:
        raise HTTPException(503, f"no model available: {e}")
    return {
        "run_id": run_id,
        "target": CONFIG["target"]["variable"],
        "min_history_days": MIN_HISTORY_DAYS,
        "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"),
    }


@app.post("/predict", response_model=Prediction)
def predict(req: PredictRequest):
    """Score a caller-supplied window of daily observations."""
    df = pd.DataFrame([o.model_dump() for o in req.observations])
    df["date"] = pd.to_datetime(df["date"])

    # Same feature code as training - one definition, so no train/serve skew.
    feats = build_features(df, drop_incomplete=False)
    row = feats.iloc[[-1]]

    try:
        model, run_id = _load_model()
    except Exception as e:
        raise HTTPException(503, f"no model available: {e}")

    cols = _feature_columns(row)
    if row[cols].isna().any().any():
        missing = row[cols].columns[row[cols].isna().any()].tolist()
        raise HTTPException(
            422,
            f"not enough history to build features (incomplete: {missing[:5]}). "
            f"send at least {MIN_HISTORY_DAYS} consecutive days.",
        )

    pred = float(np.expm1(model.predict(row[cols])[0]))
    return Prediction(
        as_of_date=str(pd.Timestamp(row["date"].iloc[0]).date()),
        predicted_streamflow_cfs=round(pred, 1),
        model_run_id=run_id,
    )


@app.get("/predict/latest", response_model=Prediction)
def predict_latest():
    """Score the newest row of the raw table this service has on disk."""
    path = resolve(CONFIG["data"]["raw_path"])
    if not path.exists():
        raise HTTPException(503, f"no raw data at {path}; mount it or run ingest")
    from .predict import predict_next_day
    try:
        result = predict_next_day(pd.read_parquet(path))
    except Exception as e:
        raise HTTPException(503, str(e))
    return Prediction(**result)


def create_app() -> FastAPI:
    return app

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
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import CONFIG, resolve
from .features import (
    DEFAULT_FLOW_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    _feature_columns,
    build_features,
)

# Tracks features.py: the newest row needs the longest lag/window before it.
MIN_HISTORY_DAYS = max(*DEFAULT_FLOW_LAGS, *DEFAULT_ROLLING_WINDOWS) + 1

_MODEL = None
_RUN_ID = None


def _load_model():
    """Resolve the deployed model once. Kept lazy so import never fails."""
    global _MODEL, _RUN_ID
    if _MODEL is None:
        from .predict import load_latest_model
        _MODEL, _RUN_ID = load_latest_model()
    return _MODEL, _RUN_ID


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # warm the model at startup but never fail here: the process must stay up so
    # /health can report why it is unhealthy
    try:
        _load_model()
        print(f"model loaded: run {_RUN_ID}")
    except Exception as e:
        print(f"model not loaded at startup: {e}")
    yield


_SITE = CONFIG["site"]
app = FastAPI(
    title="streamflow-mlops",
    version="0.1.0",
    description=f"Next-day streamflow forecast for {_SITE['usgs_site']} ({_SITE['name']}).",
    lifespan=_lifespan,
)


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
    """Liveness AND readiness. 503 without a model, so the container healthcheck
    fails instead of reporting ok on a service whose every request would 503 -
    an orchestrator must not route traffic here, and a fresh clone with no
    trained model should say so rather than look healthy."""
    try:
        _, run_id = _load_model()
    except Exception as e:
        raise HTTPException(503, f"no model loaded: {e}")
    return {"status": "ok", "model_loaded": True, "run_id": run_id}


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

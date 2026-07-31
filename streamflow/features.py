"""Feature matrix for next-day streamflow prediction.

Rows are indexed at day ``t``. Predictors use information available through day
``t`` and the target is log-transformed streamflow on day ``t + 1``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG, resolve

TARGET_COLUMN = "target_log_streamflow_next_day"
DEFAULT_FLOW_LAGS = (0, 1, 2, 3, 7, 14, 30)
DEFAULT_ROLLING_WINDOWS = (3, 7, 14, 30)


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def build_features(
    df: pd.DataFrame,
    *,
    drop_incomplete: bool = True,
    max_memory_days: int = 30,
) -> pd.DataFrame:
    """Create a first-pass feature matrix for next-day streamflow.

    Parameters
    ----------
    df:
        Raw table produced by ``streamflow.ingest``.
    drop_incomplete:
        Drop rows without enough lag history and the final row without a
        next-day target.
    max_memory_days:
        Upper bound for lag/rolling windows, matching the project idea that an
        IoT-like device may retain only recent history.
    """

    required = {"date", "streamflow_cfs", "precip_mm", "tmax_c", "tmin_c"}
    _require_columns(df, required)

    features = df.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.tz_localize(None)
    features = features.sort_values("date").reset_index(drop=True)

    for col in ("streamflow_cfs", "precip_mm", "tmax_c", "tmin_c"):
        features[col] = pd.to_numeric(features[col], errors="coerce")

    features["target_streamflow_next_day"] = features["streamflow_cfs"].shift(-1)
    features[TARGET_COLUMN] = np.log1p(features["target_streamflow_next_day"])

    lags = [lag for lag in DEFAULT_FLOW_LAGS if lag <= max_memory_days]
    for lag in lags:
        suffix = "t" if lag == 0 else f"lag_{lag}"
        shifted = features["streamflow_cfs"].shift(lag)
        features[f"streamflow_{suffix}"] = shifted
        features[f"log_streamflow_{suffix}"] = np.log1p(shifted)

    rolling_windows = [w for w in DEFAULT_ROLLING_WINDOWS if w <= max_memory_days]
    for window in rolling_windows:
        flow_roll = features["streamflow_cfs"].rolling(window=window, min_periods=window)
        precip_roll = features["precip_mm"].rolling(window=window, min_periods=window)
        features[f"streamflow_roll_mean_{window}d"] = flow_roll.mean()
        features[f"streamflow_roll_std_{window}d"] = flow_roll.std()
        features[f"streamflow_roll_min_{window}d"] = flow_roll.min()
        features[f"streamflow_roll_max_{window}d"] = flow_roll.max()
        features[f"precip_sum_{window}d"] = precip_roll.sum()
        features[f"precip_mean_{window}d"] = precip_roll.mean()

    features["temp_mean_c"] = (features["tmax_c"] + features["tmin_c"]) / 2
    features["temp_range_c"] = features["tmax_c"] - features["tmin_c"]
    for window in (7, 30):
        if window <= max_memory_days:
            features[f"temp_mean_roll_{window}d"] = (
                features["temp_mean_c"].rolling(window=window, min_periods=window).mean()
            )

    day_of_year = features["date"].dt.dayofyear
    features["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    features["month"] = features["date"].dt.month

    flood_threshold = features["streamflow_cfs"].quantile(0.99)
    features["is_high_flow_anomaly"] = (features["streamflow_cfs"] >= flood_threshold).astype(int)
    features.attrs["high_flow_threshold_cfs"] = float(flood_threshold)

    if drop_incomplete:
        features = features.dropna(subset=_model_columns(features)).reset_index(drop=True)

    feature_columns = _feature_columns(features)
    features.attrs["feature_columns"] = feature_columns
    features.attrs["target_column"] = TARGET_COLUMN

    return features


def _model_columns(df: pd.DataFrame) -> list[str]:
    return [TARGET_COLUMN, *_feature_columns(df)]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        "streamflow_cfs",
        "target_streamflow_next_day",
        TARGET_COLUMN,
        "approval_status",
        "qualifier",
        "last_modified",
    }
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def build_and_save(
    raw_path: str | Path | None = None,
    out_path: str | Path = "data/features/feature_matrix.parquet",
) -> pd.DataFrame:
    """Build features from the raw parquet and save a feature matrix."""

    source = Path(raw_path) if raw_path is not None else resolve(CONFIG["data"]["raw_path"])
    output = Path(out_path)
    if not output.is_absolute():
        output = resolve(str(output))

    raw = pd.read_parquet(source)
    features = build_features(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False)
    print(f"saved {len(features):,} feature rows -> {output}")
    print(f"target: {TARGET_COLUMN} | features: {len(features.attrs['feature_columns'])}")
    return features


if __name__ == "__main__":
    build_and_save()

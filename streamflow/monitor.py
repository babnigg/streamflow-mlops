"""Production monitoring: scoring, drift detection, and the retraining decision.

Covers Session 7's three pillars, with detection separated from action:

  data    quality + ranges, label-free, same day -> reject input, never retrain
  data    Evidently drift, label-free, 90d       -> evidence, not a pager
  model   persistence index, needs labels, 60d   -> retrain
  system  model age, numeric stability, latency  -> retrain / investigate

The system pillar exists because Session 7 is blunt that most production
failures are system failures, not model failures. It covers ML Test Score
Monitor 4 (staleness), 5 (numeric stability) and 6 (compute performance);
Monitor 2 and 3 are handled by validate.py and the shared feature code, and
Monitor 7 is the persistence index.

Layer 3 is not a pager on purpose. At daily cadence a 30-day PSI window fires
above 0.2 roughly 80% of the time with no drift present, so per-feature tests
here are evidence, not alarms. Hourly data would change that.

Everything is built on `score_range`, which scores any historical span. A
production run is a span of one day; the demo is a span of years. Replay is the
normal path, not a special mode.

Run:  python -m streamflow.monitor
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG, resolve
from .features import TARGET_COLUMN, _feature_columns

os.environ.setdefault("DO_NOT_TRACK", "1")   # evidently phones home by default

MON = CONFIG["monitoring"]
PERSISTENCE_COLUMN = "log_streamflow_t"      # today's flow: the naive forecast


# --------------------------------------------------------------------- scoring
def load_features() -> pd.DataFrame:
    f = pd.read_parquet(resolve(CONFIG["data"]["features_path"]))
    f["date"] = pd.to_datetime(f["date"])
    return f.sort_values("date").reset_index(drop=True)


def score_range(start=None, end=None, model=None, features=None) -> pd.DataFrame:
    """Score every day in [start, end] and return one row per day.

    The actual is already in the feature matrix (the target is next-day flow),
    so a historical span can be scored immediately - which is how the monitoring
    log gets populated without waiting for real days to elapse.
    """
    f = load_features() if features is None else features.copy()
    if start is not None:
        f = f[f["date"] >= pd.Timestamp(start)]
    if end is not None:
        f = f[f["date"] <= pd.Timestamp(end)]
    if f.empty:
        raise ValueError(f"no feature rows in range {start} .. {end}")

    if model is None:
        from .predict import load_latest_model
        model, run_id = load_latest_model()
    else:
        run_id = getattr(model, "_run_id", "supplied")

    cols = _feature_columns(f)
    pred = model.predict(f[cols])

    out = pd.DataFrame({
        "date": f["date"].values,
        "prediction": pred,
        "actual": f[TARGET_COLUMN].values,
        "persistence": f[PERSISTENCE_COLUMN].values,
        "model_run_id": run_id,
    })
    out["error"] = out["prediction"] - out["actual"]
    out["se_model"] = out["error"] ** 2
    out["se_persistence"] = (out["actual"] - out["persistence"]) ** 2
    return out


# --------------------------------------------------------------------- metrics
def persistence_index(scored: pd.DataFrame) -> float:
    """PI over a whole frame. 0 = ties persistence, <0 = worse than doing nothing."""
    ss_per = scored["se_persistence"].sum()
    if ss_per == 0:
        return float("nan")
    return float(1 - scored["se_model"].sum() / ss_per)


def rolling_pi(scored: pd.DataFrame, window_days: int | None = None) -> pd.Series:
    """Rolling PI as a ratio of rolling sums - not a mean of per-day ratios,
    which would be dominated by days when persistence happens to be perfect."""
    w = window_days or MON["pi_window_days"]
    num = scored["se_model"].rolling(w).sum()
    den = scored["se_persistence"].rolling(w).sum()
    return (1 - num / den).where(den > 0)


def calibrate_pi_threshold(scored: pd.DataFrame, window_days: int | None = None,
                           quantile: float = 0.0) -> dict:
    """Empirical floor of a healthy model's rolling PI, from NON-overlapping blocks.

    Overlapping windows are autocorrelated and make the sample look far larger
    than it is, so blocks are disjoint here. Re-run this whenever the cadence or
    target changes - at hourly resolution a 60-day window holds 24x the
    observations and the floor moves.
    """
    w = window_days or MON["pi_window_days"]
    blocks = np.arange(len(scored)) // w
    agg = scored.groupby(blocks)[["se_model", "se_persistence"]].sum()
    agg = agg[agg.index < agg.index.max()]            # drop the partial tail
    pi = (1 - agg["se_model"] / agg["se_persistence"]).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "window_days": w,
        "n_blocks": int(len(pi)),
        "median": float(pi.median()),
        "min": float(pi.min()),
        "p05": float(pi.quantile(0.05)),
        "share_below_zero": float((pi < 0).mean()),
        "suggested_threshold": float(min(0.0, pi.quantile(quantile))),
    }


# ---------------------------------------------------------- system health
def model_age_days(run_id: str) -> float:
    """Age of the deployed model (ML Test Score, Monitor 4: 'models are not too
    stale'). We already watch data freshness; a stale *model* is the failure this
    project is actually about, so it needs its own signal."""
    import mlflow
    from .config import MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        run = mlflow.tracking.MlflowClient().get_run(run_id)
    except Exception:
        return float("nan")
    started = pd.Timestamp(run.info.start_time, unit="ms", tz="UTC")
    return float((pd.Timestamp.now(tz="UTC") - started).total_seconds() / 86400)


def system_health(scored: pd.DataFrame, elapsed_s: float) -> dict:
    """The third pillar (Session 7): infrastructure and service-level signals.

    Session 7 is blunt that most production failures are system failures, not
    model failures, so these are recorded every run even when nothing is wrong -
    a slow leak in latency or a creeping model age is only visible as a trend.

    Covers ML Test Score Monitor 4 (staleness), 5 (numeric stability) and
    6 (compute performance).
    """
    run_id = scored["model_run_id"].iloc[0] if len(scored) else None
    nonfinite = int((~np.isfinite(scored["prediction"])).sum()) if len(scored) else 0
    return {
        "rows_scored": int(len(scored)),
        "duration_s": round(float(elapsed_s), 3),
        "rows_per_s": round(len(scored) / elapsed_s, 1) if elapsed_s > 0 else None,
        "nonfinite_predictions": nonfinite,
        "model_age_days": round(model_age_days(run_id), 2) if run_id else float("nan"),
    }


# ------------------------------------------------------------------ layer 1
def data_quality(raw: pd.DataFrame) -> dict:
    """Label-free checks available the moment data lands. Reuses the pipeline's
    own validation so serving and monitoring cannot disagree about what is valid."""
    from .validate import DataValidationError, validate
    try:
        validate(raw)
        return {"passed": True, "reason": None}
    except DataValidationError as e:
        return {"passed": False, "reason": str(e)}


# ------------------------------------------------------------------ layer 3
def seasonal_reference(df: pd.DataFrame, current: pd.DataFrame,
                       pad_days: int | None = None,
                       years: int | None = None) -> pd.DataFrame:
    """Same time of year, recent years - the reference a seasonal series needs.

    Two corrections, both measured on the clean window:
      season  a calendar-year baseline flags 100% of columns every summer purely
              because of the season. Matching day-of-year removes the calendar.
      recency an 82-year baseline spans decades of different river regime, which
              reads as drift (streamflow PSI 0.97 unlimited vs 0.14 over 10y).
    With both applied, an undisturbed window sits below the drift threshold on
    every watched column, which is what makes baseline validation meaningful.
    """
    pad = MON["reference_pad_days"] if pad_days is None else pad_days
    yrs = MON["reference_years"] if years is None else years

    lo, hi = current["date"].min(), current["date"].max()
    span = set()
    for d in pd.date_range(lo, hi, freq="D").dayofyear:
        span |= {((d - 1 + k) % 365) + 1 for k in range(-pad, pad + 1)}

    prior = df[df["date"] < lo]
    if yrs:
        prior = prior[prior["date"] >= lo - pd.DateOffset(years=yrs)]
    return prior[prior["date"].dt.dayofyear.isin(span)]


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 html_path: str | Path | None = None) -> dict:
    """Evidently data-drift + regression report over the watched columns.

    Watched columns are the raw drivers plus the prediction, not all 48 features.
    Returns a summary dict; writes the HTML artifact when a path is given.
    """
    from evidently import Dataset, DataDefinition, Regression, Report
    from evidently.presets import DataDriftPreset, RegressionPreset
    from evidently.metrics import ValueDrift

    watch = [c for c in MON["watch_columns"] if c in reference.columns and c in current.columns]
    numeric = watch + ["target", "prediction"]

    schema = DataDefinition(
        numerical_columns=numeric,
        regression=[Regression(target="target", prediction="prediction")],
    )
    keep = numeric
    ref_ds = Dataset.from_pandas(reference[keep].copy(), data_definition=schema)
    cur_ds = Dataset.from_pandas(current[keep].copy(), data_definition=schema)

    per_col = {c: MON["psi_thresholds"][c] for c in watch if c in MON.get("psi_thresholds", {})}

    report = Report(
        [
            DataDriftPreset(columns=numeric, drift_share=MON["drift_share"],
                            num_method="psi", num_threshold=MON["psi_threshold"],
                            per_column_threshold=per_col or None),
            RegressionPreset(),
        ] + [ValueDrift(column=c, method="psi",
                        threshold=per_col.get(c, MON["psi_threshold"])) for c in watch],
        include_tests=True,
    ).run(cur_ds, ref_ds)          # current first

    if html_path:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(html_path))

    return _summarize_drift(report, watch)


def _summarize_drift(report, watch: list[str]) -> dict:
    d = report.dict()
    per_column, share = {}, None
    for m in d.get("metrics", []):
        name, value = m.get("metric_name", ""), m.get("value")
        if name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            share = float(value.get("share", np.nan))
        elif name.startswith("ValueDrift"):
            col = (m.get("config") or {}).get("column")
            if col in watch and isinstance(value, (int, float)):
                per_column[col] = float(value)

    failed = [t for t in d.get("tests", [])
              if str(getattr(t.get("status"), "value", t.get("status"))) != "SUCCESS"]
    return {
        "drift_share": share,
        "per_column_psi": per_column,
        "n_failed_tests": len(failed),
        "dataset_drift": bool(share is not None and share >= MON["drift_share"]),
    }


# ------------------------------------------------------------------- decision
def decide(pi_value: float, quality: dict, drift: dict | None = None,
           threshold: float | None = None, system: dict | None = None) -> dict:
    """Turn signals into one action. Detection and action stay separate: only a
    sustained performance loss retrains, because retraining cannot fix a broken
    input feed and must not learn a flood as the new normal.

    Order matters. A broken feed or an unusable model is checked before
    performance, because a bad PI computed on bad inputs is not evidence about
    the model.
    """
    thr = MON["pi_alert_below"] if threshold is None else threshold
    system = system or {}

    if not quality.get("passed", True):
        status, action = "data_quality", "reject_input"
        reason = quality.get("reason")
    elif system.get("nonfinite_predictions"):
        status, action = "system", "investigate"
        reason = f"{system['nonfinite_predictions']} non-finite predictions"
    elif system.get("model_age_days", 0) and system["model_age_days"] > MON["max_model_age_days"]:
        status, action = "model_stale", "retrain"
        reason = (f"deployed model is {system['model_age_days']:.0f} days old "
                  f"(limit {MON['max_model_age_days']})")
    elif pi_value is not None and not np.isnan(pi_value) and pi_value < thr:
        status, action = "performance_drift", "retrain"
        reason = f"rolling PI {pi_value:.3f} < {thr:.3f} (worse than persistence)"
    elif drift and drift.get("dataset_drift"):
        status, action = "data_drift", "flag_only"
        reason = f"{drift['drift_share']:.0%} of watched columns drifted; performance still healthy"
    else:
        status, action = "ok", "none"
        reason = None

    return {"status": status, "action": action, "reason": reason,
            "pi": None if pi_value is None else float(pi_value),
            "threshold": float(thr), **system}


# ---------------------------------------------------------------------- logs
def _append(path, row: dict) -> pd.DataFrame:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if p.exists():
        new = pd.concat([pd.read_parquet(p), new], ignore_index=True)
    new.to_parquet(p, index=False)
    return new


def log_alert(decision: dict, at=None) -> pd.DataFrame:
    """Every decision is logged, including 'ok' - Session 7 asks for alert
    auditability, and a log with only failures cannot show a stable baseline."""
    row = {"logged_at": datetime.now(timezone.utc),
           "as_of": pd.Timestamp(at) if at is not None else pd.NaT, **decision}
    return _append(MON["alerts_path"], row)


def save_predictions(scored: pd.DataFrame) -> Path:
    p = resolve(MON["predictions_path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(p, index=False)
    return p


# ----------------------------------------------------------------------- run
def with_predictions(features: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Features plus target/prediction columns, the shape Evidently expects."""
    m = features[features["date"].isin(scored["date"])].reset_index(drop=True)
    return m.assign(target=scored["actual"].values, prediction=scored["prediction"].values)


def run(start=None, end=None, with_drift: bool = True) -> dict:
    """One monitoring pass: score, measure, decide, log.

    A production run passes a single day; the demo passes years. Same code.
    """
    import time
    t0 = time.perf_counter()
    features = load_features()
    scored = score_range(start, end, features=features)
    health = system_health(scored, time.perf_counter() - t0)
    save_predictions(scored)

    pi_series = rolling_pi(scored)
    pi_now = float(pi_series.dropna().iloc[-1]) if pi_series.notna().any() else float("nan")

    raw = pd.read_parquet(resolve(CONFIG["data"]["raw_path"]))
    quality = data_quality(raw)

    drift = None
    if with_drift and len(scored) > MON["drift_window_days"]:
        m = with_predictions(features, scored)
        cur = m[m["date"] > m["date"].max() - pd.Timedelta(days=MON["drift_window_days"])]
        ref = seasonal_reference(m, cur)
        if len(ref) >= MON["drift_window_days"]:
            stamp = pd.Timestamp(scored["date"].max()).date()
            drift = drift_report(ref, cur, Path(resolve(MON["reports_dir"])) / f"drift_{stamp}.html")

    decision = decide(pi_now, quality, drift, system=health)
    log_alert(decision, at=scored["date"].max())

    print(f"scored {len(scored):,} days in {health['duration_s']}s | model age "
          f"{health['model_age_days']:.1f}d | rolling PI ({MON['pi_window_days']}d) "
          f"{pi_now:.3f} | {decision['status']} -> {decision['action']}")
    if drift:
        print(f"  drift: {drift['drift_share']:.0%} of watched columns (evidence only)")
    if decision["reason"]:
        print(f"  {decision['reason']}")
    return decision


if __name__ == "__main__":
    run()

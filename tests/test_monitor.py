"""Monitoring tests: synthetic fixtures, no network, no real model.

These pin the behaviours the design depends on - especially that drift alone
never retrains, which is the whole point of separating detection from action.
"""

import numpy as np
import pandas as pd
import pytest

from streamflow import monitor


def _scored(n=180, seed=0, model_err=0.1, pers_err=0.3):
    """Days where the model beats persistence by construction."""
    rng = np.random.default_rng(seed)
    actual = rng.normal(5, 1, n)
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "actual": actual,
        "prediction": actual + rng.normal(0, model_err, n),
        "persistence": actual + rng.normal(0, pers_err, n),
    }).assign(
        se_model=lambda d: (d.prediction - d.actual) ** 2,
        se_persistence=lambda d: (d.actual - d.persistence) ** 2,
    )


# ----------------------------------------------------------------- metrics
def test_persistence_index_beats_persistence_when_model_is_better():
    pi = monitor.persistence_index(_scored())
    assert 0 < pi < 1


def test_persistence_index_negative_when_model_is_worse():
    s = _scored(model_err=0.9, pers_err=0.1)
    assert monitor.persistence_index(s) < 0


def test_rolling_pi_is_ratio_of_sums_not_mean_of_ratios():
    """A day where persistence is near-perfect would blow up a mean of ratios."""
    s = _scored(n=100)
    s.loc[50, "se_persistence"] = 1e-12
    r = monitor.rolling_pi(s, window_days=30).dropna()
    assert np.isfinite(r).all()
    assert (r > -10).all()


def test_rolling_pi_uses_the_full_window_and_no_partial_ones():
    """Pins the window itself: a shorter or partially-filled window would report
    a few days of noise as if it were a 60-day verdict."""
    s = _scored(n=100)
    r = monitor.rolling_pi(s, window_days=60)

    assert len(r) == len(s)                  # aligned to the input rows
    assert r.iloc[:59].isna().all()           # no value before the window fills
    assert r.notna().iloc[59:].all()
    # the first valid value must use exactly 60 rows
    expected = 1 - s["se_model"][:60].sum() / s["se_persistence"][:60].sum()
    assert np.isclose(r.iloc[59], expected)


def test_calibrate_measures_the_same_estimator_the_alarm_uses():
    """The alarm is an overlapping rolling PI, so calibration must be too -
    non-overlapping blocks evaluate one arbitrary phase and read far rosier."""
    s = _scored(n=300)
    out = monitor.calibrate_pi_threshold(s, window_days=60)

    assert out["n_windows"] == len(monitor.rolling_pi(s, 60).dropna())
    assert out["window_days"] == 60
    assert 0.0 <= out["share_below_zero"] <= 1.0
    assert -1.0 <= out["suggested_threshold"] <= 0.0   # clamped both ways


def test_calibrate_refuses_to_guess_from_too_little_data():
    """Returning a plausible threshold from an unfillable window would be
    indistinguishable from a real calibration."""
    with pytest.raises(ValueError):
        monitor.calibrate_pi_threshold(_scored(n=30), window_days=60)


# ---------------------------------------------------------------- decisions
def test_data_quality_failure_takes_precedence_and_never_retrains():
    d = monitor.decide(pi_value=-5.0, quality={"passed": False, "reason": "stale"})
    assert d["status"] == "data_quality"
    assert d["action"] == "reject_input"      # not retrain, even though PI is awful


def test_sustained_performance_loss_is_the_only_retrain_trigger():
    d = monitor.decide(-0.5, {"passed": True}, threshold=0.0)
    assert d["action"] == "retrain"


def test_flood_flags_but_never_retrains():
    """A flood is the rarest data we have; refitting on it teaches the extreme
    as normal."""
    d = monitor.decide(0.35, {"passed": True},
                       event={"is_anomaly": True, "flow": 3420.0,
                              "seasonal_p99": 3375.0, "times_p99": 1.01})
    assert d["status"] == "anomaly"
    assert d["action"] == "flag_only"


def test_drift_alone_does_not_retrain():
    """A flood drifts the inputs while the model is still fine. Retraining on it
    would teach the rarest data as the new normal."""
    d = monitor.decide(0.35, {"passed": True},
                       drift={"dataset_drift": True, "drift_share": 0.8})
    assert d["status"] == "data_drift"
    assert d["action"] == "flag_only"


def test_healthy_system_reports_ok():
    d = monitor.decide(0.35, {"passed": True}, drift={"dataset_drift": False, "drift_share": 0.1})
    assert d["status"] == "ok" and d["action"] == "none"


def test_unmeasured_pi_is_not_reported_as_healthy():
    """A span too short for the PI window says nothing about the model. Logging
    it as 'ok' would hide a monitor that never actually ran."""
    d = monitor.decide(float("nan"), {"passed": True})
    assert d["action"] == "none"
    assert d["status"] == "insufficient_history"     # not "ok"


def test_skipped_drift_is_distinguishable_from_clean_drift():
    never_ran = monitor.decide(0.35, {"passed": True}, drift=None)
    ran_clean = monitor.decide(0.35, {"passed": True},
                               drift={"dataset_drift": False, "drift_share": 0.1})
    assert never_ran["drift_evaluated"] is False
    assert ran_clean["drift_evaluated"] is True
    assert ran_clean["drift_share"] == 0.1


def test_unknown_model_age_is_investigated_not_assumed_fresh():
    """An MLflow outage must not silently switch the staleness check off."""
    d = monitor.decide(0.35, {"passed": True},
                       system={"model_age_days": float("nan"), "nonfinite_predictions": 0})
    assert d["status"] == "system" and d["action"] == "investigate"


def test_persistence_index_does_not_reward_missing_predictions():
    """Skipping NaN per-column would score the model on the days it managed to
    predict against persistence on all days - failing more would look better."""
    s = pd.DataFrame({"se_model": np.full(100, 0.04),        # 2x persistence error
                      "se_persistence": np.full(100, 0.01)})
    assert np.isclose(monitor.persistence_index(s), -3.0)

    broken = s.copy()
    broken.loc[:79, "se_model"] = np.nan          # model emits NaN on 80% of days
    # skipna per column would give +0.2 here - the sign flips and a broken model
    # reads as better than persistence
    assert monitor.persistence_index(broken) < 0


# ---------------------------------------------------------------- reference
def test_seasonal_reference_matches_time_of_year_not_recency():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", "2026-08-01", freq="D")})
    current = df[(df.date >= "2026-06-01") & (df.date <= "2026-07-31")]
    ref = monitor.seasonal_reference(df, current, pad_days=5)

    assert (ref["date"] < current["date"].min()).all()        # strictly earlier
    months = set(ref["date"].dt.month.unique())
    assert months <= {5, 6, 7, 8}                             # same season only
    assert ref["date"].dt.year.nunique() > 1                  # several prior years


# --------------------------------------------------------------------- log
def test_log_alert_appends_and_keeps_ok_rows(tmp_path, monkeypatch):
    """'ok' rows are logged too - a log of only failures cannot show a baseline."""
    monkeypatch.setitem(monitor.MON, "alerts_path", str(tmp_path / "alerts.parquet"))
    monkeypatch.setattr(monitor, "resolve", lambda p: tmp_path / "alerts.parquet")

    monitor.log_alert({"status": "ok", "action": "none", "reason": None,
                       "pi": 0.3, "threshold": 0.0})
    out = monitor.log_alert({"status": "performance_drift", "action": "retrain",
                             "reason": "x", "pi": -0.2, "threshold": 0.0})
    assert len(out) == 2
    assert set(out["status"]) == {"ok", "performance_drift"}


# ------------------------------------------------- system pillar (Session 7)
def test_system_health_reports_compute_and_staleness_signals():
    """ML Test Score Monitor 4/5/6: staleness, numeric stability, compute cost."""
    s = _scored(n=50)
    s["model_run_id"] = None
    h = monitor.system_health(s, elapsed_s=2.0)
    assert h["rows_scored"] == 50
    assert h["duration_s"] == 2.0 and h["rows_per_s"] == 25.0
    assert h["nonfinite_predictions"] == 0


def test_non_finite_predictions_are_investigated_not_retrained():
    """A model emitting NaN is broken, not stale - retraining is the wrong response."""
    d = monitor.decide(0.3, {"passed": True}, system={"nonfinite_predictions": 4})
    assert d["status"] == "system" and d["action"] == "investigate"


def test_stale_model_triggers_retrain_before_performance_degrades():
    """Monitor 4: age alone is a risk, ahead of any measured quality drop."""
    d = monitor.decide(0.35, {"passed": True},
                       system={"model_age_days": 999, "nonfinite_predictions": 0})
    assert d["status"] == "model_stale" and d["action"] == "retrain"


def test_broken_feed_outranks_every_other_signal():
    d = monitor.decide(-9.0, {"passed": False, "reason": "stale feed"},
                       drift={"dataset_drift": True, "drift_share": 1.0},
                       event={"is_anomaly": True, "flow": 9e9, "seasonal_p99": 1.0,
                              "times_p99": 9e9},
                       system={"model_age_days": 999, "nonfinite_predictions": 7})
    assert d["action"] == "reject_input"


def test_sustained_loss_outranks_a_concurrent_flood():
    """A flood during real degradation must not downgrade a retrain to a flag."""
    d = monitor.decide(-0.5, {"passed": True}, threshold=0.0,
                       event={"is_anomaly": True, "flow": 3420.0,
                              "seasonal_p99": 3375.0, "times_p99": 1.01})
    assert d["action"] == "retrain"


def test_system_signals_are_carried_into_the_alert_row():
    """Slow leaks in latency or model age are only visible as a logged trend."""
    d = monitor.decide(0.3, {"passed": True},
                       system={"model_age_days": 2.0, "nonfinite_predictions": 0,
                               "duration_s": 1.5, "rows_scored": 10})
    assert d["duration_s"] == 1.5 and d["model_age_days"] == 2.0

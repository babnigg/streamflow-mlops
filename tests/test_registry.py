"""Promotion gate tests.

The gate is what stands between "monitoring fired" and "a worse model is
serving traffic", so the comparison logic is tested on its own with plain
dicts, and the registry round-trip is tested against a real temporary store.
"""

import numpy as np
import pandas as pd
import pytest

from streamflow import registry


# ------------------------------------------------------------------ comparison
def test_first_model_is_promoted_when_there_is_no_champion():
    v = registry.compare({"pi": 0.30}, None)
    assert v["promote"] is True
    assert "no champion" in v["reason"]


def test_better_candidate_is_promoted():
    v = registry.compare({"pi": 0.35}, {"pi": 0.30})
    assert v["promote"] is True
    assert v["margin"] == pytest.approx(0.05)


def test_materially_worse_candidate_is_rejected():
    """The whole point: monitoring can fire on inputs that are valid but wrong,
    and the retrain it triggers must not reach production."""
    v = registry.compare({"pi": 0.10}, {"pi": 0.30}, tolerance=0.02)
    assert v["promote"] is False
    assert "loses to" in v["reason"]


def test_equal_candidate_ships_so_fresher_data_is_not_blocked():
    v = registry.compare({"pi": 0.30}, {"pi": 0.30}, tolerance=0.02)
    assert v["promote"] is True


def test_tolerance_is_a_boundary_not_a_suggestion():
    assert registry.compare({"pi": 0.28}, {"pi": 0.30}, tolerance=0.02)["promote"] is True
    assert registry.compare({"pi": 0.279}, {"pi": 0.30}, tolerance=0.02)["promote"] is False


def test_lower_is_better_metrics_invert_the_comparison():
    assert registry.compare({"rmse": 0.20}, {"rmse": 0.30}, metric="rmse")["promote"] is True
    assert registry.compare({"rmse": 0.40}, {"rmse": 0.30}, metric="rmse",
                            tolerance=0.02)["promote"] is False


def test_unmeasurable_candidate_never_promotes():
    """An unmeasurable model is not a better one."""
    for bad in (float("nan"), float("inf"), None):
        assert registry.compare({"pi": bad}, {"pi": 0.30})["promote"] is False


def test_unmeasurable_champion_yields_to_a_measurable_candidate():
    v = registry.compare({"pi": 0.30}, {"pi": float("nan")})
    assert v["promote"] is True


# -------------------------------------------------------------------- scoring
class _Fixed:
    """Model stub returning canned predictions."""
    def __init__(self, preds): self._p = np.asarray(preds, dtype=float)
    def predict(self, X): return self._p[:len(X)]


def _window(n=50, seed=0):
    rng = np.random.default_rng(seed)
    actual = rng.normal(5, 1, n)
    X = pd.DataFrame({"log_streamflow_t": actual + rng.normal(0, 0.3, n),
                      "other": rng.normal(0, 1, n)})
    return X, pd.Series(actual)


def test_evaluate_scores_both_models_on_the_same_window():
    X, y = _window()
    good = registry.evaluate(_Fixed(y.to_numpy() + 0.01), X, y)
    poor = registry.evaluate(_Fixed(y.to_numpy() + 0.90), X, y)

    assert good["pi"] > poor["pi"]
    assert good["rmse"] < poor["rmse"]
    assert good["pi"] < 1.0


def test_evaluate_returns_nan_pi_without_the_persistence_column():
    X, y = _window()
    out = registry.evaluate(_Fixed(y.to_numpy()), X.drop(columns=["log_streamflow_t"]), y)
    assert np.isnan(out["pi"])
    assert np.isfinite(out["rmse"])


# ----------------------------------------------------- registry round-trip
@pytest.fixture
def store(tmp_path, monkeypatch):
    """A real MLflow file store, isolated per test."""
    import mlflow
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setattr(registry, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(registry, "REGISTERED_MODEL", "test-model")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test-promotion")
    return uri


def _log_version(seed=0, promoted=True):
    """Train something trivial and register it; returns the version string.

    `promoted` writes the same tag train.py writes, so rollback sees the real
    shape of the store: rejected candidates are registered too.

    pip_requirements is pinned so MLflow skips dependency inference, which
    dominates the runtime of these tests and tells us nothing about promotion.
    """
    import mlflow
    import mlflow.xgboost
    from xgboost import XGBRegressor
    rng = np.random.default_rng(seed)
    X, y = rng.random((30, 2)), rng.random(30)
    with mlflow.start_run():
        mlflow.set_tag("promoted", str(promoted).lower())
        m = XGBRegressor(n_estimators=2).fit(X, y)
        info = mlflow.xgboost.log_model(m, name="model",
                                        registered_model_name=registry.REGISTERED_MODEL,
                                        pip_requirements=["xgboost"])
    # str: MLflow hands back an int here and a string from search, and the
    # registry normalizes to string so the two can be compared at all
    return str(info.registered_model_version)


def test_no_champion_before_anything_is_promoted(store):
    assert registry.load_champion() is None
    assert registry.champion_version() is None


def test_version_identifiers_are_strings_everywhere(store):
    """MLflow returns version as int from the alias lookup and str from search.
    Mixing them makes rollback treat the current champion as a rollback target,
    so every version this module hands out is a string."""
    v = _log_version(seed=1)
    registry.promote(v)

    assert isinstance(registry.champion_version(), str)
    assert all(isinstance(r["version"], str) for r in registry.history())
    assert registry.champion_version() == v

    registry.promote(_log_version(seed=2))
    assert isinstance(registry.rollback(), str)


def test_promote_moves_the_alias_and_serving_follows_it(store):
    v1 = _log_version(seed=1)
    registry.promote(v1)
    assert registry.champion_version() == v1
    assert registry.load_champion() is not None

    v2 = _log_version(seed=2)
    assert registry.champion_version() == v1, "registering must not deploy"

    registry.promote(v2)
    assert registry.champion_version() == v2


def test_rejected_candidate_stays_registered_but_unserved(store):
    """A losing retrain is kept for audit; it simply never becomes champion."""
    v1 = _log_version(seed=1)
    registry.promote(v1)
    v2 = _log_version(seed=2, promoted=False)   # gate said no -> no promote call

    versions = [h["version"] for h in registry.history()]
    assert v1 in versions and v2 in versions
    assert registry.champion_version() == v1


def test_rollback_returns_to_the_previous_version(store):
    v1 = _log_version(seed=1)
    registry.promote(v1)
    v2 = _log_version(seed=2)
    registry.promote(v2)
    assert registry.champion_version() == v2

    restored = registry.rollback()
    assert restored == v1
    assert registry.champion_version() == v1


def test_a_second_rollback_keeps_going_back(store):
    """`promoted` is the gate's verdict, not what is deployed, so the version a
    rollback moved away from still carries it. Without a bound below the current
    champion, rolling back twice returns to the model we just backed out of."""
    v1 = _log_version(seed=1)
    registry.promote(v1)
    v2 = _log_version(seed=2)
    registry.promote(v2)
    v3 = _log_version(seed=3)
    registry.promote(v3)

    assert registry.rollback() == v2
    assert registry.rollback() == v1


def test_rollback_skips_versions_the_gate_rejected(store):
    """Counting backwards by version number would deploy the model the gate
    just refused - rejected candidates stay registered for audit."""
    good = _log_version(seed=1)
    registry.promote(good)
    newer_good = _log_version(seed=2)
    registry.promote(newer_good)
    rejected = _log_version(seed=3, promoted=False)
    assert int(rejected) > int(good)

    restored = registry.rollback()
    assert restored == good, "rolled back onto a rejected candidate"
    assert registry.champion_version() == good


def test_rollback_refuses_when_there_is_nothing_to_fall_back_to(store):
    registry.promote(_log_version(seed=1))
    with pytest.raises(RuntimeError, match="no previously promoted version"):
        registry.rollback()


def test_rollback_refuses_when_only_rejected_candidates_exist(store):
    registry.promote(_log_version(seed=1))
    _log_version(seed=2, promoted=False)
    _log_version(seed=3, promoted=False)
    with pytest.raises(RuntimeError, match="no previously promoted version"):
        registry.rollback()

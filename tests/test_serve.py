"""API contract tests.

The health check is the one that matters operationally: it reporting "ok" while
no model was loaded is what made a fresh clone look like a working service whose
every prediction returned 503.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from streamflow import serve


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(serve, "_MODEL", None)
    monkeypatch.setattr(serve, "_RUN_ID", None)
    return TestClient(serve.app, raise_server_exceptions=False)


class _Model:
    def predict(self, X):
        return [5.0] * len(X)


def _loaded(monkeypatch):
    monkeypatch.setattr(serve, "_load_model", lambda: (_Model(), "run123"))


def _unloaded(monkeypatch):
    def boom():
        raise RuntimeError("no champion registered -- run python -m streamflow.train first")
    monkeypatch.setattr(serve, "_load_model", boom)


# --------------------------------------------------------------------- health
def test_health_is_503_when_no_model_is_loaded(client, monkeypatch):
    """A fresh clone has no model. Reporting 200 here makes an unusable service
    look healthy and lets an orchestrator route traffic to it."""
    _unloaded(monkeypatch)
    r = client.get("/health")

    assert r.status_code == 503
    assert "streamflow.train" in r.json()["detail"], "error must say how to fix it"


def test_health_is_200_with_a_model(client, monkeypatch):
    _loaded(monkeypatch)
    r = client.get("/health")

    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
    assert r.json()["run_id"] == "run123"


def test_model_endpoint_is_503_without_a_model(client, monkeypatch):
    _unloaded(monkeypatch)
    assert client.get("/model").status_code == 503


# -------------------------------------------------------------------- predict
def _obs(n, start="2026-01-01"):
    dates = pd.date_range(start, periods=n, freq="D")
    return [{"date": d.strftime("%Y-%m-%d"), "streamflow_cfs": 400.0,
             "precip_mm": 1.0, "tmax_c": 15.0, "tmin_c": 5.0} for d in dates]


def test_predict_rejects_too_short_a_window(client, monkeypatch):
    _loaded(monkeypatch)
    r = client.post("/predict", json={"observations": _obs(serve.MIN_HISTORY_DAYS - 1)})
    assert r.status_code == 422


def test_predict_rejects_out_of_range_values(client, monkeypatch):
    """The Pydantic bounds are the data-quality gate at the API boundary."""
    _loaded(monkeypatch)
    obs = _obs(serve.MIN_HISTORY_DAYS)
    obs[0]["streamflow_cfs"] = -5.0
    assert client.post("/predict", json={"observations": obs}).status_code == 422


def test_predict_accepts_a_well_formed_window(client, monkeypatch):
    _loaded(monkeypatch)
    r = client.post("/predict", json={"observations": _obs(serve.MIN_HISTORY_DAYS + 5)})

    assert r.status_code == 200
    body = r.json()
    assert body["model_run_id"] == "run123"
    assert body["predicted_streamflow_cfs"] > 0

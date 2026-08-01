"""Shared pytest fixtures for the tests/ directory.

pytest auto-discovers conftest.py -- fixtures defined here are available to
every test file in this directory without an import.
"""

import numpy as np
import pandas as pd
import pytest
import mlflow


@pytest.fixture(autouse=True)
def mlflow_tmp_tracking(tmp_path, monkeypatch):
    """Point MLflow at a throwaway sqlite store per test and create the
    'test_experiment' experiment that the xgboost tests log into, so tests
    never touch a real mlruns folder or tracking server."""
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow_test.db'}")
    mlflow.set_experiment("test_experiment")
    yield


@pytest.fixture
def synthetic_data():
    """Small synthetic regression dataset for exercising the tuning
    functions end-to-end without touching the real streamflow parquet."""
    n_train, n_test, n_features = 80, 20, 4

    def make_xy(n, seed):
        rng = np.random.RandomState(seed)
        X = pd.DataFrame(
            rng.uniform(-1, 1, size=(n, n_features)),
            columns=[f"f{i}" for i in range(n_features)],
        )
        y = pd.Series(X.sum(axis=1) + rng.normal(0, 0.1, size=n), name="target")
        return X, y

    X_train, y_train = make_xy(n_train, seed=1)
    X_test, y_test = make_xy(n_test, seed=2)
    return X_train, X_test, y_train, y_test

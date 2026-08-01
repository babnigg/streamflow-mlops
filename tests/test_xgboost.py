"""
Unit tests for train_xgboost_mlflow.py.

Run with:
    pytest test_train_xgboost_mlflow.py -v
"""

import numpy as np
import pandas as pd
import pytest
import mlflow

from streamflow import tune as mod


TINY_GRID = {"max_depth": [2, 3], "min_child_weight": [1, 2]}
TINY_BASE_PARAMS = {"learning_rate": 0.2, "n_estimators": 15, "reg_lambda": 1.0}

# ---------------------------------------------------------------------------
# compute_convergence
# ---------------------------------------------------------------------------
def test_compute_convergence_empty_input_returns_empty_list():
    assert mod.compute_convergence([]) == []


def test_compute_convergence_single_value():
    assert mod.compute_convergence([5.0]) == [5.0]


def test_compute_convergence_strictly_decreasing_scores_track_each_new_best():
    assert mod.compute_convergence([5.0, 4.0, 3.0]) == [5.0, 4.0, 3.0]


def test_compute_convergence_non_improving_scores_hold_at_running_best():
    assert mod.compute_convergence([3.0, 5.0, 4.0, 6.0]) == [3.0, 3.0, 3.0, 3.0]


def test_compute_convergence_mixed_sequence():
    scores = [10.0, 8.0, 9.0, 3.0, 7.0, 3.0]
    assert mod.compute_convergence(scores) == [10.0, 8.0, 8.0, 3.0, 3.0, 3.0]


def test_compute_convergence_output_is_monotonically_non_increasing():
    rng = np.random.RandomState(1)
    scores = list(rng.uniform(0, 100, size=50))
    convergence = mod.compute_convergence(scores)
    diffs = np.diff(convergence)
    assert (diffs <= 0).all()


def test_compute_convergence_output_length_matches_input_length():
    scores = [1.0, 2.0, 3.0, 4.0]
    assert len(mod.compute_convergence(scores)) == len(scores)


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------
def _dated_df(n_days, target_column="target_log_streamflow_next_day", extra_cols=None):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    data = {
        "date": dates,
        "a": np.arange(n_days, dtype=float),
        "b": np.arange(n_days, dtype=float) * 2,
        target_column: np.arange(n_days, dtype=float) * 0.1,
    }
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


def test_load_data_derives_feature_columns_excluding_metadata(monkeypatch):
    df = _dated_df(20, extra_cols={"approval_status": ["Approved"] * 20})
    monkeypatch.setattr(mod.pd, "read_parquet", lambda path: df)

    X_train, X_test, y_train, y_test, returned_columns = mod.load_data(test_size=0.2, purge_days=0)

    assert returned_columns == ["a", "b"]
    assert list(X_train.columns) == ["a", "b"]
    assert "approval_status" not in X_train.columns
    assert "date" not in X_train.columns


def test_load_data_test_size_controls_holdout_length(monkeypatch):
    df = _dated_df(100)
    monkeypatch.setattr(mod.pd, "read_parquet", lambda path: df)

    X_train, X_test, y_train, y_test, _ = mod.load_data(test_size=0.2, purge_days=0)

    assert len(X_test) == 20
    assert len(X_train) == 80


def test_load_data_purge_gap_is_dropped_from_train(monkeypatch):
    df = _dated_df(100)
    monkeypatch.setattr(mod.pd, "read_parquet", lambda path: df)

    X_train, X_test, y_train, y_test, _ = mod.load_data(test_size=0.2, purge_days=10)

    # test_size=0.2 of 100 -> 20 test rows -> test starts at index 80;
    # purge_days=10 -> train ends at index 70, so 10 rows sit in the embargo.
    assert len(X_test) == 20
    assert len(X_train) == 70


def test_load_data_train_precedes_test_chronologically(monkeypatch):
    df = _dated_df(50)
    monkeypatch.setattr(mod.pd, "read_parquet", lambda path: df)

    X_train, X_test, y_train, y_test, _ = mod.load_data(test_size=0.2, purge_days=5)

    assert X_train.index.max() < X_test.index.min()


def test_load_data_split_is_deterministic(monkeypatch):
    df = _dated_df(30)
    monkeypatch.setattr(mod.pd, "read_parquet", lambda path: df)

    result_1 = mod.load_data(test_size=0.2, purge_days=0)
    result_2 = mod.load_data(test_size=0.2, purge_days=0)

    pd.testing.assert_frame_equal(result_1[0], result_2[0])  # X_train
    pd.testing.assert_series_equal(result_1[2], result_2[2])  # y_train


# ---------------------------------------------------------------------------
# eval_and_log_final_model
# ---------------------------------------------------------------------------
def test_eval_and_log_final_model_returns_finite_metrics_and_logs_a_run(synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    params = {"max_depth": 3, "n_estimators": 20, "learning_rate": 0.1}

    rmse, mae, nse = mod.eval_and_log_final_model(
        params, X_train, y_train, X_test, y_test, run_name="test_model_run"
    )

    assert np.isfinite(rmse) and rmse >= 0
    assert np.isfinite(mae) and mae >= 0
    assert nse <= 1.0  # NSE can be negative for a bad model, but never > 1

    runs = mlflow.search_runs(
        experiment_names=["test_experiment"],
        filter_string="tags.mlflow.runName = 'test_model_run'",
    )
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.test_rmse"] == pytest.approx(rmse)
    assert runs.iloc[0]["params.max_depth"] == "3"


def test_eval_and_log_final_model_extra_tags_are_logged(synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    params = {"max_depth": 2, "n_estimators": 10}

    mod.eval_and_log_final_model(
        params, X_train, y_train, X_test, y_test,
        run_name="tagged_run",
        extra_tags={"tuning_method": "unit_test"},
    )

    runs = mlflow.search_runs(
        experiment_names=["test_experiment"],
        filter_string="tags.mlflow.runName = 'tagged_run'",
    )
    assert runs.iloc[0]["tags.tuning_method"] == "unit_test"


# ---------------------------------------------------------------------------
# run_grid_search
# ---------------------------------------------------------------------------
def test_run_grid_search_returns_expected_result_contract(synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    result = mod.run_grid_search(
        X_train, y_train, X_test, y_test,
        param_grid=TINY_GRID, base_params=TINY_BASE_PARAMS, cv_folds=2,
    )

    assert result["method"] == "grid_search"
    assert result["n_trials"] == 4  # 2 x 2 grid
    assert len(result["convergence"]) == result["n_trials"]
    assert np.isfinite(result["test_rmse"])
    assert np.isfinite(result["best_cv_rmse"])
    assert "max_depth" in result["best_params"]
    assert result["best_params"]["learning_rate"] == 0.2


def test_run_grid_search_convergence_is_non_increasing(synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    result = mod.run_grid_search(
        X_train, y_train, X_test, y_test,
        param_grid=TINY_GRID, base_params=TINY_BASE_PARAMS, cv_folds=2,
    )
    diffs = np.diff(result["convergence"])
    assert (diffs <= 1e-9).all()


# ---------------------------------------------------------------------------
# run_random_search
# ---------------------------------------------------------------------------
def test_run_random_search_returns_expected_result_contract(synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    param_distributions = {
        "max_depth": [2, 3, 4],
        "subsample": [0.7, 0.85, 1.0],
    }

    result = mod.run_random_search(
        X_train, y_train, X_test, y_test,
        param_distributions=param_distributions, n_iter=3, cv_folds=2,
    )

    assert result["method"] == "random_search"
    assert result["n_trials"] == 3
    assert len(result["convergence"]) == 3
    assert np.isfinite(result["test_rmse"])


# ---------------------------------------------------------------------------
# run_bayesian_search
# ---------------------------------------------------------------------------
def test_run_bayesian_search_returns_expected_result_contract(synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    result = mod.run_bayesian_search(
        X_train, y_train, X_test, y_test, n_trials=3, cv_folds=2,
    )

    assert result["method"] == "bayesian_optuna"
    assert result["n_trials"] == 3
    assert len(result["convergence"]) == 3
    assert np.isfinite(result["test_rmse"])


def test_run_bayesian_search_convergence_never_increases(synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    result = mod.run_bayesian_search(
        X_train, y_train, X_test, y_test, n_trials=4, cv_folds=2,
    )
    diffs = np.diff(result["convergence"])
    assert (diffs <= 1e-9).all()


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_results():
    return [
        {"method": "grid_search", "best_cv_rmse": 1.5, "test_rmse": 1.6, "test_nse": 0.5, "n_trials": 4, "time_sec": 2.0},
        {"method": "random_search", "best_cv_rmse": 1.2, "test_rmse": 1.3, "test_nse": 0.6, "n_trials": 10, "time_sec": 3.0},
        {"method": "bayesian_optuna", "best_cv_rmse": 1.1, "test_rmse": 1.2, "test_nse": 0.7, "n_trials": 10, "time_sec": 4.0},
    ]


def test_summarize_dataframe_has_one_row_per_method(fake_results, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # summarize() writes tuning_comparison.csv to cwd
    df = mod.summarize(fake_results)

    assert len(df) == len(fake_results)
    assert set(df["method"]) == {"grid_search", "random_search", "bayesian_optuna"}


def test_summarize_best_method_by_cv_rmse_is_identifiable(fake_results, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = mod.summarize(fake_results)
    best_row = df.loc[df["best_cv_rmse"].idxmin()]
    assert best_row["method"] == "bayesian_optuna"


def test_summarize_writes_csv_artifact(fake_results, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod.summarize(fake_results)
    assert (tmp_path / "tuning_comparison.csv").exists()


def test_summarize_logs_a_summary_run_to_mlflow(fake_results, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod.summarize(fake_results)

    runs = mlflow.search_runs(
        experiment_names=["test_experiment"],
        filter_string="tags.mlflow.runName = 'method_comparison_summary'",
    )
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.grid_search_best_cv_rmse"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# ensure_data_pulled
#
# There's no DVC remote configured for this repo, so `dvc pull` always
# fails -- ensure_data_pulled instead checks whether the feature parquet
# already exists and, if not, runs the actual ingest + feature-engineering
# pipeline (streamflow.ingest / streamflow.features).
# ---------------------------------------------------------------------------
def test_ensure_data_pulled_skips_pipeline_if_file_already_exists(tmp_path, monkeypatch):
    feature_path = tmp_path / "feature_matrix.parquet"
    feature_path.write_bytes(b"fake parquet bytes")
    monkeypatch.setattr(mod, "resolve", lambda rel: feature_path)

    calls = []
    monkeypatch.setattr("streamflow.ingest.build_and_save", lambda: calls.append("ingest"))
    monkeypatch.setattr("streamflow.features.build_and_save", lambda: calls.append("features"))

    mod.ensure_data_pulled()

    assert calls == []


def test_ensure_data_pulled_runs_pipeline_if_file_missing(tmp_path, monkeypatch):
    feature_path = tmp_path / "feature_matrix.parquet"  # never created
    monkeypatch.setattr(mod, "resolve", lambda rel: feature_path)

    calls = []
    monkeypatch.setattr("streamflow.ingest.build_and_save", lambda: calls.append("ingest"))
    monkeypatch.setattr("streamflow.features.build_and_save", lambda: calls.append("features"))

    mod.ensure_data_pulled()

    assert calls == ["ingest", "features"]

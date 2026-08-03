import pandas as pd
import pytest

from streamflow.features import TARGET_COLUMN, build_features


def _sample_raw(days: int = 45) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame({
        "date": dates,
        "streamflow_cfs": [100 + i for i in range(days)],
        "approval_status": ["Approved"] * (days - 2) + ["Provisional"] * 2,
        "qualifier": [None] * days,
        "last_modified": dates,
        "precip_mm": [0, 1, 2, 0, 4] * (days // 5),
        "tmax_c": [10 + (i % 7) for i in range(days)],
        "tmin_c": [2 + (i % 5) for i in range(days)],
    })


def test_build_features_next_day_target_and_lags():
    features = build_features(_sample_raw(), drop_incomplete=True)

    assert TARGET_COLUMN in features.columns
    assert "streamflow_lag_30" in features.columns
    assert "streamflow_roll_mean_30d" in features.columns
    assert "precip_sum_3d" in features.columns
    assert "is_winter_spring" not in features.columns
    assert "approval_status_Approved" not in features.columns
    assert "qualifier_none" not in features.columns
    assert not any(col.startswith("approval_status_") for col in features.columns)
    assert not any(col.startswith("qualifier_") for col in features.columns)
    assert features[TARGET_COLUMN].isna().sum() == 0

    first = features.iloc[0]
    assert first["streamflow_t"] == 130
    assert first["streamflow_lag_30"] == 100
    assert first["target_streamflow_next_day"] == 131


def test_build_features_does_not_zscore_for_xgboost():
    features = build_features(_sample_raw())

    assert "streamflow_t_z" not in features.columns
    assert not any(col.endswith("_z") for col in features.columns)
    assert "scaler_params" not in features.attrs
    assert "scaled_feature_columns" not in features.attrs


def test_missing_required_columns_raises():
    bad = _sample_raw().drop(columns=["tmax_c"])

    with pytest.raises(ValueError, match="tmax_c"):
        build_features(bad)


def test_day_of_year_sin_cos_bounded():
    features = build_features(_sample_raw(), drop_incomplete=False)

    assert features["day_of_year_sin"].between(-1, 1).all()
    assert features["day_of_year_cos"].between(-1, 1).all()


def test_lag_1_matches_manual_shift():
    raw_df = _sample_raw()
    features = build_features(raw_df, drop_incomplete=False)
    expected = raw_df["streamflow_cfs"].shift(1)

    pd.testing.assert_series_equal(
        features["streamflow_lag_1"],
        expected.rename("streamflow_lag_1"),
    )


def test_rolling_mean_matches_manual_computation():
    raw_df = _sample_raw()
    features = build_features(raw_df, drop_incomplete=False)
    expected = raw_df["streamflow_cfs"].rolling(window=7, min_periods=7).mean()

    pd.testing.assert_series_equal(
        features["streamflow_roll_mean_7d"],
        expected.rename("streamflow_roll_mean_7d"),
    )


def test_max_memory_days_truncates_rolling_windows():
    features = build_features(_sample_raw(), drop_incomplete=False, max_memory_days=10)

    assert "streamflow_roll_mean_7d" in features.columns
    assert "streamflow_roll_mean_14d" not in features.columns


def test_build_features_never_adds_zscore_scaling_columns():
    features = build_features(_sample_raw(), drop_incomplete=False)

    assert not any(col.endswith("_z") for col in features.columns)
    assert "scaler_params" not in features.attrs
    assert "scaled_feature_columns" not in features.attrs

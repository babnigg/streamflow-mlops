import pandas as pd

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
    assert "approval_status_Approved" in features.columns
    assert "qualifier_none" in features.columns
    assert features[TARGET_COLUMN].isna().sum() == 0

    first = features.iloc[0]
    assert first["streamflow_t"] == 130
    assert first["streamflow_lag_30"] == 100
    assert first["target_streamflow_next_day"] == 131


def test_build_features_optional_scaling_metadata():
    features = build_features(_sample_raw(), scale_numeric=True)

    assert "streamflow_t_z" in features.columns
    assert "scaler_params" in features.attrs
    assert "streamflow_t" in features.attrs["scaler_params"]
    assert "scaled_feature_columns" in features.attrs

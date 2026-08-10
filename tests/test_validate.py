"""Data-quality gate tests: every check must fail loudly on the failure it owns.

validate() is what stands between a broken upstream feed and the model, and it
is reused by monitor.py - a check that cannot fail is worse than no check,
because it reads as a passing gate.
"""

import pandas as pd
import pytest

from streamflow import validate as V


def _raw(days: int = 30_000) -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    return pd.DataFrame({
        "date": dates,
        "streamflow_cfs": 400.0,
        "precip_mm": 1.0,
        "tmax_c": 15.0,
        "tmin_c": 5.0,
    })


def test_clean_data_passes():
    V.validate(_raw())


def test_missing_column_is_a_quality_error_not_a_keyerror():
    """monitor.py catches DataValidationError only - a KeyError would crash the
    monitor instead of being reported as a failed gate."""
    with pytest.raises(V.DataValidationError, match="tmax_c"):
        V.validate(_raw().drop(columns=["tmax_c"]))


def test_calendar_gap_is_rejected():
    """features.py shifts positionally, so a hole silently redefines every lag."""
    df = _raw().drop(index=range(100, 110))
    with pytest.raises(V.DataValidationError, match="missing days"):
        V.check_calendar(df)


def test_duplicate_dates_are_rejected():
    df = pd.concat([_raw(), _raw().tail(1)], ignore_index=True)
    with pytest.raises(V.DataValidationError, match="duplicate"):
        V.check_calendar(df)


def test_recent_nulls_are_caught_even_though_the_record_is_long():
    """A year of missing weather is 1% of an 82-year record - under a
    whole-record threshold it would pass."""
    df = _raw()
    df.loc[df.index[-200:], "precip_mm"] = None
    with pytest.raises(V.DataValidationError, match="precip_mm"):
        V.check_missing_values(df, columns=["precip_mm"])


def test_truncated_fetch_is_rejected():
    with pytest.raises(V.DataValidationError, match="rows ingested"):
        V.check_row_count(_raw(days=5_000))


def test_stale_feed_is_rejected():
    df = _raw()
    df["date"] = df["date"] - pd.Timedelta(days=30)
    with pytest.raises(V.DataValidationError, match="days old"):
        V.check_data_recency(df)


def test_impossible_values_are_rejected():
    neg = _raw()
    neg.loc[0, "streamflow_cfs"] = -1
    with pytest.raises(V.DataValidationError, match="Negative"):
        V.check_value_ranges(neg)

    # sentinel / unit swap: the record high at this gauge is 10,700 cfs
    huge = _raw()
    huge.loc[0, "streamflow_cfs"] = 999_999
    with pytest.raises(V.DataValidationError, match="exceeds"):
        V.check_value_ranges(huge)

    hot = _raw()
    hot.loc[0, "tmax_c"] = 80
    with pytest.raises(V.DataValidationError, match="Temperature"):
        V.check_value_ranges(hot)

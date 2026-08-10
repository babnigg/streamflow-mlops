"""Data quality checks on the raw ingested dataset.

Run right after ingest, and reused by monitor.py so serving and monitoring
cannot disagree about what counts as valid. Catches upstream API issues -
schema changes, calendar gaps, null weather, stale or impossible values -
before they propagate into features and training.
"""

import pandas as pd

REQUIRED_COLUMNS = ["date", "streamflow_cfs", "precip_mm", "tmax_c", "tmin_c"]
RECENT_DAYS = 365          # null checks look here, not over the whole record


class DataValidationError(Exception):
    """Raised when the ingested data fails a quality check."""


def check_schema(df: pd.DataFrame, columns: list[str] = REQUIRED_COLUMNS) -> None:
    """Missing columns must fail as a quality error, not a KeyError - callers
    catch DataValidationError, so a renamed upstream field would otherwise crash
    the monitor instead of being reported as a failed gate."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def check_calendar(df: pd.DataFrame, date_col: str = "date") -> None:
    """No duplicate or missing days. features.py shifts positionally, so a hole
    in the calendar silently turns lag_1 into 'whatever came before', and every
    rolling window then spans the wrong period."""
    d = pd.to_datetime(df[date_col]).sort_values()
    dupes = int(d.duplicated().sum())
    if dupes:
        raise DataValidationError(f"{dupes} duplicate dates")
    gaps = pd.date_range(d.min(), d.max(), freq="D").difference(d)
    if len(gaps):
        raise DataValidationError(f"{len(gaps)} missing days, first {gaps[0].date()}")


def check_missing_values(df: pd.DataFrame, columns: list[str], max_null_frac: float = 0.01,
                         recent_days: int = RECENT_DAYS) -> None:
    """Checked over the recent window: 1% of an 82-year record is 300 days, so a
    whole year of missing weather would pass a whole-record threshold."""
    recent = df.tail(recent_days)
    for col in columns:
        null_frac = recent[col].isna().mean()
        if null_frac > max_null_frac:
            raise DataValidationError(
                f"Column '{col}' has {null_frac:.1%} missing values in the last "
                f"{len(recent)} rows (max allowed: {max_null_frac:.1%})"
            )


def check_row_count(df: pd.DataFrame, min_rows: int = 25_000) -> None:
    # the gauge has recorded daily since 1943; a much shorter table means a
    # truncated fetch, not a short history.
    if len(df) < min_rows:
        raise DataValidationError(f"Only {len(df)} rows ingested (expected at least {min_rows})")


def check_data_recency(df: pd.DataFrame, date_col: str = "date", max_staleness_days: int = 3) -> None:
    # USGS publishes provisional values a few days behind real time.
    most_recent = pd.to_datetime(df[date_col]).max().tz_localize(None)
    staleness = pd.Timestamp.now() - most_recent
    if staleness.days > max_staleness_days:
        raise DataValidationError(
            f"Most recent data is {staleness.days} days old (max allowed: {max_staleness_days})"
        )


def check_value_ranges(df: pd.DataFrame, max_flow_cfs: float = 100_000) -> None:
    if (df["streamflow_cfs"] < 0).any():
        raise DataValidationError("Negative streamflow values found")
    # an upper bound catches sentinels and unit swaps; the record high here is
    # 10,700 cfs, so this only fires on something physically impossible.
    if df["streamflow_cfs"].max() > max_flow_cfs:
        raise DataValidationError(
            f"Streamflow {df['streamflow_cfs'].max():,.0f} cfs exceeds {max_flow_cfs:,.0f}"
        )
    if df["tmax_c"].max() > 60 or df["tmin_c"].min() < -60:
        raise DataValidationError("Temperature values outside plausible range")


def validate(df: pd.DataFrame) -> None:
    """Run all checks. Raises DataValidationError on first failure."""
    check_schema(df)
    check_row_count(df)
    check_calendar(df)
    check_missing_values(df, columns=["streamflow_cfs", "precip_mm", "tmax_c", "tmin_c"])
    check_data_recency(df)
    check_value_ranges(df)
    print(f"data validation passed: {len(df):,} rows, most recent date {df['date'].max().date()}")

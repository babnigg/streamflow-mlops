"""Data quality checks on the raw ingested dataset.

Run as part of the daily pipeline, right after ingest — catches upstream
API issues (missing days, null weather, stale data) before they propagate
into feature engineering and training.
"""

import pandas as pd


class DataValidationError(Exception):
    """Raised when the ingested data fails a quality check."""


def check_missing_values(df: pd.DataFrame, columns: list[str], max_null_frac: float = 0.01) -> None:
    for col in columns:
        null_frac = df[col].isna().mean()
        if null_frac > max_null_frac:
            raise DataValidationError(
                f"Column '{col}' has {null_frac:.1%} missing values (max allowed: {max_null_frac:.1%})"
            )


def check_row_count(df: pd.DataFrame, min_rows: int = 1000) -> None:
    if len(df) < min_rows:
        raise DataValidationError(f"Only {len(df)} rows ingested (expected at least {min_rows})")


def check_data_recency(df: pd.DataFrame, date_col: str = "date", max_staleness_days: int = 1) -> None:
    most_recent = pd.to_datetime(df[date_col]).max()
    staleness = pd.Timestamp.now() - most_recent
    if staleness.days > max_staleness_days:
        raise DataValidationError(
            f"Most recent data is {staleness.days} days old (max allowed: {max_staleness_days})"
        )


def check_value_ranges(df: pd.DataFrame) -> None:
    if (df["streamflow_cfs"] < 0).any():
        raise DataValidationError("Negative streamflow values found")
    if df["tmax_c"].max() > 60 or df["tmin_c"].min() < -60:
        raise DataValidationError("Temperature values outside plausible range")


def validate(df: pd.DataFrame) -> None:
    """Run all checks. Raises DataValidationError on first failure."""
    check_row_count(df)
    check_missing_values(df, columns=["streamflow_cfs", "precip_mm", "tmax_c", "tmin_c"])
    check_data_recency(df)
    check_value_ranges(df)
    print(f"data validation passed: {len(df):,} rows, most recent date {df['date'].max().date()}")

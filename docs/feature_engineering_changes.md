# Feature Engineering Changes

This branch keeps the feature matrix focused on predictors that are suitable for
tree-based XGBoost training.

## Removed from model features

- Z-score scaling: XGBoost tree boosters do not need standardized feature
  scales, and raw/log-transformed units are easier to interpret.
- `is_winter_spring`: this binary season flag was redundant with `month`,
  `day_of_year_sin`, and `day_of_year_cos`.
- `approval_status_*` and `qualifier_*` dummy variables: these USGS metadata
  fields are useful for monitoring and data-quality review, but they can create
  shortcuts that are not physical hydrology/weather signals.

## Kept

- Raw hydrology and weather values.
- Log-transformed streamflow lags to reduce the influence of flood spikes while
  preserving their signal.
- Streamflow lag and rolling-window features.
- Precipitation rolling sums/means.
- Temperature level, range, and rolling mean features.
- Month and day-of-year sine/cosine seasonal encodings.
- `is_high_flow_anomaly` as a flood/anomaly indicator.

The saved feature table still retains raw `approval_status` and `qualifier`
columns for monitoring/reference, but they are excluded from
`features.attrs["feature_columns"]` and are not one-hot encoded for model
training.

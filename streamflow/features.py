"""Feature matrix. [stub]

Planned: discharge lags (t..t-3), 7/30-day rolling means, precip + 3-day sum,
tmax/tmin, day-of-year sin/cos. Target: log-discharge shifted -1 day.
"""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError

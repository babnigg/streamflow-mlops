"""Smoke tests. Live API test only runs with RUN_LIVE=1."""

import os
import pandas as pd

from streamflow.config import CONFIG, resolve
from streamflow import ingest


def test_config_keys():
    for key in ("site", "usgs", "weather", "data", "target"):
        assert key in CONFIG
    assert CONFIG["site"]["id"] == "USGS-05532500"


def test_saved_table_schema():
    path = resolve(CONFIG["data"]["raw_path"])
    if not path.exists():
        return
    df = pd.read_parquet(path)
    expected = {"date", "streamflow_cfs", "approval_status", "qualifier",
                "last_modified", "precip_mm", "tmax_c", "tmin_c"}
    assert expected.issubset(df.columns)
    assert df["streamflow_cfs"].isna().sum() == 0


def test_live_fetch_weather():
    if os.environ.get("RUN_LIVE") != "1":
        return
    wx = ingest.fetch_weather("2026-07-01", "2026-07-05")
    assert len(wx) == 5
    assert {"precip_mm", "tmax_c", "tmin_c"}.issubset(wx.columns)

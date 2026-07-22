"""Ingest: join USGS daily streamflow (OGC API) with Open-Meteo ERA5 weather.

Keeps approval_status / qualifier / last_modified per row - USGS revises the
provisional tail, which is drift we monitor later.

Run:  python -m streamflow.ingest
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from .config import CONFIG, resolve

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "streamflow-mlops/0.1 (ADSP 32021 coursework)"})


def fetch_usgs_daily(cfg: dict = CONFIG) -> pd.DataFrame:
    """Full mean-streamflow record via cursor pagination."""
    u = cfg["usgs"]
    url = f"{u['ogc_base']}/collections/daily/items"
    params = {
        "monitoring_location_id": cfg["site"]["id"],
        "parameter_code": u["parameter_code"],
        "statistic_id": u["statistic_id"],
        "limit": u["page_size"],
    }
    rows, page = [], 0
    while url:
        resp = SESSION.get(url, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        for feat in payload.get("features", []):
            p = feat["properties"]
            q = p.get("qualifier")   # list -> comma string
            if q is not None and not isinstance(q, str):
                q = ",".join(str(x) for x in q)
            rows.append({
                "date": p.get("time"),
                "streamflow_cfs": p.get("value"),
                "approval_status": p.get("approval_status"),
                "qualifier": q,
                "last_modified": p.get("last_modified"),
            })
        nxt = [l["href"] for l in payload.get("links", []) if l.get("rel") == "next"]
        url, params, page = (nxt[0], None, page + 1) if nxt else (None, None, page)
        print(f"  usgs page {page}: {len(rows):,} records", end="\r")
    print()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["streamflow_cfs"] = pd.to_numeric(df["streamflow_cfs"], errors="coerce")
    df["last_modified"] = pd.to_datetime(df["last_modified"], errors="coerce", utc=True)
    start = pd.to_datetime(cfg["data"]["start_date"])
    return df[df["date"] >= start].sort_values("date").reset_index(drop=True)


def fetch_weather(start: str, end: str, cfg: dict = CONFIG) -> pd.DataFrame:
    """Daily precip/tmax/tmin at the gauge point (ERA5 archive)."""
    w, s = cfg["weather"], cfg["site"]
    resp = SESSION.get(w["archive_url"], timeout=90, params={
        "latitude": s["latitude"], "longitude": s["longitude"],
        "start_date": start, "end_date": end,
        "daily": ",".join(w["daily_vars"]), "timezone": w["timezone"],
    })
    resp.raise_for_status()
    d = resp.json()["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "precip_mm": d["precipitation_sum"],
        "tmax_c": d["temperature_2m_max"],
        "tmin_c": d["temperature_2m_min"],
    })


def build_dataset(cfg: dict = CONFIG) -> pd.DataFrame:
    print("fetching USGS streamflow...")
    flow = fetch_usgs_daily(cfg)
    lo, hi = flow["date"].min().date(), flow["date"].max().date()
    print(f"  {len(flow):,} daily values, {lo} -> {hi}")

    print("fetching Open-Meteo weather...")
    wx = fetch_weather(str(lo), str(hi), cfg)
    print(f"  {len(wx):,} daily weather rows")

    return flow.merge(wx, on="date", how="left")


def build_and_save(cfg: dict = CONFIG) -> pd.DataFrame:
    df = build_dataset(cfg)
    out = resolve(cfg["data"]["raw_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    prov = (df["approval_status"] == "Provisional").sum()
    print(f"\nsaved {len(df):,} rows -> {out}")
    print(f"  provisional: {prov} | missing weather: {df['precip_mm'].isna().sum()}")
    return df


if __name__ == "__main__":
    t0 = time.time()
    build_and_save()
    print(f"done in {time.time() - t0:.1f}s")

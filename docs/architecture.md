# architecture

Next-day streamflow forecast for USGS 05532500, graded on engineering rigor /
reproducibility / system design - not accuracy.

## pipeline

```
                                  [ orchestrator: Prefect or Airflow - TBD ]
                                                  |
USGS OGC daily ──┐                                v
                 ├─► ingest ─► features ─► train ─► serve ─► monitor
Open-Meteo ERA5 ─┘  (done)     (stub)     (stub)   (stub)    (stub)
                       │           │         │        │         │
                  raw parquet   matrix    MLflow   FastAPI   Evidently
                  (DVC)                  registry  + Docker
```

## daily-forecast design

Daily cadence: ingest latest gauge + weather -> forecast next day -> score when
the actual lands -> monitor drift -> retrain on threshold. Real data won't drift
in a few weeks, so we replay historical days at accelerated speed through the
same flow (each date = one tick), then run it live for the demo.

## drift hooks (real, in the data now)

- provisional tail (46 rows) that USGS silently revises; `qualifier=REVISED` (150 rows)
- winter ice: `ESTIMATED`/`ICE` qualifiers, concentrated Dec-Feb
- seasonal regimes + flood spikes
- required artificial corruption (out-of-bounds, swapped cols, schema) layered on in stage 4

## metric

NSE + log-RMSE vs persistence (tomorrow = today). Log space (flow spans 0-10,700 cfs).

## work split (draft)

One graded axis each: data/DVC (stage 1, mostly built) | train+MLflow (2) |
FastAPI+Docker (3) | Evidently drift (4). Orchestrator wiring shared.

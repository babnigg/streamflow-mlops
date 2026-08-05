# architecture

Next-day streamflow forecast for USGS 05532500, graded on engineering rigor /
reproducibility / system design - not accuracy.

## pipeline

```
                          [ orchestrator: Prefect ]
                                     |
USGS OGC daily ──┐                   v
                 ├─► ingest ─► validate ─► features ─► tune ─► train ─► serve ─► monitor
Open-Meteo ERA5 ─┘   (done)    (done)      (done)     (done)  (done)   (done)   (stub)
                        │         │           │          │       │        │        │
                   raw parquet  quality    matrix     Optuna  MLflow   FastAPI  Evidently
                     (DVC)      checks      (DVC)    +MLflow  tracking + Docker
```

Retraining is a separate on-demand flow, not part of the daily run: the daily
pipeline ingests, validates, rebuilds features and predicts.

## daily-forecast design

Daily cadence: ingest latest gauge + weather -> validate -> forecast next day ->
score when the actual lands -> monitor drift -> retrain on threshold. Real data
won't drift in a few weeks, so we replay historical days at accelerated speed
through the same flow (each date = one tick), then run it live for the demo.

## metrics

Modelled in log space (flow spans 0-10,700 cfs).

- **NSE** - reported, the hydrology convention. Scores against the observed mean.
- **Persistence Index** - *monitored*. Scores against persistence (tomorrow =
  today), so 0 means the model adds nothing and negative means it is worse than
  doing nothing.

NSE alone is misleading here: on this autocorrelated series persistence by itself
scores NSE 0.86, so a model at NSE 0.90 is only a 17% error reduction (PI 0.31).
A stale model measured on a later month still posts NSE 0.12 - which reads as
"degraded but working" - while its PI is -1.08, i.e. five times worse than
persistence. The alert threshold is therefore `PI < 0` (config
`target.monitor_alert_below`), which needs no arbitrary tuning.

## monitoring plan (stage 4)

Two signal families, deliberately separated:

- **data drift** (PSI on numeric features, chi-square on the `approval_status` /
  `qualifier` metadata, plus drift on the prediction distribution). No labels
  needed, so it is a *leading* indicator - available the same day.
- **performance drift** (PI). Needs tomorrow's actual value, so it is *lagging*.

Alert on the **share** of drifted features, not any single one: with ~48 heavily
correlated lag/rolling features, per-feature p-values would fire by chance on
every run.

Only sustained PI < 0 triggers retraining. A flood trips the distribution tests
too - extreme but legitimate values - so data drift alone must not retrain, or we
would retrain on every flood using the rarest data we have.

## drift hooks (real, in the data now)

- provisional tail that USGS silently revises; `qualifier=REVISED`
- winter ice: `ESTIMATED`/`ICE` qualifiers, concentrated Dec-Feb
- seasonal regimes + flood spikes
- required artificial corruption (out-of-bounds, swapped cols, schema) layered on
  in stage 4; the API rejects schema/range violations at the Pydantic boundary
  before they reach the model

## work split

data/DVC (stage 1) | tune + train + MLflow (2) | Prefect orchestration (2) |
FastAPI + Docker (3) | Evidently drift (4).

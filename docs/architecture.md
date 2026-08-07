# architecture

Next-day streamflow forecast for USGS 05532500, graded on engineering rigor /
reproducibility / system design - not accuracy.

## pipeline

```
                          [ orchestrator: Prefect ]
                                     |
USGS OGC daily ──┐                   v
                 ├─► ingest ─► validate ─► features ─► tune ─► train ─► serve ─► monitor
Open-Meteo ERA5 ─┘   (done)    (done)      (done)     (done)  (done)   (done)   (done)
                        │         │           │          │       │        │        │
                   raw parquet  quality    matrix     Optuna  MLflow   FastAPI  Evidently
                     (DVC)      checks      (DVC)    +MLflow  tracking + Docker  + PI
```

The daily flow ingests, validates, rebuilds features, predicts and monitors.
Retraining branches off it only when monitoring asks for it; it is also
available as a standalone flow.

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
persistence. The alert threshold is therefore `PI < 0` over a 60-day window
(config `monitoring.pi_alert_below`) - see the calibration below.

## monitoring (stage 4)

Session 7's three pillars, with detection separated from action
(`streamflow/monitor.py`):

| pillar | signal | labels | window | acts as |
|---|---|---|---|---|
| data | quality + ranges | no | same day | reject input, never retrain |
| data | Evidently drift | no | 90 d | evidence, not a pager |
| model | persistence index | yes (t+1) | 60 d | retrain |
| system | model age, numeric stability, latency | no | per run | retrain / investigate |

Signals are checked in that order: a broken feed or an unusable model is caught
before performance, since a PI computed on bad inputs says nothing about the
model.

Coverage against the ML Test Score monitoring tests (Table IV): Monitor 2 and 3
via `validate.py` and the single shared feature path, 4/5/6 via the system
pillar, 7 via the persistence index. Monitor 1 (dependency-change notification)
is not implemented - dependencies are pinned, but nothing watches them.

Everything runs through `score_range(start, end)`, which scores any historical
span. A production run is a span of one day; the demo is a span of years, so
replay is the normal path rather than a separate harness.

**Thresholds are calibrated, not borrowed.** Two measurements set them:

- *PI < 0 at 60 days.* Across a 10.5-year backtest, no 60-day non-overlapping
  block of a healthy model fell below zero (min +0.04). At 30 days 6% of blocks
  do, and at 14 days 16% - so shorter windows cannot carry an alarm.
- *Per-column PSI, above the clean maximum.* The textbook PSI 0.2 cutoff flags
  **every** known-clean 90-day window here (streamflow alone sits near 0.68 in a
  quiet quarter). With per-column thresholds calibrated from clean history,
  0/15 clean windows alarm while a -15 C temperature shift (PSI 7.4) and 5x
  rainfall (PSI 1.4) are both caught.

**Why drift is not a pager.** At daily cadence a 30-day PSI window exceeds 0.2
about 80% of the time with no drift present; 90 days is needed for 7%. Hourly
data would put 2,160 rows in the same window and could promote drift to a real
alarm - one of the concrete arguments for that refactor.

**Reference choice matters as much as the test.** The drift reference is the
same season in recent years. A calendar-year baseline marks every summer as
100% drifted, and an 82-year baseline reads decades of changed river regime as
drift (streamflow PSI 0.97 unlimited vs 0.14 over ten years).

Only sustained PI < 0 retrains, and the candidate must beat the champion on a
held-out window before promotion. A flood trips the distribution tests while the
model is still healthy - retraining on it would teach the rarest data we have as
the new normal.

## drift hooks (real, in the data now)

- provisional tail that USGS silently revises; `qualifier=REVISED`
- winter ice: `ESTIMATED`/`ICE` qualifiers, concentrated Dec-Feb
- seasonal regimes + flood spikes
- artificial corruption (out-of-bounds, swapped cols, schema): the API rejects
  schema/range violations at the Pydantic boundary, and Evidently catches
  shifts that are valid but wrong (notebooks/05_monitoring)

## work split

data/DVC (stage 1) | tune + train + MLflow (2) | Prefect orchestration (2) |
FastAPI + Docker (3) | monitoring + drift (4).

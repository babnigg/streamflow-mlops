# architecture

Next-day streamflow forecast for USGS 05532500, graded on engineering rigor /
reproducibility / system design - not accuracy.

## pipeline

```
                          [ orchestrator: Prefect ]
                                     |
USGS OGC daily ──┐                   v
                 ├─► ingest ─► validate ─► features ─► tune ─► train ─► gate ─► serve ─► monitor
Open-Meteo ERA5 ─┘      │         │           │          │       │       │       │        │
                   raw parquet  quality    matrix     Optuna  MLflow  champion FastAPI  Evidently
                     (DVC)      checks      (DVC)    +MLflow  registry  vs      +Docker  + PI
                                                                      challenger
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
scores NSE 0.84, so the model's NSE 0.89 is only an 18% error reduction
(PI 0.32). A stale model - trained on January, serving June - posts NSE -0.04
while its PI is -3.04: four times worse than doing nothing. The alert threshold
is therefore `PI < 0` over a 90-day window (config `monitoring.pi_alert_below`)
- see the calibration below.

## monitoring (stage 4)

Session 7's three pillars, with detection separated from action
(`streamflow/monitor.py`):

| pillar | signal | labels | window | acts as |
|---|---|---|---|---|
| data | quality + ranges | no | same day | reject input, never retrain |
| data | flow vs seasonal p99 | no | same day | flag the event, never retrain |
| data | Evidently drift | no | 90 d | evidence, not a pager |
| model | persistence index | yes (t+1) | 90 d | retrain |
| system | model age, numeric stability, latency | no | per run | retrain / investigate |

Signals are checked in that order: a broken feed or an unusable model is caught
before performance, since a PI computed on bad inputs says nothing about the
model.

Coverage against the ML Test Score monitoring tests (Table IV): Monitor 1 via
Dependabot, 2 and 3 via `validate.py` and the single shared feature path, 4/5/6
via the system pillar, 7 via the persistence index.

Everything runs through `score_range(start, end)`, which scores any historical
span. A production run is a span of one day; the demo is a span of years, so
replay is the normal path rather than a separate harness.

**Thresholds are calibrated, not borrowed.** Two measurements set them:

- *PI < 0 at 90 days.* Calibration measures the false-alarm rate of the same
  overlapping rolling window the alarm uses, on a 10.5-year healthy backtest:

  | window | windows below 0 | spurious episodes | worst |
  |---|---|---|---|
  | 30 d | 8.28% | 31 | -13.12 |
  | 60 d | 1.99% | 9 | -0.84 |
  | 90 d | **0.61%** | **3** | -0.36 |

  An induced stale model is caught on 100% of days at either 60 or 90 (min PI
  -3.21), so the wider window costs nothing in detection. Requiring N
  consecutive breaches is not a substitute - healthy dips run up to 19 days.

  Non-overlapping blocks would report 0/64 below zero for the same model, which
  is why they are not used: they evaluate one arbitrary phase of the block grid,
  and the deployed alarm evaluates every phase.
- *Per-column PSI, above the clean maximum.* The textbook PSI 0.2 cutoff flags
  **every** known-clean 90-day window here (streamflow reaches 2.20 in a quiet
  winter quarter). With per-column thresholds set above the clean maximum,
  0/15 clean windows alarm while a -15 C shift (PSI 7.5), 5x rainfall (0.82)
  and a 10x gauge fault (3.10) are all caught.

  The share counts only the four watched inputs. `target` and `prediction` have
  no calibrated threshold, so at the default cutoff both read as drifted in
  every clean window - counting them would hold the share permanently at 2/6 and
  let the alarm fire on columns that were never validated.

**A flood is an event, not a distribution shift.** PSI cannot see one at this
cadence: the July 2026 flood scores 3.33 on streamflow over a 7-day window and
0.19 over the 90-day window the sample size supports - two days of record flow
diluted by eighty-eight ordinary ones. Shortening the window is not available
either, per the null test below. So the event is detected directly, as today's
flow against the seasonal 99th percentile: 1.0% of days over six years, 9
distinct events. Like drift, it only ever flags.

**Why drift is not a pager.** Null test - two samples drawn from the *same*
distribution, so no drift exists by construction. PSI still exceeds 0.2 on
98.5% of 30-day draws, 46.5% at 90 days, and only 2.2% at 180. At daily cadence
the windows we can afford are underpowered, which is why per-column thresholds
are calibrated rather than borrowed, and why drift never pages on its own.
Hourly data would put 2,160 rows in a 90-day window and could promote drift to a
real alarm - one of the concrete arguments for that refactor.

**Reference choice matters as much as the test.** The drift reference is the
same season in recent years. Measured on the same current window: a calendar
baseline gives PSI 0.57 purely from the season, an unlimited-history baseline
1.41 (decades of changed river regime), and the season-matched last-10-years
reference 0.13.

Only sustained PI < 0 retrains. A flood trips the distribution tests while the
model is still healthy - retraining on it would teach the rarest data we have as
the new normal.

## promotion gate

A retrain produces a *candidate*, not a deployment (`streamflow/registry.py`).
Serving loads the version carrying the `champion` alias, so a retrain that loses
its comparison stays registered and unserved.

```
train -> register version -> score candidate and champion on the SAME window
                          -> PI within tolerance? -> move alias : leave it
```

Both models are re-scored on the current hold-out. The champion's own recorded
metrics are not comparable - it was fitted when the split ended somewhere else,
so its stored score describes different days.

Measured end to end, with a deliberately crippled retrain standing in for one
triggered on bad-but-valid inputs (ICE/ESTIMATED qualifiers pass validation):

| version | test PI | promoted | served |
|---|---|---|---|
| v1 | +0.3170 | yes (first model) | |
| v2 | +0.3170 | yes (ties, within tolerance) | champion |
| v3 | **-6.0157** | **no** | never |

`predict` continued to load v2. Under the previous newest-run-wins rule v3 would
have gone straight into production while the alert log recorded the retrain as
remediation.

Rollback is one call (`registry.rollback()`) and skips rejected candidates -
counting back by version number would redeploy the model the gate just refused.
Recovery does not depend on the training pipeline being healthy.

Tolerance is `promotion.tolerance` (0.02 PI): an exactly-equal retrain on fresher
data still ships, a materially worse one never does.

## drift hooks (real, in the data now)

- provisional tail that USGS silently revises; `qualifier=REVISED`
- winter ice: `ESTIMATED`/`ICE` qualifiers, concentrated Dec-Feb
- seasonal regimes + flood spikes
- artificial corruption (out-of-bounds, swapped cols, schema): the API rejects
  schema/range violations at the Pydantic boundary, and Evidently catches
  shifts that are valid but wrong (notebooks/05_monitoring)

## ci

`.github/workflows/ci.yml` runs ruff and the full suite on every push and PR,
plus a Docker build so a broken image fails the build rather than the demo.

The suite needs no data, model, network or secrets - fixtures are synthetic and
the two tests that would need a live API skip themselves - so CI is the same
command a teammate runs locally (~6 min, 90 passed / 2 skipped).

There is no deploy job: nothing is deployed to. The deployment step in this
project is the promotion gate moving the `champion` alias, which is what serving
resolves.

Dependabot watches pip and actions weekly. That is ML Test Score Monitor 1 -
pinning records what we depend on, but only a watcher notices when it moves, and
a model trained under one xgboost and served under another is silent skew.

## work split

| member | contribution |
|---|---|
| Janvi | ingestion, data validation, EDA, feature engineering (stage 1) + DVC |
| Saya | hyperparameter tuning, training, MLflow tracking (stage 2) |
| Tori | Prefect orchestration, scheduled flows (stage 2) |
| Daniel | FastAPI + Docker (stage 3), monitoring + drift + retrain trigger (stage 4) |

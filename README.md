# streamflow-mlops

MLOps pipeline forecasting next-day streamflow for the Des Plaines River at
Riverside, IL (USGS 05532500). ADSP 32021 final project - Team 3 (Tori, Janvi,
Daniel, Saya).

## stages

| stage | what | where |
|-------|------|-------|
| 1 | ingestion, validation, features (USGS + Open-Meteo, 1944-present) | `ingest.py` `validate.py` `features.py` |
| 2 | tuning (grid / random / bayesian), training, MLflow tracking | `tune.py` `train.py` |
| 2 | model registry + champion/challenger promotion gate | `registry.py` |
| 2 | orchestration | `flows/pipeline.py` |
| 3 | FastAPI + Docker deployment | `serve.py` `Dockerfile` |
| 4 | monitoring, drift, retraining trigger | `monitor.py` |

Design details and metric justification: [`docs/architecture.md`](docs/architecture.md).

## data

- **streamflow:** USGS OGC API (`api.waterdata.usgs.gov`), daily mean cfs, ~30k days.
  Rows keep `approval_status` + `last_modified` (USGS revises the provisional tail).
- **weather:** Open-Meteo ERA5 daily precip/tmax/tmin at the gauge.
- both public + keyless; the pipeline rebuilds from scratch.

## setup

```bash
pip install -r requirements.txt
python -m streamflow.ingest      # ~20s -> data/raw/streamflow.parquet
python -m streamflow.features    # -> data/features/feature_matrix.parquet
python -m streamflow.train       # fits with config/best_params.yaml, logs to MLflow
python -m streamflow.predict     # next-day forecast from the latest model
```

Data is DVC-tracked: the `.dvc` pointers are committed, the parquets aren't.
Re-run ingest/features to rebuild (no shared DVC remote is configured yet).

Re-tune hyperparameters (slow, writes `config/best_params.yaml`):

```bash
python -m streamflow.tune
```

## promotion gate

A retrain is a candidate, not a deployment. `train` registers a new version,
scores it and the current champion on the same hold-out, and moves the
`champion` alias only if the candidate wins. Serving follows the alias.

```bash
python -m streamflow.registry            # versions, metrics, which one serves
```

```
streamflow-xgb  (champion alias -> version 2)
   v3   run 38e5266d  pi -6.0157  promoted=false
 * v2   run 1a16043c  pi +0.3170  promoted=true
   v1   run 66a5ed60  pi +0.3170  promoted=true
```

v3 is a crippled retrain: registered for audit, never served. Rollback is one
call and skips rejected candidates:

```python
from streamflow import registry
registry.rollback()          # champion alias -> last version that passed the gate
```

## orchestration

```bash
prefect server start                    # then, in another shell:
python flows/pipeline.py                # one daily run: ingest -> validate -> features -> predict -> monitor
prefect deploy --all                    # register the scheduled deployments
```

## serving

The API loads the most recent model from MLflow and exposes it over HTTP.

```bash
uvicorn streamflow.serve:app --reload   # http://127.0.0.1:8000/docs
```

| endpoint | purpose |
|---|---|
| `GET /health` | liveness + whether a model is loaded |
| `GET /model` | run id, target, tracking uri |
| `POST /predict` | caller supplies >= 31 days of observations |
| `GET /predict/latest` | scores the newest row of the raw table on disk |
| `GET /docs` | Swagger UI |

### docker

The image is serving-only (`requirements-serve.txt`) - no optuna, prefect or
jupyter. The model and data are mounted rather than baked in, so a retrain is
picked up without rebuilding.

```bash
docker build -t streamflow-api .
docker run --rm -p 8000:8000 -v "${PWD}/mlruns:/app/mlruns" -v "${PWD}/data:/app/data" streamflow-api
```

(one line, and `${PWD}` rather than `$(pwd)`, so it runs in PowerShell as well as bash)

Then `curl localhost:8000/health` or open `http://localhost:8000/docs`.
Run `python -m streamflow.train` at least once first, or there is no model to load.

## monitoring

```bash
python -m streamflow.monitor            # scores every day on disk, ~12s
```

```python
from streamflow import monitor
monitor.run("2026-01-01", "2026-06-30")  # any span; production passes one day
```

Writes `data/monitoring/predictions.parquet`, appends the decision to
`alerts.parquet`, and saves an Evidently report to `reports/drift_<date>.html`.
Prints one line:

```
scored 30,138 days in 11.8s | model age 0.0d | rolling PI (90d) 0.370 | ok -> none
```

Three pillars, checked in order - a bad feed or an unusable model outranks
performance, since a PI computed on bad inputs says nothing about the model:

| signal | acts as |
|---|---|
| data quality | `reject_input` - retraining cannot fix a broken feed |
| non-finite predictions | `investigate` - broken, not stale |
| flow above the seasonal p99 | `flag_only` - a flood is an event, not a new normal |
| model age > 30d, or unknown | `retrain` / `investigate` |
| rolling PI < 0 (90d) | `retrain` |
| Evidently drift | `flag_only` - a flood drifts inputs while the model is fine |

`notebooks/05_monitoring.ipynb` derives the thresholds and walks the failure
scenarios. Evidently is in `requirements.txt` but deliberately not in
`requirements-serve.txt`, so the API image stays lean.

## layout

```
config/config.yaml        # gauge, urls, params, target - single source of truth
config/best_params.yaml   # hyperparameters chosen by streamflow.tune
streamflow/
  ingest.py               # USGS + Open-Meteo -> raw parquet
  validate.py             # data-quality checks
  features.py             # lags, rolling windows, seasonality -> feature matrix
  tune.py                 # grid / random / bayesian search, logged to MLflow
  train.py                # fit with best params, log model
  predict.py              # score the latest row
  serve.py                # FastAPI inference app
  monitor.py              # scoring, drift, retraining decision
flows/pipeline.py         # Prefect flows: daily, retrain, tuning
prefect.yaml              # deployment definitions
notebooks/                # 01 eda | 02 features | 03 tuning | 05 monitoring
data/                     # DVC-tracked
docs/architecture.md
tests/
Dockerfile
```

# streamflow-mlops

MLOps pipeline forecasting next-day streamflow for the Des Plaines River at
Riverside, IL (USGS 05532500). ADSP 32021 final project - Team 3 (Tori, Janvi,
Daniel, Saya).

## status

| stage | what | status |
|-------|------|--------|
| 1 | ingestion (USGS + Open-Meteo -> daily parquet, 1944-present) | done |
| 1 | data-quality validation | done |
| 1 | EDA + feature engineering | done |
| - | DVC data versioning | done (no shared remote yet) |
| 2 | hyperparameter tuning (grid / random / bayesian) | done |
| 2 | training + MLflow tracking | done |
| 2 | orchestration (Prefect) | done |
| 3 | FastAPI + Docker deployment | done |
| 4 | Evidently monitoring + drift | todo |

Design details: [`docs/architecture.md`](docs/architecture.md).

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

## orchestration

```bash
prefect server start                    # then, in another shell:
python flows/pipeline.py                # one daily run: ingest -> validate -> features -> predict
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
docker run --rm -p 8000:8000 \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -v "$(pwd)/data:/app/data" \
  streamflow-api
```

Then `curl localhost:8000/health` or open `http://localhost:8000/docs`.
Run `python -m streamflow.train` at least once first, or there is no model to load.

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
  monitor.py              # Evidently drift  [stub]
flows/pipeline.py         # Prefect flows: daily, retrain, tuning
prefect.yaml              # deployment definitions
notebooks/                # eda, feature engineering, tuning, drift experiment
data/                     # DVC-tracked
docs/architecture.md
tests/
Dockerfile
```

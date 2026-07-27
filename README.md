# streamflow-mlops

MLOps pipeline forecasting next-day streamflow for the Des Plaines River at
Riverside, IL (USGS 05532500). ADSP 32021 final project - Team 3 (Tori, Janvi,
Daniel, Saya).

## status

| stage | what | status |
|-------|------|--------|
| 1 | ingestion (USGS + Open-Meteo -> daily parquet, 1944-present) | done |
| 1 | EDA | done |
| - | DVC data versioning | done |
| 1 | feature engineering first pass | done |
| 2 | training + MLflow registry | todo |
| 2 | orchestration (Prefect/Airflow, TBD) | todo |
| 3 | FastAPI + Docker deployment | todo |
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
python -m streamflow.ingest        # ~10s -> data/raw/streamflow.parquet
jupyter notebook notebooks/01_eda.ipynb
```

Data is DVC-tracked: the `.dvc` pointer is committed, the parquet isn't.
Re-run ingest to rebuild (or `dvc pull` once a shared remote exists).

## layout

```
config/config.yaml        # gauge, urls, params, target - single source of truth
streamflow/               # ingest [done] | features, train, serve, monitor [stubs]
flows/pipeline.py         # orchestrator-agnostic step list
notebooks/01_eda.ipynb
data/raw/                 # DVC-tracked
docs/architecture.md
tests/
```

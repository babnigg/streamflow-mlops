"""streamflow-mlops: next-day streamflow forecasting pipeline.

    config    - load config/config.yaml
    ingest    - USGS streamflow + Open-Meteo weather -> parquet  [done]
    features  - modeling matrix                                 [stub]
    train     - MLflow tracking + registry                      [stub]
    serve     - FastAPI inference                               [stub]
    monitor   - Evidently drift                                 [stub]
"""

__version__ = "0.1.0"

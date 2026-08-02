"""streamflow-mlops: next-day streamflow forecasting pipeline.

    config    - load config/config.yaml
    ingest    - USGS streamflow + Open-Meteo weather -> parquet  [done]
    validate  - data quality checks on the raw table             [done]
    features  - modeling matrix                                  [done]
    tune      - hyperparameter search (grid/random/bayesian)      [done]
    train     - fit with best params, log to MLflow               [done]
    predict   - score the latest row with the deployed model      [done]
    serve     - FastAPI inference                                 [done]
    monitor   - Evidently drift                                   [stub]
"""

__version__ = "0.1.0"

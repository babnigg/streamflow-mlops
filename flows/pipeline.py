"""Prefect orchestration for the streamflow pipeline: ingest -> features -> train.
Run:  python -m flows.pipeline
"""

from prefect import flow, task

from streamflow.ingest import build_and_save as ingest_build_and_save
from streamflow.features import build_and_save as features_build_and_save
from streamflow.train import main as train_main
from streamflow.tune import main as tune_main
from streamflow.config import CONFIG

# ---------------------------------------------------------------------------
# Daily pipeline: ingest -> features -> train (uses existing best_params.yaml)
# ---------------------------------------------------------------------------

@task(retries=2, retry_delay_seconds=60, log_prints=True)
def ingest_task():
    return ingest_build_and_save(CONFIG)

@task(log_prints=True)
def features_task(raw_df):
    return features_build_and_save()

@task(log_prints=True)
def train_task(feature_df):
    model, run_id = train_main()
    print(f"MLflow run: {run_id}")
    return run_id

@flow(name="streamflow-daily-pipeline")
def daily_pipeline():
    raw_df = ingest_task()
    feature_df = features_task(raw_df)
    run_id = train_task(feature_df)
    print(f"pipeline complete — run: {run_id}")
    return run_id


# ---------------------------------------------------------------------------
# Separate tuning flow: run on demand, not on the daily schedule
# ---------------------------------------------------------------------------

@task(log_prints=True)
def tune_task():
    return tune_main()  # writes config/best_params.yaml when done

@flow(name="streamflow-tuning")
def tuning_flow():
    tune_task()

if __name__ == "__main__":
    daily_pipeline()

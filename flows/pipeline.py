from prefect import flow, task

from streamflow.ingest import build_and_save as ingest_build_and_save
from streamflow.features import build_and_save as features_build_and_save
from streamflow.train import main as train_main
from streamflow.tune import main as tune_main
from streamflow.validate import validate as validate_data
from streamflow.predict import predict_next_day
from streamflow.config import CONFIG

@task(retries=2, retry_delay_seconds=60, log_prints=True)
def ingest_task():
    return ingest_build_and_save(CONFIG)

@task(log_prints=True)
def data_check_task(raw_df):
    validate_data(raw_df)
    return raw_df

@task(log_prints=True)
def features_task(raw_df):
    return features_build_and_save()

@task(log_prints=True)
def predict_task(feature_df, raw_df):
    return predict_next_day(raw_df)

@flow(name="streamflow-daily-pipeline")
def daily_pipeline():
    raw_df = ingest_task()
    checked_df = data_check_task(raw_df)
    feature_df = features_task(checked_df)
    prediction = predict_task(feature_df, checked_df)
    print(f"pipeline complete — prediction: {prediction}")
    return prediction

@task(log_prints=True)
def train_task():
    model, run_id = train_main()
    print(f"MLflow run: {run_id}")
    return run_id

@flow(name="streamflow-retrain")
def retrain_flow():
    train_task()

@task(log_prints=True)
def tune_task():
    return tune_main()

@flow(name="streamflow-tuning")
def tuning_flow():
    tune_task()

if __name__ == "__main__":
    daily_pipeline()

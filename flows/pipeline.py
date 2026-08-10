from prefect import flow, task

from streamflow.ingest import build_and_save as ingest_build_and_save
from streamflow.features import build_and_save as features_build_and_save
from streamflow.train import main as train_main
from streamflow.tune import main as tune_main
from streamflow.validate import validate as validate_data
from streamflow.predict import predict_next_day
from streamflow.monitor import run as monitor_run
from streamflow.registry import champion_version
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
    decision = monitor_task(prediction)

    # Only a sustained performance loss retrains. Data-quality failures and
    # input drift are logged and left alone: retraining cannot fix a broken
    # feed, and a flood drifts the inputs while the model is still fine.
    if decision["action"] == "retrain":
        print(f"retraining: {decision['reason']}")
        train_task()

    print(f"pipeline complete — prediction: {prediction}")
    return {"prediction": prediction, "decision": decision}

@task(log_prints=True)
def monitor_task(prediction):
    """Score, measure drift, decide. Returns the decision; acting on it is the
    flow's job, so detection and action stay separable."""
    return monitor_run()

@task(log_prints=True)
def train_task():
    """Retrain and put the candidate through the promotion gate.

    train_main registers a new version and only moves the champion alias if the
    candidate wins on the held-out window, so this task can run on a false alarm
    without changing what is served.
    """
    model, run_id = train_main()
    print(f"MLflow run: {run_id} | champion now v{champion_version()}")
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

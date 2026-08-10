"""Load config/config.yaml (repo-root relative, works from any cwd)."""

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel: str) -> Path:
    """Repo-relative config path -> absolute path."""
    return REPO_ROOT / rel


CONFIG = load_config()

def _default_tracking_uri() -> str:
    """Anchor a relative file store to the repo root, not the cwd.

    `file:./mlruns` resolves against whatever directory the process started in,
    so running from notebooks/ creates a second empty store and the registry
    then reports no champion.
    """
    uri = CONFIG["mlflow"]["tracking_uri"]
    prefix = "file:"
    if uri.startswith(prefix):
        path = uri[len(prefix):]
        if not Path(path).is_absolute():
            return (REPO_ROOT / path.lstrip("./")).as_uri()
    return uri


# The serving container must not import the tuning stack, so MLflow settings
# live here rather than in tune.py. Env var wins: the container points at a mount.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI") or _default_tracking_uri()
EXPERIMENT_DEPLOY = CONFIG["mlflow"]["experiment_deploy"]
EXPERIMENT_TUNE = CONFIG["mlflow"]["experiment_tune"]
REGISTERED_MODEL = CONFIG["mlflow"]["registered_model"]
CHAMPION_ALIAS = CONFIG["mlflow"]["champion_alias"]

# We run MLflow on a file store deliberately - no tracking server to stand up.
# mlflow 3 requires this opt-in to allow that.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

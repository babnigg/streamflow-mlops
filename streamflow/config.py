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


# Lives here rather than in tune.py so the serving path can reach it without
# importing the tuning module (and therefore optuna) into the container.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")

# mlflow 3.x refuses a local file store unless this is set. We use one on
# purpose (no tracking server for a course project), so opt in once here and
# keep host and container behaving identically.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

CONFIG = load_config()

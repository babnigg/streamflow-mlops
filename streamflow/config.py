"""Load config/config.yaml (repo-root relative, works from any cwd)."""

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

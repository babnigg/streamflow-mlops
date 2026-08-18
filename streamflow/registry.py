"""Model registry and the champion/challenger promotion gate.

A retrain produces a *candidate*. Serving loads whatever carries the champion
alias. Nothing moves between the two without winning a comparison, because
monitoring can fire on inputs that are valid but wrong - USGS ESTIMATED and ICE
qualifiers pass validation - and an unguarded retrain would then ship a worse
model while the alert log records the problem as remediated.

Candidate and champion are scored on the *same* held-out window. The champion's
own training metrics are not comparable: it was fitted when the split ended
somewhere else, so its recorded score describes different days.

    python -m streamflow.registry          # show the registry state
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import mlflow
import mlflow.xgboost
import numpy as np

from .config import CONFIG, MLFLOW_TRACKING_URI, REGISTERED_MODEL, CHAMPION_ALIAS

PROMOTION = CONFIG["promotion"]


def _client():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.tracking.MlflowClient()


def champion_uri() -> str:
    return f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}"


def load_champion():
    """The deployed model, or None when nothing has been promoted yet.

    The alias makes the *lookup* portable, but a file store still records an
    absolute artifact path, so a store written on the host is unreadable from
    inside the container. When the alias URI cannot be opened we rebuild the
    path from the tracking root we were actually pointed at.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        return mlflow.xgboost.load_model(champion_uri())
    except Exception:
        pass

    local = _champion_local_path()
    if local is None:
        return None
    try:
        return mlflow.xgboost.load_model(str(local))
    except Exception:
        return None


def _champion_local_path():
    """Where the champion's artifacts sit under this tracking root, if local."""
    root = MLFLOW_TRACKING_URI
    if not root.startswith("file:"):
        return None
    root = Path(url2pathname(urlparse(root).path))

    try:
        mv = _client().get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
    except Exception:
        return None

    model_id = str(mv.source).rsplit("/", 1)[-1]      # models:/m-<id>
    for exp in root.glob("*/models"):
        candidate = exp / model_id / "artifacts"
        if (candidate / "MLmodel").exists():
            return candidate
    return None


def champion_version() -> str | None:
    """Version carrying the champion alias, always as a string - MLflow returns
    it as an int here and as a string from search, and an int/str mismatch would
    silently make rollback treat the current champion as a rollback target."""
    try:
        v = _client().get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS).version
        return str(v)
    except Exception:
        return None


def evaluate(model, X_test, y_test, persistence_column: str = "log_streamflow_t") -> dict:
    """Score a model on a held-out window. PI needs the persistence column, which
    is a feature, so this takes X_test rather than predictions alone."""
    preds = model.predict(X_test)
    err = np.asarray(preds) - np.asarray(y_test)
    out = {"rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean())}

    y = np.asarray(y_test, dtype=float)
    ss_res, ss_tot = float((err ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    out["nse"] = float(1 - ss_res / ss_tot) if ss_tot else float("nan")

    if persistence_column in getattr(X_test, "columns", []):
        pers = np.asarray(X_test[persistence_column], dtype=float)
        ss_per = float(((y - pers) ** 2).sum())
        out["pi"] = float(1 - ss_res / ss_per) if ss_per else float("nan")
    else:
        out["pi"] = float("nan")
    return out


def compare(candidate: dict, champion: dict | None, metric: str | None = None,
            tolerance: float | None = None) -> dict:
    """Decide whether the candidate should be promoted.

    Higher is better for pi/nse, lower for rmse/mae. A missing champion promotes
    unconditionally: something has to serve. A non-finite candidate score never
    promotes - an unmeasurable model is not a better one.
    """
    metric = metric or PROMOTION["metric"]
    tol = PROMOTION["tolerance"] if tolerance is None else tolerance
    lower_is_better = metric in ("rmse", "mae")

    cand = candidate.get(metric, float("nan"))
    if cand is None or not np.isfinite(cand):
        return {"promote": False, "metric": metric, "candidate": cand, "champion": None,
                "reason": f"candidate {metric} is not finite"}

    if champion is None:
        return {"promote": True, "metric": metric, "candidate": float(cand), "champion": None,
                "reason": "no champion yet - first model is promoted by default"}

    champ = champion.get(metric, float("nan"))
    if champ is None or not np.isfinite(champ):
        return {"promote": True, "metric": metric, "candidate": float(cand), "champion": champ,
                "reason": f"champion {metric} is not measurable on this window"}

    margin = (champ - cand) if lower_is_better else (cand - champ)
    promote = bool(margin >= -tol)
    verb = "beats" if margin >= 0 else "is within tolerance of" if promote else "loses to"
    return {"promote": promote, "metric": metric,
            "candidate": float(cand), "champion": float(champ), "margin": float(margin),
            "reason": (f"candidate {metric} {cand:+.4f} {verb} champion {champ:+.4f} "
                       f"(margin {margin:+.4f}, tolerance {tol})")}


def promote(version: str) -> None:
    _client().set_registered_model_alias(REGISTERED_MODEL, CHAMPION_ALIAS, version)


def rollback() -> str:
    """Move the champion alias back to the last version that passed the gate.

    Not simply the previous version: rejected candidates stay registered for
    audit, so counting backwards would deploy the very model the gate refused.
    Deliberately one call rather than a retrain - recovery from a bad promotion
    should not depend on the training pipeline being healthy.

    Only versions below the current champion are eligible. `promoted` records the
    gate's verdict, not what is deployed, so a version an earlier rollback moved
    away from still carries it - without the bound, a second rollback rolls
    forward onto the model we just backed out of.
    """
    client = _client()
    current = champion_version()
    versions = sorted(client.search_model_versions(f"name='{REGISTERED_MODEL}'"),
                      key=lambda v: int(v.version), reverse=True)

    for v in versions:
        if current is not None and int(v.version) >= int(current):
            continue
        if client.get_run(v.run_id).data.tags.get("promoted") == "true":
            promote(v.version)
            return str(v.version)

    raise RuntimeError(
        "no previously promoted version to roll back to - fewer than two have passed the gate"
    )


def history(limit: int = 10) -> list[dict]:
    client = _client()
    champ = champion_version()
    rows = []
    for v in sorted(client.search_model_versions(f"name='{REGISTERED_MODEL}'"),
                    key=lambda v: int(v.version), reverse=True)[:limit]:
        run = client.get_run(v.run_id)
        rows.append({"version": str(v.version), "champion": str(v.version) == champ,
                     "run_id": v.run_id[:8],
                     "pi": run.data.metrics.get("test_pi"),
                     "rmse": run.data.metrics.get("test_rmse"),
                     "promoted": run.data.tags.get("promoted")})
    return rows


def main():
    rows = history()
    if not rows:
        print(f"no versions registered for '{REGISTERED_MODEL}' - run python -m streamflow.train")
        return rows
    print(f"{REGISTERED_MODEL}  (champion alias -> version {champion_version()})")
    for r in rows:
        mark = "*" if r["champion"] else " "
        pi = f"{r['pi']:+.4f}" if r["pi"] is not None else "   n/a"
        print(f" {mark} v{r['version']:<3} run {r['run_id']}  pi {pi}  promoted={r['promoted']}")
    return rows


if __name__ == "__main__":
    main()

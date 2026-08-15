"""Dependency-drift tests.

Two checks, run independently since they catch different things:
  static   requirements.txt and requirements-serve.txt agree on every shared
           pin -- verified from the files alone, no install required.
  runtime  the environment these tests execute in actually has what's pinned
           installed -- verified against importlib.metadata.

Run:  pytest tests/test_dependencies.py -v
"""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _parse_pins(path: Path) -> dict[str, str]:
    """name -> pinned version, from a `pkg==1.2.3` / `pkg[extra]==1.2.3` line.

    Anything not an exact pin (a comment, a blank line, a bare `-r other.txt`)
    is skipped rather than failing the parse -- an unpinned line is a separate,
    louder problem the header comment in requirements.txt already forbids.
    """
    pins = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "==" not in line:
            continue
        name, _, ver = line.partition("==")
        name = re.sub(r"\[.*\]", "", name).strip().lower()
        pins[name] = ver.strip()
    return pins


REQUIREMENTS = _parse_pins(ROOT / "requirements.txt")
REQUIREMENTS_SERVE = _parse_pins(ROOT / "requirements-serve.txt")

# requirements-serve.txt deliberately ships slimmer *distributions* of the
# same project for two packages (see its header comment): xgboost-cpu drops
# ~400MB of unused CUDA, mlflow-skinny drops the server/UI stack. Different
# PyPI name, same code, so the skew check must compare through this alias
# rather than treating them as unrelated packages.
SERVE_NAME_ALIASES = {
    "xgboost-cpu": "xgboost",
    "mlflow-skinny": "mlflow",
}


# --------------------------------------------------------------- static skew
@pytest.mark.parametrize("name", sorted(REQUIREMENTS_SERVE))
def test_serve_pin_matches_training_pin(name):
    """Every package requirements-serve.txt ships must be pinned to the exact
    version requirements.txt trained against -- a silent gap here is the
    pickle-compat break requirements-serve.txt's own header warns about."""
    training_name = SERVE_NAME_ALIASES.get(name, name)
    assert training_name in REQUIREMENTS, (
        f"{name} is pinned in requirements-serve.txt but {training_name} is "
        f"missing from requirements.txt -- serving on a package training never used"
    )
    assert REQUIREMENTS_SERVE[name] == REQUIREMENTS[training_name], (
        f"{training_name}: requirements.txt pins {REQUIREMENTS[training_name]}, "
        f"requirements-serve.txt pins {name}=={REQUIREMENTS_SERVE[name]}"
    )


# -------------------------------------------------------------- runtime drift
@pytest.mark.parametrize("name,pinned", sorted(REQUIREMENTS.items()))
def test_installed_version_matches_pin(name, pinned):
    """The environment running this test suite must be the environment the
    pins describe. A resolver that silently satisfied a pin with a nearby
    version (or a venv that predates the last bump) is train/serve skew by
    another name -- it just shows up as a wrong answer instead of an error."""
    try:
        installed = version(name)
    except PackageNotFoundError:
        pytest.fail(f"{name}=={pinned} is pinned but not installed in this environment")
    assert installed == pinned, (
        f"{name}: pinned {pinned}, installed {installed} -- "
        f"run `pip install -r requirements.txt` to resync"
    )

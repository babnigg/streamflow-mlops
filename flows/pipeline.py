"""Pipeline as plain callables - orchestrator (Prefect/Airflow) wraps these later.

Run:  python -m flows.pipeline
"""

from streamflow import features, ingest

STEPS = [
    ("ingest", ingest.build_and_save),
    ("features", features.build_and_save),
    # ("train", ...), ("monitor", ...)
]


def run():
    for name, fn in STEPS:
        print(f"[flow] {name}")
        fn()


if __name__ == "__main__":
    run()

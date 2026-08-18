# submission checklist

Requirements quoted from Session 3 p93 (final project) and p92 (logistics).
Hold the slides against this. `[x]` = evidence exists, `[ ]` = gap.

## deliverables

- [x] **1. GitHub Repository** — `github.com/babnigg/streamflow-mlops`, CI on push/PR
- [ ] **2. Presentation Slides (PPT)** — not in the project tree; commit the file
      alongside the repo so both deliverables submit together
- [x] Per-member contribution statement (p92) — `docs/architecture.md` work split.
      Must also appear on a slide.

## required core stages

- [x] **Data Ingestion & Baselines** — `ingest.py` (USGS OGC + Open-Meteo, keyless,
      30,180 days), `validate.py`, `features.py`. Persistence is the denominator of
      the monitored metric, so the baseline is structural rather than a footnote.
- [x] **Pipeline Automation & Experiment Tracking** — `flows/pipeline.py` (Prefect
      daily/retrain/tuning), CI on push/PR, MLflow on DagsHub.
- [x] **Containerization & Deployment** — `Dockerfile` (serving-only image),
      `serve.py`. Verified 2026-08-18: image builds, container answers `/health`,
      `/model`, `/predict/latest` against the remote registry with no local mlruns.
- [x] **Production Monitoring & Drift Simulation** — `monitor.py` three pillars,
      Evidently, calibrated thresholds, `05_monitoring.ipynb`.

## slide outline (the de facto rubric)

| # | requirement | evidence | state |
|---|---|---|---|
| 1 | Problem Statement & EDA | `01_eda.ipynb`; `eda_overview` `record_full` `seasonality` `target_skew` `rain_response` `drivers_correlation` | ready |
| 2 | Evaluation Metric justification | `architecture.md` §metrics — persistence scores NSE 0.87 alone, so NSE 0.91 is an 18% error reduction; a stale model reads NSE +0.37 but PI −1.46 | ready |
| 3 | System Architecture diagram | `architecture.md` has ASCII only | **needs a rendered diagram — rubric says "visual"** |
| 4 | Experimentation Tracking dashboard | tuning experiment live on DagsHub: grid 0.2859 / random 0.2863 / bayesian 0.2851 CV RMSE | **needs a screenshot committed** |
| 5 | Deployment & Monitoring | `api_swagger.jpg`; `drift_2026-08-16.html` | ready |
| 6 | Drift Analysis on corrupted data | `corruption_psi.png` — clean quiet on all 4 columns; −15 C → tmax 8.98, 5x rain → precip 1.24, 10x gauge → streamflow 3.86 | ready |
| 7 | Repository link visible | — | **put the URL on a slide** |

## gaps, in priority order

1. **Deck does not exist.** 30% of the grade, named deliverable.
2. **Architecture diagram** must be a rendered visual, not ASCII.
3. **DagsHub tuning screenshot** — the dashboard is the literal wording of the
   requirement. Needs a browser logged into DagsHub.
4. **Repo URL on a slide.**

## keep separate on the slides

Drift analysis and the promotion gate answer different questions and only the
first is on the rubric:

- **Drift Analysis** — corrupted *inputs*, monitoring responds (`corruption_psi.png`)
- **Promotion gate** — a crippled *model*, the gate rejects it (registry v3)

Presenting the gate under "Drift Analysis" leaves the graded line without evidence.

## verified 2026-08-18

- clean clone: 60 files, no `.env`, no `.dvc/config.local`, no tokens in any
  tracked file
- `dvc pull` from a cold clone: fails without credentials, and with a token
  returns both parquets at the md5s the committed pointers name
- container serves champion v2 (`run d6476b51`) from DagsHub with no local mlruns
- suite 124 passed / 1 skipped

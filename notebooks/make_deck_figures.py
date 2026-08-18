"""Regenerate the EDA and model figures used in the presentation.

    python notebooks/make_deck_figures.py     ->  reports/figures/*.png

Deterministic for a given feature matrix, so rerunning after an ingest picks up
new data without any hand-editing of the deck. The monitoring figures
(flood_hydrograph, pi_healthy_vs_stale, psi_dilution) come from
notebooks/05_monitoring.ipynb, which has to fit models to produce them.

All styling lives in figstyle.py - the palette is the presentation template's
own theme, so figures sit on a slide rather than in a white box.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # writes files only; no display, no Qt

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figstyle import apply, save, WIDE, HALF, STACK, TEAL, CLAY, MUTED, GOLD, BROWN, INK  # noqa: E402

from streamflow.config import CONFIG, resolve  # noqa: E402
from streamflow.features import TARGET_COLUMN  # noqa: E402

apply()


def load():
    raw = pd.read_parquet(resolve(CONFIG["data"]["raw_path"]))
    raw["date"] = pd.to_datetime(raw["date"])
    feats = pd.read_parquet(resolve(CONFIG["data"]["features_path"]))
    feats["date"] = pd.to_datetime(feats["date"])
    return raw, feats


def _record_panel(ax, raw):
    ax.plot(raw.date, raw.streamflow_cfs, color=TEAL, lw=0.4)
    ax.set_yscale("log")
    ax.set_ylabel("streamflow (cfs, log)")
    ax.axhline(raw.streamflow_cfs.median(), color=MUTED, lw=1.2, ls="--")


def _seasonality_panel(ax, raw):
    by_month = [raw.loc[raw.date.dt.month == m, "streamflow_cfs"].dropna()
                for m in range(1, 13)]
    bp = ax.boxplot(by_month, showfliers=False, patch_artist=True, widths=.6,
                    medianprops=dict(color=INK, lw=1.6))
    for patch in bp["boxes"]:
        patch.set_facecolor(TEAL); patch.set_alpha(.55); patch.set_edgecolor(MUTED)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set_color(MUTED)
    ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_ylabel("streamflow (cfs)")


def fig_eda_overview(raw):
    """Both EDA panels in one figure.

    Two full-width figures stacked do not fit under a title on a 5.625in slide,
    and hand-placing them means they never line up. One figure, one paste.
    """
    fig, axes = plt.subplots(2, 1, figsize=STACK)
    _record_panel(axes[0], raw)
    _seasonality_panel(axes[1], raw)
    save(fig, "eda_overview")
    plt.close(fig)


def fig_record(raw):
    """Standalone, for a slide that wants only the record."""
    fig, ax = plt.subplots(figsize=WIDE)
    _record_panel(ax, raw)
    save(fig, "record_full")
    plt.close(fig)


def fig_seasonality(raw):
    """Standalone, for a slide that wants only the seasonal view."""
    fig, ax = plt.subplots(figsize=WIDE)
    _seasonality_panel(ax, raw)
    save(fig, "seasonality")
    plt.close(fig)


def fig_drivers(feats):
    """What actually predicts tomorrow.

    The honest version of 'which feature matters': today's flow dominates, and
    precipitation is only the strongest *weather* driver. That gap is the whole
    reason persistence is a hard baseline and why we score against it.
    """
    pairs = [("today's flow (log)", "log_streamflow_t"),
             ("today's flow", "streamflow_t"),
             ("precip, 7d sum", "precip_sum_7d"),
             ("precip, 3d sum", "precip_sum_3d"),
             ("precip today", "precip_mm"),
             ("max temp", "tmax_c"),
             ("min temp", "tmin_c")]
    pairs = [(lbl, c) for lbl, c in pairs if c in feats.columns]
    corr = [abs(feats[c].corr(feats[TARGET_COLUMN])) for _, c in pairs]
    labels = [lbl for lbl, _ in pairs]

    order = np.argsort(corr)
    corr = [corr[i] for i in order]
    labels = [labels[i] for i in order]
    colors = [TEAL if "flow" in l else GOLD if "precip" in l else BROWN for l in labels]

    fig, ax = plt.subplots(figsize=HALF)
    ax.barh(labels, corr, color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("|correlation| with next-day flow")
    save(fig, "drivers_correlation")
    plt.close(fig)


def fig_target_skew(raw):
    """Why the target is modelled in log space."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    flow = raw.streamflow_cfs.dropna()
    axes[0].hist(flow, bins=60, color=CLAY)
    axes[0].set_ylabel("days"); axes[0].set_xlabel("streamflow (cfs)")
    axes[1].hist(np.log1p(flow), bins=60, color=TEAL)
    axes[1].set_xlabel("log1p(streamflow)")
    save(fig, "target_skew")
    plt.close(fig)


def fig_persistence(feats):
    """Today's flow against tomorrow's, on the 1:1 line.

    The single most useful EDA panel for this project: it shows the baseline we
    have to beat. Points sitting on the diagonal are days where "tomorrow equals
    today" was already the right answer, which is most of them - and that is the
    whole justification for scoring against persistence rather than the mean.
    """
    d = feats[["log_streamflow_t", TARGET_COLUMN]].dropna()
    fig, ax = plt.subplots(figsize=HALF)
    hb = ax.hexbin(d["log_streamflow_t"], d[TARGET_COLUMN], gridsize=55,
                   bins="log", mincnt=1, cmap="BuGn", linewidths=0)
    lo = float(min(d["log_streamflow_t"].min(), d[TARGET_COLUMN].min()))
    hi = float(max(d["log_streamflow_t"].max(), d[TARGET_COLUMN].max()))
    ax.plot([lo, hi], [lo, hi], color=CLAY, lw=1.8, ls="--")
    ax.set_xlabel("log flow today")
    ax.set_ylabel("log flow tomorrow")
    cb = fig.colorbar(hb, ax=ax, pad=.02)
    cb.set_label("days", color=MUTED)
    cb.outline.set_edgecolor(MUTED)
    save(fig, "persistence_scatter")
    plt.close(fig)


def fig_rain_response(raw):
    """What a heavy rain day does to flow over the following week.

    Justifies the rolling precipitation features: the response is not same-day,
    it builds over two to three days, which is why 3/7/14/30-day precipitation
    sums are in the feature set rather than precipitation alone.
    """
    r = raw.dropna(subset=["streamflow_cfs", "precip_mm"]).reset_index(drop=True)
    heavy = r.index[r["precip_mm"] > r["precip_mm"].quantile(0.99)]
    heavy = heavy[(heavy > 2) & (heavy < len(r) - 8)]

    offsets = range(-2, 8)
    curves = []
    for i in heavy:
        base = r.loc[i - 1, "streamflow_cfs"]
        if base and base > 0:
            curves.append([r.loc[i + k, "streamflow_cfs"] / base for k in offsets])
    arr = np.array(curves)
    med = np.median(arr, axis=0)
    lo, hi = np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)

    fig, ax = plt.subplots(figsize=HALF)
    ax.fill_between(list(offsets), lo, hi, color=TEAL, alpha=.22, lw=0)
    ax.plot(list(offsets), med, color=TEAL)
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.axvline(0, color=GOLD, lw=1.6)
    ax.set_xlabel("days from a top-1% rainfall day")
    ax.set_ylabel("flow relative to the day before")
    save(fig, "rain_response")
    plt.close(fig)


def fig_corruption_psi(feats):
    """Per-column PSI under injected corruption, against the clean control.

    A table of PSI values proves monitoring responds; a chart shows the two
    things that matter at a glance - the clean window sits under every
    threshold, and each corruption moves only the column it touched.

    Needs scored predictions: run `python -m streamflow.monitor` first.
    """
    from streamflow import monitor
    from streamflow.monitor import MON

    scored = pd.read_parquet(resolve("data/monitoring/predictions.parquet"))
    m = monitor.with_predictions(feats, scored)
    cur = m[m.date > m.date.max() - pd.Timedelta(days=MON["drift_window_days"])]
    ref = monitor.seasonal_reference(m, cur)

    scenarios = [("clean", None, None),
                 ("-15 C temp", "tmax_c", lambda s: s - 15),
                 ("5x rainfall", "precip_mm", lambda s: s * 5),
                 ("gauge 10x", "streamflow_t", lambda s: s * 10)]
    rows = []
    for label, col, fn in scenarios:
        c2 = cur if col is None else cur.assign(**{col: fn(cur[col])})
        d = monitor.drift_report(ref, c2)
        rows.append((label, d["per_column_psi"], d["dataset_drift"]))

    watch = MON["watch_columns"]
    x, width = np.arange(len(watch)), 0.2
    fig, ax = plt.subplots(figsize=WIDE)
    for i, ((label, psi, alarm), colour) in enumerate(zip(rows, [MUTED, TEAL, GOLD, CLAY])):
        ax.bar(x + (i - 1.5) * width, [psi.get(c, np.nan) for c in watch], width,
               label=f"{label}{' - alarm' if alarm else ''}", color=colour)

    # thresholds are calibrated per column, so they are marks on each group
    # rather than one line across the axis
    for j, c in enumerate(watch):
        t = MON["psi_thresholds"][c]
        ax.plot([j - 0.42, j + 0.42], [t, t], color=INK, lw=1.4, ls=(0, (4, 2)),
                zorder=5, label="threshold" if j == 0 else None)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(watch)
    ax.set_ylabel("PSI (log scale)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save(fig, "corruption_psi")
    plt.close(fig)

    for label, psi, alarm in rows:
        print(f"    {label:<12} alarm={str(alarm):<5} "
              + " ".join(f"{c}={psi.get(c, float('nan')):.2f}" for c in watch))


def main():
    raw, feats = load()
    print(f"raw {len(raw):,} rows  features {len(feats):,} rows")
    fig_eda_overview(raw)
    fig_record(raw)
    fig_seasonality(raw)
    fig_drivers(feats)
    fig_persistence(feats)
    fig_rain_response(raw)
    fig_target_skew(raw)
    fig_corruption_psi(feats)
    print("\nremaining monitoring figures come from notebooks/05_monitoring.ipynb")


if __name__ == "__main__":
    main()

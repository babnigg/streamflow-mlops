"""Shared figure styling for the deck.

Colours are the presentation template's own theme palette, so exported figures
sit on a slide without looking pasted in. Backgrounds are transparent for the
same reason - a white box around a plot is the giveaway.

Type is sized for a projector, not a laptop: anything under ~11pt in a figure
scaled to half a slide is unreadable from the back of a room.

    from figstyle import apply, save, TEAL, CLAY, INK
    apply()
    fig, ax = plt.subplots(figsize=WIDE)
    ...
    save(fig, "name")        # -> reports/figures/name.png
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# template theme (ppt/theme/theme1.xml)
INK = "#312f2b"        # dk1  - text and axes
MUTED = "#8a837c"      # dk1 lightened - secondary text, reference lines
TEAL = "#3f8f8c"       # accent1 darkened for contrast on a light slide
CLAY = "#b0553c"       # alarm / failure
OLIVE = "#4c5e37"      # accent2
GOLD = "#c2b25f"       # lt2 darkened - secondary series
BROWN = "#806952"      # accent3

FIGDIR = Path(__file__).resolve().parent.parent / "reports" / "figures"

# Sized against the deck's 10 x 5.625in slide, leaving room for a title. STACK
# is two panels in the space one WIDE figure would occupy - stacking two WIDE
# exports instead overflows the slide and never lines up.
WIDE = (10, 3.6)       # full-width, one panel
STACK = (10, 4.3)      # full-width, two panels
HALF = (8, 4.964)      # half-width, tall enough for a categorical axis
COMPACT = (7.5, 3.4)   # sits beside body text without shrinking to nothing


def apply():
    """Idempotent: safe to call from every notebook."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.transparent": True,
        "savefig.bbox": "tight",
        "font.family": ["Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "figure.constrained_layout.use": True,
    })


def save(fig, name: str) -> Path:
    """Write a transparent PNG for the deck. Deterministic for given data."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / f"{name}.png"
    fig.savefig(path)
    print(f"  saved reports/figures/{name}.png")
    return path

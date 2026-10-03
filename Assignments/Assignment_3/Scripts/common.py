"""
Shared paths and plot style for the Assignment 3 scripts.

Importing this module puts the project root (which holds the sampler library:
Metropolis_Hastings.py, Simulated_Annealing.py, Simulated_Tempering.py, ...)
on sys.path, so the scripts can be run from any working directory.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ASSIGNMENT_DIR.parents[1]
FIG_DIR = ASSIGNMENT_DIR / "Figures"
RESULTS_DIR = ASSIGNMENT_DIR / "Results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Categorical slots (fixed order) and text colours from the reference data-viz palette.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
LINESTYLES = ["-", "--", "-.", ":"]       # secondary encoding when series overlap
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": TEXT_2,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "figure.dpi": 100,
    "savefig.bbox": "tight",
})


def save_fig(fig, name):
    """Save a figure as PDF (for the report) and PNG (for quick viewing)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  saved Figures/{name}.pdf and .png")


def save_results(name, results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  saved Results/{name}.json")

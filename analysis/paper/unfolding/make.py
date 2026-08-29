"""Unfolded cross section, uncertainty budget and post-fit correlations.

    python -m analysis.paper.unfolding.make [names...] [--label L]
"""
from analysis.paper._driver import run

FIGURES = {
    "unfolded": ("analysis.paper.unfolding.fig_unfolded_lep", "main"),
    "budget":   ("analysis.paper.unfolding.fig_budget", "main"),
    "corr":     ("analysis.paper.unfolding.fig_correlation", "main"),
}

DEFAULT_LABEL = "sec5f10"


if __name__ == "__main__":
    run("analysis.paper.unfolding.make", FIGURES, label=DEFAULT_LABEL)

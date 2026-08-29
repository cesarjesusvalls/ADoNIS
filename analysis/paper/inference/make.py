"""Closure fit, coverage and corner plots, all from one fit label.

    python -m analysis.paper.inference.make [names...] [--label L]
"""
from analysis.paper._driver import run

FIGURES = {
    "rates":   ("analysis.paper.inference.fig_rates", "main"),
    "closure": ("analysis.paper.inference.fig_closure_summary", "main", lambda a: (a.label, f"{a.label}_ens")),
    "corner":  ("analysis.paper.inference.fig_corner_all", "main", lambda a: (a.label, a.label)),
}

DEFAULT_LABEL = "sec4_P2"


if __name__ == "__main__":
    run("analysis.paper.inference.make", FIGURES, label=DEFAULT_LABEL)

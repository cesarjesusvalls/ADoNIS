"""Shared paths for analysis/.

The output root comes from adonis.io.output_root ($ADONIS_OUT, else configs/paths.yaml, else
<repo>/output); results and figures are named subdirectories of it.
"""
from __future__ import annotations

from pathlib import Path

from adonis.io import output_root


def results_dir() -> Path:
    """Analysis results: the Jacobian, the fit stages, the benchmarks, the unfolding."""
    return output_root() / "results"


def figures_dir() -> Path:
    """Rendered figures.  ADONIS_PAPER_OUT still overrides, for rebuilding beside a published set."""
    import os
    return Path(os.environ.get("ADONIS_PAPER_OUT") or output_root() / "paper")


def result(label: str, suffix: str = ".npz") -> Path:
    """The file a run label maps to, with the results directory created."""
    d = results_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{label}{suffix}"

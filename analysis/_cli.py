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

FLAT_PRIOR_SCALE = 1e6


def timed_log(prefix=""):
    """A logger stamping seconds since it was created."""
    import time
    t0 = time.time()
    def log(m):
        print(f"[{time.time() - t0:7.1f}s] {prefix}{m}", flush=True)
    return log


def apply_prior_scale(eng, cfg, log=None):
    """Widen the engine's prior by cfg.fit.prior_scale; 0 means flat, applied as FLAT_PRIOR_SCALE.

    Returns the untouched prior.
    """
    import numpy as np
    prior_true = np.asarray(eng.prior, float).copy()
    scale = cfg.fit.prior_scale
    if scale != 1.0:
        eng.prior = eng.prior * (FLAT_PRIOR_SCALE if scale == 0.0 else scale)
    if log:
        log(f"estimator {cfg.fit.estimator.upper()} (prior x{scale:g})")
    return prior_true


def shard_range(base, count, total):
    """The half-open slice of `total` items this shard owns.

    `base` < 0 means every item; otherwise the shard starts at `base` and takes `count`.
    """
    if base < 0:
        return 0, total
    return base, min(base + count, total)


def fig_cfg(z):
    """The `figures:` block of the config that produced `z`, read from the npz rather than the yaml,
    which can change after the fit ran.  Files carrying no cfg_json yield {}, so module defaults stand.
    """
    import json
    if "cfg_json" in z.files:
        return json.loads(str(z["cfg_json"])).get("figures", {}) or {}
    return {}

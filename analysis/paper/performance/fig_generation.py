"""Event-generation throughput, CPU against GPU, split by stage.

Reads the `stage_seconds` block that adonis.workflow.generate_bank writes into every bank manifest, so
the numbers come from real production runs rather than from a benchmark that approximates one.  Point
it at one bank directory per device:

    python -m analysis.paper.performance.fig_generation \
        "EPYC 7713 (1 core)=<dir>" "RTX 2080 Ti=<dir>" "A100-SXM4=<dir>"

WHY THE STAGES ARE SEPARATED.  A single events/s number hides the only thing worth knowing here: the
two stages behave completely differently.

  pre-FSI    the hard vertex -- flux sampling, spectral-function draws, the amplitude.  Host-bound, and
             it gains NOTHING from a GPU: measured 1817 ev/s on one EPYC 7713 core against 1749 on an
             A100 and 902 on a 2080 Ti, i.e. one CPU core beats both cards.
  cascade    the intranuclear cascade, jitted end to end.  This is the part a GPU is for: 55 ev/s on
             the same core against 718 on the A100.
  record/IO  building the flat FSI record and writing the npz.  Host-side, ~150 s/chunk on every
             machine measured, which is what drags the end-to-end speedup from 13x down to ~5x.

Chunk 0 carries JIT compilation and is dropped; the steady state is the median over the rest.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "adonis").is_dir())
sys.path.insert(0, str(ROOT))
from analysis.paper import style

STAGES = [("pre_fsi", "pre-FSI"), ("cascade", "cascade"), ("record_io", "record + I/O")]
S_COLOR = {"pre_fsi": style.C_QE, "cascade": style.C_RES, "record_io": "0.55"}


def read_bank(path):
    """(device-independent) steady-state seconds per stage and events per chunk, from a bank manifest."""
    man = json.loads((Path(path) / "manifest.json").read_text())
    st = man.get("stage_seconds")
    if not st:
        raise SystemExit(f"{path}/manifest.json has no 'stage_seconds' -- regenerate the bank, or the "
                         f"bank predates the per-stage timing (adonis.workflow.generate_bank).")
    rows = st[1:] if len(st) > 1 else st
    n = float(np.median([r["n_events"] for r in rows]))
    return {k: float(np.median([r[k] for r in rows])) for k, _ in STAGES}, n, len(rows)


def main(*specs):
    """Each spec is 'Label=path/to/bank'."""
    if not specs:
        raise SystemExit(__doc__)
    devs = []
    for s in specs:
        lab, _, path = str(s).partition("=")
        t, n, nch = read_bank(path)
        devs.append((lab, t, n, nch))
        tot = sum(t.values())
        print(f"{lab:26s} {nch} steady chunks, {n:,.0f} ev/chunk | "
              + "  ".join(f"{nm} {t[k]:7.1f}s ({n/t[k]:6.0f} ev/s)" for k, nm in STAGES)
              + f" | end-to-end {n/tot:6.0f} ev/s")

    fig, ax = plt.subplots(1, 2, figsize=(style.PANEL_W * 2.1, style.PANEL_H * 1.25))
    y = np.arange(len(devs))

    for k, nm in STAGES:
        ax[0].barh(y + 0.26 - 0.26 * [s[0] for s in STAGES].index(k), 0, height=0)
    h = 0.26
    for si, (k, nm) in enumerate(STAGES):
        ax[0].barh(y + (1 - si) * h, [d[2] / d[1][k] for d in devs], height=h,
                   color=S_COLOR[k], label=nm, edgecolor="white", lw=0.6)
    ax[0].set_xscale("log")
    ax[0].set_yticks(y); ax[0].set_yticklabels([d[0] for d in devs], fontsize=8)
    ax[0].set_xlabel("events / s", fontsize=9)
    ax[0].legend(fontsize=7.5, loc="lower right", frameon=False)
    ax[0].tick_params(labelsize=8, top=False, right=False)
    ax[0].set_title("(a) throughput by stage", fontsize=9, loc="left")

    left = np.zeros(len(devs))
    for k, nm in STAGES:
        w = np.array([d[1][k] / sum(d[1].values()) for d in devs])
        ax[1].barh(y, w, left=left, height=0.55, color=S_COLOR[k], label=nm, edgecolor="white", lw=0.6)
        left += w
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_xlim(0, 1); ax[1].set_xlabel("fraction of wall clock", fontsize=9)
    ax[1].tick_params(labelsize=8, top=False, right=False)
    ax[1].set_title("(b) time budget per chunk", fontsize=9, loc="left")
    for i, d in enumerate(devs):
        ax[1].text(1.01, y[i], f"{d[2]/sum(d[1].values()):.0f} ev/s", va="center", fontsize=7.5)

    fig.tight_layout()
    style.save(fig, "performance_generation")


if __name__ == "__main__":
    style.use()
    main(*sys.argv[1:])

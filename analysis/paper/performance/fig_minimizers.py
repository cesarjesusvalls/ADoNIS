"""Minimiser cost with and without gradients: Gauss--Newton against MIGRAD.

Reads the workload surface written by `python -m adonis.fit.bench_fair` (output/altgen/bench_fair_*.npz),
whose `rows` is one record per (n dials, statistical realisation, method) with the wall time and the
number of full passes over the resident events.

    python -m analysis.paper.performance.fig_minimizers [npz-stem]

THE THREE ARMS differ only in which derivative object the algorithm may ask for -- the cost function,
the events and the convergence target are identical:

  gn          Gauss--Newton on the autodiff Jacobian: one forward-mode JVP per dial per iteration.
  migrad+g    MIGRAD driven by the reverse-mode gradient: one VJP per call, dial count irrelevant.
  migrad      MIGRAD with no gradient, so it builds one by finite differences: ~2n extra cost-function
              calls per gradient, each a full pass over ~2.4M events.

MEDIAN OVER STATISTICAL REALISATIONS, min--max as the band.  The fits are run on closure truth plus
per-bin Gaussian noise, not on Asimov data, because Gauss--Newton drops a Hessian term proportional to
the residual: on a perfect closure that term vanishes and GN silently becomes Newton, which flatters it
everywhere.  The spread across realisations is carried because GN's advantage is seed-dependent.

Compilation is excluded -- every callable is warmed up on its exact argument shapes before the clock
starts, which is the difference between measuring arithmetic and measuring XLA.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "adonis").is_dir())
sys.path.insert(0, str(ROOT))
from analysis.paper import style                                      # noqa: E402

METHODS = [("gn", "Gauss–Newton (autodiff Jacobian)", style.C_QE, "o", "-"),
           ("migrad+g", "MIGRAD + reverse-mode gradient", style.C_RES, "s", "-"),
           ("migrad", "MIGRAD, finite differences", style.C_TOTAL, "^", "--")]


def load(stem):
    z = np.load(style.ALTGEN / f"{stem}.npz", allow_pickle=True)
    rows = [dict(r) for r in np.asarray(z["rows"])]
    return rows, z


def _series(rows, meth, key):
    """median / min / max of `key` against dial count, over the NOISE realisations only."""
    ns = sorted({r["n"] for r in rows})
    med, lo, hi = [], [], []
    for n in ns:
        v = [r[key] for r in rows
             if r["n"] == n and r["method"] == meth and str(r["tag"]).startswith("noise")
             and r.get("converged", True)]
        if not v:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        med.append(float(np.median(v))); lo.append(float(min(v))); hi.append(float(max(v)))
    return np.array(ns, float), np.array(med), np.array(lo), np.array(hi)


def main(stem="bench_fair_amp_N60000"):
    rows, z = load(stem)
    N = int(z["sig_cap"]); nlive = int(z["nlive"])

    fig, ax = plt.subplots(1, 2, figsize=(style.PANEL_W * 2.1, style.PANEL_H * 1.25))
    for key, A, ylab, title in ((("wall"), ax[0], "wall clock to converge [s]", "(a) time"),
                                (("passes"), ax[1], "full passes over the events", "(b) work")):
        for meth, lab, col, mk, ls in METHODS:
            n, m, lo, hi = _series(rows, meth, key)
            ok = np.isfinite(m)
            A.plot(n[ok], m[ok], ls, color=col, marker=mk, ms=4.5, lw=1.4, label=lab)
            A.fill_between(n[ok], lo[ok], hi[ok], color=col, alpha=0.18, lw=0)
        A.set_yscale("log"); A.set_xlabel("fitted dials $n$", fontsize=9)
        A.set_ylabel(ylab, fontsize=9)
        A.set_title(title, fontsize=9, loc="left")
        A.tick_params(labelsize=8, top=False, right=False)
    ax[0].legend(fontsize=7.5, loc="upper left", frameon=False)

    # The ratio at the largest n is the sentence this figure exists to support; print it rather than
    # writing it on the canvas, so the caption quotes a number the run actually produced.
    for key in ("wall", "passes"):
        n, g, _, _ = _series(rows, "gn", key)
        _, mg, _, _ = _series(rows, "migrad+g", key)
        _, mf, _, _ = _series(rows, "migrad", key)
        i = int(np.nanargmax(np.where(np.isfinite(g), n, np.nan)))
        print(f"  n={n[i]:.0f}, {N:,} ev/sample, {nlive} live bins: {key}  GN {g[i]:.3g}  "
              f"MIGRAD+g {mg[i]:.3g} ({mg[i]/g[i]:.1f}x)  MIGRAD {mf[i]:.3g} ({mf[i]/g[i]:.1f}x)")

    fig.tight_layout()
    style.save(fig, "performance_minimizers")


if __name__ == "__main__":
    style.use()
    main(*(sys.argv[1:2] or []))

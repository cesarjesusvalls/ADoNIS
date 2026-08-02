"""Sec 5 scaling / differentiability-advantage figure (real measured prefactors).

Reads sec5_timing_<tag>.npz (per-op costs on the multisample model) and turns them into the three
quantitative arguments for the differentiable engine:
  (a) non-Gaussian corner cost: Taylor (derivatives once + analytic) vs exact-grid (conditional) vs
      exact-profile -- log scale, the seconds-vs-hours headline.
  (b) fit gradient: autodiff Jacobian (16 jvp, EXACT) vs central finite differences (32 evals, noisy) --
      what a non-differentiable generator pays per gradient.
  (c) derivative-tensor cost vs order m: ~ C(n+m-1,m) sweeps (n=16: 16,136,816,3876) x per-sweep cost.

Usage:  python -m analysis.paper.sec5_methods.fig_scaling [tag]     (default 4ch)
"""
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

C_DIFF, C_GRID, C_PROF = "#1f4b9c", "#7b6f9e", "#c44"


def _fmt(t):
    if t < 1:    return f"{t*1e3:.0f} ms"
    if t < 90:   return f"{t:.1f} s"
    if t < 5400: return f"{t/60:.0f} min"
    return f"{t/3600:.1f} hr"


def main(tag="4ch"):
    style.use()
    z = np.load(style.ALTGEN / f"sec5_timing_{tag}.npz", allow_pickle=True)
    tm = float(z["t_model"]); tj = float(z["t_jac"]); tfd = float(z["t_fdjac"]); td2 = float(z["t_d2m"])
    n = int(z["n_dials"]); nev = int(z["n_events"])
    G, PAIRS, STEPS = 19, n * (n - 1) // 2, 4                     # corner grid / pairs / inner-fit steps

    t_taylor = tj + (n * (n + 1) // 2) * td2                      # J + n(n+1)/2 second derivatives
    t_grid = PAIRS * G * G * tm
    t_profile = PAIRS * G * G * STEPS * tj

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.5, 4.4))

        # (a) corner cost --------------------------------------------------------------------------- #
        vals = [t_taylor, t_grid, t_profile]; cols = [C_DIFF, C_GRID, C_PROF]
        labs = ["Taylor\n(autodiff)", "exact grid\n(conditional)", "exact profile\n(re-fit each node)"]
        a.bar(range(3), vals, color=cols, width=0.62, alpha=0.9, log=True)
        for i, v in enumerate(vals):
            a.text(i, v * 1.4, _fmt(v), ha="center", fontsize=9, fontweight="bold")
        a.set_xticks(range(3)); a.set_xticklabels(labs, fontsize=8.5)
        a.set_ylabel("cost for the 120-pair corner", fontsize=9.5)
        a.set_ylim(t_taylor * 0.3, t_profile * 6)
        a.set_title("(a)  non-Gaussian corner", fontsize=10.5, loc="left")

        # (b) gradient: autodiff vs finite difference ----------------------------------------------- #
        b.bar([0, 1], [tj * 1e3, tfd * 1e3], color=[C_DIFF, C_GRID], width=0.6, alpha=0.9)
        for i, v in enumerate([tj, tfd]):
            b.text(i, v * 1e3 * 1.02, _fmt(v), ha="center", fontsize=9, fontweight="bold")
        b.set_xticks([0, 1]); b.set_xticklabels([f"autodiff\n({n} jvp, exact)", f"finite diff\n({2*n} evals, noisy)"],
                                                fontsize=8.5)
        b.set_ylabel("cost per Jacobian [ms]", fontsize=9.5)
        b.set_title("(b)  fit gradient", fontsize=10.5, loc="left")
        for sp in ("top", "right"):
            b.spines[sp].set_visible(False)

        # (c) derivative-tensor cost vs order m ----------------------------------------------------- #
        ms = [1, 2, 3, 4]; sweeps = [math.comb(n + m - 1, m) for m in ms]
        cost = [s * td2 for s in sweeps]
        c.bar(ms, cost, color=C_DIFF, width=0.6, alpha=0.9, log=True)
        for m, s, v in zip(ms, sweeps, cost):
            c.text(m, v * 1.4, f"{s}", ha="center", fontsize=8.5)
        c.set_xticks(ms); c.set_xlabel("derivative order $m$", fontsize=9.5)
        c.set_ylabel(r"derivative-tensor cost ($\sim\!n^m/m!$ sweeps)", fontsize=9)
        c.set_title("(c)  higher-order autodiff", fontsize=10.5, loc="left")
        for sp in ("top", "right"):
            c.spines[sp].set_visible(False)

        fig.suptitle(f"Sec 5  differentiability advantage — measured on the multisample model "
                     f"({nev/1e6:.1f}M events, {n} dials)", fontsize=11.5, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        style.save(fig, "sec5_fig_scaling")
    print(f"  corner: Taylor {_fmt(t_taylor)} | grid {_fmt(t_grid)} | profile {_fmt(t_profile)}")
    print(f"  gradient: autodiff {_fmt(tj)} vs finite-diff {_fmt(tfd)}")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos[:1] or []))

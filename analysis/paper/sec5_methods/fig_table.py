"""Sec 5.5 -- the differentiability-advantage table (rendered from the MEASURED per-op costs).

Per statistical task, the cost with the differentiable engine vs what a finite-difference generator pays.
Numbers come from sec5_timing_<tag>.npz (t_model / t_jac / t_fdjac / t_d2m on the multisample model); the
corner costs are the same conditional-grid / true-profile / Taylor estimates as the scaling figure.

Usage:  python -m analysis.paper.sec5_methods.fig_table [tag]     (default 4ch)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

C_DIFF, C_FD = "#e8eefc", "#fbe6e6"


def _fmt(t):
    if t < 1:    return f"{t*1e3:.0f} ms"
    if t < 90:   return f"{t:.0f} s"
    if t < 5400: return f"{t/60:.0f} min"
    return f"{t/3600:.0f} hr"


def main(tag="4ch"):
    style.use()
    z = np.load(style.ALTGEN / f"sec5_timing_{tag}.npz", allow_pickle=True)
    tm, tj, tfd, td2 = (float(z[k]) for k in ("t_model", "t_jac", "t_fdjac", "t_d2m"))
    n = int(z["n_dials"]); G, P, ST = 19, n * (n - 1) // 2, 4
    t_taylor = tj + (n * (n + 1) // 2) * td2
    t_grid, t_prof = P * G * G * tm, P * G * G * ST * tj

    rows = [
        ("gradient / Fisher (per fit step)",
         f"autodiff Jacobian ({n} jvp) — {_fmt(tj)}, exact",
         f"{2*n} model evals — {_fmt(tfd)}, noisy + step tuning"),
        ("Hessian curvature",
         "from the same Jacobian (JᵀWJ), exact",
         f"~{n*n} model evals, noisy"),
        ("error propagation (covariance)",
         "exact (V = A⁻¹), stable",
         "noisy; unstable near boundaries"),
        ("non-Gaussian corner (120 pairs)",
         f"Taylor (derivs once) — {_fmt(t_taylor)}",
         f"exact grid {_fmt(t_grid)} / true profile {_fmt(t_prof)}"),
        ("m-th derivative (skew/kurtosis)",
         "autodiff / jet — ~nᵐ/m! sweeps, exact",
         "~nᵐ evals, catastrophic cancellation"),
        ("hybrid (grid-bad × Taylor-good)",
         "exact in the few bad dials, analytic in the good",
         "full grid — exponential in dimension"),
    ]

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"]}):
        fig, ax = plt.subplots(figsize=(13.5, 4.6)); ax.axis("off")
        cols = ["statistical task", "differentiable engine", "finite-difference generator"]
        tab = ax.table(cellText=[[r[0], r[1], r[2]] for r in rows], colLabels=cols,
                       colWidths=[0.28, 0.36, 0.36], cellLoc="left", loc="center")
        tab.auto_set_font_size(False); tab.set_fontsize(9.5); tab.scale(1, 2.1)
        for (r, c), cell in tab.get_celld().items():
            cell.set_edgecolor("0.8"); cell.set_linewidth(0.6)
            if r == 0:
                cell.set_facecolor("0.2"); cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("white" if c == 0 else (C_DIFF if c == 1 else C_FD))
                if c == 0:
                    cell.get_text().set_fontweight("bold")
        fig.suptitle(f"Sec 5.5  differentiability advantage per task  "
                     f"(measured, {int(z['n_events'])/1e6:.1f}M events, {n} dials)", fontsize=12, y=0.93)
        style.save(fig, "sec5_fig_table")
    print(f"  taylor {_fmt(t_taylor)} | grid {_fmt(t_grid)} | profile {_fmt(t_prof)} | "
          f"jac {_fmt(tj)} | fdjac {_fmt(tfd)}")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos[:1] or []))

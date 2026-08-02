"""Sec 4.2 -- example pre/post-fit distributions across the probe types.

Reads the closure npz (data, model_nom = pre-fit/nominal, fit_model = post-fit best-fit, per-obs edges) and
overlays them for four observables spanning the sample set: two neutrino (T2K + MINERvA CC0pi delta pT),
one (e,e') electron (QE omega), one hadron beam (pi+ -> C reaction sigma).  Makes the fit concrete: the
nominal misses, the best-fit lands on the closure data (for the MLE closure it matches it exactly).

Usage:  python -m analysis.paper.sec4_closure.fig_dists [label]     (default sec4_closure_r16_noprior)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

C_FIT, C_NOM = "#1f4b9c", "0.55"
PANELS = [("dpt",        r"T2K CC0$\pi$   $\delta p_T$ [MeV/c]",       r"$d\sigma/d\delta p_T$"),
          ("mnv_dpt",    r"MINERvA CC0$\pi$   $\delta p_T$ [MeV/c]",   r"$d\sigma/d\delta p_T$"),
          ("e_qe",       r"$(e,e')$ C   $\omega_{\rm QE}$ [MeV]",      r"$d\sigma/d\omega$"),
          ("pip_react",  r"$\pi^+$–C   $\sigma_{\rm reac}(p)$",        r"$\sigma$ [mb]")]


def main(label="sec4_closure_r16_noprior"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}.npz", allow_pickle=True)
    dskeys = [str(x) for x in z["dskeys"]]; row0 = np.asarray(z["row0"])
    data = np.asarray(z["data"]); sigma = np.asarray(z["sigma"])
    m_nom = np.asarray(z["model_nom"]); m_fit = np.asarray(z["fit_model"])

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
        for ax, (key, xl, yl) in zip(axes.ravel(), PANELS):
            if key not in dskeys:
                ax.set_visible(False); continue
            j = dskeys.index(key); sl = slice(int(row0[j]), int(row0[j + 1]))
            edges = np.asarray(z[f"{key}_edges"]); x = 0.5 * (edges[1:] + edges[:-1]); xe = 0.5 * np.diff(edges)
            xs = np.repeat(edges, 2)[1:-1]
            ax.plot(xs, np.repeat(m_nom[sl], 2), color=C_NOM, ls=":", lw=1.6, label="nominal (pre-fit)")
            ax.plot(xs, np.repeat(m_fit[sl], 2), color=C_FIT, ls="-", lw=1.7, label="best-fit (post-fit)")
            ax.errorbar(x, data[sl], xerr=xe, yerr=sigma[sl], fmt="o", color="k", ms=3, lw=0.8,
                        capsize=0, zorder=4, label="closure data")
            ax.set_xlabel(xl, fontsize=9.5); ax.set_ylabel(yl, fontsize=9.5); ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0, 0].legend(fontsize=8.5, loc="upper right", framealpha=0.95)
        fig.suptitle("Sec 4.2  pre/post-fit distributions across the probes "
                     "($\\nu$ · $\\nu$ · $(e,e')$ · hadron beam)", fontsize=11.5, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        style.save(fig, "sec4_fig42_dists")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos[:1] or []))

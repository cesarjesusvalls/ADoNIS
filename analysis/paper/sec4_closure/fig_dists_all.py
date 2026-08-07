"""Sec 4.2 (survey) -- pre/post-fit for EVERY fitted observable, not just the four representatives.

Same content as fig_dists, but one panel per dataset block in the closure npz, so you can see the whole
fitted sample set at once and pick which observables are worth showing in the paper.  Reads ONLY the
persisted npz (data / model_nom = pre-fit / fit_model = post-fit / per-obs edges) -- no bank, no refit.

Usage:  python -m analysis.paper.sec4_closure.fig_dists_all [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

C_FIT, C_NOM = "#1f4b9c", "0.55"

# axis labels per observable suffix; anything unlisted falls back to the raw key
XL = {"dpt": r"$\delta p_T$ [MeV/c]", "dat": r"$\delta\alpha_T$ [rad]", "pmu": r"$p_\mu$ [MeV/c]",
      "cosmu": r"$\cos\theta_\mu$", "pn": r"$p_n$ [MeV/c]", "dptt": r"$\delta p_{TT}$ [MeV/c]",
      "daT": r"$\delta\alpha_T$ [rad]", "pt": r"$p_T^\mu$ [MeV/c]", "pz": r"$p_\parallel^\mu$ [MeV/c]",
      "omega": r"$\omega$ [MeV]", "react": r"$p$ [MeV/c]", "abs": r"$p$ [MeV/c]",
      "pipro": r"$p$ [MeV/c]"}
YL_XSEC = r"$d\sigma/dx$"
YL_MB = r"$\sigma$ [mb]"


def _labels(key):
    """(xlabel, ylabel, title) for a namespaced dskey like 't2k_cc0pi:dpt' or a beam key 'pip_react'."""
    if ":" in key:
        samp, obs = key.split(":", 1)
    else:                                   # beam blocks: '<beam>_<obs>'
        samp, _, obs = key.partition("_")
    xl = XL.get(obs, obs)
    yl = YL_MB if obs in ("react", "abs", "pipro") else YL_XSEC
    return xl, yl, f"{samp}  ·  {obs}"


def main(label="sec4_250k"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}.npz", allow_pickle=True)
    dskeys = [str(x) for x in z["dskeys"]]; row0 = np.asarray(z["row0"])
    data = np.asarray(z["data"]); sigma = np.asarray(z["sigma"])
    m_nom = np.asarray(z["model_nom"]); m_fit = np.asarray(z["fit_model"])

    n = len(dskeys)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.35 * ncol, 2.5 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, key in zip(axes, dskeys):
            j = dskeys.index(key); sl = slice(int(row0[j]), int(row0[j + 1]))
            nb = int(row0[j + 1]) - int(row0[j])
            ek = f"{key}_edges"
            edges = np.asarray(z[ek]) if ek in z.files else np.arange(nb + 1, dtype=float)
            x = 0.5 * (edges[1:] + edges[:-1]); xe = 0.5 * np.diff(edges)
            xs = np.repeat(edges, 2)[1:-1]
            ax.plot(xs, np.repeat(m_nom[sl], 2), color=C_NOM, ls=":", lw=1.4)
            ax.plot(xs, np.repeat(m_fit[sl], 2), color=C_FIT, ls="-", lw=1.5)
            sg = np.where(np.isfinite(sigma[sl]), sigma[sl], 0.0)
            ax.errorbar(x, data[sl], xerr=xe, yerr=sg, fmt="o", color="k", ms=2.2, lw=0.6,
                        capsize=0, zorder=4)
            xl, yl, ti = _labels(key)
            ax.set_title(f"{ti}   ({nb} bins)", fontsize=8.5)
            ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
            ax.set_ylim(bottom=0); ax.tick_params(labelsize=7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        for ax in axes[n:]:
            ax.set_visible(False)
        # one shared legend on the first empty slot (or the first axis)
        h = [plt.Line2D([], [], color=C_NOM, ls=":", lw=1.6),
             plt.Line2D([], [], color=C_FIT, ls="-", lw=1.7),
             plt.Line2D([], [], color="k", marker="o", ls="", ms=3)]
        lab = ["nominal (pre-fit)", "best-fit (post-fit)", "closure data"]
        (axes[n] if n < len(axes) else axes[0]).legend(h, lab, fontsize=9, loc="center", frameon=False)
        if n < len(axes):
            axes[n].set_visible(True); axes[n].axis("off")
        fig.suptitle(f"Sec 4.2 (survey)  pre/post-fit for ALL {n} fitted observables  [{label}]",
                     fontsize=12, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        style.save(fig, "sec4_fig42_dists_all")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(pos[0] if pos else "sec4_250k")

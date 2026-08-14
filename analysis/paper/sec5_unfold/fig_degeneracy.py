"""FIGURE H -- what a single sample cannot do: separate the flux from the signal.

Templates and flux both scale signal events.  They are distinguishable only through the 17% background,
which responds to the flux and not to the templates, and through the fact that they partition the events
differently -- the templates by true (delta-p_T, delta-alpha_T), the flux by true E_nu.  That is very
little to work with, and the fake-data studies show exactly how little.

(a) INJECTED vs RECOVERED flux.  A 20% distortion of the peak comes back as ~2%: the fit leaves the flux
    at its prior.
(b) the excess has to go somewhere, and it goes into the templates -- shown in units of the quoted
    uncertainty, which is the only fair way to read a bias.

The conclusion is the physics, not a defect: a flux shape cannot be measured from the signal sample it
multiplies, which is why experiments constrain flux with near detectors and hadron-production data
instead.  The uncertainty is honest -- every bias here is under half a sigma -- but the central value
moves, and a figure that showed only the closure would hide that.

Usage:  python -m analysis.paper.sec5_unfold.fig_degeneracy [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

STUDIES = [("flux120", r"flux peak $\times1.20$", "#1f4b9c", "o"),
           ("fluxtilt", "flux tilt 0.85$\\to$1.15", "#e08214", "s"),
           ("combo", r"tilt $+$ signal $\times1.15$", "#1a9e57", "^")]


def main(label="sec5"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_unfold.npz", allow_pickle=True)
    edges = np.asarray(z["flux_edges"], float).copy()
    if not np.isfinite(edges[-1]):
        edges[-1] = edges[-2] + (edges[-2] - edges[-3])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    nt = len(z["asimov_c"])

    fig = plt.figure(figsize=(9.8, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    # x-offsets: the three studies recover almost the same flux, so without them the markers and their
    # (large) error bars sit exactly on top of one another and two of the three become invisible.
    dx = 0.018 * (edges[-1] - edges[0])
    for i, (st, lab, col, mk) in enumerate(STUDIES):
        ft, ff, fe = z[f"{st}_f_true"], z[f"{st}_f"], z[f"{st}_f_err"]
        ax.step(np.r_[edges[0], edges], np.r_[ft[0], ft, ft[-1]][:len(edges) + 1], where="post",
                color=col, ls="--", lw=1.4, alpha=0.75, zorder=1)
        ax.errorbar(ctr + (i - 1) * dx, ff, yerr=fe, fmt=mk, ms=4.5, color=col, lw=0.9, capsize=0,
                    elinewidth=0.9, alpha=0.95, zorder=3, label=lab)
    ax.axhline(1.0, color="k", lw=0.7, alpha=0.5)
    ax.set_xlabel(r"true $E_\nu$ [MeV]"); ax.set_ylabel(r"flux parameter $f_b$")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 0.13))
    ax.text(0.02, 0.03, "dashed = injected,  markers = recovered", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.5, color="0.35")
    ax.set_title("(a)  the flux is not recovered", fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(nt)
    for st, lab, col, mk in STUDIES:
        pull = (z[f"{st}_c"] - z[f"{st}_c_true"]) / z[f"{st}_c_err"]
        ax.plot(x, pull, mk + "-", color=col, ms=4, lw=1.1, label=lab)
    for lv, sty in ((1.0, ":"), (-1.0, ":")):
        ax.axhline(lv, color="0.4", ls=sty, lw=0.8)
    ax.axhline(0.0, color="k", lw=0.7, alpha=0.5)
    ax.set_xlabel("truth cell"); ax.set_ylabel(r"$(c_j-c_j^{\rm true})/\sigma(c_j)$")
    ax.set_xticks(x); ax.set_ylim(-1.4, 1.4)
    ax.text(0.02, 0.95, r"$\pm1\sigma$", transform=ax.transAxes, va="top", fontsize=7.5, color="0.4")
    ax.set_title("(b)  the excess lands on the templates", fontsize=9.5, loc="left")

    style.save(fig, f"{label}_figH")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))

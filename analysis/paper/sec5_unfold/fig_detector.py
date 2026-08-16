"""FIGURE E -- the detector: what the smearing does, and what survives it.

Three statements, because an unfolding result is only readable if you know what was unfolded:

  (a) the RECO rate in the (delta-p_T, delta-alpha_T) plane -- everything the "experiment" sees, signal
      and background, on the 10 x 6 reco grid.
  (b) the TRUTH rate, signal only, on the 3 x 3 truth grid.  Same plane, same units; the difference
      between the two panels is the detector plus the selection.
  (c) EFFICIENCY per truth cell -- how much of the signal survives reco selection.
  (d) PURITY per reco bin -- how much of what is reconstructed there is signal at all.

Usage:  python -m analysis.paper.sec5_unfold.fig_detector [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

# Series colours are section 4's, so the two halves of the paper that share a fit read alike.
C_EFF, C_PUR = "#1f4b9c", "#e08214"


def _finite(e):
    """Edges for drawing: the open top bin is closed at twice its neighbour's width."""
    e = np.asarray(e, float).copy()
    if not np.isfinite(e[-1]):
        e[-1] = e[-2] + (e[-2] - e[-3])
    return e


def _rate_panel(ax, fig, R, edpt, edat, title, cmap="viridis"):
    """Rate on the physical (delta-p_T, delta-alpha_T) plane, drawn on its own edges."""
    m = ax.pcolormesh(_finite(edpt), _finite(edat), np.asarray(R).T, cmap=cmap, shading="flat")
    cb = fig.colorbar(m, ax=ax, pad=0.02)
    cb.set_label("events", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel(r"$\delta p_T$ [MeV/c]", fontsize=9)
    ax.set_ylabel(r"$\delta\alpha_T$ [rad]", fontsize=9)
    ax.tick_params(labelsize=7.5)
    ax.set_title(title, fontsize=9.5, loc="left")


def main(label="sec5"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_unfold.npz", allow_pickle=True)
    eff, pur = np.asarray(z["eff"]), np.asarray(z["purity"])
    reco_dpt, reco_dat = np.asarray(z["reco_dpt"]), np.asarray(z["reco_dat"])
    true_dpt, true_dat = np.asarray(z["true_dpt"]), np.asarray(z["true_dat"])
    nrd, nra = len(reco_dpt) - 1, len(reco_dat) - 1
    ntd, nta = len(true_dpt) - 1, len(true_dat) - 1
    reco_tot = (np.asarray(z["n_sig_reco"]) + np.asarray(z["n_bkg_reco"])).reshape(nrd, nra)
    true_sig = np.asarray(z["n_true"]).reshape(ntd, nta)
    ntrue, nreco = len(eff), len(pur)

    ctx = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
           "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}
    fig = plt.figure(figsize=(12.6, 3.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.0, 1.0, 1.0], wspace=0.42)

    with plt.rc_context(ctx):
        _rate_panel(fig.add_subplot(gs[0, 0]), fig, reco_tot, reco_dpt, reco_dat,
                    f"(a)  reco rate, all events  ({reco_tot.sum():.0f})")
        _rate_panel(fig.add_subplot(gs[0, 1]), fig, true_sig, true_dpt, true_dat,
                    f"(b)  truth rate, signal  ({true_sig.sum():.0f})", cmap="magma")

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(np.arange(ntrue), eff, color=C_EFF, width=0.72)
    ax.axhline(np.nanmean(eff), color="k", ls="--", lw=0.9, label=f"mean {np.nanmean(eff):.2f}")
    ax.set_xlabel("truth cell"); ax.set_ylabel("efficiency"); ax.set_ylim(0.5, 0.85)
    ax.set_xticks(range(ntrue)); ax.legend(frameon=False, fontsize=8)
    ax.set_title("(c)  efficiency", fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[0, 3])
    ax.step(np.arange(nreco), pur, where="mid", color=C_PUR, lw=1.3)
    ax.axhline(np.nanmean(pur), color="k", ls="--", lw=0.9, label=f"mean {np.nanmean(pur):.2f}")
    ax.set_xlabel("reco bin"); ax.set_ylabel("purity"); ax.set_ylim(0.70, 0.95)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(d)  purity", fontsize=9.5, loc="left")

    style.save(fig, f"{label}_figE")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))

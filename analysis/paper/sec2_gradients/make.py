"""Paper section 2 -- "we have exact gradient information for all 27 knobs".

The object is the SAME Jacobian section 3 gates on (physfit_gate1_full_v2.npz): J_ik = d(dsigma/dx)_i/dtheta_k
for every knob k and every bin i, obtained by ONE jax.jvp per knob through bank_reweight.weight_jit --
autodiff through the frozen walk, not finite differences, not a surrogate.  Sections 2 and 3 therefore
cost a single bank pass between them.

Plotted in the natural dimensionless form
    S_ik = (dtheta_k^prior) * J_ik / sigma_i        [ = the knob's per-bin pull, in units of the error ]
which is exactly what the Fisher F = S^T S accumulates -- so the figure IS the Fisher's integrand, and a
row that is everywhere pale is precisely a knob the sample cannot constrain.

Usage:  python -m analysis.paper.sec2_gradients.make [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "altgen"))
from analysis.paper import physical_fit as PF
from analysis.paper import style

DSLABEL = {"dpt": "CC0$\\pi$\n$\\delta p_T$", "dat": "CC0$\\pi$\n$\\delta\\alpha_T$",
           "pmu": "CC0$\\pi$\n$p_\\mu$", "cosmu": "CC0$\\pi$\n$\\cos\\theta_\\mu$",
           "pn": "CC1$\\pi$\n$p_N$", "dptt": "CC1$\\pi$\n$\\delta p_{TT}$",
           "daT": "CC1$\\pi$\n$\\delta\\alpha_T$", "ppi": "CC1$\\pi$\n$p_\\pi$",
           "cospi": "CC1$\\pi$\n$\\cos\\theta_\\pi$", "n_p": "incl\n$N_p$", "n_chpi": "incl\n$N_{\\pi^\\pm}$"}


def main(label="physfit_gate1_full_v2"):
    style.use()
    d = np.load(style.ALTGEN / f"{label}.npz", allow_pickle=True)
    J, sigma, prior = d["J"], d["sigma"], d["prior"]
    pnames = [str(x) for x in d["pnames"]]
    dskeys = [str(x) for x in d["dskeys"]]
    row0 = np.asarray(d["row0"])
    shrink = d["shrink"]

    S = (prior[None, :] * J) / sigma[:, None]        # (nbins, nknob) per-bin pull in units of sigma
    S = S.T                                          # -> (nknob, nbins)

    # kF_sf pulls ~10 sigma/bin while the FF knobs pull ~0.01 -- a linear scale saturates on kF_sf and
    # whites out everything else, which would UNDERSTATE the claim.  Signed-log keeps the absolute scale
    # (rows stay comparable, unlike a per-panel renormalization) and still resolves the small gradients.
    # linthresh = 0.1 sigma: below that a per-bin pull is negligible and STAYS pale (an honest "no
    # information here"); above it, two decades of colour separate the knobs that actually move the data.
    vmax = float(np.max(np.abs(S)))
    norm = mcolors.SymLogNorm(linthresh=0.1, vmin=-vmax, vmax=vmax, base=10)
    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    im = ax.imshow(S, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")

    for r in row0[1:-1]:                             # dataset boundaries
        ax.axvline(r - 0.5, color="k", lw=0.8)
    ctr = [(row0[j] + row0[j + 1]) / 2 - 0.5 for j in range(len(dskeys))]
    ax.set_xticks(ctr)
    ax.set_xticklabels([DSLABEL.get(k, k) for k in dskeys], fontsize=7)
    ax.set_yticks(range(len(pnames)))
    ax.set_yticklabels([f"{p}   " for p in pnames], fontsize=7)
    # mark the knobs the data actually determines (Gate I), so section 2 hands section 3 the baton
    for k, p in enumerate(pnames):
        if shrink[k] < 0.5:
            ax.get_yticklabels()[k].set_color("#d62728")
            ax.get_yticklabels()[k].set_weight("bold")
    ax.set_xlabel("bin  (grouped by observable)")
    ax.set_title("Exact per-bin gradients, all 27 knobs:  "
                 "$\\sigma^{prior}_k\\,\\partial(d\\sigma/dx)_i/\\partial\\theta_k\\;/\\;\\sigma_i$   "
                 "(autodiff through the cascade; red label = passes Gate I)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01,
                 label="per-bin pull  [$\\sigma$]  for a 1-$\\sigma$ prior move")
    style.save(fig, "sec2_gradients_all27")

    # ---- per-knob SHAPE: each ROW normalized to its own peak |pull|, so the gradient's DISTRIBUTION
    # across the bins is visible for every knob regardless of its overall magnitude (kF_sf and the tiny
    # FF knobs become equally readable).  Answers "given this knob, WHERE and with what sign does it pull
    # across the observable?" -- pure shape, magnitude removed.  Linear [-1,1] (rows are already comparable).
    rowmax = np.max(np.abs(S), axis=1, keepdims=True)
    Sshape = np.where(rowmax > 0, S / rowmax, 0.0)
    reach = np.sqrt((S ** 2).sum(axis=1))                  # per-knob overall magnitude = sqrt(F_kk) [sigma]
    fig = plt.figure(figsize=(15.0, 7.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[6, 1.05, 0.16], wspace=0.04)
    axh = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1], sharey=axh); cax = fig.add_subplot(gs[2])
    im = axh.imshow(Sshape, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    for r in row0[1:-1]:
        axh.axvline(r - 0.5, color="k", lw=0.8)
    axh.set_xticks(ctr); axh.set_xticklabels([DSLABEL.get(k, k) for k in dskeys], fontsize=7)
    axh.set_yticks(range(len(pnames))); axh.set_yticklabels([f"{p}   " for p in pnames], fontsize=7)
    for k, p in enumerate(pnames):
        if shrink[k] < 0.5:
            axh.get_yticklabels()[k].set_color("#d62728"); axh.get_yticklabels()[k].set_weight("bold")
    axh.set_xlabel("bin  (grouped by observable)")
    axh.set_title("Per-knob gradient SHAPE:  each row normalized to its own peak pull  "
                  "(where each knob pulls across the bins; magnitude removed)", fontsize=9)
    # right panel: the overall magnitude that the row-normalization removed = sqrt(F_kk)
    axb.barh(range(len(pnames)), reach,
             color=["#d62728" if shrink[k] < 0.5 else "#9e9e9e" for k in range(len(pnames))])
    axb.set_xscale("log"); axb.axvline(2.0, color="k", ls=":", lw=0.8)
    axb.grid(axis="x", ls=":", alpha=0.4); axb.tick_params(labelsize=7)
    plt.setp(axb.get_yticklabels(), visible=False)
    axb.set_xlabel(r"magnitude  $\sqrt{F_{kk}}$  [$\sigma$]", fontsize=8)
    axb.set_title("overall magnitude\n(2$\\sigma$ = dotted)", fontsize=8)
    fig.colorbar(im, cax=cax, label="relative pull  (row-normalized, signed)")
    style.save(fig, "sec2_gradients_shape")

    # per-knob reach: the total pull a 1-sigma prior move produces, = sqrt(F_kk) (degeneracy-blind)
    reach = np.sqrt((S**2).sum(axis=1))
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    o = np.argsort(reach)
    col = ["#d62728" if shrink[k] < 0.5 else "#9e9e9e" for k in o]
    ax.barh(range(len(o)), reach[o], color=col)
    ax.axvline(2.0, color="k", ls=":", lw=.8)
    ax.set_yticks(range(len(o))); ax.set_yticklabels([pnames[k] for k in o], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("gradient reach  $\\sqrt{F_{kk}}$  [$\\sigma$ of pull per 1-$\\sigma$ prior move]")
    ax.set_title("Every knob has a gradient; not every gradient is information\n"
                 "(red = passes Gate I on the full set)", fontsize=9)
    style.save(fig, "sec2_gradient_reach")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))

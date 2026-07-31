"""The exact per-bin gradients for the knobs the data can constrain.

The object is the SAME Jacobian the Fisher figure gates on (multisample_carbon.npz): J_ik =
d(dsigma/dx)_i/dtheta_k for every knob k and every bin i, obtained by ONE jax.jvp per knob through
bank_reweight.weight_jit -- autodiff through the frozen walk, not finite differences, not a surrogate.

Plotted as the per-knob gradient SHAPE for the Gate-I FITTABLE subset (marginalized shrinkage < 0.5 on
the combined fit): each knob's row of the dimensionless pull
    S_ik = (dtheta_k^prior) * J_ik / sigma_i        [ the knob's per-bin pull, in units of the error ]
is normalized to its own peak, so WHERE each knob pulls across the bins is readable; the pull is SIGNED
(a knob raises or lowers a bin), hence the diverging map.  Rows grouped by physics, styled to match the
Fisher figure (analysis/paper/sec3_fisher) -- the two read as a pair: which knobs are constrainable, and
where their information comes from.

Usage:  python -m analysis.paper.sec2_gradients.make [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "altgen"))
from analysis.paper import style
from analysis.paper import fisher_engine as FE

# every observable label carries its EXPERIMENT (+ topology): line 1 = experiment/sample, line 2 = the
# observable variable (the thin inclusive-multiplicity rows use a compact single-line superscript form).
DSLABEL = {"dpt": "T2K CC0$\\pi$\n$\\delta p_T$", "dat": "T2K CC0$\\pi$\n$\\delta\\alpha_T$",
           "pmu": "T2K CC0$\\pi$\n$p_\\mu$", "cosmu": "T2K CC0$\\pi$\n$\\cos\\theta_\\mu$",
           "pn": "T2K CC1$\\pi$\n$p_N$", "dptt": "T2K CC1$\\pi$\n$\\delta p_{TT}$",
           "daT": "T2K CC1$\\pi$\n$\\delta\\alpha_T$", "ppi": "T2K CC1$\\pi$\n$p_\\pi$",
           "cospi": "T2K CC1$\\pi$\n$\\cos\\theta_\\pi$",
           "n_p": "T2K $N_p^{\\rm incl}$", "n_chpi": "T2K $N_{\\pi^\\pm}^{\\rm incl}$",
           "mnv_dat": "MINERvA\nCC0$\\pi$ $\\delta\\alpha_T$", "mnv_pn": "MINERvA\nCC0$\\pi$ $p_n$",
           "mnv_dpt": "MINERvA\nCC0$\\pi$ $\\delta p_T$",
           "pip_react": "$\\pi^+$C\n$\\sigma_{\\rm reac}$", "pip_abs": "$\\pi^+$C\n$\\sigma_{\\rm abs}$",
           "prot_react": "$p$C\n$\\sigma_{\\rm reac}$", "prot_pipro": "$p$C\n$\\sigma_{\\pi\\rm prod}$",
           "neut_react": "$n$C\n$\\sigma_{\\rm reac}$", "neut_pipro": "$n$C\n$\\sigma_{\\pi\\rm prod}$",
           "e_qe": "$(e,e')$C\n$\\omega_{\\rm QE}$", "e_res": "$(e,e')$C\n$\\omega_{\\rm RES}$",
           "t2k_pcos": "T2K CC0$\\pi$\n2D $p_\\mu$-$\\cos\\theta_\\mu$",
           "mnv_ptmu": "MINERvA\n$p_T^\\mu$", "mnv_pzmu": "MINERvA\n$p_\\parallel^\\mu$",
           "mnv_ptpl": "MINERvA qe\n2D $p_T$-$p_\\parallel$"}



FIT_CUT = 0.5     # a knob is shown when the COMBINED fit's marginalized shrinkage < this (Gate I)


def main(label="multisample_carbon"):
    """Per-bin gradient SHAPE for the Gate-I fittable knobs -- the paper's gradient figure.

    Rows = the knobs the combined data can actually constrain (marginalized shrinkage < FIT_CUT),
    ordered by physics block; columns = bins grouped by observable.  Each row is normalized to its own
    peak |pull|, so WHERE each knob pulls is visible; the pull is SIGNED (a knob raises or lowers a bin),
    hence the diverging blue<->orange map (anchored on the §1 blue).  Styled to match the §3 Fisher figure.
    """
    style.use()
    d = np.load(style.ALTGEN / f"{label}.npz", allow_pickle=True)
    J, sigma, prior = d["J"], d["sigma"], d["prior"]
    pnames = [str(x) for x in d["pnames"]]
    dskeys = [str(x) for x in d["dskeys"]]
    row0 = np.asarray(d["row0"])

    marg = FE.gate1(J, sigma, prior, np.arange(J.shape[0]))[3]     # marginalized shrinkage, combined fit
    S = (prior[None, :] * J) / sigma[:, None]                      # (nbin, nknob) per-bin pull [sigma units]
    Sk = S.T                                                       # (nknob, nbin)
    rowmax = np.max(np.abs(Sk), axis=1, keepdims=True)
    shape = np.where(rowmax > 0, Sk / rowmax, 0.0)                 # per-knob normalized, in [-1, 1]

    idx = [k for k in range(len(pnames)) if marg[k] < FIT_CUT]     # the fittable subset (Gate I)
    idx = sorted(idx, key=lambda k: (style.knob_group(pnames[k]), k))
    Z = shape[idx]
    plabels = [style.plab(pnames[k]) for k in idx]
    gid = [style.knob_group(pnames[k]) for k in idx]
    nk, nb = Z.shape
    ctr = [(row0[j] + row0[j + 1]) / 2 - 0.5 for j in range(len(dskeys))]   # observable-block centres

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, ax = plt.subplots(figsize=(max(13.0, 0.03 * nb + 3.4), 0.36 * nk + 2.2))
        im = ax.imshow(Z, aspect="auto", cmap=style.CMAP_GRAD_DIV, vmin=-1, vmax=1, interpolation="nearest")
        for r in row0[1:-1]:                                       # observable column separators
            ax.axvline(r - 0.5, color="0.35", lw=0.7)
        ax.set_xticks(ctr); ax.set_xticklabels([DSLABEL.get(k, k) for k in dskeys], fontsize=6.5)
        ax.xaxis.set_ticks_position("top"); ax.tick_params(axis="x", length=0)
        style.knob_group_tabs(ax, gid, tabx=-0.055, tabw=0.008, labx=-0.072)   # narrow tabs (wide figure)
        ax.set_yticks(range(nk)); ax.set_yticklabels(plabels, fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.020, pad=0.015, ticks=[-1, 0, 1])
        cb.set_label("per-bin gradient (normalized per knob)", fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5)
        style.save(fig, "sec2_gradients_shape")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]      # positional label
    main(*(pos[:1] or []))

"""Per-bin gradient shapes for the knobs the data can constrain.

Reads the persisted Jacobian J_ik = d(dsigma/dx)_i/dtheta_k (one jax.jvp per knob through
bank_reweight.weight_jit).  Plots the signed, per-row-normalized pull S_ik = (dtheta_k^prior)*J_ik/sigma_i
for the Gate-I fittable knobs (marginalized shrinkage < 0.5), grouped by physics block, styled to match
the Fisher figure (grad_info/fisher.py).

Usage:  python -m analysis.paper.grad_info.make [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.paper import style
from adonis.stats import fisher as FE

DSLABEL = {"t2k_cc0pi:dpt": "CC0$\\pi$ $\\delta p_T$", "t2k_cc0pi:dalphat": "CC0$\\pi$ $\\delta\\alpha_T$",
           "t2k_cc0pi:pmu": "CC0$\\pi$ $p_\\mu$", "t2k_cc0pi:cos_mu": "CC0$\\pi$ $\\cos\\theta_\\mu$",
           "t2k_cc1pi_ch:pn": "CC1$\\pi$ $p_N$", "t2k_cc1pi_ch:dptt": "CC1$\\pi$ $\\delta p_{TT}$",
           "t2k_cc1pi_ch:dalphat": "CC1$\\pi$ $\\delta\\alpha_T$",
           "minerva_stv:dalphat": "CC0$\\pi$ $\\delta\\alpha_T$", "minerva_stv:pn": "CC0$\\pi$ $p_n$",
           "minerva_stv:dpt": "CC0$\\pi$ $\\delta p_T$",
           "minerva_ptpz:pt": "incl. $p_T^\\mu$", "minerva_ptpz:pz": "incl. $p_\\parallel^\\mu$",
           "minerva_cc1pip_tpi:tpi": "CC1$\\pi^+$ $T_\\pi$",
           "minerva_cc1pip_q2:q2": "CC1$\\pi^+$ $Q^2$",
           "ee_omega:omega": "$(e,e')$ $\\omega$",
           "pip_react": "$\\pi^+$ $\\sigma_{\\rm reac}$", "pip_abs": "$\\pi^+$ $\\sigma_{\\rm abs}$",
           "prot_react": "$p$ $\\sigma_{\\rm reac}$", "prot_pipro": "$p$ $\\sigma_{\\pi\\rm prod}$",
           "neut_react": "$n$ $\\sigma_{\\rm reac}$", "neut_pipro": "$n$ $\\sigma_{\\pi\\rm prod}$"}


SAMPLE_GROUP = [("T2K", ("t2k_",), "#4a4a4a"),
                ("MINERvA", ("minerva_",), "#8f8f8f"),
                ("$e$ beam", ("ee_",), "#4a4a4a"),
                ("hadron beam", ("pip_", "prot_", "neut_"), "#8f8f8f")]


def _group_of(key):
    for gi, (_n, pref, _c) in enumerate(SAMPLE_GROUP):
        if key.startswith(pref):
            return gi
    return len(SAMPLE_GROUP) - 1


FIT_CUT = 0.5
MC_CUT = 0.05
SYST = 0.05


def main(label="multisample_carbon"):
    """Per-bin gradient SHAPE for the Gate-I fittable knobs.

    Rows = knobs the combined data can constrain (marginalized shrinkage < FIT_CUT), grouped by physics
    block; columns = bins grouped by observable.  Each row is normalized to its own peak; sign shows
    whether a knob raises or lowers a bin.
    """
    style.use()
    d = np.load(style.ALTGEN / f"{label}.npz", allow_pickle=True)
    J, sigma, prior = d["J"], d["sigma"], d["prior"]
    pnames = [str(x) for x in d["pnames"]]
    dskeys = [str(x) for x in d["dskeys"]]
    row0 = np.asarray(d["row0"])

    marg = FE.fisher_shrinkage(J, sigma, prior, np.arange(J.shape[0]))[3]
    S = (prior[None, :] * J) / sigma[:, None]
    Sk = S.T
    rowmax = np.max(np.abs(Sk), axis=1, keepdims=True)
    shape = np.where(rowmax > 0, Sk / rowmax, 0.0)

    keep = None
    _mp = style.ALTGEN / "sec4_live_mask.npz"
    if _mp.exists():
        _m = np.load(_mp, allow_pickle=True)
        parts = []
        for j, k in enumerate(dskeys):
            nb_j = row0[j + 1] - row0[j]
            parts.append(np.asarray(_m[f"{k}_keep"], bool) if f"{k}_keep" in _m.files
                         else np.ones(nb_j, bool))
        keep = np.concatenate(parts)
        print(f"[sec3] sec4 live-bin mask: {int(keep.sum())}/{len(keep)} bins kept "
              f"(from {_mp.name}, the fit's own statistics)")

    sig_all = np.asarray(sigma, float)
    nomc = []
    if keep is None:
      keep = np.ones(J.shape[0], bool)
      for j, k in enumerate(dskeys):
        sl = slice(row0[j], row0[j + 1])
        if f"{k}_central" not in d.files:
            nomc.append(k)
            continue
        c0 = np.abs(np.asarray(d[f"{k}_central"], float))
        var = sig_all[sl] ** 2 - (SYST * c0) ** 2
        mc = np.sqrt(np.clip(var, 0.0, None))
        keep[sl] = np.isfinite(sig_all[sl]) & (c0 > 0) & (mc <= MC_CUT * np.where(c0 > 0, c0, 1.0))
      print(f"[sec3] FALLBACK mask on this Jacobian's own statistics: {int(keep.sum())}/{len(keep)} "
            f"bins kept -- NOT the fit's mask; run the engine mask dump for that"
            + (f"; no central for {nomc}" if nomc else ""))

    idx = [k for k in range(len(pnames)) if marg[k] < FIT_CUT]
    idx = sorted(idx, key=lambda k: (style.knob_group(pnames[k]), k))
    border = sorted(range(len(dskeys)), key=lambda j: (_group_of(dskeys[j]), j))
    dskeys = [dskeys[j] for j in border]
    cols = np.concatenate([np.arange(row0[j], row0[j + 1])[keep[row0[j]:row0[j + 1]]] for j in border]) \
        if len(border) else np.zeros(0, int)
    nkeep = [int(keep[row0[j]:row0[j + 1]].sum()) for j in border]
    row0k = np.cumsum([0] + nkeep)
    Z = shape[idx][:, cols]
    plabels = [style.plab(pnames[k]) for k in idx]
    gid = [style.knob_group(pnames[k]) for k in idx]
    nk, nb = Z.shape
    ctr = [(row0k[j] + row0k[j + 1]) / 2 - 0.5 for j in range(len(dskeys))]

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, ax = plt.subplots(figsize=(max(10.0, 0.023 * nb + 2.8), 0.29 * nk + 1.7))
        im = ax.imshow(Z, aspect="auto", cmap=style.CMAP_GRAD_DIV, vmin=-1, vmax=1, interpolation="nearest")
        for r in row0k[1:-1]:
            ax.axvline(r - 0.5, color="0.35", lw=0.7)
        ax.set_xticks(ctr)
        ax.set_xticklabels([DSLABEL.get(k, k) for k in dskeys], fontsize=7.5, rotation=90,
                           ha="center", va="bottom")
        ax.xaxis.set_ticks_position("top"); ax.tick_params(axis="x", length=0, pad=2)

        gids = [_group_of(k) for k in dskeys]
        runs, st = [], 0
        for j in range(1, len(gids) + 1):
            if j == len(gids) or gids[j] != gids[st]:
                runs.append((gids[st], row0k[st], row0k[j])); st = j
        fig.canvas.draw()
        inv = ax.transAxes.inverted()
        tops = [inv.transform(t.get_window_extent().corners())[:, 1].max()
                for t in ax.get_xticklabels() if t.get_text()]
        BAR_Y = (max(tops) if tops else 1.0) + 0.012
        LBL_Y = BAR_Y + 0.010
        tr = ax.get_xaxis_transform()
        for _gi, a, _b in runs[1:]:
            ax.axvline(a - 0.5, color="0.22", lw=1.6)
        for gi, a, b in runs:
            nm, _pref, col = SAMPLE_GROUP[gi]
            ax.plot([a - 0.4, b - 0.6], [BAR_Y, BAR_Y], transform=tr, color=col, lw=2.6,
                    solid_capstyle="butt", clip_on=False, zorder=5)
            ax.text((a + b) / 2 - 0.5, LBL_Y, nm, transform=tr, ha="center", va="bottom",
                    fontsize=9.5, color=col, clip_on=False)
        ax.set_yticks(range(nk)); ax.set_yticklabels(plabels, fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        fig.canvas.draw()
        inv = ax.transAxes.inverted()
        lefts = [inv.transform(t.get_window_extent().corners())[:, 0].min()
                 for t in ax.get_yticklabels() if t.get_text()]
        _tabx = (min(lefts) if lefts else 0.0) - 0.024
        style.knob_group_tabs(ax, gid, tabx=_tabx, tabw=0.007, labx=_tabx - 0.018)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.011, pad=0.012, ticks=[-1, 0, 1])
        cb.ax.tick_params(labelsize=7.5)
        style.save(fig, "multisample_grad_per_bin")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos[:1] or []))

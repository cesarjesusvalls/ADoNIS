"""The exact per-bin gradients for the knobs the data can constrain.

The object is the SAME Jacobian the Fisher figure gates on (multisample_carbon.npz): J_ik =
d(dsigma/dx)_i/dtheta_k for every knob k and every bin i, obtained by ONE jax.jvp per knob through
bank_reweight.weight_jit -- autodiff through the frozen walk, not finite differences, not a surrogate.

Plotted as the per-knob gradient SHAPE for the Gate-I FITTABLE subset (marginalized shrinkage < 0.5 on
the combined fit): each knob's row of the dimensionless pull
    S_ik = (dtheta_k^prior) * J_ik / sigma_i        [ the knob's per-bin pull, in units of the error ]
is normalized to its own peak, so WHERE each knob pulls across the bins is readable; the pull is SIGNED
(a knob raises or lowers a bin), hence the diverging map.  Rows grouped by physics, styled to match the
Fisher figure (analysis/paper/grad_info) -- the two read as a pair: which knobs are constrainable, and
where their information comes from.

Usage:  python -m analysis.paper.grad_info.make [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "altgen"))
from analysis.paper import style
from adonis.stats import fisher as FE

# every observable label carries its EXPERIMENT (+ topology): line 1 = experiment/sample, line 2 = the
# observable variable (the thin inclusive-multiplicity rows use a compact single-line superscript form).
# dskeys are namespaced `sample:obs` (AnaSample/SampleSet); the beam keys stay bare.
# The EXPERIMENT is carried by the colour bar across the top, so each label only has to name the
# TOPOLOGY and the OBSERVABLE.  Repeating "T2K"/"MINERvA" on every one of 21 two-line labels at 6.5 pt
# was what made the axis unreadable.  Labels are ONE LINE and drawn VERTICALLY: the observable blocks
# differ in width by more than 10x once the sparse-bin mask is applied, and a horizontal label centred
# on a 3-bin block runs straight into its neighbours (the T2K CC1pi block did exactly that).
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


# EXPERIMENT groups, drawn as a colour bar across the top.  Twenty-one two-line observable labels at
# 6.5 pt are unreadable as a block; the bar carries the experiment so each label only names the
# observable.  Same device as the physics-block tabs down the left, and the same muted palette.
# NEUTRAL greys on purpose.  The clay/sage/violet/rose palette belongs to the KNOB physics blocks down
# the left; reusing it here would imply that "T2K" corresponds to "cross-section" and so on, which is
# false.  Two alternating greys are enough to separate adjacent spans, and they read as structure rather
# than as a second colour code.
SAMPLE_GROUP = [("T2K", ("t2k_",), "#4a4a4a"),
                ("MINERvA", ("minerva_",), "#8f8f8f"),
                ("$e$ beam", ("ee_",), "#4a4a4a"),
                ("hadron beam", ("pip_", "prot_", "neut_"), "#8f8f8f")]


def _group_of(key):
    for gi, (_n, pref, _c) in enumerate(SAMPLE_GROUP):
        if key.startswith(pref):
            return gi
    return len(SAMPLE_GROUP) - 1


FIT_CUT = 0.5     # a knob is shown when the COMBINED fit's marginalized shrinkage < this (Gate I)
MC_CUT = 0.05     # the sec4 sparse-bin mask: drop bins whose MC error exceeds this fraction of central
SYST = 0.05       # the fractional systematic sigma was built with (configs/samples/*.yaml)


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

    # SPARSE-BIN MASK, exactly the sec4 fit's: a bin is dropped when its MC error exceeds MC_CUT of the
    # central value (or the central is non-positive).  Drawing bins the fit discards overstates where the
    # information is -- the discarded ones are precisely the sparse tails, which look striking here and
    # contribute nothing there.
    # THE MASK COMES FROM THE FIT, not from this Jacobian.  The two do not see the same statistics: the
    # sec4 engine caps selected events per sample (banks.sig_cap = 250k) while the Jacobian streams the
    # whole bank, so recomputing the 5% cut here drops 4 bins where the fit drops 43.  A figure captioned
    # "the bins the fit uses" has to use the fit's own mask, so the engine dumps it
    # (output/altgen/sec4_live_mask.npz) and this reads it.  The recovery below is the fallback.
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

    # mcerr is RECOVERED from what the npz already stores rather than requiring a rebuild:
    #     sigma = sqrt((syst*central)^2 + mcerr^2)   =>   mcerr = sqrt(sigma^2 - (syst*central)^2)
    # (adonis.analysis.sample._bin_sigma / fisher_engine.bin_sigma).  Exact, not an approximation.  An
    # observable with no stored central -- the beam blocks -- keeps all its bins: it has no MC error to
    # test, and inventing one would silently drop bins for the wrong reason.
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
        mc = np.sqrt(np.clip(var, 0.0, None))                  # clip: rounding can make this -1e-30
        keep[sl] = np.isfinite(sig_all[sl]) & (c0 > 0) & (mc <= MC_CUT * np.where(c0 > 0, c0, 1.0))
      print(f"[sec3] FALLBACK mask on this Jacobian's own statistics: {int(keep.sum())}/{len(keep)} "
            f"bins kept -- NOT the fit's mask; run the engine mask dump for that"
            + (f"; no central for {nomc}" if nomc else ""))

    idx = [k for k in range(len(pnames)) if marg[k] < FIT_CUT]     # the fittable subset (Gate I)
    idx = sorted(idx, key=lambda k: (style.knob_group(pnames[k]), k))
    # Renumber the columns over the KEPT bins only, so each observable block shrinks to the bins the fit
    # actually uses and the separators still land between blocks.
    # BLOCK ORDER: group the observables by experiment so each colour span across the top is contiguous.
    # (e,e') sits between two MINERvA samples in the config's sample order, which would otherwise print
    # "MINERvA" twice with a gap.  Display only -- within a group the config order is preserved, and the
    # underlying Jacobian is untouched.
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
    ctr = [(row0k[j] + row0k[j + 1]) / 2 - 0.5 for j in range(len(dskeys))]  # observable-block centres

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        # Smaller canvas at the same font sizes: the type reads larger relative to the panel.
        fig, ax = plt.subplots(figsize=(max(10.0, 0.023 * nb + 2.8), 0.29 * nk + 1.7))
        im = ax.imshow(Z, aspect="auto", cmap=style.CMAP_GRAD_DIV, vmin=-1, vmax=1, interpolation="nearest")
        for r in row0k[1:-1]:                                      # observable column separators
            ax.axvline(r - 0.5, color="0.35", lw=0.7)
        ax.set_xticks(ctr)
        ax.set_xticklabels([DSLABEL.get(k, k) for k in dskeys], fontsize=7.5, rotation=90,
                           ha="center", va="bottom")
        ax.xaxis.set_ticks_position("top"); ax.tick_params(axis="x", length=0, pad=2)

        # EXPERIMENT BAR across the top: one coloured span per experiment, so the per-observable labels
        # below it only have to name the observable.  Drawn in axes coords above the matrix.
        gids = [_group_of(k) for k in dskeys]
        runs, st = [], 0
        for j in range(1, len(gids) + 1):
            if j == len(gids) or gids[j] != gids[st]:
                runs.append((gids[st], row0k[st], row0k[j])); st = j
        # Place the bar just above the tick labels by MEASURING them.  The labels are rotated, so the
        # fraction of the axes they occupy grows as the canvas shrinks; a hardcoded offset is correct at
        # exactly one figure size and leaves a gap or an overlap at every other.
        fig.canvas.draw()
        inv = ax.transAxes.inverted()
        tops = [inv.transform(t.get_window_extent().corners())[:, 1].max()
                for t in ax.get_xticklabels() if t.get_text()]
        BAR_Y = (max(tops) if tops else 1.0) + 0.012
        LBL_Y = BAR_Y + 0.010
        tr = ax.get_xaxis_transform()
        # EXPERIMENT separators, heavier than the per-observable ones -- same weight and colour as the
        # physics-block separators across the rows, so the two groupings read as the same kind of
        # structure in each direction.
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
        # Tabs go OUTSIDE the widest knob label, measured.  Their offsets are axes fractions, so the
        # narrower canvas shrank them in absolute terms and the long NN-FSI symbols (s^el_NN,pp) ran into
        # the block tabs.  Measuring keeps the clearance right at any canvas size.
        fig.canvas.draw()
        inv = ax.transAxes.inverted()
        lefts = [inv.transform(t.get_window_extent().corners())[:, 0].min()
                 for t in ax.get_yticklabels() if t.get_text()]
        _tabx = (min(lefts) if lefts else 0.0) - 0.024
        style.knob_group_tabs(ax, gid, tabx=_tabx, tabw=0.007, labx=_tabx - 0.018)
        for s in ax.spines.values():
            s.set_visible(False)
        # No colourbar label: the quantity is stated in the caption, and at this width the text was
        # taller than the bar it annotated.
        cb = fig.colorbar(im, ax=ax, fraction=0.011, pad=0.012, ticks=[-1, 0, 1])
        cb.ax.tick_params(labelsize=7.5)
        style.save(fig, "multisample_grad_per_bin")   # renamed 2026-08-14 (was multisample_carbon_shape)


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]      # positional label
    main(*(pos[:1] or []))

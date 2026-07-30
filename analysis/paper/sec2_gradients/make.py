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

# compact, physics-intuitive LaTeX for each knob (physical_fit SPEC names -> symbol).  QE/RES form-factor
# strengths S_{A,V}; Sachs FFs; C5A (res axial); pion-pole; Delta P33 strength; FSI scale factors s_*
# (pi absorption / piN elastic+cex / conversion / NN elastic+inelastic per pp,pn,nn; NN charge-exchange
# fraction); spectral function k_F / removal-energy shift / SF & SRC norms; QE & RES channel norms.
PLATEX = {
    "M_A_qe": r"$M_A^{\rm QE}$", "M_A_res": r"$M_A^{\rm RES}$",
    "axial_strength": r"$S_A^{\rm QE}$", "vector_strength": r"$S_V^{\rm QE}$",
    "mu_p": r"$\mu_p$", "mu_n": r"$\mu_n$", "gep": r"$G_E^p$", "gen": r"$G_E^n$",
    "res_axial_strength": r"$C_5^A$", "pion_pole": r"$F_{\rm pp}$", "delta_strength": r"$S_\Delta$",
    "sabs": r"$s_{\rm abs}^{\pi}$", "s_piN_elastic": r"$s^{\rm el}_{\pi N}$",
    "s_piN_cex": r"$s^{\rm cex}_{\pi N}$", "s_conv": r"$s_{\rm conv}$",
    "s_NN_elastic[0]": r"$s^{\rm el}_{NN,pp}$", "s_NN_elastic[1]": r"$s^{\rm el}_{NN,pn}$",
    "s_NN_elastic[2]": r"$s^{\rm el}_{NN,nn}$",
    "s_NN_inelastic[0]": r"$s^{\rm inel}_{NN,pp}$", "s_NN_inelastic[1]": r"$s^{\rm inel}_{NN,pn}$",
    "s_NN_inelastic[2]": r"$s^{\rm inel}_{NN,nn}$", "f_NN_cex": r"$f^{\rm cex}_{NN}$",
    "kF_sf": r"$k_F$", "Eb_shift": r"$\Delta E_b$", "sf_norm": r"$N_{\rm SF}$",
    "src_tail": r"$N_{\rm SRC}$", "qe_norm": r"$N_{\rm QE}$", "res_norm": r"$N_{\rm RES}$",
}


def _plab(p):
    return PLATEX.get(p, p)

# Published-data x-RANGE per observable, in that observable's native bin units, for the PER-BIN data
# highlight (`--data`): a bin is marked when its CENTRE falls in [lo, hi], so only the individual bins the
# measurement actually covers light up -- not the whole observable block.  Sources: T2K CC0pi & CC1pi STV,
# MINERvA CC0piNp STV, pi+/proton-carbon scattering.  Observables NOT listed have no measurement in this
# exact signal definition (lepton/pion single kinematics, multiplicities) and get no marks.  FIRST-PASS
# ranges -- easily refined per release.
DATA_RANGE = {
    "dpt": (0.0, 1000.0), "dat": (0.0, 180.0),                              # T2K CC0pi STV
    "pn": (0.0, 1000.0), "dptt": (-700.0, 700.0), "daT": (0.0, 180.0),      # T2K CC1pi STV
    "mnv_dat": (0.0, 180.0), "mnv_pn": (0.0, 800.0), "mnv_dpt": (0.0, 2000.0),  # MINERvA CC0piNp STV
    "pip_react": (80.0, 450.0), "pip_abs": (80.0, 450.0),                  # pi+ C reaction/absorption (DUET/Ashery)
    "prot_react": (100.0, 1000.0),                                         # p C reaction
    # e_qe / e_res: JLab (e,e') omega data exists but the exact acceptance range is a follow-up -> unset.
}


def _data_bins(dskeys, row0, edges_by):
    """Global indices of the individual bins whose CENTRE lies in the observable's measured DATA_RANGE."""
    marked = []
    for j, dk in enumerate(dskeys):
        rng, e = DATA_RANGE.get(dk), edges_by.get(dk)
        if rng is None or e is None:
            continue
        cen = 0.5 * (np.asarray(e, float)[:-1] + np.asarray(e, float)[1:])
        marked += [row0[j] + b for b, c in enumerate(cen) if rng[0] <= c <= rng[1]]
    return np.asarray(marked, int)


def _mark_data(ax, marked, nknob, transpose=False):
    """Green margin bar next to each INDIVIDUAL bin that currently has published data (contiguous data
    bins merge into a solid segment; gaps show where the measurement stops)."""
    import matplotlib.collections as mcoll
    if len(marked) == 0:
        return
    # one short bin-wide segment per data bin, drawn just outside the heatmap in the bin-axis margin
    segs = [[(-1.2, i - 0.5), (-1.2, i + 0.5)] if transpose
            else [(i - 0.5, nknob - 0.4), (i + 0.5, nknob - 0.4)] for i in marked]
    lc = mcoll.LineCollection(segs, colors="#2ca02c", linewidths=4.0, zorder=7)
    lc.set_clip_on(False)
    ax.add_collection(lc)


# figure basenames: the canonical T2K npz keeps the historical names; any other label (e.g. a
# multi-sample stack) renders to its OWN <label>_* files so it never clobbers the T2K figures.
_CANON = {"physfit_gate1", "physfit_gate1_full_v2"}


def _fig_names(label):
    if label in _CANON:
        return {"all": "sec2_gradients_all27", "shape": "sec2_gradients_shape", "reach": "sec2_gradient_reach"}
    return {"all": f"{label}_all27", "shape": f"{label}_shape", "reach": f"{label}_reach"}


def main(label="physfit_gate1_full_v2", mark_data=False, transpose=False):
    style.use()
    nm = _fig_names(label)
    d = np.load(style.ALTGEN / f"{label}.npz", allow_pickle=True)
    J, sigma, prior = d["J"], d["sigma"], d["prior"]
    pnames = [str(x) for x in d["pnames"]]
    dskeys = [str(x) for x in d["dskeys"]]
    row0 = np.asarray(d["row0"])
    shrink = d["shrink"]

    S = (prior[None, :] * J) / sigma[:, None]        # (nbins, nknob) per-bin pull in units of sigma
    S = S.T                                          # -> (nknob, nbins)
    nknob, nbin = S.shape
    ctr = [(row0[j] + row0[j + 1]) / 2 - 0.5 for j in range(len(dskeys))]   # observable-block centres
    dslab = [DSLABEL.get(k, k) for k in dskeys]
    edges_by = {k: d[f"{k}_edges"] for k in dskeys if f"{k}_edges" in d.files}
    marked = _data_bins(dskeys, row0, edges_by) if mark_data else np.array([], int)

    # kF_sf pulls ~10 sigma/bin while the FF knobs pull ~0.01 -- a linear scale saturates on kF_sf and
    # whites out everything else, which would UNDERSTATE the claim.  Signed-log keeps the absolute scale
    # (rows stay comparable, unlike a per-panel renormalization) and still resolves the small gradients.
    # linthresh = 0.1 sigma: below that a per-bin pull is negligible and STAYS pale (an honest "no
    # information here"); above it, two decades of colour separate the knobs that actually move the data.
    vmax = float(np.max(np.abs(S)))
    norm = mcolors.SymLogNorm(linthresh=0.1, vmin=-vmax, vmax=vmax, base=10)
    title_all = (f"Exact per-bin gradients, all {nknob} knobs:  "
                 "$\\sigma^{prior}_k\\,\\partial(d\\sigma/dx)_i/\\partial\\theta_k\\;/\\;\\sigma_i$   "
                 "(autodiff through the cascade)")

    if transpose:                                    # bins DOWN the Y axis, knobs across the TOP (tall)
        W = max(7.0, 0.36 * nknob + 2.0); H = max(9.0, 0.026 * nbin + 2.0)
        fig, ax = plt.subplots(figsize=(W, H))
        im = ax.imshow(S.T, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
        for r in row0[1:-1]:
            ax.axhline(r - 0.5, color="k", lw=0.8)
        ax.set_yticks(ctr); ax.set_yticklabels(dslab, fontsize=7)
        ax.set_xticks(range(nknob)); ax.set_xticklabels([_plab(p) for p in pnames], rotation=90, fontsize=8)
        ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
        ax.set_ylabel("bin  (grouped by observable)")
        ax.set_title(title_all + "\n", fontsize=8, pad=30)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01,
                     label="per-bin pull  [$\\sigma$]  for a 1-$\\sigma$ prior move")
        if mark_data:
            _mark_data(ax, marked, nknob, transpose=True)
            ax.text(0.0, 1.006, "green = bins with published data", transform=ax.transAxes,
                    color="#2ca02c", fontsize=8)
        style.save(fig, nm["all"])
    else:
        fig, ax = plt.subplots(figsize=(12.5, 7.4))
        im = ax.imshow(S, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
        for r in row0[1:-1]:                          # dataset boundaries
            ax.axvline(r - 0.5, color="k", lw=0.8)
        ax.set_xticks(ctr); ax.set_xticklabels(dslab, fontsize=7)
        ax.set_yticks(range(nknob)); ax.set_yticklabels([f"{_plab(p)}  " for p in pnames], fontsize=8)
        ax.set_xlabel("bin  (grouped by observable)")
        ax.set_title(title_all, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01,
                     label="per-bin pull  [$\\sigma$]  for a 1-$\\sigma$ prior move")
        if mark_data:
            _mark_data(ax, marked, nknob)
            ax.text(0.995, 1.006, "green = bins with published data", transform=ax.transAxes,
                    ha="right", color="#2ca02c", fontsize=8)
        style.save(fig, nm["all"])

    # ---- per-knob SHAPE: each knob's ROW normalized to its own peak |pull|, so the gradient's
    # DISTRIBUTION across the bins is visible for every knob regardless of magnitude (kF_sf and the tiny
    # FF knobs become equally readable) -- magnitude removed.  The OVERALL Fisher magnitude sqrt(F_kk)
    # deliberately does NOT appear here: aggregating the per-bin pulls into a Fisher (and asking which
    # knobs that makes fittable) is section 3's job -- section 2 shows only the raw per-bin gradients.
    rowmax = np.max(np.abs(S), axis=1, keepdims=True)
    Sshape = np.where(rowmax > 0, S / rowmax, 0.0)
    shape_title = ("Per-knob gradient SHAPE:  each column normalized to its own peak pull  "
                   "(where each knob pulls across the bins; magnitude removed)")

    if transpose:                                    # bins DOWN Y, knobs across the TOP (tall)
        W = max(7.0, 0.36 * nknob + 2.0); H = max(9.0, 0.026 * nbin + 2.0)
        fig, axh = plt.subplots(figsize=(W, H))
        im = axh.imshow(Sshape.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        for r in row0[1:-1]:
            axh.axhline(r - 0.5, color="k", lw=0.8)
        axh.set_yticks(ctr); axh.set_yticklabels(dslab, fontsize=7)
        axh.set_xticks(range(nknob)); axh.set_xticklabels([_plab(p) for p in pnames], rotation=90, fontsize=8)
        axh.xaxis.set_ticks_position("top"); axh.xaxis.set_label_position("top")
        # no y-axis label, no title; colorbar horizontal underneath with only -1/0/1 ticks
        fig.colorbar(im, ax=axh, orientation="horizontal", fraction=0.03, pad=0.03, ticks=[-1, 0, 1])
        if mark_data:
            _mark_data(axh, marked, nknob, transpose=True)
            axh.text(0.0, 1.006, "green = bins with published data", transform=axh.transAxes,
                     color="#2ca02c", fontsize=8)
        style.save(fig, nm["shape"])
    else:
        W = max(13.0, 0.03 * nbin + 3.0); H = max(6.0, 0.26 * nknob + 1.5)
        fig, axh = plt.subplots(figsize=(W, H))
        im = axh.imshow(Sshape, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        for r in row0[1:-1]:
            axh.axvline(r - 0.5, color="k", lw=0.8)
        for i in range(1, nknob):                     # thin separators between knob rows -> read across
            axh.axhline(i - 0.5, color="k", lw=0.3)
        axh.set_xticks(ctr); axh.set_xticklabels(dslab, fontsize=7)
        axh.set_yticks(range(nknob)); axh.set_yticklabels([f"{_plab(p)}  " for p in pnames], fontsize=8)
        axh.xaxis.set_ticks_position("top"); axh.xaxis.set_label_position("top")   # sample labels on top
        axh.tick_params(axis="x", which="both", top=True, bottom=False, direction="out")   # top ticks outward
        axh.tick_params(axis="y", which="both", left=True, right=False, direction="out")   # left only, outward
        # no x/y label, no title; thin horizontal colorbar underneath with only -1/0/1 ticks
        fig.colorbar(im, ax=axh, orientation="horizontal", fraction=0.012, pad=0.05, shrink=0.4, ticks=[-1, 0, 1])
        if mark_data:
            _mark_data(axh, marked, nknob)
            axh.text(0.995, 1.006, "green = bins with published data", transform=axh.transAxes,
                     ha="right", color="#2ca02c", fontsize=8)
        style.save(fig, nm["shape"])


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]      # positional label
    main(*(pos[:1] or []), mark_data="--data" in sys.argv[1:],     # --data: outline the data-backed obs
         transpose="--transpose" in sys.argv[1:])                  # --transpose: bins on Y, knobs on top

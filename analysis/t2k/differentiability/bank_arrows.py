"""Arrow grids where the ARROW LENGTH is the ACTUAL change in the bin (real dsigma/dx units) for a +20%
and +50% variation of each knob -- TWO arrows per bin, drawn against each panel's own y-axis (no abstract
scaling).  Change = the stored Taylor series C(D) = B1*D + B2*D^2/2 + B3*D^3/6 (1st+2nd+3rd derivatives
from the bank), so it is accurate even where the linear term alone is not.  D = frac * ref, ref = |nominal|
(=1 for the multiplicative knobs; Eb_shift uses 1 MeV, f_NN_cex uses 0.5).

7 observables, one 27-knob page each:
  dpt, dat (STV, with T2K acceptance) ; Q2, W, cos(theta_mu), p_mu, p_lead (CC0pi, no acceptance).

  python analysis/t2k/differentiability/bank_arrows.py [bankdir]
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import analysis.t2k.differentiability.bank_plot as BP
from analysis.t2k.differentiability import grad_arrows as GA
import analysis.t2k.differentiability.bank_arrows_obs as BO

CONV_X = 1e-33 / 12.0 * 1e38          # per raw-unit after /binwidth (Q2/W/... inclusive observables)
FRACS = (0.20, 0.50)
_NOM = {"Eb_shift": 0.0, "f_NN_cex": 0.5}
# variation reference scale per knob: Delta = frac * ref.  Multiplicative knobs -> ref = |nominal| (so
# frac is literally +20%/+50%).  ADDITIVE knobs with nominal 0 need a PHYSICAL absolute scale instead:
# Eb_shift is a removal-energy shift in MeV -> ref 20 MeV (so the arrows are +4 / +10 MeV shifts).
_REF_ABS = {"Eb_shift": 20.0}


def _var_grid_fig(name, edges, h0, B1, B2, B3, labels, refs, xlabel, xsc):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    nk, nb = B1.shape
    ctr = 0.5 * (edges[1:] + edges[:-1]) / xsc; xed = edges / xsc
    ncol = 6; nrow = int(np.ceil(nk / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.95 * ncol, 2.35 * nrow))
    axes = np.atleast_1d(axes).ravel()
    styles = [(0.50, 0.018, "50%"), (0.20, 0.011, "20%")]   # (alpha, width, label) drawn faint->solid
    for j, lab in enumerate(labels):
        ax = axes[j]
        D = {f: f * refs[j] for f in FRACS}
        C = {f: B1[j] * D[f] + 0.5 * B2[j] * D[f] ** 2 + B3[j] * D[f] ** 3 / 6.0 for f in FRACS}
        allv = np.concatenate([h0] + [h0 + C[f] for f in FRACS])
        ylo = min(0.0, float(allv.min())); yhi = float(allv.max())
        pad = 0.08 * (yhi - ylo + 1e-300); ylo -= pad; yhi += pad
        ax.fill_between(xed, np.append(h0, h0[-1]), step="post", color="0.92", zorder=0)
        ax.step(xed, np.append(h0, h0[-1]), where="post", color="0.6", lw=1.0, zorder=1)
        ax.axhline(0, color="0.6", lw=0.5)
        for f, (al, w, _) in zip(FRACS, styles):
            col = np.where(C[f] >= 0, "#c0392b", "#2471a3")
            ax.quiver(ctr, h0, np.zeros(nb), C[f], angles="xy", scale_units="xy", scale=1.0,
                      color=col, width=w, headwidth=4, headlength=5, alpha=al, zorder=2 + (f == FRACS[0]))
        ax.set_title(f"{lab}  $|\\Delta|_{{50\\%}}^{{max}}$={np.max(np.abs(C[0.50])):.1e}", fontsize=8)
        ax.set_ylim(ylo, yhi); ax.tick_params(labelsize=7)
        if j >= nk - ncol:
            ax.set_xlabel(xlabel, fontsize=8)
        if j % ncol == 0:
            ax.set_ylabel(r"d$\sigma$/dx", fontsize=8)
    for j in range(nk, len(axes)):
        axes[j].axis("off")
    leg = [Line2D([0], [0], color="0.3", lw=3, alpha=0.5, label="+50% variation"),
           Line2D([0], [0], color="0.3", lw=2, alpha=1.0, label="+20% variation"),
           Line2D([0], [0], color="#c0392b", lw=3, label="bin increases (+)"),
           Line2D([0], [0], color="#2471a3", lw=3, label="bin decreases (−)")]
    fig.legend(handles=leg, loc="lower right", fontsize=9, ncol=2)
    fig.suptitle(f"d$\\sigma$/d({name}): per-bin change for +20% / +50% knob variation "
                 f"(arrow length = real $\\Delta$(d$\\sigma$/dx), Taylor 3rd order; from bank)", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.985])
    return fig


def knob_variation_weights(B, grids, fracs=(0.20, 0.50)):
    """EXACT per-event weights for each knob varied by frac (others nominal).  Returns (w0, Wdict, labels)
    where Wdict[(label,frac)] is the per-event weight (N,) and labels are the 26 plotted knobs."""
    import numpy as _np
    from analysis.t2k.differentiability import bank_reweight as BR, grad_arrows as GA
    from analysis.t2k.differentiability.full_knobs import nominal_knobs
    NOM = nominal_knobs(); SP = GA._specs(NOM)
    JB = BR.to_jax(B)                       # one-time host->device; jitted reweight reuses it
    w0 = _np.asarray(BR.weight_jit(JB, NOM, grids)); W = {}
    for name, idx, label, nomv in SP:
        ref = _REF_ABS.get(label, abs(nomv) or 1.0)
        for fr in fracs:
            k = dict(NOM); v = nomv + fr * ref
            if idx is None:
                k[name] = v
            else:
                b = list(NOM[name]); b[idx] = v; k[name] = tuple(b)
            W[(label, fr)] = _np.asarray(BR.weight_jit(JB, k, grids))
    return w0, W, [s[2] for s in SP]


def ratio_page(name, edges, xsc, xlabel, idx, mask, w0, W, labels, fracs=(0.20, 0.50)):
    """Per-knob grid; each cell = top (nominal dsigma/dx) + bottom EXACT ratio (predicted/nominal per bin)
    for +20% (solid) / +50% (dashed).  Uses the exact reweighted weights W (no Taylor)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    nb = len(edges) - 1; xed = edges / xsc
    def binsum(w): return np.bincount(idx[mask], weights=w[mask].astype(np.float64), minlength=nb)
    h0 = binsum(w0); hsafe = np.where(h0 > 0, h0, np.nan)
    ncol = 6; nrow = int(np.ceil(len(labels) / ncol))
    fig = plt.figure(figsize=(3.0 * ncol, 3.0 * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.45, wspace=0.38)
    sty = {fracs[0]: ("-", "#c0392b"), fracs[1]: ("--", "#2471a3")}
    for j, lab in enumerate(labels):
        r, c = divmod(j, ncol)
        cell = outer[r, c].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.05)
        at = fig.add_subplot(cell[0]); ab = fig.add_subplot(cell[1], sharex=at)
        at.fill_between(xed, np.append(h0, h0[-1]), step="post", color="0.9", zorder=0)
        at.step(xed, np.append(h0, h0[-1]), where="post", color="0.5", lw=1.0)
        at.set_title(lab, fontsize=8); at.tick_params(labelbottom=False, labelsize=6); at.set_ylim(bottom=0)
        allr = []
        for fr in fracs:
            ratio = binsum(W[(lab, fr)]) / hsafe; allr.append(ratio)
            ls, col = sty[fr]
            ab.step(xed, np.append(ratio, ratio[-1]), where="post", color=col, lw=1.3, ls=ls)
        ab.axhline(1.0, color="0.5", lw=0.7, ls=":")
        a = np.concatenate(allr); a = a[np.isfinite(a)]
        if a.size:
            dev = max(0.04, float(np.max(np.abs(a - 1.0))) * 1.45); ab.set_ylim(1 - dev, 1 + dev)
        ab.tick_params(labelsize=6)
        if r == nrow - 1: ab.set_xlabel(xlabel, fontsize=8)
        if c == 0: at.set_ylabel(r"d$\sigma$/dx", fontsize=7); ab.set_ylabel("ratio", fontsize=7)
    leg = [Line2D([0], [0], color="#c0392b", lw=1.5, ls="-", label=f"+{int(fracs[0]*100)}%"),
           Line2D([0], [0], color="#2471a3", lw=1.5, ls="--", label=f"+{int(fracs[1]*100)}%"),
           Line2D([0], [0], color="0.5", lw=0.7, ls=":", label="nominal (=1)")]
    fig.legend(handles=leg, loc="lower right", fontsize=9, ncol=3)
    fig.suptitle(f"d$\\sigma$/d({name}): per-bin EXACT ratio to nominal for +20% / +50% knob variation "
                 f"(bottom; 1.0=no change; full reweight, no Taylor)", fontsize=12)
    return fig


def _ratio_grid_fig(name, edges, h0, B1, B2, B3, labels, refs, xlabel, xsc):
    """Per-knob grid; each cell = top (nominal dsigma/dx) + bottom RATIO panel = predicted/nominal per bin
    for +20% (solid) and +50% (dashed) knob variation (1.0 = no change, 1.04 = +4%).  Taylor 3rd order."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    nk, nb = B1.shape
    xed = edges / xsc
    ncol = 6; nrow = int(np.ceil(nk / ncol))
    fig = plt.figure(figsize=(3.0 * ncol, 3.0 * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.45, wspace=0.38)
    hsafe = np.where(h0 > 0, h0, np.nan)
    for j, lab in enumerate(labels):
        r, c = divmod(j, ncol)
        cell = outer[r, c].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.05)
        at = fig.add_subplot(cell[0]); ab = fig.add_subplot(cell[1], sharex=at)
        at.fill_between(xed, np.append(h0, h0[-1]), step="post", color="0.9", zorder=0)
        at.step(xed, np.append(h0, h0[-1]), where="post", color="0.5", lw=1.0)
        at.set_title(lab, fontsize=8); at.tick_params(labelbottom=False, labelsize=6); at.set_ylim(bottom=0)
        ratios = []
        for D, sty, col in ((0.20, "-", "#c0392b"), (0.50, "--", "#2471a3")):
            d = D * refs[j]
            C = B1[j] * d + 0.5 * B2[j] * d ** 2 + B3[j] * d ** 3 / 6.0
            ratio = 1.0 + C / hsafe
            ratios.append(ratio)
            ab.step(xed, np.append(ratio, ratio[-1]), where="post", color=col, lw=1.3, ls=sty)
        ab.axhline(1.0, color="0.5", lw=0.7, ls=":")
        allr = np.concatenate(ratios); allr = allr[np.isfinite(allr)]
        if allr.size:
            dev = max(0.04, float(np.nanmax(np.abs(allr - 1.0))) * 1.45)   # generous Y-margin
            ab.set_ylim(1 - dev, 1 + dev)
        ab.tick_params(labelsize=6)
        if r == nrow - 1:
            ab.set_xlabel(xlabel, fontsize=8)
        if c == 0:
            at.set_ylabel(r"d$\sigma$/dx", fontsize=7); ab.set_ylabel("ratio", fontsize=7)
    leg = [Line2D([0], [0], color="#c0392b", lw=1.5, ls="-", label="+20%"),
           Line2D([0], [0], color="#2471a3", lw=1.5, ls="--", label="+50%"),
           Line2D([0], [0], color="0.5", lw=0.7, ls=":", label="nominal (=1)")]
    fig.legend(handles=leg, loc="lower right", fontsize=9, ncol=3)
    fig.suptitle(f"d$\\sigma$/d({name}): per-bin RATIO to nominal for +20% / +50% knob variation "
                 f"(bottom panel; 1.0=no change; Taylor 3rd order; from bank)", fontsize=12)
    return fig


def _binned_derivs(B, mask, idx, nb, conv, bw):
    nk = B["D1"].shape[1]
    out = [np.zeros((nk, nb)) for _ in range(3)]
    for o, key in enumerate(("D1", "D2", "D3")):
        for k in range(nk):
            out[o][k] = np.bincount(idx[mask], weights=B[key][mask, k].astype(np.float64), minlength=nb) / bw * conv
    h0 = np.bincount(idx[mask], weights=B["w0"][mask].astype(np.float64), minlength=nb) / bw * conv
    return h0, out[0], out[1], out[2]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    chan = "cc1pi" if "cc1pi" in args else "cc0pi"
    bankdir = next((a for a in args if a not in ("cc0pi", "cc1pi", "stv")), "output/event_bank")
    sys.argv = [sys.argv[0], "dpt"]
    from analysis.t2k.differentiability import tune as T
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    B = BP.load_bank(bankdir)
    labels = B["labels"]
    refs = np.array([_REF_ABS.get(l, abs(_NOM.get(l, 1.0)) or 1.0) for l in labels])
    INCL = {"Q2": "Q^2", "W": "W", "cosmu": "\\cos\\theta_\\mu", "pmu": "p_\\mu", "plead": "p_{lead}"}
    if chan == "cc0pi":
        lead, has_p = BP.leading_proton(B)
        base = (B["prim_pi_pid"] == 0); acc = BP.acceptance(B, lead)
        ed_dpt, c_dpt, _ = GA._obs_binning(T, "dpt"); ed_dat, c_dat, _ = GA._obs_binning(T, "dat")
        SPEC = [
            ("\\delta p_T", BP.dpt(B, lead), ed_dpt, 1e3, r"$\delta p_T$ [GeV/c]", c_dpt, base & acc & has_p),
            ("\\delta\\alpha_T", BP.dat(B, lead), ed_dat, 1.0, r"$\delta\alpha_T$ [rad]", c_dat, base & acc & has_p),
        ]
        for nm, (fn, edges, xsc, xlabel, needp) in BO.OBSDEF.items():
            SPEC.append((INCL[nm], fn(B, lead), edges, xsc, xlabel, CONV_X, base & (has_p if needp else True)))
    else:                                                              # CC1pi+ topological (no acceptance; good
        sig = BP.signal_cc1pi_stv if "stv" in args else BP.signal_cc1pi   # stats). 'stv' arg -> T2K acceptance.
        m1, lead, pip = sig(B); kmu = B["k_mu"].astype(np.float64)
        SPEC = [
            ("\\delta p_T^{1\\pi}", BP.dpt_1pi(kmu, lead, pip), np.linspace(0, 1000, 11), 1e3,
             r"$\delta p_T$ [GeV/c]", CONV_X, m1),
            ("\\delta\\alpha_T^{1\\pi}", BP.dat_1pi(kmu, lead, pip), np.linspace(0, np.pi, 9), 1.0,
             r"$\delta\alpha_T$ [rad]", CONV_X, m1),
            ("\\delta p_{TT}", BP.dptt_1pi(kmu, lead, pip), np.array([-700., -500, -300, -100, 100, 300, 500, 700]),
             1e3, r"$\delta p_{TT}$ [GeV/c]", CONV_X, m1),
            ("p_N", BP.pN_1pi(kmu, lead, pip), np.linspace(0, 800, 11), 1e3, r"$p_N$ [GeV/c]", CONV_X, m1),
            ("p_\\pi", np.linalg.norm(pip[:, 1:], axis=1), np.linspace(0, 1000, 11), 1e3,
             r"$p_\pi$ [GeV/c]", CONV_X, m1),
        ]
        for nm, (fn, edges, xsc, xlabel, _needp) in BO.OBSDEF.items():
            SPEC.append((INCL[nm], fn(B, lead), edges, xsc, xlabel, CONV_X, m1))
    import analysis.t2k.differentiability.bank_reweight as BR
    grids = BR.default_grids()
    print("computing exact per-knob variation weights ...", flush=True)
    w0, W, _lab = knob_variation_weights(B, grids)
    os.makedirs("output/figures", exist_ok=True)
    suffix = "_stv" if (chan == "cc1pi" and "stv" in args) else ""
    out = f"output/figures/{chan}_arrows_variation{suffix}.pdf"
    with PdfPages(out) as pdf:
        for name, vals, edges, xsc, xlabel, _conv, mask in SPEC:
            idx = np.clip(np.searchsorted(edges, vals) - 1, 0, len(edges) - 2)
            fig = ratio_page(name, edges, xsc, xlabel, idx, mask, w0, W, labels)
            pdf.savefig(fig); plt.close(fig)
            print(f"  page: {name}  ({int(mask.sum())} events)", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

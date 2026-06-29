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
    os.makedirs("output/figures", exist_ok=True)
    suffix = "_stv" if (chan == "cc1pi" and "stv" in args) else ""
    out = f"output/figures/{chan}_arrows_variation{suffix}.pdf"
    with PdfPages(out) as pdf:
        for name, vals, edges, xsc, xlabel, conv, mask in SPEC:
            nb = len(edges) - 1; bw = np.diff(edges)
            idx = np.clip(np.searchsorted(edges, vals) - 1, 0, nb - 1)
            h0, B1, B2, B3 = _binned_derivs(B, mask, idx, nb, conv, bw)
            fig = _var_grid_fig(name, edges, h0, B1, B2, B3, labels, refs, xlabel, xsc)
            pdf.savefig(fig); plt.close(fig)
            print(f"  page: {name}  ({int(mask.sum())} events)", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

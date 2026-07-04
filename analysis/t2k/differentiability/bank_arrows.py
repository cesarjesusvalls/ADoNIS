"""EXACT per-bin knob-variation figures from the full-record event bank (no Taylor anywhere): each knob is
varied by +20%/+50% (others nominal), the bank is exactly reweighted (bank_reweight), and each page shows,
per knob, the nominal d(sigma)/dx (top) + the EXACT predicted/nominal per-bin ratio (bottom).

Modes (merged from the former bank_arrows.py / bank_arrows_incl.py / bank_arrows_obs.py):

  cc0pi (default) -- STV dpt/dat (T2K acceptance) + inclusive-CC0pi Q2, W, cos(theta_mu), p_mu, p_lead:
     python analysis/t2k/differentiability/bank_arrows.py [bankdir]
  cc1pi [stv]     -- CC1pi+ topological (or 'stv' = T2K CC1pi+Np acceptance) TKI + kinematics pages:
     python analysis/t2k/differentiability/bank_arrows.py cc1pi [stv] [bankdir]
  incl            -- INCLUSIVE (no signal cut) multiplicities N(p)/N(n)/N(pi+-0) + ranked nucleon momenta:
     python analysis/t2k/differentiability/bank_arrows.py incl [bankdir]

Additive knobs with nominal 0-like values use a physical reference scale instead of a fraction (Eb_shift:
+4/+10 MeV).  bankdir defaults to output/event_bank; the 1M bank lives in the MAIN checkout (set
ADONIS_EVENT_BANK or pass the path).
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import analysis.t2k.differentiability.bank_plot as BP
from analysis.t2k.differentiability import grad_arrows as GA

CONV_X = 1e-33 / 12.0 * 1e38          # per raw-unit after /binwidth (Q2/W/... inclusive observables)
FRACS = (0.20, 0.50)
_NOM = {"Eb_shift": 0.0, "f_NN_cex": 0.5}
# variation reference scale per knob: Delta = frac * ref.  Multiplicative knobs -> ref = |nominal| (so
# frac is literally +20%/+50%).  ADDITIVE knobs with nominal 0 need a PHYSICAL absolute scale instead:
# Eb_shift is a removal-energy shift in MeV -> ref 20 MeV (so the arrows are +4 / +10 MeV shifts).
_REF_ABS = {"Eb_shift": 20.0}


# ---- extra CC0pi observables (former bank_arrows_obs.OBSDEF): name -> (fn, edges, xsc, xlabel, needs_p) - #
def _Q2(B, lead):
    q = B["k_nu"].astype(np.float64) - B["k_mu"].astype(np.float64)
    return np.sum(q[:, 1:] ** 2, 1) - q[:, 0] ** 2                      # MeV^2
def _W(B, lead):
    q = B["k_nu"].astype(np.float64) - B["k_mu"].astype(np.float64)
    had = B["p_struck"].astype(np.float64) + q
    return np.sqrt(np.clip(had[:, 0] ** 2 - np.sum(had[:, 1:] ** 2, 1), 0, None))   # MeV
def _cosmu(B, lead):
    p = B["k_mu"].astype(np.float64)[:, 1:]; return p[:, 2] / np.linalg.norm(p, axis=1)
def _pmu(B, lead):  return np.linalg.norm(B["k_mu"].astype(np.float64)[:, 1:], axis=1)     # MeV
def _plead(B, lead):  return np.linalg.norm(lead[:, 1:], axis=1)                            # MeV

OBSDEF = {
    "Q2":     (_Q2,    np.linspace(0.0, 1.5e6, 16), 1e6,  r"$Q^2$ [GeV$^2$]", False),
    "W":      (_W,     np.linspace(900.0, 1700.0, 17), 1e3, r"$W$ [GeV]", False),
    "cosmu":  (_cosmu, np.linspace(-0.2, 1.0, 13), 1.0,   r"$\cos\theta_\mu$", False),
    "pmu":    (_pmu,   np.linspace(0.0, 1600.0, 17), 1e3, r"$p_\mu$ [GeV/c]", False),
    "plead":  (_plead, np.linspace(0.0, 1200.0, 13), 1e3, r"$p_{\mathrm{lead}}$ [GeV/c]", True),
}


def knob_variation_weights(B, grids, fracs=(0.20, 0.50)):
    """EXACT per-event weights for each knob varied by frac (others nominal).  Returns (w0, Wdict, labels)
    where Wdict[(label,frac)] is the per-event weight (N,) and labels are the plotted knobs."""
    import numpy as _np
    from analysis.t2k.differentiability import bank_reweight as BR
    from analysis.t2k.differentiability.full_knobs import nominal_knobs, knob_specs
    NOM = nominal_knobs(); SP = knob_specs(NOM)
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


def _spec_cc0pi(B, T):
    lead, has_p = BP.leading_proton(B)
    base = (B["prim_pi_pid"] == 0); acc = BP.acceptance(B, lead)
    ed_dpt, c_dpt, _ = GA._obs_binning(T, "dpt"); ed_dat, c_dat, _ = GA._obs_binning(T, "dat")
    INCL = {"Q2": "Q^2", "W": "W", "cosmu": "\\cos\\theta_\\mu", "pmu": "p_\\mu", "plead": "p_{lead}"}
    SPEC = [
        ("\\delta p_T", BP.dpt(B, lead), ed_dpt, 1e3, r"$\delta p_T$ [GeV/c]", c_dpt, base & acc & has_p),
        ("\\delta\\alpha_T", BP.dat(B, lead), ed_dat, 1.0, r"$\delta\alpha_T$ [rad]", c_dat, base & acc & has_p),
    ]
    for nm, (fn, edges, xsc, xlabel, needp) in OBSDEF.items():
        SPEC.append((INCL[nm], fn(B, lead), edges, xsc, xlabel, CONV_X, base & (has_p if needp else True)))
    return SPEC


def _spec_cc1pi(B, stv):
    sig = BP.signal_cc1pi_stv if stv else BP.signal_cc1pi   # 'stv' -> T2K acceptance; else topological
    m1, lead, pip = sig(B); kmu = B["k_mu"].astype(np.float64)
    INCL = {"Q2": "Q^2", "W": "W", "cosmu": "\\cos\\theta_\\mu", "pmu": "p_\\mu", "plead": "p_{lead}"}
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
    for nm, (fn, edges, xsc, xlabel, _needp) in OBSDEF.items():
        SPEC.append((INCL[nm], fn(B, lead), edges, xsc, xlabel, CONV_X, m1))
    return SPEC


def _spec_incl(B):
    """Former bank_arrows_incl: multiplicities + ranked nucleon momenta, ALL events, no signal cut."""
    allm = np.ones(len(B["w0"]), bool)
    np_, nn = BP.n_protons(B), BP.n_neutrons(B)
    npip, npi0, npim = BP.pion_counts(B)
    lp, lp_h = BP.ranked_mom(B, 2212, 0); sp, sp_h = BP.ranked_mom(B, 2212, 1)
    ln, ln_h = BP.ranked_mom(B, 2112, 0); sn, sn_h = BP.ranked_mom(B, 2112, 1)
    ce = np.arange(-0.5, 7.5); me = np.linspace(0.0, 1200.0, 13)
    return [
        ("N_p", np_, ce, 1.0, r"$N_p$", 1.0, allm), ("N_n", nn, ce, 1.0, r"$N_n$", 1.0, allm),
        ("N_{\\pi^+}", npip, ce, 1.0, r"$N_{\pi^+}$", 1.0, allm),
        ("N_{\\pi^0}", npi0, ce, 1.0, r"$N_{\pi^0}$", 1.0, allm),
        ("N_{\\pi^-}", npim, ce, 1.0, r"$N_{\pi^-}$", 1.0, allm),
        ("p_{lead}", lp, me, 1e3, r"$p_{\mathrm{lead}}$ [GeV/c]", 1.0, lp_h),
        ("p_{sublead}", sp, me, 1e3, r"$p_{\mathrm{sublead}}$ [GeV/c]", 1.0, sp_h),
        ("n_{lead}", ln, me, 1e3, r"$n_{\mathrm{lead}}$ [GeV/c]", 1.0, ln_h),
        ("n_{sublead}", sn, me, 1e3, r"$n_{\mathrm{sublead}}$ [GeV/c]", 1.0, sn_h),
    ]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    mode = ("cc1pi" if "cc1pi" in args else "incl" if "incl" in args else "cc0pi")
    bankdir = next((a for a in args if a not in ("cc0pi", "cc1pi", "stv", "incl")),
                   os.environ.get("ADONIS_EVENT_BANK", "output/event_bank"))
    sys.argv = [sys.argv[0], "dpt"]
    from analysis.t2k.differentiability import tune as T
    import analysis.t2k.differentiability.bank_reweight as BR
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    B = BP.load_bank(bankdir)
    if mode == "cc0pi":
        SPEC = _spec_cc0pi(B, T); out = "output/figures/cc0pi_arrows_variation.pdf"
    elif mode == "cc1pi":
        stv = "stv" in args
        SPEC = _spec_cc1pi(B, stv); out = f"output/figures/cc1pi_arrows_variation{'_stv' if stv else ''}.pdf"
    else:
        SPEC = _spec_incl(B); out = "output/figures/incl_arrows_variation.pdf"
    grids = BR.default_grids()
    print("computing exact per-knob variation weights ...", flush=True)
    w0, W, labels = knob_variation_weights(B, grids)
    os.makedirs("output/figures", exist_ok=True)
    with PdfPages(out) as pdf:
        for name, vals, edges, xsc, xlabel, _conv, mask in SPEC:
            idx = np.clip(np.searchsorted(edges, vals) - 1, 0, len(edges) - 2)
            fig = ratio_page(name, edges, xsc, xlabel, idx, mask, w0, W, labels)
            pdf.savefig(fig); plt.close(fig)
            print(f"  page: {name}  ({int(mask.sum())} events)", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

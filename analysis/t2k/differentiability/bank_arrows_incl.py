"""Variation-arrow grids for INCLUSIVE (no signal definition) distributions from the event bank:
  * particle MULTIPLICITIES:  dsigma/dN for N(p), N(n), N(pi+), N(pi0), N(pi-)
  * ranked NUCLEON MOMENTA:   dsigma/d|p| for leading/subleading proton, leading/subleading neutron
All events (QE+RES), NO signal cut.  Same arrow convention as bank_arrows: arrow length = real
Delta(dsigma/dx) for +20% (solid) / +50% (faint) knob variation, all 27 knobs; Eb_shift uses the 20 MeV
physical reference; built from the per-event w0 + D1/D2/D3 (Taylor 3rd order).

  python analysis/t2k/differentiability/bank_arrows_incl.py [bankdir]
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import analysis.t2k.differentiability.bank_plot as BP
import analysis.t2k.differentiability.bank_arrows as BA


def main():
    bankdir = sys.argv[1] if len(sys.argv) > 1 else "output/event_bank"
    B = BP.load_bank(bankdir)
    labels = B["labels"]
    refs = np.array([BA._REF_ABS.get(l, abs(BA._NOM.get(l, 1.0)) or 1.0) for l in labels])
    allm = np.ones(len(B["w0"]), bool)                       # NO signal definition
    np_, nn = BP.n_protons(B), BP.n_neutrons(B)
    npip, npi0, npim = BP.pion_counts(B)
    lp, lp_h = BP.ranked_mom(B, 2212, 0); sp, sp_h = BP.ranked_mom(B, 2212, 1)
    ln, ln_h = BP.ranked_mom(B, 2112, 0); sn, sn_h = BP.ranked_mom(B, 2112, 1)
    cnt_edges = np.arange(-0.5, 7.5)                          # multiplicity bins 0..6
    mom_edges = np.linspace(0.0, 1200.0, 13)                  # |p| bins (MeV)
    # (name, values, edges, xsc, xlabel, mask)
    SPEC = [
        ("N_p",      np_,  cnt_edges, 1.0, r"$N_p$",      allm),
        ("N_n",      nn,   cnt_edges, 1.0, r"$N_n$",      allm),
        ("N_{\\pi^+}", npip, cnt_edges, 1.0, r"$N_{\pi^+}$", allm),
        ("N_{\\pi^0}", npi0, cnt_edges, 1.0, r"$N_{\pi^0}$", allm),
        ("N_{\\pi^-}", npim, cnt_edges, 1.0, r"$N_{\pi^-}$", allm),
        ("p_{lead}",    lp, mom_edges, 1e3, r"$p_{\mathrm{lead}}$ [GeV/c]",    lp_h),
        ("p_{sublead}", sp, mom_edges, 1e3, r"$p_{\mathrm{sublead}}$ [GeV/c]", sp_h),
        ("n_{lead}",    ln, mom_edges, 1e3, r"$n_{\mathrm{lead}}$ [GeV/c]",    ln_h),
        ("n_{sublead}", sn, mom_edges, 1e3, r"$n_{\mathrm{sublead}}$ [GeV/c]", sn_h),
    ]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs("output/figures", exist_ok=True)
    out = "output/figures/incl_arrows_variation.pdf"
    with PdfPages(out) as pdf:
        for name, vals, edges, xsc, xlabel, mask in SPEC:
            nb = len(edges) - 1; bw = np.diff(edges)
            idx = np.clip(np.searchsorted(edges, vals) - 1, 0, nb - 1)
            h0, B1, B2, B3 = BA._binned_derivs(B, mask, idx, nb, BA.CONV_X, bw)
            fig = BA._var_grid_fig(name, edges, h0, B1, B2, B3, labels, refs, xlabel, xsc)
            pdf.savefig(fig); plt.close(fig)
            print(f"  page: {name}  ({int(mask.sum())} events)", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

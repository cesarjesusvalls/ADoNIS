"""EXACT per-bin ratio plots for INCLUSIVE (no signal definition) distributions from the full-record bank:
  * particle MULTIPLICITIES:  dsigma/dN for N(p), N(n), N(pi+), N(pi0), N(pi-)
  * ranked NUCLEON MOMENTA:   dsigma/d|p| for leading/subleading proton, leading/subleading neutron
All events (QE+RES), NO signal cut.  Each knob panel = top (nominal dsigma/dx) + bottom EXACT ratio
predicted/nominal per bin for +20%/+50% (full reweight via bank_reweight, NO Taylor).

  python analysis/t2k/differentiability/bank_arrows_incl.py [bankdir]
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import analysis.t2k.differentiability.bank_plot as BP
import analysis.t2k.differentiability.bank_arrows as BA
import analysis.t2k.differentiability.bank_reweight as BR


def main():
    bankdir = sys.argv[1] if len(sys.argv) > 1 else "output/event_bank"
    B = BP.load_bank(bankdir)
    grids = BR.default_grids()
    print("computing exact per-knob variation weights ...", flush=True)
    w0, W, labels = BA.knob_variation_weights(B, grids)
    allm = np.ones(len(w0), bool)
    np_, nn = BP.n_protons(B), BP.n_neutrons(B)
    npip, npi0, npim = BP.pion_counts(B)
    lp, lp_h = BP.ranked_mom(B, 2212, 0); sp, sp_h = BP.ranked_mom(B, 2212, 1)
    ln, ln_h = BP.ranked_mom(B, 2112, 0); sn, sn_h = BP.ranked_mom(B, 2112, 1)
    ce = np.arange(-0.5, 7.5); me = np.linspace(0.0, 1200.0, 13)
    SPEC = [
        ("N_p", np_, ce, 1.0, r"$N_p$", allm), ("N_n", nn, ce, 1.0, r"$N_n$", allm),
        ("N_{\\pi^+}", npip, ce, 1.0, r"$N_{\pi^+}$", allm),
        ("N_{\\pi^0}", npi0, ce, 1.0, r"$N_{\pi^0}$", allm),
        ("N_{\\pi^-}", npim, ce, 1.0, r"$N_{\pi^-}$", allm),
        ("p_{lead}", lp, me, 1e3, r"$p_{\mathrm{lead}}$ [GeV/c]", lp_h),
        ("p_{sublead}", sp, me, 1e3, r"$p_{\mathrm{sublead}}$ [GeV/c]", sp_h),
        ("n_{lead}", ln, me, 1e3, r"$n_{\mathrm{lead}}$ [GeV/c]", ln_h),
        ("n_{sublead}", sn, me, 1e3, r"$n_{\mathrm{sublead}}$ [GeV/c]", sn_h),
    ]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs("output/figures", exist_ok=True)
    out = "output/figures/incl_arrows_variation.pdf"
    with PdfPages(out) as pdf:
        for name, vals, edges, xsc, xlabel, mask in SPEC:
            idx = np.clip(np.searchsorted(edges, vals) - 1, 0, len(edges) - 2)
            fig = BA.ratio_page(name, edges, xsc, xlabel, idx, mask, w0, W, labels)
            pdf.savefig(fig); plt.close(fig)
            print(f"  page: {name}  ({int(mask.sum())} events)", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

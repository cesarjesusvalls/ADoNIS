"""dsigma/dW & dsigma/dQ2 figures (absolute norm, step-ratio + per-bin chi2) from the large-stat
PION-FIX files vs the matched 1M ACHILLES references.  Free proton (2M) and 12C (1M).
Reuses parse_hepmc + panel from diagnostic_WQ2.  Usage: python scripts/make_fix_figs.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_figures"))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from diagnostic_WQ2 import parse_hepmc, panel

ACH = Path("/Users/homelab/Lab/Playground/projects/DIFFGEN/Achilles")

def make(npz, hepmc, out, lbins, qbins, title, w_is_total_over_N=False):
    d = np.load(npz); dlx, dq2, dw = d["W"], d["Q2"], d["w"]
    if w_is_total_over_N:                      # free-proton npz: w sums to N*sigma -> make absolute
        dw = dw / len(dw)
    alx, aq2, aw, sig_ach = parse_hepmc(hepmc, "res")
    sig_ado = dw.sum()
    print(f"{title}: ACH {len(alx):,} ev sigma={sig_ach:.4e}  ADoNIS {int((dw>0).sum()):,} ev "
          f"sigma={sig_ado:.4e}  ratio={sig_ado/sig_ach:.4f}", flush=True)
    fig, ax = plt.subplots(3, 2, figsize=(14, 9.5), height_ratios=[3, 1, 1], sharex="col")
    c2W = panel(ax[0, 0], ax[1, 0], alx, aw, dlx, dw, lbins, "W=M(Nπ) [MeV]", axc=ax[2, 0])
    c2Q = panel(ax[0, 1], ax[1, 1], aq2, aw, dq2, dw, qbins, "Q² [GeV²]", axc=ax[2, 1])
    fig.suptitle(f"{title} (pion-mass fix) — total ADoNIS/ACH = {sig_ado/sig_ach:.4f}   "
                 f"(N_ACH={len(alx):,}, N_ADO={int((dw>0).sum()):,})   χ²/ndf  W={c2W:.2f}  Q²={c2Q:.2f}")
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"  wrote {out}  (chi2/ndf W={c2W:.2f} Q2={c2Q:.2f})", flush=True)

# Free proton (mono 1 GeV): tight bins; npz w sums to N*sigma
make("scripts/free_proton_2M_mpi0.npz", str(ACH / "_resrun_out/nofsi_res_H_mono_1M.hepmc"),
     "paper_figures/res_WQ2_free_proton_mpi0_2M.png",
     np.linspace(1080, 1560, 22), np.linspace(0, 1.15, 22),
     "RES free proton (mono 1 GeV) ABSOLUTE", w_is_total_over_N=True)

# 12C (T2K flux): wider bins; npz w already sums to sigma
make("scripts/res_events_spline_1M_mpi0.npz", str(ACH / "_drvout/nofsi_res_hi_1M.hepmc"),
     "paper_figures/res_WQ2_12C_mpi0_1M.png",
     np.linspace(1080, 2000, 16), np.linspace(0, 2.0, 18),
     "RES 12C (T2K flux) ABSOLUTE", w_is_total_over_N=False)

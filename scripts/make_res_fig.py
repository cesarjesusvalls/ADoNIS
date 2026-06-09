"""Make the RES dsigma/dW & dsigma/dQ2 figure (step ratio panels) from a SAVED ADoNIS .npz
and an ACHILLES hepmc. Reuses parse_hepmc + panel (step ratios) from diagnostic_WQ2.
Usage: python scripts/make_res_fig.py <adonis.npz> <achilles.hepmc> <out.png>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_figures"))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from diagnostic_WQ2 import parse_hepmc, panel

npz = sys.argv[1] if len(sys.argv) > 1 else "scripts/res_events_1M.npz"
hepmc = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parents[2] / "Achilles/_resrun_out/nofsi_res_hi.hepmc")
out = sys.argv[3] if len(sys.argv) > 3 else "paper_figures/res_WQ2_shapes_matched.png"

d = np.load(npz)
dlx, dq2, dw = d["W"], d["Q2"], d["w"]
alx, aq2, aw, sig_ach = parse_hepmc(hepmc, "res")
sig_ado = dw.sum()
print(f"ACHILLES: {len(alx)} ev, sigma={sig_ach:.4e} nb   ADoNIS: {len(dlx)} ev, sigma={sig_ado:.4e} nb"
      f"   total ADoNIS/ACH={sig_ado/sig_ach:.3f}")

lbins = np.linspace(1080, 2000, 16); qbins = np.linspace(0, 2.0, 18)
fig, axes = plt.subplots(3, 2, figsize=(14, 9.5), height_ratios=[3, 1, 1], sharex="col")
c2W = panel(axes[0, 0], axes[1, 0], alx, aw, dlx, dw, lbins, "W=M(Nπ) [MeV]", axc=axes[2, 0])
c2Q = panel(axes[0, 1], axes[1, 1], aq2, aw, dq2, dw, qbins, "Q² [GeV²]", axc=axes[2, 1])
fig.suptitle(f"RES primary (no FSI): ACHILLES vs ADoNIS — total ADoNIS/ACH = {sig_ado/sig_ach:.3f}"
             f"   (N_ACH={len(alx):,}, N_ADO={len(dlx):,})   χ²/ndf  W={c2W:.2f}  Q²={c2Q:.2f}")
fig.tight_layout(); fig.savefig(out, dpi=120)
print(f"chi2/ndf  W={c2W:.2f}  Q2={c2Q:.2f}"); print("wrote", out)

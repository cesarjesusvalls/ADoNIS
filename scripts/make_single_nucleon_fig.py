"""Single-nucleon (free proton, mono 1 GeV) RES dsigma/dW & dsigma/dQ2, ADoNIS vs ACHILLES,
step-ratio style. ADoNIS = free_proton_gen.generate (free proton, fix applied); ACHILLES =
nofsi_res_H_mono.hepmc.  ABSOLUTE normalization: ADoNIS weights sum to sigma (= mean weight),
ACHILLES weighted to its GenCrossSection (pb->nb) -- so the ratio panel shows the true ~1%
absolute deficit, NOT area-matched.  Saves the ADoNIS (W,Q2,w) to npz. Usage: python ... <N>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_figures"))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import jax; jax.config.update("jax_enable_x64", True)
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"   # bit-faithful to ACHILLES
from scripts.free_proton_gen import generate, parse_oracle, ACHHEPMC
from diagnostic_WQ2 import panel

def ach_sigma_nb(path):
    g = None
    with open(path) as fh:
        for line in fh:
            if line[:1] == "A" and "GenCrossSection" in line:
                g = float(line.split()[3])      # pb, cumulative (last wins)
    return g * 1e-3                              # pb -> nb

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
NPZ = Path("scripts/single_nucleon_events.npz")
# Reuse saved events unless regen requested (pass "regen" as 2nd arg). Generation is the ~75s
# cost (spline amps2); plotting is 0.3s -- so cosmetic re-plots load from the npz instantly.
if NPZ.exists() and "regen" not in sys.argv:
    d = np.load(NPZ); W, Q2, w = d["W"], d["Q2"], d["w"]
    print(f"loaded {len(W):,} events from {NPZ} (pass 'regen' to regenerate)")
else:
    W, Q2, w = generate(N)
    w = w / N                                    # absolute: weights now sum to sigma (mean weight)
    m = w > 0; W, Q2, w = W[m], Q2[m], w[m]
    np.savez_compressed(NPZ, W=W, Q2=Q2, w=w)
    print(f"generated + saved {len(W):,} events -> {NPZ}")
sig_ado = w.sum(); sig_ach = ach_sigma_nb(ACHHEPMC)
aW, aQ2 = parse_oracle(); aw = np.ones(len(aW)) / len(aW) * sig_ach   # ABSOLUTE (hepmc GenCrossSection)
print(f"ADoNIS {len(W):,} ev  sigma={sig_ado:.4e} nb   ACHILLES {len(aW):,} ev  sigma={sig_ach:.4e} nb"
      f"   ratio ADoNIS/ACH={sig_ado/sig_ach:.4f}")

lbins = np.linspace(1080, 1500, 22); qbins = np.linspace(0, 1.15, 22)
fig, axes = plt.subplots(3, 2, figsize=(14, 9.5), height_ratios=[3, 1, 1], sharex="col")
c2W = panel(axes[0, 0], axes[1, 0], aW, aw, W, w, lbins, "W=M(Nπ) [MeV]", axc=axes[2, 0])
c2Q = panel(axes[0, 1], axes[1, 1], aQ2, aw, Q2, w, qbins, "Q² [GeV²]", axc=axes[2, 1])
fig.suptitle(f"RES single proton (free, mono 1 GeV) ABSOLUTE: ACHILLES vs ADoNIS — total ADoNIS/ACH"
             f" = {sig_ado/sig_ach:.3f}   (N_ACH={len(aW):,}, N_ADO={len(W):,})   χ²/ndf  W={c2W:.2f}  Q²={c2Q:.2f}")
fig.tight_layout(); fig.savefig("paper_figures/res_WQ2_single_nucleon.png", dpi=120)
print("wrote paper_figures/res_WQ2_single_nucleon.png")

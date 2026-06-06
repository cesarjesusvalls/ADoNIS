"""Paper Fig 14 (arXiv:2508.19213v2): N N -> N N pi cross sections pp -> pn pi+ and pp -> pp pi0
vs beam momentum / sqrt(s) -- model-only (ACHILLES vs the GiBUU parameterisation).

ADoNIS evaluates the Dmitriev-Sushkov / GiBUU N N -> N Delta -> N N pi cross section directly
(adonis/fsi/mb/nn_delta.py, an exact port of ACHILLES ResonanceHelper.cc + NucleonNucleon.cc),
so the ADoNIS curve IS the GiBUU parameterisation the ACHILLES cascade samples.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from adonis.fsi.mb import nn_delta as nd
ROOT = Path(__file__).resolve().parents[1]


def pbeam(sqrts):
    El = (sqrts ** 2 - 2 * nd.M_N ** 2) / (2 * nd.M_N)
    return np.sqrt(El ** 2 - nd.M_N ** 2)


sqrts = np.linspace(2.02, 3.2, 60)
pn = np.array([nd.sigma_pp_channels(s)[0] for s in sqrts])
pp = np.array([nd.sigma_pp_channels(s)[1] for s in sqrts])
pb = np.array([pbeam(s) for s in sqrts])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(sqrts, pn, color="tab:blue", lw=2, label=r"$pp\to pn\pi^+$")
ax1.plot(sqrts, pp, color="tab:red", lw=2, label=r"$pp\to pp\pi^0$")
ax1.set_xlabel(r"$\sqrt{s}$ [GeV]"); ax1.set_ylabel(r"$\sigma$ [mb]")
ax1.set_title("N N -> N N pi (GiBUU Dmitriev-Sushkov)"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(pb, pn, color="tab:blue", lw=2, label=r"$pp\to pn\pi^+$")
ax2.plot(pb, pp, color="tab:red", lw=2, label=r"$pp\to pp\pi^0$")
ax2.set_xlabel(r"$p_{beam}$ [GeV]"); ax2.set_ylabel(r"$\sigma$ [mb]")
ax2.set_xlim(0.8, 3.5)
ax2.set_title(f"peak {pn.max():.0f}:{pp.max():.0f} mb  (5:1 isospin)"); ax2.legend(); ax2.grid(alpha=0.3)
fig.suptitle("Fig 14 — N N -> N N pi: ADoNIS (= GiBUU parameterisation = ACHILLES INC input)")
fig.tight_layout()
out = ROOT / "paper_figures" / "fig14_nn_nnpi.png"
fig.savefig(out, dpi=130)
print("wrote", out, f" peak pn pi+ {pn.max():.1f} mb, pp pi0 {pp.max():.1f} mb")

"""Phase A1 validation figure: NC single-pion σ(E_nu) within the [10,90]° lepton angular
acceptance, the four EM channels, ADoNIS vs ACHILLES (stationary 1H + 1N, e⁻ probe).

The EM total is 1/Q⁴-divergent forward, so this is σ-in-acceptance (matching ACHILLES's
AngleTheta cut), the EM analog of A3's free-nucleon σ(E_ν). A single universal constant c
(fit over the whole energy × 4-channel grid) bridges the model (relative units) to ACHILLES
(nb) — so the figure shows the parameter-free agreement of energy dependence + channel ratios.

  ADONIS_A2_N=150000 python scripts/make_a2_nc_figure.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adonis.params import PhysicsParams
from adonis.primary.dcc.sigma_enu import nc_sigma_channels_at

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("ADONIS_A2_N", "150000"))
LABELS = [r"$\nu p\to \nu p\,\pi^0$", r"$\nu p\to \nu n\,\pi^+$",
          r"$\nu n\to \nu n\,\pi^0$", r"$\nu n\to \nu p\,\pi^-$"]
COL = ["tab:green", "tab:blue", "tab:olive", "tab:red"]

ref = np.loadtxt(ROOT / "data" / "oracle" / "freenucleon_nc_sigma.csv")
E, ach = ref[:, 0], ref[:, 1:5]

mod = np.zeros_like(ach)
for i, e in enumerate(E):
    _, sc = nc_sigma_channels_at(PhysicsParams(), jax.random.fold_in(jax.random.PRNGKey(7), i),
                                 float(e), N, chunk=50_000)
    mod[i] = np.asarray(sc)
c = float(np.exp(np.mean(np.log(ach / mod))))
pred = c * mod
rel = np.abs(pred - ach) / ach

fig, (ax, axr) = plt.subplots(2, 1, figsize=(6, 6), height_ratios=[3, 1], sharex=True)
for ci in range(4):
    ax.plot(E / 1000, ach[:, ci], "o", color=COL[ci], ms=5, label=f"ACHILLES {LABELS[ci]}")
    ax.plot(E / 1000, pred[:, ci], "-", color=COL[ci], lw=1.5, label=f"ADoNIS {LABELS[ci]}")
    axr.plot(E / 1000, 100 * (pred[:, ci] - ach[:, ci]) / ach[:, ci], "o-", color=COL[ci], ms=3)
ax.set_ylabel(r"$\sigma$ in [nb]")
ax.set_title(f"A1: NC single-pion $\\sigma(E_nu)$ — max rel {rel.max():.1%}, mean {rel.mean():.1%}")
ax.legend(fontsize=6, ncol=2)
axr.axhline(0, color="k", lw=0.5)
axr.set_ylabel("model−ACH [%]"); axr.set_xlabel(r"$E_nu$ [GeV]"); axr.set_ylim(-8, 8)
fig.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "a2_nc_sigma.png", dpi=130)
print(f"wrote {out/'a2_nc_sigma.png'}  (c={c:.4g}, max rel {rel.max():.3f})")

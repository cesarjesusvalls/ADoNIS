"""Phase A1 validation figure: EM single-pion σ(E_e) within the [10,90]° lepton angular
acceptance, the four EM channels, ADoNIS vs ACHILLES (stationary 1H + 1N, e⁻ probe).

The EM total is 1/Q⁴-divergent forward, so this is σ-in-acceptance (matching ACHILLES's
AngleTheta cut), the EM analog of A3's free-nucleon σ(E_ν). A single universal constant c
(fit over the whole energy × 4-channel grid) bridges the model (relative units) to ACHILLES
(nb) — so the figure shows the parameter-free agreement of energy dependence + channel ratios.

  ADONIS_A1_N=150000 python scripts/make_a1_em_figure.py
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
from adonis.primary.dcc.sigma_enu import em_sigma_channels_at

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("ADONIS_A1_N", "150000"))
# Validated comparison set: the two proton channels + the neutron TOTAL (the neutron
# pi0/pi- split is a known open EM-isospin item -- see docs/phases/PHASE_A1.md).
LABELS = [r"$e p\to e p\,\pi^0$", r"$e p\to e n\,\pi^+$", r"$e n\to e N\pi$ (total)"]
COL = ["tab:green", "tab:blue", "tab:purple"]

ref = np.loadtxt(ROOT / "data" / "oracle" / "freenucleon_em_sigma.csv")
E, ach4 = ref[:, 0], ref[:, 1:5]
ach = np.stack([ach4[:, 0], ach4[:, 1], ach4[:, 2] + ach4[:, 3]], axis=1)

mod4 = np.zeros_like(ach4)
for i, e in enumerate(E):
    _, sc = em_sigma_channels_at(PhysicsParams(), jax.random.fold_in(jax.random.PRNGKey(7), i),
                                 float(e), N, chunk=50_000)
    mod4[i] = np.asarray(sc)
mod = np.stack([mod4[:, 0], mod4[:, 1], mod4[:, 2] + mod4[:, 3]], axis=1)
c = float(np.exp(np.mean(np.log(ach / mod))))
pred = c * mod
rel = np.abs(pred - ach) / ach

fig, (ax, axr) = plt.subplots(2, 1, figsize=(6, 6), height_ratios=[3, 1], sharex=True)
for ci in range(3):
    ax.plot(E / 1000, ach[:, ci], "o", color=COL[ci], ms=5, label=f"ACHILLES {LABELS[ci]}")
    ax.plot(E / 1000, pred[:, ci], "-", color=COL[ci], lw=1.5, label=f"ADoNIS {LABELS[ci]}")
    axr.plot(E / 1000, 100 * (pred[:, ci] - ach[:, ci]) / ach[:, ci], "o-", color=COL[ci], ms=3)
ax.set_ylabel(r"$\sigma$ in $[10,90]^\circ$ acceptance [nb]")
ax.set_title(f"A1: EM single-pion $\\sigma(E_e)$ — max rel {rel.max():.1%}, mean {rel.mean():.1%}")
ax.legend(fontsize=6, ncol=2)
axr.axhline(0, color="k", lw=0.5)
axr.set_ylabel("model−ACH [%]"); axr.set_xlabel(r"$E_e$ [GeV]"); axr.set_ylim(-8, 8)
fig.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "a1_em_sigma.png", dpi=130)
print(f"wrote {out/'a1_em_sigma.png'}  (c={c:.4g}, max rel {rel.max():.3f})")

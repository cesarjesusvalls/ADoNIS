"""Phase A3 validation figure: free-nucleon single-pion σ(E_ν), the three CC isospin
channels, ADoNIS vs ACHILLES (stationary 1H + 1N, ν_e).

Reproduces the structure of the paper's Fig. anl_bnl (minus the ANL/BNL data overlay,
which lands with A3.5).  The model is in relative units; a single universal constant c
(fit over the whole 9-energy × 3-channel grid) bridges to ACHILLES — so this figure
shows the parameter-free agreement of the **energy dependence and channel ratios**.

  ADONIS_A3_N=150000 python scripts/make_a3_sigma_figure.py
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
from adonis.primary.dcc.sigma_enu import sigma_channels_at, CM2_1E38_PER_NB

ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("ADONIS_A3_N", "150000"))
LABELS = [r"$\nu n\to\ell^- p\,\pi^0$", r"$\nu n\to\ell^- n\,\pi^+$",
          r"$\nu p\to\ell^- p\,\pi^+$"]
COL = ["tab:green", "tab:orange", "tab:blue"]

ref = np.loadtxt(ROOT / "data" / "oracle" / "freenucleon_nue_sigma.csv")
E, ach = ref[:, 0], ref[:, 1:4]

mod = np.zeros_like(ach)
for i, e in enumerate(E):
    _, sc = sigma_channels_at(PhysicsParams(), jax.random.fold_in(jax.random.PRNGKey(11), i),
                              float(e), N, chunk=50_000)
    mod[i] = np.asarray(sc)
c = float(np.exp(np.mean(np.log(ach / mod))))          # single bridging constant
pred = c * mod
rel = np.abs(pred - ach) / ach
U = CM2_1E38_PER_NB                                     # nb -> 10^-38 cm^2 (Fig-2 axis)

fig, (ax, axr) = plt.subplots(2, 1, figsize=(6, 6), height_ratios=[3, 1], sharex=True)
for ci in range(3):
    ax.plot(E / 1000, ach[:, ci] * U, "o", color=COL[ci], ms=5, label=f"ACHILLES {LABELS[ci]}")
    ax.plot(E / 1000, pred[:, ci] * U, "-", color=COL[ci], lw=1.5,
            label=f"ADoNIS {LABELS[ci]}")
    axr.plot(E / 1000, 100 * (pred[:, ci] - ach[:, ci]) / ach[:, ci], "o-", color=COL[ci], ms=3)
ax.set_ylabel(r"$\sigma$  [$10^{-38}\,\mathrm{cm}^2$]")
ax.set_title(f"A3: free-nucleon single-pion $\\sigma(E_\\nu)$  —  "
             f"max rel {rel.max():.1%}, mean {rel.mean():.1%} (one constant c)")
ax.legend(fontsize=7, ncol=1)
axr.axhline(0, color="k", lw=0.5)
axr.set_ylabel("model−ACH [%]"); axr.set_xlabel(r"$E_\nu$ [GeV]")
axr.set_ylim(-8, 8)
fig.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "a3_freenucleon_sigma.png", dpi=130)
print(f"wrote {out/'a3_freenucleon_sigma.png'}  (c={c:.4g}, max rel {rel.max():.3f})")

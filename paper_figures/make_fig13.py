"""Paper Fig 13 (arXiv:2508.19213v2): meson-baryon scattering -- pi N -> pi N total sigma(W)
(left) and the dsigma/dOmega angular distribution (right).  Model-only in the paper (ACHILLES
INC vs the ANL-Osaka DCC analytic); ADoNIS evaluates the ANL-Osaka DCC partial-wave amplitudes
directly (Phase E, adonis/fsi/mb/anl_xsec.py) -- the SAME amplitudes the ACHILLES cascade's
MesonBaryonInteraction samples, so the two coincide by construction.

Reproduces the paper's pi N content (the eta N / K Lambda channels need the ANL_0-{1,2,3}
amplitude tables, which are absent from the extracted data dump).
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from adonis.fsi.mb.anl_xsec import pip_p_total, pim_p_elastic, pim_p_cex, dsigma_dOmega
ROOT = Path(__file__).resolve().parents[1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
W, sp = pip_p_total(); _, se = pim_p_elastic(); _, sc = pim_p_cex()
ax1.plot(W, sp, color="tab:red", lw=2, label=r"$\pi^+p\to\pi^+p$")
ax1.plot(W, sc, color="tab:green", lw=2, label=r"$\pi^-p\to\pi^0n$ (cex)")
ax1.plot(W, se, color="tab:blue", lw=2, label=r"$\pi^-p\to\pi^-p$")
ax1.axvline(1232, ls=":", color="gray"); ax1.set_xlim(1080, 1400)
m = (W >= 1150) & (W <= 1350); pk = lambda s: s[m].max()
ax1.set_xlabel("W [MeV]"); ax1.set_ylabel(r"$\sigma$ [mb]")
ax1.set_title(f"ANL-Osaka DCC $\\pi N$ total $\\sigma$  (Δ ratio {pk(sp)/pk(se):.1f}:{pk(sc)/pk(se):.1f}:1)")
ax1.legend()

c = np.linspace(-1, 1, 121)
for W0, col in [(1232.0, "tab:red"), (1160.0, "tab:purple"), (1320.0, "tab:orange")]:
    d = dsigma_dOmega(W0, c); ax2.plot(c, d / d.mean(), color=col, label=f"W={W0:.0f}")
ax2.plot(c, (1 + 3 * c ** 2) / np.mean(1 + 3 * c ** 2), "k--", lw=1, label=r"$1+3\cos^2\theta$")
ax2.set_xlabel(r"$\cos\theta_{cm}$"); ax2.set_ylabel(r"$d\sigma/d\Omega$ (norm.)")
ax2.set_title(r"$\pi^+p$ angular (P33 at the Δ)"); ax2.legend(fontsize=8)
fig.suptitle("Fig 13 — meson-baryon scattering: ANL-Osaka DCC (= ACHILLES INC input), ADoNIS Phase E")
fig.tight_layout()
out = ROOT / "paper_figures" / "fig13_meson_baryon.png"
fig.savefig(out, dpi=130)
print("wrote", out, f" Delta isospin ratio {pk(sp)/pk(se):.2f}:{pk(sc)/pk(se):.2f}:1")

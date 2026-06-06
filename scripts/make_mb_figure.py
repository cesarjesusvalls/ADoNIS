"""Phase E1 figure: ANL-Osaka piN scattering -- total sigma(W) for the three piN channels
(the 9:2:1 Delta isospin pattern) and the dsigma/dOmega angular shape at vs off the Delta.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.fsi.mb.anl_xsec import pip_p_total, pim_p_elastic, pim_p_cex, dsigma_dOmega, _CEX_CG
ROOT = Path(__file__).resolve().parents[1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# left: total cross sections vs W
W, sp = pip_p_total()
_, se = pim_p_elastic()
_, sc = pim_p_cex()
ax1.plot(W, sp, label=r"$\pi^+p\to\pi^+p$ (I=3/2)", color="tab:red")
ax1.plot(W, sc, label=r"$\pi^-p\to\pi^0n$ (cex)", color="tab:green")
ax1.plot(W, se, label=r"$\pi^-p\to\pi^-p$ (elastic)", color="tab:blue")
ax1.axvline(1232, ls=":", color="gray", lw=1)
ax1.set_xlim(1080, 1400); ax1.set_xlabel("W [MeV]"); ax1.set_ylabel(r"$\sigma$ [mb]")
m = (W >= 1150) & (W <= 1350); pk = lambda s: s[m].max()
ax1.set_title(f"$\\pi N$ total $\\sigma$  (9:2:1 at Δ: "
              f"{pk(sp)/pk(se):.1f}:{pk(sc)/pk(se):.1f}:1)")
ax1.legend()

# right: angular distribution at the Delta vs off-resonance
c = np.linspace(-1, 1, 121)
for W0, col in [(1232.0, "tab:red"), (1160.0, "tab:purple"), (1320.0, "tab:orange")]:
    d = dsigma_dOmega(W0, c)
    ax2.plot(c, d / d.mean(), color=col, label=f"W={W0:.0f} MeV")
ax2.plot(c, (1 + 3 * c ** 2) / np.mean(1 + 3 * c ** 2), "k--", lw=1, label=r"$1+3\cos^2\theta$")
ax2.set_xlabel(r"$\cos\theta_{cm}$"); ax2.set_ylabel(r"$d\sigma/d\Omega$ (norm. to mean)")
ax2.set_title(r"$\pi^+p$ angular shape (P33 at the Δ)")
ax2.legend()

fig.suptitle("ANL-Osaka meson-baryon (DCC PWA) -- Phase E1")
fig.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "mb_pin_scattering.png", dpi=130)
print("wrote", out / "mb_pin_scattering.png")

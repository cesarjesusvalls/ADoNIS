"""Phase B2 validation: ADoNIS inclusive (e,e') vs the ACHILLES oracle, with ratio + chi2.

For each component -- QE (QE_Spectral_Func) and 1pi/Delta (RES_Spectral_Func) -- overlays the
ADoNIS model on the ACHILLES omega spectrum (E=2.222 GeV, theta~15.5 deg, 12C), with a
ratio (model/oracle) sub-panel and a chi2/ndf, the standard model-vs-oracle comparison.
Both are unit-area-normalised over the comparison window (shape comparison).
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from adonis.nuclear.qe_inclusive import qe_dsigma_domega
from adonis.nuclear.inclusive_1pi import onepi_dsigma_domega
from adonis.core.validation import chi2_ndf
ROOT = Path(__file__).resolve().parents[1]
E, TH = 2222.0, 15.541

COMPONENTS = [
    dict(key="qe", csv="inclusive_ee_12C_qe.csv", title="QE", color="tab:orange",
         model=lambda w: qe_dsigma_domega(E, TH, w, "pke12p_tot.data", 6, 6)),
    dict(key="res", csv="inclusive_ee_12C_res.csv", title=r"1$\pi$ (Δ) — RES", color="tab:green",
         model=lambda w: onepi_dsigma_domega(E, TH, w, "pke12p_tot.data", 12)),
]

fig = plt.figure(figsize=(11, 5.5))
gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.25)

for j, c in enumerate(COMPONENTS):
    d = np.loadtxt(ROOT / "data" / "oracle" / c["csv"])
    ow, osh, oerr = d[:, 0], d[:, 1], d[:, 2]
    m = c["model"](ow)
    # unit-area normalise over the window (uniform bins -> sum; matches chi2_ndf)
    msum = m.sum(); osum = osh.sum()
    mn, on, oen = m / msum, osh / osum, oerr / osum
    chi2, ndf = chi2_ndf(m, np.zeros_like(m), osh, oerr, floor=0.05)

    ax = fig.add_subplot(gs[0, j])
    ax.plot(ow, mn, "-", color=c["color"], lw=2, label="ADoNIS")
    ax.errorbar(ow, on, yerr=oen, fmt="ks", ms=4, capsize=2, label="ACHILLES")
    ax.set_title(f"{c['title']}   χ²/ndf = {chi2/max(ndf,1):.2f}  (ndf {ndf})")
    ax.set_ylabel("dσ/dω (unit area)"); ax.legend(); ax.set_xticklabels([])
    ax.grid(alpha=0.3)

    axr = fig.add_subplot(gs[1, j], sharex=ax)
    good = on > 0.05 * on.max()
    ratio = np.where(good, mn / np.where(on > 0, on, np.nan), np.nan)
    rerr = np.where(good, ratio * oen / np.where(on > 0, on, np.nan), np.nan)
    axr.errorbar(ow[good], ratio[good], yerr=rerr[good], fmt="o", ms=3, color=c["color"])
    axr.axhline(1.0, ls="--", color="gray"); axr.set_ylim(0.4, 1.6)
    axr.set_xlabel(r"$\omega$ [MeV]"); axr.set_ylabel("model/ACH")
    axr.grid(alpha=0.3)

fig.suptitle(r"ADoNIS vs ACHILLES inclusive (e,e') on $^{12}$C, $E$=2.222 GeV, $\theta$=15.5°")
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "inclusive_ee_validation_c12.png", dpi=130, bbox_inches="tight")
print("wrote", out / "inclusive_ee_validation_c12.png")
for c in COMPONENTS:
    d = np.loadtxt(ROOT / "data" / "oracle" / c["csv"])
    chi2, ndf = chi2_ndf(c["model"](d[:, 0]), np.zeros(len(d)), d[:, 1], d[:, 2], floor=0.05)
    print(f"  {c['key']}: chi2/ndf = {chi2/max(ndf,1):.2f}  (ndf {ndf})")

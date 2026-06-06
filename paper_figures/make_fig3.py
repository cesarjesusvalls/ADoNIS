"""Paper Fig 3 (arXiv:2508.19213v2): pi+ on 12C reaction & absorption cross section sigma(p_pi),
ACHILLES vs ADoNIS.

This is the pion-nucleus FSI observable that the cascade is built to reproduce.  Both curves
use the SAME, untuned in-medium cross sections (the exact Oset absorption + ANL-Osaka DCC
meson-baryon scatter); the comparison is therefore a direct validation of the differentiable
ADoNIS cascade transport (adonis/fsi/cascade_real.py) against the ACHILLES Virtual-Resonances
cascade (achilles:cascade, Mode: CrossSection -> data/oracle/cascade_pip_c12_virt_abs.csv,
this repo's scripts/cascade_abs_from_hepmc.py).

Key result (see memory cascade-absorption-fraction-029): the ABSORPTION FRACTION agrees at the
Delta -- ADoNIS 0.28-0.30 vs ACHILLES 0.29-0.31 -- and the sigma(p) SHAPE agrees; ADoNIS sits
~1.3x above ACHILLES in absolute normalisation (flat in p), the residual transport difference
(continuum mean-free-path through the smooth rho(r) vs ACHILLES's discrete impact-parameter
walk over correlated QMC nucleon configs).  As elsewhere in this project, ADoNIS is bridged to
the ACHILLES oracle by ONE constant and we report chi2/ndf of the shape.

DUET / Ashery pi+-C absorption data (the paper's experimental overlay) are digitised separately
in the paper and are NOT in NUISANCE; the experimental points are pending that digitisation.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ach = np.loadtxt(ROOT / "data" / "oracle" / "cascade_pip_c12_virt_abs.csv")
ado = np.loadtxt(ROOT / "data" / "oracle" / "cascade_pip_c12_virt_abs_adonis.csv")
p_a, sr_a, sa_a, nacc = ach[:, 0], ach[:, 1], ach[:, 2], ach[:, 3]
p_d, sr_d, sa_d = ado[:, 0], ado[:, 1], ado[:, 2]
assert np.allclose(p_a, p_d), "oracle binning mismatch"
# ACHILLES Poisson stat error per bin: sigma scales with sqrt(n)/n = 1/sqrt(n)
err_r = sr_a / np.sqrt(np.clip(nacc, 1, None))
# absorbed-event count per bin ~ nacc * (sa/sr); its Poisson error propagates to sa
nabs = nacc * np.divide(sa_a, sr_a, out=np.zeros_like(sa_a), where=sr_a > 0)
err_a = sa_a / np.sqrt(np.clip(nabs, 1, None))

PANELS = [
    dict(key="reaction", title=r"$\pi^+\,^{12}$C reaction $\sigma$", ya=sr_a, ey=err_r, yd=sr_d, col="tab:blue"),
    dict(key="absorption", title=r"$\pi^+\,^{12}$C absorption $\sigma$", ya=sa_a, ey=err_a, yd=sa_d, col="tab:red"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 6), height_ratios=[3, 1], sharex=True)
for j, pn in enumerate(PANELS):
    ax, axr = axes[0, j], axes[1, j]
    ya, yd, ey = pn["ya"], pn["yd"], pn["ey"]
    m = (ya > 0) & (yd > 0)
    dlt = m & (p_a >= 200) & (p_a <= 360)           # the Delta region (physics-relevant)
    # bridge constant over the Delta region (where the shape agrees), error-weighted
    w = (ya[dlt] / ey[dlt]) ** 2
    c = float(np.exp(np.sum(w * np.log(ya[dlt] / yd[dlt])) / np.sum(w)))
    pred = c * yd
    chi2d = float(np.sum(((pred[dlt] - ya[dlt]) / ey[dlt]) ** 2)); ndd = int(dlt.sum()) - 1
    chi2f = float(np.sum(((pred[m] - ya[m]) / ey[m]) ** 2)); ndf = int(m.sum()) - 1
    ax.errorbar(p_a, ya, yerr=ey, fmt="o-", color="0.3", lw=1.5, ms=4, capsize=2, label="ACHILLES (VirtRes)")
    ax.plot(p_d, yd, ":", color=pn["col"], lw=1.5, label="ADoNIS (raw)")
    ax.plot(p_d, pred, "-", color=pn["col"], lw=2, label=f"ADoNIS (bridged ×{c:.2f})")
    ax.axvspan(200, 360, color="gold", alpha=0.08)
    ax.set_title(f"{pn['title']}   χ²/ndf: Δ-region {chi2d/max(ndd,1):.1f}, full {chi2f/max(ndf,1):.1f}", fontsize=10)
    ax.set_ylim(bottom=0)
    if j == 0:
        ax.set_ylabel(r"$\sigma$ [mb]")
    ax.legend(fontsize=8)
    # ratio panel: ADoNIS(raw)/ACHILLES  and the bridged ratio (with ACHILLES error band)
    axr.fill_between(p_a[m], 1 - ey[m] / ya[m], 1 + ey[m] / ya[m], color="0.8")
    axr.plot(p_a[m], yd[m] / ya[m], ":", color=pn["col"], lw=1.5)
    axr.plot(p_a[m], pred[m] / ya[m], "-", color=pn["col"], lw=2)
    axr.axhline(1, ls="--", color="0.5")
    axr.set_ylim(0.5, 2.2); axr.set_xlabel(r"$p_\pi$ [MeV]")
    if j == 0:
        axr.set_ylabel("ADoNIS/ACH")
# absorption-fraction annotation
fa = sa_a.sum() / sr_a.sum(); fd = sa_d.sum() / sr_d.sum()
fig.suptitle(r"Fig 3 — $\pi^+\,^{12}$C reaction & absorption $\sigma(p_\pi)$: ACHILLES vs ADoNIS "
             f"(abs. fraction ACH {fa:.2f} / ADoNIS {fd:.2f}; same untuned Oset+DCC xsecs).  "
             "DUET/Ashery data pending digitisation.", fontsize=9)
fig.tight_layout()
out = ROOT / "paper_figures" / "fig3_pion_carbon_xsec.png"
fig.savefig(out, dpi=130)
print("wrote", out)
print(f"abs fraction: ACHILLES {fa:.3f}  ADoNIS {fd:.3f}")

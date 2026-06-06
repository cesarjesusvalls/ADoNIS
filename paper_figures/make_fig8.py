"""Paper Fig 8 (arXiv:2508.19213v2): T2K CC1pi+ single-transverse-variable (STV) cross sections
-- delta_pTT, p_N, delta_alphaT -- EXPERIMENT (T2K, arXiv:2102.03346) vs ACHILLES vs ADoNIS.

Data: NUISANCE local text release (../nuisance/data/T2K/CC1pipNp_STV/xsec_{dpTT,pN,daT}.txt),
absolute dsigma/dx per nucleon + covariance.  ACHILLES: the paper run_T2K_virtresonances full
event generation (achilles:fullcascade), CC1pi+ signal extracted by scripts/extract_t2k_cc1pi_tki.
ADoNIS: T2K-flux-folded DCC CC1pi+ with the spectral-function nuclear model + REAL pion (Oset+
DCC) and nucleon (GiBUU NN-elastic) FSI (scripts/gen_t2k_cc1pi_adonis).

Shape comparison (area-normalised dsigma/dx): ADoNIS and ACHILLES are each compared to the data
shape; chi2/ndf uses the data diagonal errors.  FSI (pion absorption/charge-exchange + proton
rescattering) is what bends the predictions toward the data tails.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DDIR = ROOT.parent / "nuisance" / "data" / "T2K" / "CC1pipNp_STV"


def parse_stv(fname):
    """Parse xsec_*.txt -> (edges, dsigma/dx values, diag errors from the covariance)."""
    lines = (DDIR / fname).read_text().splitlines()
    edges = np.array([float(x) for x in lines[0].split(":")[1].split()])
    vals = np.array([float(x) for x in lines[1].split(":")[1].split()])
    # the covariance matrix follows the line starting "Cross-section covariance matrix"
    i = next(k for k, l in enumerate(lines) if l.startswith("Cross-section covariance matrix"))
    cov = np.array([[float(x) for x in lines[i + 1 + r].split()] for r in range(len(vals))])
    return edges, vals, np.sqrt(np.diag(cov))


def hist_shape(x, w, edges):
    """area-normalised dsigma/dx on the given edges (= shape)."""
    h, _ = np.histogram(x, bins=edges, weights=w)
    width = np.diff(edges)
    dens = h / width
    area = np.sum(dens * width)
    return dens / area if area > 0 else dens


OBS = [
    dict(key="dptt", file="xsec_dpTT.txt", label=r"$\delta p_{TT}$ [MeV]", conv=1.0),
    dict(key="pn", file="xsec_pN.txt", label=r"$p_N$ [MeV]", conv=1.0),
    dict(key="dalphat", file="xsec_daT.txt", label=r"$\delta\alpha_T$ [deg]", conv=np.degrees(1.0)),
]

ach = np.load(ROOT / "data" / "oracle" / "t2k_cc1pi_tki_achilles.npz")
ado = np.load(ROOT / "data" / "oracle" / "t2k_cc1pi_tki_adonis_fsi.npz")
ado0 = np.load(ROOT / "data" / "oracle" / "t2k_cc1pi_tki_adonis_nofsi.npz")

fig, axes = plt.subplots(2, 3, figsize=(14, 6), height_ratios=[3, 1])
for j, o in enumerate(OBS):
    ax, axr = axes[0, j], axes[1, j]
    edges, dval, derr = parse_stv(o["file"])
    cen = 0.5 * (edges[1:] + edges[:-1]); width = np.diff(edges)
    # normalise data to unit area (shape)
    area = np.sum(dval * width); dval_n, derr_n = dval / area, derr / area
    xa = ach[o["key"]] * o["conv"]; xd = ado[o["key"]] * o["conv"]; xd0 = ado0[o["key"]] * o["conv"]
    sa = hist_shape(xa, ach["w"], edges)
    sd = hist_shape(xd, ado["w"], edges)
    sd0 = hist_shape(xd0, ado0["w"], edges)

    def chi2(model):
        good = derr_n > 0
        return float(np.sum(((model[good] - dval_n[good]) / derr_n[good]) ** 2)), int(good.sum())
    c2a, nd = chi2(sa); c2d, _ = chi2(sd)

    ax.errorbar(cen, dval_n, yerr=derr_n, xerr=width / 2, fmt="o", color="k", ms=4, capsize=2, label="T2K data")
    ax.step(cen, sa, where="mid", color="0.4", lw=1.8, label=f"ACHILLES (χ²/ndf={c2a/max(nd,1):.1f})")
    ax.step(cen, sd, where="mid", color="tab:red", lw=2, label=f"ADoNIS+FSI (χ²/ndf={c2d/max(nd,1):.1f})")
    ax.step(cen, sd0, where="mid", color="tab:red", lw=1, ls=":", alpha=0.7, label="ADoNIS no-FSI")
    ax.set_title(o["label"], fontsize=10); ax.set_ylim(bottom=0)
    if j == 0:
        ax.set_ylabel(r"$(1/\sigma)\,d\sigma/dx$"); ax.legend(fontsize=7)
    axr.axhspan(1 - 0, 1 + 0, color="0.9")
    axr.errorbar(cen, dval_n / np.clip(sd, 1e-12, None), yerr=derr_n / np.clip(sd, 1e-12, None),
                 fmt="o", color="tab:red", ms=3)
    axr.axhline(1, ls="--", color="0.5"); axr.set_ylim(0.3, 1.9); axr.set_xlabel(o["label"])
    if j == 0:
        axr.set_ylabel("data/ADoNIS")
fig.suptitle("Fig 8 — T2K CC1π⁺ STV (CH): data vs ACHILLES vs ADoNIS (spectral-fn + real pion & "
             "nucleon FSI; area-normalised shapes)", fontsize=10)
fig.tight_layout()
out = ROOT / "paper_figures" / "fig8_t2k_cc1pi_stv.png"
fig.savefig(out, dpi=130)
print("wrote", out)

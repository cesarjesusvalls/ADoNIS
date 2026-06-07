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
    """Area-normalised dsigma/dx AND its per-bin MC stat error (shape/sqrt(Neff_bin)).

    Neff_bin = (Sigma w)^2/Sigma(w^2); the relative error sqrt(Sigma w^2)/Sigma w rides
    straight through the global area rescale.  Returns (shape, abs_error)."""
    h, _ = np.histogram(x, bins=edges, weights=w)
    h2, _ = np.histogram(x, bins=edges, weights=w ** 2)
    width = np.diff(edges)
    dens = h / width
    area = np.sum(dens * width)
    if area <= 0:
        return dens, np.zeros_like(dens)
    shp = dens / area
    rel = np.sqrt(h2) / np.clip(h, 1e-300, None)
    return shp, shp * rel


OBS = [
    dict(key="dptt", file="xsec_dpTT.txt", label=r"$\delta p_{TT}$ [MeV]", conv=1.0),
    dict(key="pn", file="xsec_pN.txt", label=r"$p_N$ [MeV]", conv=1.0),
    dict(key="dalphat", file="xsec_daT.txt", label=r"$\delta\alpha_T$ [deg]", conv=np.degrees(1.0)),
]

ach = np.load(ROOT / "data" / "oracle" / "t2k_cc1pi_tki_achilles.npz")
# bit-exact RES primary (adonis/xsec) if present, else the older DCC-bridge surrogate
_x = ROOT / "data" / "oracle" / "t2k_cc1pi_tki_adonis_xsec.npz"
ado = np.load(_x) if _x.exists() else np.load(ROOT / "data" / "oracle" / "t2k_cc1pi_tki_adonis_fsi.npz")
ado0 = np.load(ROOT / "data" / "oracle" / "t2k_cc1pi_tki_adonis_nofsi.npz")

fig, axes = plt.subplots(2, 3, figsize=(14, 6), height_ratios=[3, 1])
for j, o in enumerate(OBS):
    ax, axr = axes[0, j], axes[1, j]
    edges, dval, derr = parse_stv(o["file"])
    cen = 0.5 * (edges[1:] + edges[:-1]); width = np.diff(edges)
    # normalise data to unit area (shape)
    area = np.sum(dval * width); dval_n, derr_n = dval / area, derr / area
    xa = ach[o["key"]] * o["conv"]; xd = ado[o["key"]] * o["conv"]; xd0 = ado0[o["key"]] * o["conv"]
    sa, sa_e = hist_shape(xa, ach["w"], edges)
    sd, sd_e = hist_shape(xd, ado["w"], edges)
    sd0, _ = hist_shape(xd0, ado0["w"], edges)

    def chi2(model, merr):
        """chi2 vs data folding BOTH data and model (MC) stat errors into the denominator."""
        good = derr_n > 0; denom = derr_n[good] ** 2 + merr[good] ** 2
        return float(np.sum((model[good] - dval_n[good]) ** 2 / denom)), int(good.sum())
    c2a, nd = chi2(sa, sa_e); c2d, _ = chi2(sd, sd_e)

    ax.errorbar(cen, dval_n, yerr=derr_n, xerr=width / 2, fmt="o", color="k", ms=4, capsize=2, label="T2K data")
    ax.step(cen, sa, where="mid", color="0.4", lw=1.8, label=f"ACHILLES (χ²/ndf={c2a/max(nd,1):.1f})")
    ax.step(cen, sd, where="mid", color="tab:red", lw=2, label=f"ADoNIS+FSI (χ²/ndf={c2d/max(nd,1):.1f})")
    ax.step(cen, sd0, where="mid", color="tab:red", lw=1, ls=":", alpha=0.7, label="ADoNIS no-FSI")
    ax.set_title(f"{o['label']}   χ²/ndf vs data: ADoNIS {c2d/max(nd,1):.1f}, ACH {c2a/max(nd,1):.1f}",
                 fontsize=9); ax.set_ylim(bottom=0)
    if j == 0:
        ax.set_ylabel(r"$(1/\sigma)\,d\sigma/dx$"); ax.legend(fontsize=7)
    # closure metric: ACHILLES/ADoNIS per-bin ratio with propagated MC stat errors (+/-3% band)
    sac = np.clip(sa, 1e-12, None); sdc = np.clip(sd, 1e-12, None)
    rca = sa / sdc
    rca_e = rca * np.sqrt((sa_e / sac) ** 2 + (sd_e / sdc) ** 2)
    axr.axhspan(0.97, 1.03, color="tab:green", alpha=0.15)
    axr.step(cen, rca, where="mid", color="tab:red", lw=1.5)
    axr.errorbar(cen, rca, yerr=rca_e, fmt="o", color="tab:red", ms=3, capsize=2, lw=1)
    axr.axhline(1, ls="--", color="0.5"); axr.set_ylim(0.3, 1.9); axr.set_xlabel(o["label"])
    miss = np.abs(rca - 1) - rca_e > 0.03
    axr.set_title(f"max|r-1|={100*np.max(np.abs(rca-1)):.0f}%  ({int(miss.sum())} off>3σ)", fontsize=8)
    if j == 0:
        axr.set_ylabel("ACHILLES / ADoNIS")
fig.suptitle("Fig 8 — T2K CC1π⁺ STV (CH): data vs ACHILLES vs ADoNIS (spectral-fn + real pion & "
             "nucleon FSI; area-normalised shapes)", fontsize=10)
fig.tight_layout()
out = ROOT / "paper_figures" / "fig8_t2k_cc1pi_stv.png"
fig.savefig(out, dpi=130)
print("wrote", out)

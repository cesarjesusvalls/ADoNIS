"""Paper Fig 7 (arXiv:2508.19213v2): T2K CC0pi-Np single-transverse-variable cross sections --
delta_pT and delta_alphaT -- EXPERIMENT (T2K, arXiv:1802.05078) vs ACHILLES vs ADoNIS.

Tight signal phase space (NUISANCE T2K_CC0pi_STV, PRD): p_mu>250, cos_mu>-0.6, leading proton
450-1000 MeV/c with cos_p>0.4, no mesons.  Data from the NUISANCE ROOT release; ACHILLES from
the paper run_T2K_virtresonances full-event generation (CC0pi extracted by
scripts/extract_t2k_cc0pi_tki.py); ADoNIS = T2K-flux-folded CCQE (Llewellyn-Smith) off the
spectral function + discrete-Glauber proton FSI (scripts/gen_t2k_cc0pi_adonis.py).  Area-
normalised shapes; chi2/ndf vs data reported for both ADoNIS and ACHILLES.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DDIR = ROOT.parent / "nuisance" / "data" / "T2K" / "CC0pi" / "STV"


def load_root(fname):
    r = uproot.open(DDIR / fname)["Result"]
    v, e = r.to_numpy(); return e, v, r.errors()


def shape(x, w, edges):
    """Area-normalised shape AND its per-bin statistical error.

    A weighted histogram bin has variance Sigma(w^2); the effective count is
    Neff_bin = (Sigma w)^2 / Sigma(w^2) so the *relative* stat error on the bin is
    sqrt(Sigma w^2)/Sigma w = 1/sqrt(Neff_bin).  Area-normalisation is a global
    rescale, so it carries that relative error straight onto the shape:
    err_bin = shape_bin / sqrt(Neff_bin).  Returns (shape, abs_error)."""
    h, _ = np.histogram(x, bins=edges, weights=w)
    h2, _ = np.histogram(x, bins=edges, weights=w ** 2)         # Sigma w^2 per bin
    wd = np.diff(edges)
    d = h / wd; s = np.sum(d * wd)
    if s <= 0:
        return d, np.zeros_like(d)
    shp = d / s
    rel = np.sqrt(h2) / np.clip(h, 1e-300, None)                # 1/sqrt(Neff_bin)
    return shp, shp * rel


ach = np.load(ROOT / "data" / "oracle" / "t2k_cc0pi_tki_achilles.npz")
# ADoNIS CC0pi = bit-exact spectral-fn QE + RES-with-pion-absorbed (adonis/xsec), each at its OWN
# first-principles absolute nb weight (sigma_QE, sigma_RES) -- the faithful ACHILLES composition,
# NOT a fitted fraction.  Falls back to the older combined/CCQE samples if the xsec one is absent.
for _name in ("t2k_cc0pi_tki_adonis_xsec.npz", "t2k_cc0pi_tki_adonis_combined.npz",
              "t2k_cc0pi_tki_adonis_fsi.npz"):
    _p = ROOT / "data" / "oracle" / _name
    if _p.exists():
        ado = np.load(_p); break
ado0 = np.load(ROOT / "data" / "oracle" / "t2k_cc0pi_tki_adonis_nofsi.npz")

PANELS = [
    dict(key="dpt", root="dptResults.root", label=r"$\delta p_T$ [GeV]", conv=1e-3),   # MeV->GeV
    dict(key="dalphat", root="datResults.root", label=r"$\delta\alpha_T$ [rad]", conv=1.0),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 6), height_ratios=[3, 1])
for j, pn in enumerate(PANELS):
    ax, axr = axes[0, j], axes[1, j]
    edges, dval, derr = load_root(pn["root"])
    cen = 0.5 * (edges[1:] + edges[:-1]); width = np.diff(edges)
    area = np.sum(dval * width); dn, en = dval / area, derr / area
    xa = ach[pn["key"]] * pn["conv"]; xd = ado[pn["key"]] * pn["conv"]; xd0 = ado0[pn["key"]] * pn["conv"]
    sa, sa_e = shape(xa, ach["w"], edges); sd, sd_e = shape(xd, ado["w"], edges); sd0, _ = shape(xd0, ado0["w"], edges)

    # CLOSURE metric (THE acceptance test): chi2 of ACHILLES vs ADoNIS, folding BOTH MC stat
    # errors into the denominator -- are the two codes statistically consistent bin-by-bin?
    gc = (sa_e > 0) | (sd_e > 0)
    denom_c = sa_e[gc] ** 2 + sd_e[gc] ** 2
    c2c = float(np.sum((sa[gc] - sd[gc]) ** 2 / np.clip(denom_c, 1e-300, None)))
    ndc = int(gc.sum())
    print(f"[{pn['key']}] ACHILLES-vs-ADoNIS closure chi2/ndf = {c2c/max(ndc,1):.2f} ({c2c:.1f}/{ndc})")
    rab = sd / np.clip(sa, 1e-12, None)                 # ADoNIS/ACHILLES per-bin ratio (THE metric)
    print(f"[{pn['key']}] ADoNIS/ACHILLES per-bin ratio:", np.array2string(rab, precision=3))
    print(f"   max|r-1| = {100*np.max(np.abs(rab-1)):.1f}%  "
          f"({'PASS <3%' if np.max(np.abs(rab-1))<0.03 else 'FAIL'})")
    ax.errorbar(cen, dn, yerr=en, xerr=width / 2, fmt="o", color="k", ms=4, capsize=2, label="T2K data")
    ax.step(cen, sa, where="mid", color="0.4", lw=1.8, label="ACHILLES")
    ax.step(cen, sd, where="mid", color="tab:blue", lw=2, label="ADoNIS+FSI")
    ax.step(cen, sd0, where="mid", color="tab:blue", lw=1, ls=":", alpha=0.7, label="ADoNIS no-FSI")
    ax.set_title(f"{pn['label']}   ACHILLES-vs-ADoNIS closure χ²/ndf = {c2c/max(ndc,1):.2f}", fontsize=9)
    ax.set_ylim(bottom=0)
    if j == 0:
        ax.set_ylabel(r"$(1/\sigma)\,d\sigma/dx$"); ax.legend(fontsize=8)
    # THE acceptance criterion: ACHILLES / ADoNIS per-bin ratio (target 1 +/- 3%).
    # Error propagated from BOTH MC samples: r*sqrt((sa_e/sa)^2 + (sd_e/sd)^2).
    sac = np.clip(sa, 1e-12, None); sdc = np.clip(sd, 1e-12, None)
    rca = sa / sdc
    rca_e = rca * np.sqrt((sa_e / sac) ** 2 + (sd_e / sdc) ** 2)
    axr.axhspan(0.97, 1.03, color="tab:green", alpha=0.15)          # +/-3% band
    axr.step(cen, rca, where="mid", color="tab:blue", lw=1.8)
    axr.errorbar(cen, rca, yerr=rca_e, fmt="o", color="tab:blue", ms=3, capsize=2, lw=1)
    axr.axhline(1, ls="--", color="0.5"); axr.set_ylim(0.85, 1.15); axr.set_xlabel(pn["label"])
    # how many bins are >3% away by MORE than their own error bar (a real, not statistical, miss)
    miss = np.abs(rca - 1) - rca_e > 0.03
    mx = 100 * np.max(np.abs(rca - 1))
    axr.set_title(f"max |ratio-1| = {mx:.1f}%   ({int(miss.sum())} bins off >3sigma)", fontsize=8)
    if j == 0:
        axr.set_ylabel("ACHILLES / ADoNIS")
fig.suptitle("Fig 7 — T2K CC0π-Np STV (CH): data vs ACHILLES vs ADoNIS "
             "(spectral-fn CCQE + discrete-Glauber proton FSI; tight phase space)", fontsize=10)
fig.tight_layout()
out = ROOT / "paper_figures" / "fig7_t2k_cc0pi_stv.png"
fig.savefig(out, dpi=130); print("wrote", out)

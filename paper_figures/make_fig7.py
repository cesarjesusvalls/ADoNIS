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
    h, _ = np.histogram(x, bins=edges, weights=w); wd = np.diff(edges)
    d = h / wd; s = np.sum(d * wd); return d / s if s > 0 else d


ach = np.load(ROOT / "data" / "oracle" / "t2k_cc0pi_tki_achilles.npz")
ado = np.load(ROOT / "data" / "oracle" / "t2k_cc0pi_tki_adonis_fsi.npz")
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
    sa = shape(xa, ach["w"], edges); sd = shape(xd, ado["w"], edges); sd0 = shape(xd0, ado0["w"], edges)

    def chi2(m):
        g = en > 0; return float(np.sum(((m[g] - dn[g]) / en[g]) ** 2)), int(g.sum())
    c2a, nd = chi2(sa); c2d, _ = chi2(sd)
    ax.errorbar(cen, dn, yerr=en, xerr=width / 2, fmt="o", color="k", ms=4, capsize=2, label="T2K data")
    ax.step(cen, sa, where="mid", color="0.4", lw=1.8, label="ACHILLES")
    ax.step(cen, sd, where="mid", color="tab:blue", lw=2, label="ADoNIS+FSI")
    ax.step(cen, sd0, where="mid", color="tab:blue", lw=1, ls=":", alpha=0.7, label="ADoNIS no-FSI")
    ax.set_title(f"{pn['label']}   χ²/ndf vs data: ADoNIS {c2d/max(nd,1):.1f}, ACH {c2a/max(nd,1):.1f}", fontsize=9)
    ax.set_ylim(bottom=0)
    if j == 0:
        ax.set_ylabel(r"$(1/\sigma)\,d\sigma/dx$"); ax.legend(fontsize=8)
    axr.errorbar(cen, dn / np.clip(sd, 1e-12, None), yerr=en / np.clip(sd, 1e-12, None),
                 fmt="o", color="tab:blue", ms=3)
    axr.axhline(1, ls="--", color="0.5"); axr.set_ylim(0.3, 1.9); axr.set_xlabel(pn["label"])
    if j == 0:
        axr.set_ylabel("data/ADoNIS")
fig.suptitle("Fig 7 — T2K CC0π-Np STV (CH): data vs ACHILLES vs ADoNIS "
             "(spectral-fn CCQE + discrete-Glauber proton FSI; tight phase space)", fontsize=10)
fig.tight_layout()
out = ROOT / "paper_figures" / "fig7_t2k_cc0pi_stv.png"
fig.savefig(out, dpi=130); print("wrote", out)

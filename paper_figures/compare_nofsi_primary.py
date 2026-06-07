"""PRIMARY (pre-FSI) closure: ACHILLES vs ADoNIS, T2K CC0pi-Np STV, freshly generated under
IDENTICAL conditions (QE_Spectral_Func, T2K flux, 12C, Cascade:Run:False, same CC0pi-Np cuts).

ACHILLES: docker `run_nofsi_qe.yml` -> nofsi_qe.hepmc -> extract_t2k_cc0pi_tki (50k events).
ADoNIS:   scripts/gen_nofsi_qe_adonis.py (importance QE, current bit-exact code, same cuts).
Comparing the two PRE-FSI predictions isolates the hard generator from the cascade.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ach = np.load(ROOT / "data" / "oracle" / "nofsi_qe_achilles.npz")
ado = np.load(ROOT / "data" / "oracle" / "nofsi_qe_adonis.npz")

PANELS = [dict(key="dpt", label=r"$\delta p_T$ [MeV]", edges=np.linspace(0, 800, 9)),
          dict(key="dalphat", label=r"$\delta\alpha_T$ [rad]", edges=np.linspace(0, np.pi, 9))]


def shape(x, w, edges):
    h, _ = np.histogram(x, bins=edges, weights=w)
    h2, _ = np.histogram(x, bins=edges, weights=w ** 2)
    wd = np.diff(edges); d = h / wd; s = np.sum(d * wd)
    if s <= 0:
        return d, np.zeros_like(d)
    shp = d / s; rel = np.sqrt(h2) / np.clip(h, 1e-300, None)
    return shp, shp * rel


fig, axes = plt.subplots(2, 2, figsize=(11, 6), height_ratios=[3, 1])
for j, pn in enumerate(PANELS):
    ax, axr = axes[0, j], axes[1, j]
    edges = pn["edges"]; cen = 0.5 * (edges[1:] + edges[:-1])
    sa, sa_e = shape(ach[pn["key"]], ach["w"], edges); sd, sd_e = shape(ado[pn["key"]], ado["w"], edges)
    gc = (sa_e > 0) | (sd_e > 0)
    c2 = float(np.sum((sa[gc] - sd[gc]) ** 2 / np.clip(sa_e[gc] ** 2 + sd_e[gc] ** 2, 1e-300, None)))
    ndc = int(gc.sum())
    ax.errorbar(cen, sa, yerr=sa_e, fmt="s", color="0.4", ms=4, capsize=2, label="ACHILLES (no FSI)")
    ax.errorbar(cen, sd, yerr=sd_e, fmt="o", color="tab:blue", ms=3, capsize=2, label="ADoNIS (no FSI)")
    ax.step(cen, sa, where="mid", color="0.4", lw=1.2, alpha=0.6)
    ax.step(cen, sd, where="mid", color="tab:blue", lw=1.2, alpha=0.6)
    ax.set_title(f"{pn['label']}   primary closure χ²/ndf = {c2/max(ndc,1):.2f}", fontsize=9)
    ax.set_ylim(bottom=0)
    if j == 0:
        ax.set_ylabel(r"$(1/\sigma)\,d\sigma/dx$"); ax.legend(fontsize=8)
    sac = np.clip(sa, 1e-12, None); sdc = np.clip(sd, 1e-12, None)
    rca = sa / sdc; rca_e = rca * np.sqrt((sa_e / sac) ** 2 + (sd_e / sdc) ** 2)
    axr.axhspan(0.97, 1.03, color="tab:green", alpha=0.15)
    axr.step(cen, rca, where="mid", color="tab:blue", lw=1.5)
    axr.errorbar(cen, rca, yerr=rca_e, fmt="o", color="tab:blue", ms=3, capsize=2, lw=1)
    axr.axhline(1, ls="--", color="0.5"); axr.set_ylim(0.5, 1.5); axr.set_xlabel(pn["label"])
    miss = np.abs(rca - 1) - rca_e > 0.03
    axr.set_title(f"max|r-1|={100*np.max(np.abs(rca-1)):.0f}%  ({int(miss.sum())} off>3σ)", fontsize=8)
    if j == 0:
        axr.set_ylabel("ACHILLES / ADoNIS")
    print(f"[{pn['key']}] primary closure χ²/ndf={c2/max(ndc,1):.2f}")
fig.suptitle("PRIMARY (pre-FSI) closure — ACHILLES vs ADoNIS, T2K CC0π-Np QE\n"
             "freshly generated, identical conditions (QE_Spectral_Func, T2K flux, ¹²C, no cascade)",
             fontsize=10)
fig.tight_layout()
out = ROOT / "paper_figures" / "compare_nofsi_primary.png"
fig.savefig(out, dpi=130); print("wrote", out)

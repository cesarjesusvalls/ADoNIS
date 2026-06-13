"""CC0pi-Np ACHILLES/ADoNIS absolute ratios, broken down by TRUE process (QE vs RES).
ACHILLES truth tag = NuHepMC signal_process_id (200 = QE, 401/402 = RES single-pion).
ADoNIS truth = the disaggregated generator cells (QE-C, RES-C+RES-H for CH).

Columns: QE no-FSI | QE with-FSI | RES with-FSI.   (QE/RES no-FSI->RES is identically 0
on both sides: the RES pion survives the no-meson cut, so it is omitted.)
Rows: vertex W, Q2, delta_pT, delta_alphaT.   First-principles absolute nb, no fit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from adonis.data.oracle.normalization import weight_to_nb_of
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ORA = Path("data/oracle")
dis = np.load(ORA / "cc0pi_disaggregated.npz")
ACH_FSI = np.load(ORA / "t2k_cc0pi_tki_achilles.npz")          # 2M FSI   (has proc)
ACH_NOF = np.load(ORA / "t2k_cc0pi_tki_achilles_nofsi.npz")    # 200k no-FSI
SCALE_FSI = weight_to_nb_of(ACH_FSI)
SCALE_NOF = weight_to_nb_of(ACH_NOF)
KEYS = ("W", "Q2", "dalphat", "dpt")


def ach_cut(d, scale, sel):
    """ACHILLES absolute-nb arrays for the process-selected subset."""
    return {k: np.asarray(d[k])[sel] for k in KEYS} | {"w": np.asarray(d["w"])[sel] * scale}


def ado_cells(cells, tag):
    return {k: np.concatenate([dis[f"{c}_{tag}_{k}"] for c in cells]) for k in KEYS + ("w",)}


# (column label, ACHILLES subset, ADoNIS subset)
pn = np.asarray(ACH_NOF["proc"]); pf = np.asarray(ACH_FSI["proc"])
COLS = [
    ("QE  no-FSI", ach_cut(ACH_NOF, SCALE_NOF, pn == 200), ado_cells(["QE-C"], "False")),
    ("QE  with-FSI", ach_cut(ACH_FSI, SCALE_FSI, pf == 200), ado_cells(["QE-C"], "True")),
    ("RES  with-FSI", ach_cut(ACH_FSI, SCALE_FSI, pf >= 400), ado_cells(["RES-C", "RES-H"], "True")),
]
VARS = [("W", np.linspace(850, 1500, 21), "vertex W [MeV]"),
        ("Q2", np.linspace(0, 1.4, 21), r"$Q^2$ [GeV$^2$]"),
        ("dpt", np.linspace(0, 800, 21), r"$\delta p_T$ [MeV]"),
        ("dalphat", np.linspace(0, np.pi, 21), r"$\delta\alpha_T$ [rad]")]


def he(v, w, bins):
    v = np.asarray(v); m = np.isfinite(v) & np.isfinite(w)
    h, _ = np.histogram(v[m], bins=bins, weights=w[m])
    h2, _ = np.histogram(v[m], bins=bins, weights=w[m] ** 2)
    return h, np.sqrt(h2)


fig, axes = plt.subplots(len(VARS), len(COLS), figsize=(5.0 * len(COLS), 3.9 * len(VARS)))
for ci, (clab, ach, ado) in enumerate(COLS):
    sig_a, sig_d = ach["w"].sum(), ado["w"].sum()
    for ri, (key, bins, xlab) in enumerate(VARS):
        ax = axes[ri, ci]; ctr = 0.5 * (bins[1:] + bins[:-1])
        ha, ea = he(ach[key], ach["w"], bins); hd, ed = he(ado[key], ado["w"], bins)
        msk = (ha > 0) & (hd > 0)
        r = np.where(msk, ha / np.where(hd > 0, hd, 1), np.nan)
        re = r * np.sqrt(np.where(msk, (ea / np.where(ha > 0, ha, 1)) ** 2
                                  + (ed / np.where(hd > 0, hd, 1)) ** 2, 0))
        chi2 = float(np.nansum(((ha[msk] - hd[msk]) ** 2) / (ea[msk] ** 2 + ed[msk] ** 2)))
        ndf = int(msk.sum())
        ax.axhspan(0.95, 1.05, color="green", alpha=0.10); ax.axhspan(0.90, 1.10, color="green", alpha=0.06)
        ax.axhline(1.0, ls="--", color="0.4", lw=1)
        ax.errorbar(ctr[msk], r[msk], yerr=re[msk], fmt="o", color="C3", ms=4, capsize=2, lw=1.1)
        ax.set_ylim(0.5, 1.6); ax.set_xlabel(xlab)
        if ci == 0:
            ax.set_ylabel("ACHILLES / ADoNIS")
        ax.set_title(f"{xlab}    $\\chi^2$/ndf = {chi2/max(ndf,1):.2f}", fontsize=10)
    axes[0, ci].annotate(f"{clab}\n$\\sigma_{{ACH}}$={sig_a:.3e}  $\\sigma_{{ADO}}$={sig_d:.3e} nb"
                         f"   (ratio {sig_a/sig_d:.3f})",
                         xy=(0.5, 1.30), xycoords="axes fraction", ha="center",
                         fontsize=11, weight="bold")
fig.suptitle("CC0$\\pi$-Np ACHILLES/ADoNIS absolute ratio, by TRUE process "
             "(ACHILLES signal_process_id; ADoNIS generator truth) — no fit", fontsize=13, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "paper_figures/cc0pi_ratios_byproc.png"; fig.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out)
for clab, ach, ado in COLS:
    print(f"  {clab:14s}: sigma ACH={ach['w'].sum():.4e}  ADO={ado['w'].sum():.4e} nb  "
          f"ratio ACH/ADO={ach['w'].sum()/ado['w'].sum():.3f}")

"""ACHILLES/ADoNIS absolute (first-principles) CC0pi-Np ratios per observable, with stat errors
propagated from BOTH MC samples.  TWO rows: no-FSI and with-FSI.  FOUR columns: W, Q2, dpt, dat.

ADoNIS = CH target, taken from the self-consistent disaggregated sample (carries vertex W):
  no-FSI : QE-C only          (RES no-FSI = 0; QE-H = 0)
  FSI    : QE-C + RES-C + RES-H
ACHILLES weights are NuHepMC CV "W": absolute nb = w * GenXS[pb->nb] / sum_w_all (per run).
err(R)/R = sqrt( (sqrt(Sum w^2)/Sum w)_ACH^2 + (...)_ADO^2 )  per bin.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ORA = Path("data/oracle")
dis = np.load(ORA / "cc0pi_disaggregated.npz")

# ACHILLES absolute scales: GenXS[pb] * 1e-3 (->nb) / sum_w_all  for each run
ACH_FSI = np.load(ORA / "t2k_cc0pi_tki_achilles.npz")          # 2M FSI
ACH_NOF = np.load(ORA / "t2k_cc0pi_tki_achilles_nofsi.npz")    # 200k no-FSI
SCALE_FSI = 6.266697e-02 * 1e-3 / 8.905269e+05
SCALE_NOF = 6.25784081e-02 * 1e-3 / 8.589350e+04


def ado(fsi):
    """ADoNIS CH absolute (nb): concat the contributing disaggregated cells (weights already nb)."""
    cells = ["QE-C"] if not fsi else ["QE-C", "RES-C", "RES-H"]
    tag = "True" if fsi else "False"
    out = {}
    for k in ("W", "Q2", "dalphat", "dpt", "w"):
        out[k] = np.concatenate([dis[f"{c}_{tag}_{k}"] for c in cells]) if cells else np.array([])
    return out


ROWS = [("no FSI", ACH_NOF, SCALE_NOF, ado(False)),
        ("with FSI", ACH_FSI, SCALE_FSI, ado(True))]
VARS = [("W", np.linspace(850, 1500, 21), "vertex W [MeV]"),
        ("Q2", np.linspace(0, 1.4, 21), r"$Q^2$ [GeV$^2$]"),
        ("dpt", np.linspace(0, 800, 21), r"$\delta p_T$ [MeV]"),
        ("dalphat", np.linspace(0, np.pi, 21), r"$\delta\alpha_T$ [rad]")]


def he(v, w, bins):
    v = np.asarray(v); m = np.isfinite(v) & np.isfinite(w)
    h, _ = np.histogram(v[m], bins=bins, weights=w[m])
    h2, _ = np.histogram(v[m], bins=bins, weights=w[m] ** 2)
    return h, np.sqrt(h2)


fig, axes = plt.subplots(len(ROWS), len(VARS), figsize=(5.2 * len(VARS), 4.3 * len(ROWS)))
for ri, (rlab, ach, scale, ad) in enumerate(ROWS):
    ach_w = np.asarray(ach["w"]) * scale
    sig_a = ach_w.sum(); sig_d = ad["w"].sum()
    for ci, (key, bins, xlab) in enumerate(VARS):
        ax = axes[ri, ci]; ctr = 0.5 * (bins[1:] + bins[:-1])
        ha, ea = he(ach[key], ach_w, bins)
        hd, ed = he(ad[key], ad["w"], bins)
        msk = (ha > 0) & (hd > 0)
        r = np.where(msk, ha / np.where(hd > 0, hd, 1), np.nan)
        re = r * np.sqrt(np.where(msk, (ea / np.where(ha > 0, ha, 1)) ** 2
                                  + (ed / np.where(hd > 0, hd, 1)) ** 2, 0))
        chi2 = float(np.nansum(((ha[msk] - hd[msk]) ** 2) / (ea[msk] ** 2 + ed[msk] ** 2)))
        ndf = int(msk.sum())
        ax.axhspan(0.95, 1.05, color="green", alpha=0.10); ax.axhspan(0.90, 1.10, color="green", alpha=0.06)
        ax.axhline(1.0, ls="--", color="0.4", lw=1)
        ax.errorbar(ctr[msk], r[msk], yerr=re[msk], fmt="o", color="C3", ms=4, capsize=2, lw=1.2)
        ax.set_xlabel(xlab); ax.set_ylabel("ACHILLES / ADoNIS")
        ax.set_title(f"{rlab}  {xlab}    $\\chi^2$/ndf = {chi2/max(ndf,1):.2f}")
        ax.set_ylim(0.6, 1.5)
    axes[ri, 0].annotate(f"{rlab}: $\\sigma_{{ACH}}$={sig_a:.3e}  $\\sigma_{{ADO}}$={sig_d:.3e} nb"
                         f"  (ratio {sig_a/sig_d:.3f})",
                         xy=(0.0, 1.18), xycoords="axes fraction", fontsize=10, weight="bold")
fig.suptitle("CC0$\\pi$-Np ACHILLES/ADoNIS absolute ratio (first-principles, no fit; CH target) — "
             "stat errors from both MC samples", fontsize=13, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "paper_figures/cc0pi_ratios.png"; fig.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out)
print(f"no-FSI : sigma ACH={ (np.asarray(ACH_NOF['w'])*SCALE_NOF).sum():.4e}  ADO={ado(False)['w'].sum():.4e} nb")
print(f"FSI    : sigma ACH={ (np.asarray(ACH_FSI['w'])*SCALE_FSI).sum():.4e}  ADO={ado(True)['w'].sum():.4e} nb")

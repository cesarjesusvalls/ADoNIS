"""CC0pi-Np (T2K signal) absolute dsigma/dx: ADoNIS discrete cascade in CYLINDER vs GAUSSIAN
interaction-probability modes, both vs ACHILLES.  W, Q^2, delta_alphaT, delta_pT; top = dsigma/dx
(3 curves), bottom = ratio to ACHILLES (Cyl/ACH and Gauss/ACH; the Cyl-vs-Gauss gap is the spread
between the two ratio curves).  First-principles absolute nb, no fit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ORA = Path("data/oracle")
cyl = np.load(ORA / "cc0pi_disaggregated.npz")
gau = np.load(ORA / "cc0pi_disaggregated_gaussian.npz")
ach = np.load(ORA / "t2k_cc0pi_tki_achilles.npz")
SCALE = 6.266697e-02 * 1e-3 / 8.905269e+05
CELLS = ["QE-C", "RES-C", "RES-H"]


def ado(d):
    return {k: np.concatenate([d[f"{c}_True_{k}"] for c in CELLS]) for k in ("W", "Q2", "dalphat", "dpt", "w")}


C, G = ado(cyl), ado(gau)
ach_w = np.asarray(ach["w"]) * SCALE
VARS = [("W", np.linspace(850, 1500, 22), "vertex W [MeV]"),
        ("Q2", np.linspace(0, 1.4, 22), r"$Q^2$ [GeV$^2$]"),
        ("dalphat", np.linspace(0, np.pi, 22), r"$\delta\alpha_T$ [rad]"),
        ("dpt", np.linspace(0, 800, 22), r"$\delta p_T$ [MeV]")]


def he(v, w, bins):
    v = np.asarray(v); m = np.isfinite(v) & np.isfinite(w)
    h, _ = np.histogram(v[m], bins=bins, weights=w[m]); h2, _ = np.histogram(v[m], bins=bins, weights=w[m] ** 2)
    return h, np.sqrt(h2)


print(f"sigma_CC0pi nb:  ACH={ach_w.sum():.4e}  Cyl={C['w'].sum():.4e}  Gauss={G['w'].sum():.4e}")
print(f"  Cyl/ACH={C['w'].sum()/ach_w.sum():.3f}  Gauss/ACH={G['w'].sum()/ach_w.sum():.3f}  "
      f"Gauss/Cyl={G['w'].sum()/C['w'].sum():.3f}")

fig, axes = plt.subplots(2, len(VARS), figsize=(5.6 * len(VARS), 7), height_ratios=[3, 1], sharex="col")
for c, (key, bins, xlab) in enumerate(VARS):
    bw = np.diff(bins); ctr = 0.5 * (bins[1:] + bins[:-1])
    da, ea = he(ach[key], ach_w, bins); dc, ec = he(C[key], C["w"], bins); dg, eg = he(G[key], G["w"], bins)
    da, ea, dc, ec, dg, eg = [x / bw for x in (da, ea, dc, ec, dg, eg)]
    ax, axr = axes[0, c], axes[1, c]
    ax.step(bins, np.append(da, da[-1]), where="post", color="k", lw=1.6, label="ACHILLES")
    ax.step(bins, np.append(dc, dc[-1]), where="post", color="C0", lw=1.4, label="ADoNIS Cylinder")
    ax.step(bins, np.append(dg, dg[-1]), where="post", color="C1", lw=1.4, ls="--", label="ADoNIS Gaussian")
    ax.set_ylabel(r"d$\sigma$/dx [nb/unit]"); ax.set_ylim(bottom=0); ax.set_title(f"CC0$\\pi$-Np  {xlab}")
    if c == 0:
        ax.legend(fontsize=9)
    with np.errstate(divide="ignore", invalid="ignore"):
        rc = dc / da; rce = rc * np.sqrt((ec / dc) ** 2 + (ea / da) ** 2)
        rg = dg / da; rge = rg * np.sqrt((eg / dg) ** 2 + (ea / da) ** 2)
    m = (da > 0)
    axr.axhspan(0.95, 1.05, color="green", alpha=0.10); axr.axhline(1.0, ls="--", color="0.4", lw=1)
    axr.errorbar(ctr[m], rc[m], yerr=rce[m], fmt="o", color="C0", ms=3, capsize=2, lw=1, label="Cyl/ACH")
    axr.errorbar(ctr[m], rg[m], yerr=rge[m], fmt="s", color="C1", ms=3, capsize=2, lw=1, label="Gauss/ACH")
    axr.set_ylim(0.6, 1.4); axr.set_ylabel("ADoNIS / ACHILLES"); axr.set_xlabel(xlab)
    if c == 0:
        axr.legend(fontsize=8, ncol=2)
fig.suptitle("CC0$\\pi$-Np absolute d$\\sigma$/dx: ADoNIS Cylinder vs Gaussian vs ACHILLES   "
             f"$\\sigma$ Cyl/ACH={C['w'].sum()/ach_w.sum():.3f}  Gauss/ACH={G['w'].sum()/ach_w.sum():.3f}",
             fontsize=13)
fig.tight_layout()
out = "paper_figures/cc0pi_cyl_vs_gauss.png"; fig.savefig(out, dpi=120); print("wrote", out)

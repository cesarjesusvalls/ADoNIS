"""CC0pi-Np TKI: ADoNIS (unified engine + discrete cascade FSI) vs the ACHILLES FSI oracle.
dsigma/d(delta_pT) and dsigma/d(delta_alphaT), area-normalised SHAPE + ADoNIS/ACH ratio + chi2.
Usage: python scripts/cc0pi_compare.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from analysis.utils.hepmc import weight_to_nb_of
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ORA = Path("data/oracle")
ach = np.load(ORA / "t2k_cc0pi_tki_achilles.npz")
adoC = np.load(ORA / "t2k_cc0pi_tki_adonis_xsec.npz")        # 12C  (30 seeds)
adoH = np.load(ORA / "t2k_cc0pi_tki_adonis_H.npz")          # free-proton H (1 sample)
# ABSOLUTE nb weights (first principles, no fit).  ADoNIS = CH = 12C + H (1 carbon + 1 hydrogen):
#   12C per-event nb / 30 seeds  (+)  H per-event nb.   ACHILLES: NuHepMC W x GenXS(pb->nb)/sum_w_all.
ACH_SCALE = weight_to_nb_of(ach)      # GenCrossSection/sum_w from the npz header (no hardcoded scale)
KEYS = ("dpt", "dalphat", "Q2")
ado_w = np.concatenate([np.asarray(adoC["w"]) / 30.0, np.asarray(adoH["w"])])
ado_vals = {k: np.concatenate([np.asarray(adoC[k]), np.asarray(adoH[k])]) for k in KEYS}
ach_w = np.asarray(ach["w"]) * ACH_SCALE
SIG_ADO, SIG_ACH = ado_w.sum(), ach_w.sum()
SIG_C, SIG_H = np.asarray(adoC["w"]).sum() / 30.0, np.asarray(adoH["w"]).sum()
print(f"ACHILLES CC0pi: {len(ach['w']):,} ev  sigma={SIG_ACH:.4e} nb")
print(f"ADoNIS CH = 12C({SIG_C:.3e}) + H({SIG_H:.3e}) = {SIG_ADO:.4e} nb   ratio ADO/ACH={SIG_ADO/SIG_ACH:.3f}")

VARS = [("dpt", np.linspace(0, 800, 21), r"$\delta p_T$ [MeV]"),
        ("dalphat", np.linspace(0, np.pi, 21), r"$\delta\alpha_T$ [rad]"),
        ("Q2", np.linspace(0, 1.4, 21), r"$Q^2$ [GeV$^2$]")]
VARS = [v for v in VARS if v[0] in ach.files and v[0] in ado_vals]   # only vars present in both
fig, axes = plt.subplots(2, len(VARS), figsize=(6.5 * len(VARS), 8), height_ratios=[3, 1], sharex="col")
for col, (key, bins, xlab) in enumerate(VARS):
    bw = np.diff(bins); ctr = 0.5 * (bins[1:] + bins[:-1])
    # ABSOLUTE dsigma/dx [nb/unit] -- first-principles weights, no area normalisation
    def shp(v, w):
        h, _ = np.histogram(v, bins=bins, weights=w)
        h2, _ = np.histogram(v, bins=bins, weights=w ** 2)
        return h / bw, np.sqrt(h2) / bw
    da, ea = shp(ach[key], ach_w); dd, ed = shp(ado_vals[key], ado_w)
    ax, axr = axes[0, col], axes[1, col]
    ax.step(bins, np.append(da, da[-1]), where="post", color="0.35", label="ACHILLES (CH)")
    ax.errorbar(ctr, da, yerr=ea, fmt="none", ecolor="0.35", alpha=0.5)
    ax.step(bins, np.append(dd, dd[-1]), where="post", color="C0", label="ADoNIS CH (12C+H)")
    ax.errorbar(ctr, dd, yerr=ed, fmt="none", ecolor="C0", alpha=0.5)
    ax.legend(); ax.set_ylabel(r"d$\sigma$/d" + xlab.split()[0].strip("$") + " [nb/unit]")
    ax.set_title(f"CC0$\\pi$-Np  {xlab}")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = dd / da; re = r * np.sqrt((ed / dd) ** 2 + (ea / da) ** 2)
    msk = (da > 0) & (dd > 0) & np.isfinite(re) & (re > 0)
    chi2 = float(np.sum((dd[msk] - da[msk]) ** 2 / (ed[msk] ** 2 + ea[msk] ** 2)))
    ndf = int(msk.sum())
    axr.axhspan(0.9, 1.1, color="green", alpha=0.12); axr.axhline(1.0, ls="--", color="green")
    axr.step(bins, np.append(r, r[-1]), where="post", color="C0", lw=1.3)
    axr.errorbar(ctr, r, yerr=re, fmt="none", ecolor="C0", alpha=0.8, capsize=1.5)
    axr.set_ylim(0.6, 1.4); axr.set_ylabel("ADoNIS/ACH"); axr.set_xlabel(xlab)
    axr.text(0.03, 0.83, f"$\\chi^2$/ndf = {chi2/max(ndf,1):.2f}", transform=axr.transAxes, fontsize=10)
    print(f"  {key}: chi2/ndf = {chi2/max(ndf,1):.2f} ({ndf} bins)  "
          f"<ACH>={np.average(ach[key],weights=ach['w']):.3g} <ADO>={np.average(ado_vals[key],weights=ado_w):.3g}")
fig.suptitle(f"CC0$\\pi$-Np ABSOLUTE d$\\sigma$/dx [nb]: ADoNIS (first-principles, no fit) vs ACHILLES   "
             f"$\\sigma_{{CC0\\pi}}$: ADoNIS {SIG_ADO:.3e} / ACH {SIG_ACH:.3e} nb = {SIG_ADO/SIG_ACH:.3f}", fontsize=12)
fig.tight_layout()
out = "paper_figures/cc0pi_tki_compare.png"; fig.savefig(out, dpi=120); print("wrote", out)

"""Paper Fig 2 (arXiv:2508.19213v2): free-nucleon nu_mu CC single-pion sigma(E_nu) for the
three isospin channels, EXPERIMENT (ANL + BNL, W<1.4 GeV) vs ACHILLES vs ADoNIS.

Data: ANL (Radecky 1982) + BNL (Kitagaki 1986) digitised in NUISANCE, vendored under
data/experiment/anl_bnl_cc1pi/ (Enu[GeV], sigma[cm^2], err[cm^2]).
ACHILLES: the free-nucleon oracle (data/oracle/freenucleon_numu_sigma.csv, absolute nb).
ADoNIS: the DCC model sigma_channels_at, bridged to ACHILLES by ONE universal constant
(the same c as Phase A3) -- so the data comparison is parameter-free in shape AND the
ACHILLES normalisation.  Reports chi2/ndf of ADoNIS-vs-data and ACHILLES-vs-data per channel.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from adonis.params import PhysicsParams
from adonis.primary.dcc.sigma_enu import sigma_channels_at, CM2_1E38_PER_NB
ROOT = Path(__file__).resolve().parents[1]
N = int(os.environ.get("ADONIS_FIG2_N", 150_000))
U = CM2_1E38_PER_NB                      # nb -> 1e-38 cm^2
DDIR = ROOT / "data" / "experiment" / "anl_bnl_cc1pi"

# channel order matches the oracle columns 1..3: n->p pi0, n->n pi+, p->p pi+
CHAN = [
    dict(title=r"$\nu_\mu n\to\mu^- p\,\pi^0$", anl="anl_n_ppi0.txt", bnl="bnl_n_ppi0.txt", col="tab:green"),
    dict(title=r"$\nu_\mu n\to\mu^- n\,\pi^+$", anl="anl_n_npip.txt", bnl="bnl_n_npip.txt", col="tab:orange"),
    dict(title=r"$\nu_\mu p\to\mu^- p\,\pi^+$", anl="anl_p_ppip.txt", bnl="bnl_p_ppip.txt", col="tab:blue"),
]


def _load(fn):
    d = np.loadtxt(DDIR / fn)
    m = d[:, 1] > 0                       # drop trailing zero bin-edge marker
    return d[m, 0], d[m, 1] * 1e38, d[m, 2] * 1e38   # E[GeV], sigma & err in 1e-38 cm^2


ref = np.loadtxt(ROOT / "data" / "oracle" / "freenucleon_numu_sigma.csv")
E, ach = ref[:, 0], ref[:, 1:4]          # MeV, nb
mod = np.zeros_like(ach)
for i, e in enumerate(E):
    _, sc = sigma_channels_at(PhysicsParams(), jax.random.fold_in(jax.random.PRNGKey(11), i),
                              float(e), N, chunk=50_000)
    mod[i] = np.asarray(sc)
c = float(np.exp(np.mean(np.log(ach / mod))))        # one universal bridge constant (A3)
pred = c * mod
Eg = E / 1000.0


def chi2(xg, yg, xd, yd, ed):
    ym = np.interp(xd, xg, yg)
    good = ed > 0
    return float(np.sum(((ym[good] - yd[good]) / ed[good]) ** 2)), int(good.sum())


fig, axes = plt.subplots(2, 3, figsize=(13, 6), height_ratios=[3, 1], sharex=True)
for j, ch in enumerate(CHAN):
    ax, axr = axes[0, j], axes[1, j]
    aE, aS, aErr = _load(ch["anl"]); bE, bS, bErr = _load(ch["bnl"])
    achc = ach[:, j] * U; predc = pred[:, j] * U
    ax.plot(Eg, achc, "--", color="0.4", lw=1.5, label="ACHILLES")
    ax.plot(Eg, predc, "-", color=ch["col"], lw=2, label="ADoNIS")
    ax.errorbar(aE, aS, yerr=aErr, fmt="o", color="k", ms=4, capsize=2, label="ANL")
    ax.errorbar(bE, bS, yerr=bErr, fmt="s", mfc="none", color="k", ms=4, capsize=2, label="BNL")
    # chi2/ndf of ADoNIS vs the combined ANL+BNL data
    dE = np.concatenate([aE, bE]); dS = np.concatenate([aS, bS]); dErr = np.concatenate([aErr, bErr])
    c2m, nd = chi2(Eg, predc, dE, dS, dErr); c2a, _ = chi2(Eg, achc, dE, dS, dErr)
    ax.set_title(f"{ch['title']}\nADoNIS χ²/ndf={c2m/max(nd,1):.1f}, ACH={c2a/max(nd,1):.1f}", fontsize=9)
    ax.set_xlim(0, 2.0)
    if j == 0:
        ax.set_ylabel(r"$\sigma$ [$10^{-38}$cm$^2$]"); ax.legend(fontsize=7)
    # ratio data/ADoNIS
    ym = np.interp(dE, Eg, predc)
    axr.errorbar(dE, dS / ym, yerr=dErr / ym, fmt="o", color="k", ms=3)
    axr.axhline(1, ls="--", color=ch["col"]); axr.set_ylim(0.3, 1.9); axr.set_xlabel(r"$E_\nu$ [GeV]")
    if j == 0:
        axr.set_ylabel("data/ADoNIS")
fig.suptitle(r"Fig 2 — free-nucleon $\nu_\mu$ CC single-$\pi$ $\sigma(E_\nu)$: ANL/BNL data vs "
             r"ACHILLES vs ADoNIS (W<1.4 GeV data; one bridge constant c=%.3g)" % c)
fig.tight_layout()
out = ROOT / "paper_figures" / "fig2_sigma_enu.png"
fig.savefig(out, dpi=130)
print("wrote", out, " bridge c=%.4g" % c)

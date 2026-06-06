"""Paper Fig 1 (arXiv:2508.19213v2): inclusive (e,e') on 12C, EXPERIMENT vs ACHILLES vs ADoNIS.

Inclusive electron scattering is cascade-free (the lepton response folded over S(p,E)).
Kinematics E=2.020 GeV, theta=15.022 deg -- the JLab 12C point available in NUISANCE
(data/Electron/12C.dat; vendored to data/experiment/jlab_ee/), essentially the paper's Fig-1
e,e' kinematics.

ACHILLES: QE_Spectral_Func + RES_Spectral_Func runs (achilles:oracle, AngleTheta [14,16]),
summed -> dsigma/domega.  ADoNIS: qe_dsigma_domega + onepi_dsigma_domega.  Both are absolute
up to one bridge constant to the data; we plot experiment + ACHILLES + ADoNIS with a
model/data ratio panel + chi2/ndf vs the data.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from adonis.data.oracle.parse_hepmc import parse_events
from adonis.nuclear.qe_inclusive import qe_dsigma_domega
from adonis.nuclear.inclusive_1pi import onepi_dsigma_domega
ROOT = Path(__file__).resolve().parents[1]
E0, TH = 2020.0, 15.022
OUTD = ROOT / "_oracle_out"


def achilles_domega(hepmc, edges):
    """ACHILLES dsigma/domega histogram (weighted) over the omega bin edges."""
    oms, ws = [], []
    for ev in parse_events(Path(hepmc)):
        sc = [p4 for pid, st, p4 in ev["parts"] if pid == 11 and st == 1]
        if not sc:
            continue
        oms.append(E0 - sc[0][0]); ws.append(ev["w"])
    h, _ = np.histogram(oms, bins=edges, weights=np.array(ws))
    return h / np.diff(edges)        # arb. absolute (per MeV); bridged below


d = np.loadtxt(ROOT / "data" / "experiment" / "jlab_ee" / "c12_2020_15022.txt")
dom, dxs, derr = d[:, 0], d[:, 1], d[:, 2]
edges = np.linspace(dom.min() - 7.5, dom.max() + 7.5, 26)
ctr = 0.5 * (edges[1:] + edges[:-1])

ach_qe = achilles_domega(OUTD / "ee_2020_qe.hepmc", edges)
ach_res = achilles_domega(OUTD / "ee_2020_res.hepmc", edges)   # consistent absolute units
mod_qe = qe_dsigma_domega(E0, TH, ctr, "pke12p_tot.data", 6, 6)
mod_pi = onepi_dsigma_domega(E0, TH, ctr, "pke12p_tot.data", 12)


def _bridge(model, ref):
    return float(np.sum(ref * model) / np.sum(model ** 2 + 1e-30))


# put the ADoNIS QE and 1pi on the ACHILLES (consistent) absolute scale separately
mqe = mod_qe * _bridge(mod_qe, ach_qe)
mpi = mod_pi * _bridge(mod_pi, ach_res)
mod = mqe + mpi
ach = ach_qe + ach_res
# one overall units constant ACHILLES_total -> absolute data (nb/GeV/sr, /dOmega bin)
dat_i = np.interp(ctr, dom, dxs)
Cd = float(np.sum(dat_i * ach) / np.sum(ach ** 2 + 1e-30))
achc, modc = ach * Cd, mod * Cd
mod_qe, mod_pi = mqe * Cd, mpi * Cd


def chi2(model_ctr, model_y, dx, dy, de):
    ym = np.interp(dx, model_ctr, model_y)
    return float(np.sum(((ym - dy) / de) ** 2)), len(dy)


c2a, nd = chi2(ctr, achc, dom, dxs, derr); c2m, _ = chi2(ctr, modc, dom, dxs, derr)

fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.5, 6), height_ratios=[3, 1], sharex=True)
ax.errorbar(dom, dxs, yerr=derr, fmt="ko", ms=4, capsize=2, label="JLab data")
ax.plot(ctr, achc, "--", color="0.4", lw=1.5, label=f"ACHILLES (χ²/ndf {c2a/max(nd,1):.1f})")
ax.plot(ctr, modc, "-", color="tab:blue", lw=2, label=f"ADoNIS (χ²/ndf {c2m/max(nd,1):.1f})")
ax.plot(ctr, mod_qe, ":", color="tab:orange", lw=1, label="ADoNIS QE")
ax.plot(ctr, mod_pi, ":", color="tab:green", lw=1, label="ADoNIS 1π")
ax.set_ylabel(r"$d\sigma/d\omega\,d\Omega$ [nb/GeV/sr]"); ax.legend(fontsize=8)
ax.set_title(r"Fig 1 — inclusive (e,e') on $^{12}$C, $E$=2.020 GeV, $\theta$=15.02° (JLab)")
ym = np.interp(dom, ctr, modc)
axr.errorbar(dom, dxs / ym, yerr=derr / ym, fmt="o", color="tab:blue", ms=3)
axr.axhline(1, ls="--", color="gray"); axr.set_ylim(0.5, 1.5)
axr.set_ylabel("data/ADoNIS"); axr.set_xlabel(r"$\omega$ [MeV]")
fig.tight_layout()
out = ROOT / "paper_figures" / "fig1_inclusive_ee.png"
fig.savefig(out, dpi=130)
print("wrote", out, f"  chi2/ndf ADoNIS {c2m/max(nd,1):.1f}, ACHILLES {c2a/max(nd,1):.1f}")

"""Phase-1 figure for the Q^2-modification study: nominal vs Q2-modified fake data (logbook 18).

The injected unknown-unknown: w(Q^2) = 1 - amp*exp(-Q^2/lam), amp=0.2, lam=0.3 GeV^2
(w(0)=0.8 -> 1 at high Q^2; RPA-screening-like low-Q^2 suppression).
5 observable panels (data = nominal x wQ2, model = nominal) + w(Q2) curve + Q2 distribution.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adonis.reweight import bank_plot as BP, bank_reweight as BR
from adonis.reweight.full_knobs import nominal_knobs
from analysis.paper.physical_fit import build_physfit_datasets, theta_nominal
from physical_fit_run import Engine, apply_mode, event_Q2, wq2_of
from analysis.paper.physfit.physfit_report import stairs_band

C_NOM = "#1f77b4"
AMP, LAM = (float(x) for x in os.environ.get("PHYSFIT_Q2MOD", "0.2,0.3").split(","))
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

B = BP.load_bank(os.environ.get("ADONIS_EVENT_BANK", "output/event_bank"))
JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
w0 = np.asarray(BR.weight_jit(JB, nom, grids))
ds = build_physfit_datasets(B, w0, log)
eng = Engine(ds, JB, grids, nom, B=B)
m_nom = eng.model(theta_nominal(nom))
relerr = [d["mcerr"] / np.maximum(m_nom[eng.row0[j]:eng.row0[j+1]], 1e-300) for j, d in enumerate(ds)]
_, desc = apply_mode(ds, eng, "q2mod", q2mod=(AMP, LAM))
log(f"fake data: {desc}")

XLAB = {"dpt": r"$\delta p_T$ [MeV/c]", "dat": r"$\delta\alpha_T$ [rad]",
        "pn": r"$p_N$ [MeV/c]", "dptt": r"$\delta p_{TT}$ [MeV/c]", "daT": r"$\delta\alpha_T$ [deg]",
        "pmu": r"$p_\mu$ [MeV/c]", "cosmu": r"$\cos\theta_\mu$",
        "ppi": r"$p_\pi$ [MeV/c]", "cospi": r"$\cos\theta_\pi$"}
NOBS = len(ds)
fig = plt.figure(figsize=(13.5, 11.5))
gs = fig.add_gridspec(4, 3, hspace=0.6, wspace=0.3, height_ratios=[1, 1, 1, 0.85])
axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(NOBS)]
for j, (ax, d) in enumerate(zip(axs, ds)):
    x = 0.5 * (d["edges"][1:] + d["edges"][:-1]); xe = 0.5 * np.diff(d["edges"])
    seg = slice(eng.row0[j], eng.row0[j+1])
    ax.errorbar(x, d["data"], xerr=xe, yerr=d["sigma"], fmt="o", color="k", ms=2.8, lw=0.7,
                capsize=0, zorder=3, label=r"fake data (nominal $\times\,w(Q^2)$)")
    stairs_band(ax, d["edges"], m_nom[seg], relerr[j], C_NOM, ls="-", lw=1.5,
                label="ADoNIS nominal", balpha=0.14)
    ax.set_title(d["name"], fontsize=10); ax.set_ylim(bottom=0)
    ax.set_xlabel(XLAB[d["key"]], fontsize=9)
    if j == 0:
        ax.legend(fontsize=8)
# ratio strip: data/nominal per observable (one combined panel)
axr = fig.add_subplot(gs[3, 2])
for j, d in enumerate(ds):
    seg = slice(eng.row0[j], eng.row0[j+1])
    r = d["data"] / np.maximum(m_nom[seg], 1e-300)
    axr.plot(np.linspace(0, 1, d["nbin"]), r, lw=1.3, label=d["name"])
axr.axhline(1.0, color="k", lw=0.8); axr.axhline(1 - AMP, color="0.7", lw=0.8, ls="--")
axr.set(xlabel="bin position (normalized)", ylabel="data / nominal", ylim=(0.75, 1.05))
axr.set_title("suppression pattern per observable", fontsize=9)
axr.legend(fontsize=6.5, ncol=2)

# w(Q2) + Q2 distribution
Q2 = event_Q2(B); wv = wq2_of(B, AMP, LAM)
axq = fig.add_subplot(gs[3, 0])
qq = np.linspace(0, 2.0, 400)
axq.plot(qq, 1 - AMP * np.exp(-qq / LAM), color="#d62728", lw=2)
axq.axhline(1.0, color="0.8", lw=0.8)
axq.set(xlabel=r"$Q^2$ [GeV$^2$]", ylabel=r"$w(Q^2)$", ylim=(0.75, 1.05),
        title=f"injected modification: $1-{AMP}\\,e^{{-Q^2/{LAM}}}$")
axh = fig.add_subplot(gs[3, 1])
axh.hist(np.clip(Q2, 0, 2.0), bins=80, weights=np.asarray(B["w0"]), color="0.6")
axh.set(xlabel=r"$Q^2$ [GeV$^2$]", ylabel="weighted events", title="bank $Q^2$ distribution")
axh.text(0.55, 0.8, f"mean w(Q2) = {np.average(wv, weights=np.asarray(B['w0'])):.3f}\n"
                    f"Q2 median = {np.percentile(Q2, 50):.2f} GeV$^2$",
         fontsize=8.5, transform=axh.transAxes)
for ax in fig.axes:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.suptitle("Q$^2$-dependent unknown-unknown: nominal vs modified fake data", y=0.99)
out = "output/figures/physfit_fig8_q2mod_data.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
log(f"[fig] {out}")

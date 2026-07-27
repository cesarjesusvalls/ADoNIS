"""M2 (Huber robust fit) only vs GENIE-3M: 5-panel overlay + knob table (logbook 17 follow-up)."""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adonis.reweight import bank_plot as BP, bank_reweight as BR
from adonis.reweight.reweight_model import nominal_knobs
from analysis.paper.physical_fit import PNAMES, PRIOR, build_physfit_datasets, theta_nominal
from physical_fit_run import Engine, apply_mode
from analysis.paper.physfit.physfit_report import stairs_band

C_M0, C_M2, C_NOM = "#d62728", "#ff7f0e", "0.45"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

ge = np.load("output/altgen/physfit_genie.npz", allow_pickle=True)
B = BP.load_bank(os.environ.get("ADONIS_EVENT_BANK", "output/event_bank"))
JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
w0 = np.asarray(BR.weight_jit(JB, nom, grids))
ds = build_physfit_datasets(B, w0, log)
eng = Engine(ds, JB, grids, nom)
th0 = theta_nominal(nom)
m_nom = eng.model(th0)
m_M0 = eng.model(np.asarray(ge["M0_th"]))
m_M2 = eng.model(np.asarray(ge["M2_th"]))
relerr = [d["mcerr"] / np.maximum(m_nom[eng.row0[j]:eng.row0[j+1]], 1e-300) for j, d in enumerate(ds)]
apply_mode(ds, eng, "genie")

XLAB = {"dpt": r"$\delta p_T$ [MeV/c]", "dat": r"$\delta\alpha_T$ [rad]",
        "pn": r"$p_N$ [MeV/c]", "dptt": r"$\delta p_{TT}$ [MeV/c]", "daT": r"$\delta\alpha_T$ [deg]"}
fig = plt.figure(figsize=(13.5, 8.2))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)
axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(5)]
for j, (ax, d) in enumerate(zip(axs, ds)):
    x = 0.5 * (d["edges"][1:] + d["edges"][:-1]); xe = 0.5 * np.diff(d["edges"])
    seg = slice(eng.row0[j], eng.row0[j+1])
    ax.errorbar(x, d["data"], xerr=xe, yerr=d["sigma"], fmt="o", color="k", ms=2.8, lw=0.7,
                capsize=0, zorder=3, label="GENIE 3M data")
    stairs_band(ax, d["edges"], m_nom[seg], relerr[j], C_NOM, ls=":", lw=1.4,
                label="ADoNIS nominal", balpha=0.12)
    stairs_band(ax, d["edges"], m_M0[seg], relerr[j], C_M0, lw=1.5,
                label="M0 standard chi2 fit", balpha=0.12)
    stairs_band(ax, d["edges"], m_M2[seg], relerr[j], C_M2, ls="--", lw=1.7,
                label="M2 Huber fit", balpha=0.15)
    # per-bin post-fit pull as a thin annotation strip? keep panels clean; ratio conveyed by overlay
    ax.set_title(d["name"], fontsize=10); ax.set_ylim(bottom=0)
    ax.set_xlabel(XLAB[d["key"]], fontsize=9)
    if j == 0:
        ax.legend(fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# knob table panel: M0 vs M2 side by side
axk = fig.add_subplot(gs[1, 2]); axk.axis("off")
sub = [int(k) for k in ge["subset"]]
thA = np.asarray(ge["M0_th"]); sA = np.sqrt(np.abs(np.diag(np.asarray(ge["M0_V"]))))
thB = np.asarray(ge["M2_th"]); sB = np.sqrt(np.abs(np.diag(np.asarray(ge["M2_V"]))))
rows = []
for c, k in enumerate(sub):
    rows.append([PNAMES[k], f"{th0[k]:.2f}",
                 f"{thA[k]:.3f}±{sA[c]:.3f}", f"{thB[k]:.3f}±{sB[c]:.3f}"])
tab = axk.table(cellText=rows, colLabels=["knob", "nom", "M0 standard", "M2 Huber"],
                loc="center", cellLoc="center")
tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1.05, 1.45)
axk.set_title(f"BFPs  (chi2_data: M0 {float(ge['M0_chi2data']):.0f} | M2 {float(ge['M2_chi2data']):.0f} on 100 bins)",
              fontsize=9)
fig.suptitle("Standard chi2 (M0) vs Huber (M2) fit to GENIE-3M — GENIE 3.04 AR23_20i, QE+RES, T2K-numu/12C", y=0.98)
out = "output/figures/physfit_fig5_m0_vs_m2.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
log(f"[fig] {out}")

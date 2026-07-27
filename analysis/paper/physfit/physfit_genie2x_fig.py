"""GENIE+2x-tail overlay: data, nominal, M0/M2/M1 curves + stability table (logbook 17 follow-up)."""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adonis.reweight import bank_plot as BP, bank_reweight as BR
from adonis.reweight.full_knobs import nominal_knobs
from analysis.paper.physical_fit import PNAMES, build_physfit_datasets, theta_nominal
from physical_fit_run import Engine, apply_mode
from analysis.paper.physfit.physfit_report import stairs_band

C_M0, C_M1, C_M2, C_NOM = "#d62728", "#1f77b4", "#ff7f0e", "0.45"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

g2 = np.load("output/altgen/physfit_genie2x.npz", allow_pickle=True)
geA = np.load("output/altgen/physfit_genie.npz", allow_pickle=True)     # baseline M0/M2 (new binning)
def base_th(key):
    return np.asarray(geA[f"{key}_th"])
B = BP.load_bank(os.environ.get("ADONIS_EVENT_BANK", "output/event_bank"))
JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
w0 = np.asarray(BR.weight_jit(JB, nom, grids))
ds = build_physfit_datasets(B, w0, log)
eng = Engine(ds, JB, grids, nom)
th0 = theta_nominal(nom)
m_nom = eng.model(th0)
m = {k: eng.model(np.asarray(g2[f"{k}_th"])) for k in ("M0", "M2")}
relerr = [d["mcerr"] / np.maximum(m_nom[eng.row0[j]:eng.row0[j+1]], 1e-300) for j, d in enumerate(ds)]
apply_mode(ds, eng, "genie2x")

XLAB = {"dpt": r"$\delta p_T$ [MeV/c]", "dat": r"$\delta\alpha_T$ [rad]",
        "pn": r"$p_N$ [MeV/c]", "dptt": r"$\delta p_{TT}$ [MeV/c]", "daT": r"$\delta\alpha_T$ [deg]"}
fig = plt.figure(figsize=(13.5, 8.2))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)
axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(5)]
for j, (ax, d) in enumerate(zip(axs, ds)):
    x = 0.5 * (d["edges"][1:] + d["edges"][:-1]); xe = 0.5 * np.diff(d["edges"])
    seg = slice(eng.row0[j], eng.row0[j+1])
    if d["key"] == "dpt":
        ax.axvspan(300, d["edges"][-1], color="#f5e6c8", zorder=0)
        ax.text(0.98, 0.65, "injected ×2\nregion", transform=ax.transAxes, ha="right",
                fontsize=8, color="0.35")
    ax.errorbar(x, d["data"], xerr=xe, yerr=d["sigma"], fmt="o", color="k", ms=2.8, lw=0.7,
                capsize=0, zorder=3, label="GENIE 3M + ×2 tail")
    stairs_band(ax, d["edges"], m_nom[seg], relerr[j], C_NOM, ls=":", lw=1.4,
                label="ADoNIS nominal", balpha=0.10)
    stairs_band(ax, d["edges"], m["M0"][seg], relerr[j], C_M0, lw=1.5,
                label="M0 standard", balpha=0.10)
    stairs_band(ax, d["edges"], m["M2"][seg], relerr[j], C_M2, ls="--", lw=1.6,
                label="M2 Huber", balpha=0.10)
    ax.set_title(d["name"], fontsize=10); ax.set_ylim(bottom=0)
    ax.set_xlabel(XLAB[d["key"]], fontsize=9)
    if j == 0:
        ax.legend(fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# stability table: baseline GENIE -> GENIE+2x per method
axk = fig.add_subplot(gs[1, 2]); axk.axis("off")
sub = [int(k) for k in g2["subset"]]
rows = []
for c, k in enumerate(sub):
    if PNAMES[k] == "Eb_shift":
        continue
    r = [PNAMES[k]]
    for key in ("M0", "M2"):
        r.append(f"{float(base_th(key)[k]):.2f}"
                 + r"$\rightarrow$" + f"{float(np.asarray(g2[f'{key}_th'])[k]):.2f}")
    rows.append(r)
tab = axk.table(cellText=rows, colLabels=["knob", "M0 standard", "M2 Huber"],
                loc="center", cellLoc="center")
tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1.08, 1.5)
axk.set_title("stability: GENIE baseline → GENIE+×2 tail\n(same binning; artifact should NOT move knobs)",
              fontsize=9)
fig.suptitle("GENIE-3M + ×2 tail (δp$_T$>300 MeV): which fit survives an artifact on real foreign data?", y=0.98)
out = "output/figures/physfit_fig6_genie2x.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
log(f"[fig] {out}")

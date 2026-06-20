"""Per-event weight distributions, QE vs RES, for C and Ar (the c6h/ar6h overnight banks).
Plots the normalized distribution of w / mean(w) on a log axis; annotates N_eff/N (the quantity that
sets the statistical power).  Weights are the absolute per-event nb weights stored by the generator."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BANKS = {
    ("C", "QE"):  "data/oracle/t2k_cc0pi_engine_rich_c6h.npz",
    ("C", "RES"): "data/oracle/t2k_cc1pi_engine_rich_c6h.npz",
    ("Ar", "QE"):  "data/oracle/t2k_cc0pi_engine_rich_ar6h.npz",
    ("Ar", "RES"): "data/oracle/t2k_cc1pi_engine_rich_ar6h.npz",
}
def neffN(w): w = w[w > 0]; return (w.sum()**2 / np.sum(w**2)) / len(w)

edges = np.logspace(-4, 2, 61)       # w / mean(w)
ctr = np.sqrt(edges[:-1]*edges[1:])
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, mat in zip(axes, ("C", "Ar")):
    for ch, color in (("QE", "navy"), ("RES", "crimson")):
        b = np.load(BANKS[(mat, ch)]); w = np.asarray(b["w"]); w = w[w > 0]
        r = w / w.mean()
        h, _ = np.histogram(r, bins=edges); h = h / h.sum()
        ax.step(ctr, h, where="mid", color=color, lw=2,
                label=f"{ch}: N={len(w):,}  N_eff/N={neffN(w):.3f}")
    ax.set_xscale("log"); ax.set_xlabel("$w\\,/\\,\\langle w\\rangle$")
    ax.set_title(f"{mat}: per-event weight distribution (QE vs RES)")
    ax.axvline(1.0, ls=":", c="gray", lw=1); ax.legend(fontsize=9)
axes[0].set_ylabel("fraction of events")
plt.tight_layout(); out = "paper_figures/weight_dist_QE_RES_C_Ar.png"
plt.savefig(out, dpi=120); print("wrote", out)
# also print the tail stats
for (mat, ch), p in BANKS.items():
    w = np.asarray(np.load(p)["w"]); w = w[w > 0]
    print(f"{mat:2s} {ch:3s}: N={len(w):>8,}  N_eff/N={neffN(w):.3f}  "
          f"p99/med={np.percentile(w,99)/np.median(w):.0f}  max/med={w.max()/np.median(w):.0f}")

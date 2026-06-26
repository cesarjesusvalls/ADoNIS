"""Per-event weight distributions for the C _cv5 banks (RES = Vegas grid), QE vs RES, w/<w> on a log
axis, with N_eff/N annotated.  Overlay the resonance-only RES (_c6h) to show the Vegas tightening."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BANKS = {
    "QE (cv5)":            ("data/oracle/t2k_cc0pi_cv5.npz", "navy", "-"),
    "RES Vegas (cv5)":     ("data/oracle/t2k_cc1pi_cv5.npz", "crimson", "-"),
    "RES resonance (c6h)": ("data/oracle/t2k_cc1pi_c6h.npz", "darkorange", "--"),
}
def neffN(w): w = w[w > 0]; return (w.sum()**2 / np.sum(w**2)) / len(w)

edges = np.logspace(-4, 2, 61); ctr = np.sqrt(edges[:-1]*edges[1:])
fig, ax = plt.subplots(figsize=(8.5, 5.2))
print(f"{'bank':22s} {'N':>9s} {'N_eff/N':>8s} {'p99/med':>8s} {'max/med':>9s}")
for lab, (path, col, ls) in BANKS.items():
    w = np.asarray(np.load(path)["w"]); w = w[w > 0]
    r = w / w.mean(); h = np.histogram(r, bins=edges)[0]; h = h / h.sum()
    ax.step(ctr, h, where="mid", color=col, ls=ls, lw=2, label=f"{lab}: N_eff/N={neffN(w):.3f}")
    print(f"{lab:22s} {len(w):9d} {neffN(w):8.3f} {np.percentile(w,99)/np.median(w):8.0f} {w.max()/np.median(w):9.0f}")
ax.set_xscale("log"); ax.set_xlabel(r"$w/\langle w\rangle$"); ax.set_ylabel("fraction of events")
ax.set_title("C: per-event weight distribution — QE vs RES (Vegas vs resonance-only)")
ax.axvline(1.0, ls=":", c="gray", lw=1); ax.legend()
plt.tight_layout(); out = "paper_figures/weight_dist_QE_RES_C_cv5.png"
plt.savefig(out, dpi=120); print("wrote", out)

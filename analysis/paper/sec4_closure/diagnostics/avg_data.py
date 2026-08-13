"""What did the DATA average to, in the toys that ran down the valley?

Per bin, the mean pull (data - nominal)/sigma over a toy subgroup, with its error on the mean.  Each toy's
noise is iid N(0,1) per bin, so for a subgroup of n the expectation is 0 +- 1/sqrt(n): any bin that sits
away from zero by several of those is a bin whose fluctuation SELECTED those toys.  The core group is
drawn alongside as the null -- it should be flat at zero, and if it is not the selection is picking up
something other than the noise.
"""
import os, sys, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analysis.paper import style
style.use()

z = np.load(style.ALTGEN / "sec4_P1.npz", allow_pickle=True)
sig = np.asarray(z["sigma"]); row0 = np.asarray(z["row0"]); dsk = [str(x) for x in z["dskeys"]]
ok = np.isfinite(sig) & (sig > 0)
Z = np.load(style.ALTGEN / "sec4_P1_pulls.npz")
P = np.asarray(Z["P"]); lo = np.asarray(Z["lo"]); hi = np.asarray(Z["hi"]); core = np.asarray(Z["core"])

GR = [("$S_\\Delta<-1$", lo, "#1f4b9c", 0), ("$S_\\Delta>+1$", hi, "#c33", 0.18),
      ("core", core, "0.55", -0.18)]
nd = len(dsk); nc = 3; nr = int(np.ceil(nd / nc))
fig, ax = plt.subplots(nr, nc, figsize=(4.4 * nc, 2.1 * nr))
for i, k in enumerate(dsk):
    A = ax.flat[i]; a, b = row0[i], row0[i + 1]; m = ok[a:b]
    x = np.arange(b - a)[m]
    for lab, g, col, dx in GR:
        p = P[g, a:b][:, m]
        mu = p.mean(0); se = p.std(0, ddof=1) / np.sqrt(g.sum())
        A.errorbar(x + dx, mu, yerr=se, fmt="o", ms=2.6, lw=0, elinewidth=1.0, color=col,
                   label=f"{lab} (n={int(g.sum())})" if i == 0 else None)
    A.axhline(0, color="k", lw=0.8)
    A.set_title(k, fontsize=7.5); A.tick_params(labelsize=6)
    A.set_ylim(-1.15, 1.15); A.set_xlim(-0.8, (b - a) - 0.2)
    if i == 0: A.legend(fontsize=6, frameon=False, loc="lower left")
for j in range(nd, nr * nc): ax.flat[j].axis("off")
fig.suptitle("Mean data fluctuation per bin, by toy subgroup  (pull units; nominal = 0)",
             fontsize=12, x=0.01, ha="left")
fig.supylabel(r"$\langle(\mathrm{data}-\mathrm{nominal})/\sigma\rangle$", fontsize=9)
fig.supxlabel("bin index within sample", fontsize=9)
fig.tight_layout(rect=(0.015, 0.01, 1, 0.97))
style.save(fig, "sec4_P1_avg_data")

print(f"{'sample':>26} {'bin':>4} {'<pull> S<-1':>14} {'core':>14}")
for i, k in enumerate(dsk):
    a, b = row0[i], row0[i + 1]; m = ok[a:b]
    if not m.sum(): continue
    p = P[lo, a:b][:, m]; mu = p.mean(0); se = p.std(0, ddof=1) / np.sqrt(lo.sum())
    pc = P[core, a:b][:, m]; muc = pc.mean(0); sec = pc.std(0, ddof=1) / np.sqrt(core.sum())
    j = int(np.argmax(np.abs(mu / se)))
    if abs(mu[j] / se[j]) > 3:
        print(f"{k:>26} {int(np.arange(b-a)[m][j]):4d} {mu[j]:+7.3f}+-{se[j]:.3f} "
              f"({mu[j]/se[j]:+5.1f}s) {muc[j]:+7.3f}+-{sec[j]:.3f}")

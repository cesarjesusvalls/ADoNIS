"""Q2-nuisance verdict figure — instant, from persisted npz only (logbook 18).

Left: the fitted g(Q^2) spline (knots ± errors) vs the injected truth w(Q^2) — the unknown-unknown
reconstructed. Right: knob pulls (BFP-truth)/sigma, naive fit vs nuisance fit — closure restored."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physical_fit import NPAR

AMP, LAM = 0.2, 0.3
KNOTS = [0.02, 0.08, 0.18, 0.45, 1.2]
zn = np.load("output/altgen/p9_closq2_nuis.npz", allow_pickle=True)      # with nuisance
zb = np.load("output/altgen/p9_closure_q2mod.npz", allow_pickle=True)    # naive (biased)
pn = list(zn["pnames"])
truth = np.asarray(zn["truth"])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

# ---- left: g(Q2) reconstruction ------------------------------------------------------------------ #
qq = np.geomspace(5e-3, 3.0, 400)
a1.plot(qq, 1 - AMP * np.exp(-qq / LAM), color="#d62728", lw=2.2,
        label=r"injected truth  $w(Q^2)=1-0.2\,e^{-Q^2/0.3}$")
sub = [int(k) for k in zn["M0_sub"]]
V = np.asarray(zn["M0_V"]); th = np.asarray(zn["M0_th"])
kidx = [c for c, k in enumerate(sub) if k >= NPAR]
gval = [th[sub[c]] for c in kidx]
gerr = [np.sqrt(max(V[c, c], 0)) for c in kidx]
a1.errorbar(KNOTS, gval, yerr=gerr, fmt="o", color="#1f77b4", ms=7, capsize=4, lw=1.6,
            label="fitted nuisance g(Q$^2$) knots", zorder=3)
a1.plot(qq, np.interp(np.log(qq), np.log(KNOTS), gval), color="#1f77b4", lw=1.2, ls="--", alpha=0.6)
a1.axhline(1.0, color="0.8", lw=0.8)
a1.set(xscale="log", xlabel=r"$Q^2$ [GeV$^2$]", ylabel=r"$g(Q^2)$", ylim=(0.55, 1.25),
       title="the unknown-unknown, RECONSTRUCTED")
a1.legend(fontsize=8.5, loc="lower right")

# ---- right: knob pulls, naive vs nuisance -------------------------------------------------------- #
knobs = [k for k in [int(x) for x in zn["M0_sub"]] if k < NPAR]
yy = np.arange(len(knobs))
for z, lab, col, mk, dy in [(zb, "naive fit (no nuisance)", "#d62728", "s", -0.15),
                            (zn, "with g(Q$^2$) nuisance", "#1f77b4", "o", 0.15)]:
    subz = [int(x) for x in z["M0_sub"]]
    Vz = np.asarray(z["M0_V"]); thz = np.asarray(z["M0_th"])
    pulls, errs = [], []
    for k in knobs:
        c = subz.index(k)
        s = np.sqrt(max(Vz[c, c], 1e-24))
        pulls.append((thz[k] - truth[k]) / s)
    a2.errorbar(pulls, yy + dy, xerr=1.0, fmt=mk, color=col, ms=6, capsize=3, lw=1.4, label=lab)
a2.axvline(0, color="k", lw=1.0); a2.axvspan(-1, 1, color="0.94", zorder=0)
a2.set_yticks(yy); a2.set_yticklabels([pn[k] for k in knobs])
a2.set_xlabel(r"(BFP $-$ truth) / $\sigma$"); a2.set_xlim(-3.2, 3.2)
a2.set_title("closure restored (band = ±1σ)")
a2.legend(fontsize=8.5); a2.invert_yaxis()
for ax in (a1, a2):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.suptitle("Closure despite the Q$^2$ unknown-unknown: fit a flexible g(Q$^2$) alongside the physics knobs", y=1.0)
fig.tight_layout()
out = "output/figures/physfit_fig9_q2nuis.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"[fig] {out}")

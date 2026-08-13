"""Where do the toys with |S_Delta| < 1 sigma land in every OTHER dial?

Selecting a slice in one dial and looking at the rest is the direct way to see the valley structure:
on a degenerate direction the complement should not scatter symmetrically, it should pile up on the
partner dials in a fixed sense.  Units are the same as every other figure: (theta_fit - theta_true)/sigma_post.
"""
import os, sys, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analysis.paper import style
from adonis.analysis import knobs as K
style.use()

CUT = float(os.environ.get("SLICE_CUT", "-1.0"))
zp = np.load(style.ALTGEN / "sec4_P1_profile.npz", allow_pickle=True)
sp = np.asarray(zp["sigma_post"]); sub = [int(k) for k in zp["subset"]]
pn = [str(x) for x in zp["pnames"]]; names = [pn[k] for k in sub]
ef = [f for f in sorted(glob.glob(str(style.ALTGEN / "sec4_P1_ens_*.npz"))) if "_conv" not in f]
E = [np.load(f, allow_pickle=True) for f in ef]
th = np.concatenate([e["th_fit"] for e in E]); st = np.concatenate([e["th_star"] for e in E])
V = (th - st) / sp[None, :]                      # every dial in sigma_post about its own truth

cs = names.index("delta_strength")
sel = V[:, cs] < CUT          # ONE-SIDED, as asked: the low-S_Delta branch only
print(f"{len(V)} toys | S_Delta < {CUT}: {sel.sum()} ({100*sel.mean():.1f}%) | "
      f"rest {int((~sel).sum())}")

nd = len(names); nc = 4; nr = int(np.ceil(nd / nc))
fig, ax = plt.subplots(nr, nc, figsize=(3.9 * nc, 2.5 * nr))
print(f"\n{'dial':>20} {'ALL mean':>10} {'S<cut':>10} {'S>=cut':>10}   {'shift of complement':>20}")
for c, nm in enumerate(names):
    A = ax.flat[c]
    v = V[:, c]
    lo, hi = np.percentile(v, [0.2, 99.8]); pad = 0.08 * (hi - lo)
    b = np.linspace(lo - pad, hi + pad, 46)
    # weights:each histogram normalised by the TOTAL toy count, so bar heights are directly comparable
    # and the area under each curve is that subset's fraction of the ensemble
    w = np.full(len(v), 1.0 / len(v) / (b[1] - b[0]))
    A.hist(v, bins=b, weights=w, histtype="stepfilled", color="0.75", alpha=.8, label=f"all ({len(v)})")
    A.hist(v[sel], bins=b, weights=w[sel], histtype="step", lw=1.5, color="#1f4b9c",
           label=rf"$S_\Delta<{CUT:g}$ ({sel.sum()})")
    A.hist(v[~sel], bins=b, weights=w[~sel], histtype="step", lw=1.5, color="#c33",
           label=rf"$S_\Delta\geq{CUT:g}$ ({(~sel).sum()})")
    A.axvline(0, color="k", lw=.8, ls=":")
    A.set_title(style.plab(nm), fontsize=9)
    A.tick_params(labelsize=7); A.set_xlim(b[0], b[-1])
    if c == 0: A.legend(fontsize=6, frameon=False)
    print(f"{nm:>20} {v.mean():+10.3f} {v[sel].mean():+10.3f} {v[~sel].mean():+10.3f}   "
          f"{v[~sel].mean()-v[sel].mean():+10.3f}")
for j in range(nd, nr * nc): ax.flat[j].axis("off")
fig.suptitle(rf"Toys with $S_\Delta<{CUT:g}\sigma$ -- where they map "
             f"({len(V)} toys, P1)", fontsize=12, x=0.01, ha="left")
fig.supxlabel(r"$(\theta_{\rm fit}-\theta_{\rm true})/\sigma_{\rm post}$", fontsize=9)
fig.tight_layout(rect=(0, 0.01, 1, 0.97))
style.save(fig, f"sec4_P1_slice_sdelta_lo")

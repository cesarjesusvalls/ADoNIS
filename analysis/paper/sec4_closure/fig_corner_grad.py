"""Grad-mode corner: gradient information across the FULL PHYSICAL range of every dial.

One panel per dial pair, axes in PHYSICAL units (GeV / MeV / dimensionless), spanning each dial's whole
allowed range rather than a sigma_post window.  The claim it supports: the differentiable model returns a
usable gradient EVERYWHERE, not just near the best fit -- so a fit can be started anywhere.

FIELD selects which vector field is drawn:
  gn    -A(theta)^-1 grad chi2, the Gauss-Newton step the optimiser actually takes (DEFAULT)
  grad  raw grad chi2 restricted to the pair -- the locally steepest direction

Both are computed from a Jacobian RECOMPUTED at every node (not frozen at the best fit), which is what
makes the statement hold across the full range.  They are NOT the same field: comparing the DOWNHILL
gradient with the GN step, the measured median angle is 54 deg over the 15 views (20.6 deg for
delta_strength x sabs, 87.7 deg for Eb x kF).  So steepest descent is materially misaligned with the
step the optimiser actually takes -- A^-1 removes several decades of ill-conditioning -- but the two are
NOT opposed.  (An earlier version of this docstring claimed ~123 deg; that compared the uphill gradient
against the downhill GN step and was an artefact of the sign convention.)

There are deliberately NO confidence contours here: these are conditional surfaces, so their levels are
NOT 68/90% regions (see fig_corner_prof for those).  Colour is log10 chi2 only, as context for the arrows.

Usage:  FIELD=gn|grad python -m analysis.paper.sec4_closure.fig_corner_grad [label]
"""
import os
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

FIELD = os.environ.get("FIELD", "gn").lower()
C_BFP = "#1f4b9c"
C_ARR = {"gn": "#186", "grad": "#c33"}[FIELD]


def main(label="sec4_ref"):
    style.use()
    fs = sorted(glob.glob(str(style.ALTGEN / f"{label}_corner2d_grad_*.npz")))
    if not fs:
        raise SystemExit(f"no grad shards for {label}")
    Z = [np.load(f, allow_pickle=True) for f in fs]
    dials = [str(x) for x in Z[0]["dials"]]
    pn = [str(x) for x in Z[0]["pnames"]]
    sub = [int(k) for k in Z[0]["subset"]]
    pos = [int(p) for p in Z[0]["sel_pos"]]
    bfp = np.asarray(Z[0]["bfp"])

    # SIGN CONVENTION -- the driver stores these differently:
    #   gn_step = -A^-1 grad chi2   ALREADY the downhill step the optimiser takes
    #   grad2d  = +grad chi2        the raw gradient, which points UPHILL
    # so only grad2d gets negated to display a downhill arrow.  Negating both (the earlier bug) drew the
    # GN field pointing AWAY from the minimum and inflated the measured angle between the two fields from
    # 54 deg to 126 deg -- an artefact of the convention, not a property of the likelihood.
    key = "gn_step" if FIELD == "gn" else "grad2d"
    SGN = 1.0 if FIELD == "gn" else -1.0
    G = {}
    for z in Z:                                   # one shard per view
        for i, (a, b) in enumerate(np.asarray(z["pair_idx"])):
            G[(int(a), int(b))] = (np.asarray(z["axes_phys"])[i],
                                   np.asarray(z["dchi2"])[i], np.asarray(z[key])[i])

    nd = len(dials)
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, axes = plt.subplots(nd - 1, nd - 1, figsize=(2.25 * (nd - 1), 2.25 * (nd - 1)))
        for a in range(nd - 1):
            for b in range(nd - 1):
                A = axes[a, b]
                i, j = b, a + 1
                if j <= i or (i, j) not in G:
                    A.axis("off"); continue
                (axa, axb), d, F = G[(i, j)]
                X, Y = np.meshgrid(axa, axb, indexing="ij")
                # colour: log10 chi2 as faint context.  chi2 spans ~6 decades over a full physical range,
                # so this is orientation only -- the arrows carry the message.
                A.pcolormesh(X, Y, np.ma.masked_invalid(np.log10(np.maximum(d, 1e-3))),
                             cmap="Greys", shading="auto", rasterized=True, alpha=0.75)
                U, V = F[..., 0], F[..., 1]
                n = np.hypot(U, V); n = np.where(n > 0, n, 1.0)
                s = max(1, len(axa) // 11)                     # subsample so arrows stay legible
                A.quiver(X[::s, ::s], Y[::s, ::s], (SGN * U / n)[::s, ::s], (SGN * V / n)[::s, ::s],
                         angles="xy", scale=20, width=0.006, color=C_ARR, alpha=0.9)
                ka, kb = sub[pos[i]], sub[pos[j]]
                A.plot(bfp[ka], bfp[kb], "*", color=C_BFP, ms=13, mec="white", mew=0.7, zorder=8)
                A.set_xlim(axa[0], axa[-1]); A.set_ylim(axb[0], axb[-1])
                A.tick_params(labelsize=6, top=False, right=False)
                if a == nd - 2:
                    A.set_xlabel(style.plab(pn[ka]), fontsize=8)
                if b == 0:
                    A.set_ylabel(style.plab(pn[kb]), fontsize=8)
        lab = ("Gauss-Newton step  $-A^{-1}\\nabla\\chi^2$" if FIELD == "gn"
               else "raw gradient  $\\nabla\\chi^2$")
        h = [plt.Line2D([], [], color=C_ARR, marker=r"$\rightarrow$", ls="", ms=10),
             plt.Line2D([], [], color=C_BFP, marker="*", ls="", ms=11)]
        fig.legend(h, [f"{lab}  (downhill, unit-normalised)", "best fit"],
                   loc="upper right", fontsize=9, frameon=False, bbox_to_anchor=(0.99, 0.99))
        fig.suptitle("gradient information across the full physical range of every dial",
                     fontsize=11, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        style.save(fig, f"sec4_corner_grad_{FIELD}")


if __name__ == "__main__":
    pos_ = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos_[:1] or []))

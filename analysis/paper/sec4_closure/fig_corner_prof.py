"""Figure D -- 2-D PROFILED confidence contours.

For every dial pair: chi2 minimised over the other 14 dials at each grid node, so the levels really are
2-D confidence regions:  Dchi2 = 2.30 (68%) and 4.61 (90%).

This is the ONLY corner in the section whose contours mean that.  fig_corner_minimizer shows a CONDITIONAL
surface (others frozen), whose levels are far tighter than the marginal errors -- measured factor ~12 for
M_A_res -- and fig_corner_grad shows gradient fields with no contours at all.  Keeping the distinction is
the point: a conditional slice read as a confidence region overstates the precision badly.

Physical bounds are SHADED, not masked: the model clamps beyond them so the flat chi2 there is real.

Usage:  python -m analysis.paper.sec4_closure.fig_corner_prof [label]
"""
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from adonis.analysis import knobs as K

L68, L90 = 2.30, 4.61          # 2-D Delta-chi2 levels
C68, C90, C_BFP = "#1f4b9c", "#7aa7dd", "#d24"


def main(label="sec4_ref"):
    style.use()
    fs = sorted(glob.glob(str(style.ALTGEN / f"{label}_corner2d_prof_*.npz")))
    if not fs:
        raise SystemExit(f"no prof shards for {label}")
    Z = [np.load(f, allow_pickle=True) for f in fs]
    dials = [str(x) for x in Z[0]["dials"]]
    ax = np.asarray(Z[0]["axis_sigma"])
    pn = [str(x) for x in Z[0]["pnames"]]
    sub = [int(k) for k in Z[0]["subset"]]
    pos = [int(p) for p in Z[0]["sel_pos"]]
    spost = np.asarray(Z[0]["sigma_post"]); bfp = np.asarray(Z[0]["bfp"])

    G, part = {}, []
    for z in Z:
        ok = bool(z["complete"]) if "complete" in z.files else True
        for i, (a, b) in enumerate(np.asarray(z["pair_idx"])):
            G[(int(a), int(b))] = np.asarray(z["dchi2"])[i]
            if not ok:
                part.append((int(a), int(b)))
    if part:
        print(f"[warn] {len(part)} view(s) came from an INCOMPLETE shard (checkpoint): {part}")

    nd = len(dials)
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, axes = plt.subplots(nd - 1, nd - 1, figsize=(2.1 * (nd - 1), 2.1 * (nd - 1)),
                                 sharex="col", sharey="row")
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        for a in range(nd - 1):
            for b in range(nd - 1):
                A = axes[a, b]
                i, j = b, a + 1
                if j <= i or (i, j) not in G:
                    A.axis("off"); continue
                d = G[(i, j)]
                A.contourf(X, Y, d, levels=[0, L68, L90], colors=[C68, C90], alpha=0.75)
                A.contour(X, Y, d, levels=[L68, L90], colors=["k", "0.35"], linewidths=[1.0, 0.7])
                for which, (kk, cc) in (("x", (sub[pos[i]], pos[i])), ("y", (sub[pos[j]], pos[j]))):
                    lo = K.phys_lo(pn[kk])
                    if lo is None:
                        continue
                    xb = (lo - bfp[kk]) / spost[cc]
                    if ax[0] < xb < ax[-1]:
                        (A.axvspan if which == "x" else A.axhspan)(ax[0], xb, facecolor="0.5",
                                                                   alpha=0.30, zorder=3, lw=0)
                        (A.axvline if which == "x" else A.axhline)(xb, color="k", lw=0.9, ls="--", zorder=4)
                A.plot(0, 0, "*", color=C_BFP, ms=11, mec="white", mew=0.6, zorder=8)
                A.set_xlim(ax[0], ax[-1]); A.set_ylim(ax[0], ax[-1])
                A.tick_params(labelsize=6, top=False, right=False)
                if a == nd - 2:
                    A.set_xlabel(style.plab(pn[sub[pos[i]]]), fontsize=8)
                if b == 0:
                    A.set_ylabel(style.plab(pn[sub[pos[j]]]), fontsize=8)
        h = [plt.Rectangle((0, 0), 1, 1, fc=C68, alpha=0.75),
             plt.Rectangle((0, 0), 1, 1, fc=C90, alpha=0.75),
             plt.Line2D([], [], color=C_BFP, marker="*", ls="", ms=11),
             plt.Rectangle((0, 0), 1, 1, fc="0.5", alpha=0.30)]
        fig.legend(h, [r"68%  ($\Delta\chi^2=2.30$)", r"90%  ($\Delta\chi^2=4.61$)", "best fit",
                       "unphysical (model clamps)"],
                   loc="upper right", fontsize=8.5, frameon=False, bbox_to_anchor=(0.99, 0.99))
        fig.supxlabel(r"$(\theta-\hat\theta)/\sigma_{\rm post}$", fontsize=9)
        fig.suptitle("2-D profiled confidence regions (minimised over the other 14 dials)",
                     fontsize=11, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        style.save(fig, "sec4_corner_prof")


if __name__ == "__main__":
    p = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(p[:1] or []))

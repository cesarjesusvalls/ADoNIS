"""Figure C -- the corner the MINIMISER sees.

Per dial pair: the CONDITIONAL chi2 surface (other 14 dials held at the best fit -- the surface the
optimiser actually faces, not a profiled one), the FULL 16-D Gauss-Newton step field projected onto the
pair, and the recorded LM trajectory projected the same way.

Read this figure as "how the fit got there", NOT "what the uncertainty is" -- that is figure D.

Two honest caveats, both stated in the caption rather than hidden:
  * the step field uses A = J^T W J frozen at the best fit.  Recomputing J at every grid node would cost
    more than the surface itself, and near convergence that IS the model the fitter linearises.  It is not
    the field seen on the first step from nominal, where J differed.
  * a projected 16-D step can point "uphill" in a 2-D panel -- it is buying chi2 in the other 14
    directions.  That disagreement is the point: a 2-D corner cannot explain a 16-D fit.

Usage:  python -m analysis.paper.sec4_closure.fig_corner_minimizer [label]
"""
import sys
import glob
from pathlib import Path

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, AsinhNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from adonis.analysis import knobs as K

C_TRAJ, C_BFP = "#d24", "#1f4b9c"
SCALE = os.environ.get("CORNER_SCALE", "sqrt")     # sqrt | log
CMAP = os.environ.get("CORNER_CMAP", "Blues_r")


def main(label="sec4_ref"):
    style.use()
    fs = sorted(glob.glob(str(style.ALTGEN / f"{label}_corner2d_cond_*.npz")))
    if not fs:
        raise SystemExit(f"no cond shards for {label}")
    Z = [np.load(f, allow_pickle=True) for f in fs]
    dials = [str(x) for x in Z[0]["dials"]]
    ax = np.asarray(Z[0]["axis_sigma"])
    pn = [str(x) for x in Z[0]["pnames"]]
    sub = [int(k) for k in Z[0]["subset"]]
    spost = np.asarray(Z[0]["sigma_post"]); bfp = np.asarray(Z[0]["bfp"])
    pos = [int(p) for p in Z[0]["sel_pos"]]                       # positions of the shown dials in `subset`

    # gather all shards into {(a,b): (dchi2, gn)}
    G = {}
    for z in Z:
        for i, (a, b) in enumerate(np.asarray(z["pair_idx"])):
            G[(int(a), int(b))] = (np.asarray(z["dchi2"])[i], np.asarray(z["gn_step"])[i])

    # Trajectory.  The GRID label and the CLOSURE label can differ (a finer grid is run under its own
    # label to keep the coarse one banked), so read the trajectory from TRAJ_LABEL rather than assuming
    # the grid label owns a closure npz -- otherwise the path silently vanishes from the figure.
    traj = None
    tlab = os.environ.get("CORNER_TRAJ_LABEL", label)
    cf = style.ALTGEN / f"{tlab}.npz"
    if cf.exists():
        zc = np.load(cf, allow_pickle=True)
        if "traj_theta" in zc.files:
            traj = np.asarray(zc["traj_theta"])                   # (nstep, 28) full knob vector

    nd = len(dials)
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, axes = plt.subplots(nd - 1, nd - 1, figsize=(2.15 * (nd - 1), 2.15 * (nd - 1)),
                                 sharex="col", sharey="row")
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        for a in range(nd - 1):
            for b in range(nd - 1):
                A = plt.subplot(nd - 1, nd - 1, a * (nd - 1) + b + 1) if False else axes[a, b]
                i, j = b, a + 1                                   # column dial i, row dial j
                if j <= i or (i, j) not in G:
                    A.axis("off"); continue
                d, gn = G[(i, j)]
                # Dchi2 spans 0 -> ~4000 across a panel, so a LINEAR map saturates to one colour.
                # CORNER_SCALE picks how to compress it:
                #   sqrt -> sqrt(Dchi2) = distance in sigma; linear in sigma, good near the minimum
                #   log  -> LogNorm over Dchi2; shows structure across all four decades, at the cost of
                #           exaggerating numerically-tiny differences right at the minimum
                if SCALE == "asinh":
                    # linear below linear_width, logarithmic above: keeps the minimum localised (unlike
                    # log, where everything under ~1 sigma flattens to one colour) while still filling
                    # the panel out to Dchi2 ~ 1e3 (unlike sqrt, which leaves the tails near-white).
                    A.pcolormesh(X, Y, np.ma.masked_invalid(d), cmap=CMAP,
                                 norm=AsinhNorm(linear_width=1.0, vmin=0, vmax=1e3),
                                 shading="auto", rasterized=True)
                elif SCALE == "log":
                    A.pcolormesh(X, Y, np.ma.masked_invalid(np.maximum(d, 1e-2)), cmap=CMAP,
                                 norm=LogNorm(vmin=1e-2, vmax=1e3), shading="auto", rasterized=True)
                else:
                    A.pcolormesh(X, Y, np.ma.masked_invalid(np.sqrt(d)), cmap=CMAP, vmin=0, vmax=4.5,
                                 shading="auto", rasterized=True)
                A.contour(X, Y, d, levels=[1.0, 2.30, 4.61], colors=["0.15", "0.35", "0.55"],
                          linewidths=[1.0, 0.8, 0.8])
                # GN step field (subsampled) -- the direction the algorithm takes
                s = max(1, len(ax) // 9)
                U = gn[..., 0][::s, ::s]; Vv = gn[..., 1][::s, ::s]
                nrm = np.hypot(U, Vv); nrm[nrm == 0] = 1
                A.quiver(X[::s, ::s], Y[::s, ::s], U / nrm, Vv / nrm, angles="xy",
                         scale=22, width=0.006, color="#2a7", alpha=0.8)
                if traj is not None:                              # projected LM path, in sigma units
                    ki, kj = sub[pos[i]], sub[pos[j]]
                    tx = (traj[:, ki] - bfp[ki]) / spost[pos[i]]
                    ty = (traj[:, kj] - bfp[kj]) / spost[pos[j]]
                    R = ax[-1]
                    inside = (np.abs(tx) <= R) & (np.abs(ty) <= R)
                    A.plot(tx, ty, "--", color=C_TRAJ, lw=0.8, alpha=0.55, zorder=5)   # incoming leg
                    A.plot(tx[inside], ty[inside], "-o", color=C_TRAJ, ms=3, lw=1.4, zorder=6,
                           mec="white", mew=0.5)
                    if inside[0]:
                        A.plot(tx[0], ty[0], "s", color=C_TRAJ, ms=6, zorder=7, mec="white", mew=0.6)
                    else:
                        r0 = max(abs(tx[0]), abs(ty[0]))
                        A.text(0.03, 0.97, f"start {r0:.0f}$\\sigma$ off-panel", transform=A.transAxes,
                               fontsize=5.5, va="top", color=C_TRAJ)
                # PHYSICAL BOUNDS: shade (do not mask) the clamped region.  The chi2 there is REAL -- the
                # model clamps, so the likelihood is genuinely flat, which is the whole point of the E_b
                # wall.  Hiding it would delete the evidence; shading says "not free parameter space"
                # without pretending the values are undefined.
                for which, (kk, cc) in (("x", (sub[pos[i]], pos[i])), ("y", (sub[pos[j]], pos[j]))):
                    lo, hi = K.phys_lo(pn[kk]), K.phys_hi(pn[kk])
                    for bnd, side in ((lo, "lo"), (hi, "hi")):
                        if bnd is None:
                            continue
                        xb = (bnd - bfp[kk]) / spost[cc]           # bound position in sigma units
                        if not (ax[0] < xb < ax[-1]):
                            continue                               # bound outside the panel -> nothing to draw
                        span = (ax[0], xb) if side == "lo" else (xb, ax[-1])
                        if which == "x":
                            A.axvspan(*span, facecolor="0.5", alpha=0.30, zorder=3, lw=0)
                            A.axvline(xb, color="k", lw=1.0, ls="--", zorder=4)
                        else:
                            A.axhspan(*span, facecolor="0.5", alpha=0.30, zorder=3, lw=0)
                            A.axhline(xb, color="k", lw=1.0, ls="--", zorder=4)
                A.plot(0, 0, "*", color=C_BFP, ms=11, zorder=8, mec="white", mew=0.6)
                A.set_xlim(ax[0], ax[-1]); A.set_ylim(ax[0], ax[-1])
                A.tick_params(labelsize=6)
                if a == nd - 2:
                    A.set_xlabel(style.plab(pn[sub[pos[i]]]), fontsize=8)
                if b == 0:
                    A.set_ylabel(style.plab(pn[sub[pos[j]]]), fontsize=8)
        h = [plt.Line2D([], [], color=C_TRAJ, marker="o", ms=4, lw=1.3),
             plt.Line2D([], [], color=C_TRAJ, marker="s", ls="", ms=6),
             plt.Line2D([], [], color="#2a7", marker=r"$\rightarrow$", ls="", ms=9),
             plt.Line2D([], [], color=C_BFP, marker="*", ls="", ms=10)]
        fig.legend(h, ["LM trajectory (projected)", "start (nominal)",
                       "Gauss-Newton step, full 16-D, projected", "best fit"],
                   loc="upper right", fontsize=8, frameon=False, bbox_to_anchor=(0.99, 0.99))
        fig.supxlabel(r"$(\theta-\hat\theta)/\sigma_{\rm post}$", fontsize=9)
        fig.tight_layout()
        style.save(fig, "sec4_corner_minimizer")


if __name__ == "__main__":
    pos_ = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos_[:1] or []))

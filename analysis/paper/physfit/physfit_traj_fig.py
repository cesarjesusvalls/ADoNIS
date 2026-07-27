"""PLOT the closure-fit diagnostics computed by physfit_traj.py (decoupled: reads the npz only, so
style iterations never re-run the fit).

  fig A  <label>_params_vs_iter.png : each fitted knob vs LM/GN iteration, with the injected truth,
                                      the nominal start, and the final +-1sigma band; plus chi2 panel.
  fig B  <label>_corner.png         : corner over all fitted-knob pairs.
        lower triangle : GN chi2 landscape (Delta-chi2 of the 2D quadratic slice, others at BFP)
                         as filled contours; -grad(chi2) STREAMLINES (grey); the actual fit
                         trajectory (blue line+dots); truth (red star), start (grey square).
        diagonal       : profiled 1D Delta-chi2 parabola (from V) + truth/start/BFP markers.

Usage:  python analysis/paper/physfit/physfit_traj_fig.py [label]         (default physfit_traj_closure5)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = sys.argv[1] if len(sys.argv) > 1 else "physfit_traj_closure5"
z = np.load(f"output/altgen/{LABEL}.npz", allow_pickle=True)
P = [str(p) for p in z["pnames_sub"]]
traj = np.asarray(z["traj"]); nsub = traj.shape[1]
A = np.asarray(z["A"]); V = np.asarray(z["V"])
bfp = np.asarray(z["th_bfp"]); truth = np.asarray(z["th_truth"]); nom = np.asarray(z["th_nom"])
sig = np.sqrt(np.abs(np.diag(V)))
c_traj = np.asarray(z["chi2_traj"]); cd_traj = np.asarray(z["chi2_data_traj"])
OUT = Path("output/figures"); OUT.mkdir(parents=True, exist_ok=True)

C_TRAJ, C_TRUTH, C_NOM = "#1f77b4", "#d62728", "0.45"

# exact chi2 grids (physfit_traj.py EXACT_PAIR mode), keyed by (j, i) with j < i
PAIRS = [(a, b) for b in range(nsub) for a in range(b)]
EXACT = {}
for pidx, (j, i) in enumerate(PAIRS):
    f = Path(f"output/altgen/{LABEL}_exact_p{pidx}.npz")
    if f.exists():
        z_ = np.load(f, allow_pickle=True)
        EXACT[(j, i)] = (np.asarray(z_["xr"]), np.asarray(z_["yr"]), np.asarray(z_["chi2"]))
print(f"exact grids found: {len(EXACT)}/{len(PAIRS)}")
CHI2_MIN = float(z["chi2_min"])


# ============================== fig A: parameters vs iteration =================================== #
def fig_params():
    it = np.arange(len(traj))
    ncol = min(nsub + 1, 3); nrow = int(np.ceil((nsub + 1) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.0 * nrow), squeeze=False)
    for i in range(nsub):
        ax = axs[i // ncol, i % ncol]
        ax.axhspan(bfp[i] - sig[i], bfp[i] + sig[i], color=C_TRAJ, alpha=0.12, lw=0,
                   label=r"final $\pm1\sigma$")
        ax.axhline(truth[i], color=C_TRUTH, ls="--", lw=1.2, label="injected truth")
        ax.axhline(nom[i], color=C_NOM, ls=":", lw=1.0, label="nominal (start)")
        ax.plot(it, traj[:, i], "-o", color=C_TRAJ, ms=3.5, lw=1.2, label="LM/GN trajectory")
        ax.set_title(P[i], fontsize=10)
        ax.set_xlabel("iteration", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7, loc="best")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axc = axs[nsub // ncol, nsub % ncol]
    axc.plot(it, c_traj - c_traj[-1], "-o", color="k", ms=3.5, lw=1.2, label=r"$\chi^2-\chi^2_{\min}$")
    axc.plot(it, cd_traj - cd_traj[-1], "-s", color="0.55", ms=3, lw=1.0, label="data term")
    axc.set_yscale("symlog", linthresh=1e-2)
    axc.set_title(r"$\chi^2$ convergence", fontsize=10); axc.set_xlabel("iteration", fontsize=8)
    axc.legend(fontsize=7)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    for j in range(nsub + 1, nrow * ncol):
        axs[j // ncol, j % ncol].axis("off")
    fig.suptitle(f"closure fit trajectory — {LABEL}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / f"{LABEL}_params_vs_iter.png"
    fig.savefig(out, dpi=150); print(f"[fig] {out}")


# ============================== fig B: corner with GN landscape + streamlines ==================== #
def _pair_range(i):
    # tight window around the BFP so the 1/2/3-sigma ellipses FILL the panel; the trajectory simply
    # enters from the edge (its faraway start would otherwise shrink the contours to a dot).
    half = max(4.0 * sig[i], 1.25 * abs(truth[i] - bfp[i]))
    return bfp[i] - half, bfp[i] + half


def fig_corner():
    fig, axs = plt.subplots(nsub, nsub, figsize=(2.75 * nsub, 2.55 * nsub), squeeze=False)
    lev = [2.30, 6.18, 11.83]                            # 2D 1/2/3 sigma Delta-chi2
    for i in range(nsub):
        for j in range(nsub):
            ax = axs[i, j]
            if j > i:
                ax.axis("off"); continue
            if i == j:                                   # profiled 1D parabola
                xr = np.linspace(*_pair_range(i), 300)
                ax.plot(xr, ((xr - bfp[i]) / sig[i]) ** 2, color="k", lw=1.3)
                ax.axvline(truth[i], color=C_TRUTH, ls="--", lw=1.1)
                ax.axvline(nom[i], color=C_NOM, ls=":", lw=1.0)
                ax.axvline(bfp[i], color=C_TRAJ, lw=1.0)
                ax.set_ylim(0, 9); ax.set_ylabel(r"$\Delta\chi^2$" if j == 0 else "", fontsize=8)
            else:                                        # 2D conditional GN slice
                Ap = A[np.ix_([j, i], [j, i])]           # conditional: others clamped at BFP
                xr = np.linspace(*_pair_range(j), 60); yr = np.linspace(*_pair_range(i), 60)
                X, Y = np.meshgrid(xr, yr)
                dx, dy = X - bfp[j], Y - bfp[i]
                CH = Ap[0, 0] * dx ** 2 + 2 * Ap[0, 1] * dx * dy + Ap[1, 1] * dy ** 2
                ax.contourf(X, Y, CH, levels=[0] + lev, colors=["#9ecae1", "#c6dbef", "#deebf7"],
                            alpha=0.75)
                ax.contour(X, Y, CH, levels=lev, colors="0.35", linewidths=0.6)
                # -grad chi2 streamlines of the SAME quadratic slice
                U = -(2 * Ap[0, 0] * dx + 2 * Ap[0, 1] * dy)
                Vv = -(2 * Ap[0, 1] * dx + 2 * Ap[1, 1] * dy)
                ax.streamplot(xr, yr, U, Vv, density=0.65, color="0.62", linewidth=0.55,
                              arrowsize=0.55)
                if (j, i) in EXACT:                     # exact chi2 contours (dashed, same levels)
                    ex, ey, ec = EXACT[(j, i)]
                    ax.contour(ex, ey, ec - CHI2_MIN, levels=lev, colors="#b30000",
                               linestyles="--", linewidths=0.9, zorder=4)
                ax.plot(traj[:, j], traj[:, i], "-o", color=C_TRAJ, ms=2.6, lw=1.1, zorder=5)
                ax.plot(nom[j], nom[i], "s", color=C_NOM, ms=5, zorder=6)
                ax.plot(truth[j], truth[i], "*", color=C_TRUTH, ms=11, zorder=6)
                ax.set_xlim(*_pair_range(j)); ax.set_ylim(*_pair_range(i))
            if i == nsub - 1:
                ax.set_xlabel(P[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(P[i], fontsize=8)
            elif j > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6.5)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[
        Line2D([], [], color=C_TRAJ, marker="o", ms=4, lw=1.2, label="LM/GN trajectory"),
        Line2D([], [], color=C_TRUTH, marker="*", ms=11, lw=0, label="injected truth"),
        Line2D([], [], color=C_NOM, marker="s", ms=5, lw=0, label="nominal (start)"),
        Line2D([], [], color="0.62", lw=1.0, label=r"$-\nabla\chi^2$ streamlines"),
        Patch(facecolor="#9ecae1", edgecolor="0.35", label=r"$\Delta\chi^2 \leq 2.3$  ($1\sigma$)"),
        Patch(facecolor="#c6dbef", edgecolor="0.35", label=r"$\Delta\chi^2 \leq 6.2$  ($2\sigma$)"),
        Patch(facecolor="#deebf7", edgecolor="0.35", label=r"$\Delta\chi^2 \leq 11.8$  ($3\sigma$)"),
        Line2D([], [], color="#b30000", ls="--", lw=1.1, label=r"EXACT $\Delta\chi^2$ (full model)"),
    ], loc="upper right", fontsize=10, frameon=False, bbox_to_anchor=(0.97, 0.96))
    fig.suptitle(f"Gauss–Newton landscape + fit trajectory — {LABEL}", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT / f"{LABEL}_corner.png"
    fig.savefig(out, dpi=150); print(f"[fig] {out}")


# ============================== fig C: quadratic-fidelity matrix ================================= #
def fig_fidelity():
    """Per pair: RMS(exact - quadratic Delta-chi2) over grid points inside the 3sigma (quad) region."""
    if not EXACT:
        print("[skip] no exact grids"); return
    M = np.full((nsub, nsub), np.nan)
    for (j, i), (ex, ey, ec) in EXACT.items():
        Ap = A[np.ix_([j, i], [j, i])]
        X, Y = np.meshgrid(ex, ey)
        dx, dy = X - bfp[j], Y - bfp[i]
        quad = Ap[0, 0] * dx**2 + 2 * Ap[0, 1] * dx * dy + Ap[1, 1] * dy**2
        exact = ec - CHI2_MIN
        m = quad <= 11.83
        if m.sum() >= 8:
            M[i, j] = float(np.sqrt(np.mean((exact[m] - quad[m]) ** 2)))
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=np.nanmax(M))
    for i in range(nsub):
        for j in range(nsub):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="k" if M[i, j] < 0.7 * np.nanmax(M) else "w")
    ax.set_xticks(range(nsub)); ax.set_xticklabels(P, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(nsub)); ax.set_yticklabels(P, fontsize=8)
    fig.colorbar(im, ax=ax, label=r"RMS$(\Delta\chi^2_{\rm exact}-\Delta\chi^2_{\rm quad})$ inside $3\sigma$")
    ax.set_title(f"Gauss–Newton quadratic fidelity — {LABEL}\n(0 = exact landscape is quadratic;"
                 r" $\Delta\chi^2$ scale: $1\sigma$=2.3)", fontsize=10)
    fig.tight_layout()
    out = OUT / f"{LABEL}_quadfidelity.png"
    fig.savefig(out, dpi=150); print(f"[fig] {out}")


fig_params()
fig_corner()
fig_fidelity()

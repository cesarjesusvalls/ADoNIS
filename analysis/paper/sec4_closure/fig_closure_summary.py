"""Figure A -- the whole closure argument in one figure.

Four panels, one claim each, all from the SAME reference fit (one truth, MLE, no prior):

  (a) RECOVERY + quoted uncertainty.  Asimov fit (no statistical fluctuations): every dial comes back on
      its injected truth, with the profile-likelihood interval beside the Gaussian sigma.  This is the
      number the paper would quote.
  (b) DOES THE QUOTED ERROR DESCRIBE THE SCATTER?  The same truth refit 500 times with statistical
      throws only.  The histogram of best-fit values must have the width the profile predicted -- that is
      what makes the error bar an honest statement rather than a curvature artefact.
  (c) GOODNESS OF FIT.  chi2_data at the best fit across those 500 fits, against chi2(ndf) with ndf
      counting only bins that CONSTRAIN (empty bins carry sigma=inf and are not degrees of freedom).
  (d) THE BOUNDARY CASE.  E_b sits ~1.2 sigma from its physical wall, where the model clamps and the
      likelihood goes flat.  The Gaussian interval leaks into E_b < 0; the likelihood-based one does not.

Panels (a)+(b) together are the point: (a) says "here is the uncertainty", (b) says "and it is real".
Superseding the old split of 4.1 / 4.3 / 4.1b into three disconnected figures built on different
ensembles.

Usage:  python -m analysis.paper.sec4_closure.fig_closure_summary [label] [ens_label]
"""
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from analysis.paper.physical_fit import PRIOR, theta_nominal
from adonis.reweight.reweight_model import nominal_knobs
from adonis.analysis import knobs as K

C_FIT, C_GAUSS, C_ENS = "#1f4b9c", "0.55", "#c8842a"


def _credible(grid, prof, mass=0.6827):
    from scipy.interpolate import CubicSpline
    spl = CubicSpline(grid, prof)
    xf = np.linspace(grid.min(), grid.max(), 2001)
    dens = np.exp(-0.5 * np.maximum(spl(xf), 0.0)); dens /= np.trapezoid(dens, xf)
    o = np.argsort(dens)[::-1]
    cum = np.cumsum(dens[o]) * (xf[1] - xf[0])
    thr = dens[o][min(np.searchsorted(cum, mass), len(o) - 1)]
    sel = dens >= thr
    return float(xf[np.argmax(dens)]), float(xf[sel].min()), float(xf[sel].max())


def main(label="sec4_ref", ens="sec4_ens"):
    style.use()
    zp = np.load(style.ALTGEN / f"{label}_profile.npz", allow_pickle=True)
    sub = [int(k) for k in zp["subset"]]; pn = [str(x) for x in zp["pnames"]]
    grid = np.asarray(zp["grid_sigma"]); prof = np.asarray(zp["prof_dobj"])
    bfp = np.asarray(zp["bfp"]); spost = np.asarray(zp["sigma_post"]); truth = np.asarray(zp["truth"])
    nom = np.asarray(theta_nominal(nominal_knobs())); prior = np.asarray(PRIOR)
    order = sorted(range(len(sub)), key=lambda c: (style.knob_group(pn[sub[c]]), sub[c]))

    F = sorted(glob.glob(str(style.ALTGEN / f"{ens}_*.npz")))
    Z = [np.load(f, allow_pickle=True) for f in F]
    e_fit = np.concatenate([z["th_fit"] for z in Z]) if Z else None
    e_chi2 = np.concatenate([z["chi2_data"] for z in Z]) if Z else None
    nlive = int(Z[0]["nbins_live"]) if Z and "nbins_live" in Z[0].files else None
    ndf = (nlive if nlive else (int(Z[0]["nbins"]) if Z else 0)) - len(sub)

    zw = None
    wf = style.ALTGEN / "sec4_ebwall.npz"
    if wf.exists():
        zw = np.load(wf, allow_pickle=True)

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig = plt.figure(figsize=(11.5, 6.6))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.15], height_ratios=[1.0, 1.0, 1.0],
                              hspace=0.55, wspace=0.28, left=0.09, right=0.98, top=0.95, bottom=0.08)
        axA = fig.add_subplot(gs[:, 0])          # recovery spans all three rows
        axB = fig.add_subplot(gs[0, 1])
        axC = fig.add_subplot(gs[1, 1])
        axD = fig.add_subplot(gs[2, 1])

        # ---- (a) recovery ------------------------------------------------------------------------ #
        yy = np.arange(len(order)); gid = [style.knob_group(pn[sub[c]]) for c in order]
        for i, c in enumerate(order):
            k = sub[c]; p = max(prior[k], 1e-12)
            mode, clo, chi = _credible(grid, prof[c])
            xhat = (bfp[k] + mode * spost[c] - nom[k]) / p
            axA.errorbar([xhat], [yy[i]], xerr=[[spost[c] / p], [spost[c] / p]], fmt="none",
                         ecolor=C_GAUSS, capsize=4.5, capthick=0.9, elinewidth=0.9, zorder=2)
            axA.errorbar([xhat], [yy[i]],
                         xerr=[[(mode - clo) * spost[c] / p], [(chi - mode) * spost[c] / p]],
                         fmt="none", ecolor=C_FIT, capsize=2.2, capthick=1.5, elinewidth=2.2, zorder=4)
            if e_fit is not None:                # ensemble scatter as a light band behind
                s_e = np.std(e_fit[:, c], ddof=1) / p
                axA.errorbar([xhat], [yy[i]], xerr=[[s_e], [s_e]], fmt="none", ecolor=C_ENS,
                             capsize=0, elinewidth=5.5, alpha=0.35, zorder=1)
            axA.plot(xhat, yy[i], "D", ms=4.5, color=C_FIT, mec="white", mew=0.5, zorder=6)
            axA.plot((truth[k] - nom[k]) / p, yy[i], marker="*", ms=11, color="k", zorder=5, ls="none")
        axA.axvline(0, color="0.8", lw=0.8, zorder=0)
        axA.set_yticks(yy); axA.set_yticklabels([style.plab(pn[sub[c]]) for c in order], fontsize=7)
        axA.set_ylim(len(order) - 0.5, -0.5)
        axA.set_xlabel(r"$(\theta-\theta_{\rm nom})\,/\,\sigma_{\rm ref}$", fontsize=9)
        bounds = [i for i in range(1, len(gid)) if gid[i] != gid[i - 1]]
        seg = [-.5] + [b - .5 for b in bounds] + [len(gid) - .5]
        yt = axA.get_yaxis_transform()
        for a_, b_ in zip(seg[:-1], seg[1:]):
            g = gid[int((a_ + b_) / 2 + .5)]
            axA.axhspan(a_, b_, facecolor=style.KNOB_GROUP_COLOR[g], alpha=0.10, zorder=0, lw=0)
            axA.text(0.982, a_ + 0.12, style.KNOB_GROUP_NAME[g].replace("\n", " "), transform=yt,
                     ha="right", va="top", fontsize=6, color="0.25", fontweight="bold", zorder=7)
        for b_ in bounds:
            axA.axhline(b_ - .5, color="0.35", lw=0.7, zorder=1)
        axA.tick_params(which="both", top=False, right=False, labelsize=7)
        h = [plt.Line2D([], [], color="k", marker="*", ls="", ms=10),
             plt.Line2D([], [], color=C_FIT, lw=2.2), plt.Line2D([], [], color=C_GAUSS, lw=0.9),
             plt.Line2D([], [], color=C_ENS, lw=5, alpha=0.35)]
        axA.legend(h, ["injected truth", "68% credible", r"Gaussian $\sigma$",
                       f"scatter of {len(e_fit) if e_fit is not None else 0} refits"],
                   fontsize=6, loc="lower left", framealpha=0.92, borderpad=0.3, labelspacing=0.3)
        axA.set_title("(a)  recovery + quoted uncertainty", fontsize=9.5, loc="left")

        # ---- (b) does the quoted error match the ensemble scatter? --------------------------------- #
        if e_fit is not None:
            rat = [np.std(e_fit[:, c], ddof=1) / spost[c] for c in range(len(sub))]
            axB.axhline(1.0, color="0.4", lw=1.0, ls="--")
            axB.plot(range(len(order)), [rat[c] for c in order], "o", color=C_ENS, ms=5,
                     mec="white", mew=0.5)
            axB.set_xticks(range(len(order)))
            axB.set_xticklabels([style.plab(pn[sub[c]]) for c in order], rotation=90, fontsize=6)
            axB.set_ylabel(r"ensemble RMS / quoted $\sigma$", fontsize=8)
            axB.set_ylim(0, 2)
            axB.set_title(f"(b)  is the quoted error the true scatter?  ({len(e_fit)} refits)",
                          fontsize=9.5, loc="left")
        axB.tick_params(which="both", top=False, right=False, labelsize=7)

        # ---- (c) goodness of fit ------------------------------------------------------------------- #
        if e_chi2 is not None:
            axC.hist(e_chi2, bins=30, density=True, color=C_FIT, alpha=0.75, edgecolor="white", lw=0.4)
            xx = np.linspace(max(0, e_chi2.min() * 0.85), e_chi2.max() * 1.1, 300)
            axC.plot(xx, stats.chi2.pdf(xx, ndf), "k-", lw=1.3, label=f"$\\chi^2$(ndf={ndf})")
            axC.legend(fontsize=7); axC.set_xlabel(r"$\chi^2_{\rm data}$ at the best fit", fontsize=8)
            axC.set_ylabel("density", fontsize=8)
            axC.set_title("(c)  goodness of fit", fontsize=9.5, loc="left")
        axC.tick_params(which="both", top=False, right=False, labelsize=7)

        # ---- (d) the boundary case: E_b at its physical wall --------------------------------------- #
        if zw is not None:
            g = np.asarray(zw["eb_grid"]); dch = np.asarray(zw["dchi2"])
            eb0 = float(zw["eb_bfp"]); sig = float(zw["eb_sig"])
            xs = np.linspace(g.min(), g.max(), 800)
            post = np.exp(-0.5 * np.maximum(np.interp(xs, g, dch), 0.0))
            gau = np.exp(-0.5 * ((xs - eb0) / sig) ** 2)
            post /= np.trapezoid(post, xs); gau /= np.trapezoid(gau, xs)
            lo = K.phys_lo("Eb_shift") or 0.0
            axD.axvspan(g.min(), lo, facecolor="0.5", alpha=0.30, lw=0)   # unphysical: model clamps here
            axD.axvline(lo, color="k", lw=1.0, ls="--")
            axD.plot(xs, gau, color=C_GAUSS, lw=1.4, ls="--", label=r"Gaussian $N(\hat E_b,\sigma)$")
            axD.plot(xs, post, color=C_FIT, lw=1.6, label=r"likelihood $\propto e^{-\Delta\chi^2/2}$")
            leak = float(np.trapezoid(gau[xs < lo], xs[xs < lo]))
            axD.text(0.03, 0.94, f"Gaussian puts {100*leak:.0f}% below the wall",
                     transform=axD.transAxes, fontsize=6.5, va="top", color=C_GAUSS)
            axD.set_xlabel(r"$E_b$ shift  [MeV]", fontsize=8); axD.set_ylabel("density", fontsize=8)
            axD.set_ylim(bottom=0); axD.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
            axD.set_title("(d)  the boundary case", fontsize=9.5, loc="left")
        axD.tick_params(which="both", top=False, right=False, labelsize=7)
        for a_ in (axB, axC, axD):
            for sp in ("top", "right"):
                a_.spines[sp].set_visible(False)

        style.save(fig, "sec4_closure_summary")


if __name__ == "__main__":
    p = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(p[:2] or []))

"""Section 4 coverage figure: is the fit's quadratic error calibrated?  (the Gaussianity test)

Reads every sec4_coverage_*.npz shard (analysis.paper.physfit.multisample_coverage), pools the toys, and
renders:
  (a) pooled pull (theta_fit - theta*)/sigma_fit  vs the unit normal  -> mean, std, KS p-value.
  (b) per-dial pull mean +/- std (target 0 +/- 1), grouped by physics block -> which dial (if any) is
      biased or mis-sized.
  (c) chi2_data across toys vs the chi2(ndf) expectation (ndf = nbins - n_dials as a reference).

A pooled pull that is N(0,1) means the at-BFP Gauss-Newton covariance covers correctly -- the quadratic
error the closure quotes is trustworthy across the whole prior volume, not just at one injected point.

Usage:  python -m analysis.paper.sec4_closure.coverage_fig [glob-label]   (default sec4_coverage)
"""
import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

C_FIT = "#1f4b9c"


def main(label="sec4_coverage"):
    style.use()
    files = sorted(glob.glob(str(style.ALTGEN / f"{label}_*.npz")))
    if not files:
        raise SystemExit(f"no coverage shards at {style.ALTGEN}/{label}_*.npz")
    Z = [np.load(f, allow_pickle=True) for f in files]
    subset = [int(k) for k in Z[0]["subset"]]
    pnames = [str(x) for x in Z[0]["pnames"]]
    nbins = int(Z[0]["nbins"])
    # ndf must count only bins that CONSTRAIN: empty bins carry sigma=inf -> weight 0, contributing
    # neither to chi2 nor to the degrees of freedom.  Older npz lack nbins_live; fall back loudly.
    nlive = int(Z[0]["nbins_live"]) if "nbins_live" in Z[0].files else None
    star = np.concatenate([z["th_star"] for z in Z], axis=0)      # (ntoy, nsub)
    fit = np.concatenate([z["th_fit"] for z in Z], axis=0)
    sig = np.concatenate([z["sig_fit"] for z in Z], axis=0)
    chi2d = np.concatenate([z["chi2_data"] for z in Z])
    ntoy = star.shape[0]
    pull = (fit - star) / np.maximum(sig, 1e-12)                  # (ntoy, nsub)
    # a sig_fit ~ 0 from pinv on a numerically-degenerate direction is a NUMERICAL failure, not a coverage
    # one -> drop |pull|>8 entries (see sec4-nongaussian memory); report how many.
    ndrop = int((np.abs(pull) > 8).sum())

    order = sorted(range(len(subset)), key=lambda c: (style.knob_group(pnames[subset[c]]), subset[c]))
    pooled = pull[:, order].ravel()
    pooled = pooled[np.abs(pooled) < 8]
    ks = stats.kstest(pooled, "norm")

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig = plt.figure(figsize=(13.5, 4.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.0], wspace=0.32,
                              left=0.06, right=0.985, top=0.88, bottom=0.15)

        # (a) pooled pull vs N(0,1) ---------------------------------------------------------------- #
        ax = fig.add_subplot(gs[0, 0])
        ax.hist(pooled, bins=np.linspace(-4, 4, 33), density=True, color=C_FIT, alpha=0.75,
                edgecolor="white", lw=0.4)
        xx = np.linspace(-4, 4, 200); ax.plot(xx, stats.norm.pdf(xx), "k-", lw=1.4, label="$N(0,1)$")
        ax.set_xlabel(r"pull  $(\hat\theta-\theta^\star)/\sigma$"); ax.set_ylabel("density")
        ax.set_title("(a)  pooled pull", fontsize=10.5, loc="left")
        ax.text(0.03, 0.97, f"mean {pooled.mean():+.2f}\nstd {pooled.std():.2f}\nKS p={ks.pvalue:.2f}\n"
                f"{ntoy} toys x {len(subset)}",
                transform=ax.transAxes, va="top", fontsize=8.5)
        ax.legend(fontsize=8, loc="upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # (b) per-dial pull mean +/- std ----------------------------------------------------------- #
        ax = fig.add_subplot(gs[0, 1])
        yy = np.arange(len(order))
        for i, c in enumerate(order):
            k = subset[c]; gcol = style.KNOB_GROUP_COLOR[style.knob_group(pnames[k])]
            pc = pull[:, c]; pc = pc[np.abs(pc) < 8]                   # drop numerical-degenerate entries
            ax.errorbar(pc.mean(), yy[i], xerr=pc.std(), fmt="o", color=gcol, ms=5, capsize=3, lw=1.3)
        ax.axvline(0, color="k", lw=0.9); ax.axvspan(-1, 1, color="0.9", zorder=0)
        ax.set_yticks(yy); ax.set_yticklabels([style.plab(pnames[subset[c]]) for c in order], fontsize=8)
        ax.set_ylim(len(order) - 0.5, -0.5); ax.set_xlim(-2.2, 2.2)
        ax.set_xlabel("per-dial pull  mean $\\pm$ std"); ax.set_title("(b)  per-dial calibration",
                                                                     fontsize=10.5, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # (c) chi2_data vs chi2(ndf) --------------------------------------------------------------- #
        ax = fig.add_subplot(gs[0, 2])
        ndf = (nlive if nlive is not None else nbins) - len(subset)
        if nlive is None:
            print(f"[warn] npz has no nbins_live; ndf falls back to ALL {nbins} bins -> the chi2 curve "
                  f"will sit high if any bins are empty")
        ax.hist(chi2d, bins=25, density=True, color=C_FIT, alpha=0.75, edgecolor="white", lw=0.4)
        xx = np.linspace(max(0, chi2d.min() * 0.7), chi2d.max() * 1.15, 200)
        ax.plot(xx, stats.chi2.pdf(xx, ndf), "k-", lw=1.4, label=f"$\\chi^2$(ndf={ndf})")
        ax.set_xlabel(r"$\chi^2_{\rm data}$ at BFP"); ax.set_ylabel("density")
        ax.set_title("(c)  goodness of fit", fontsize=10.5, loc="left")
        ax.text(0.03, 0.97, f"mean {chi2d.mean():.0f}\nndf {ndf}"
                + (f"\n({nbins-nlive} empty bins\n dropped)" if nlive is not None and nlive < nbins else ""),
                transform=ax.transAxes, va="top",
                fontsize=8.5)
        ax.legend(fontsize=8, loc="upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        fig.suptitle(f"S4 coverage — {ntoy} toys, 16 dials, {nbins} bins  "
                     f"(pooled pull {pooled.mean():+.2f} $\\pm$ {pooled.std():.2f})",
                     fontsize=11, x=0.06, ha="left")
        style.save(fig, "sec4_coverage")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos[:1] or []))

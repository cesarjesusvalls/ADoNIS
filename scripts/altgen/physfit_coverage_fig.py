"""PLOT the coverage-toy ensemble (physfit_coverage.py toys in output/altgen/coverage/).

  fig 1  coverage_chi2.png  : postfit chi2 distributions (total incl. prior term, and data-only)
                              with the expected chi2 pdf overlaid + KS p-values.
                              Linear-Gaussian expectation: chi2_total ~ chi2(nbins)  [the prior acts
                              as nsub extra measurements, the fit removes nsub dof]; the data-only
                              term has E = nbins - k_eff with k_eff <= nsub (prior-regularized).
  fig 2  coverage_pulls.png : per-knob pull (th_fit - th_star)/sig_fit distributions + pooled pull
                              vs N(0,1) -- the parameter-coverage check.

Usage:  python scripts/altgen/physfit_coverage_fig.py
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

COVDIR = sys.argv[1] if len(sys.argv) > 1 else "output/altgen/coverage"
TAG = os.path.basename(COVDIR.rstrip("/"))
files = sorted(glob.glob(f"{COVDIR}/toy_*.npz"))
if not files:
    raise SystemExit(f"no toys found under {COVDIR}/")
T = [np.load(f, allow_pickle=True) for f in files]
P = [str(p) for p in T[0]["pnames_sub"]]
nsub = int(T[0]["nsub"]); nbins = int(T[0]["nbins"])
c_tot = np.array([float(t["chi2_total"]) for t in T])
c_dat = np.array([float(t["chi2_data"]) for t in T])
pulls = np.array([(t["th_fit"] - t["th_star"]) / t["sig_fit"] for t in T])   # (ntoy, nsub)
N = len(T)
print(f"{N} toys | nbins={nbins} nsub={nsub}")

# ---------------- fig 1a/1b: chi2 distributions (ONE PANEL PER FILE) ------------------------------ #
for c, ndf, lab, stem in ((c_tot, nbins, "total (data + prior)", "chi2_total"),
                          (c_dat, nbins - nsub, "data term", "chi2_data")):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    lo, hi = min(c.min(), ndf - 4 * np.sqrt(2 * ndf)), max(c.max(), ndf + 4 * np.sqrt(2 * ndf))
    bins = np.linspace(lo, hi, 96)
    ax.hist(c, bins=bins, density=True, color="#9ecae1", edgecolor="0.4", lw=0.5,
            label=f"toys (N={N})")
    xx = np.linspace(lo, hi, 400)
    ax.plot(xx, stats.chi2.pdf(xx, ndf), color="#d62728", lw=1.6, label=f"$\\chi^2$(ndf={ndf})")
    ks = stats.kstest(c, "chi2", args=(ndf,))
    ax.set_title(f"postfit $\\chi^2$ — {lab}  [{TAG}]", fontsize=10)
    ax.text(0.03, 0.93, f"mean {c.mean():.1f}  (exp {ndf})\nKS p = {ks.pvalue:.3f}",
            transform=ax.transAxes, fontsize=9, va="top")
    ax.set_xlabel(r"$\chi^2$"); ax.legend(fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = f"output/figures/coverage_{stem}.png"
    fig.savefig(out, dpi=150); print(f"[fig] {out}")

# ---------------- fig 2: pulls ------------------------------------------------------------------- #
ncol = 5; nrow = int(np.ceil((nsub + 1) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.7 * nrow), squeeze=False)
xx = np.linspace(-4, 4, 300)
for i in range(nsub):
    ax = axs[i // ncol, i % ncol]
    ax.hist(pulls[:, i], bins=np.linspace(-4, 4, 21), density=True,
            color="#9ecae1", edgecolor="0.4", lw=0.6)
    ax.plot(xx, stats.norm.pdf(xx), color="#d62728", lw=1.4)
    ax.set_title(P[i], fontsize=9)
    ax.text(0.04, 0.92, f"$\\mu$={pulls[:, i].mean():+.2f}\n$\\sigma$={pulls[:, i].std():.2f}",
            transform=ax.transAxes, fontsize=7.5, va="top")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
ax = axs[nsub // ncol, nsub % ncol]
pooled = pulls.ravel()
ax.hist(pooled, bins=np.linspace(-4, 4, 25), density=True, color="#c6dbef", edgecolor="0.4", lw=0.6)
ax.plot(xx, stats.norm.pdf(xx), color="#d62728", lw=1.4)
ks = stats.kstest(pooled, "norm")
ax.set_title("pooled", fontsize=9)
ax.text(0.04, 0.92, f"$\\mu$={pooled.mean():+.2f}  $\\sigma$={pooled.std():.2f}\nKS p={ks.pvalue:.3f}",
        transform=ax.transAxes, fontsize=7.5, va="top")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for j in range(nsub + 1, nrow * ncol):
    axs[j // ncol, j % ncol].axis("off")
fig.suptitle(r"coverage toys: pulls $(\theta_{\rm fit}-\theta^{*})/\sigma_{\rm fit}$ vs $\mathcal{N}(0,1)$",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("output/figures/coverage_pulls.png", dpi=150)
print("[fig] output/figures/coverage_pulls.png")

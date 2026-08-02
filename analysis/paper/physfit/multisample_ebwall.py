"""Sec 4 non-Gaussian headline: the E_b hard wall.

E_b >= 0 is physical (the model flattens for E_b<0: sf_reweight clamps it), so a posterior that sits near
the wall is TRUNCATED, not Gaussian.  Inject a small E_b (~0.5 MeV) so the MLE posterior bumps the wall,
then scan E_b finely -- INCLUDING below zero -- profiling the other 15 dials, to expose the one-sided
profile likelihood.  The figure then contrasts:
  * Gaussian sigma  -> a symmetric interval that leaks into UNPHYSICAL E_b < 0,
  * profile Dchi2=1 -> a one-sided interval that stops at the wall.

Caches the 1-D profile curve (offline figure).  MLE (no prior) so it is the pure data likelihood.

Env: PHYSFIT_INJECT (default "Eb_shift=0.5")  EB_LO/EB_HI/EB_N (scan, default -1.0 / 2.5 / 41)
     + the S4_*_CHUNKS bank caps.  Writes output/altgen/sec4_ebwall.npz.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper.physfit.multisample import build_multisample_engine, MULTISAMPLE_NPZ
from analysis.paper.physfit.physical_fit_run import lm_fit, parse_inject
from analysis.paper.physical_fit import PNAMES, theta_nominal
from adonis.reweight.reweight_model import nominal_knobs


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    INJECT = os.environ.get("PHYSFIT_INJECT", "Eb_shift=0.5")
    EB_LO = float(os.environ.get("EB_LO", "-1.0"))
    EB_HI = float(os.environ.get("EB_HI", "2.5"))
    EB_N = int(os.environ.get("EB_N", "41"))

    eng = build_multisample_engine(log)
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    subset = [int(i) for i in np.where(g["shrink"] < 0.5)[0]]
    eb_k = PNAMES.index("Eb_shift"); eb_pos = subset.index(eb_k)
    nom = np.asarray(theta_nominal(nominal_knobs()))

    truth, _ = parse_inject(INJECT, nominal_knobs())              # small Eb injected, others nominal
    eng.set_closure_data(truth)
    data = np.concatenate([d["data"] for d in eng.ds]); sigma = np.concatenate([d["sigma"] for d in eng.ds])
    W = np.where(np.isfinite(sigma) & (sigma > 0), 1.0 / sigma**2, 0.0)
    def chi2d(th):
        m = eng.model(th); return float(np.sum(((m - data) / sigma)**2))

    bfp = truth.copy()                                           # MLE minimum == injected truth
    J = eng.jac(bfp, subset); A = J.T @ (J * W[:, None]); V = np.linalg.pinv(A, rcond=1e-12)
    eb_sig = float(np.sqrt(abs(V[eb_pos, eb_pos])))              # Gaussian (data-only) sigma on Eb
    chi2_min = chi2d(bfp)
    log(f"Eb: bfp={bfp[eb_k]:.3f} MeV, Gaussian sigma={eb_sig:.3f} MeV, wall at 0; scan [{EB_LO},{EB_HI}]")

    free = [k for k in subset if k != eb_k]
    grid = np.linspace(EB_LO, EB_HI, EB_N)
    dchi2 = np.zeros(EB_N)
    for i, val in enumerate(grid):
        th_init = bfp.copy(); th_init[eb_k] = val                # pinned (model clamps val<0 internally -> flat)
        th, _V, _J, _m, _c, _cd = lm_fit(eng, free, f"eb{val:+.2f}", nit=40, th_init=th_init)
        th[eb_k] = val
        dchi2[i] = chi2d(th) - chi2_min
        if (i + 1) % 10 == 0:
            log(f"  {i+1}/{EB_N}  Eb={val:+.2f} Dchi2={dchi2[i]:.2f}")

    out = "output/altgen/sec4_ebwall.npz"
    np.savez(out, eb_grid=grid, dchi2=dchi2, eb_bfp=bfp[eb_k], eb_sig=eb_sig, eb_nom=nom[eb_k],
             chi2_min=chi2_min, inject=INJECT)
    log(f"[out] {out}")


if __name__ == "__main__":
    main()

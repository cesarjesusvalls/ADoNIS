"""Sec 5.4 hybrid (grid-bad x Taylor-good) ingredients: (E_b, X) 2-D likelihood near the E_b wall.

Taylor (a smooth polynomial from J + d2m) CANNOT represent the E_b>=0 wall -- it is analytic, so it extends
symmetrically below the boundary.  The hybrid GRIDS the one bad dial (E_b, exact model, respecting the
wall) and keeps TAYLOR for the good dial X (+ analytic nuisance profile).  This runner caches everything a
3-way comparison (pure Taylor / hybrid / exact) needs, at a near-wall injection (E_b* ~ 0.15 MeV):
  Jb, Bb          model 1st/2nd derivatives at the fit  (for the Taylor / hybrid good-dim part)
  W               1/sigma^2
  eb_1d           exact 1-D E_b profile (the walled backbone the hybrid uses)
  grid2d[X]       exact 2-D (E_b, X) profiled Delta-chi2 (ground truth) for a few good dials X
  bfp, sigma_post, subset, pnames, eb_axis, x_axis

Env: PHYSFIT_INJECT (default Eb_shift=0.15)  S4_HYBRID_X (comma dials, default kF_sf,M_A_res)  + S4_*_CHUNKS.
Writes output/altgen/sec5_hybrid.npz.
"""
import os
import sys
import time
import itertools
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper.physfit.multisample import build_multisample_engine, MULTISAMPLE_NPZ
from analysis.paper.physfit.physical_fit_run import lm_fit, parse_inject
from analysis.paper.physical_fit import PNAMES, NPAR
from adonis.reweight.reweight_model import nominal_knobs


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    INJECT = os.environ.get("PHYSFIT_INJECT", "Eb_shift=0.15")
    XNAMES = os.environ.get("S4_HYBRID_X", "kF_sf,M_A_res").split(",")

    eng = build_multisample_engine(log)
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    subset = [int(i) for i in np.where(g["shrink"] < 0.5)[0]]
    eb_k = PNAMES.index("Eb_shift"); eb_pos = subset.index(eb_k)
    truth, _ = parse_inject(INJECT, nominal_knobs())
    eng.set_closure_data(truth)
    data = np.concatenate([d["data"] for d in eng.ds]); sigma = np.concatenate([d["sigma"] for d in eng.ds])
    W = np.where(np.isfinite(sigma) & (sigma > 0), 1.0 / sigma**2, 0.0)
    def chi2d(th):
        m = eng.model(th); return float(np.sum(((m - data) / sigma)**2))
    bfp = truth.copy(); chi2_min = chi2d(bfp)

    # ---- derivatives at the fit (for pure-Taylor + hybrid good-dim part) --------------------------- #
    def evec(idxs):
        v = np.zeros(NPAR)
        for i in idxs:
            v[subset[i]] = 1.0
        return v
    log("J + d2m at the near-wall fit")
    Jb = np.column_stack([eng.dd(bfp, evec([i]), 1) for i in range(len(subset))])
    Bii = [eng.dd(bfp, evec([i]), 2) for i in range(len(subset))]
    Bb = np.zeros((len(sigma), len(subset), len(subset)))
    for i in range(len(subset)):
        Bb[:, i, i] = Bii[i]
    for i, j in itertools.combinations(range(len(subset)), 2):
        d = 0.5 * (eng.dd(bfp, evec([i, j]), 2) - Bii[i] - Bii[j])
        Bb[:, i, j] = d; Bb[:, j, i] = d
    A = Jb.T @ (Jb * W[:, None]); V = np.linalg.pinv(A, rcond=1e-12); spost = np.sqrt(np.abs(np.diag(V)))

    # ---- exact 1-D E_b profile (the walled backbone) ---------------------------------------------- #
    free_eb = [k for k in subset if k != eb_k]
    eb_axis = np.linspace(-0.6, bfp[eb_k] + 3 * spost[eb_pos], 21)
    eb_1d = np.zeros(len(eb_axis))
    for i, v in enumerate(eb_axis):
        ti = bfp.copy(); ti[eb_k] = v
        eb_1d[i] = lm_fit(eng, free_eb, "eb", nit=40, th_init=ti)[5] - chi2_min
    log(f"exact 1-D Eb profile done ({len(eb_axis)} nodes)")

    # ---- exact 2-D (E_b, X) profiled grids (ground truth) ----------------------------------------- #
    grids = {}; xax = {}
    for xn in XNAMES:
        xk = PNAMES.index(xn); xpos = subset.index(xk)
        free2 = [k for k in subset if k not in (eb_k, xk)]
        xa = np.linspace(bfp[xk] - 3 * spost[xpos], bfp[xk] + 3 * spost[xpos], 15)
        Z = np.zeros((len(eb_axis), len(xa)))
        for ie, ve in enumerate(eb_axis):
            for ix, vx in enumerate(xa):
                ti = bfp.copy(); ti[eb_k] = ve; ti[xk] = vx
                Z[ie, ix] = lm_fit(eng, free2, "2d", nit=40, th_init=ti)[5] - chi2_min
        grids[xn] = Z; xax[xn] = xa
        log(f"exact 2-D (Eb,{xn}) grid done")

    out = "output/altgen/sec5_hybrid.npz"
    np.savez(out, subset=subset, pnames=list(PNAMES), eb_pos=eb_pos, bfp=bfp, V=V, W=W, Jb=Jb, Bb=Bb,
             sigma_post=spost, eb_axis=eb_axis, eb_1d=eb_1d, xnames=XNAMES,
             **{f"grid_{xn}": grids[xn] for xn in XNAMES}, **{f"xax_{xn}": xax[xn] for xn in XNAMES})
    log(f"[out] {out}")


if __name__ == "__main__":
    main()

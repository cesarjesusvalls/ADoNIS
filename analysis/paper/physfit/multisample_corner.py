"""S4D corner: exact (non-Gaussian) 2D contours of the 16 dials vs the Gaussian covariance ellipses.

Naive importance sampling degenerates in 16-D (ESS ~1%), and grid-profiling 120 pairs (re-minimising 14
nuisances at every node) is intractable.  Instead, for each pair (a,b) we scan the 2D plane and set the
other 14 dials to their GAUSSIAN-CONDITIONAL MEAN given (a,b) -- the linear shift  V_{o,ab} V_{ab,ab}^-1
(x_ab - bfp_ab)  -- then evaluate the EXACT chi2.  This is the exact profile to leading (linear) order in
the nuisances, so it captures BOTH the parameter correlations (via the conditional shift) and the genuine
non-Gaussian curvature of the (a,b) plane (via the exact chi2), at one model eval per node (no inner fit).

Reads the MLE closure npz (bfp = injected truth) for the truth; recomputes V = (J^T W J)^-1 at the bank cap
used here so the exact grid and the Gaussian ellipse are internally consistent.  Caches the per-pair
Delta-chi2 grids; the corner (Gaussian ellipse from V vs exact contour) renders offline.

Env: ADONIS_LABEL (default sec4_closure_r16_noprior)  S4_CORNER_N (per-axis nodes, default 19)
     S4_CORNER_RANGE (half-width in sigma, default 3.5)  + the S4_*_CHUNKS bank caps.
Writes output/altgen/<label>_corner.npz.
"""
import os
import sys
import time
import itertools
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper.physfit.multisample import build_multisample_engine


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    LABEL = os.environ.get("ADONIS_LABEL", "sec4_closure_r16_noprior")
    N = int(os.environ.get("S4_CORNER_N", "19"))
    RANGE = float(os.environ.get("S4_CORNER_RANGE", "3.5"))

    z = np.load(f"output/altgen/{LABEL}.npz", allow_pickle=True)
    sub = [int(k) for k in z["subset"]]; pn = [str(x) for x in z["pnames"]]
    truth = np.asarray(z["truth"]); ndim = len(sub)

    eng = build_multisample_engine(log)
    eng.set_closure_data(truth)                                   # noiseless closure: chi2 minimum at truth
    data = np.concatenate([d["data"] for d in eng.ds]); sigma = np.concatenate([d["sigma"] for d in eng.ds])
    def chi2d(th):
        m = eng.model(th); return float(np.sum(((m - data) / sigma)**2))

    bfp = truth.copy()                                            # MLE minimum == injected truth (chi2=0)
    J = eng.jac(bfp, sub); W = 1.0 / sigma**2
    A = J.T @ (J * W[:, None]); V = np.linalg.pinv(A, rcond=1e-12)  # data-only (MLE) covariance
    spost = np.sqrt(np.abs(np.diag(V)))
    log(f"corner grid: {ndim} dials, {N}x{N} nodes, +/-{RANGE} sigma, chi2@bfp={chi2d(bfp):.2e}")

    ax = np.linspace(-RANGE, RANGE, N)                            # per-axis nodes in sigma_post units
    pairs = list(itertools.combinations(range(ndim), 2))         # 120 unordered pairs
    dchi2 = np.full((len(pairs), N, N), np.nan)                  # exact Delta-chi2 per pair grid
    for pi, (a, b) in enumerate(pairs):
        ka, kb = sub[a], sub[b]
        others = [i for i in range(ndim) if i not in (a, b)]
        ko = [sub[i] for i in others]
        Vpp = V[np.ix_([a, b], [a, b])]; Vop = V[np.ix_(others, [a, b])]
        M = Vop @ np.linalg.pinv(Vpp, rcond=1e-12)               # (14,2) conditional-mean coefficients
        for ia, ta in enumerate(ax):
            for ib, tb in enumerate(ax):
                dpair = np.array([ta * spost[a], tb * spost[b]])
                th = bfp.copy(); th[ka] = bfp[ka] + dpair[0]; th[kb] = bfp[kb] + dpair[1]
                dsh = M @ dpair
                for j, ki in enumerate(ko):
                    th[ki] = bfp[ki] + dsh[j]
                    if pn[ki] == "Eb_shift":
                        th[ki] = max(th[ki], 1e-2)
                if pn[ka] == "Eb_shift": th[ka] = max(th[ka], 1e-2)
                if pn[kb] == "Eb_shift": th[kb] = max(th[kb], 1e-2)
                dchi2[pi, ia, ib] = chi2d(th)
        if (pi + 1) % 20 == 0:
            log(f"  {pi+1}/{len(pairs)} pairs")

    out = f"output/altgen/{LABEL}_corner.npz"
    np.savez(out, subset=sub, pnames=pn, bfp=bfp, truth=truth, V=V, A=A, sigma_post=spost,
             axis_sigma=ax, pairs=np.array(pairs), dchi2=dchi2, N=N, range=RANGE)
    log(f"[out] {out}  ({len(pairs)} pairs, {N}x{N})")


if __name__ == "__main__":
    main()

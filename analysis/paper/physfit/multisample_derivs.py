"""S4D (higher-order): the model's 1st and 2nd derivatives at the best-fit -- the ingredients for an
ANALYTIC non-Gaussian corner, computed by autodiff of the differentiable generator (no bank grid).

Around the MLE closure minimum theta_hat the residual is zero, so
    chi2(theta_hat + D) = sum_b W_b ( (J.D)_b + 1/2 (D.d2m.D)_b + ... )^2 ,   W_b = 1/sigma_b^2 .
Truncating the MODEL at 2nd order (m ~ m0 + J.D + 1/2 D.d2m.D) and squaring gives a chi2 that is Gaussian
at leading order (D^T (J^T W J) D) with the leading NON-Gaussianity carried by d2m -- and is >= 0 by
construction (a sum of squares), so it never diverges in the tails the way a raw Taylor series would.

This script caches the two tensors needed, computed ONCE at theta_hat:
    Jb  (nbin, nsub)          binned 1st directional derivatives  (J . e_i)
    Bb  (nbin, nsub, nsub)    binned 2nd directional derivatives  (e_i . d2m . e_j), via polarization
    W   (nbin,)               1/sigma^2  (empty/inf-sigma bins -> 0)
d2m is obtained by NESTED jvp (jvp-of-jvp) -- the same jvp rules that already give J, so no jax.jet and no
extra primitive support is needed.  The corner (Gaussian ellipse from J^T W J vs the exact 2nd-order-model
chi2, profiled over the other 14 dials as a cheap polynomial) then renders OFFLINE from this npz.

Env: ADONIS_LABEL (default sec4_closure_r16_noprior) + the S4_*_CHUNKS bank caps.
Writes output/altgen/<label>_derivs.npz.
"""
import os
import sys
import time
import itertools
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper.physfit.multisample import build_multisample_engine
from analysis.paper.physical_fit import NPAR


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    LABEL = os.environ.get("ADONIS_LABEL", "sec4_closure_r16_noprior")
    z = np.load(f"output/altgen/{LABEL}.npz", allow_pickle=True)
    sub = [int(k) for k in z["subset"]]; pn = [str(x) for x in z["pnames"]]
    truth = np.asarray(z["truth"]); nsub = len(sub)

    eng = build_multisample_engine(log)
    eng.set_closure_data(truth)
    sigma = np.concatenate([d["sigma"] for d in eng.ds])
    W = np.where(np.isfinite(sigma) & (sigma > 0), 1.0 / sigma**2, 0.0)
    bfp = truth.copy()                                            # MLE minimum == injected truth
    nbin = len(sigma)

    def evec(idxs):                                               # unit tangent(s) in the 28-dim theta space
        v = np.zeros(NPAR)
        for i in idxs:
            v[sub[i]] = 1.0
        return v

    # ---- J : binned 1st directional derivative along each dial ------------------------------------- #
    log(f"J: {nsub} directional 1st derivatives at the BFP")
    Jb = np.column_stack([eng.dd(bfp, evec([i]), 1) for i in range(nsub)])   # (nbin, nsub)

    # ---- d2m : binned 2nd directional derivatives, assembled by polarization ----------------------- #
    log(f"d2m: {nsub} diagonal + {nsub*(nsub-1)//2} off-diagonal 2nd derivatives (nested jvp)")
    Bii = [eng.dd(bfp, evec([i]), 2) for i in range(nsub)]                   # (nbin,) each  = B_ii
    Bb = np.zeros((nbin, nsub, nsub))
    for i in range(nsub):
        Bb[:, i, i] = Bii[i]
    for c, (i, j) in enumerate(itertools.combinations(range(nsub), 2)):
        dij = eng.dd(bfp, evec([i, j]), 2)                        # = B_ii + 2 B_ij + B_jj
        Bij = 0.5 * (dij - Bii[i] - Bii[j])
        Bb[:, i, j] = Bij; Bb[:, j, i] = Bij
        if (c + 1) % 30 == 0:
            log(f"  {c+1}/{nsub*(nsub-1)//2} off-diagonal pairs")

    A = Jb.T @ (Jb * W[:, None])                                  # Gauss-Newton Hessian (= Gaussian curvature)

    # ---- d3m : binned MIXED 3rd derivatives (order-3 model -> chi2 exact to 4th order) -------------- #
    Cb = None
    ORDER = int(os.environ.get("S4_ORDER", "2"))
    if ORDER >= 3:
        ntri = nsub * (nsub + 1) * (nsub + 2) // 6
        log(f"d3m: {ntri} mixed 3rd derivatives (triple-nested jvp)")
        Cb = np.zeros((nbin, nsub, nsub, nsub))
        for c, (i, j, k) in enumerate(itertools.combinations_with_replacement(range(nsub), 3)):
            g = eng.dd3(bfp, evec([i]), evec([j]), evec([k]))     # = C_ijk (fully symmetric)
            for p in set(itertools.permutations((i, j, k))):      # symmetrise into all slots
                Cb[:, p[0], p[1], p[2]] = g
            if (c + 1) % 100 == 0:
                log(f"  {c+1}/{ntri} triples")

    out = f"output/altgen/{LABEL}_derivs.npz"
    kw = dict(subset=sub, pnames=pn, bfp=bfp, truth=truth, sigma=sigma, W=W,
              Jb=Jb, Bb=Bb, A=A, V=np.linalg.pinv(A, rcond=1e-12))
    if Cb is not None:
        kw["Cb"] = Cb
    np.savez(out, **kw)
    log(f"[out] {out}  (Jb {Jb.shape}, Bb {Bb.shape}" + (f", Cb {Cb.shape}" if Cb is not None else "") + ")")


if __name__ == "__main__":
    main()

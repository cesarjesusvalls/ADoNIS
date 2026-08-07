"""2-D corner ingredients for a SELECTED set of dials, in two modes.

MODE=cond  (figure C -- "what the minimiser sees")
    chi2(theta_i, theta_j) with the other 14 dials HELD at the best fit: the conditional surface the
    optimiser actually faces, not a profiled one.  Also stores the FULL 16-D Gauss-Newton step at every
    node, projected onto (i,j) -- gn = -A^-1 grad, A = J^T W J and grad = 2 J^T W r.  That is the direction
    the algorithm takes, so it lines up with the recorded trajectory even where it looks "uphill" in 2D
    (the step is buying chi2 in the other 14 directions).  A is held at the BFP: recomputing J at every
    node would cost more than the surface itself, and near convergence that IS the model the fitter uses.

MODE=prof  (figure D -- inference)
    chi2 PROFILED over the other 14 at every node -> honest 68/90% 2-D contours (Dchi2 = 2.30 / 4.61).
    Warm-started along a snake path so each node starts from its neighbour's solution (1-2 LM iterations
    instead of ~10).  Shard with S4_PAIR_BASE / S4_NPAIR across a SLURM array; one npz per shard.

Both modes take the BFP from PHYSFIT_INJECT: for an Asimov closure the minimum sits exactly on the
injected truth (verified chi2@bfp ~ 1e-28), so neither needs the closure npz and both can run in parallel
with it.

Env: S4_CORNER_DIALS (comma list, default the 6 agreed), S4_CORNER_N (grid per axis, default 17),
     S4_CORNER_RANGE (+- sigma_post, default 3.0), MODE (cond|prof), S4_PAIR_BASE/S4_NPAIR,
     PHYSFIT_INJECT, ALTGEN_NIT (inner LM iterations for prof, default 3) + the S4_*_CHUNKS caps.
Writes output/altgen/<label>_corner2d_<mode>_<pairbase>.npz
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
from analysis.paper.physical_fit import PNAMES
from adonis.reweight.reweight_model import nominal_knobs
from adonis.analysis import knobs as K

DEFAULT_DIALS = "M_A_res,delta_strength,Eb_shift,sabs,f_NN_cex,kF_sf"

# PHYSICAL ranges for MODE=grad (the "we have gradient information everywhere" figure).  Not sigma_post
# windows: the claim is about the whole physically allowed region, so the grid must span it.
PHYS_RANGE = {"M_A_res": (0.05, 2.0), "delta_strength": (0.05, 2.0), "Eb_shift": (0.01, 5.0),
              "sabs": (0.05, 2.0), "f_NN_cex": (0.01, 0.99), "kF_sf": (0.05, 2.0),
              "M_A_qe": (0.05, 2.0), "s_piN_elastic": (0.05, 2.0), "s_conv": (0.05, 2.0),
              "src_tail": (0.05, 2.0), "axial_strength": (0.05, 2.0)}


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    MODE = os.environ.get("MODE", "cond").lower()
    LABEL = os.environ.get("ADONIS_LABEL", "sec4_ref")
    N = int(os.environ.get("S4_CORNER_N", "17"))
    RANGE = float(os.environ.get("S4_CORNER_RANGE", "3.0"))
    NIT = int(os.environ.get("ALTGEN_NIT", "3"))
    INJECT = os.environ["PHYSFIT_INJECT"]
    want = [s.strip() for s in os.environ.get("S4_CORNER_DIALS", DEFAULT_DIALS).split(",") if s.strip()]

    eng = build_multisample_engine(log)
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    subset = [int(i) for i in np.where(g["shrink"] < 0.5)[0]]
    pn = list(eng.pnames)

    truth, _ = parse_inject(INJECT, nominal_knobs())
    eng.set_closure_data(truth)                     # Asimov: the minimum sits ON the injected truth
    bfp = truth.copy()
    data = np.concatenate([d["data"] for d in eng.ds])
    sigma = np.concatenate([d["sigma"] for d in eng.ds])
    W = np.where(np.isfinite(sigma) & (sigma > 0), 1.0 / sigma ** 2, 0.0)

    def resid(th):
        return eng.model(th) - data

    def chi2(th):
        r = resid(th); return float(np.sum(W * r * r))

    J = eng.jac(bfp, subset)                        # (nbin, nsub) at the best fit
    A = J.T @ (J * W[:, None])
    V = np.linalg.pinv(A, rcond=1e-12)
    spost = np.sqrt(np.abs(np.diag(V)))
    log(f"chi2@bfp = {chi2(bfp):.3e}   (Asimov -> ~0 confirms bfp == injected truth)")

    idx = [subset.index(pn.index(w)) for w in want]          # positions within `subset`
    pairs_all = list(itertools.combinations(range(len(idx)), 2))
    PB = int(os.environ.get("S4_PAIR_BASE", "0"))
    NP = int(os.environ.get("S4_NPAIR", str(len(pairs_all))))
    pairs = pairs_all[PB:PB + NP]
    log(f"dials {want}\n  {len(pairs_all)} pairs total, this shard does {len(pairs)} from index {PB}")

    ax = np.linspace(-RANGE, RANGE, N)
    chi = np.full((len(pairs), N, N), np.nan)
    gn = np.full((len(pairs), N, N, 2), np.nan)      # projected full-16D GN step
    g2 = np.full((len(pairs), N, N, 2), np.nan)      # RAW 2-D gradient on the shown pair (grad mode)
    axes_phys = np.full((len(pairs), 2, N), np.nan)  # the physical grid per pair (grad mode)

    def _axis(kk, cc):
        """Grid for one dial: PHYSICAL range in grad mode, bfp +- RANGE*sigma_post otherwise."""
        if MODE != "grad":
            return bfp[kk] + ax * spost[cc]
        lo, hi = PHYS_RANGE.get(pn[kk].split("[", 1)[0], (bfp[kk] - 3 * spost[cc], bfp[kk] + 3 * spost[cc]))
        lo = max(lo, K.phys_lo(pn[kk]) or -np.inf)        # never grid outside the model's validity
        hi = min(hi, K.phys_hi(pn[kk]) or np.inf)
        return np.linspace(lo, hi, N)

    for pi, (a, b) in enumerate(pairs):
        ca, cb = idx[a], idx[b]                      # positions in `subset`
        ka, kb = subset[ca], subset[cb]
        others = [c for c in range(len(subset)) if c not in (ca, cb)]
        ko = [subset[c] for c in others]
        axa, axb = _axis(ka, ca), _axis(kb, cb)
        axes_phys[pi] = np.stack([axa, axb])
        # snake order so every node starts from its neighbour's solution
        order = [(ia, ib) for ia in range(N) for ib in (range(N) if ia % 2 == 0 else range(N - 1, -1, -1))]
        warm = bfp.copy()
        for (ia, ib) in order:
            th = (warm if MODE == "prof" else bfp).copy()
            th[ka], th[kb] = axa[ia], axb[ib]
            for kk in (ka, kb):                      # respect hard boundaries (E_b >= 0)
                th[kk] = K.clip_phys(pn[kk], th[kk])
            if MODE == "prof":
                th, *_ = lm_fit(eng, ko, f"p{pi}", nit=NIT, th_init=th)
                th[ka], th[kb] = axa[ia], axb[ib]    # lm_fit never moves the pinned pair, re-assert
                for kk in (ka, kb):
                    th[kk] = K.clip_phys(pn[kk], th[kk])
                warm = th.copy()
            r = resid(th)
            chi[pi, ia, ib] = float(np.sum(W * r * r))
            if MODE == "cond":
                grad = 2.0 * J.T @ (W * r)                # J FROZEN at the BFP (cheap, near-BFP only)
                step = -V @ grad
                gn[pi, ia, ib] = (step[ca] / spost[ca], step[cb] / spost[cb])
            elif MODE == "grad":
                # J RECOMPUTED at this node (16 jvps) -- the whole point: over a full physical range a
                # BFP-frozen linearisation is meaningless.  One Jacobian gives BOTH fields, so running
                # "raw 2-D gradient" and "full 16-D GN step" together costs the same as GN alone.
                Jn = eng.jac(th, subset)
                gradn = 2.0 * Jn.T @ (W * r)
                g2[pi, ia, ib] = (gradn[ca], gradn[cb])   # raw gradient, PHYSICAL units
                An = Jn.T @ (Jn * W[:, None])
                stepn = -np.linalg.pinv(An, rcond=1e-12) @ gradn
                gn[pi, ia, ib] = (stepn[ca], stepn[cb])   # full 16-D GN step, projected, PHYSICAL units
            if MODE == "grad" and (ia * N + ib + 1) % 100 == 0:
                log(f"    pair {pi+1}: node {ia*N+ib+1}/{N*N}")
        log(f"  pair {pi+1}/{len(pairs)} ({pn[ka]},{pn[kb]}) done")

    chi -= np.nanmin(chi)
    out = f"output/altgen/{LABEL}_corner2d_{MODE}_{PB:02d}.npz"
    np.savez(out, mode=MODE, dials=want, pair_idx=np.array(pairs), pair_base=PB, axis_sigma=ax,
             dchi2=chi, gn_step=gn, grad2d=g2, axes_phys=axes_phys, bfp=bfp, truth=truth,
             subset=subset, pnames=pn, sigma_post=spost, V=V, A=A, sel_pos=np.array(idx))
    log(f"[out] {out}")


if __name__ == "__main__":
    main()

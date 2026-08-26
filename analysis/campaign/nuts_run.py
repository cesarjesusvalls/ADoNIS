"""Run the NUTS sampler against the paper's multisample posterior.

The sampler itself is adonis.fit.nuts -- leapfrog, tree building, Stan-style warm-up -- and knows
nothing about this campaign.  This module is the part that did: it builds the engine from a fit
config, reads NUTS_CHAIN to pick a chain, and writes output/altgen/<label>_nutsown_<chain>.npz.
Keeping it here is what lets adonis.fit.nuts be imported without the bank machinery.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

from adonis.fit.nuts import nuts_sample, warmup_stan, _metric
from adonis.analysis import knobs as K


def run_real():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    from analysis.campaign.stages.multisample import build_multisample_engine, MULTISAMPLE_NPZ
    from adonis.fit.fitters import parse_inject
    from adonis.reweight.reweight_model import nominal_knobs
    CH = int(os.environ.get("NUTS_CHAIN", "0"))      # which chain -- per-job, set by the runner
    from adonis.fit.config import FitConfig
    cfg = FitConfig.load(os.environ.get("ADONIS_FIT_CONFIG", "configs/fits/sec4_P1.yaml"))
    eng = build_multisample_engine(log, cfg)
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    sub = [int(i) for i in np.where(g["shrink"] < 0.5)[0]]
    # TRUTH: taken from PHYSFIT_INJECT.
    st = cfg.stage("nuts")
    LABEL, NS, NW = cfg.name, int(st["samples"]), int(st["warmup"])
    global MAXDEPTH
    MAXDEPTH = int(st.get("max_depth", 8))
    _inj = cfg.inject_string()
    if not _inj:
        # No silent fallback.  This used to read a truth16.txt in a session scratchpad, which (a) is not
        # reproducible outside that session and (b) held the OLD 16-dial truth, so once
        # res_axial_strength joined the fitted set the sampler would explore a posterior whose data was
        # generated at a different point than the closure it is compared against.
        raise SystemExit("PHYSFIT_INJECT is required: the truth the Asimov data is built at, e.g.\n"
                         "  PHYSFIT_INJECT='M_A_qe=1.12,M_A_res=0.85,...'")
    star, _ = parse_inject(_inj, nominal_knobs())
    log(f"truth: {_inj[:90]}...")
    eng.set_closure_data(star)
    data, sigma = eng.data_sigma()
    ok = np.isfinite(sigma) & (sigma > 0)
    W = np.where(ok, 1.0 / np.where(ok, sigma, 1.0) ** 2, 0.0)
    zc = np.load(f"output/altgen/{LABEL}.npz", allow_pickle=True)
    bfp = np.asarray(zc["fit_th"]); V = np.asarray(zc["fit_V"])
    _fs = [int(k) for k in zc["fit_sub"]] if "fit_sub" in zc.files else sub
    if _fs != list(sub):
        raise SystemExit(f"subset mismatch: gate-I gives {len(sub)} dials, {LABEL}.npz was fitted with "
                         f"{len(_fs)}.  The mass matrix would be built from the wrong covariance.")
    log(f"{len(sub)} dials: " + ",".join(eng.pnames[k] for k in sub))
    sp = np.sqrt(np.abs(np.diag(V))); idx = np.array(sub)
    NF = [0]

    # TRUNCATION to the physical region.  Without it the sampler is not sampling a posterior at all:
    # below E_b's wall the model CLAMPS, so chi2 is exactly flat, the density is flat and unbounded, the
    # posterior is improper and every marginal is undefined.  Measured on the untruncated run, 94.3% of
    # 6000 draws had E_b outside its range (down to -288 in physical units) -- and because those draws
    # still carry values for the other 15 dials, they corrupted ALL the marginals, not just E_b's.
    # Returning -inf outside the box makes NUTS reject the step, which is the correct treatment of a
    # hard physical boundary.
    from adonis.analysis import knobs as _K
    _lo = np.array([-np.inf if _K.phys_lo(eng.pnames[k]) is None else _K.phys_lo(eng.pnames[k])
                    for k in sub])
    _hi = np.array([np.inf if _K.phys_hi(eng.pnames[k]) is None else _K.phys_hi(eng.pnames[k])
                    for k in sub])
    NREJ = [0]

    def gradf(u):
        """log p and its gradient in sigma_post units about the BFP.  Exact: grad chi2 = 2 J^T W r."""
        th = bfp.copy(); th[idx] = bfp[idx] + sp * u
        if np.any(th[idx] < _lo) or np.any(th[idx] > _hi):
            NREJ[0] += 1
            return -np.inf, np.zeros(len(sub))       # outside the physical box
        NF[0] += 1
        r = eng.model(th) - data
        J = eng.jac(th, sub)
        return -0.5 * float(np.sum(W * r * r)), -(J.T @ (W * r)) * sp

    # mass matrix from the KNOWN fit covariance (in sigma_post units) -- no window adaptation needed,
    # and it absorbs the -0.995 pair, which is what keeps the tree shallow.
    D = np.diag(1.0 / sp)
    C = D @ V @ D
    C = 0.5 * (C + C.T) + 1e-9 * np.eye(len(sub))
    Minv = C
    Mchol = np.linalg.cholesky(np.linalg.inv(C))
    lp0, g0 = gradf(np.zeros(len(sub)))
    log(f"logp at BFP = {lp0:.6e}   |grad| = {np.linalg.norm(g0):.3e}")
    t = time.time()
    for _ in range(3): gradf(np.zeros(len(sub)))
    log(f"gradient cost {(time.time()-t)/3*1000:.0f} ms   (pure_callback route was 14000 ms)")

    rng = np.random.default_rng(20260809 + CH)
    eps = float(st.get("eps", 0.35))

    # RESUME (NUTS_RESUME=<npz>): continue an existing chain instead of starting a new one.  A Markov
    # chain's future depends only on its current state, so appending to it is exact -- PROVIDED the kernel
    # is unchanged.  The step size is therefore read back from the file rather than re-derived, and the
    # mass matrix is deterministic from fit_V, so both match the original run by construction.  Warmup is
    # skipped: re-warming would change eps and break stationarity of the pooled samples.
    RES = os.environ.get("NUTS_RESUME", "").strip()
    q_start = np.zeros(len(sub))
    if RES:
        zr = np.load(RES, allow_pickle=True)
        q_start = np.asarray(zr["u"])[-1].copy()
        eps = float(zr["eps"])
        NW = 0
        log(f"RESUMING from {RES}: {len(zr['u'])} existing samples, eps={eps:.4f} (warmup skipped)")

    if NW:
        log(f"warmup {NW} samples to tune eps (start {eps})")
        _, _, a, _ = nuts_sample(rng.standard_normal(len(sub)) * 0.1, gradf, eps, Minv, Mchol, NW, rng,
                                 log=log, tag="warm ")
        am = float(np.mean(a))
        eps *= (am / 0.8) ** 0.5 if am > 0 else 1.0
        log(f"warmup acceptance {am:.3f} -> eps {eps:.4f}")
    s, dep, acc, nf = nuts_sample(q_start, gradf, eps, Minv, Mchol, NS, rng, log=log, tag="")
    out = f"output/altgen/{LABEL}_nutsown_{CH}.npz"
    if RES:                       # never overwrite the run being extended
        out = f"output/altgen/{LABEL}_nutsown_{CH}_ext{os.environ.get('NUTS_EXT','1')}.npz"
    np.savez(out, u=s, bfp=bfp, sigma_post=sp, subset=sub,
             pnames=eng.pnames, truth=star, depth=dep, accept=acc, ngrad=nf, eps=eps,
             resumed_from=RES)
    log(f"[out] {out}  {NS} samples, "
        f"{nf.mean():.1f} grads/sample, acc {acc.mean():.3f}, {NF[0]} gradient calls, "
        f"{NREJ[0]} out-of-bounds rejections")


if __name__ == "__main__":
    run_real()

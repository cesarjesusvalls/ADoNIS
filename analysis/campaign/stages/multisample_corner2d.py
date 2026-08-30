"""2-D corner ingredients for a selected set of dials, in two modes.

MODE=cond ("what the minimiser sees")
    chi2(theta_i, theta_j) with the other dials HELD at the best fit: the conditional surface the
    optimiser actually faces, not a profiled one.  Also stores the full Gauss-Newton step at every node,
    projected onto (i,j) -- gn = -A^-1 grad, A = J^T W J and grad = 2 J^T W r, i.e. the direction the
    algorithm actually takes.  A is held fixed at the BFP.

MODE=prof (inference)
    chi2 PROFILED over the other dials at every node -> honest 68/90% 2-D contours (Dchi2 = 2.30 / 4.61).
    Warm-started along a snake path so each node starts from its neighbour's solution.  Shard with
    ADONIS_PAIR_BASE / ADONIS_PAIR_COUNT across a SLURM array; one npz per shard.

Both modes take the BFP from the injected truth: for an Asimov closure the minimum sits exactly on it,
so neither mode needs the closure npz and both can run in parallel with it.

Env: ADONIS_FIT_CONFIG, ADONIS_FIT_STAGE (profile2d|gradient2d), ADONIS_PAIR_BASE/ADONIS_PAIR_COUNT,
     ADONIS_ROW_BASE/ADONIS_ROW_COUNT (row-level sharding within a pair).
Writes <results>/<label>_corner2d_<mode>_<pairbase>.npz
"""

from analysis._cli import results_dir, FLAT_PRIOR_SCALE, timed_log, shard_range
import os
import sys
import time
import itertools
from pathlib import Path

import numpy as np

from analysis.campaign.stages.multisample import build_multisample_engine, MULTISAMPLE_NPZ, fit_subset
from adonis.fit.fitters import logdet_cov, lm_fit, trf_fit, parse_inject
from adonis.fit import provenance
from adonis.reweight.knobs import PNAMES
from adonis.reweight.reweight_model import nominal_knobs
from adonis.reweight import knobs as K

DEFAULT_DIALS = "M_A_res,delta_strength,Eb_shift,sabs,f_NN_cex,kF_sf"

PHYS_RANGE = {"M_A_res": (0.05, 2.0), "delta_strength": (0.05, 2.0), "Eb_shift": (0.01, 5.0),
              "res_axial_strength": (0.05, 2.0),
              "sabs": (0.05, 2.0), "f_NN_cex": (0.01, 0.99), "kF_sf": (0.05, 2.0),
              "M_A_qe": (0.05, 2.0), "s_piN_elastic": (0.05, 2.0), "s_conv": (0.05, 2.0),
              "src_tail": (0.05, 2.0), "axial_strength": (0.05, 2.0)}


def main():
    log = timed_log()

    STAGE = os.environ.get("ADONIS_FIT_STAGE", "profile2d")
    MODE = {"profile2d": "prof", "gradient2d": "grad"}[STAGE]

    from adonis.fit.config import FitConfig
    cfg = FitConfig.load(os.environ.get("ADONIS_FIT_CONFIG", "configs/fits/sec4_P1.yaml"))
    eng = build_multisample_engine(log, cfg)
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    subset = fit_subset(g, eng.pnames, cfg, log)
    pn = list(eng.pnames)
    st = cfg.stage(STAGE)
    LABEL = cfg.name
    N = int(st["n"])
    RANGE = float(st.get("range", 3.0))
    NIT = int(st.get("max_nfev", 8))
    INJECT = cfg.inject_string()
    want = list(st["dials"])
    _INNER = trf_fit if cfg.fit.minimizer.method == "trf" else lm_fit
    log(f"inner fitter: {_INNER.__name__}")
    PRIOR_SCALE = cfg.fit.prior_scale
    if PRIOR_SCALE != 1.0:
        eng.prior = eng.prior * (FLAT_PRIOR_SCALE if PRIOR_SCALE == 0.0 else PRIOR_SCALE)
    log(f"estimator: {cfg.fit.estimator.upper()}"
        + (" (data-only, no prior)" if cfg.fit.estimator == "mle"
           else f" (prior width x{PRIOR_SCALE:g})"))

    truth, _ = parse_inject(INJECT, nominal_knobs())
    eng.set_closure_data(truth)
    bfp = truth.copy()
    data = np.concatenate([d["data"] for d in eng.ds])
    sigma = np.concatenate([d["sigma"] for d in eng.ds])
    W = np.where(np.isfinite(sigma) & (sigma > 0), 1.0 / sigma ** 2, 0.0)

    def resid(th):
        return eng.model(th) - data

    def chi2(th):
        r = resid(th); return float(np.sum(W * r * r))

    J = eng.jac(bfp, subset)
    A = J.T @ (J * W[:, None])
    V = np.linalg.pinv(A, rcond=1e-12)
    spost = np.sqrt(np.abs(np.diag(V)))
    log(f"chi2@bfp = {chi2(bfp):.3e}   (Asimov -> ~0 confirms bfp == injected truth)")

    idx = [subset.index(pn.index(w)) for w in want]
    pairs_all = list(itertools.combinations(range(len(idx)), 2))
    PB = int(os.environ.get("ADONIS_PAIR_BASE", "0"))
    NP = int(os.environ.get("ADONIS_PAIR_COUNT", str(len(pairs_all))))
    pairs = pairs_all[PB:PB + NP]
    log(f"dials {want}\n  {len(pairs_all)} pairs total, this shard does {len(pairs)} from index {PB}")

    ax = np.linspace(-RANGE, RANGE, N)
    chi = np.full((len(pairs), N, N), np.nan)
    gn = np.full((len(pairs), N, N, 2), np.nan)
    g2 = np.full((len(pairs), N, N, 2), np.nan)
    ldv = np.full((len(pairs), N, N), np.nan)
    axes_phys = np.full((len(pairs), 2, N), np.nan)

    _PROF_AX = None
    if st.get("axes_from") == "profile":
        _pf = str(results_dir() / f"{LABEL}_profile.npz")
        if os.path.exists(_pf):
            _z = np.load(_pf, allow_pickle=True)
            _gs = np.asarray(_z["grids_sigma"])
            _PROF_AX = {int(kk): (float(np.nanmin(_gs[cc])), float(np.nanmax(_gs[cc])))
                        for cc, kk in enumerate([int(q) for q in _z["subset"]])
                        if np.isfinite(_gs[cc]).any()}
            log(f"axes from {_pf}")
        else:
            log(f"[warn] {_pf} missing -- falling back to +-{RANGE} sigma")

    def _axis(kk, cc):
        """Grid for one dial: PHYSICAL range in grad mode, else the profile's adaptive span (or +-RANGE)."""
        if MODE != "grad":
            if _PROF_AX and kk in _PROF_AX:
                a0, a1 = _PROF_AX[kk]
                return bfp[kk] + np.linspace(a0, a1, N) * spost[cc]
            return bfp[kk] + ax * spost[cc]
        lo, hi = PHYS_RANGE.get(pn[kk].split("[", 1)[0], (bfp[kk] - 3 * spost[cc], bfp[kk] + 3 * spost[cc]))
        lo = max(lo, K.phys_lo(pn[kk]) or -np.inf)
        hi = min(hi, K.phys_hi(pn[kk]) or np.inf)
        return np.linspace(lo, hi, N)

    _RB = int(os.environ.get("ADONIS_ROW_BASE", "-1"))
    _NR = int(os.environ.get("ADONIS_ROW_COUNT", "1"))
    _ROWS = range(*shard_range(_RB, _NR, N))
    out = (str(results_dir() / f"{LABEL}_corner2d_{MODE}_n{N:02d}_{PB:02d}.npz") if _RB < 0
           else str(results_dir() / f"{LABEL}_corner2d_{MODE}_n{N:02d}_{PB:02d}_r{_RB:03d}.npz"))

    def _save(final=False):
        """Checkpoint: runs are long and often land on preemptable nodes.  Partial grids keep NaN where
        not yet computed, so a consumer can tell what is missing instead of silently reading zeros."""
        c = chi - np.nanmin(chi) if np.isfinite(chi).any() else chi
        _axsig = np.stack([np.linspace(*_PROF_AX[subset[idx[a]]], N) if (_PROF_AX and subset[idx[a]] in _PROF_AX)
                           else ax for a, _ in pairs]) if pairs else np.array([ax])
        np.savez(out, **provenance.stamp(row_base=max(_RB, 0), n_row=len(_ROWS), n_grid=N),
                 mode=MODE, dials=want, pair_idx=np.array(pairs), pair_base=PB, axis_sigma=ax,
                 axis_sigma_pair=_axsig,
                 dchi2=c, chi2_abs=chi, logdet_Vnuis=ldv, gn_step=gn, grad2d=g2, axes_phys=axes_phys, bfp=bfp, truth=truth,
                 subset=subset, pnames=pn, sigma_post=spost, V=V, A=A, sel_pos=np.array(idx),
                 complete=bool(final), n_done=int(np.isfinite(chi).sum()))
        if final:
            log(f"[out] {out}")

    for pi, (a, b) in enumerate(pairs):
        ca, cb = idx[a], idx[b]
        ka, kb = subset[ca], subset[cb]
        others = [c for c in range(len(subset)) if c not in (ca, cb)]
        ko = [subset[c] for c in others]
        axa, axb = _axis(ka, ca), _axis(kb, cb)
        axes_phys[pi] = np.stack([axa, axb])
        order = [(ia, ib) for ia in _ROWS for ib in (range(N) if ia % 2 == 0 else range(N - 1, -1, -1))]
        warm = bfp.copy()
        for (ia, ib) in order:
            th = (warm if MODE == "prof" else bfp).copy()
            th[ka], th[kb] = axa[ia], axb[ib]
            for kk in (ka, kb):
                th[kk] = K.clip_phys(pn[kk], th[kk])
            if MODE == "prof":
                th, Vn, *_ = _INNER(eng, ko, f"p{pi}", nit=NIT, th_init=th)
                _ld, _nk = logdet_cov(eng, ko, th)
                ldv[pi, ia, ib] = _ld
                th[ka], th[kb] = axa[ia], axb[ib]
                for kk in (ka, kb):
                    th[kk] = K.clip_phys(pn[kk], th[kk])
                warm = th.copy()
            r = resid(th)
            chi[pi, ia, ib] = float(np.sum(W * r * r))
            if MODE == "cond":
                grad = 2.0 * J.T @ (W * r)
                step = -V @ grad
                gn[pi, ia, ib] = (step[ca] / spost[ca], step[cb] / spost[cb])
            elif MODE == "grad":
                Jn = eng.jac(th, subset)
                gradn = 2.0 * Jn.T @ (W * r)
                g2[pi, ia, ib] = (gradn[ca], gradn[cb])
                An = Jn.T @ (Jn * W[:, None])
                stepn = -np.linalg.pinv(An, rcond=1e-12) @ gradn
                gn[pi, ia, ib] = (stepn[ca], stepn[cb])
            done = ia * N + ib + 1
            if done % 100 == 0:
                log(f"    pair {pi+1}: node {done}/{N*N}")
                _save()
        log(f"  pair {pi+1}/{len(pairs)} ({pn[ka]},{pn[kb]}) done")

    _save(final=True)



if __name__ == "__main__":
    main()

"""Profile study: is the fit's QUADRATIC (Gauss-Newton, at-BFP) error actually the likelihood?

The closure quotes sigma = sqrt(diag(V)), V = (J^T W J + prior)^-1 at the BFP -- the parabolic (Laplace)
approximation of the likelihood curvature.  This checks it directly, per dial, by PROFILING the EXACT
nonlinear chi2:

  for each fitted dial k:
    * scan theta_k over BFP +/- 3 sigma_post,k  (marginal sigma from V),
    * at each grid node FIX theta_k and re-minimize chi2 over the other dials (LM) -> profile chi2(theta_k),
    * compare Delta chi2_profile(theta_k)  vs  the parabola (theta_k - BFP_k)^2 / sigma_post,k^2.

If the two agree over +/-3 sigma the marginal error is trustworthy; a dial whose exact profile is skewed
or walls off shows up as a departure.  Reads the BFP + injected data from an existing closure run, so it
profiles the SAME likelihood.

Env: ADONIS_FIT_CONFIG, S4_PROF_BASE/S4_PROF_N (dial-level sharding).  Writes
<results>/<label>_profile.npz.
"""

from analysis._cli import results_dir
import os
import sys
import time
from pathlib import Path

import numpy as np

from analysis.campaign.stages.multisample import build_multisample_engine
from adonis.fit.fitters import logdet_cov, lm_fit, trf_fit
from adonis.fit import provenance
from adonis.reweight.knobs import PNAMES
from adonis.reweight import knobs as K


_INNER = None


def _inner(eng, free, tag, nit, th_init):
    """One profiled re-minimisation.  lm_fit takes kwargs trf_fit does not (huber/record/tol)."""
    return _INNER(eng, free, tag, nit=nit, th_init=th_init)


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    from adonis.fit.config import FitConfig
    cfg = FitConfig.load(os.environ.get("ADONIS_FIT_CONFIG", "configs/fits/sec4_P1.yaml"))
    st = cfg.stage("profile")
    LABEL = cfg.name
    NG = (int(st["n"]) - 1) // 2
    NIT = int(st.get("max_nfev", cfg.fit.minimizer.max_nfev))
    global _INNER
    _INNER = trf_fit if cfg.fit.minimizer.method == "trf" else lm_fit

    z = np.load(str(results_dir() / f"{LABEL}.npz"), allow_pickle=True)
    subset = [int(k) for k in z["subset"]]
    bfp = np.asarray(z["fit_th"] if "fit_th" in z.files else z["M0_th"])
    V = np.asarray(z["fit_V"] if "fit_V" in z.files else z["M0_V"])
    spost = np.sqrt(np.abs(np.diag(V)))
    truth = np.asarray(z["truth"])
    log(f"profiling {LABEL}: {len(subset)} dials, BFP loaded")

    eng = build_multisample_engine(log, cfg)
    nom0 = eng.th0.copy()
    eng.set_closure_data(truth)
    data = np.concatenate([d["data"] for d in eng.ds]); sigma = np.concatenate([d["sigma"] for d in eng.ds])
    PRIOR_SCALE = cfg.fit.prior_scale
    if PRIOR_SCALE != 1.0:
        eng.prior = eng.prior * (1e6 if PRIOR_SCALE == 0.0 else PRIOR_SCALE)
    log(f"estimator: {cfg.fit.estimator.upper()}"
        + (" (data-only, no prior)" if cfg.fit.estimator == "mle"
           else f" (prior width x{PRIOR_SCALE:g})"))

    def objective(th):
        """FULL negative-log-posterior = chi2_data + prior penalty (the curvature sigma_post encodes)."""
        cd = float(np.sum(((eng.model(th) - data) / sigma)**2))
        pri = float(np.sum([(th[j] - nom0[j])**2 / eng.prior[j]**2 for j in subset]))
        return cd + pri, cd

    L_min, cd_min = objective(bfp)
    log(f"objective at BFP: total={L_min:.2f}  chi2_data={cd_min:.2f}")

    NGRID = 2 * NG + 1
    grid = np.linspace(-3.0, 3.0, NGRID)
    grids = np.full((len(subset), NGRID), np.nan)
    prof = np.full((len(subset), NGRID), np.nan)
    profd = np.full((len(subset), NGRID), np.nan)
    logdetV = np.full((len(subset), NGRID), np.nan)
    PB = int(os.environ.get("S4_PROF_BASE", "-1"))
    PN = int(os.environ.get("S4_PROF_N", "1"))
    mine = range(len(subset)) if PB < 0 else range(PB, min(PB + PN, len(subset)))
    if PB >= 0:
        log(f"shard: dials {[eng.pnames[subset[c]] for c in mine]}")
    for c in mine:
        k = subset[c]
        free = [j for j in subset if j != k]
        lo_p, hi_p = K.phys_lo(eng.pnames[k]), K.phys_hi(eng.pnames[k])
        REACH = float(st.get("reach_dchi2", 25.0))
        SPAN = float(st.get("span", 3.0))
        def _edge(sign):
            """Walk outward from the BFP until Dchi2 > REACH or the physical bound."""
            bound = (lo_p if sign < 0 else hi_p)
            b = None if bound is None else (bound - bfp[k]) / spost[c]
            x = sign * SPAN
            for _ in range(6):
                if b is not None and abs(x) >= abs(b):
                    return b
                val = K.clip_phys(eng.pnames[k], bfp[k] + x * spost[c])
                th_i = bfp.copy(); th_i[k] = val
                th_e, *_ = _inner(eng, free, f"reach[{eng.pnames[k]}]{x:+.1f}", NIT, th_i)
                th_e[k] = val
                if objective(th_e)[0] - L_min > REACH:
                    return x
                x *= 1.8
            return x if b is None else (b if abs(b) < abs(x) else x)
        g_lo, g_hi = _edge(-1), _edge(+1)
        if g_lo < 0.0 < g_hi:
            n_lo = int(round((0.0 - g_lo) / (g_hi - g_lo) * (NGRID - 1)))
            n_lo = min(max(n_lo, 1), NGRID - 2)
            gk = np.concatenate([np.linspace(g_lo, 0.0, n_lo + 1),
                                 np.linspace(0.0, g_hi, NGRID - n_lo)[1:]])
        else:
            gk = np.linspace(g_lo, g_hi, NGRID)
        grids[c] = gk
        if g_lo > -3.0 or g_hi < 3.0:
            log(f"  [{eng.pnames[k]}] BOUNDED -> scanning [{g_lo:+.3f}, {g_hi:+.3f}] sigma "
                f"(physical range), not the full +-3")
        for gi, gv in enumerate(gk):
            val = K.clip_phys(eng.pnames[k], bfp[k] + gv * spost[c])
            th_init = bfp.copy(); th_init[k] = val
            th, _V, _J, _m, _c, _cd = _inner(eng, free, f"prof[{eng.pnames[k]}]{gi}", NIT, th_init)
            th[k] = val
            Lval, cdval = objective(th)
            prof[c, gi] = Lval - L_min; profd[c, gi] = cdval - cd_min
            logdetV[c, gi], _nkept = logdet_cov(eng, free, th)
            if gi == 0:
                log(f"    [{eng.pnames[k]}] exact-hessian logdet {logdetV[c, gi]:.3f} "
                    f"({_nkept}/{len(free)} directions kept)")
        _d0, _d1 = prof[c, 0], prof[c, -1]
        _bad = [w for w, d, e in (("lo", _d0, g_lo), ("hi", _d1, g_hi))
                if d < 9.0 and abs(e - ((lo_p if w == "lo" else hi_p) - bfp[k]) / spost[c]
                                   if (lo_p if w == "lo" else hi_p) is not None else 1) > 1e-6]
        log(f"  [{eng.pnames[k]}] profiled ({2*NG+1} nodes) over [{g_lo:+.2f},{g_hi:+.2f}] sigma  "
            f"Dobj@edges=[{_d0:.1f},{_d1:.1f}]  density@edges="
            f"[{np.exp(-0.5*_d0):.3f},{np.exp(-0.5*_d1):.3f}]"
            + ("   *** STILL NOT DECAYED ***" if _bad else ""))

    out = (str(results_dir() / f"{LABEL}_profile.npz") if PB < 0
           else str(results_dir() / f"{LABEL}_profile_sh{PB:02d}.npz"))
    np.savez(out, **provenance.stamp(dial_base=max(PB, 0), n_dial=len(mine), n_subset=len(subset)),
             subset=subset, pnames=eng.pnames, grid_sigma=grid, grids_sigma=grids,
             prof_dobj=prof, prof_dchi2=profd, logdet_Vnuis=logdetV,
             bfp=bfp, sigma_post=spost, truth=truth, chi2_min=cd_min)
    log(f"[out] {out}")


if __name__ == "__main__":
    main()

"""Section 4 profile study: is the fit's QUADRATIC (Gauss-Newton, at-BFP) error actually the likelihood?

The closure quotes sigma = sqrt(diag(V)), V = (J^T W J + prior)^-1 at the BFP -- the parabolic (Laplace)
approximation of the likelihood curvature.  This checks it directly, per dial, by PROFILING the EXACT
nonlinear chi2:

  for each fitted dial k:
    * scan theta_k over BFP +/- 3 sigma_post,k  (marginal sigma from V),
    * at each grid node FIX theta_k and re-minimize chi2 over the OTHER 15 dials (LM) -> profile chi2(theta_k),
    * compare Delta chi2_profile(theta_k)  vs  the parabola (theta_k - BFP_k)^2 / sigma_post,k^2.

If the two agree over +/-3 sigma the marginal error is trustworthy; a dial whose exact profile is skewed or
walls off (Eb_shift's floor is the candidate) shows up as a departure.  Reuses the closure's engine + fit;
reads the BFP + injected data from an existing closure run (so it profiles the SAME likelihood).

Env: ADONIS_LABEL (closure npz to read + profile, default sec4_closure_random16),
     S4_PROFILE_N (grid nodes per side, default 6), ALTGEN_NIT (inner-fit iters, default 20)
     + the S4_*_CHUNKS bank caps.  Writes output/altgen/<label>_profile.npz.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adonis.fit.stages.multisample import build_multisample_engine
from adonis.fit.fitters import logdet_cov, lm_fit, trf_fit
from adonis.fit import provenance
from adonis.analysis.knobs import PNAMES     # core copy; identical to the paper-side one
from adonis.analysis import knobs as K


# INNER FITTER for the profile scan.  Each node re-minimises the other N-1 dials, which is exactly the
# flat-direction problem LM fails at: it clips the step onto the box, inflates lambda until the step
# underflows, and exits via `norm(dth) < 1e-12` reported as convergence.  On the RES axial valley that
# leaves every node's chi2 too high near the truth and DISPLACES the profile minimum -- measured -0.32
# sigma for res_axial_strength on an Asimov fit whose global minimum is the truth to 1e-15.  TRF puts the
# bounds inside the trust-region subproblem and certifies its own optimality.
_INNER = None      # bound from cfg.fit.minimizer.method in main()


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
    NG = (int(st["n"]) - 1) // 2                # config gives TOTAL nodes; the scan wants per side
    NIT = int(st.get("max_nfev", cfg.fit.minimizer.max_nfev))
    global _INNER
    _INNER = trf_fit if cfg.fit.minimizer.method == "trf" else lm_fit

    z = np.load(f"output/altgen/{LABEL}.npz", allow_pickle=True)
    subset = [int(k) for k in z["subset"]]
    bfp = np.asarray(z["fit_th"] if "fit_th" in z.files else z["M0_th"])
    V = np.asarray(z["fit_V"] if "fit_V" in z.files else z["M0_V"])
    spost = np.sqrt(np.abs(np.diag(V)))                    # marginal sigma per dial (subset order)
    truth = np.asarray(z["truth"])
    log(f"profiling {LABEL}: {len(subset)} dials, BFP loaded")

    eng = build_multisample_engine(log, cfg)
    nom0 = eng.th0.copy()                                  # true nominal = the prior centre (never mutated)
    eng.set_closure_data(truth)                            # reproduce the closure's (noiseless) data
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

    # PER-DIAL grid.  A bounded dial must be scanned from its BOUNDARY upward, NOT from bfp-3sigma with
    # the sub-boundary nodes clipped: clipping evaluates the SAME point many times and stores it against
    # x-coordinates that do not exist, so the profile looks "flat" below the wall.  Integrating that
    # fictitious region inflated E_b's 68% credible interval by 34% (half-width 1.366 vs 0.862 restricted,
    # against an ensemble RMS of 1.016).  30% of E_b's old grid lay below the wall.
    NGRID = 2 * NG + 1
    grid = np.linspace(-3.0, 3.0, NGRID)                   # nominal axis, kept for back-compat
    grids = np.full((len(subset), NGRID), np.nan)          # the axis ACTUALLY scanned, per dial
    prof = np.full((len(subset), NGRID), np.nan)           # EXACT profiled Delta(total objective)
    profd = np.full((len(subset), NGRID), np.nan)          # (also keep Delta chi2_data)
    # LOG-DET of the NUISANCE covariance at each profiled node.  Profiling MAXIMISES over the other 15
    # dials; marginalising INTEGRATES over them.  To Laplace order the two differ by the nuisance-volume
    # (Occam) factor, so the marginal posterior is
    #     p(theta_k) ~ exp(-Dchi2_prof/2) * sqrt(det V_nuis(theta_k)) ,
    # i.e. profiling silently assumes that volume is constant along the scan.  For a strongly correlated,
    # CURVED pair the nuisance widths change as you move along the valley, and dropping that variation is
    # a candidate explanation for the profile posterior coming out lopsided about the best fit
    # (measured: 67.6/32.4 for M_A_res, 28.8/71.2 for S_Delta, against 50/50 for the toys).
    logdetV = np.full((len(subset), NGRID), np.nan)
    # SHARDING: dials are independent -- each scan re-minimises the other 15 from its own warm start, so
    # one SLURM task per dial turns a ~1 h serial scan into ~1 wall-clock scan.  Unset -> all dials, and
    # the output filename is unchanged, so the unsharded path behaves exactly as before.
    PB = int(os.environ.get("S4_PROF_BASE", "-1"))
    PN = int(os.environ.get("S4_PROF_N", "1"))
    mine = range(len(subset)) if PB < 0 else range(PB, min(PB + PN, len(subset)))
    if PB >= 0:
        log(f"shard: dials {[eng.pnames[subset[c]] for c in mine]}")
    for c in mine:
        k = subset[c]
        free = [j for j in subset if j != k]               # re-minimise over the OTHER 15 dials
        lo_p, hi_p = K.phys_lo(eng.pnames[k]), K.phys_hi(eng.pnames[k])
        # ADAPTIVE REACH.  A fixed +-3 sigma_post window is not a domain, it is a guess -- and for a
        # shallow direction the curve has not decayed there, so exp(-Dchi2/2) normalised over it is set by
        # where the scan stopped rather than by the likelihood.  Measured: at +-3 sigma the density was
        # still 40% of peak for M_A_res and 60% for S_Delta (vs ~1% for the other 13 dials), which
        # invalidates every mass-based interval (HPD, percentiles) for exactly the two dials of interest.
        # Extend outward in steps until Dchi2 > REACH or a physical bound stops us, then report how far
        # each side actually got so a truncated scan can never again pass silently.
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
        # THE BEST FIT MUST BE A NODE.  linspace(g_lo, g_hi, N) contains 0 only when the range is
        # SYMMETRIC, and the adaptive reach routinely is not (M_A_res [-5.40,+3.00], C5A [-3.00,+5.40]).
        # Without a node at 0 the scan never evaluates its own minimum: Dchi2 is interpolated between the
        # straddling nodes, which displaced the apparent minimum by -0.32 sigma for C5A on an Asimov fit
        # whose global minimum is the truth to 1e-15 -- and skewed every interval read off the curve.
        # Split the nodes either side of 0 in proportion to the reach, keeping NGRID total.
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
            th_init = bfp.copy(); th_init[k] = val          # warm-start from BFP with dial k pinned
            th, _V, _J, _m, _c, _cd = _inner(eng, free, f"prof[{eng.pnames[k]}]{gi}", NIT, th_init)
            th[k] = val                                     # ensure the pinned value (lm_fit never moves it)
            Lval, cdval = objective(th)
            prof[c, gi] = Lval - L_min; profd[c, gi] = cdval - cd_min
            # THE OCCAM LOG-DET COMES FROM THE EXACT HESSIAN, not from the fit's (J^T W J)^-1.
            # The Gauss-Newton matrix drops sum_b r_b d2m_b: a ~2.4e-04 elementwise perturbation, so it
            # is invisible in any sigma -- but a log-determinant sums over ALL eigen-directions and is
            # dominated by the worst-constrained ones, and on this near-degenerate nuisance block
            # (cond ~260) it moves log det V by up to 0.25.  The Occam term enters the profile in units
            # where Delta chi2 = 1 is one sigma, so that is not a rounding difference.  Measured in
            # docs/bench_fair_report.md; the exact hessian costs ~n HVPs.
            # Directions truncated by the pseudo-inverse are dropped and counted: slogdet returns sgn=0
            # the moment one is, which used to give NaN at every node.
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

    out = (f"output/altgen/{LABEL}_profile.npz" if PB < 0
           else f"output/altgen/{LABEL}_profile_sh{PB:02d}.npz")
    # WHICH DIALS this shard was assigned, stamped: the merge then states which are missing from the
    # assignment, instead of inferring it from an all-NaN row (indistinguishable from a scan that ran and
    # produced nothing).
    np.savez(out, **provenance.stamp(dial_base=max(PB, 0), n_dial=len(mine), n_subset=len(subset)),
             subset=subset, pnames=eng.pnames, grid_sigma=grid, grids_sigma=grids,
             prof_dobj=prof, prof_dchi2=profd, logdet_Vnuis=logdetV,
             bfp=bfp, sigma_post=spost, truth=truth, chi2_min=cd_min)
    log(f"[out] {out}")


if __name__ == "__main__":
    main()

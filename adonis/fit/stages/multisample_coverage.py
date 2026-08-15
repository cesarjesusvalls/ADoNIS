"""Section 4 coverage toys: does the fit's (quadratic, at-BFP) error actually cover?

One ensemble, the SAME 16 dials + 20 samples as the closure.  Per toy:
  1. theta*  = nominal + prior * N(0,1)   on the 16 fitted dials     (throw from the prior; Eb_shift
     floored at its physical boundary so it stays recoverable).
  2. data    = model(theta*)  [exact nonlinear reweight]  +  N(0, sigma_bin)   (STAT throw at the fit's
     own per-bin error -- WITHOUT the noise the Asimov self-fit is degenerate, pull == 0).
  3. blind LM fit from nominal -> theta_fit, V.
  4. record pull_k = (theta_fit_k - theta*_k)/sqrt(V_kk) and chi2_data.

Over the ensemble: the pooled pull must be ~N(0,1) (the quadratic error is calibrated / the likelihood is
locally Gaussian) and chi2_data ~ chi2(nbins - k_eff).  coverage_fig.py renders both.

The engine (6 banks) is loaded ONCE; the toys loop over it.  Shard with S4_TOY_BASE across a SLURM array;
the per-shard npz are concatenated by coverage_fig.py.  Saves incrementally (preemption-safe).

Env: S4_NTOYS (default 40)  S4_TOY_BASE (default 0)  ALTGEN_NIT (default 30)  ADONIS_LABEL (sec4_coverage)
     + the S4_*_CHUNKS bank caps from multisample.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from adonis.fit.stages.multisample import build_multisample_engine, MULTISAMPLE_NPZ, fit_subset
from adonis.fit.fitters import lm_fit, trf_fit
from analysis.paper.physical_fit import PNAMES
from adonis.analysis import knobs as K   # PHYS_BOUND / phys_lo: one source of truth for hard boundaries


# Physical reach for a dial whose nominal sits ON a bound and whose prior has been widened away.
# E_b_shift: the prior is +-4 MeV, so a flat draw over [wall, wall+4] spans the physically interesting
# range without inheriting the 1e6 widening.
THROW_REACH = {"Eb_shift": (0.0, 4.0)}


def _throw_truth(eng, subset, real_prior, rng):
    """One toy truth drawn from the REAL prior (legacy ensemble; see S4_FIXED_TRUTH for the other mode).

    Throwing the truth from the prior -- not holding it fixed -- is what makes the toy spread comparable
    to a MAP/posterior width.  At a FIXED truth sitting at the prior centre the estimator is shrunk toward
    the right answer, so Var(theta_hat) = (F+P)^-1 F (F+P)^-1 < (F+P)^-1 and the pull comes out at
    sqrt(F/(F+P)) < 1 by construction -- narrowest exactly on the prior-dominated dials.  Averaged over
    truths drawn from the prior it is (F+P)^-1 and the pull is 1.
    """
    star = eng.th0.copy()
    for k in subset:
        lo, hi = K.phys_lo(eng.pnames[k]), K.phys_hi(eng.pnames[k])
        # UNCONSTRAINED dials (S4_PRIOR_FREE widens the prior by 1e6) have no prior to draw from -- held
        # at nominal.  Without this the widened width feeds straight into the draw: E_b would be thrown
        # uniformly out to ~1e6.
        # UNCONSTRAINED dials would be thrown out to ~1e6 by the widened prior, so they cannot use it --
        # but "hold at nominal" is WRONG for a dial whose nominal IS its boundary.  E_b_shift is stored
        # as 1e-2 with a hard floor at 1e-2, and real_prior 4.0 > 100 x 1e-2 fires this guard, so every
        # coverage toy was thrown with E_b pinned exactly on its wall -- and the flat-draw branch below,
        # written for precisely this dial, was unreachable.  A boundary-coverage number built that way
        # measures the pinning, not the statistics.  Freeze only dials that are NOT on a bound; a dial
        # sitting on one is thrown flat over its physical reach instead.
        if real_prior[k] > 100.0 * max(abs(eng.th0[k]), 1e-12):
            near_bound = (lo is not None and abs(eng.th0[k] - lo) <= 1e-9 * max(abs(lo), 1.0)) or \
                         (hi is not None and abs(eng.th0[k] - hi) <= 1e-9 * max(abs(hi), 1.0))
            if not near_bound:
                continue
            reach = float(np.max(np.abs(THROW_REACH.get(eng.pnames[k], (0.0, 1.0)))))
            star[k] = rng.uniform(lo if lo is not None else eng.th0[k] - reach,
                                  (lo if lo is not None else eng.th0[k]) + reach)
            continue
        # FLAT draw only when the Gaussian actually STRADDLES a boundary (the old E_b case: sigma 4.0 about
        # a nominal of 0.01 put half the draws below zero, and clamping them piled 255/500 toys on the
        # identical truth).  With a 20% prior the bound is many sigma away for every dial, and applying the
        # flat draw unconditionally would replace N(1.0, 0.2) by U(0.05, 0.4) -- a different ensemble.
        if lo is not None and eng.th0[k] - 2.0 * real_prior[k] < lo:
            star[k] = rng.uniform(lo, eng.th0[k] + 2.0 * real_prior[k])
            continue
        for _ in range(100):                            # Gaussian, resampled back inside the box
            v = eng.th0[k] + real_prior[k] * rng.standard_normal()
            if (lo is None or v > lo) and (hi is None or v < hi):
                star[k] = v; break
        else:
            star[k] = float(np.clip(eng.th0[k], lo if lo is not None else -np.inf,
                                    hi if hi is not None else np.inf))
    return star


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    BASE = int(os.environ.get("S4_TOY_BASE", "0"))     # shard offset -- per-job, set by the runner

    from adonis.fit.config import FitConfig
    cfg = FitConfig.load(os.environ.get("ADONIS_FIT_CONFIG", "configs/fits/sec4_P1.yaml"))
    eng = build_multisample_engine(log, cfg)
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    assert [str(x) for x in g["pnames"]] == list(PNAMES)
    subset = fit_subset(g, eng.pnames, cfg, log)
    # throws always use the REAL prior (the population of true values); the FIT's prior can be scaled --
    # S4_PRIOR_SCALE=0 -> unregularised (MLE) coverage of the data-only errors, consistent with S4A.
    real_prior = eng.prior.copy()
    st = cfg.stage("toys")
    NTOYS = int(st.get("shard_size", 50))
    NIT = int(st.get("max_nfev", cfg.fit.minimizer.max_nfev))
    LABEL = f"{cfg.name}_ens"
    PRIOR_SCALE = cfg.fit.prior_scale
    if PRIOR_SCALE != 1.0:
        eng.prior = eng.prior * (1e6 if PRIOR_SCALE == 0.0 else PRIOR_SCALE)
    log(f"estimator: {cfg.fit.estimator.upper()}"
        + (" (data-only, no prior)" if cfg.fit.estimator == "mle"
           else f" (prior width x{PRIOR_SCALE:g})"))
    log(f"coverage: {NTOYS} toys from base {BASE}, {len(subset)} dials, {eng.row0[-1]} bins")

    # FIXED-TRUTH mode (S4_FIXED_TRUTH=<inject string>): every toy shares ONE truth and only the DATA
    # fluctuates.  That is the ensemble the quoted uncertainty actually describes -- a sigma computed at a
    # reference fit is a statement about scatter AT THAT TRUTH.  Throwing a new truth per toy instead mixes
    # in the variation of sigma across parameter space (the model is nonlinear), which muddles "is my error
    # bar right" with "how does my error bar move".  Empty -> legacy prior-thrown truths.
    FIXED = cfg.inject_string() if st.get("fixed_truth") else ""
    fixed_star = None
    if FIXED:
        from adonis.fit.fitters import parse_inject
        from adonis.reweight.reweight_model import nominal_knobs
        fixed_star, _ = parse_inject(FIXED, nominal_knobs())
        log("truth: FIXED for every toy (statistics-only ensemble) -> " +
            " ".join(f"{eng.pnames[k]}={fixed_star[k]:.3f}" for k in subset))

    th_star, th_fit, sig_fit, chi2d = [], [], [], []
    out = f"output/altgen/{LABEL}_{BASE:04d}.npz"
    for t in range(NTOYS):
        seed = BASE + t
        rng = np.random.default_rng(1_000_000 + seed)
        star = fixed_star.copy() if fixed_star is not None else _throw_truth(eng, subset, real_prior, rng)
        eng.set_closure_data(star)                              # nonlinear data at theta*
        for d in eng.ds:                                        # + per-bin STAT throw at the fit's sigma
            sd = np.where(np.isfinite(d["sigma"]), d["sigma"], 0.0)   # empty bins have sigma=inf -> no throw
            d["data"] = d["data"] + rng.normal(0.0, sd)
        # FITTER.  LM clips its step onto the box and, when that stops improving, inflates lambda until
        # the step underflows and exits via `norm(dth) < 1e-12` -- reported as convergence with the Newton
        # decrement still at ~1e-3.  Measured on 48 toys it parks E_b exactly on its floor where TRF, whose
        # bounds are inside the subproblem, pulls it off and reaches a LOWER chi2.  That inflates the
        # boundary atom this ensemble exists to measure, so TRF is the default here.
        _fit = trf_fit if cfg.fit.minimizer.method == "trf" else lm_fit
        th, V, J, m, c, cd = _fit(eng, subset, f"toy{seed}", nit=NIT)
        s = np.sqrt(np.abs(np.diag(V)))
        th_star.append([star[k] for k in subset]); th_fit.append([th[k] for k in subset])
        sig_fit.append(list(s)); chi2d.append(cd)
        log(f"toy {seed}: chi2_data={cd:.1f}  max|pull|="
            f"{max(abs(th[k]-star[k])/max(s[c],1e-12) for c,k in enumerate(subset)):.2f}"
            f"")
        if (t + 1) % 5 == 0 or t == NTOYS - 1:                  # incremental save (preemption-safe)
            # nbins_live = bins that actually CONSTRAIN.  Empty bins get sigma=inf (-> weight 0), so they
            # contribute nothing to chi2 and must not be counted as degrees of freedom either: using the
            # raw bin count puts the chi2(ndf) reference curve to the right of the toys.
            _sig = np.concatenate([d["sigma"] for d in eng.ds])
            np.savez(out, subset=subset, pnames=eng.pnames, nbins=int(eng.row0[-1]),
                     nbins_live=int(np.sum(np.isfinite(_sig) & (_sig > 0))),
                     th_star=np.array(th_star), th_fit=np.array(th_fit),
                     sig_fit=np.array(sig_fit), chi2_data=np.array(chi2d))
    log(f"[out] {out} ({len(th_star)} toys)")


if __name__ == "__main__":
    main()

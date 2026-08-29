"""Coverage toys: does the fit's (quadratic, at-BFP) error actually cover?

One ensemble, the SAME dials + samples as the closure.  Per toy:
  1. theta*  = nominal + prior * N(0,1) on the fitted dials (throw from the prior; Eb_shift floored at
     its physical boundary so it stays recoverable), or a FIXED truth if the config asks for one.
  2. data    = model(theta*) [exact nonlinear reweight] + N(0, sigma_bin) (a STAT throw at the fit's own
     per-bin error -- without it the Asimov self-fit is degenerate, pull == 0).
  3. blind LM fit from nominal -> theta_fit, V.
  4. record pull_k = (theta_fit_k - theta*_k)/sqrt(V_kk) and chi2_data.

Over the ensemble: the pooled pull should be ~N(0,1) (the quadratic error is calibrated / the likelihood
is locally Gaussian) and chi2_data ~ chi2(nbins - k_eff).

The engine is loaded ONCE; the toys loop over it.  Shard with S4_TOY_BASE across a SLURM array; the
per-shard npz are meant to be concatenated downstream.  Saves incrementally (preemption-safe).

Env: S4_TOY_BASE, ADONIS_FIT_CONFIG; toy count/iterations/fixed-truth come from the config's "toys" stage.
"""

from analysis._cli import results_dir
import os
import sys
import time
from pathlib import Path

import numpy as np

from analysis.campaign.stages.multisample import build_multisample_engine, MULTISAMPLE_NPZ, fit_subset
from adonis.fit.fitters import lm_fit, trf_fit
from adonis.reweight.knobs import PNAMES
from adonis.reweight import knobs as K


THROW_REACH = {"Eb_shift": (0.0, 4.0)}


def _throw_truth(eng, subset, real_prior, rng):
    """One toy truth drawn from the prior (see the config's fixed_truth option for the alternative mode).

    Throwing from the prior -- rather than holding truth fixed at its centre -- is what makes the
    ensemble pull match N(0,1); a fixed truth at the prior centre under-covers on prior-dominated dials.
    """
    star = eng.th0.copy()
    for k in subset:
        lo, hi = K.phys_lo(eng.pnames[k]), K.phys_hi(eng.pnames[k])
        if real_prior[k] > 100.0 * max(abs(eng.th0[k]), 1e-12):
            near_bound = (lo is not None and abs(eng.th0[k] - lo) <= 1e-9 * max(abs(lo), 1.0)) or \
                         (hi is not None and abs(eng.th0[k] - hi) <= 1e-9 * max(abs(hi), 1.0))
            if not near_bound:
                continue
            reach = float(np.max(np.abs(THROW_REACH.get(eng.pnames[k], (0.0, 1.0)))))
            star[k] = rng.uniform(lo if lo is not None else eng.th0[k] - reach,
                                  (lo if lo is not None else eng.th0[k]) + reach)
            continue
        if lo is not None and eng.th0[k] - 2.0 * real_prior[k] < lo:
            star[k] = rng.uniform(lo, eng.th0[k] + 2.0 * real_prior[k])
            continue
        for _ in range(100):
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

    BASE = int(os.environ.get("S4_TOY_BASE", "0"))

    from adonis.fit.config import FitConfig
    cfg = FitConfig.load(os.environ.get("ADONIS_FIT_CONFIG", "configs/fits/sec4_P1.yaml"))
    eng = build_multisample_engine(log, cfg)
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    assert [str(x) for x in g["pnames"]] == list(PNAMES)
    subset = fit_subset(g, eng.pnames, cfg, log)
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

    FIXED = cfg.inject_string() if st.get("fixed_truth") else ""
    fixed_star = None
    if FIXED:
        from adonis.fit.fitters import parse_inject
        from adonis.reweight.reweight_model import nominal_knobs
        fixed_star, _ = parse_inject(FIXED, nominal_knobs())
        log("truth: FIXED for every toy (statistics-only ensemble) -> " +
            " ".join(f"{eng.pnames[k]}={fixed_star[k]:.3f}" for k in subset))

    th_star, th_fit, sig_fit, chi2d = [], [], [], []
    out = str(results_dir() / f"{LABEL}_{BASE:04d}.npz")
    for t in range(NTOYS):
        seed = BASE + t
        rng = np.random.default_rng(1_000_000 + seed)
        star = fixed_star.copy() if fixed_star is not None else _throw_truth(eng, subset, real_prior, rng)
        eng.set_closure_data(star)
        for d in eng.ds:
            sd = np.where(np.isfinite(d["sigma"]), d["sigma"], 0.0)
            d["data"] = d["data"] + rng.normal(0.0, sd)
        _fit = trf_fit if cfg.fit.minimizer.method == "trf" else lm_fit
        th, V, J, m, c, cd = _fit(eng, subset, f"toy{seed}", nit=NIT)
        s = np.sqrt(np.abs(np.diag(V)))
        th_star.append([star[k] for k in subset]); th_fit.append([th[k] for k in subset])
        sig_fit.append(list(s)); chi2d.append(cd)
        log(f"toy {seed}: chi2_data={cd:.1f}  max|pull|="
            f"{max(abs(th[k]-star[k])/max(s[c],1e-12) for c,k in enumerate(subset)):.2f}"
            f"")
        if (t + 1) % 5 == 0 or t == NTOYS - 1:
            _sig = np.concatenate([d["sigma"] for d in eng.ds])
            np.savez(out, subset=subset, pnames=eng.pnames, nbins=int(eng.row0[-1]),
                     nbins_live=int(np.sum(np.isfinite(_sig) & (_sig > 0))),
                     th_star=np.array(th_star), th_fit=np.array(th_fit),
                     sig_fit=np.array(sig_fit), chi2_data=np.array(chi2d))
    log(f"[out] {out} ({len(th_star)} toys)")


if __name__ == "__main__":
    main()

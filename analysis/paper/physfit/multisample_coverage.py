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
from analysis.paper.physfit.multisample import build_multisample_engine, MULTISAMPLE_NPZ
from analysis.paper.physfit.physical_fit_run import lm_fit
from analysis.paper.physical_fit import PNAMES
from adonis.analysis import knobs as K   # PHYS_BOUND / phys_lo: one source of truth for hard boundaries


def _throw_truth(eng, subset, real_prior, rng):
    """One toy truth drawn from the REAL prior (legacy ensemble; see S4_FIXED_TRUTH for the other mode)."""
    star = eng.th0.copy()
    for k in subset:
        lo = K.phys_lo(eng.pnames[k])                   # None unless the dial has a hard boundary
        if lo is not None:
            # A bounded dial (E_b >= 0) whose Gaussian prior straddles its boundary: sigma 4.0 MeV about a
            # nominal of 0.01 puts HALF the draws below zero.  Clamping those to the floor made 255/500 toys
            # share the identical truth E_b = 0.01 -- half the ensemble sitting exactly ON the boundary,
            # where the Gaussian pull is not a meaningful diagnostic and the pull width is inflated by an
            # artefact of the throw rather than by the fit.  Draw FLAT on [boundary, 2 sigma] instead; the
            # lower edge comes from the PHYS_BOUND registry, not a literal.
            star[k] = rng.uniform(lo, 2.0 * real_prior[k])
        else:
            star[k] = eng.th0[k] + real_prior[k] * rng.standard_normal()
    return star


def main():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    NTOYS = int(os.environ.get("S4_NTOYS", "40"))
    BASE = int(os.environ.get("S4_TOY_BASE", "0"))
    NIT = int(os.environ.get("ALTGEN_NIT", "30"))
    LABEL = os.environ.get("ADONIS_LABEL", "sec4_coverage")

    eng = build_multisample_engine(log)
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    assert [str(x) for x in g["pnames"]] == list(PNAMES)
    subset = [int(i) for i in np.where(g["shrink"] < 0.5)[0]]
    # throws always use the REAL prior (the population of true values); the FIT's prior can be scaled --
    # S4_PRIOR_SCALE=0 -> unregularised (MLE) coverage of the data-only errors, consistent with S4A.
    real_prior = eng.prior.copy()
    PRIOR_SCALE = float(os.environ.get("S4_PRIOR_SCALE", "0.0"))   # DEFAULT 0.0 = MLE (data-only)
    if PRIOR_SCALE != 1.0:
        eng.prior = eng.prior * (1e6 if PRIOR_SCALE == 0.0 else PRIOR_SCALE)
    log("estimator: " + ("MLE (data-only, no prior)" if PRIOR_SCALE == 0.0
                         else f"MAP (prior width x{PRIOR_SCALE})"))
    log(f"coverage: {NTOYS} toys from base {BASE}, {len(subset)} dials, {eng.row0[-1]} bins")

    # FIXED-TRUTH mode (S4_FIXED_TRUTH=<inject string>): every toy shares ONE truth and only the DATA
    # fluctuates.  That is the ensemble the quoted uncertainty actually describes -- a sigma computed at a
    # reference fit is a statement about scatter AT THAT TRUTH.  Throwing a new truth per toy instead mixes
    # in the variation of sigma across parameter space (the model is nonlinear), which muddles "is my error
    # bar right" with "how does my error bar move".  Empty -> legacy prior-thrown truths.
    FIXED = os.environ.get("S4_FIXED_TRUTH", "").strip()
    fixed_star = None
    if FIXED:
        from analysis.paper.physfit.physical_fit_run import parse_inject
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
        th, V, J, m, c, cd = lm_fit(eng, subset, f"toy{seed}", huber=False, nit=NIT)
        s = np.sqrt(np.abs(np.diag(V)))
        th_star.append([star[k] for k in subset]); th_fit.append([th[k] for k in subset])
        sig_fit.append(list(s)); chi2d.append(cd)
        log(f"toy {seed}: chi2_data={cd:.1f}  max|pull|="
            f"{max(abs(th[k]-star[k])/max(s[c],1e-12) for c,k in enumerate(subset)):.2f}")
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

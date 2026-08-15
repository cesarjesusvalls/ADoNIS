"""How often does a minimiser park E_b on its wall, and does MIGRAD's tolerance control it?

Companion to bench_minimizers.py, which found MIGRAD returning Eb_shift = 0.010 -- its lower bound --
on a closure whose truth was 0.500.  That is the failure mode fitters.py says TRF was adopted to cure
("21.6% of the ensemble was sitting on the floor, ~83% of that being optimiser failure rather than
censoring"), so the question is whether MIGRAD reproduces it and whether `tol` is the knob that fixes it.

THE EXPECTATION, and why this ensemble makes it exactly zero.  Toy truths come from the coverage stage's
thrower for every dial EXCEPT E_b, which is drawn flat over (wall, wall + 2*prior) by _throw() below --
see there for why the upstream thrower cannot supply it (it freezes E_b at nominal, i.e. at the wall).
Thrown E_b truths are therefore uniform in ~(0.01, 8.01) and none are AT the wall.

With ASIMOV data (the default here, --noise off) the maximum-likelihood point IS the truth, exactly.  So
the expected fraction of fits ending on the wall is 0, and every one that does is an optimiser failure --
no censoring to subtract, no modelling assumption in the comparison.  Turn --noise on and the expectation
stops being zero: a toy thrown close to the wall can have its unconstrained MLE pushed below it by the
stat fluctuation and get clamped, which is CENSORING and entirely legitimate.  The script reports the
analytic censoring rate P(e + N(0,sigma_post) < lo) averaged over the thrown truths, so the two effects
are separated rather than conflated.

Usage:
    srun --jobid=<ID> --overlap python -m adonis.fit.bench_tol_boundary [config] \
        [--toys N] [--tols 0.1,0.01,1e-3] [--noise] [--sig-cap N]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from adonis.analysis import knobs as K

EB = "Eb_shift"


def _throw(eng, subset, real_prior, rng, wall_dial=EB):
    """Toy truth, with `wall_dial` deliberately thrown AWAY from its wall.

    multisample_coverage._throw_truth cannot be used unmodified here, and the reason is a bug in it:
    its first guard skips any dial whose prior exceeds 100x its nominal, meant to catch the S4_PRIOR_FREE
    1e6 widening.  Eb_shift trips it on its REAL prior (4.0 vs a nominal of 0.01, ratio 400), so the
    function `continue`s and leaves E_b at nominal -- which IS the wall.  The flat-draw branch right
    below, whose comment says it exists for exactly the E_b case, is unreachable for E_b.  Measured:
    3/3 smoke toys came back with Eb* = 0.010 and every fit "correctly" sat on the wall, so the ensemble
    tested nothing.

    Here E_b is drawn FLAT over (wall, wall + 2*prior).  A flat draw, not a Gaussian, because a Gaussian
    of width 4.0 about a nominal of 0.01 puts half its mass below the wall and piles those toys onto one
    identical clamped truth -- the pathology that motivated the branch in the first place.
    """
    star = _throw_truth_upstream(eng, subset, real_prior, rng)
    k = eng.pnames.index(wall_dial)
    lo = K.phys_lo(wall_dial)
    star[k] = rng.uniform(lo, lo + 2.0 * real_prior[k])
    return star


def _at_bound(name, v, rtol=1e-3):
    """Is this dial sitting ON its lower wall?

    phys_lo is the bound OFFSET INWARD by FLOOR_EPS (knobs.py:78), so a fit that has converged onto the
    wall returns phys_lo itself, not the mathematical bound.  Compare against phys_lo, with a tolerance,
    rather than against zero -- testing `v < 1e-6` would score every wall-parked fit as free.
    """
    lo = K.phys_lo(name)
    return lo is not None and abs(v - lo) <= rtol * max(abs(lo), 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="configs/fits/sec4_P1.yaml")
    ap.add_argument("--toys", type=int, default=30)
    ap.add_argument("--tols", default="0.1,0.01,1e-3,1e-4", help="MIGRAD tol values to scan")
    ap.add_argument("--noise", action="store_true", help="add the per-bin stat throw (expectation != 0)")
    ap.add_argument("--sig-cap", type=int, default=60000)
    ap.add_argument("--nit", type=int, default=200)
    ap.add_argument("--max-calls", type=int, default=50000)
    ap.add_argument("--fixed-truth", action="store_true",
                    help="use the CONFIG's injected truth for every toy and vary only the statistical "
                         "throw. This is the closure repeated under noise -- one likelihood surface, N "
                         "data realisations -- which is what a coverage/boundary statement is about. "
                         "Without it each toy also re-throws the other 16 dial truths, which averages "
                         "over different surfaces and conflates landscape variation with the effect.")
    ap.add_argument("--nograd", action="store_true",
                    help="also run derivative-free MIGRAD at the first tol (finite-difference gradients)")
    ap.add_argument("--eb-truth", default="",
                    help="fix the Eb_shift TRUTH for every toy: a number, or 'wall'. With 'wall' plus "
                         "--noise this is the textbook Chernoff setup -- a parameter whose true value "
                         "sits ON its boundary -- for which the asymptotic boundary mass is exactly 1/2.")
    ap.add_argument("--eb-start", type=float, default=0.0,
                    help="start Eb_shift HERE instead of at nominal (which IS its wall). 0 = nominal. "
                         "Tests whether the wall trapping is a property of the START point rather than "
                         "of the minimiser's tolerance.")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--out", default="output/altgen/bench_tol_boundary.npz")
    a = ap.parse_args(argv)
    tols = [float(s) for s in a.tols.split(",") if s.strip()]

    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    import jax
    log(f"jax {jax.__version__}  devices={jax.devices()}")

    import dataclasses
    from adonis.fit.config import FitConfig
    from adonis.fit.fitters import trf_fit
    from adonis.fit.stages.multisample import (MULTISAMPLE_NPZ, build_multisample_engine, fit_subset)
    from adonis.fit.stages.multisample_coverage import _throw_truth as _tu
    globals()['_throw_truth_upstream'] = _tu
    from adonis.fit.bench_minimizers import _bounds, _migrad
    from adonis.fit.fitters import parse_inject
    from adonis.reweight.reweight_model import nominal_knobs

    cfg = FitConfig.load(a.config)
    if a.sig_cap:
        cfg = dataclasses.replace(cfg, banks=dataclasses.replace(cfg.banks, sig_cap=a.sig_cap))
        log(f"[override] sig_cap -> {a.sig_cap:,} (a smaller fit: fewer events, more masked bins)")
    eng = build_multisample_engine(log, cfg)
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    subset = fit_subset(g, eng.pnames, cfg, log)
    idx = np.array(subset, int)
    prior_true = eng.prior[idx].copy()
    real_prior = eng.prior.copy()
    if cfg.fit.prior_scale != 1.0:
        eng.prior = eng.prior * (1e6 if cfg.fit.prior_scale == 0.0 else cfg.fit.prior_scale)
    truth_cfg, _ = parse_inject(cfg.inject_string(), nominal_knobs())   # the closure's own injection
    names = [eng.pnames[k] for k in subset]
    keb = names.index(EB)
    lo, hi = _bounds(eng, subset)
    x0 = np.asarray(eng.th0)[idx].copy()
    th_init = np.asarray(eng.th0).copy()
    if a.eb_start:
        # Move BOTH minimisers' start, so the comparison stays paired -- shifting only MIGRAD's would
        # confound "start point" with "different problem".
        x0[keb] = a.eb_start
        th_init[subset[keb]] = a.eb_start
        log(f"[start] {EB} starts at {a.eb_start} instead of nominal {eng.th0[subset[keb]]:.3g} (= the wall)")
    log(f"{len(subset)} dials; {EB} wall at {K.phys_lo(EB)}, prior {prior_true[keb]}")

    # objectives compiled ONCE, outside every clock (see bench_minimizers for why this matters)
    f_only = eng.chi2_fn(subset)
    vg = eng.chi2_grad_fn(subset)
    log("objectives compiled")

    rng = np.random.default_rng(a.seed)
    methods = ["gn"] + [f"migrad+g@{t:g}" for t in tols]
    if a.nograd:
        methods.append(f"migrad@{tols[0]:g}")           # no gradient supplied -> 2n calls per gradient
    # FIXED TRUTH: draw it once, outside the loop, so every toy shares one likelihood surface and the
    # only thing varying is the data.  Both minimisers see the SAME data in a given toy by construction
    # (it is built once per toy, before either runs), so the comparison is paired throw by throw.
    fixed_star = None
    if a.fixed_truth:
        fixed_star = truth_cfg.copy()
        log(f"[truth] FIXED at the config's injection for all {a.toys} toys; only the stat throw varies")
    star_eb, stars, fits = [], [], {m: [] for m in methods}
    times = {m: [] for m in methods}
    chi2s = {m: [] for m in methods}
    ncalls = {m: [] for m in methods}

    for it in range(a.toys):
        star = fixed_star.copy() if fixed_star is not None else _throw(eng, subset, real_prior, rng)
        if a.eb_truth:
            star[subset[keb]] = K.phys_lo(EB) if a.eb_truth == "wall" else float(a.eb_truth)
        eng.set_closure_data(star)
        if a.noise:
            for d in eng.ds:
                sd = np.where(np.isfinite(d["sigma"]), d["sigma"], 0.0)
                d["data"] = d["data"] + sd * rng.standard_normal(len(sd))
        # Retarget the COMPILED objectives at this toy's data.  Rebuilding them instead would recompile
        # (~1-2 min of XLA) once per toy and dwarf the ensemble itself.
        Dt, Wt = eng.chi2_data_dev()
        f_only.set_data(Dt, Wt); vg.set_data(Dt, Wt)
        star_eb.append(float(star[subset[keb]])); stars.append(star[idx].copy())

        for m in methods:
            if m == "gn":
                t1 = time.perf_counter()
                th, *_rest, c_data = trf_fit(eng, subset, "toy", nit=a.nit, th_init=th_init)
                dt = time.perf_counter() - t1
                x = th[idx]
            else:
                tol = float(m.split("@")[1])
                x, nfg, c_data, dt = _migrad(eng, subset, x0, lo, hi, m.startswith("migrad+g"),
                                             tol, a.max_calls, f_only, vg)
            if m == "gn":
                nfg = (int(getattr(eng, "last_nfev", -1)), int(getattr(eng, "last_njev", -1)))
            fits[m].append(x.copy()); times[m].append(dt); chi2s[m].append(float(c_data))
            ncalls[m].append(nfg)
        log(f"  toy {it+1}/{a.toys}: Eb*={star_eb[-1]:6.3f} -> "
            + "  ".join(f"{m.split('@')[-1]}:{fits[m][-1][keb]:6.3f}" for m in methods))

    star_eb = np.array(star_eb)
    wall = K.phys_lo(EB)
    # ---- A. EXPECTED boundary ("Chernoff") fraction ------------------------------------------------ #
    # Chernoff: when a parameter's TRUE value sits on its boundary, the MLE lands exactly on that
    # boundary with asymptotic probability 1/2 -- half the data fluctuations push the unconstrained
    # optimum outside the allowed region and get clamped.  That is a property of the STATISTICS, not of
    # the minimiser, so a correct minimiser must reproduce 1/2 and any excess is optimiser failure.
    #   truth AT the wall + noise -> 1/2, exactly, no nuisance parameters needed.
    #   truth OFF the wall + noise -> the same argument with the truth displaced: P(MLE < wall) =
    #     Phi((wall - e)/sigma_post), averaged over the thrown truths.
    #   no noise (Asimov)          -> 0: the MLE IS the truth, there is nothing to fluctuate.
    from scipy import stats as sstats
    at_wall_truth = bool(a.eb_truth == "wall")
    if not a.noise:
        exp_frac, exp_txt = 0.0, "0.0%  (Asimov: the MLE IS the truth, nothing to fluctuate)"
    elif at_wall_truth:
        exp_frac, exp_txt = 0.5, "50.0%  (Chernoff: true value ON the boundary)"
    else:
        s_post = float(np.std([f[keb] - e for f, e in zip(fits["gn"], star_eb)])) or 1e-9
        exp_frac = float(np.mean(sstats.norm.cdf((wall - star_eb) / s_post)))
        exp_txt = f"{100*exp_frac:.1f}%  (censoring at sigma_post~{s_post:.3f})"

    print(f"\n==== E_b boundary study: {a.toys} toys, {'stat-thrown' if a.noise else 'Asimov'} data, "
          f"Eb truth {'AT THE WALL' if at_wall_truth else 'thrown'}, start Eb={a.eb_start or 'nominal'} ====")
    print(f"A) EXPECTED fraction at the boundary: {exp_txt}\n")
    print(f"{'method':>16} | {'A: at wall':>18} | {'B: median s':>11} {'value':>6} {'deriv':>6} "
          f"| {'C: median chi2':>14} {'chi2 IQR':>16}")
    S = np.array(stars)
    n_ok = 0
    for m in methods:
        F = np.array(fits[m]); N = np.array(ncalls[m]); C = np.array(chi2s[m])
        nb = int(sum(_at_bound(EB, v) for v in F[:, keb])); fr = nb / len(F)
        # binomial 1-sigma on the measured fraction, so "agrees with expectation" is decidable
        se = float(np.sqrt(max(fr * (1 - fr), 1e-12) / len(F)))
        z = (fr - exp_frac) / se if se > 0 else np.inf
        q1, q3 = np.percentile(C, [25, 75])
        print(f"{m:>16} | {nb:3d}/{len(F):<3d} {100*fr:5.1f}+-{100*se:4.1f}% {z:+5.1f}sig | "
              f"{np.median(times[m]):11.2f} {int(np.median(N[:, 0])):6d} {int(np.median(N[:, 1])):6d} "
              f"| {np.median(C):14.4e} [{q1:.3e},{q3:.3e}]")
    np.savez(a.out, methods=np.array(methods, dtype=object), star_eb=star_eb, wall=wall,
             exp_frac=exp_frac, noise=a.noise, toys=a.toys, eb_start=a.eb_start, prior_true=prior_true, stars=S,
             names=np.array(names, dtype=object), keb=keb,
             **{f"{m}_x": np.array(fits[m]) for m in methods},
             **{f"{m}_t": np.array(times[m]) for m in methods},
             **{f"{m}_chi2": np.array(chi2s[m]) for m in methods},
             **{f"{m}_n": np.array(ncalls[m]) for m in methods})
    log(f"[out] {a.out}")


if __name__ == "__main__":
    sys.exit(main())

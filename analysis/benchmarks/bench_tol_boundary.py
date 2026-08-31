"""How often does a minimiser park E_b on its wall, and does MIGRAD's tolerance control it?

Companion to bench_minimizers.py, which found MIGRAD returning Eb_shift at its lower bound on a closure
whose truth was away from it -- the failure mode fitters.py adopted TRF to cure.  Toy truths are drawn
from the coverage stage's thrower for every dial except E_b, which _throw() below draws flat over
(wall, wall + 2*prior) so that none land AT the wall.

With ASIMOV data (the default, --noise off) the MLE IS the truth exactly, so the expected fraction of
fits ending on the wall is 0 and every one that does is an optimiser failure.  With --noise the
expectation is instead the analytic censoring rate P(e + N(0,sigma_post) < lo) averaged over the thrown
truths, reported separately so censoring and optimiser failure are not conflated.

Usage:
    srun --jobid=<ID> --overlap python -m analysis.benchmarks.bench_tol_boundary [config] \
        [--toys N] [--tols 0.1,0.01,1e-3] [--noise] [--sig-cap N]
"""
from __future__ import annotations

from analysis._cli import results_dir, FLAT_PRIOR_SCALE, timed_log

import argparse
import sys
import time

import numpy as np

from adonis.reweight import knobs as K

EB = "Eb_shift"


def _throw(eng, subset, real_prior, rng, wall_dial=EB):
    """Toy truth, with `wall_dial` deliberately thrown AWAY from its wall.

    Delegates to multisample_coverage._throw_truth for every other dial, then overrides `wall_dial`:
    drawn FLAT over (wall, wall + 2*prior) rather than Gaussian, so no mass lands below the wall.
    """
    from analysis.campaign.stages.multisample_coverage import _throw_truth
    star = _throw_truth(eng, subset, real_prior, rng)
    k = eng.pnames.index(wall_dial)
    lo = K.phys_lo(wall_dial)
    star[k] = rng.uniform(lo, lo + 2.0 * real_prior[k])
    return star


def _at_bound(name, v, rtol=1e-3):
    """Is this dial sitting ON its lower wall?

    phys_lo is the bound OFFSET INWARD by FLOOR_EPS, so a wall-parked fit returns phys_lo itself, not
    the mathematical bound.  Compare against phys_lo with a tolerance, not against zero.
    """
    lo = K.phys_lo(name)
    return lo is not None and abs(v - lo) <= rtol * max(abs(lo), 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="configs/fits/closure.yaml")
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
    ap.add_argument("--out", default=str(results_dir() / "bench_tol_boundary.npz"))
    a = ap.parse_args(argv)
    tols = [float(s) for s in a.tols.split(",") if s.strip()]

    log = timed_log()

    import jax
    log(f"jax {jax.__version__}  devices={jax.devices()}")

    import dataclasses
    from adonis.fit.config import FitConfig
    from adonis.fit.fitters import trf_fit
    from analysis.campaign.stages.multisample import (MULTISAMPLE_NPZ, build_multisample_engine, fit_subset)
    from analysis.benchmarks._shared import _bounds, _migrad
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
        eng.prior = eng.prior * (FLAT_PRIOR_SCALE if cfg.fit.prior_scale == 0.0 else cfg.fit.prior_scale)
    truth_cfg, _ = parse_inject(cfg.inject_string(), nominal_knobs())
    names = [eng.pnames[k] for k in subset]
    keb = names.index(EB)
    lo, hi = _bounds(eng, subset)
    x0 = np.asarray(eng.th0)[idx].copy()
    th_init = np.asarray(eng.th0).copy()
    if a.eb_start:
        x0[keb] = a.eb_start
        th_init[subset[keb]] = a.eb_start
        log(f"[start] {EB} starts at {a.eb_start} instead of nominal {eng.th0[subset[keb]]:.3g} (= the wall)")
    log(f"{len(subset)} dials; {EB} wall at {K.phys_lo(EB)}, prior {prior_true[keb]}")

    f_only = eng.chi2_fn(subset)
    vg = eng.chi2_grad_fn(subset)
    log("objectives compiled")

    rng = np.random.default_rng(a.seed)
    methods = ["gn"] + [f"migrad+g@{t:g}" for t in tols]
    if a.nograd:
        methods.append(f"migrad@{tols[0]:g}")
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

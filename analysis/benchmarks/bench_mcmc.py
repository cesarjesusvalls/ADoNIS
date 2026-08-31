"""Gradient-free vs gradient-based MCMC on one ADoNIS posterior: Metropolis-Hastings vs NUTS.

Both samplers query the same `FitKernel` (MH: `chi2`, NUTS: `chi2_and_grad`), so likelihood, box,
dataset, precision and device are shared by construction -- the only difference is whether the sampler
uses a gradient.  The posterior is exp(-chi2/2) truncated to the box (MLE, prior widened x1e6).

MH is preconditioned by the LAPLACE COVARIANCE (J^T W J)^-1 -- the same geometry NUTS gets as its mass
matrix -- with its step scale adapted during warm-up to 0.234 acceptance (Roberts-Gelman-Gilks), so the
comparison is against the best standard MH, not a strawman.

Each chain warms up for a fixed iteration count then samples for a fixed WALL-CLOCK budget.  ESS/s is
hardware-specific, so it is reported beside ESS per MODEL EVALUATION and per EVENT-PASS (portable), and
both bulk and tail ESS are given since a random walk can mix in the body while barely crossing the
quantiles a physics result quotes.

    srun ... python -m analysis.benchmarks.bench_mcmc --ndials 8 --method mh --chain 0
"""
from __future__ import annotations

from analysis._cli import results_dir, FLAT_PRIOR_SCALE, timed_log

import argparse
import sys
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="configs/fits/closure.yaml")
    ap.add_argument("--sig-cap", type=int, default=15000)
    ap.add_argument("--ndials", type=int, required=True)
    ap.add_argument("--dials", default="", help="explicit comma-separated dial NAMES; overrides the "
                    "nested shrinkage subset.  Lets a FIXED n be run with different parameter sets, "
                    "which is the only way to separate dimensional scaling from the individual role of "
                    "a tricky parameter -- the nested subsets confound the two by construction.")
    ap.add_argument("--tag", default="", help="label for the output file when --dials is used")
    ap.add_argument("--method", choices=("mh", "nuts"), required=True)
    ap.add_argument("--chain", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=0, help="0 = method default (MH 4000, NUTS 300)")
    ap.add_argument("--sample-seconds", type=float, default=300.0)
    ap.add_argument("--noise-seed", type=int, default=1, help="the pseudo-data throw, SHARED by both")
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--adapt-metric", action="store_true",
                    help="NUTS only: Stan-style warm-up (dual averaging + WINDOWED METRIC adaptation) "
                         "instead of the fixed Laplace metric.  This breaks the strict fairness of the "
                         "MH comparison -- NUTS then acquires geometry MH does not have -- so it is a "
                         "SEPARATE arm answering 'what would production do', not a replacement for the "
                         "matched-geometry run.")
    ap.add_argument("--ref", default=str(results_dir() / "closure.npz"))
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    log = timed_log()

    import dataclasses
    import jax
    from adonis.fit import nuts as NU
    from analysis.benchmarks._shared import _dial_order, freeze_sample, throw
    from adonis.fit.config import FitConfig
    from adonis.fit.fitters import parse_inject
    from adonis.fit.kernels import FitKernel
    from adonis.fit.minimizers import gn_fit
    from analysis.campaign.stages import multisample as MS
    from adonis.reweight.reweight_model import nominal_knobs

    log(f"jax {jax.__version__} devices={jax.devices()} x64={jax.config.jax_enable_x64}")
    if jax.devices()[0].platform != "gpu":
        raise SystemExit("not on a GPU")
    NU.MAXDEPTH = int(a.max_depth)

    cfg = FitConfig.load(a.config)
    cfg = dataclasses.replace(cfg, banks=dataclasses.replace(cfg.banks, sig_cap=a.sig_cap))
    eng = MS.build_multisample_engine(log, cfg)
    g = np.load(MS.MULTISAMPLE_NPZ, allow_pickle=True)
    if a.dials:
        want = [w.strip() for w in a.dials.split(",") if w.strip()]
        miss = [w for w in want if w not in list(eng.pnames)]
        if miss:
            raise SystemExit(f"--dials: unknown {miss}")
        subset = sorted(list(eng.pnames).index(w) for w in want)
        if len(subset) != a.ndials:
            raise SystemExit(f"--dials gave {len(subset)} dials, --ndials says {a.ndials}")
    else:
        subset = sorted(_dial_order(g, eng.pnames)[:a.ndials])
    nlive = freeze_sample(eng, np.load(a.ref, allow_pickle=True), log)

    th0 = np.asarray(eng.th0, float); idx = np.asarray(subset, int)
    truth_full, _ = parse_inject(cfg.inject_string(), nominal_knobs())
    truth = th0.copy(); truth[idx] = truth_full[idx]
    eng.set_closure_data(truth)
    throw(eng, np.random.default_rng(1_000_000 + a.noise_seed))
    if cfg.fit.prior_scale != 1.0:
        eng.prior = eng.prior * (FLAT_PRIOR_SCALE if cfg.fit.prior_scale == 0.0 else cfg.fit.prior_scale)

    kern = FitKernel(eng, subset)
    kern.warmup(which=("residuals", "jac", "chi2", "chi2_and_grad"))
    names = list(kern.pnames)
    lo, hi = kern.bounds()
    log(f"n={a.ndials} dials {names}")

    x_start = th0[idx].copy()
    if "Eb_shift" in names:
        x_start[names.index("Eb_shift")] = 2.0
    fit = gn_fit(kern, x_start, max_nfev=200, gtol=cfg.fit.minimizer.gtol)
    xb = fit.x
    V = kern.covariance_gn(fit.J)
    V = 0.5 * (V + V.T)
    spost = np.sqrt(np.abs(np.diag(V)))
    Lv = np.linalg.cholesky(V + 1e-12 * np.eye(len(xb)) * np.trace(V) / len(xb))
    log(f"BFP chi2 {fit.chi2:.3f}, sigma_post {np.round(spost, 4)}")

    inbox = lambda q: bool(np.all(q > lo) and np.all(q < hi))

    def logp(q):
        if not inbox(q):
            return -np.inf
        return -0.5 * kern.chi2(q)

    def logp_grad(q):
        if not inbox(q):
            return -np.inf, np.zeros_like(q)
        c, gr = kern.chi2_and_grad(q)
        return -0.5 * c, -0.5 * gr

    rng = np.random.default_rng(20260817 + 977 * a.chain + 13 * a.ndials)
    for _ in range(200):
        q = xb + 2.0 * (Lv @ rng.standard_normal(len(xb)))
        if inbox(q):
            break
    else:
        q = xb.copy()
    log(f"chain {a.chain} start {np.round((q - xb) / spost, 2)} sigma from the BFP")

    kern.reset_counts()
    draws, extra = [], {}
    t_warm0 = time.time()

    if a.method == "mh":
        nw = a.warmup or 4000
        s = 2.38 / np.sqrt(len(xb))
        lp = logp(q); nacc = 0
        for i in range(nw):
            qp = q + s * (Lv @ rng.standard_normal(len(xb)))
            lpp = logp(qp)
            if np.log(rng.random()) < lpp - lp:
                q, lp = qp, lpp; nacc += 1
            if (i + 1) % 100 == 0:
                rate = nacc / 100.0
                s *= float(np.exp((rate - 0.234) * 1.0))
                nacc = 0
        t_warm = time.time() - t_warm0
        log(f"MH warm-up {nw} its, scale {s:.4f}")
        kern.reset_counts()
        t1 = time.time(); nacc = 0; nit = 0
        while time.time() - t1 < a.sample_seconds:
            for _ in range(50):
                qp = q + s * (Lv @ rng.standard_normal(len(xb)))
                lpp = logp(qp)
                if np.log(rng.random()) < lpp - lp:
                    q, lp = qp, lpp; nacc += 1
                draws.append(q.copy()); nit += 1
        t_samp = time.time() - t1
        extra = dict(accept=nacc / max(nit, 1), scale=s, ndiv=0, ngrad=0, depth=0.0, ndiv_frac=0.0)

    else:
        nw = a.warmup or 300
        Minv = V
        Mchol = np.linalg.cholesky(np.linalg.inv(V) + 1e-12 * np.eye(len(xb)))
        eps = 0.5
        if a.adapt_metric:
            q, _lp, _g, eps, Minv, Mchol, _inf = NU.warmup_stan(
                q, logp_grad, Minv, max(nw, 2000), rng, log=log)
        else:
            for _ in range(6):
                blk = max(10, nw // 6)
                sm, dep, acc, nf = NU.nuts_sample(q, logp_grad, eps, Minv, Mchol, blk, rng,
                                                  log=lambda m: None)
                q = sm[-1]
                am = float(np.mean(acc))
                eps *= (am / 0.8) ** 0.5 if am > 0 else 1.0
        t_warm = time.time() - t_warm0
        log(f"NUTS warm-up {nw} its, eps {eps:.4f}")
        kern.reset_counts()
        t1 = time.time(); deps = []; accs = []; dv = []
        while time.time() - t1 < a.sample_seconds:
            sm, dep, acc, nf = NU.nuts_sample(q, logp_grad, eps, Minv, Mchol, 10, rng,
                                              log=lambda m: None, ndiv_out=dv)
            q = sm[-1]
            draws.extend(list(sm))
            deps.append(np.mean(dep)); accs.append(np.mean(acc))
        t_samp = time.time() - t1
        ndiv = int(np.sum(dv))
        extra = dict(accept=float(np.mean(accs)), scale=eps, ndiv=ndiv,
                     ngrad=int(kern.counts["grad"]), depth=float(np.mean(deps)),
                     ndiv_frac=float(ndiv / max(len(draws), 1)))

    D = np.asarray(draws)
    c = dict(kern.counts)
    out = a.out or str(results_dir() / f"mcmc_{a.method}_n{a.ndials}_c{a.chain}.npz")
    np.savez(out, draws=D, names=np.array(names, dtype=object), method=a.method, ndials=a.ndials,
             chain=a.chain, sig_cap=a.sig_cap, nlive=nlive, bfp=xb, V=V, spost=spost,
             t_sample=t_samp, t_warm=t_warm, n_logp=c["chi2"], n_vg=c["vg"], n_grad=c["grad"],
             visits=kern.visits(),
             passes=kern.event_passes(), counts=np.array([c], dtype=object),
             noise_seed=a.noise_seed, **{k: v for k, v in extra.items()})
    log(f"{a.method} n={a.ndials} chain {a.chain}: {len(D):,} draws in {t_samp:.1f}s "
        f"| logp {c['chi2']:,} grad {c['grad']:,} passes {kern.event_passes():,.0f} "
        f"| accept {extra['accept']:.3f}")
    log(f"[out] {out}")


if __name__ == "__main__":
    sys.exit(main())

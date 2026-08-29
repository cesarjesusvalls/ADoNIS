"""Workload (wall time, event-passes) to reach common convergence targets, as a function of N events/sample
and n fitted dials, for Gauss-Newton and MIGRAD (with/without gradient) on one shared FitKernel.

n indexes NESTED subsets of the Gate-I dials, ordered by shrinkage (best-constrained first).  Sigma and
the live-bin mask are frozen from --ref rather than recomputed at each N.  Data is closure truth on the
fitted dials plus per-bin Gaussian noise (one Asimov fit per cell is kept as a contrast).  Times are
compared to a COMMON chi2/parameter-distance target, not to each method's native stopping rule.

Usage:
    srun --jobid=<ID> --overlap python -m analysis.benchmarks.bench_fair --sig-cap 60000 --ndials 2,4,8,12,17
"""
from __future__ import annotations

import argparse
import gc
import sys
import time

import numpy as np


CHI2_TARGETS = (1.0, 1e-2, 1e-4, 1e-8)
DIST_TARGETS = (1e-1, 1e-3, 1e-6)


def _dial_order(g, pnames):
    """The Gate-I dials, best-constrained first.  Deterministic, so the n-subsets are nested."""
    shrink = np.asarray(g["shrink"], float)
    gate1 = np.where(shrink < 0.5)[0]
    return [int(k) for k in gate1[np.argsort(shrink[gate1])]]


def freeze_sample(eng, ref, log):
    """Impose the REFERENCE run's sigma and live-bin mask on this engine, per dataset key.

    Keys are matched by name, so a sample-composition mismatch raises KeyError rather than silently
    misaligning.
    """
    keys = [str(k) for k in ref["dskeys"]]
    row0 = np.asarray(ref["row0"], int)
    sig_ref = np.asarray(ref["sigma"], float)
    by_key = {k: sig_ref[row0[i]:row0[i + 1]] for i, k in enumerate(keys)}
    nlive = 0
    for s in eng.samples:
        for d in s.ds:
            sg = by_key[d["key"]]
            if len(sg) != d["nbin"]:
                raise SystemExit(f"{d['key']}: reference has {len(sg)} bins, engine has {d['nbin']}")
            d["sigma"] = sg.copy()
            d["_sigma0"] = sg.copy()
            nlive += int(np.isfinite(sg).sum())
    log(f"  sample frozen from reference: {nlive} live bins, sigma fixed (independent of sig_cap)")
    return nlive


def throw(eng, rng, log=None):
    """data <- data + N(0, sigma) on live bins.  Dead bins are left alone; their sigma is inf."""
    for s in eng.samples:
        for d in s.ds:
            sg = np.asarray(d["sigma"], float)
            ok = np.isfinite(sg) & (sg > 0)
            d["data"] = np.asarray(d["data"], float) + np.where(ok, rng.normal(0, np.where(ok, sg, 1.0)), 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="configs/fits/sec4_P1.yaml")
    ap.add_argument("--sig-cap", type=int, required=True, help="events per sample for this job")
    ap.add_argument("--ndials", default="2,4,8,12,17")
    ap.add_argument("--ref", default="output/altgen/sec4_P1.npz",
                    help="run whose sigma + live-bin mask define the sample at EVERY N")
    ap.add_argument("--noise-seeds", default="1,2,3", help="statistical realisations; '' for none")
    ap.add_argument("--asimov", action="store_true", default=True)
    ap.add_argument("--no-asimov", dest="asimov", action="store_false")
    ap.add_argument("--eb-start", type=float, default=2.0,
                    help="Eb_shift start, in MeV, away from its lower bound at 0.01")
    ap.add_argument("--methods", default="gn,migrad+g,migrad")
    ap.add_argument("--nit", type=int, default=200)
    ap.add_argument("--tol", type=float, default=0.1, help="iminuit tol")
    ap.add_argument("--max-calls", type=int, default=100000)
    ap.add_argument("--jac-batch", type=int, default=0, help="0 = all dials in one dispatch")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit a CPU run, for the CPU-vs-GPU comparison; never mix platforms "
                         "in one grid")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    import dataclasses
    import jax

    from adonis.fit.config import FitConfig
    from adonis.fit.fitters import parse_inject
    from adonis.fit.kernels import FitKernel
    from adonis.fit.minimizers import dist_to, gn_fit, migrad_fit, time_to
    from analysis.campaign.stages import multisample as MS
    from adonis.reweight.reweight_model import nominal_knobs

    log(f"jax {jax.__version__}  devices={jax.devices()}  x64={jax.config.jax_enable_x64}")
    if jax.devices()[0].platform != "gpu" and not a.allow_cpu:
        raise SystemExit("not on a GPU: rows from different machines are not comparable. "
                         "Pass --allow-cpu to time this machine deliberately; the platform is "
                         "recorded in the output either way.")
    if not jax.config.jax_enable_x64:
        raise SystemExit("jax_enable_x64 is off: the objective would be float32")

    ndials = [int(s) for s in a.ndials.split(",") if s.strip()]
    seeds = [int(s) for s in a.noise_seeds.split(",") if s.strip()]
    methods = [s.strip() for s in a.methods.split(",") if s.strip()]

    cfg = FitConfig.load(a.config)
    cfg = dataclasses.replace(cfg, banks=dataclasses.replace(cfg.banks, sig_cap=a.sig_cap))
    log(f"config {cfg.path}  digest {cfg.digest()}  sig_cap {a.sig_cap:,}")
    eng = MS.build_multisample_engine(log, cfg)
    g = np.load(MS.MULTISAMPLE_NPZ, allow_pickle=True)
    order = _dial_order(g, eng.pnames)
    log(f"dial order (Gate-I, best constrained first): "
        + " ".join(eng.pnames[k] for k in order))

    ref = np.load(a.ref, allow_pickle=True)
    nlive_ref = freeze_sample(eng, ref, log)

    truth_full, _ = parse_inject(cfg.inject_string(), nominal_knobs())
    th0 = np.asarray(eng.th0, float)
    prior_true = np.asarray(eng.prior, float).copy()
    if cfg.fit.prior_scale != 1.0:
        eng.prior = eng.prior * (1e6 if cfg.fit.prior_scale == 0.0 else cfg.fit.prior_scale)
    log(f"estimator {cfg.fit.estimator.upper()} (prior x{cfg.fit.prior_scale:g})")

    rows, out, kern = [], {}, None
    for n in sorted(ndials, reverse=True):
        subset = sorted(order[:n])
        names = [eng.pnames[k] for k in subset]
        idx = np.asarray(subset, int)

        truth = th0.copy(); truth[idx] = truth_full[idx]
        eng.set_closure_data(truth)
        data0 = [np.asarray(d["data"], float).copy() for s in eng.samples for d in s.ds]

        if kern is not None:
            del kern
            kern = None
        gc.collect()
        jax.clear_caches()
        try:
            kern = FitKernel(eng, subset, jac_batch=(a.jac_batch or n))
            kern.warmup()
        except Exception as e:
            s_ = str(e)
            if "RESOURCE_EXHAUSTED" in s_ or "OUT_OF_MEMORY" in s_.upper():
                try:
                    mm = jax.local_devices()[0].memory_stats()
                    log(f"  device at failure: {mm['bytes_in_use']/2**30:.2f}/"
                        f"{mm['bytes_limit']/2**30:.2f} GB in use, peak {mm['peak_bytes_in_use']/2**30:.2f} GB")
                except Exception:
                    pass
                log(f"  XLA said:\n{s_}")
                raise SystemExit(
                    f"device OOM building the n={n} jacobian at batch {a.jac_batch or n} "
                    f"(see the allocation breakdown above).")
            raise
        log(f"n={n:2d}: {kern.describe()}  compile {kern.compile_s:.1f}s  "
            f"live {kern.n_live} (ref {nlive_ref})")
        assert kern.n_live == nlive_ref, "frozen mask did not hold -- the N axis would be confounded"

        x0 = th0[idx].copy()
        if "Eb_shift" in names:
            x0[names.index("Eb_shift")] = a.eb_start
        xt = truth[idx]

        for tag, seed in ([("asimov", None)] if a.asimov else []) + [(f"noise{s}", s) for s in seeds]:
            k = 0
            for s in eng.samples:
                for d in s.ds:
                    d["data"] = data0[k].copy(); k += 1
            if seed is not None:
                throw(eng, np.random.default_rng(1_000_000 + seed))
            kern.refresh_data()

            fits = {}
            for meth in methods:
                if meth == "gn":
                    r = gn_fit(kern, x0, max_nfev=a.nit, gtol=cfg.fit.minimizer.gtol,
                               xtol=cfg.fit.minimizer.xtol, ftol=cfg.fit.minimizer.ftol)
                elif meth in ("migrad+g", "migrad"):
                    r = migrad_fit(kern, x0, tol=a.tol, max_calls=a.max_calls,
                                   use_grad=(meth == "migrad+g"))
                else:
                    raise SystemExit(f"unknown method {meth}")
                fits[meth] = r
                log(f"  n={n:2d} {tag:>7} {meth:>9}: {r.wall:8.2f}s  passes {r.passes:8.0f}  "
                    f"chi2 {r.chi2:.6e}  nfev {r.nfev} njev {r.njev}  "
                    f"{'ok' if r.converged else 'NOT CONVERGED'}")

            c_ref = min(f.chi2 for f in fits.values())
            best = min(fits.values(), key=lambda f: f.chi2)

            spost, cond, smin = None, np.nan, np.nan
            if "gn" in fits and fits["gn"].J is not None:
                J = fits["gn"].J
                V = fits["gn"].covariance()
                spost = np.sqrt(np.abs(np.diag(V)))
                sv = np.linalg.svd(J, compute_uv=False)
                cond, smin = float(sv[0] / max(sv[-1], 1e-300)), float(sv[-1])
            scale = spost if spost is not None else prior_true[idx]

            eb_wall = None
            if "Eb_shift" in names:
                j = names.index("Eb_shift")
                eb_wall = {m: bool(abs(f.x[j] - 0.01) < 1e-6) for m, f in fits.items()}

            for meth, f in fits.items():
                tt = time_to(f.trace, c_ref, CHI2_TARGETS)
                _t, dd, _p = dist_to(f.trace, best.x, scale)
                dt = {}
                for eps in DIST_TARGETS:
                    hit = np.where(dd < eps)[0]
                    dt[eps] = (float(_t[hit[0]]), float(_p[hit[0]])) if len(hit) else None
                rows.append(dict(n=n, N=a.sig_cap, tag=tag, method=meth, wall=f.wall,
                                 passes=f.passes, chi2=f.chi2, chi2_ref=c_ref,
                                 nfev=f.nfev, njev=f.njev, converged=f.converged,
                                 counts=f.counts, t_chi2=tt, t_dist=dt, cond=cond, smin=smin,
                                 eb_wall=(eb_wall or {}).get(meth), x=f.x,
                                 bias=float(np.max(np.abs((f.x - xt) / prior_true[idx])))))
                out[f"trace_{n}_{tag}_{meth}"] = np.column_stack(f.trace.arrays()[:3])

            for f_ in fits.values():
                f_.trace = None
            del fits, best

            log(f"  n={n:2d} {tag:>7} chi2_ref {c_ref:.6e}  cond(J) {cond:.3e}  "
                + (f"Eb on wall: {[m for m, v in eb_wall.items() if v] or 'none'}" if eb_wall else ""))

    print(f"\n==== workload at {a.sig_cap:,} events/sample ({10*a.sig_cap:,} resident) ====")
    print(f"{'n':>3} {'case':>7} {'method':>9} {'wall s':>9} {'passes':>9} {'nfev':>6} {'njev':>5} "
          f"{'chi2':>12} {'t(1e-4)':>9} {'p(1e-4)':>9} {'conv':>5}")
    for r in rows:
        h = r["t_chi2"].get(1e-4)
        print(f"{r['n']:>3} {r['tag']:>7} {r['method']:>9} {r['wall']:9.2f} {r['passes']:9.0f} "
              f"{r['nfev']:6d} {r['njev']:5d} {r['chi2']:12.4e} "
              f"{(f'{h[0]:9.2f}' if h else '        -')} {(f'{h[1]:9.0f}' if h else '        -')} "
              f"{'y' if r['converged'] else 'N':>5}")

    outfile = a.out or f"output/altgen/bench_fair_N{a.sig_cap}.npz"
    np.savez(outfile, rows=np.array(rows, dtype=object), sig_cap=a.sig_cap,
             ndials=np.array(ndials), order=np.array(order), nlive=nlive_ref,
             chi2_targets=np.array(CHI2_TARGETS), dist_targets=np.array(DIST_TARGETS),
             platform=jax.devices()[0].platform, eb_start=a.eb_start, ref=a.ref, **out)
    log(f"[out] {outfile}")


if __name__ == "__main__":
    sys.exit(main())

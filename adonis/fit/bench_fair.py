"""Computing workload as a function of (number of events, number of parameters), for GN and MIGRAD.

Phases 4-5 of docs/bench_fair_plan.md.  Everything both minimisers touch is ONE `FitKernel`, so the only
difference between the arms is which derivative object the algorithm asks for.  What this produces is a
workload surface -- time AND event-passes to reach COMMON accuracy targets -- not a headline number.

THE AXES

  N   events per sample (banks.sig_cap).  Ten samples, so the resident set is 10 x N and that is what
      every model evaluation walks.
  n   fitted dials: NESTED SUBSETS of the Gate-I 17, ordered by shrinkage (best-constrained first), so
      n=17 is exactly the sec4 fit and every smaller n is a genuine sub-problem of it.  The estimator
      stays MLE throughout, as in sec4.  The axis deliberately stops at 17: going to the full 28-knob
      basis would drag in dials that fail Gate I, and the only way to keep that well-posed would be to
      turn the prior back on -- i.e. to benchmark a DIFFERENT fit than the paper's.

WHAT IS HELD FIXED ACROSS THE GRID

  The sample definition.  sigma and the live-bin mask are FROZEN from a reference run (the production
  250k configuration) rather than recomputed at each N.  Without this the N axis is confounded: fewer
  events means a larger per-bin MC error, which trips the sparse-bin cut, which changes ndf -- measured
  136/175/219/282 live bins at 150k/300k/600k/2.4M total.  A "scaling with statistics" curve built that
  way is partly a curve of how many bins were dropped.  Freezing them makes N mean one thing: how many
  events a model evaluation walks.

  The start point.  Nominal, except that Eb_shift starts OFF its lower bound.  Starting a parameter on a
  wall measures time-to-fail, not time-to-converge -- MIGRAD's bounded-parameter sine transform has a
  vanishing derivative there, so it sees zero internal gradient and parks.  That behaviour is already a
  completed, separately reported result; repeating it inside every cell of a workload scan would just
  contaminate the scan.

  The data.  Closure truth (injected on the FITTED dials only, so every n is a reachable closure) plus
  per-bin Gaussian noise at the fit's own sigma.  The throw is generated ONCE per realisation and every
  method fits that same data.  Noise is the primary regime because Gauss-Newton drops a Hessian term
  proportional to the residual: on a perfect closure that term vanishes and GN *becomes* Newton, so an
  Asimov-only surface would flatter it everywhere.  One Asimov fit per cell is kept as the contrast.

HOW THE TIMES ARE COMPARED

  Not by their native stopping rules -- GN stops on the projected gradient, MIGRAD on EDM, and they
  therefore stop at different accuracies.  Every objective evaluation is timestamped and tagged with the
  kernel's event-pass count, and the report is time (and passes) to reach a COMMON target, measured
  against the lowest chi2 any method reached in that cell.  A method that never reaches a target is
  recorded as not reaching it; a failure is never scored as a fast time.

Usage:
    srun --jobid=<ID> --overlap python -m adonis.fit.bench_fair --sig-cap 60000 --ndials 2,4,8,12,17
"""
from __future__ import annotations

import argparse
import gc
import sys
import time

import numpy as np


# Accuracy targets, common to every method.  The deepest is bounded from below by the objective's own
# floor (|chi2_kernel - chi2_host| ~ 1.6e-11 at the closure truth, measured by verify_kernels): a target
# below that would be measuring float noise rather than a minimiser.
CHI2_TARGETS = (1.0, 1e-2, 1e-4, 1e-8)
DIST_TARGETS = (1e-1, 1e-3, 1e-6)


def _dial_order(g, pnames):
    """The 17 Gate-I dials, best-constrained first.  Deterministic, so the n-subsets are nested."""
    shrink = np.asarray(g["shrink"], float)
    gate1 = np.where(shrink < 0.5)[0]
    return [int(k) for k in gate1[np.argsort(shrink[gate1])]]


def freeze_sample(eng, ref, log):
    """Impose the REFERENCE run's sigma and live-bin mask on this engine, per dataset key.

    Both are properties of the measurement, not of how many MC events happen to be resident, and letting
    them follow the event count is what makes an N scan uninterpretable.  Keys are matched by name, so a
    sample-composition change is a loud KeyError rather than a silent misalignment.
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
            d["_sigma0"] = sg.copy()          # so set_closure_data cannot recompute it
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
    from adonis.fit.stages import multisample as MS
    from adonis.reweight.reweight_model import nominal_knobs

    log(f"jax {jax.__version__}  devices={jax.devices()}  x64={jax.config.jax_enable_x64}")
    if jax.devices()[0].platform != "gpu":
        raise SystemExit("not on a GPU: every number here would be off a different machine")
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
    if cfg.fit.prior_scale != 1.0:               # MLE: data-only, as in sec4
        eng.prior = eng.prior * (1e6 if cfg.fit.prior_scale == 0.0 else cfg.fit.prior_scale)
    log(f"estimator {cfg.fit.estimator.upper()} (prior x{cfg.fit.prior_scale:g})")

    rows, out, kern = [], {}, None
    # DESCENDING n: the biggest jacobian is built first, on a device that has not yet been
    # fragmented by anything else.  Cell order does not affect any result -- each n is independent.
    for n in sorted(ndials, reverse=True):
        subset = sorted(order[:n])
        names = [eng.pnames[k] for k in subset]
        idx = np.asarray(subset, int)

        # TRUTH ON THE FITTED DIALS ONLY.  Injecting the full 17-dial truth and then freezing the ones
        # outside the subset makes the closure unreachable: the fit chases a data vector it cannot
        # represent, chi2_min is hundreds instead of ndf, and the n axis stops being apples-to-apples.
        truth = th0.copy(); truth[idx] = truth_full[idx]
        eng.set_closure_data(truth)
        data0 = [np.asarray(d["data"], float).copy() for s in eng.samples for d in s.ds]

        # RELEASE THE PREVIOUS KERNEL BEFORE BUILDING THE NEXT.  Each kernel holds five compiled XLA
        # executables per sample plus their CUDA graphs, and those do NOT go away when the python object
        # is rebound.  Building the n=17 kernel while the n=2 kernel's executables were still resident
        # is what exhausted an 11 GB card at 20k events/sample -- where the actual tangent data is ~3 MB,
        # so it was never the arithmetic.  Drop the reference, then clear jax's compilation cache.
        #
        # AND NO RETRY-ON-OOM.  An earlier version halved the batch and tried again; after a device OOM
        # the CUDA context is poisoned ("Recorded commands are not empty ... can be recorded at most
        # once") and the retry fails with an INTERNAL error that is not an OOM, so the fallback both
        # failed and disguised why.  A clean, loud failure with the flag to set is worth more than a
        # recovery that cannot be trusted.
        if kern is not None:
            del kern
            kern = None
        gc.collect()
        jax.clear_caches()
        try:
            kern = FitKernel(eng, subset, jac_batch=(a.jac_batch or n))
            kern.warmup()
        except Exception as e:                        # noqa: BLE001
            s_ = str(e)
            if "RESOURCE_EXHAUSTED" in s_ or "OUT_OF_MEMORY" in s_.upper():
                # PRINT XLA'S OWN MESSAGE.  Replacing it with a friendlier one threw away the
                # allocation size and the memory breakdown -- the only facts that identify WHAT is too
                # big -- and cost an hour of guessing at a ceiling that could have been read off directly.
                try:
                    mm = jax.local_devices()[0].memory_stats()
                    log(f"  device at failure: {mm['bytes_in_use']/2**30:.2f}/"
                        f"{mm['bytes_limit']/2**30:.2f} GB in use, peak {mm['peak_bytes_in_use']/2**30:.2f} GB")
                except Exception:                             # noqa: BLE001
                    pass
                log(f"  XLA said:\n{s_}")
                raise SystemExit(
                    f"device OOM building the n={n} jacobian at batch {a.jac_batch or n} "
                    f"(see the allocation breakdown above).")
            raise
        log(f"n={n:2d}: {kern.describe()}  compile {kern.compile_s:.1f}s  "
            f"live {kern.n_live} (ref {nlive_ref})")
        assert kern.n_live == nlive_ref, "frozen mask did not hold -- the N axis would be confounded"

        # start: nominal, with Eb_shift lifted off its lower bound
        x0 = th0[idx].copy()
        if "Eb_shift" in names:
            x0[names.index("Eb_shift")] = a.eb_start
        xt = truth[idx]

        for tag, seed in ([("asimov", None)] if a.asimov else []) + [(f"noise{s}", s) for s in seeds]:
            # restore the Asimov data, then throw THIS realisation on top of it
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

            # ONE reference minimum per cell: the lowest any method reached.  Grading each method
            # against its own final chi2 would score a method that stopped early as perfectly accurate.
            c_ref = min(f.chi2 for f in fits.values())
            best = min(fits.values(), key=lambda f: f.chi2)

            # sigma_post from the GN jacobian if we have one -- the covariance is a by-product of the
            # fit, computed AFTER the clock stopped, never inside it
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

            # DROP EVERY REFERENCE TO THE KERNEL BEFORE THE NEXT ONE IS BUILT.  Trace keeps `self.k` so
            # it can tag each objective evaluation with an event-pass count, and FitResult keeps the
            # Trace -- so the previous cell's `fits` dict pins the whole kernel, and the `del kern` at the
            # top of the next n block frees nothing.  That is why N=125k built n=17 and then OOMed on
            # n=12, a SMALLER kernel: the memory was never the new kernel's, it was the old one's.
            # The trace arrays are already copied into `out` above, so nothing is lost here.
            for f_ in fits.values():
                f_.trace = None
            del fits, best

            log(f"  n={n:2d} {tag:>7} chi2_ref {c_ref:.6e}  cond(J) {cond:.3e}  "
                + (f"Eb on wall: {[m for m, v in eb_wall.items() if v] or 'none'}" if eb_wall else ""))

    # ---- report --------------------------------------------------------------------------------- #
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

"""How the three minimisers scale in the number of PARAMETERS and the number of EVENTS.

bench_minimizers.py measures one point of this surface (17 dials, one bank size).  The claim the paper
wants to make is about scaling, and scaling has to be measured on both axes because the methods do not
depend on them the same way:

  in the DIAL COUNT n
    chi2 evaluation      O(1)   -- one pass over the events, whatever n is
    reverse-mode grad    O(1)   -- one VJP, independent of n; this is the whole point of reverse mode
    forward jacobian     O(n)   -- one JVP per dial (batched), so GN's per-iteration cost grows
    finite-diff grad     O(n)   -- 2n chi2 calls, which is what derivative-free MIGRAD pays per step
    numerical hessian    O(n^2) -- 2n^2 calls (HESSE)
  in the EVENT COUNT N
    every one of the above is O(N), because each is some number of passes over the resident events.
    So the event axis should rescale all methods TOGETHER and leave the ratios flat -- if it does not,
    something other than the model evaluation is dominating and the comparison needs re-reading.

The event axis needs a separate engine build (a bank load, several minutes) so it is the OUTER loop and
is driven by re-running this script per --sig-cap.  The dial axis is free within one build and is scanned
internally.  Timing rules are bench_minimizers': only the minimisation is clocked, objectives are
compiled and warmed outside it, and each dial count gets its own compiled objectives.

Usage:
    srun --jobid=<ID> --overlap python -m analysis.benchmarks.bench_scaling --sig-cap 60000 --dials 4,8,12,17
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from analysis.benchmarks.bench_minimizers import _bounds, _gn, _migrad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="configs/fits/sec4_P1.yaml")
    ap.add_argument("--dials", default="4,8,12,17", help="dial counts to scan (prefix of the Gate-I set)")
    ap.add_argument("--sig-cap", type=int, default=60000, help="events per sample for THIS run")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.1)
    ap.add_argument("--nit", type=int, default=200)
    ap.add_argument("--max-calls", type=int, default=100000)
    ap.add_argument("--methods", default="gn,migrad+g,migrad")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    counts = [int(s) for s in a.dials.split(",") if s.strip()]
    methods = [s.strip() for s in a.methods.split(",") if s.strip()]
    out = a.out or f"output/altgen/bench_scaling_{a.sig_cap}.npz"

    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    import dataclasses
    import jax
    log(f"jax {jax.__version__}  devices={jax.devices()}")

    from adonis.fit.config import FitConfig
    from adonis.fit.fitters import parse_inject
    from analysis.campaign.stages.multisample import (MULTISAMPLE_NPZ, build_multisample_engine, fit_subset)
    from adonis.reweight.reweight_model import nominal_knobs

    cfg = FitConfig.load(a.config)
    cfg = dataclasses.replace(cfg, banks=dataclasses.replace(cfg.banks, sig_cap=a.sig_cap))
    log(f"sig_cap {a.sig_cap:,}  (fewer events also masks more bins: each cap is its own fit)")
    eng = build_multisample_engine(log, cfg)
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    full = fit_subset(g, eng.pnames, cfg, log)
    truth_full, _ = parse_inject(cfg.inject_string(), nominal_knobs())
    if cfg.fit.prior_scale != 1.0:
        eng.prior = eng.prior * (1e6 if cfg.fit.prior_scale == 0.0 else cfg.fit.prior_scale)
    nbin = len(eng.data_sigma()[0])
    log(f"{len(full)} dials available, {nbin} bins total; scanning {counts}")

    rows = []
    for nd in counts:
        if nd > len(full):
            log(f"[skip] {nd} dials requested, only {len(full)} in the Gate-I set")
            continue
        sub = list(full[:nd])
        idx = np.array(sub, int)
        star = np.asarray(eng.th0).copy()
        star[idx] = np.asarray(truth_full)[idx]
        eng.set_closure_data(star)
        xt = star[idx]
        x0 = np.asarray(eng.th0)[idx].copy()
        lo, hi = _bounds(eng, sub)
        _d, _s = eng.data_sigma()
        nlive = int((np.isfinite(_s) & (_s > 0)).sum())

        f_only = eng.chi2_fn(sub)
        vg = eng.chi2_grad_fn(sub)
        f_only(x0); vg(x0); eng.jac(np.asarray(eng.th0), sub)

        def _t(fn, n=5):
            fn(); ts = []
            for _ in range(n):
                s = time.perf_counter(); fn(); ts.append(time.perf_counter() - s)
            return float(np.median(ts))
        c_f = _t(lambda: f_only(x0))
        c_vg = _t(lambda: vg(x0)[1])
        c_J = _t(lambda: eng.jac(np.asarray(eng.th0), sub))
        log(f"  n={nd:2d}  live bins {nlive}/{nbin}  per-call: chi2 {1e3*c_f:6.1f} ms | "
            f"+grad {1e3*c_vg:6.1f} ms ({c_vg/c_f:4.2f}x) | jacobian {1e3*c_J:7.1f} ms ({c_J/c_f:5.1f}x)")

        for meth in methods:
            ts, ns = [], []
            for _r in range(a.reps):
                if meth == "gn":
                    x, n_, c_, dt = _gn(eng, sub, cfg.fit.minimizer.newton_tol, a.nit)
                else:
                    x, n_, c_, dt = _migrad(eng, sub, x0, lo, hi, meth == "migrad+g",
                                            a.tol, a.max_calls, f_only, vg)
                ts.append(dt); ns.append(n_)
            ns = np.array(ns)
            rows.append(dict(n=nd, method=meth, t=float(np.median(ts)),
                             nval=int(np.median(ns[:, 0])), nder=int(np.median(ns[:, 1])),
                             chi2=float(c_), c_f=c_f, c_vg=c_vg, c_J=c_J, nlive=nlive,
                             bias=float(np.abs(x - xt).max())))
            log(f"  n={nd:2d}  {meth:9s} {np.median(ts):8.2f}s  value {int(np.median(ns[:,0])):5d} "
                f"deriv {int(np.median(ns[:,1])):4d}")

    _nl = rows[0]["nlive"] if rows else 0
    print(f"\n==== scaling at {a.sig_cap:,} events/sample ({10*a.sig_cap:,} total), "
          f"{_nl}/{nbin} live bins ({jax.devices()[0].platform}) ====")
    print("    truth injected on the FITTED dials only, so chi2_min = 0 at every n")
    print(f"{'n':>3} {'method':>9} {'median s':>9} {'value':>6} {'deriv':>6} "
          f"{'chi2/call ms':>12} {'jac/call ms':>12} {'vs GN':>7}")
    for nd in counts:
        base = next((r["t"] for r in rows if r["n"] == nd and r["method"] == "gn"), None)
        for r in [r for r in rows if r["n"] == nd]:
            rel = f"{r['t']/base:6.2f}x" if base else "     -"
            print(f"{r['n']:3d} {r['method']:>9} {r['t']:9.2f} {r['nval']:6d} {r['nder']:6d} "
                  f"{1e3*r['c_f']:12.1f} {1e3*r['c_J']:12.1f} {rel:>7}   chi2 {r['chi2']:.2e}")
    np.savez(out, sig_cap=a.sig_cap, nbin=nbin, counts=np.array(counts),
             rows=np.array([(r["n"], r["method"], r["t"], r["nval"], r["nder"], r["chi2"],
                             r["c_f"], r["c_vg"], r["c_J"], r["bias"], r["nlive"]) for r in rows],
                            dtype=object))
    log(f"[out] {out}")


if __name__ == "__main__":
    sys.exit(main())

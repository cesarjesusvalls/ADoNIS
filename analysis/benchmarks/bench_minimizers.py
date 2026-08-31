"""Minimiser benchmark on the section-4 closure: Gauss-Newton vs MIGRAD, with and without gradients.

    gn          Gauss-Newton / trust-region-reflective.  Full FORWARD Jacobian (one JVP per dial),
                normal equations -- second-order information for free since the model is a sum of squares.
    migrad+g    MIGRAD driven by the REVERSE-mode gradient.  One VJP per call regardless of dial count.
    migrad      MIGRAD with no gradient supplied, so it builds one by FINITE DIFFERENCES: ~2 x ndial
                extra cost-function calls per gradient.

Only the minimisation is timed: bank loading, closure-data construction and JIT compilation happen
before the clock starts, and every callable is warmed up on its exact argument shapes first.  MIGRAD is
CPU-only; the COST FUNCTION (the model evaluation over the resident events) runs on the GPU for all
three methods alike.  Fits are deterministic (Asimov data, fixed start), so repeats measure TIMING
spread only; the script asserts results are identical across repeats.

Usage:
    srun --jobid=<ID> --overlap python -m analysis.benchmarks.bench_minimizers [config] [--reps N] [--methods ...]
"""
from __future__ import annotations

from analysis._cli import results_dir, FLAT_PRIOR_SCALE, timed_log

import argparse
import sys
import time

import numpy as np

from analysis.benchmarks._shared import _bounds, _gn, _migrad

from adonis.reweight import knobs as K








def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="configs/fits/closure.yaml")
    ap.add_argument("--reps", type=int, default=3, help="timed repeats per method")
    ap.add_argument("--methods", default="gn,migrad+g,migrad")
    ap.add_argument("--tol", type=float, default=0.1, help="iminuit tol (EDM target = tol*errordef*1e-3)")
    ap.add_argument("--max-calls", type=int, default=20000, help="MIGRAD call ceiling")
    ap.add_argument("--nit", type=int, default=200, help="GN max_nfev")
    ap.add_argument("--sig-cap", type=int, default=0,
                    help="override banks.sig_cap (0 = use the config's). The differentiable objective "
                         "needs device room ON TOP of the resident banks, which a 11GB card does not "
                         "have at the config's 250k/sample; lowering this trades absolute times for a "
                         "comparison that still runs. The METHOD RATIOS are what the benchmark is for.")
    ap.add_argument("--jac-bench", action="store_true",
                    help="also time the DERIVATIVE objects themselves: MINUIT finite differences vs "
                         "ADoNIS forward/reverse autodiff, on the identical cost function")
    ap.add_argument("--out", default=str(results_dir() / "bench_minimizers.npz"))
    a = ap.parse_args(argv)
    methods = [s.strip() for s in a.methods.split(",") if s.strip()]

    log = timed_log()

    import jax
    log(f"jax {jax.__version__}  devices={jax.devices()}  default={jax.devices()[0].platform}")
    if jax.devices()[0].platform != "gpu":
        log("[warn] NOT on a GPU -- these timings are not the numbers you want")

    from adonis.fit.config import FitConfig
    from adonis.fit.fitters import parse_inject
    from analysis.campaign.stages.multisample import (MULTISAMPLE_NPZ, build_multisample_engine, fit_subset)
    from adonis.reweight.reweight_model import nominal_knobs

    cfg = FitConfig.load(a.config)
    log(f"config {cfg.path}  digest {cfg.digest()}")
    if a.sig_cap:
        import dataclasses
        log(f"[override] banks.sig_cap {cfg.banks.sig_cap:,} -> {a.sig_cap:,}  -- fewer events raises the "
            f"per-bin MC error and MASKS MORE BINS: a different, smaller fit. Method ratios transfer; "
            f"absolute times do not.")
        cfg = dataclasses.replace(cfg, banks=dataclasses.replace(cfg.banks, sig_cap=a.sig_cap))
    eng = build_multisample_engine(log, cfg)
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    subset = fit_subset(g, eng.pnames, cfg, log)
    truth, _ = parse_inject(cfg.inject_string(), nominal_knobs())
    eng.set_closure_data(truth)
    prior_true = eng.prior[np.array(subset, int)].copy()
    if cfg.fit.prior_scale != 1.0:
        eng.prior = eng.prior * (FLAT_PRIOR_SCALE if cfg.fit.prior_scale == 0.0 else cfg.fit.prior_scale)
    nd = len(subset)
    x0 = np.asarray(eng.th0)[np.array(subset, int)].copy()
    xt = np.asarray(truth)[np.array(subset, int)]
    lo, hi = _bounds(eng, subset)
    names_all = [eng.pnames[k] for k in subset]
    log(f"{nd} dials, start at nominal, truth {np.abs(xt - x0).max():.3f} away (max abs)")

    log("warm-up (JIT compile + first-touch), not timed")
    tw = time.perf_counter()
    f_only = eng.chi2_fn(subset); v0 = float(f_only(x0))
    vg = eng.chi2_grad_fn(subset); vv, gg = vg(x0)
    m0 = eng.model(np.asarray(eng.th0)); J0 = eng.jac(np.asarray(eng.th0), subset)
    log(f"  compiled in {time.perf_counter()-tw:.1f}s   chi2(start)={v0:.4g}  |grad|={np.linalg.norm(gg):.4g}"
        f"  model{m0.shape} jac{J0.shape}")
    assert abs(v0 - vv) < 1e-6 * max(1.0, abs(v0)), "value-only and value-and-grad objectives disagree"

    data, sigma = eng.data_sigma()
    okb = np.isfinite(sigma) & (sigma > 0)
    Wb = np.where(okb, 1.0 / np.where(okb, sigma, 1.0), 0.0)

    def _chi2_host(x):
        th = np.asarray(eng.th0).copy(); th[np.array(subset, int)] = x
        return float(np.sum(((eng.model(th) - data) * Wb) ** 2))

    th_t = np.asarray(eng.th0).copy(); th_t[np.array(subset, int)] = xt
    m_host, m_dev = eng.model(th_t), np.asarray(eng.model_jax(th_t))
    rel = np.abs(m_host - m_dev) / np.maximum(np.abs(m_host), 1e-300)
    log(f"  host vs device model at truth: max rel {rel.max():.3e} (bin {int(np.argmax(rel))}), "
        f"median {np.median(rel):.3e}")
    log(f"  chi2 at truth : host {_chi2_host(xt):.6e} | device {float(f_only(xt)):.6e}")
    log(f"  chi2 at start : host {_chi2_host(x0):.6e} | device {v0:.6e}")

    import os as _os
    from analysis.campaign.stages import multisample as _MS
    audit, fatal = {}, []

    audit["ADONIS_JAX_BINNING"] = _os.environ.get("ADONIS_JAX_BINNING", "<unset>")
    if not _MS._JAX_BIN:
        fatal.append("ADONIS_JAX_BINNING is off: GN bins on the host, MIGRAD's objective on the device. "
                     "Set ADONIS_JAX_BINNING=1 or the timings are not comparable.")

    audit["ADONIS_JAC_BATCH"] = _MS.JAC_BATCH
    audit["n_dispatch"] = int(np.ceil(nd / max(1, _MS.JAC_BATCH)))
    if _MS.JAC_BATCH < nd:
        fatal.append(f"JAC_BATCH={_MS.JAC_BATCH} < {nd} dials -> {audit['n_dispatch']} dispatches, "
                     f"one extra primal pass each. Set ADONIS_JAC_BATCH>={nd}.")

    audit["x64"] = bool(jax.config.jax_enable_x64)
    if not audit["x64"]:
        fatal.append("jax_enable_x64 is off: the objective is float32.")

    audit["platform"] = jax.devices()[0].platform
    if audit["platform"] != "gpu":
        fatal.append(f"running on {audit['platform']}, not gpu.")

    d_start = abs(_chi2_host(x0) - v0) / max(abs(v0), 1e-300)
    d_truth = abs(_chi2_host(xt) - float(f_only(xt)))
    audit["chi2_reldiff_start"], audit["chi2_absdiff_truth"] = d_start, d_truth
    if d_start > 1e-10:
        fatal.append(f"host and device chi2 differ by {d_start:.2e} at the start point.")

    _pw = 1.0 / eng.prior[np.array(subset, int)]
    pri_start = float(np.sum(((x0 - np.asarray(eng.th0)[np.array(subset, int)]) * _pw) ** 2))
    pri_truth = float(np.sum(((xt - np.asarray(eng.th0)[np.array(subset, int)]) * _pw) ** 2))
    audit["prior_block_at_truth"] = pri_truth
    audit["prior_frac_of_chi2"] = pri_truth / max(v0, 1e-300)
    if audit["prior_frac_of_chi2"] > 1e-6:
        fatal.append(f"prior block is {pri_truth:.3e}, {audit['prior_frac_of_chi2']:.2e} of chi2 -- "
                     f"GN and MIGRAD are minimising materially different objectives.")

    n_live = int(okb.sum()); n_dead = int((~okb).sum())
    audit["bins_live"], audit["bins_masked"] = n_live, n_dead
    _Wdev = np.concatenate([np.asarray(w_) for w_ in eng.chi2_data_dev()[1]])
    if int((_Wdev > 0).sum()) != n_live:
        fatal.append(f"live-bin count differs: host {n_live} vs device {int((_Wdev > 0).sum())}.")

    audit["start_at_nominal"] = bool(np.allclose(x0, np.asarray(eng.th0)[np.array(subset, int)]))
    audit["n_bounded_below"] = int(np.isfinite(lo).sum())

    log("  --- comparability audit ---")
    for k, v in audit.items():
        log(f"      {k:24s} {v}")
    if fatal:
        for m in fatal:
            log(f"  [FATAL] {m}")
        raise SystemExit("comparability audit failed; timings would not be apples-to-apples")
    log("  audit PASSED: both arms use the device binning path, one jacobian dispatch, float64, "
        "the same objective to 1e-10, the same live bins and the same start.")

    def _t(fn, n=5):
        fn(); ts = []
        for _ in range(n):
            s = time.perf_counter(); fn(); ts.append(time.perf_counter() - s)
        return float(np.median(ts))
    c_f = _t(lambda: float(f_only(x0)))
    c_vg = _t(lambda: float(vg(x0)[0]))
    c_J = _t(lambda: eng.jac(np.asarray(eng.th0), subset))
    c_m = _t(lambda: eng.model(np.asarray(eng.th0)))
    log(f"  per-call: chi2 {1e3*c_f:.1f} ms | chi2+grad {1e3*c_vg:.1f} ms ({c_vg/c_f:.2f}x)"
        f" | full jacobian {1e3*c_J:.1f} ms ({c_J/c_f:.1f}x) | model {1e3*c_m:.1f} ms")
    GN_TAIL = c_m + 2 * c_J

    if a.jac_bench:
        from iminuit import Minuit

        def _fd_grad(x, h=1e-4):
            """Central-difference gradient: the 2n calls MIGRAD makes when handed no derivative."""
            g = np.empty(len(x))
            for i in range(len(x)):
                xp = x.copy(); xp[i] += h
                xm = x.copy(); xm[i] -= h
                g[i] = (f_only(xp) - f_only(xm)) / (2 * h)
            return g

        t_vjp = _t(lambda: vg(x0)[1], n=5)
        t_fd = _t(lambda: _fd_grad(x0), n=3)
        t_jacf = c_J
        def _hesse_once():
            mh = Minuit(lambda *aa: f_only(np.asarray(aa, float)), *xt, name=names_all)
            mh.errordef = Minuit.LEAST_SQUARES
            for i, nm in enumerate(names_all):
                mh.limits[nm] = (None if not np.isfinite(lo[i]) else lo[i],
                                 None if not np.isfinite(hi[i]) else hi[i])
            t_ = time.perf_counter(); mh.hesse(); return time.perf_counter() - t_
        _hs = [_hesse_once() for _ in range(3)]
        t_hesse = float(np.median(_hs))
        log(f"  HESSE repeats: {['%.2f' % v for v in _hs]}s -> median {t_hesse:.2f}s")
        print(f"\n==== derivatives at {nd} dials (same objective, {jax.devices()[0].platform}) ====")
        print(f"{'object':>34} {'seconds':>9} {'obj-calls':>10} {'vs autodiff':>12}")
        print(f"{'MINUIT gradient (2n fin.diff.)':>34} {t_fd:9.3f} {2*nd:10d} {t_fd/t_vjp:11.1f}x")
        print(f"{'ADoNIS gradient (1 reverse VJP)':>34} {t_vjp:9.3f} {'~2':>10} {1.0:11.1f}x")
        print(f"{'MINUIT hessian (HESSE, ~2n^2)':>34} {t_hesse:9.3f} {'~'+str(2*nd*nd):>10} "
              f"{t_hesse/t_jacf:11.1f}x")
        print(f"{'ADoNIS jacobian (n forward JVPs)':>34} {t_jacf:9.3f} {'~'+str(nd):>10} {1.0:11.1f}x"
              f"   -> full {len(data)}x{nd} residual jacobian; J^T J is the GN hessian, no extra cost")

    res = {}
    for meth in methods:
        ts, ns, cs, xs = [], [], [], []
        for r in range(a.reps):
            if meth == "gn":
                x, n, c, dt = _gn(eng, subset, cfg.fit.minimizer.newton_tol, a.nit)
            elif meth == "migrad+g":
                x, n, c, dt = _migrad(eng, subset, x0, lo, hi, True, a.tol, a.max_calls, f_only, vg)
            elif meth == "migrad":
                x, n, c, dt = _migrad(eng, subset, x0, lo, hi, False, a.tol, a.max_calls, f_only, vg)
            else:
                raise SystemExit(f"unknown method {meth}")
            ts.append(dt); ns.append(n); cs.append(c); xs.append(x)
            log(f"  {meth:9s} rep {r+1}/{a.reps}: {dt:8.2f}s  value={n[0]:5d} deriv={n[1]:4d}"
                f"  chi2={c:.3e}"
                f"  max|x-truth|/prior={np.abs((x - xt) / prior_true).max():.2e}")
        xs = np.array(xs)
        tolx = 1e-6 * prior_true
        if a.reps > 1 and not np.all(np.abs(xs - xs[0]) <= tolx):
            log(f"  [warn] {meth}: repeats differ beyond GPU round-off "
                f"(max {np.abs(xs - xs[0]).max():.2e})")
        res[meth] = dict(t=np.array(ts), n=np.array(ns), chi2=np.array(cs), x=xs[0])

    print(f"\n==== minimiser benchmark ({nd} dials, {a.reps} reps, {jax.devices()[0].platform}) ====")
    print(f"{'method':>10} {'median s':>10} {'min-max s':>16} {'value':>6} {'deriv':>6} {'chi2':>11} "
          f"{'max bias/prior':>15}")
    corr = {m: (np.median(res[m]["t"]) - (GN_TAIL if m == "gn" else 0.0)) for m in methods}
    base = corr.get("gn")
    for meth in methods:
        d = res[meth]
        b = np.abs((d["x"] - xt) / prior_true).max()
        rel = "" if not base else f"   {corr[meth]/base:7.1f}x GN"
        nv, ng = int(np.median(d["n"][:, 0])), int(np.median(d["n"][:, 1]))
        print(f"{meth:>10} {np.median(d['t']):10.2f} {d['t'].min():7.2f}-{d['t'].max():<8.2f}"
              f" {nv:6d} {ng:6d} {np.median(d['chi2']):11.3e} {b:15.2e}{rel}")
    print(f"\nminimisation only (GN less its {GN_TAIL:.2f}s post-fit covariance/diagnostic tail): "
          + "  ".join(f"{m} {corr[m]:.2f}s" for m in methods))
    _acc = {m: float(np.abs((res[m]["x"] - xt) / prior_true).max()) for m in methods}
    print("\nSTOPPING RULES DIFFER -- times are each method to ITS OWN criterion, not to equal accuracy:")
    for m in methods:
        print(f"    {m:>10}  max|dtheta|/prior {_acc[m]:.3e}"
              + ("   <- reference" if m == "gn" else
                 f"   ({_acc[m]/max(_acc.get('gn', 1e-30), 1e-30):.1e}x GN's residual error)"))

    np.savez(a.out, methods=np.array(methods, dtype=object), subset=np.array(subset), truth=xt,
             per_call_chi2=c_f, per_call_chi2grad=c_vg, per_call_jac=c_J, per_call_model=c_m, prior_true=prior_true,
             gn_tail=GN_TAIL, platform=jax.devices()[0].platform, reps=a.reps,
             **{f"{m}_{k}": res[m][k] for m in methods for k in ("t", "n", "chi2", "x")})
    log(f"[out] {a.out}")


if __name__ == "__main__":
    sys.exit(main())

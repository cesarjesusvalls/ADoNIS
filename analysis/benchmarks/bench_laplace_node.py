"""Cost of the Laplace/Occam log-det correction: autodiff vs MINUIT, timed at a real profile node.

autodiff gets V = (J^T W J + P)^-1 as a by-product of the inner Gauss-Newton fit (one cheap
decomposition, nothing extra evaluated).  MINUIT's defensible route is m.hesse() (~2(n-1)^2 objective
evaluations), since m.covariance after migrad() is only an approximate BFGS-style covariance.  Measured
at a real node (one dial fixed away from the best fit, the rest re-minimised), not by scaling a
free-parameter number, since fixing a dial changes both the dimension and the point the Hessian is
taken at.

Usage:
    srun --jobid=<ID> --overlap python -m analysis.benchmarks.bench_laplace_node [--sig-cap N] [--nodes K]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from analysis.benchmarks.bench_minimizers import _bounds

N_NODES_1D = 221
N_NODES_2D = 7938


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="configs/fits/sec4_P1.yaml")
    ap.add_argument("--sig-cap", type=int, default=60000)
    ap.add_argument("--nodes", type=int, default=4, help="how many scan nodes to time")
    ap.add_argument("--offset", type=float, default=1.0, help="node offset from the BFP, in sigma_post")
    ap.add_argument("--tol", type=float, default=0.1)
    ap.add_argument("--nit", type=int, default=200)
    ap.add_argument("--out", default="output/altgen/bench_laplace_node.npz")
    a = ap.parse_args(argv)

    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    import dataclasses
    import jax
    from iminuit import Minuit
    log(f"jax {jax.__version__}  devices={jax.devices()}")

    from adonis.fit.config import FitConfig
    from adonis.fit.fitters import parse_inject, trf_fit
    from analysis.campaign.stages.multisample import (MULTISAMPLE_NPZ, build_multisample_engine, fit_subset)
    from adonis.reweight.reweight_model import nominal_knobs

    cfg = FitConfig.load(a.config)
    cfg = dataclasses.replace(cfg, banks=dataclasses.replace(cfg.banks, sig_cap=a.sig_cap))
    eng = build_multisample_engine(log, cfg)
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    subset = fit_subset(g, eng.pnames, cfg, log)
    truth, _ = parse_inject(cfg.inject_string(), nominal_knobs())
    eng.set_closure_data(truth)
    if cfg.fit.prior_scale != 1.0:
        eng.prior = eng.prior * (1e6 if cfg.fit.prior_scale == 0.0 else cfg.fit.prior_scale)
    n = len(subset)
    idx = np.array(subset, int)
    log(f"{n} dials; timing {a.nodes} nodes at {a.offset} sigma from the BFP")

    th_b, V_b, *_ = trf_fit(eng, subset, "bfp", nit=a.nit)
    spost = np.sqrt(np.abs(np.diag(V_b)))
    log(f"BFP done; sigma_post {np.round(spost, 4)}")

    rows = []
    for k in range(min(a.nodes, n)):
        free = [s for j, s in enumerate(subset) if j != k]
        pin = float(th_b[idx[k]] + a.offset * spost[k])
        th_i = th_b.copy(); th_i[idx[k]] = pin

        t1 = time.perf_counter()
        th_n, V_n, *_ = trf_fit(eng, free, "node", nit=a.nit, th_init=th_i)
        t_fit = time.perf_counter() - t1
        t2 = time.perf_counter()
        sign, ld = np.linalg.slogdet(V_n)
        t_logdet = time.perf_counter() - t2

        f_only = eng.chi2_fn(subset)
        f_only(th_b[idx])
        names = [eng.pnames[s] for s in subset]
        lo, hi = _bounds(eng, subset)

        def _hesse_once():
            m = Minuit(lambda *aa: f_only(np.asarray(aa, float)), *th_n[idx], name=names)
            m.errordef = Minuit.LEAST_SQUARES
            for i, nm in enumerate(names):
                m.limits[nm] = (None if not np.isfinite(lo[i]) else lo[i],
                                None if not np.isfinite(hi[i]) else hi[i])
            m.fixed[names[k]] = True
            t_ = time.perf_counter(); m.hesse(); return time.perf_counter() - t_, m

        hs = []
        for _ in range(3):
            dt_, m_ = _hesse_once(); hs.append(dt_)
        t_hesse = float(np.median(hs))
        rows.append((k, t_fit, t_logdet, t_hesse))
        log(f"  node {k} ({eng.pnames[idx[k]]}): inner fit {t_fit:6.2f}s | logdet {1e6*t_logdet:7.1f}us "
            f"| HESSE {t_hesse:6.2f}s  {['%.2f' % v for v in hs]}")

    R = np.array([(r[1], r[2], r[3]) for r in rows])
    t_fit, t_ld, t_he = np.median(R[:, 0]), np.median(R[:, 1]), np.median(R[:, 2])
    print(f"\n==== Laplace cost per profile node ({n} dials, {n-1} free, "
          f"{10*a.sig_cap:,} events, {jax.devices()[0].platform}) ====")
    print(f"  inner fit (both routes pay this)      {t_fit:8.2f} s")
    print(f"  autodiff: log det of the fit's own V  {1e6*t_ld:8.1f} us   ({t_ld/t_fit*100:.5f}% of the fit)")
    print(f"  MINUIT  : hesse() on {n-1} free dials  {t_he:8.2f} s   ({t_he/t_fit:.2f}x the fit itself)")
    print(f"\n  extrapolated EXTRA cost of the MINUIT route:")
    for tag, nn in (("1-D profile", N_NODES_1D), ("2-D corner", N_NODES_2D)):
        print(f"    {tag:>12} ({nn:5d} nodes): {nn*t_he/3600:8.2f} h   vs {nn*t_ld:8.3f} s by autodiff")
    print(f"  (at {10*a.sig_cap:,} events; production is 2,409,416, ~4x)")
    np.savez(a.out, rows=R, ndial=n, sig_cap=a.sig_cap, t_fit=t_fit, t_logdet=t_ld, t_hesse=t_he)
    log(f"[out] {a.out}")


if __name__ == "__main__":
    sys.exit(main())

"""Cost and accuracy of the Laplace/Occam log-det term: Gauss-Newton by-product vs autodiff exact Hessian
vs MINUIT HESSE, at a real profile node (one dial pinned away from the best fit, the rest re-minimised).

  A  GAUSS-NEWTON   V = (J^T W J)^-1 from the inner fit's own J -- free, but drops the Hessian's
                    residual term sum_b r_b d2m_b, exact only in the small-residual limit.
  B  AUTODIFF EXACT  forward-over-reverse Hessian-vector products, O(n) passes, exact.
  C  MINUIT HESSE    finite differences, ~2(n-1)^2 objective evaluations; the honest MINUIT route, since
                    the covariance migrad() accumulates is a path-dependent quasi-Newton approximation.

A log-det difference `d` shifts the Occam-corrected chi2 by `d`, so A and B are compared directly
against the Delta-chi2=1 scale rather than against each other's cost.

    srun ... python -m analysis.benchmarks.bench_laplace --sig-cap 60000 --nodes 4
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
    ap.add_argument("--sig-cap", type=int, default=60000)
    ap.add_argument("--ndials", type=int, default=17)
    ap.add_argument("--nodes", type=int, default=4, help="how many scan nodes to measure")
    ap.add_argument("--stage", choices=("bfp", "node"), default="bfp")
    ap.add_argument("--node", type=int, default=0, help="which scan node (stage=node)")
    ap.add_argument("--bfp", default="", help="npz written by stage=bfp")
    ap.add_argument("--offset", type=float, default=1.0, help="node offset from the BFP, in sigma_post")
    ap.add_argument("--noise-seed", type=int, default=1, help="0 = Asimov")
    ap.add_argument("--ref", default=str(results_dir() / "closure.npz"))
    ap.add_argument("--tol", type=float, default=0.1)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    log = timed_log()

    import dataclasses
    import jax
    from iminuit import Minuit

    from analysis.benchmarks._shared import _dial_order, freeze_sample, throw
    from adonis.fit.config import FitConfig
    from adonis.fit.fitters import parse_inject
    from adonis.fit.kernels import FitKernel
    from adonis.fit.minimizers import gn_fit
    from analysis.campaign.stages import multisample as MS
    from adonis.reweight.reweight_model import nominal_knobs

    log(f"jax {jax.__version__} devices={jax.devices()} x64={jax.config.jax_enable_x64}")
    cfg = FitConfig.load(a.config)
    cfg = dataclasses.replace(cfg, banks=dataclasses.replace(cfg.banks, sig_cap=a.sig_cap))
    eng = MS.build_multisample_engine(log, cfg)
    g = np.load(MS.MULTISAMPLE_NPZ, allow_pickle=True)
    subset = sorted(_dial_order(g, eng.pnames)[:a.ndials])
    freeze_sample(eng, np.load(a.ref, allow_pickle=True), log)

    th0 = np.asarray(eng.th0, float)
    idx = np.asarray(subset, int)
    truth_full, _ = parse_inject(cfg.inject_string(), nominal_knobs())
    truth = th0.copy(); truth[idx] = truth_full[idx]
    eng.set_closure_data(truth)
    if a.noise_seed:
        throw(eng, np.random.default_rng(1_000_000 + a.noise_seed))
        log(f"statistical noise thrown, seed {a.noise_seed} (the regime the correction is used in)")
    if cfg.fit.prior_scale != 1.0:
        eng.prior = eng.prior * (FLAT_PRIOR_SCALE if cfg.fit.prior_scale == 0.0 else cfg.fit.prior_scale)

    if a.stage == "bfp":
        kern = FitKernel(eng, subset)
        kern.warmup(which=("residuals", "jac"))
        x0 = th0[idx].copy()
        if "Eb_shift" in kern.pnames:
            x0[kern.pnames.index("Eb_shift")] = 2.0
        r_bfp = gn_fit(kern, x0, max_nfev=200, gtol=cfg.fit.minimizer.gtol)
        Vb = kern.covariance_gn(r_bfp.J)
        spost = np.sqrt(np.abs(np.diag(Vb)))
        log(f"BFP: chi2 {r_bfp.chi2:.4f}, {r_bfp.nfev} evals, sigma_post {np.round(spost, 4)}")
        kern.warmup(which=("hess",))
        H = kern.hessian(r_bfp.x)
        Vex = np.linalg.pinv(0.5 * H, rcond=1e-12)
        _, ld_gn = np.linalg.slogdet(Vb)
        _, ld_ex = np.linalg.slogdet(Vex)
        rel = np.max(np.abs(0.5 * H - (r_bfp.J.T @ r_bfp.J))) / np.max(np.abs(r_bfp.J.T @ r_bfp.J))
        log(f"CONTROL at the {'ASIMOV ' if not a.noise_seed else ''}BFP (chi2={r_bfp.chi2:.4g}): "
            f"log det V  GN {ld_gn:+.8f} | exact {ld_ex:+.8f} | diff {ld_gn-ld_ex:+.3e}")
        log(f"  max|H/2 - J^T J| / max|J^T J| = {rel:.3e}"
            + ("   <- must be ~0: the residual is zero here" if not a.noise_seed else
               "   (residual is NOT zero here, so a difference is expected)"))
        out = a.out or str(results_dir() / f"laplace_bfp_N{a.sig_cap}_s{a.noise_seed}.npz")
        np.savez(out, xb=r_bfp.x, spost=spost, subset=np.array(subset), chi2=r_bfp.chi2,
                 ld_gn_bfp=ld_gn, ld_exact_bfp=ld_ex, hess_rel=rel)
        log(f"[out] {out}")
        return

    z = np.load(a.bfp, allow_pickle=True)
    xb, spost = np.asarray(z["xb"], float), np.asarray(z["spost"], float)
    assert list(np.asarray(z["subset"], int)) == subset, "bfp file is for a different dial subset"
    rows = []
    for k in [a.node]:
        kdial = subset[k]
        pin = float(xb[k] + a.offset * spost[k])
        free = [s for s in subset if s != kdial]
        th_node = th0.copy(); th_node[idx] = xb; th_node[kdial] = pin
        kn = FitKernel(eng, free, th_fixed=th_node)
        kn.warmup(which=("residuals", "chi2", "grad", "jac", "hess"))
        xn0 = np.array([xb[j] for j, s in enumerate(subset) if s != kdial])

        rn = gn_fit(kn, xn0, max_nfev=200, gtol=cfg.fit.minimizer.gtol)
        xn = rn.x

        kn.reset_counts()
        t1 = time.perf_counter()
        Va = kn.covariance_gn(rn.J)
        sa, lda = np.linalg.slogdet(Va)
        ta, pa = time.perf_counter() - t1, kn.event_passes()

        kn.reset_counts()
        t1 = time.perf_counter()
        Vb_ex = kn.covariance_exact(xn)
        sb, ldb = np.linalg.slogdet(Vb_ex)
        tb, pb = time.perf_counter() - t1, kn.event_passes()

        lo, hi = kn.bounds()
        kn.reset_counts()
        names = list(kn.pnames)
        m = Minuit(lambda *aa: kn.chi2(np.asarray(aa, float)), *xn, name=names)
        m.errordef = Minuit.LEAST_SQUARES
        for i, nm in enumerate(names):
            m.limits[nm] = (None if not np.isfinite(lo[i]) else lo[i],
                            None if not np.isfinite(hi[i]) else hi[i])
        t1 = time.perf_counter()
        m.hesse()
        tc, pc = time.perf_counter() - t1, kn.event_passes()
        nc = kn.counts["chi2"]
        Vc = np.asarray(m.covariance)
        sc, ldc = np.linalg.slogdet(Vc)

        rows.append(dict(k=k, dial=eng.pnames[kdial], pin=pin, chi2=rn.chi2, nfree=kn.n,
                         ld_gn=lda, ld_exact=ldb, ld_hesse=ldc,
                         t_gn=ta, t_exact=tb, t_hesse=tc,
                         p_gn=pa, p_exact=pb, p_hesse=pc, n_hesse_calls=nc))
        log(f"  node {k} ({eng.pnames[kdial]} pinned at {pin:.4f}): inner fit {rn.wall:.2f}s")
        log(f"    log det V:  GN {lda:+.6f} | exact {ldb:+.6f} | HESSE {ldc:+.6f}")
        log(f"    cost     :  GN {ta*1e6:.1f}us / {pa:.0f} passes | exact {tb:.3f}s / {pb:.0f} | "
            f"HESSE {tc:.2f}s / {pc:.0f} ({nc} calls)")

    R = rows
    print(f"\n==== Laplace/Occam factor per profile node "
          f"({len(subset)} dials, {len(subset)-1} free, {10*a.sig_cap:,} events, "
          f"{'noise' if a.noise_seed else 'asimov'}) ====")
    print(f"{'node':>18} {'logdet GN':>12} {'logdet exact':>13} {'logdet HESSE':>13} "
          f"{'GN-exact':>10} {'HESSE-exact':>12}")
    for r in R:
        print(f"{r['dial']:>18} {r['ld_gn']:12.6f} {r['ld_exact']:13.6f} {r['ld_hesse']:13.6f} "
              f"{r['ld_gn']-r['ld_exact']:10.2e} {r['ld_hesse']-r['ld_exact']:12.2e}")
    dga = np.abs([r["ld_gn"] - r["ld_exact"] for r in R])
    dha = np.abs([r["ld_hesse"] - r["ld_exact"] for r in R])
    print(f"\n  A difference d in log det V shifts the Occam-corrected chi2 by d, so compare with 1.")
    print(f"    |GN - exact|    median {np.median(dga):.3e}  max {dga.max():.3e}")
    print(f"    |HESSE - exact| median {np.median(dha):.3e}  max {dha.max():.3e}")

    print(f"\n{'route':>28} {'seconds':>10} {'event-passes':>14} {'obj calls':>11}")
    print(f"{'A  Gauss-Newton (by-product)':>28} {np.median([r['t_gn'] for r in R]):10.6f} "
          f"{np.median([r['p_gn'] for r in R]):14.0f} {0:11d}")
    print(f"{'B  exact hessian (autodiff)':>28} {np.median([r['t_exact'] for r in R]):10.3f} "
          f"{np.median([r['p_exact'] for r in R]):14.0f} {0:11d}")
    print(f"{'C  MINUIT HESSE (2n^2 f.d.)':>28} {np.median([r['t_hesse'] for r in R]):10.3f} "
          f"{np.median([r['p_hesse'] for r in R]):14.0f} "
          f"{int(np.median([r['n_hesse_calls'] for r in R])):11d}")

    N1D, N2D = 221, 7938
    tg, te, th_ = (np.median([r[k] for r in R]) for k in ("t_gn", "t_exact", "t_hesse"))
    pg, pe, ph = (np.median([r[k] for r in R]) for k in ("p_gn", "p_exact", "p_hesse"))
    print(f"\n  over the real scans (extra cost ON TOP of the inner fits, which all routes pay):")
    print(f"{'scan':>14} {'A seconds':>12} {'B seconds':>12} {'C hours':>10} "
          f"{'A passes':>10} {'B passes':>10} {'C passes':>12}")
    for tag, nn in (("1-D profile", N1D), ("2-D corner", N2D)):
        print(f"{tag:>14} {nn*tg:12.4f} {nn*te:12.2f} {nn*th_/3600:10.2f} "
              f"{nn*pg:10.0f} {nn*pe:10.0f} {nn*ph:12.0f}")

    out = a.out or (str(results_dir() / f"bench_laplace_N{a.sig_cap}_s{a.noise_seed}_k{a.node}.npz"))
    np.savez(out, rows=np.array(R, dtype=object), sig_cap=a.sig_cap, ndials=a.ndials,
             noise_seed=a.noise_seed, n_nodes_1d=N1D, n_nodes_2d=N2D)
    log(f"[out] {out}")


if __name__ == "__main__":
    sys.exit(main())

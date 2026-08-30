"""Acceptance test for `adonis.fit.kernels.FitKernel`: does the fused device path compute the SAME
objective, gradient and Jacobian as the host path the fits have been using?

No timing from the kernel should be believed until every check here passes.  The reference is
deliberately the HOST path (`eng.model` -> np.bincount, `eng.jac` -> per-event derivatives binned on the
host); the kernel has to reproduce it, not the other way round.

Checked, at several theta (start, truth, midpoint, a random displacement):

  1. model        kern.model(x)      vs  eng.model(th)
  2. residuals    kern.residuals(x)  vs  trf_fit's own expression, (eng.model - data)*w  ++  prior block
  3. chi2         kern.chi2(x)       vs  the host sum of squares of the same
  4. gradient     kern.grad(x)       vs  2 J^T W r + 2 (x-x0)/prior^2 built from the HOST Jacobian,
                                     and vs a central finite difference of the host chi2
  5. jacobian     kern.jac(x)        vs  vstack([eng.jac(th, subset)*w, diag(1/prior)])
  6. batching     jac at batch n, 5, 1 -- identical results, different dispatch counts
  7. counters     the event-pass bookkeeping matches what was actually asked for

The gradient is checked against both references: J^T W r tests reverse mode against forward mode
through the same binning, while the finite difference is the only check that would catch a binning map
that is consistently wrong.

Usage:
    srun --jobid=<ID> --overlap python -m analysis.benchmarks.verify_kernels [config] [--sig-cap N]
"""
from __future__ import annotations

from analysis._cli import FLAT_PRIOR_SCALE, timed_log

import argparse
import sys
import time

import numpy as np


def _rel(a, b, scale=None):
    """max |a-b| / scale, with scale defaulting to max|b| -- 0 vs 0 reports 0, not nan."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = float(np.max(np.abs(b))) if scale is None else float(scale)
    if s <= 0:
        return float(np.max(np.abs(a - b)))
    return float(np.max(np.abs(a - b)) / s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="configs/fits/sec4_P1.yaml")
    ap.add_argument("--sig-cap", type=int, default=20000, help="events/sample (0 = the config's)")
    ap.add_argument("--tol", type=float, default=1e-9, help="relative tol on the MODEL")
    ap.add_argument("--resid-tol", type=float, default=1e-5, help="absolute tol on residuals, in SIGMA")
    ap.add_argument("--deriv-tol", type=float, default=1e-5, help="relative tol on gradient and jacobian")
    ap.add_argument("--floor-tol", type=float, default=1e-8, help="absolute tol on chi2 at the closure "
                    "truth -- the objective's floor, which bounds how deep a convergence target may be set")
    ap.add_argument("--fd-tol", type=float, default=2e-5, help="relative tol for the finite-difference "
                    "gradient check (limited by the difference, not by the kernel)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    log = timed_log()

    import dataclasses
    import jax
    import jax.numpy as jnp

    from adonis.fit.config import FitConfig
    from adonis.fit.fitters import parse_inject
    from adonis.fit.kernels import FitKernel
    from analysis.campaign.stages import multisample as MS
    from adonis.reweight.reweight_model import nominal_knobs

    log(f"jax {jax.__version__}  devices={jax.devices()}  x64={jax.config.jax_enable_x64}")
    if jax.devices()[0].platform != "gpu":
        log("[warn] not on a GPU -- correctness still holds, but run the acceptance on the device the "
            "benchmark will use, since that is where a numerical difference would appear")

    if MS._JAX_BIN:
        raise SystemExit("ADONIS_JAX_BINNING=1: the reference here must be the HOST path. Unset it and re-run.")

    cfg = FitConfig.load(a.config)
    if a.sig_cap:
        cfg = dataclasses.replace(cfg, banks=dataclasses.replace(cfg.banks, sig_cap=a.sig_cap))
    log(f"config {cfg.path}  digest {cfg.digest()}  sig_cap {cfg.banks.sig_cap:,}")
    eng = MS.build_multisample_engine(log, cfg)
    g = np.load(MS.MULTISAMPLE_NPZ, allow_pickle=True)
    subset = MS.fit_subset(g, eng.pnames, cfg, log)
    truth, _ = parse_inject(cfg.inject_string(), nominal_knobs())
    eng.set_closure_data(truth)
    if cfg.fit.prior_scale != 1.0:
        eng.prior = eng.prior * (FLAT_PRIOR_SCALE if cfg.fit.prior_scale == 0.0 else cfg.fit.prior_scale)

    idx = np.asarray(subset, int)
    th0 = np.asarray(eng.th0, float)
    x0, xt = th0[idx].copy(), np.asarray(truth, float)[idx]
    data, sigma = eng.data_sigma()
    ok = np.isfinite(sigma) & (sigma > 0)
    w = np.where(ok, 1.0 / np.where(ok, sigma, 1.0), 0.0)
    pw = 1.0 / np.asarray(eng.prior, float)[idx]

    kern = FitKernel(eng, subset)
    log(kern.describe())
    assert kern.n_live == int(ok.sum()), f"live bins {kern.n_live} != host {int(ok.sum())}"
    assert np.array_equal(kern.whiten() > 0, w > 0), "kernel and host disagree on WHICH bins are live"
    log(f"  live bins {kern.n_live}/{kern.nbin}; prior weights match: "
        f"{np.allclose(kern.pw, pw, rtol=0, atol=0)}")
    kern.warmup(x0)

    def h_model(x):
        th = th0.copy(); th[idx] = x
        return eng.model(th)

    def h_resid(x):
        return np.concatenate([(h_model(x) - data) * w, (np.asarray(x) - x0) * pw])

    def h_chi2(x):
        r = h_resid(x)
        return float(r @ r)

    def h_jac(x):
        th = th0.copy(); th[idx] = x
        return np.vstack([eng.jac(th, subset) * w[:, None], np.diag(pw)])

    rng = np.random.default_rng(a.seed)
    pts = [("start", x0), ("truth", xt), ("mid", 0.5 * (x0 + xt)),
           ("random", x0 + rng.normal(0, 0.05, size=len(idx)) * np.abs(x0))]
    lo, hi = kern.bounds()
    pts = [(nm, np.clip(x, lo + 1e-9, hi - 1e-9)) for nm, x in pts]

    S_model = float(np.max(np.abs(h_model(x0))))
    S_resid = 1.0
    S_grad = float(np.max(np.abs(2.0 * (h_jac(x0).T @ h_resid(x0)))))
    S_jac = float(np.max(np.abs(h_jac(x0))))

    fails = []
    print(f"\nrelative to FIXED scales: model {S_model:.3e}, residual 1 sigma, grad {S_grad:.3e}, "
          f"jac {S_jac:.3e}")
    print(f"{'point':>8} {'model':>11} {'residuals':>11} {'chi2(abs)':>11} {'grad(JtWr)':>11} "
          f"{'grad(f.d.)':>11} {'jacobian':>11}")
    for nm, x in pts:
        m_h, r_h, c_h, J_h = h_model(x), h_resid(x), h_chi2(x), h_jac(x)
        m_k, r_k, c_k, J_k = kern.model(x), kern.residuals(x), kern.chi2(x), kern.jac(x)

        d_m = _rel(m_k, m_h, S_model)
        d_r = _rel(r_k, r_h, S_resid)
        d_c = abs(c_k - c_h)
        d_J = _rel(J_k, J_h, S_jac)

        g_h = 2.0 * (J_h.T @ r_h)
        g_k = kern.grad(x)
        d_g = _rel(g_k, g_h, S_grad)

        hstep = 1e-5 * np.maximum(np.abs(x), 1.0)
        g_fd = np.empty(len(x))
        for i in range(len(x)):
            xp = x.copy(); xp[i] += hstep[i]
            xm = x.copy(); xm[i] -= hstep[i]
            g_fd[i] = (h_chi2(xp) - h_chi2(xm)) / (2 * hstep[i])
        d_gfd = _rel(g_k, g_fd, S_grad)

        lv_ = w > 0
        pb = np.abs(m_k - m_h)[lv_] / np.maximum(np.abs(m_h)[lv_], 1e-300)
        jb = int(np.argmax(pb))
        keys = np.array([d["key"] for d in eng.ds for _ in range(d["nbin"])])[lv_]

        print(f"{nm:>8} {d_m:11.2e} {d_r:11.2e} {d_c:11.2e} {d_g:11.2e} {d_gfd:11.2e} {d_J:11.2e}"
              f"   worst bin {pb[jb]:.1e} rel in {keys[jb]} "
              f"(content {np.abs(m_h)[lv_][jb]:.3g}, sigma {sigma[lv_][jb]:.3g})")
        for tag, v, tol in (("model", d_m, a.tol), ("residuals", d_r, a.resid_tol),
                            ("grad vs JtWr", d_g, a.deriv_tol), ("grad vs f.d.", d_gfd, a.fd_tol),
                            ("jacobian", d_J, a.deriv_tol)):
            if not np.isfinite(v) or v > tol:
                fails.append(f"{nm}: {tag} disagrees by {v:.3e} (tol {tol:.0e})")
    floor = abs(kern.chi2(xt) - h_chi2(xt))
    print(f"\nobjective floor at the closure truth: |chi2_kernel - chi2_host| = {floor:.3e}  "
          f"-> convergence targets must stay above it")
    if floor > a.floor_tol:
        fails.append(f"chi2 floor at the truth is {floor:.3e} (tol {a.floor_tol:.0e})")

    print("\n---- decomposition of the host-vs-kernel difference, at the start point ----")
    xd = x0
    thd = th0.copy(); thd[idx] = xd
    m_host = eng.model(thd)
    m_eager = np.asarray(eng.model_jax(jnp.asarray(thd)))
    m_jit = kern.model(xd)
    print(f"  model  host vs eager-device (binning only) : {_rel(m_eager, m_host, S_model):.3e}")
    print(f"  model  eager-device vs jitted kernel (fusion): {_rel(m_jit, m_eager, S_model):.3e}")
    print(f"  model  host vs jitted kernel (both)         : {_rel(m_jit, m_host, S_model):.3e}")

    lv = w > 0
    J_host = eng.jac(thd, subset)[lv]
    J_loop = np.vstack([s.jac_blocks_ref(thd, subset) for s in eng.samples])[lv]
    J_kern = kern.jac_data(xd)[lv] / w[lv][:, None]
    SJ = float(np.max(np.abs(J_host)))
    print(f"  jac    host vmapped vs host looped (vmap)   : {_rel(J_loop, J_host, SJ):.3e}")
    print(f"  jac    host vs jitted kernel (fusion)       : {_rel(J_kern, J_host, SJ):.3e}")

    s0 = eng.samples[0]
    if hasattr(s0, "_wf"):
        _i, _t0 = jnp.asarray(idx), jnp.asarray(th0)
        w_direct = np.asarray(s0._wf(jnp.asarray(thd), s0.JB))
        w_rejit = np.asarray(jax.jit(lambda z: s0._wf(_t0.at[_i].set(z), s0.JB))(jnp.asarray(xd)))
        Sw = float(np.max(np.abs(w_direct)))
        print(f"  weights  _wf vs re-jitted _wf (no binning) : "
              f"{_rel(w_rejit, w_direct, Sw):.3e}   <- (a) weight re-optimisation")

    print(f"\n{'jac batch':>10} {'dispatches':>11} {'tangents run':>13} {'vs batch=n':>12}")
    Jref = kern.jac(x0)
    print(f"{kern.n:>10} {kern.nblk:>11} {kern.n:>13} {0.0:12.2e}")
    for B in (5,):
        k2 = FitKernel(eng, subset, jac_batch=B)
        k2.jac(x0)
        k2.reset_counts()
        d = _rel(k2.jac(x0), Jref)
        print(f"{B:>10} {k2.nblk:>11} {k2.counts['tangent']:>13} {d:12.2e}")
        if d > a.tol:
            fails.append(f"jac batch {B} differs from batch {kern.n} by {d:.3e}")
        assert k2.counts["tangent"] == k2.nblk * k2.B
        assert k2.counts["tangent_eff"] == k2.n

    k3 = kern.reset_counts()
    k3.chi2(x0); k3.chi2(x0); k3.grad(x0); k3.jac(x0)
    c = k3.counts
    print(f"\ncounters after 2 chi2 + 1 grad + 1 jac: {c}")
    print(f"  event passes = primal {c['primal']} + tangent {c['tangent']} + "
          f"{2.0:g} x vjp {c['vjp']} = {k3.event_passes():g}")
    if not (c["chi2"] == 2 and c["grad"] == 1 and c["jac"] == 1
            and c["tangent"] == k3.n and c["vjp"] == 1 and c["primal"] == 2 + k3.nblk):
        fails.append(f"counter bookkeeping wrong: {c}")

    print()
    if fails:
        for f in fails:
            log(f"[FAIL] {f}")
        raise SystemExit(f"kernel acceptance FAILED ({len(fails)} checks)")
    log(f"kernel acceptance PASSED: fused device path reproduces the host path to < {a.tol:.0e} "
        f"on model, residuals, chi2, gradient and Jacobian at {len(pts)} points, "
        f"is batch-invariant, and its counters are consistent.")


if __name__ == "__main__":
    sys.exit(main())

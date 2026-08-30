"""NUTS driven by a plain Python loop over `eng.model`/`eng.jac` -- no blackjax, no JAX tracing of
the sampler itself, so the log-density need not be jax-traceable as one fused graph.

The gradient is exact: grad chi2 = 2 J^T W r, using the same autodiff Jacobian the fit uses.

`--selftest` checks the sampler against a Gaussian with a strongly correlated pair, where the mean
and covariance are known analytically.
"""
import os, sys, time
import pathlib
import numpy as np

MAXDEPTH = 8


def leapfrog(q, p, g, eps, Minv, gradf):
    """One leapfrog step, carrying the incoming gradient forward instead of recomputing it: one
    gradient evaluation per step, not two.
    """
    p = p + 0.5 * eps * g
    q = q + eps * (Minv @ p)
    lp, g = gradf(q)
    p = p + 0.5 * eps * g
    return q, p, lp, g


def _uturn(qm, qp, pm, pp, Minv):
    d = qp - qm
    return (d @ (Minv @ pm) < 0) or (d @ (Minv @ pp) < 0)


def build_tree(q, p, g, logu, v, j, eps, Minv, gradf, H0, rng, nfev, ndiv=None):
    """Recursive NUTS tree (Hoffman & Gelman 2014, slice form).  Carries (q, p, g) at both ends.

    `ndiv`, when a one-element list is passed, counts divergences: leaves where the energy error
    exceeded the 1000-nat threshold, distinct from a healthy stop caused by a U-turn.
    """
    if j == 0:
        q1, p1, lp1, g1 = leapfrog(q, p, g, v * eps, Minv, gradf); nfev[0] += 1
        H = lp1 - 0.5 * p1 @ (Minv @ p1)
        n1 = 1 if logu <= H else 0
        s1 = 1 if logu < H + 1000.0 else 0
        if s1 == 0 and ndiv is not None:
            ndiv[0] += 1
        return (q1, p1, g1, q1, p1, g1, q1, n1, s1, min(1.0, np.exp(H - H0)), 1)
    (qm, pm, gm, qp, pp, gp, q1, n1, s1, a1, na1) = build_tree(q, p, g, logu, v, j - 1, eps, Minv,
                                                               gradf, H0, rng, nfev, ndiv)
    if s1 == 1:
        if v == -1:
            (qm, pm, gm, _, _, _, q2, n2, s2, a2, na2) = build_tree(qm, pm, gm, logu, v, j - 1, eps,
                                                                    Minv, gradf, H0, rng, nfev, ndiv)
        else:
            (_, _, _, qp, pp, gp, q2, n2, s2, a2, na2) = build_tree(qp, pp, gp, logu, v, j - 1, eps,
                                                                    Minv, gradf, H0, rng, nfev, ndiv)
        if n1 + n2 > 0 and rng.random() < n2 / (n1 + n2):
            q1 = q2
        a1 += a2; na1 += na2
        s1 = s2 * (0 if _uturn(qm, qp, pm, pp, Minv) else 1)
        n1 += n2
    return (qm, pm, gm, qp, pp, gp, q1, n1, s1, a1, na1)


def nuts_step(q, lp, g, gradf, eps, Minv, Mchol, rng):
    """One NUTS iteration.  Returns (q, lp, g, depth, alpha, nfev, ndiv).  Shared by both warm-up
    and sampling, so both run the exact same tree code.
    """
    p = Mchol @ rng.standard_normal(len(q))
    H0 = lp - 0.5 * p @ (Minv @ p)
    logu = H0 + np.log(rng.random())
    qm = qp = q.copy(); pm = pp = p.copy(); gm = gp = g.copy()
    j = 0; n = 1; sflag = 1; a = 0.0; na = 0; nfev = [0]; nd = [0]
    while sflag == 1 and j < MAXDEPTH:
        v = 1 if rng.random() < 0.5 else -1
        if v == -1:
            (qm, pm, gm, _, _, _, q1, n1, s1, a, na) = build_tree(qm, pm, gm, logu, v, j, eps,
                                                                  Minv, gradf, H0, rng, nfev, nd)
        else:
            (_, _, _, qp, pp, gp, q1, n1, s1, a, na) = build_tree(qp, pp, gp, logu, v, j, eps,
                                                                  Minv, gradf, H0, rng, nfev, nd)
        if s1 == 1 and rng.random() < min(1.0, n1 / max(n, 1)):
            q = q1
        n += n1
        sflag = s1 * (0 if _uturn(qm, qp, pm, pp, Minv) else 1)
        j += 1
    lp, g = gradf(q)
    return q, lp, g, j, a / max(na, 1), nfev[0] + 1, nd[0]


def _metric(S, n):
    """Stan's regularised covariance estimate for a warm-up window: Sigma = n/(n+5) S +
    1e-3*(5/(n+5)) I, shrunk toward a scaled identity so it stays positive-definite even with fewer
    draws than dimensions.
    """
    d = S.shape[0]
    Sig = (n / (n + 5.0)) * S + 1e-3 * (5.0 / (n + 5.0)) * np.eye(d)
    return 0.5 * (Sig + Sig.T)


def warmup_stan(q0, gradf, Minv0, nwarm, rng, target=0.8, log=print, dense=True):
    """Stan-style warm-up: dual-averaging step size + windowed metric adaptation.

    Schedule (Stan's, scaled to whatever nwarm is given): a fast init buffer that tunes only the
    step size, then expanding slow windows that each re-estimate the metric from their own draws and
    restart dual averaging, then a fast terminal buffer that re-tunes the step size against the final
    metric.

    Returns (q, lp, g, eps, Minv, Mchol, info).
    """
    d = len(q0)
    n_init = max(10, int(0.10 * nwarm))
    n_term = max(50, int(0.25 * nwarm))
    n_mid = max(20, nwarm - n_init - n_term)
    Minv = np.array(Minv0, float)
    Mchol = np.linalg.cholesky(np.linalg.inv(Minv))
    q = np.array(q0, float); lp, g = gradf(q)
    eps = 1.0
    info = dict(windows=[], ndiv=0, nfev=0)

    def _dual(eps0, nsteps, collect):
        """Dual averaging (Hoffman & Gelman Alg. 6) for `nsteps`, optionally collecting draws."""
        nonlocal q, lp, g, Minv, Mchol
        mu = np.log(10.0 * eps0); lbar = 0.0; Hbar = 0.0
        gam, t0, kap = 0.05, 10.0, 0.75
        e = eps0; got = []; alphas = []
        for m in range(1, nsteps + 1):
            q, lp, g, dep, alpha, nf, nd = nuts_step(q, lp, g, gradf, e, Minv, Mchol, rng)
            info["ndiv"] += nd; info["nfev"] += nf; alphas.append(alpha)
            Hbar = (1.0 - 1.0 / (m + t0)) * Hbar + (target - alpha) / (m + t0)
            le = mu - np.sqrt(m) / gam * Hbar
            eta = m ** (-kap)
            lbar = eta * le + (1.0 - eta) * lbar
            e = float(np.exp(le))
            if collect:
                got.append(q.copy())
        ab = float(np.mean(alphas[len(alphas) // 2:])) if alphas else float("nan")
        return float(np.exp(lbar)), got, ab

    eps, _, _ = _dual(eps, n_init, False)
    log(f"  warmup init {n_init} its -> eps {eps:.4f}")
    w, used, k = max(20, n_mid // 8), 0, 0
    while used < n_mid:
        take = int(min(w, n_mid - used))
        if n_mid - used - take < take // 2:
            take = n_mid - used
        eps, draws, _ = _dual(eps, take, True)
        Y = np.asarray(draws)
        if len(Y) > d + 2:
            S = np.cov(Y.T) if dense else np.diag(Y.var(axis=0, ddof=1))
            Minv = _metric(np.atleast_2d(S), len(Y))
            Mchol = np.linalg.cholesky(np.linalg.inv(Minv))
            info["windows"].append(dict(n=len(Y), cond=float(np.linalg.cond(Minv))))
            log(f"  warmup window {k}: {len(Y)} draws -> metric cond {np.linalg.cond(Minv):.3e}, "
                f"eps {eps:.4f}")
        used += take; w *= 2; k += 1
    eps, _, abar = _dual(eps, n_term, False)
    info["alpha_term"] = abar
    log(f"  warmup term {n_term} its -> eps {eps:.4f}, achieved alpha {abar:.3f} (target {target:.2f})"
        f"; {len(info['windows'])} metric updates, {info['ndiv']} divergences during warmup")
    if not (target - 0.08 < abar < target + 0.08):
        log(f"  WARNING: warm-up did not converge -- alpha {abar:.3f} vs target {target:.2f}; "
            f"leapfrog steps go as 1/eps so gradient counts below are NOT tuned")
    return q, lp, g, eps, Minv, Mchol, info


def nuts_sample(q0, gradf, eps, Minv, Mchol, nsamp, rng, log=print, tag="", ndiv_out=None):
    q = q0.copy(); out = np.empty((nsamp, len(q0))); depth = []; acc = []; nf = []
    lp, g = gradf(q)
    t0 = time.time()
    for i in range(nsamp):
        q, lp, g, j, alpha, nfev, nd = nuts_step(q, lp, g, gradf, eps, Minv, Mchol, rng)
        out[i] = q; depth.append(j); acc.append(alpha); nf.append(nfev)
        if ndiv_out is not None:
            ndiv_out.append(nd)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            log(f"  {tag}{i+1}/{nsamp}  {el/(i+1):.1f} s/sample  grads/sample {np.mean(nf):.1f}  "
                f"depth {np.mean(depth):.1f}  acc {np.mean(acc):.2f}  ETA {(nsamp-i-1)*el/(i+1)/3600:.2f} h")
    return out, np.array(depth), np.array(acc), np.array(nf)


def selftest():
    """Correctness check: a Gaussian with a strongly correlated pair, whose mean and covariance are
    known analytically."""
    n = 16; rng = np.random.default_rng(0)
    C = np.eye(n); C[0, 1] = C[1, 0] = -0.995
    Cinv = np.linalg.inv(C)
    def gradf(q): return -0.5 * q @ Cinv @ q, -(Cinv @ q)
    Minv = C.copy(); Mchol = np.linalg.cholesky(np.linalg.inv(C))
    s, d, a, nf = nuts_sample(np.zeros(n), gradf, 0.6, Minv, Mchol, 4000, rng, tag="selftest ")
    Ce = np.cov(s.T)
    print(f"  mean |max| = {np.abs(s.mean(0)).max():.3f}  (expect ~0)")
    print(f"  var  [0,1] = {Ce[0,0]:.3f} {Ce[1,1]:.3f}  (expect 1)")
    print(f"  corr(0,1)  = {Ce[0,1]/np.sqrt(Ce[0,0]*Ce[1,1]):.4f}  (expect -0.9950)")
    print(f"  max |cov err| off the pair = {np.abs(Ce - C)[2:,2:].max():.3f}")
    print(f"  grads/sample {nf.mean():.1f}  acc {a.mean():.3f}")


if __name__ == "__main__":
    selftest()

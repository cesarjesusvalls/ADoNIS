"""NUTS with our own loop -- no blackjax, no JAX tracing of the sampler.

Why not blackjax: it jits its kernel, so the log-density must be jax-traceable, and both ways of
satisfying that fail here.  jax.pure_callback bounces device->host every call (measured 4.05 s against
0.83 s of real work).  A natively traceable model_jax fuses all 8 sub-samples into ONE graph and OOMs on
an 11 GB card -- eng.jac only fits because it is 8 SEPARATE jit calls with numpy in between.  Driving the
loop ourselves imposes neither constraint: this calls eng.model/eng.jac exactly as lm_fit does.

The gradient is unchanged and exact: grad chi2 = 2 J^T W r, J the same autodiff Jacobian the fit uses.

Correctness before physics: --selftest samples a 16-D Gaussian with a -0.995 correlated pair (the
M_A_res/S_Delta geometry) where mean and covariance are known analytically, and checks them.

Env: NUTS_SAMPLES, NUTS_WARMUP, NUTS_CHAIN, NUTS_MAXDEPTH, ADONIS_LABEL.
"""
import os, sys, time
import pathlib
import numpy as np

MAXDEPTH = 8      # rebound from cfg.stage("nuts")["max_depth"] in run_real()


def leapfrog(q, p, g, eps, Minv, gradf):
    """One leapfrog step, CARRYING the gradient forward: 1 gradient evaluation per step, not 2.

    The naive form recomputes grad at the current position before the half-kick, but that value is
    already known from the previous step.  Measured, the duplicate doubled the cost (20.7 s/sample
    against ~10 s expected from 11.8 grads x 0.843 s).
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

    `ndiv`, when a one-element list is passed, counts DIVERGENCES -- leaves where the energy error
    exceeded the 1000-nat threshold.  Without it a caller cannot distinguish a trajectory that stopped
    because it made a U-turn (healthy) from one that stopped because the integrator blew up (the step
    size is too large for the local geometry), and the second is exactly the failure that would quietly
    erode a gradient sampler's apparent advantage as the posterior gets harder.
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
    """ONE NUTS iteration.  Returns (q, lp, g, depth, alpha, nfev, ndiv).

    Extracted so the sampler and the warm-up run the SAME tree code -- two copies of a doubling
    recursion is how a warm-up ends up tuning a sampler that is not the one that then runs.
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
    """Stan's regularised covariance estimate for a warm-up window.

    Shrinks toward a scaled identity so a window with fewer draws than dimensions still yields a
    positive-definite metric: Sigma = n/(n+5) S + 1e-3 (5/(n+5)) I.
    """
    d = S.shape[0]
    Sig = (n / (n + 5.0)) * S + 1e-3 * (5.0 / (n + 5.0)) * np.eye(d)
    return 0.5 * (Sig + Sig.T)


def warmup_stan(q0, gradf, Minv0, nwarm, rng, target=0.8, log=print, dense=True):
    """Stan-style warm-up: dual-averaging step size + WINDOWED METRIC ADAPTATION.

    Why this exists.  A fixed metric taken from the Laplace covariance describes a posterior well only
    where that covariance describes it -- i.e. where the posterior is close to Gaussian.  On a curved or
    near-degenerate direction it does not, the integrator diverges, and the sampler answers by building
    ever deeper trees.  Measured on a three-way degenerate subset: 137 divergences and Rhat 1.0124, a
    convergence FAILURE, with a fixed metric.  Re-estimating the metric from the draws is the standard
    remedy, and this is that remedy.

    Schedule (Stan's, scaled to whatever nwarm is given): a fast init buffer tuning only the step size,
    then expanding slow windows each of which re-estimates the metric from its own draws and restarts
    dual averaging, then a fast terminal buffer that re-tunes the step size against the FINAL metric.

    Returns (q, lp, g, eps, Minv, Mchol, info).
    """
    d = len(q0)
    # THE TERMINAL BUFFER IS THE ONLY PLACE eps CONVERGES, so it gets a quarter of the budget.
    # With Stan's 10% split (60 its of 600) the step size never settled: a measured warm-up went
    # 0.4364 -> 0.4317 -> 0.2436 -> 0.4748 -> 0.3505 across windows, i.e. oscillating by 2x, and
    # sampling then ran at acceptance 0.93 against a 0.80 target at EVERY cell of the grid.  Too
    # small an eps is not a wrong answer, it is a slow one -- leapfrog steps go as 1/eps -- so the
    # whole NUTS arm was paying ~2x the gradients it needed and the sampler comparison read as a
    # property of the algorithm when it was a property of my warm-up schedule.
    #
    # Each window also restarts dual averaging with mu = log(10 eps0), which biases eps upward at
    # the start of every window; the returned exp(lbar) averages over that excursion, so a SHORT
    # window returns a value dominated by its own transient.  Only the terminal buffer runs against
    # a frozen metric, so only it can actually converge -- give it room to.
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
        # mean alpha over the LAST HALF: the first half is the mu = log(10 eps0) transient, and
        # averaging it in would report the schedule's excursion rather than where eps settled.
        ab = float(np.mean(alphas[len(alphas) // 2:])) if alphas else float("nan")
        return float(np.exp(lbar)), got, ab

    eps, _, _ = _dual(eps, n_init, False)
    log(f"  warmup init {n_init} its -> eps {eps:.4f}")
    # expanding slow windows: 25, 50, 100, ... capped so the total is n_mid
    w, used, k = max(20, n_mid // 8), 0, 0
    while used < n_mid:
        take = int(min(w, n_mid - used))
        if n_mid - used - take < take // 2:      # absorb a short tail into this window
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
    # ACHIEVED acceptance against the frozen final metric.  If this is not near `target`, eps did not
    # converge and every downstream gradient count is inflated -- so it is logged and stored, not
    # left to be inferred from the sampling-phase number after the fact.
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
    """16-D Gaussian with a -0.995 correlated pair: mean and covariance are known exactly."""
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


def run_real():
    t0 = time.time()
    def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    from adonis.fit.stages.multisample import build_multisample_engine, MULTISAMPLE_NPZ
    from adonis.fit.fitters import parse_inject
    from adonis.reweight.reweight_model import nominal_knobs
    CH = int(os.environ.get("NUTS_CHAIN", "0"))      # which chain -- per-job, set by the runner
    from adonis.fit.config import FitConfig
    cfg = FitConfig.load(os.environ.get("ADONIS_FIT_CONFIG", "configs/fits/sec4_P1.yaml"))
    eng = build_multisample_engine(log, cfg)
    g = np.load(MULTISAMPLE_NPZ, allow_pickle=True)
    sub = [int(i) for i in np.where(g["shrink"] < 0.5)[0]]
    # TRUTH: taken from PHYSFIT_INJECT.
    st = cfg.stage("nuts")
    LABEL, NS, NW = cfg.name, int(st["samples"]), int(st["warmup"])
    global MAXDEPTH
    MAXDEPTH = int(st.get("max_depth", 8))
    _inj = cfg.inject_string()
    if not _inj:
        # No silent fallback.  This used to read a truth16.txt in a session scratchpad, which (a) is not
        # reproducible outside that session and (b) held the OLD 16-dial truth, so once
        # res_axial_strength joined the fitted set the sampler would explore a posterior whose data was
        # generated at a different point than the closure it is compared against.
        raise SystemExit("PHYSFIT_INJECT is required: the truth the Asimov data is built at, e.g.\n"
                         "  PHYSFIT_INJECT='M_A_qe=1.12,M_A_res=0.85,...'")
    star, _ = parse_inject(_inj, nominal_knobs())
    log(f"truth: {_inj[:90]}...")
    eng.set_closure_data(star)
    data, sigma = eng.data_sigma()
    ok = np.isfinite(sigma) & (sigma > 0)
    W = np.where(ok, 1.0 / np.where(ok, sigma, 1.0) ** 2, 0.0)
    zc = np.load(f"output/altgen/{LABEL}.npz", allow_pickle=True)
    bfp = np.asarray(zc["fit_th"]); V = np.asarray(zc["fit_V"])
    _fs = [int(k) for k in zc["fit_sub"]] if "fit_sub" in zc.files else sub
    if _fs != list(sub):
        raise SystemExit(f"subset mismatch: gate-I gives {len(sub)} dials, {LABEL}.npz was fitted with "
                         f"{len(_fs)}.  The mass matrix would be built from the wrong covariance.")
    log(f"{len(sub)} dials: " + ",".join(eng.pnames[k] for k in sub))
    sp = np.sqrt(np.abs(np.diag(V))); idx = np.array(sub)
    NF = [0]

    # TRUNCATION to the physical region.  Without it the sampler is not sampling a posterior at all:
    # below E_b's wall the model CLAMPS, so chi2 is exactly flat, the density is flat and unbounded, the
    # posterior is improper and every marginal is undefined.  Measured on the untruncated run, 94.3% of
    # 6000 draws had E_b outside its range (down to -288 in physical units) -- and because those draws
    # still carry values for the other 15 dials, they corrupted ALL the marginals, not just E_b's.
    # Returning -inf outside the box makes NUTS reject the step, which is the correct treatment of a
    # hard physical boundary.
    from adonis.analysis import knobs as _K
    _lo = np.array([-np.inf if _K.phys_lo(eng.pnames[k]) is None else _K.phys_lo(eng.pnames[k])
                    for k in sub])
    _hi = np.array([np.inf if _K.phys_hi(eng.pnames[k]) is None else _K.phys_hi(eng.pnames[k])
                    for k in sub])
    NREJ = [0]

    def gradf(u):
        """log p and its gradient in sigma_post units about the BFP.  Exact: grad chi2 = 2 J^T W r."""
        th = bfp.copy(); th[idx] = bfp[idx] + sp * u
        if np.any(th[idx] < _lo) or np.any(th[idx] > _hi):
            NREJ[0] += 1
            return -np.inf, np.zeros(len(sub))       # outside the physical box
        NF[0] += 1
        r = eng.model(th) - data
        J = eng.jac(th, sub)
        return -0.5 * float(np.sum(W * r * r)), -(J.T @ (W * r)) * sp

    # mass matrix from the KNOWN fit covariance (in sigma_post units) -- no window adaptation needed,
    # and it absorbs the -0.995 pair, which is what keeps the tree shallow.
    D = np.diag(1.0 / sp)
    C = D @ V @ D
    C = 0.5 * (C + C.T) + 1e-9 * np.eye(len(sub))
    Minv = C
    Mchol = np.linalg.cholesky(np.linalg.inv(C))
    lp0, g0 = gradf(np.zeros(len(sub)))
    log(f"logp at BFP = {lp0:.6e}   |grad| = {np.linalg.norm(g0):.3e}")
    t = time.time()
    for _ in range(3): gradf(np.zeros(len(sub)))
    log(f"gradient cost {(time.time()-t)/3*1000:.0f} ms   (pure_callback route was 14000 ms)")

    rng = np.random.default_rng(20260809 + CH)
    eps = float(st.get("eps", 0.35))

    # RESUME (NUTS_RESUME=<npz>): continue an existing chain instead of starting a new one.  A Markov
    # chain's future depends only on its current state, so appending to it is exact -- PROVIDED the kernel
    # is unchanged.  The step size is therefore read back from the file rather than re-derived, and the
    # mass matrix is deterministic from fit_V, so both match the original run by construction.  Warmup is
    # skipped: re-warming would change eps and break stationarity of the pooled samples.
    RES = os.environ.get("NUTS_RESUME", "").strip()
    q_start = np.zeros(len(sub))
    if RES:
        zr = np.load(RES, allow_pickle=True)
        q_start = np.asarray(zr["u"])[-1].copy()
        eps = float(zr["eps"])
        NW = 0
        log(f"RESUMING from {RES}: {len(zr['u'])} existing samples, eps={eps:.4f} (warmup skipped)")

    if NW:
        log(f"warmup {NW} samples to tune eps (start {eps})")
        _, _, a, _ = nuts_sample(rng.standard_normal(len(sub)) * 0.1, gradf, eps, Minv, Mchol, NW, rng,
                                 log=log, tag="warm ")
        am = float(np.mean(a))
        eps *= (am / 0.8) ** 0.5 if am > 0 else 1.0
        log(f"warmup acceptance {am:.3f} -> eps {eps:.4f}")
    s, dep, acc, nf = nuts_sample(q_start, gradf, eps, Minv, Mchol, NS, rng, log=log, tag="")
    out = f"output/altgen/{LABEL}_nutsown_{CH}.npz"
    if RES:                       # never overwrite the run being extended
        out = f"output/altgen/{LABEL}_nutsown_{CH}_ext{os.environ.get('NUTS_EXT','1')}.npz"
    np.savez(out, u=s, bfp=bfp, sigma_post=sp, subset=sub,
             pnames=eng.pnames, truth=star, depth=dep, accept=acc, ngrad=nf, eps=eps,
             resumed_from=RES)
    log(f"[out] {out}  {NS} samples, "
        f"{nf.mean():.1f} grads/sample, acc {acc.mean():.3f}, {NF[0]} gradient calls, "
        f"{NREJ[0]} out-of-bounds rejections")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_real()

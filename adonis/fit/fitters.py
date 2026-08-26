"""The fitters: bound-constrained least squares, and the Gate-2 diagnostics built on them.

Parameters come from a config, not the environment.
"""
import os, time
import numpy as np

from adonis.analysis import knobs as _K   # PHYS_BOUND / clip_phys: one source of truth for bounds
from scipy import stats as sstats
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.reweight import bank_plot as BP, bank_reweight as BR
from adonis.reweight.reweight_model import nominal_knobs
from adonis.analysis.knobs import SPEC, theta_nominal



# ---- fitter constants ----------------------------------------------------------------------------- #
NIT = 12                # default iteration budget; callers pass cfg.fit.minimizer.max_nfev
STEP_SCALE = 1.0        # LM step scaling factor
F_RESP = 0.3            # responsive-bin threshold: |J_bk| * prior_k > F_RESP * sigma_b
HUBER_C = 1.345         # Huber tuning constant (95% efficiency at the Gaussian)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

def parse_inject(s, nom):
    th = theta_nominal(nom)
    inj = {}
    for tok in s.split(","):
        k, v = tok.split("="); inj[k.strip()] = float(v)
    for j, (key, idx, *_ ) in enumerate(SPEC):
        nm = f"{key}[{idx}]" if idx is not None else key
        if nm in inj:
            th[j] = inj[nm]
    return th, inj



_KCACHE = {}


def fit_kernel(eng, subset, mask=None):
    """The FitKernel for (engine, subset, mask), built once and REUSED.

    Every kernel loads ~50 CUBIN modules onto the device and `jax.clear_caches()` does not unload them,
    so building one per call would exhaust the CUDA context part-way through any scan.  A profile scan
    re-minimises the SAME free subset at every node -- only the pinned value moves, and that is a runtime
    argument (`set_fixed`), not a constant -- so one kernel serves the whole scan.
    """
    key = (id(eng), tuple(int(k) for k in subset),
           None if mask is None else np.asarray(mask, bool).tobytes())
    k = _KCACHE.get(key)
    if k is None:
        from adonis.fit.device_plan import resolve
        from adonis.fit.kernels import FitKernel
        nev = max([int(getattr(s_, "n_events", 0) or 0) for s_ in eng.samples] or [0])
        plan = resolve(getattr(eng, "cfg", None), len(subset), nev, log=log)
        k = FitKernel(eng, subset, mask=mask, jac_batch=(plan["jac_batch"] or None))
        k.plan = plan
        k.warmup(which=("residuals", "jac"))
        _KCACHE[key] = k
    return k


def trf_fit(eng, subset, tag, nit=NIT, mask=None, th_init=None, record=None, exact_cov=False):
    """Bound-constrained least squares by trust-region reflective, ON THE FUSED KERNEL.

    Returns (th, V, J, m, chi2_total, chi2_data).  Residuals and the analytic Jacobian come from
    `adonis.fit.kernels.FitKernel`, which bins INSIDE the jit.

    WHY TRF.  A genuine trust region bounds the step and reflection handles the box, so a step that
    would leave the feasible region is reflected rather than silently replaced by one pointing
    somewhere else.  This matters on dials with a hard physical floor (e.g. E_b >= 0):
      * LINEAR space, step then clip: projecting a rejected step onto the boundary can manufacture a
        SPURIOUS local minimum at the wall, one that even passes a KKT check.
      * LOG space: removes the floor but not the bound, at the cost of an unbounded step near it.

    exact_cov: return V from the EXACT hessian (forward-over-reverse HVPs) instead of (J^T W J)^-1.
    Off by default -- the two differ by ~1e-4 on the DIAGONAL, invisible in any quoted sigma, while
    costing n extra HVPs.  It matters for the log-DETERMINANT, which is sensitive to the
    worst-constrained directions; see `logdet_cov`.
    """
    from adonis.fit.minimizers import gn_fit

    kern = fit_kernel(eng, subset, mask)
    idx = np.array(subset, int)
    th = (eng.th0 if th_init is None else th_init).copy()
    kern.set_fixed(th)          # dials outside `subset` sit where the caller put them
    kern.refresh_data()         # data/sigma may have moved (a toy throw, a new closure)

    lo, hi = kern.bounds()
    x0 = np.clip(th[idx], lo + 1e-12, hi - 1e-12)

    _mz = getattr(getattr(eng, "cfg", None), "fit", None)
    _mz = getattr(_mz, "minimizer", None)
    # gtol is the only tolerance allowed to stop this fit early, and it is loose ON PURPOSE: stopping
    # on the PROJECTED GRADIENT is the criterion that means "this is a minimum", while xtol/ftol stay
    # tight so a small step or a small chi2 change is never mistaken for convergence on a near-
    # degenerate direction (e.g. M_A_res/delta_strength, corr ~ -0.995).
    r = gn_fit(kern, x0, bounds=(lo, hi), max_nfev=max(nit, 8),
               gtol=float(getattr(_mz, "gtol", 1e-8)),
               xtol=float(getattr(_mz, "xtol", 1e-14)),
               ftol=float(getattr(_mz, "ftol", 1e-14)),
               trace=record is not None)
    th[idx] = r.x

    # Convergence is RECORDED, not assumed, so a caller can see when the optimizer failed to reach its
    # own tolerance instead of silently treating the last iterate as a result.
    eng.last_status = 1 if r.converged else 0
    eng.last_nfev, eng.last_njev = int(r.nfev), int(r.njev)
    eng.last_passes = float(r.passes)

    J = r.J                                       # (nbin + n, n), already whitened, incl. the prior block
    V = kern.covariance_exact(r.x) if exact_cov else kern.covariance_gn(J)
    m = kern.model(r.x)
    c_tot = float(r.chi2)
    dx = (r.x - kern.x0) * kern.pw
    c_data = c_tot - float(dx @ dx)
    # The Newton decrement -- how much chi2 remains between here and the local minimum, in the same
    # units as everything we compare.  A gradient NORM alone says nothing: |g|=0.1 against curvature
    # ~100 leaves only 1e-4 of chi2 on the table.
    # chi2 = ||r||^2, so g = 2 J^T r and H = 2 J^T J; the predicted decrease is (1/2) g^T H^-1 g, i.e.
    # (1/4) g^T (J^T J)^-1 g.  Built from J and r, which the fit already has -- no extra program.
    rr = kern.residuals(r.x)
    g = 2.0 * (J.T @ rr)
    eng.last_opt = float(np.max(np.abs(g)))
    try:
        eng.last_gap = float(0.25 * g @ np.linalg.solve(J.T @ J, g))
    except np.linalg.LinAlgError:
        eng.last_gap = float(0.25 * g @ (np.linalg.pinv(J.T @ J, rcond=1e-12) @ g))

    if record is not None and r.trace is not None:
        t_, c_, p_, x_ = r.trace.arrays()
        for xi, ci in zip(x_, c_):
            ti = th.copy(); ti[idx] = xi
            record.append((ti, float(ci), float(ci)))
        if not (record and np.array_equal(record[-1][0], th)):
            record.append((th.copy(), c_tot, c_data))

    log(f"  [{tag}] trf: chi2={c_tot:.5f} (data {c_data:.5f}) nfev={r.nfev} njev={r.njev} "
        f"passes={r.passes:.0f} gap={eng.last_gap:.2e} -- "
        + " ".join(f"{eng.pnames[k]}={th[k]:.4f}" for k in subset))
    return th, V, J[:kern.nbin], m, c_tot, c_data


def logdet_cov(eng, subset, th, mask=None):
    """log det V for the Laplace/Occam factor, from the EXACT hessian.

    NOT from (J^T W J)^-1.  That drops sum_b r_b d2m_b, an elementwise perturbation invisible in any
    single sigma -- but a log-determinant sums over ALL eigen-directions and is dominated by the
    worst-constrained ones, so on an ill-conditioned nuisance block it can move log det V by an amount
    that is not negligible in Delta chi2 units.  The exact hessian costs ~n HVPs, against ~2n^2
    objective evaluations for a finite-difference HESSE.

    Returns (logdet, n_kept): directions truncated by the pseudo-inverse are dropped and counted,
    because slogdet returns sgn=0 the moment one is, which would silently poison the whole scan.
    """
    kern = fit_kernel(eng, subset, mask)
    kern.set_fixed(th)
    kern.refresh_data()
    x = np.asarray(th, float)[np.array(subset, int)]
    # HVP batch defaults to 4, not n.  The hessian program is compiled ON TOP of the residual and
    # jacobian programs this kernel already holds, and a vmap of width n can make the compiled program
    # too large for its CUBIN to load.  A narrower vmap is the same arithmetic in more dispatches: the
    # hessian is still exact and still O(n) passes, only the peak program size changes.
    hb = int(getattr(kern, "plan", {}).get("hvp_batch", 4))
    w = np.linalg.eigvalsh(0.5 * kern.hessian(x, batch=hb))
    pos = w[w > 1e-12 * max(w.max(), 1e-300)]
    return (float(-np.sum(np.log(pos))) if pos.size else np.nan), int(pos.size)


def lm_fit(eng, subset, tag, huber=False, nit=NIT, mask=None, record=None, tol=1e-6, th_init=None):
    """LM on chi2_data(+Huber) + prior penalty over `subset` knobs, restricted to `mask` bins.
    Returns th(full), V, J(full rows), m(full), chi2s (masked).

    record: if a list is passed, append (th.copy(), chi2_total, chi2_data) at the start and after every
            accepted iteration -- the optimization trajectory, for a convergence plot (no re-fit).
    tol:    convergence on the NEWTON DECREMENT g^T A^-1 g (predicted objective gap), default 1e-6.  This
            is landscape-invariant: it certifies theta is at the argmin even on flat/degenerate directions,
            where a relative-chi2 test can stall with theta still off the minimum.  Full Gauss-Newton
            steps (STEP_SCALE=1, the default) converge quadratically; use <1 only to force a slow,
            smooth trajectory for a convergence demo.
    th_init: start point (default eng.th0).  The prior is ALWAYS centred at eng.th0 -- th_init only warm-
            starts the walk (e.g. profile scans re-minimising from the BFP), it does not move the prior."""
    # Which minimiser runs is decided by the caller (cfg.fit.minimizer.method); lm_fit never
    # dispatches to another minimiser internally.  TRF is generally preferred for bound-constrained
    # fits -- see trf_fit's docstring.
    data, sigma = eng.data_sigma()
    if mask is None:
        mask = np.ones(len(data), bool)
    prior_w = 1.0 / eng.prior[subset]**2
    th = (eng.th0 if th_init is None else th_init).copy(); lam = 1e-3
    def chi2_terms(thv):
        m = eng.model(thv); u = (m - data) / sigma
        hw = np.minimum(1.0, HUBER_C / np.maximum(np.abs(u), 1e-12)) if huber else np.ones_like(u)
        c_data = float(np.sum((hw * u**2)[mask]))
        c_pri = float(np.sum(prior_w * (thv[subset] - eng.th0[subset])**2))
        return c_data + c_pri, c_data, m, hw
    c_cur, c_data, m, hw = chi2_terms(th)
    # Log-space dial fitting (u = log(theta)) is not supported here: e^u > 0 would keep a dial positive
    # but destroys the parameterisation the priors and sigma_post are quoted in, and TRF's own box
    # already handles the boundary correctly.  islog is kept as an explicit all-False array so the
    # branches below read as general rather than dial-specific; the compiler folds them away.
    islog = np.zeros(len(subset), bool)
    if islog.any():
        if np.any(th[np.array(subset)[islog]] <= 0):
            raise ValueError(f"log-fitted dial started at <= 0: "
                             f"{[eng.pnames[k] for k, L in zip(subset, islog) if L and th[k] <= 0]}")
        log(f"  [{tag}] log-space dials: {[eng.pnames[k] for k, L in zip(subset, islog) if L]}")
    log(f"  [{tag}] start chi2 {c_cur:.1f} (data {c_data:.1f}, {int(mask.sum())} bins)")
    if record is not None:
        record.append((th.copy(), c_cur, c_data))
    for it in range(nit):
        # chain rule dtheta/du = theta for log dials, 1 otherwise -- applied to J so that A, g, the step
        # and the Newton decrement are all in the space actually being minimised.
        scale = np.where(islog, th[subset], 1.0)
        J = eng.jac(th, subset) * scale
        W = np.where(mask, hw / sigma**2, 0.0)
        A = J.T @ (J * W[:, None]) + np.diag(prior_w * scale**2)
        g = J.T @ (W * (m - data)) + prior_w * (th[subset] - eng.th0[subset]) * scale
        # Newton decrement nd = g^T A^-1 g = 2 x (predicted objective gap to the quadratic minimum).
        # A is regularised by the prior (>= 1/prior^2) so this is well-defined even on the flat/degenerate
        # directions -- unlike a relative-chi2 test, which stalls there while theta is still off the argmin.
        nd = float(g @ np.linalg.solve(A, g))
        for _ in range(12):
            dth = STEP_SCALE * np.linalg.solve(A + lam * np.diag(np.maximum(np.diag(A), 1e-12)), -g)
            th_try = th.copy()
            # log dials update multiplicatively (theta <- theta*e^du), the exact map back from u-space;
            # this is what keeps them strictly positive without any clipping.
            th_try[subset] = np.where(islog, th[subset] * np.exp(np.clip(dth, -20, 20)), th[subset] + dth)
            # BOX CONSTRAINTS from the PHYS_BOUND registry.  Outside these the MODEL is not merely
            # disfavoured, it is meaningless:
            #   * M_A_* enter the dipole only as M_A^2, so an unbounded fit has a MIRROR MINIMUM at
            #     negative M_A with IDENTICAL chi2.
            #   * f_NN_cex outside [0,1] gives NEGATIVE event weights.
            #   * scale knobs <= 0 give a negative cross-section contribution; cascade rates appear as
            #     exp(-a/s), singular at s=0.
            #   * Eb_shift < 0 is clamped by sf_reweight, so the likelihood is EXACTLY flat there -- an
            #     absorbing trap for LM.  Projecting onto the boundary keeps the one-sided gradient alive.
            for _k, _L in zip(subset, islog):
                if not _L:                       # log dials are positive by construction -- never clip
                    th_try[_k] = _K.clip_phys(eng.pnames[_k], th_try[_k])
            c_try, cd_try, m_try, hw_try = chi2_terms(th_try)
            if c_try < c_cur:
                th, c_cur, c_data, m, hw = th_try, c_try, cd_try, m_try, hw_try
                lam = max(lam / 3, 1e-8); break
            lam *= 5
        log(f"  [{tag}] it {it:2d} chi2={c_cur:12.5f} (data {c_data:12.5f}) nd={nd:.2e} " +
            " ".join(f"{eng.pnames[k]}={th[k]:.4f}" for k in subset))
        if record is not None:
            record.append((th.copy(), c_cur, c_data))
        if nd < tol or np.linalg.norm(dth) < 1e-12:                # converged when the predicted gap -> 0
            log(f"  [{tag}] converged at it {it} (Newton decrement {nd:.2e} < {tol:.0e})"); break
    # V is returned in PHYSICAL (theta) space even when dials were fitted in log space: A is rebuilt from
    # the UNSCALED Jacobian, and since A_u = D A_theta D with D = diag(dtheta/du), V_theta = D V_u D
    # exactly.  So every caller keeps getting sigma_theta, and sigma_u = sigma_theta/theta is recoverable.
    J = eng.jac(th, subset)
    W = np.where(mask, hw / sigma**2, 0.0)
    A = J.T @ (J * W[:, None]) + np.diag(prior_w)
    V = np.linalg.pinv(A, rcond=1e-12)
    return th, V, J, m, c_cur, c_data



def cochran_q(eng, th, subset, J, m, mask=None):
    """Per-knob Cochran's Q on per-bin demands at the BFP. Returns dict knob->(Q,ndf,p,I2,nresp)."""
    data, sigma = eng.data_sigma()
    if mask is None:
        mask = np.ones(len(data), bool)
    r = data - m
    out = {}
    for c, k in enumerate(subset):
        Jk = J[:, c]
        resp = (np.abs(Jk) * eng.prior[k] > F_RESP * sigma) & mask
        n = int(resp.sum())
        if n < 3:
            out[k] = dict(Q=0.0, ndf=0, p=1.0, I2=0.0, nresp=n); continue
        dth = r[resp] / Jk[resp]
        w = (Jk[resp] / sigma[resp])**2
        dhat = np.sum(w * dth) / np.sum(w)
        Q = float(np.sum(w * (dth - dhat)**2)); ndf = n - 1
        p = float(sstats.chi2.sf(Q, ndf))
        I2 = max(0.0, (Q - ndf) / max(Q, 1e-12))
        out[k] = dict(Q=Q, ndf=ndf, p=p, I2=I2, nresp=n, dhat=float(dhat))
    return out



def split_half_consistency(eng, th, subset, J, m, mask=None):
    """Vector split-fit: one-step GN estimates on low/high-half regions (per observable), prior-anchored."""
    data, sigma = eng.data_sigma()
    if mask is None:
        mask = np.ones(len(data), bool)
    r = data - m
    lo_mask = np.zeros(len(data), bool)
    for j, d in enumerate(eng.ds):
        s, e = eng.row0[j], eng.row0[j + 1]
        lo_mask[s:s + d["nbin"] // 2] = True
    prior_w = np.diag(1.0 / eng.prior[subset]**2)
    est = {}
    for name, msk in (("lo", lo_mask & mask), ("hi", (~lo_mask) & mask)):
        Ji = J[msk]; Ci = 1.0 / sigma[msk]**2
        Ai = Ji.T @ (Ji * Ci[:, None]) + prior_w
        Vi = np.linalg.pinv(Ai, rcond=1e-12)
        dthi = Vi @ (Ji.T @ (Ci * r[msk]))
        est[name] = (th[subset] + dthi, Vi)
    dtheta = est["lo"][0] - est["hi"][0]
    Vsum = est["lo"][1] + est["hi"][1]
    Q = float(dtheta @ np.linalg.pinv(Vsum, rcond=1e-12) @ dtheta)
    ndf = len(subset)
    p = float(sstats.chi2.sf(Q, ndf))
    evals, evecs = np.linalg.eigh(np.linalg.pinv(Vsum, rcond=1e-12))
    worst = evecs[:, -1]
    zk = dtheta / np.sqrt(np.maximum(np.diag(Vsum), 1e-300))
    return dict(Q=Q, ndf=ndf, p=p, dtheta=dtheta, zk=zk, worst=worst,
                th_lo=est["lo"][0], th_hi=est["hi"][0])



def flags(eng, m, mask=None):
    """Contiguous runs of >=2 bins with |r/sigma|>2 at the BFP (excised bins never re-flagged)."""
    data, sigma = eng.data_sigma()
    if mask is None:
        mask = np.ones(len(data), bool)
    pull = (data - m) / sigma
    out = []
    for j, d in enumerate(eng.ds):
        s = eng.row0[j]; p = pull[s:s + d["nbin"]]
        bad = (np.abs(p) > 2.0) & mask[s:s + d["nbin"]]
        i = 0
        while i < len(bad):
            if bad[i]:
                k = i
                while k + 1 < len(bad) and bad[k + 1]:
                    k += 1
                if k - i + 1 >= 2:
                    out.append(dict(obs=d["name"], j=j, i0=i, i1=k,
                                    lo=float(d["edges"][i]), hi=float(d["edges"][k + 1]),
                                    nbins=k - i + 1, mean_pull=float(np.mean(p[i:k + 1]))))
                i = k + 1
            else:
                i += 1
    return out



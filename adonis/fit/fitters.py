"""The fitters: bound-constrained least squares, and the Gate-2 diagnostics built on them.

Split out of the old physical_fit_run.py, which was 700 lines in which these six functions sat
beside a pre-section study driver for GENIE/fakedata modes.  That driver read fourteen
PHYSFIT_*/ADONIS_* variables AT MODULE LEVEL, so merely importing trf_fit executed them -- which is
how a fit layer that is meant to take its parameters from a config kept a live path to the
environment.  The driver is gone; these are what sections 1-4 use.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

from adonis.analysis import knobs as _K   # PHYS_BOUND / clip_phys: one source of truth for bounds
from scipy import stats as sstats
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.reweight import bank_plot as BP, bank_reweight as BR
from adonis.reweight.reweight_model import nominal_knobs
from analysis.paper import info_content as IC
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.paper.physical_fit import (SPEC, NPAR, PNAMES, PRIOR, theta_nominal, knobs_of,
                          build_physfit_datasets, N_BINS, SYST)



# ---- fitter constants ----------------------------------------------------------------------------- #
# These were environment reads.  NIT and STEP_SCALE are per-run and now arrive from the config
# (fit.minimizer.max_nfev); the values here are defaults for a direct call.  S4_LOGFIT selected dials to
# fit in log space -- an abandoned experiment, dropped rather than carried as an empty set that silently
# changes the parameterisation if anyone sets it.
NIT = 12                # default iteration budget; callers pass cfg.fit.minimizer.max_nfev
STEP_SCALE = 1.0        # LM step scaling; 0.5 under-converged on flat dials and was reverted
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



def trf_fit(eng, subset, tag, nit=NIT, mask=None, th_init=None):
    """Bound-constrained least squares by trust-region reflective (scipy 'trf'), same contract as lm_fit.

    WHY this replaces the hand-rolled LM + clip.  Both earlier recipes failed on E_b for optimiser
    reasons, not statistical ones -- its profile is parabolic to 5% (Dchi2 = 1.59 at the wall vs 1.229^2
    = 1.51), so there was never any non-Gaussianity to model:

      * LINEAR space, step then clip.  Projecting a rejected step onto the boundary manufactures
        SPURIOUS local minima: E_b parks on the floor, the other 15 dials re-optimise around it and the
        point becomes self-consistent (one-sided gradient >= 0, so it even passes a KKT check).  Measured
        on 12 such toys, 10 reached a LOWER chi2 when restarted above the wall -- 21.6% of the ensemble
        was sitting on the floor, ~83% of that being optimiser failure rather than censoring.
      * LOG space, theta <- theta*exp(du).  Removes the floor but not the bound (e^u > 0 always), at the
        cost of an unbounded step: du = -g_u/A_uu scales as 1/theta, so the first iteration jumped E_b
        from 1.0 to 2e-9 (a factor e^-20, the step clip) and was ACCEPTED because the other dials improved
        enough to lower total chi2.  The remaining iterations then crawl back and stall short.

    TRF fixes the actual defect: a genuine trust region bounds the step, and bounds are handled by
    reflection rather than truncation, so a step that would leave the box is not silently replaced by one
    that points somewhere else.  Residuals are the whitened data terms stacked with the prior terms, and
    the analytic Jacobian from eng.jac is passed straight through.
    """
    from scipy.optimize import least_squares
    data, sigma = eng.data_sigma()
    if mask is None:
        mask = np.ones(len(data), bool)
    idx = np.array(subset, int)
    w = np.where(mask & np.isfinite(sigma) & (sigma > 0), 1.0 / np.where(sigma > 0, sigma, 1.0), 0.0)
    pw = 1.0 / eng.prior[idx]
    th = (eng.th0 if th_init is None else th_init).copy()
    lo = np.array([-np.inf if _K.phys_lo(eng.pnames[k]) is None else _K.phys_lo(eng.pnames[k])
                   for k in subset], float)
    hi = np.array([np.inf if _K.phys_hi(eng.pnames[k]) is None else _K.phys_hi(eng.pnames[k])
                   for k in subset], float)
    x0 = np.clip(th[idx], lo + 1e-12, hi - 1e-12)

    def _resid(x):
        t = th.copy(); t[idx] = x
        return np.concatenate([(eng.model(t) - data) * w, (x - eng.th0[idx]) * pw])

    def _jac(x):
        t = th.copy(); t[idx] = x
        return np.vstack([eng.jac(t, subset) * w[:, None], np.diag(pw)])

    # x_scale='jac' is not optional here.  The M_A_res/S_Delta block is degenerate at corr = -0.995
    # (condition number ~1e5), and with the default unit scaling TRF terminates on the step tolerance
    # while the projected gradient is still 1e-1 -- it reports a "solution" that is not a stationary
    # point.  Rescaling by the Jacobian columns makes the trust region isotropic in the variables that
    # actually matter.  tr_solver='exact' is the right choice at 16 parameters (dense, tiny).
    # GTOL is the only tolerance that is allowed to stop this fit early, and it is loose ON PURPOSE.
    # least_squares stops when ANY criterion is met, so leaving all three at 1e-14 meant TRF essentially
    # never exited before max_nfev: every warm-started node in the corner scan paid the full 8 function
    # + 7 Jacobian evaluations even when the starting point was already stationary.  Stopping on the
    # PROJECTED GRADIENT is the criterion that actually means "this is a minimum"; xtol/ftol stay tight
    # so a small step or a small chi2 change can still never be mistaken for convergence on the
    # degenerate M_A_res/S_Delta direction, which is exactly how the old LM false-converged.
    GTOL = float(os.environ.get("S4_TRF_GTOL", "1e-8"))
    r = least_squares(_resid, x0, jac=_jac, bounds=(lo, hi), method="trf",
                      x_scale="jac", tr_solver="exact",
                      max_nfev=max(nit, 8), xtol=1e-14, ftol=1e-14, gtol=GTOL)
    th[idx] = r.x
    # Convergence is RECORDED, not assumed.  The LM it replaces never once satisfied its own tolerance
    # (0/12 on the wall toys, every fit hitting the iteration cap) and nothing downstream noticed -- that
    # silence is how 21.6% of an ensemble ended up on the E_b floor and got read as physics.
    eng.last_opt = float(r.optimality)
    eng.last_status = int(r.status)
    m = eng.model(th)
    # The gradient NORM is scale-dependent and says nothing on its own: |g|=0.1 against curvature ~100
    # leaves 1e-4 of chi2 on the table, which is irrelevant next to the 0.4-4.3 gaps between basins.
    # The decidable quantity is the PREDICTED CHI2 GAP g^T A^-1 g (the Newton decrement) -- how much
    # chi2 remains between here and the local minimum, in the same units as everything we compare.
    _r = (m - data) * w
    _J = eng.jac(th, subset) * w[:, None]
    _g = _J.T @ _r + (r.x - eng.th0[idx]) * pw**2
    _A = _J.T @ _J + np.diag(pw**2)
    try:
        eng.last_gap = float(_g @ np.linalg.solve(_A, _g))
    except np.linalg.LinAlgError:
        eng.last_gap = float(_g @ (np.linalg.pinv(_A, rcond=1e-12) @ _g))
    c_data = float(np.sum(((m - data) * w) ** 2))
    c_tot = c_data + float(np.sum(((r.x - eng.th0[idx]) * pw) ** 2))
    J = eng.jac(th, subset)
    A = J.T @ (J * (w ** 2)[:, None]) + np.diag(pw ** 2)
    V = np.linalg.pinv(A, rcond=1e-12)
    log(f"  [{tag}] trf: chi2={c_tot:.5f} (data {c_data:.5f}) nfev={r.nfev} njev={r.njev} "
        f"opt={r.optimality:.2e} gap={eng.last_gap:.2e} -- " + " ".join(f"{eng.pnames[k]}={th[k]:.4f}" for k in subset))
    return th, V, J, m, c_tot, c_data



def lm_fit(eng, subset, tag, huber=False, nit=NIT, mask=None, record=None, tol=1e-6, th_init=None):
    """LM on chi2_data(+Huber) + prior penalty over `subset` knobs, restricted to `mask` bins.
    Returns th(full), V, J(full rows), m(full), chi2s (masked).

    record: if a list is passed, append (th.copy(), chi2_total, chi2_data) at the start and after every
            accepted iteration -- the optimization trajectory, for a convergence plot (no re-fit).
    tol:    convergence on the NEWTON DECREMENT g^T A^-1 g (predicted objective gap), default 1e-6.  This
            is landscape-invariant: it certifies theta is at the argmin even on flat/degenerate directions,
            where the old relative-chi2 test stalled with theta still off the minimum.  Full Gauss-Newton
            steps (PHYSFIT_STEP_SCALE=1, the default) then converge quadratically; use <1 only to force a
            slow smooth trajectory for a convergence demo.
    th_init: start point (default eng.th0).  The prior is ALWAYS centred at eng.th0 -- th_init only warm-
            starts the walk (e.g. profile scans re-minimising from the BFP), it does not move the prior."""
    # S4_FITTER=trf -> bound-constrained trust-region-reflective.  Kept because it reproduces the boundary
    # atom that LM does not: at E_b truth 0.50 (1.24 sigma from its wall) LM put 18.5% of 2000 toys ON the
    # wall against Chernoff's Phi(-1.24) = 10.7%, while TRF gave 7.2%.  Any FC belt or coverage number for
    # a boundary dial built with LM measures the minimiser, not the statistics.
    if os.environ.get("S4_FITTER", "").lower() == "trf" and not huber and record is None:
        return trf_fit(eng, subset, tag, nit=nit, mask=mask, th_init=th_init)
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
    # LOG-SPACE FITTING WAS REMOVED.  S4_LOGFIT let selected dials be fitted as u = log(theta), which
    # removes the boundary instead of projecting onto it -- but e^u > 0 keeps the bound while destroying
    # the parameterisation the priors and sigma_post are quoted in, and TRF's own box handles the wall
    # correctly.  Kept as an all-False array so the downstream branches stay readable rather than being
    # unpicked; the compiler folds it away.
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
            # BOX CONSTRAINTS from the PHYS_BOUND registry (was hardcoded for Eb_shift alone).  Outside
            # these the MODEL is not merely disfavoured, it is meaningless:
            #   * M_A_* enter the dipole only as M_A^2, so an unbounded fit has a MIRROR MINIMUM at
            #     negative M_A with IDENTICAL chi2.  Observed at M_A_qe=-1.55, delta_strength=-2.81 in
            #     the prior-thrown coverage ensemble, where they produced |pull| up to 6.7e11 that the
            #     coverage figure then silently discarded via its |pull|>8 cut.
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



def gate2_Q(eng, th, subset, J, m, mask=None):
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



def gate2_split(eng, th, subset, J, m, mask=None):
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



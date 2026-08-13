"""PHYSICAL FIT (logbook 17) — steps 2/3: fit engines M0/M1/M2 + Gate II + flagging.

Fake-data modes (PHYSFIT_MODE):
  closure  : data = exact-reweight model at INJECTED knobs (PHYSFIT_INJECT="M_A_qe=1.2,res_norm=0.8")
  inject2x : data = nominal Asimov with bins x2 in one observable above a threshold
             (PHYSFIT_2X="dpt>300", native units) — the known "unknown unknown"
  Errors: sigma^2 = (SYST*data)^2 + ADoNIS-MC^2, recomputed on the (injected) data.

Methods (PHYSFIT_METHOD, comma list or "all"):
  M0 : traditional — LM minimize chi2_data + prior penalty on the Gate-I knob subset (control).
  M1 : physical   — M0, then Gate II: per-knob Cochran's Q on per-bin demands + vector split-fit
       Q_split (one-step GN on disjoint low/high-half regions); freeze failers, refit, iterate.
       Then FLAG contiguous |r/sigma|>2 regions (unknown-unknown candidates).
  M2 : robust     — Huber IRLS (c=1.345) in the loss (the principled in-loss competitor).

Gate-I subset: knobs with shrinkage < 0.5 from GATE1_NPZ (blind to what was injected).
All fits include the prior term chi2_prior = sum ((theta-nom)/prior)^2 for FITTED knobs;
frozen knobs stay at nominal. LIVE per-iteration progress.

Calibration caveat (documented): Q_split one-step estimates share the prior anchor -> conservative
(under-flags); acceptable for v1, checked empirically in the closure step.
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

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")
GATE1_NPZ = os.environ.get("GATE1_NPZ", "output/altgen/physfit_gate1.npz")
MODE = os.environ.get("PHYSFIT_MODE", "closure")
METHODS = os.environ.get("PHYSFIT_METHOD", "all")
INJECT = os.environ.get("PHYSFIT_INJECT", "M_A_qe=1.2,res_norm=0.8")
INJ2X = os.environ.get("PHYSFIT_2X", "dpt>300")
LABEL = os.environ.get("ADONIS_LABEL", f"physfit_{MODE}")
NIT = int(os.environ.get("ALTGEN_NIT", "12"))
# LM step-length scale: <1 = shorter damped steps.  Full steps can overshoot into local minima under
# strong nonlinearity (kF_sf/Eb corners of the prior); 0.5 with a higher NIT is the robust setting
# established by the coverage-toy campaign.  Default 1.0 preserves historical behaviour.
STEP_SCALE = float(os.environ.get("PHYSFIT_STEP_SCALE", "1.0"))
# LOG-SPACE dials (comma list of knob names).  A dial bounded below at zero gives the projected LM a wall
# to park on, and the projection does not merely censor -- it STALLS: 21.6% of the E_b coverage toys
# returned E_b exactly on the floor, and re-minimising the identical toy data from a start above the wall
# reached a LOWER chi2 at E_b = 0.18 (toy 0: 272.056 vs 272.454).  Fitting u = log(theta) deletes the
# boundary instead of projecting onto it -- theta = e^u > 0 for every u -- so there is no wall to stall
# against and no atom in the sampling distribution.  Only the chain rule changes: dm/du = theta * dm/dtheta.
LOGFIT = {s.strip() for s in os.environ.get("S4_LOGFIT", "").split(",") if s.strip()}
F_RESP = 0.3            # responsive-bin threshold: |J_bk|*prior_k > F_RESP*sigma_b
P_GATE = 0.01           # Gate II p-value threshold (Q_k and Q_split)
HUBER_C = 1.345

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


def refresh_sigma(ds):
    """Recompute sigma/Cinv from the CURRENT d['data'] (syst part) + fixed MC part (+ GENIE stat)."""
    for d in ds:
        var = (SYST * d["data"])**2 + d["mcerr"]**2
        if "stat_g" in d:
            var = var + d["stat_g"]**2
        d["sigma"] = np.sqrt(var); d["Cinv"] = np.diag(1.0 / np.maximum(var, 1e-300))


def event_Q2(B):
    """Per-event Q^2 [GeV^2] from the bank 4-vectors (MeV). Junk rows (zero-weight padding) -> 0."""
    kmu = np.asarray(B["k_lep"], float); knu = np.asarray(B["k_nu"], float)
    bad = ~np.isfinite(kmu).all(axis=1) | (np.abs(kmu) > 1e6).any(axis=1)
    q = knu - np.where(bad[:, None], knu, kmu)
    Q2 = (q[:, 1]**2 + q[:, 2]**2 + q[:, 3]**2 - q[:, 0]**2) / 1e6
    return np.clip(Q2, 0.0, None)


def wq2_of(B, amp, lam):
    """The injected unknown-unknown: w(Q^2) = 1 - amp*exp(-Q^2/lam); w(0)=1-amp, ->1 at high Q^2."""
    return 1.0 - amp * np.exp(-event_Q2(B) / lam)


def apply_mode(ds, eng, mode, inject=None, inj2x=None, gst=None, fluct=0, q2mod=None):
    """Construct the fake data in-place on `ds` (single source of truth for all consumers).
    Returns (truth theta vector, description)."""
    truth = eng.th0.copy(); inj_desc = "asimov"
    if mode in ("closure", "q2mod", "closure_q2mod"):
        wev = np.asarray(BR.weight_jit(eng.JB, knobs_of(eng.th0, eng.nom), eng.grids))
        if mode != "q2mod":                                     # injected knob shifts
            truth_spec, _inj = parse_inject(inject or INJECT, eng.nom)
            truth = eng.th0.copy(); truth[:NPAR] = truth_spec
            wev = np.asarray(BR.weight_jit(eng.JB, knobs_of(truth_spec, eng.nom), eng.grids))
            inj_desc = inject or INJECT
        else:
            inj_desc = "asimov"
        if mode != "closure":                                   # the Q^2 unknown-unknown
            amp, lam = q2mod or tuple(float(x) for x in os.environ.get("PHYSFIT_Q2MOD", "0.2,0.3").split(","))
            wev = wev * wq2_of(eng.B, amp, lam)
            inj_desc += f" x wQ2(amp={amp},lam={lam}GeV2)"
        for d in ds:
            d["data"] = IC.bin_w(d, wev)
    elif mode in ("mecmix", "closure_mecmix"):
        # ADoNIS prediction (nominal or injected knobs) PLUS GENIE's MEC component only — a
        # physically-motivated OUTSIDE-MANIFOLD unknown-unknown (ADoNIS has no 2p2h channel;
        # only the missing channel is foreign, QE/RES stay ADoNIS). GENIE absolute normalization.
        import awkward as ak
        from analysis.paper.physfit.build_fakedata import extract_cc0pi, extract_cc1pi
        if mode == "closure_mecmix":
            truth_spec, _inj = parse_inject(inject or INJECT, eng.nom)
            truth = eng.th0.copy(); truth[:NPAR] = truth_spec
            wev = np.asarray(BR.weight_jit(eng.JB, knobs_of(truth_spec, eng.nom), eng.grids))
            inj_desc = inject or INJECT
        else:
            wev = np.asarray(BR.weight_jit(eng.JB, knobs_of(eng.th0[:NPAR], eng.nom), eng.grids))
            inj_desc = "asimov"
        for d in ds:
            d["data"] = IC.bin_w(d, wev)
        gst = gst or os.environ.get("ADONIS_GST", "output/altgen/genie_t2k_12C_ar23_CCQERESMEC_3M.gst.root")
        scale = float(os.environ.get("PHYSFIT_MECSCALE", "1.0"))
        E = extract_cc0pi(gst); E1 = extract_cc1pi(E)
        mec = np.asarray(ak.to_numpy(E["b"].mec)).astype(bool)
        sel0 = np.asarray(E["sel"]) & mec
        m1 = mec[np.asarray(E1["sel1"])]
        log(f"  MEC admixture: CC0pi {int(sel0.sum())} ev, CC1pi {int(m1.sum())} ev, scale={scale}")
        mvals = {"dpt": E["dpt"][sel0], "dat": E["dat"][sel0],
                 "pmu": E["pmu"][sel0], "cosmu": E["cthmu"][sel0],
                 "pn": E1["vals1"]["pn"][m1], "dptt": E1["vals1"]["dptt"][m1],
                 "daT": E1["vals1"]["daT"][m1], "ppi": E1["ppi"][m1], "cospi": E1["cospi"][m1]}
        for d in ds:
            edges = d["edges"]; eps = (edges[-1] - edges[0]) * 1e-12
            vc = np.clip(mvals[d["key"]], edges[0] + eps, edges[-1] - eps)
            cnt, _ = np.histogram(vc, bins=edges)
            if d["key"] in ("dpt", "dat", "pmu", "cosmu"):
                bw_unit = np.diff(edges) / (1000.0 if d["key"] in ("dpt", "pmu") else 1.0)
                sc = E["per_event"] / bw_unit                     # 1e-38/unit/nucleon
            else:
                sc = E1["per_event_nb_CH"] / np.diff(edges)       # nb/unit/CH
            d["data"] = d["data"] + scale * cnt * sc
            d["stat_g"] = scale * np.sqrt(cnt) * sc
        inj_desc += f" + GENIE-MEC(x{scale})"
    elif mode == "inject2x":
        obs, thr = (inj2x or INJ2X).split(">"); thr = float(thr)
        for d in ds:
            if d["key"] == obs.strip():
                hit = d["edges"][:-1] >= thr
                d["data"] = d["data"] * np.where(hit, 2.0, 1.0)
                log(f"  x2 injection: {d['name']} bins with edge>={thr} ({int(hit.sum())} bins)")
        inj_desc = inj2x or INJ2X
    elif mode == "genie":
        # GENIE 3M as data on the physfit binning: same extraction as build_fakedata (single source
        # of truth), identical overflow clipping; data sigma additionally carries GENIE Poisson stat.
        from analysis.paper.physfit.build_fakedata import extract_cc0pi, extract_cc1pi
        gst = gst or os.environ.get("ADONIS_GST", "output/altgen/genie_t2k_12C_ar23_CCQERES_3M.gst.root")
        E = extract_cc0pi(gst); E1 = extract_cc1pi(E)
        gvals = {"dpt": E["dpt"][E["sel"]], "dat": E["dat"][E["sel"]],
                 "pn": E1["vals1"]["pn"], "dptt": E1["vals1"]["dptt"], "daT": E1["vals1"]["daT"]}
        for d in ds:
            edges = d["edges"]; eps = (edges[-1] - edges[0]) * 1e-12
            vc = np.clip(gvals[d["key"]], edges[0] + eps, edges[-1] - eps)
            cnt, _ = np.histogram(vc, bins=edges)
            if d["key"] in ("dpt", "dat"):
                bw_unit = np.diff(edges) / (1000.0 if d["key"] == "dpt" else 1.0)
                scale = E["per_event"] / bw_unit
                d["data"] = cnt * scale                            # 1e-38/unit/nucleon
            else:
                scale = E1["per_event_nb_CH"] / np.diff(edges)
                d["data"] = cnt * scale + d["offset"]              # GENIE-C + frozen free-H (nb/CH)
            d["stat_g"] = np.sqrt(cnt) * scale
        inj_desc = f"genie:{Path(gst).name}"
    elif mode == "genie2x":
        # GENIE data PLUS the x2 tail artifact on top (recursive; each pass refreshes sigma)
        truth, d1 = apply_mode(ds, eng, "genie", gst=gst)
        _, d2 = apply_mode(ds, eng, "inject2x", inj2x=inj2x)
        return truth, f"{d1} + {d2}"
    refresh_sigma(ds)
    # optional fluctuation: jitter data by its sigma (null calibration of the gates); sigma unchanged
    if fluct:
        rng = np.random.default_rng(fluct)
        for d in ds:
            d["data"] = d["data"] + rng.normal(0.0, d["sigma"])
        inj_desc += f" +fluct(seed={fluct})"
    return truth, inj_desc


class Engine:
    """Shared differentiable model/Jacobian over the SPEC vector (optionally extended with a
    flexible Q^2-shape nuisance: per-event multiplicative spline g(Q^2;c), K knot coefficients
    linearly interpolated in log(Q^2), clamped beyond the end knots), restricted to a fit subset."""
    def __init__(self, ds, JB, grids, nom, B=None, q2knots=None):
        self.ds, self.JB, self.grids, self.nom, self.B = ds, JB, grids, nom, B
        self.th0 = theta_nominal(nom)
        self.prior = PRIOR.copy()
        self.pnames = list(PNAMES)
        self.npar = NPAR
        self.q2knots = q2knots
        if q2knots is not None:
            K = len(q2knots)
            self.q2x = jnp.asarray(np.log(np.asarray(q2knots, float)))
            self.q2ev = jnp.asarray(np.log(np.clip(event_Q2(B), 1e-4, None)))
            self.th0 = np.concatenate([self.th0, np.ones(K)])
            pw = float(os.environ.get("PHYSFIT_Q2NUIS_PRIOR", "0.5"))
            self.prior = np.concatenate([self.prior, np.full(K, pw)])
            self.pnames += [f"gQ2[{q:g}]" for q in q2knots]
            self.npar = NPAR + K
        self.row0 = np.cumsum([0] + [d["nbin"] for d in ds])
        def wf(th, JB):
            w = BR.bank_weight(JB, knobs_of(th[:NPAR], nom), grids)
            if q2knots is not None:
                w = w * jnp.interp(self.q2ev, self.q2x, th[NPAR:])
            return w
        self.wf_jit = jax.jit(wf)
        self.jvp = jax.jit(lambda th, tang, JB: jax.jvp(lambda t: wf(t, JB), (th,), (tang,))[1])

    def model(self, th):
        w = np.asarray(self.wf_jit(jnp.asarray(th), self.JB))
        return np.concatenate([IC.bin_w(d, w) for d in self.ds])

    def jac(self, th, subset):
        J = np.zeros((self.row0[-1], len(subset)))
        for c, k in enumerate(subset):
            g = np.asarray(self.jvp(jnp.asarray(th), jnp.zeros(self.npar).at[k].set(1.0), self.JB))
            for j, d in enumerate(self.ds):
                J[self.row0[j]:self.row0[j+1], c] = IC.bin_w0(d, g)
        return J

    def data_sigma(self):
        return (np.concatenate([d["data"] for d in self.ds]),
                np.concatenate([d["sigma"] for d in self.ds]))


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
    # dials fitted as u = log(theta): the boundary is removed, not projected onto (see LOGFIT above)
    islog = np.array([eng.pnames[k] in LOGFIT for k in subset], bool)
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


def main():
    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    w0 = np.asarray(BR.weight_jit(JB, nom, grids))
    log(f"bank {len(w0)} events | mode={MODE} methods={METHODS} label={LABEL}")
    # PHYSFIT_OBS: restrict the datasets to an OBS_SUBSETS key (build_physfit_datasets with obs=None
    # builds ALL observables incl. the multiplicities, but the GENIE-based modes can only fill the
    # keys their extractor provides: genie -> "tki", mecmix -> "kin9").
    from analysis.paper.physical_fit import OBS_SUBSETS
    _obs = os.environ.get("PHYSFIT_OBS", "")
    ds = build_physfit_datasets(B, w0, log, obs=OBS_SUBSETS[_obs] if _obs else None)
    q2n = os.environ.get("PHYSFIT_Q2NUIS", "")
    q2knots = [float(x) for x in q2n.split(",")] if q2n else None
    eng = Engine(ds, JB, grids, nom, B=B, q2knots=q2knots)

    # ---- Gate-I subset (blind to injection) + all nuisance coefficients --------------------------- #
    g1 = np.load(GATE1_NPZ, allow_pickle=True)
    subset = [int(i) for i in np.where(g1["shrink"] < 0.5)[0]]
    if q2knots:
        subset = subset + list(range(NPAR, eng.npar))
        log(f"Q2-shape nuisance ON: {len(q2knots)} knots at {q2knots} (prior ±{eng.prior[-1]:g})")
    log(f"fit subset ({len(subset)}): " + " ".join(eng.pnames[k] for k in subset))

    # ---- fake data -------------------------------------------------------------------------------- #
    truth, inj_desc = apply_mode(ds, eng, MODE, fluct=int(os.environ.get("PHYSFIT_FLUCT", "0")))
    log(f"fake data ready: {inj_desc}")

    # ---- run methods ------------------------------------------------------------------------------ #
    want = ["M0", "M1", "M2"] if METHODS == "all" else METHODS.split(",")
    results = {}
    for meth in want:
        log(f"==== {meth} ====")
        if meth in ("M0", "M2"):
            th, V, J, m, c, cd = lm_fit(eng, subset, meth, huber=(meth == "M2"))
            results[meth] = dict(th=th, V=V, sub=subset, chi2=c, chi2_data=cd,
                                 Qk=gate2_Q(eng, th, subset, J, m),
                                 split=gate2_split(eng, th, subset, J, m), flags=flags(eng, m))
        else:  # M1: gated fit with EXCISE-REFIT — freeze incoherent knobs; flag+excise bad regions;
               # refit the clean bins with the FULL Gate-I subset (knobs may become coherent once the
               # mismodeled region is removed) until no new flags.
            mask = np.ones(eng.row0[-1], bool)
            excised = []
            fl = []
            for xr in range(3):
                sub = list(subset); frozen = []
                for rnd in range(4):
                    if not sub:
                        # all frozen -> evaluate at NOMINAL (not the discarded biased point)
                        th = eng.th0.copy(); m = eng.model(th)
                        V = np.zeros((0, 0)); J = np.zeros((eng.row0[-1], 0))
                        dd, ss = eng.data_sigma()
                        cd = float(np.sum((((m - dd) / ss)**2)[mask])); c = cd
                        Qk = {}; sp = dict(Q=float("nan"), ndf=0, p=float("nan"), zk=np.array([]))
                        log(f"  [M1x{xr}] all knobs frozen -> evaluated at nominal")
                        break
                    th, V, J, m, c, cd = lm_fit(eng, sub, f"M1x{xr}r{rnd}", mask=mask)
                    Qk = gate2_Q(eng, th, sub, J, m, mask=mask)
                    sp = gate2_split(eng, th, sub, J, m, mask=mask)
                    fail = [k for k in sub if Qk[k]["p"] < P_GATE]
                    # split failure: freeze the knob with the largest |z| if the vector splits
                    if sp["p"] < P_GATE:
                        kz = sub[int(np.argmax(np.abs(sp["zk"])))]
                        if kz not in fail:
                            fail.append(kz)
                    log(f"  [M1x{xr}r{rnd}] Qk fails: {[eng.pnames[k] for k in fail]}  "
                        f"Q_split={sp['Q']:.1f}/{sp['ndf']} (p={sp['p']:.3g})")
                    if not fail:
                        break
                    for k in fail:
                        sub.remove(k); frozen.append(k)
                fl = flags(eng, m, mask=mask)
                if not fl:
                    log(f"  [M1x{xr}] no flags on the clean region -> converged"); break
                for f in fl:
                    excised.append(f)
                    mask[eng.row0[f["j"]] + f["i0"]: eng.row0[f["j"]] + f["i1"] + 1] = False
                log(f"  [M1x{xr}] excised {sum(f['nbins'] for f in fl)} bins "
                    f"({int((~mask).sum())} total) -> refit clean region")
            results[meth] = dict(th=th, V=V, sub=sub, frozen=frozen, chi2=c, chi2_data=cd,
                                 Qk=Qk, split=sp, flags=fl, excised=excised)

    # ---- report ----------------------------------------------------------------------------------- #
    print(f"\n==== PHYSICAL-FIT run [{LABEL}] mode={MODE} ({inj_desc}) ====")
    for meth, R in results.items():
        print(f"\n-- {meth} --  chi2_data={R['chi2_data']:.1f}")
        print(f"{'knob':>16} {'truth':>7} {'BFP':>8} {'+/-':>7} {'bias/sig':>8}   Qk_p")
        for c, k in enumerate(R["sub"]):
            s = np.sqrt(max(R["V"][c, c], 0.0))
            b = (R["th"][k] - truth[k]) / s if s > 0 else 0.0
            qp = R["Qk"][k]["p"] if k in R["Qk"] else float("nan")
            print(f"{eng.pnames[k]:>16} {truth[k]:7.3f} {R['th'][k]:8.3f} {s:7.3f} {b:8.2f}   {qp:.3g}")
        if R.get("frozen"):
            print(f"   frozen: {[eng.pnames[k] for k in R['frozen']]}")
        print(f"   Q_split p={R['split']['p']:.3g}")
        for f in R.get("excised", []):
            print(f"   EXCISED (unknown-unknown candidate) {f['obs']}: [{f['lo']:.0f},{f['hi']:.0f}] "
                  f"{f['nbins']} bins mean pull {f['mean_pull']:+.1f}")
        for f in R["flags"]:
            print(f"   FLAG {f['obs']}: [{f['lo']:.0f},{f['hi']:.0f}] {f['nbins']} bins mean pull {f['mean_pull']:+.1f}")
        if not R["flags"]:
            print("   FLAGS: none (clean region)")

    # persist the BINNED curves too (blueprint: figures re-render without recompute)
    m_nom_full = eng.model(eng.th0)
    np.savez(f"output/altgen/{LABEL}.npz", mode=MODE, inj=inj_desc, truth=truth, subset=subset,
             pnames=eng.pnames, row0=eng.row0, dskeys=[d["key"] for d in ds],
             data=np.concatenate([d["data"] for d in ds]),
             sigma=np.concatenate([d["sigma"] for d in ds]),
             mcerr=np.concatenate([d["mcerr"] for d in ds]),
             model_nom=m_nom_full,
             **{f"{d['key']}_edges": d["edges"] for d in ds},
             **{f"{m}_th": R["th"] for m, R in results.items()},
             **{f"{m}_V": R["V"] for m, R in results.items()},
             **{f"{m}_sub": np.array(R["sub"]) for m, R in results.items()},
             **{f"{m}_chi2data": R["chi2_data"] for m, R in results.items()},
             **{f"{m}_model": eng.model(R["th"]) for m, R in results.items()})
    log(f"[out] output/altgen/{LABEL}.npz")
    log("done")


if __name__ == "__main__":
    main()

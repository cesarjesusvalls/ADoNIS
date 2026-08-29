"""Pieces every benchmark uses: dial ordering, sample freezing, the fit calls and their box."""
from __future__ import annotations

import time

import numpy as np

from adonis.reweight import knobs as K


def _dial_order(g, pnames):
    """The Gate-I dials, best-constrained first.  Deterministic, so the n-subsets are nested."""
    shrink = np.asarray(g["shrink"], float)
    gate1 = np.where(shrink < 0.5)[0]
    return [int(k) for k in gate1[np.argsort(shrink[gate1])]]

def freeze_sample(eng, ref, log):
    """Impose the REFERENCE run's sigma and live-bin mask on this engine, per dataset key.

    Keys are matched by name, so a sample-composition mismatch raises KeyError rather than silently
    misaligning.
    """
    keys = [str(k) for k in ref["dskeys"]]
    row0 = np.asarray(ref["row0"], int)
    sig_ref = np.asarray(ref["sigma"], float)
    by_key = {k: sig_ref[row0[i]:row0[i + 1]] for i, k in enumerate(keys)}
    nlive = 0
    for s in eng.samples:
        for d in s.ds:
            sg = by_key[d["key"]]
            if len(sg) != d["nbin"]:
                raise SystemExit(f"{d['key']}: reference has {len(sg)} bins, engine has {d['nbin']}")
            d["sigma"] = sg.copy()
            d["_sigma0"] = sg.copy()
            nlive += int(np.isfinite(sg).sum())
    log(f"  sample frozen from reference: {nlive} live bins, sigma fixed (independent of sig_cap)")
    return nlive

def throw(eng, rng, log=None):
    """data <- data + N(0, sigma) on live bins.  Dead bins are left alone; their sigma is inf."""
    for s in eng.samples:
        for d in s.ds:
            sg = np.asarray(d["sigma"], float)
            ok = np.isfinite(sg) & (sg > 0)
            d["data"] = np.asarray(d["data"], float) + np.where(ok, rng.normal(0, np.where(ok, sg, 1.0)), 0.0)


def _bounds(eng, subset):
    """(lo, hi) per fitted dial from the PHYS_BOUND registry -- the same box the GN fit uses, so MIGRAD
    is not quietly given a different feasible set."""
    lo = np.array([-np.inf if K.phys_lo(eng.pnames[k]) is None else K.phys_lo(eng.pnames[k])
                   for k in subset], float)
    hi = np.array([np.inf if K.phys_hi(eng.pnames[k]) is None else K.phys_hi(eng.pnames[k])
                   for k in subset], float)
    return lo, hi

def _migrad(eng, subset, x0, lo, hi, use_grad, tol, max_calls, f, vg):
    """One MIGRAD fit.  Returns (x, (nvalue, nderiv), chi2, seconds) with ONLY migrad() in the clock.

    `f` and `vg` must be the ALREADY-COMPILED objectives, passed in rather than built here: eng.chi2_fn()
    returns a FRESH jax.jit wrapper on each call, with an empty compilation cache, so building them
    inside this function would recompile on every repeat.
    """
    from iminuit import Minuit

    names = [eng.pnames[k] for k in subset]
    if use_grad:
        nf = [0, 0]

        def fcn(*a):
            nf[0] += 1
            return f(np.asarray(a, float))

        def grd(*a):
            nf[1] += 1
            return vg(np.asarray(a, float))[1]

        m = Minuit(fcn, *x0, name=names, grad=grd)
    else:
        nf = [0, 0]

        def fcn(*a):
            nf[0] += 1
            return f(np.asarray(a, float))

        m = Minuit(fcn, *x0, name=names)

    m.errordef = Minuit.LEAST_SQUARES
    for i, n in enumerate(names):
        m.limits[n] = (None if not np.isfinite(lo[i]) else lo[i],
                       None if not np.isfinite(hi[i]) else hi[i])
    m.tol = tol
    m.strategy = 1

    t0 = time.perf_counter()
    m.migrad(ncall=max_calls)
    dt = time.perf_counter() - t0
    return np.array(m.values), (nf[0], nf[1]), float(m.fval), dt

def _gn(eng, subset, tol, nit):
    """One Gauss-Newton (TRF) fit, clock around the fit only.

    The clock includes trf_fit's POST-CONVERGENCE work (one model evaluation and two Jacobians for the
    covariance and Newton-decrement diagnostic), which is not minimisation; main() reports a corrected
    figure alongside the raw one.
    """
    from adonis.fit.fitters import trf_fit

    t0 = time.perf_counter()
    th, _V, _J, _m, _c_tot, c_data = trf_fit(eng, subset, "bench", nit=nit)
    dt = time.perf_counter() - t0
    return (th[np.array(subset, int)],
            (int(getattr(eng, "last_nfev", -1)), int(getattr(eng, "last_njev", -1))),
            float(c_data), dt)

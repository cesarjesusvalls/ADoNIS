"""Gauss-Newton and MIGRAD, both driven by the same `FitKernel` and instrumented identically.

The two arms differ only in which derivative object they ask the kernel for:

    gn        residuals + full forward Jacobian   -> normal equations
    migrad+g  scalar chi2 + one reverse-mode VJP  -> quasi-Newton
    migrad    scalar chi2 only                    -> gradient by finite differences

Objective, prior block, whitening, dead bins, bounds, start point and device all come from the
kernel, identically for both methods.  Every objective evaluation is timestamped and tagged with the
kernel's event-pass count (`Trace`), so a comparison against a common accuracy target can be made
after the fact via `time_to`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


class Trace:
    """Every objective evaluation: (wall time, chi2, event passes, x).

    `best_so_far()` derives the running-minimum curve used for time-to-accuracy: a rejected trial
    point counts toward cost but not toward the reported accuracy.
    """

    def __init__(self, kern):
        self.k = kern
        self.t, self.c, self.p, self.x = [], [], [], []
        self.t0 = None

    def start(self):
        self.t0 = time.perf_counter()
        return self

    def add(self, x, chi2):
        if self.t0 is None:
            return
        self.t.append(time.perf_counter() - self.t0)
        self.c.append(float(chi2))
        self.p.append(self.k.event_passes())
        self.x.append(np.asarray(x, float).copy())

    def arrays(self):
        return (np.asarray(self.t), np.asarray(self.c), np.asarray(self.p),
                np.asarray(self.x) if self.x else np.zeros((0, self.k.n)))

    def best_so_far(self):
        """(t, chi2_best, passes, x_best) with chi2 replaced by its running minimum."""
        t, c, p, x = self.arrays()
        if not len(c):
            return t, c, p, x
        best_i = np.empty(len(c), int)
        bi = 0
        for i in range(len(c)):
            if c[i] < c[bi]:
                bi = i
            best_i[i] = bi
        return t, c[best_i], p, x[best_i] if len(x) else x


@dataclass
class FitResult:
    method: str
    x: np.ndarray
    chi2: float
    wall: float
    counts: dict
    passes: float
    converged: bool
    message: str
    nfev: int = 0
    njev: int = 0
    trace: Trace | None = field(default=None, repr=False)
    J: np.ndarray | None = field(default=None, repr=False)

    def covariance(self):
        """(J^T J)^-1 at the solution, computed from the Jacobian the fit already built, after the
        clock has stopped.
        """
        if self.J is None:
            return None
        A = self.J.T @ self.J
        return np.linalg.pinv(A, rcond=1e-12)


def gn_fit(kern, x0, bounds=None, max_nfev=200, gtol=1e-8, xtol=1e-14, ftol=1e-14, trace=True):
    """Bound-constrained Gauss-Newton via scipy's trust-region reflective, on the kernel.

    x_scale='jac' is required: with unit scaling, TRF can stop on the step tolerance while the
    projected gradient is still large on a near-degenerate parameter block -- a reported "solution"
    that is not a stationary point.
    """
    from scipy.optimize import least_squares

    lo, hi = kern.bounds() if bounds is None else bounds
    x0 = np.clip(np.asarray(x0, float), lo + 1e-12, hi - 1e-12)
    tr = Trace(kern) if trace else None

    def f(x):
        r = kern.residuals(x)
        if tr is not None:
            tr.add(x, float(r @ r))
        return r

    kern.reset_counts()
    if tr is not None:
        tr.start()
    t0 = time.perf_counter()
    r = least_squares(f, x0, jac=kern.jac, bounds=(lo, hi), method="trf",
                      x_scale="jac", tr_solver="exact",
                      max_nfev=max(int(max_nfev), 8), xtol=xtol, ftol=ftol, gtol=gtol)
    wall = time.perf_counter() - t0
    counts = dict(kern.counts)
    passes = kern.event_passes()

    return FitResult(method="gn", x=np.asarray(r.x, float), chi2=float(2.0 * r.cost), wall=wall,
                     counts=counts, passes=passes, converged=bool(r.status > 0),
                     message=f"status {r.status}: {r.message}", nfev=int(r.nfev), njev=int(r.njev),
                     trace=tr, J=np.asarray(r.jac, float))


def migrad_fit(kern, x0, bounds=None, tol=0.1, max_calls=100000, use_grad=True, strategy=1, trace=True):
    """One MIGRAD fit on the kernel.  `use_grad=False` leaves MINUIT to build the gradient itself.

    `fcn` and `grd` are separate calls by design: MIGRAD's line search evaluates chi2 at several
    trial points before it ever requests a gradient, so they must not share a cached value_and_grad.
    """
    from iminuit import Minuit

    lo, hi = kern.bounds() if bounds is None else bounds
    x0 = np.clip(np.asarray(x0, float), lo + 1e-12, hi - 1e-12)
    tr = Trace(kern) if trace else None
    nf = [0, 0]

    def fcn(*a):
        x = np.asarray(a, float)
        nf[0] += 1
        c = kern.chi2(x)
        if tr is not None:
            tr.add(x, c)
        return c

    def grd(*a):
        nf[1] += 1
        return kern.grad(np.asarray(a, float))

    names = list(kern.pnames)
    m = Minuit(fcn, *x0, name=names, grad=(grd if use_grad else None))
    m.errordef = Minuit.LEAST_SQUARES
    for i, nm in enumerate(names):
        m.limits[nm] = (None if not np.isfinite(lo[i]) else lo[i],
                        None if not np.isfinite(hi[i]) else hi[i])
    m.tol = tol
    m.strategy = strategy

    kern.reset_counts()
    if tr is not None:
        tr.start()
    t0 = time.perf_counter()
    m.migrad(ncall=max_calls)
    wall = time.perf_counter() - t0
    counts = dict(kern.counts)
    passes = kern.event_passes()

    fm = m.fmin
    return FitResult(method=("migrad+g" if use_grad else "migrad"),
                     x=np.asarray(m.values, float), chi2=float(m.fval), wall=wall,
                     counts=counts, passes=passes,
                     converged=bool(fm.is_valid and not fm.has_reached_call_limit),
                     message=(f"valid={fm.is_valid} edm={fm.edm:.3e} "
                              f"call_limit={fm.has_reached_call_limit}"),
                     nfev=nf[0], njev=nf[1], trace=tr, J=None)


def time_to(trace, chi2_ref, targets):
    """First point at which the best-so-far chi2 comes within `eps` of `chi2_ref`, for each eps in
    `targets`.  Returns {eps: (t, passes, index)}, None where never reached.  `chi2_ref` must be the
    same value for every method being compared -- typically the lowest chi2 any of them attained.
    """
    t, cbest, p, _ = trace.best_so_far()
    out = {}
    for eps in targets:
        hit = np.where(cbest - chi2_ref < eps)[0]
        out[eps] = (float(t[hit[0]]), float(p[hit[0]]), int(hit[0])) if len(hit) else None
    return out


def dist_to(trace, x_ref, scale):
    """Best-so-far max|x - x_ref|/scale along the trace, in the same best-so-far ordering as chi2.

    Measures accuracy in parameter space rather than chi2: on a degenerate parameter direction, chi2
    proximity does not imply parameter proximity.
    """
    t, _, p, xbest = trace.best_so_far()
    if not len(xbest):
        return t, np.zeros(0), p
    d = np.max(np.abs(xbest - np.asarray(x_ref, float)) / np.asarray(scale, float), axis=1)
    return t, d, p

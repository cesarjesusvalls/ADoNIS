"""Gauss-Newton and MIGRAD, both driven by the SAME `FitKernel`, both instrumented the same way.

This is Phase 2 of docs/bench_fair_plan.md.  The only thing that differs between the two arms here is
which derivative object the algorithm asks the kernel for:

    gn        residuals + full forward Jacobian   -> normal equations, second-order information for free
    migrad+g  scalar chi2 + one reverse-mode VJP  -> quasi-Newton, curvature accumulated over steps
    migrad    scalar chi2 only                    -> gradient built by finite differences, ~2n calls each

Everything else -- objective, prior block, whitening, dead bins, box, start point, precision, device --
is the kernel's, identically for all three.

TWO THINGS THAT ARE DELIBERATELY NOT DONE HERE:

  NO POST-FIT WORK INSIDE THE CLOCK.  `trf_fit` times its own covariance and Newton-decrement
  diagnostic -- one model evaluation and TWO Jacobians (it calls eng.jac twice at the same theta) --
  which is not minimisation and was worth ~2.05 s of a 12.3 s "Gauss-Newton fit".  The covariance is
  still available: it is J^T J at the solution, and `covariance()` builds it from the Jacobian the fit
  already computed, after the clock has stopped.

  NO COMPARISON OF NATIVE STOPPING RULES.  GN stops on the projected gradient, MIGRAD on EDM; they
  therefore stop at different accuracies and a single wall-clock number each is not a comparison.  Every
  objective evaluation is timestamped and tagged with the kernel's event-pass count, so the honest
  question -- how long did each take to reach the SAME accuracy -- is answerable after the fact from
  `Trace`.  `time_to()` does that.

Both arms are instrumented at the objective, not at the iteration, because that is the one place both
algorithms genuinely share.  The recording costs a copy of n floats against a model pass of milliseconds,
and it is paid identically by both, so it cannot bias the comparison.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


class Trace:
    """Every objective evaluation: (wall time, chi2, event passes, x).

    The BEST-SO-FAR curve derived from this is what time-to-accuracy is measured on -- a trial point a
    line search rejects is work the method paid for, but it is not an answer the method would have
    returned, so it counts in the cost and not in the accuracy.
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
    wall: float                       # minimisation only
    counts: dict                      # kernel call counts over the timed region
    passes: float                     # event passes over the timed region
    converged: bool
    message: str
    nfev: int = 0
    njev: int = 0
    trace: Trace | None = field(default=None, repr=False)
    J: np.ndarray | None = field(default=None, repr=False)

    def covariance(self):
        """(J^T J)^-1 at the solution, from the Jacobian the fit already built -- AFTER the clock.

        For Gauss-Newton this is free: the fit's last Jacobian IS the one the covariance needs, which is
        the point of the comparison against HESSE's ~2n^2 objective evaluations.
        """
        if self.J is None:
            return None
        A = self.J.T @ self.J
        return np.linalg.pinv(A, rcond=1e-12)


# ---- Gauss-Newton ---------------------------------------------------------------------------------- #
def gn_fit(kern, x0, bounds=None, max_nfev=200, gtol=1e-8, xtol=1e-14, ftol=1e-14, trace=True):
    """Bound-constrained Gauss-Newton via scipy's trust-region reflective, on the kernel.

    x_scale='jac' is not optional: the M_A_res/delta_strength block is degenerate at corr ~ -0.995, and
    with unit scaling TRF stops on the step tolerance while the projected gradient is still O(0.1) --
    a reported "solution" that is not a stationary point.
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

    # status 1..4 are the three tolerances met; 0 is the evaluation cap, i.e. NOT converged.
    return FitResult(method="gn", x=np.asarray(r.x, float), chi2=float(2.0 * r.cost), wall=wall,
                     counts=counts, passes=passes, converged=bool(r.status > 0),
                     message=f"status {r.status}: {r.message}", nfev=int(r.nfev), njev=int(r.njev),
                     trace=tr, J=np.asarray(r.jac, float))


# ---- MIGRAD ---------------------------------------------------------------------------------------- #
def migrad_fit(kern, x0, bounds=None, tol=0.1, max_calls=100000, use_grad=True, strategy=1, trace=True):
    """One MIGRAD fit on the kernel.  `use_grad=False` leaves MINUIT to build the gradient itself.

    VALUE AND GRADIENT COME FROM SEPARATE CALLS ON PURPOSE.  Serving both from one cached
    `value_and_grad` is a trap that was measured: MIGRAD's line search evaluates the FUNCTION at several
    trial points before asking for a gradient, so each of those paid for a VJP it never used, and a
    one-entry cache then thrashed.  That made "MIGRAD with gradients" come out SLOWER than without --
    an artefact of the harness, not a property of the method.
    """
    from iminuit import Minuit

    lo, hi = kern.bounds() if bounds is None else bounds
    x0 = np.clip(np.asarray(x0, float), lo + 1e-12, hi - 1e-12)
    tr = Trace(kern) if trace else None
    nf = [0, 0]                                   # [value calls, gradient calls]

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
    m.errordef = Minuit.LEAST_SQUARES             # chi2: the 1-sigma contour is chi2_min + 1
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


# ---- time to a COMMON accuracy ---------------------------------------------------------------------- #
def time_to(trace, chi2_ref, targets):
    """First point at which the best-so-far chi2 came within `eps` of `chi2_ref`.

    Returns {eps: (t, passes, index)} with None where the target was never reached.  `chi2_ref` must be
    the SAME reference for every method being compared -- the lowest chi2 any of them attained -- or the
    comparison silently grades each method against its own idea of the minimum.
    """
    t, cbest, p, _ = trace.best_so_far()
    out = {}
    for eps in targets:
        hit = np.where(cbest - chi2_ref < eps)[0]
        out[eps] = (float(t[hit[0]]), float(p[hit[0]]), int(hit[0])) if len(hit) else None
    return out


def dist_to(trace, x_ref, scale):
    """Best-so-far max|x - x_ref|/scale along the trace, in the SAME best-so-far ordering as chi2.

    Accuracy in PARAMETER space, which is what a physics result is quoted in; chi2 proximity does not
    imply it on a degenerate direction, and on this problem M_A_res/delta_strength are correlated at
    -0.995, so the two curves genuinely differ.
    """
    t, _, p, xbest = trace.best_so_far()
    if not len(xbest):
        return t, np.zeros(0), p
    d = np.max(np.abs(xbest - np.asarray(x_ref, float)) / np.asarray(scale, float), axis=1)
    return t, d, p

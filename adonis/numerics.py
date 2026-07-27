"""Generic numerical helpers shared across ADoNIS -- interpolation + CDF building.

Backend-neutral utilities that no physics package should privately own (they were formerly buried in
nuclear/spectral.py, from which flux/spectrum.py reached in cross-package).  The ACHILLES Polint
(Neville) is bit-exact vs the instrumented reference; the code here is moved VERBATIM.
"""
from __future__ import annotations

import numpy as np


def polint(xa, ya, x):
    """Numerical-Recipes Polint (Neville), faithful to ACHILLES Interpolation.cc::Polint."""
    n = len(xa)
    c = np.array(ya, float); d = np.array(ya, float)
    dif = abs(x - xa[0]); ns = 0
    for i in range(n):
        dift = abs(x - xa[i])
        if dift < dif:
            ns = i; dif = dift
    y = ya[ns]; ns -= 1
    for m in range(n - 1):
        for i in range(n - m - 1):
            ho = xa[i] - x; hp = xa[i + m + 1] - x
            w = c[i + 1] - d[i]; den = ho - hp
            if den == 0:
                raise RuntimeError("Polint: zero denominator")
            den = w / den
            d[i] = hp * den; c[i] = ho * den
        if 2 * ns + 1 < (n - m - 1):
            y += c[ns + 1]
        else:
            y += d[ns]; ns -= 1
    return y


def neville_batch(xa, ya, x):
    """Vectorised Neville (Polint) over the last axis (n points) for a batch.  xa,ya (N,n)."""
    n = xa.shape[1]
    c = ya.astype(float).copy(); d = ya.astype(float).copy()
    dif = np.abs(x[:, None] - xa); ns = np.argmin(dif, axis=1)          # nearest point
    y = ya[np.arange(len(x)), ns].astype(float); ns = ns - 1
    for m in range(n - 1):
        for i in range(n - m - 1):
            ho = xa[:, i] - x; hp = xa[:, i + m + 1] - x
            w = c[:, i + 1] - d[:, i]; den = ho - hp
            den = w / den
            d[:, i] = hp * den; c[:, i] = ho * den
        take_c = (2 * ns + 1) < (n - m - 1)
        idx = np.clip(ns + 1, 0, n - 1)
        y = y + np.where(take_c, c[np.arange(len(x)), idx], d[np.arange(len(x)), np.clip(ns, 0, n - 1)])
        ns = np.where(take_c, ns, ns - 1)
    return y


def trapz_cdf(y, x=None):
    """Normalised cumulative-trapezoid CDF of a tabulated density `y`.  x=None -> uniform grid (the
    spacing cancels in the normalisation); pass `x` for a non-uniform grid.  CDF[0]=0, CDF[-1]=1."""
    y = np.clip(np.asarray(y, float), 0.0, None)
    if x is None:
        c = np.cumsum(0.5 * (y[:-1] + y[1:]))
    else:
        c = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(np.asarray(x, float)))
    cdf = np.concatenate([[0.0], c])
    tot = cdf[-1]
    return cdf / tot if tot > 0 else np.linspace(0.0, 1.0, len(y))

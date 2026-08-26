"""Diagonal Gaussian error model and Gauss-Newton covariance."""
from __future__ import annotations

import numpy as np


def bin_sigma(central, mcerr, syst):
    """Diagonal error model sigma = sqrt((syst*central)^2 + mcerr^2) with the EMPTY-BIN GUARD.

    A bin with no selected events has central=mcerr=0 AND J=0 (the bincount reduction is 0 there), so an
    unguarded sqrt(var)=0 makes J/sigma = 0/0 = NaN, and F = (J/sigma)^T(J/sigma) becomes NaN in EVERY
    entry -> every knob silently reports "freeze".  sigma=inf gives J/sigma=0 instead, which is correct
    (an empty bin carries no information).
    """
    var = (syst * np.asarray(central)) ** 2 + np.asarray(mcerr) ** 2
    return np.where(var > 0, np.sqrt(var), np.inf)


def gn_covariance(J, rcond=1e-12):
    """(J^T J)^+ -- the Gauss-Newton covariance of a whitened least-squares problem.

    Pseudo-inverse, not inverse: a degenerate direction gives a singular J^T J, and pinv returns the
    minimum-norm answer instead of raising or amplifying round-off.  Exact at an Asimov minimum, where
    the dropped term sum_b r_b d2m_b vanishes because the residuals do.
    """
    J = np.asarray(J, float)
    return np.linalg.pinv(J.T @ J, rcond=rcond)

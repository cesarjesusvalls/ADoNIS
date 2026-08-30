"""Diagonal Gaussian error model and Gauss-Newton covariance."""
from __future__ import annotations

import numpy as np


def bin_sigma(central, mcerr, syst):
    """Diagonal error model sigma = sqrt((syst*central)^2 + mcerr^2), with an EMPTY-BIN GUARD.

    An empty bin has central=mcerr=0 and J=0, so an unguarded sqrt(var)=0 would give J/sigma = 0/0
    = NaN, propagating to a NaN Fisher matrix and every knob reporting "frozen". sigma=inf instead
    gives J/sigma=0, correctly reflecting that an empty bin carries no information.
    """
    var = (syst * np.asarray(central)) ** 2 + np.asarray(mcerr) ** 2
    return np.where(var > 0, np.sqrt(var), np.inf)


def gn_covariance(J, rcond=1e-12):
    """(J^T J)^+ -- the Gauss-Newton covariance of a whitened least-squares problem.

    Uses the pseudo-inverse, not the inverse: a degenerate direction makes J^T J singular, and pinv
    returns the minimum-norm solution instead of raising or blowing up on round-off. Exact at an
    Asimov minimum, where the dropped term sum_b r_b d2m_b vanishes because the residuals do.
    """
    J = np.asarray(J, float)
    return np.linalg.pinv(J.T @ J, rcond=rcond)

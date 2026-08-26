"""Flux parameters: one scale per true-neutrino-energy interval, with a correlated prior.

A flux parameter f_b scales every event with true E_nu in interval b, signal and background alike --
unlike the templates (signal only) or the cross-section knobs (background only), so it cannot be
folded into either.

Binning: 200 MeV per parameter, one open bin above 2 GeV. The flux peaks at 600-800 MeV, so that region
is split at 100 MeV; everything below 600 MeV is merged into one bin, since a narrower bin there would
hold well under one signal event and be unconstrained by the data.

Prior: 10% per bin, correlated across neighbours,

    Sigma_bb' = sigma_b sigma_b' exp( -|E_b - E_b'| / L ),

with L a correlation length in true energy. Without the off-diagonal terms a fit can carve a sawtooth
out of the flux to absorb a fluctuation in one reco bin; the prior enforces smoothness in energy
instead. The kernel is exponential rather than squared-exponential because the squared exponential is
near-singular for smooth, closely spaced bins -- a prior that cannot be inverted is not a prior.
"""
from __future__ import annotations

import numpy as np

FLUX_EDGES = [0.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1200.0, 1400.0, 1700.0, 2000.0, np.inf]

SIGMA = 0.10
CORR_LENGTH = 400.0


def n_flux(edges=None) -> int:
    return len(FLUX_EDGES if edges is None else edges) - 1


def flux_index(e_nu, edges=None):
    """Flat flux-bin index per event.  The last bin is open, so nothing falls outside."""
    e = FLUX_EDGES if edges is None else edges
    return np.clip(np.digitize(np.asarray(e_nu, dtype=float), e) - 1, 0, n_flux(e) - 1)


def bin_centres(edges=None):
    """Representative energy per bin, for the correlation kernel.

    The open top bin is assigned the lower edge plus half the preceding bin's width, so that it has a
    finite separation from its neighbours; with an infinite centre the kernel would make it
    uncorrelated with everything."""
    e = np.array(FLUX_EDGES if edges is None else edges, dtype=float)
    c = 0.5 * (e[:-1] + e[1:])
    c[-1] = e[-2] + 0.5 * (e[-2] - e[-3])
    return c


def prior_cov(sigma=SIGMA, corr_length=CORR_LENGTH, edges=None):
    """The flux prior covariance: 10% diagonal, neighbours correlated over `corr_length` in true E_nu."""
    c = bin_centres(edges)
    rho = np.exp(-np.abs(c[:, None] - c[None, :]) / float(corr_length))
    s = np.full(n_flux(edges), float(sigma))
    return (s[:, None] * s[None, :]) * rho


def prior_chol(sigma=SIGMA, corr_length=CORR_LENGTH, edges=None):
    """Lower Cholesky factor of the prior covariance: the fit's prior residual is solve(L, f - 1), a
    whitening transform turning the correlated Gaussian prior into a plain sum of squares."""
    return np.linalg.cholesky(prior_cov(sigma, corr_length, edges))

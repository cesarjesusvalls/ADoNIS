"""Flux parameters: one scale per true-neutrino-energy interval, with a correlated prior.

A flux parameter f_b scales EVERY event whose true E_nu falls in interval b -- signal and background
alike.  That is what makes it a flux uncertainty rather than a cross-section one, and it is why the flux
block cannot be folded into either the templates (which touch only signal) or the cross-section knobs
(which touch only background): it cuts across both.

BINNING.  Nominally 200 MeV per parameter with one wide bin above 2 GeV, adjusted where the measured
spectrum demands it.  The T2K flux peaks at 600-800 MeV, so the peak is split at 100 MeV where more than
half the events sit, and everything below 600 MeV is merged: a [0, 400) parameter would have held 0.6
signal events, and a parameter with no events is not constrained by the data at all -- it would sit
wherever its prior and its neighbours put it while contributing a free direction to the fit.

PRIOR.  10% on each bin, correlated so that no single bin can be pulled far from its neighbours:

    Sigma_bb' = sigma_b sigma_b' exp( -|E_b - E_b'| / L )

with L a correlation length in true energy.  Without the off-diagonal terms a fit will happily carve a
sawtooth out of the flux to absorb a statistical fluctuation in one reco bin -- the flux is a smooth
function of energy and the prior has to say so.  The exponential kernel is used rather than a squared
exponential because the latter is close to singular for smooth, closely spaced bins, and a prior that
cannot be inverted is not a prior.
"""
from __future__ import annotations

import numpy as np

# True E_nu [MeV].  Ten parameters: below the peak, five across it, then 200 MeV steps, then one wide
# bin above 2 GeV.
FLUX_EDGES = [0.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1200.0, 1400.0, 1700.0, 2000.0, np.inf]

# 10%, not 20%.  The T2K flux uncertainty after the NA61/SHINE hadron-production constraint is roughly
# 8-10% near the peak, so 20% was a factor ~2 pessimistic -- and since flux dominated the section's
# error budget, that pessimism was setting the headline number.  It stays ABOVE the detector's 5% per
# reco bin because a flux parameter moves every bin it touches coherently while the detector dials are
# independent, so the two are not comparable at equal nominal width.
SIGMA = 0.10          # per-bin prior width
CORR_LENGTH = 400.0   # MeV; roughly two bins across the peak


# EDGES ARE A PARAMETER.  UnfoldConfig has always had a `flux.edges` field; nothing read it, so a run
# that set it recorded one binning in its npz and built the prior covariance from another.  None keeps
# the module default, which is the T2K binning these studies use.
def n_flux(edges=None) -> int:
    return len(FLUX_EDGES if edges is None else edges) - 1


def flux_index(e_nu, edges=None):
    """Flat flux-bin index per event.  The last bin is open, so nothing falls outside."""
    e = FLUX_EDGES if edges is None else edges
    return np.clip(np.digitize(np.asarray(e_nu, dtype=float), e) - 1, 0, n_flux(e) - 1)


def bin_centres(edges=None):
    """Representative energy per bin.  The open top bin is given a finite centre so it has a defined
    distance to its neighbours in the correlation kernel; without one the kernel would put it at
    infinite separation, i.e. uncorrelated with everything, which is the opposite of the truth."""
    e = np.array(FLUX_EDGES if edges is None else edges, dtype=float)
    c = 0.5 * (e[:-1] + e[1:])
    c[-1] = e[-2] + 0.5 * (e[-2] - e[-3])          # top bin: extrapolate the last finite width
    return c


def prior_cov(sigma=SIGMA, corr_length=CORR_LENGTH, edges=None):
    """The flux prior covariance: 10% diagonal, neighbours correlated over `corr_length` in true E_nu."""
    c = bin_centres(edges)
    rho = np.exp(-np.abs(c[:, None] - c[None, :]) / float(corr_length))
    s = np.full(n_flux(edges), float(sigma))
    return (s[:, None] * s[None, :]) * rho


def prior_chol(sigma=SIGMA, corr_length=CORR_LENGTH, edges=None):
    """Lower Cholesky factor L of the prior covariance, so the fit's prior residual is
    solve(L, f - 1): a whitening transform that turns the correlated Gaussian prior into a plain
    sum of squares, which is what a least-squares solver needs."""
    return np.linalg.cholesky(prior_cov(sigma, corr_length, edges))

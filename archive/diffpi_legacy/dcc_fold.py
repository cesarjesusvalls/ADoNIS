"""Spectral-function fold: elementary sigma(W,Q^2) -> nuclear dsigma/dW (step 5).

Builds the inclusive e + 12C -> e' + pi + X nuclear cross section by Monte-Carlo
over (a) the electron kinematics across the oracle acceptance and (b) the struck-
nucleon momentum/removal-energy from the spectral function, weighting each event by
the leptonic virtual-photon flux times the differentiable elementary cross section
sigma(W,Q^2; knobs).  W is smeared by Fermi motion: W^2 = (q + p_struck)^2.

This is the kind-1 reweighting estimator: the sampling (electron kinematics +
spectral function) is a FIXED, knob-independent proposal (all detached); only the
elementary-xsec weight carries the knobs, so gradients are exact (cf. Component A).

Leptonic factor (Hand virtual-photon flux, transverse only -- sigma_L omitted to
match the transverse elementary sigma):
    Gamma = (alpha/2pi^2)(E'/E)(K/Q^2)/(1-eps),  K=(W^2-M^2)/2M,
    eps = [1 + 2(|q|^2/Q^2) tan^2(theta/2)]^-1.
Overall normalisation is calibrated to the oracle (one constant); the SHAPE is a
prediction of the elementary amplitudes folded with the real spectral function.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .dcc import DCCKnobs
from .dcc_xsec import DCCCrossSection, M_N
from .spectral import load_spectral, SpectralSampler

ALPHA = 1.0 / 137.036
E_BEAM = 1108.0                       # oracle beam energy [MeV]
THETA_LO, THETA_HI = 36.5, 38.5       # oracle angular acceptance [deg]
EP_LO, EP_HI = 150.0, 1060.0          # outgoing-electron energy sampling window [MeV]
                                      # (E' down to 150 -> omega up to ~960 MeV so the
                                      #  W>1500 second-resonance region is not clipped)


def sample_lepton(key, n):
    """Sample (theta [rad], E' [MeV]) uniformly over the acceptance; detached."""
    kth, kep = jax.random.split(key)
    theta = jnp.deg2rad(THETA_LO + (THETA_HI - THETA_LO) * jax.random.uniform(kth, (n,)))
    Ep = EP_LO + (EP_HI - EP_LO) * jax.random.uniform(kep, (n,))
    return jax.lax.stop_gradient(theta), jax.lax.stop_gradient(Ep)


def leptonic_kinematics(theta, Ep):
    """Return Q2 [MeV^2], omega, q3 (|q| spatial), q4 (E,px,py,pz with beam along z)."""
    s2 = jnp.sin(theta / 2) ** 2
    Q2 = 4.0 * E_BEAM * Ep * s2
    omega = E_BEAM - Ep
    q3 = jnp.sqrt(Q2 + omega ** 2)
    # beam along +z; scattering in x-z plane: k=(E,0,0,E), k'=(E',E'sinθ,0,E'cosθ)
    qx = -Ep * jnp.sin(theta)
    qz = E_BEAM - Ep * jnp.cos(theta)
    q4 = jnp.stack([omega, qx, jnp.zeros_like(qx), qz], axis=-1)
    return Q2, omega, q3, q4


def fold_events(knobs: DCCKnobs, key, n=200000, sf="pke12p_tot.data",
                xs: DCCCrossSection | None = None, mode="EM", e_beam=E_BEAM,
                ep_lo=EP_LO, ep_hi=EP_HI, theta_max_deg=None):
    """Event-level fold: return (W [MeV], Q2 [MeV^2], w_evt) for n sampled events.

    W, Q2 are DETACHED (fixed proposal); w_evt = leptonic-flux * sigma(knobs) carries
    the knob dependence smoothly (kind-1 reweighting -> exact gradients).

    mode='EM'  -> electron scattering: vector current, 1/Q^2 photon flux, narrow
                  angular acceptance (THETA_LO..THETA_HI).
    mode='CC'  -> neutrino CC: vec+axial current, FLAT W-propagator weight (no
                  1/Q^2), full forward hemisphere (0..theta_max_deg)."""
    xs = xs or DCCCrossSection()
    sampler = SpectralSampler(load_spectral(sf))
    klep, ksf, kth = jax.random.split(key, 3)
    Ep = ep_lo + (ep_hi - ep_lo) * jax.random.uniform(klep, (n,))
    if mode == "CC":
        tmax = jnp.deg2rad(theta_max_deg if theta_max_deg is not None else 60.0)
        theta = tmax * jax.random.uniform(kth, (n,))
        current = "all"
    else:
        theta = jnp.deg2rad(THETA_LO + (THETA_HI - THETA_LO) * jax.random.uniform(kth, (n,)))
        current = "vec"
    Ep = jax.lax.stop_gradient(Ep); theta = jax.lax.stop_gradient(theta)
    s2 = jnp.sin(theta / 2) ** 2
    Q2 = 4.0 * e_beam * Ep * s2
    omega = e_beam - Ep
    q3 = jnp.sqrt(Q2 + omega ** 2)
    qx = -Ep * jnp.sin(theta); qz = e_beam - Ep * jnp.cos(theta)
    q4 = jnp.stack([omega, qx, jnp.zeros_like(qx), qz], axis=-1)
    p_vec, E_rm = sampler.sample(ksf, n)
    pi4 = jnp.concatenate([(M_N - E_rm)[:, None], p_vec], axis=1)
    tot = q4 + pi4
    W2 = tot[:, 0] ** 2 - jnp.sum(tot[:, 1:] ** 2, axis=1)
    W = jnp.sqrt(jnp.clip(W2, 1.0, None))
    if mode == "CC":
        Gamma = (Ep / e_beam) * jnp.sin(theta)
    else:
        eps = 1.0 / (1.0 + 2.0 * (q3 ** 2 / Q2) * (s2 / jnp.clip(1 - s2, 1e-9, None)))
        K = jnp.clip((W ** 2 - M_N ** 2) / (2 * M_N), 0.0, None)
        Gamma = (ALPHA / (2 * jnp.pi ** 2)) * (Ep / e_beam) * (K / Q2) / jnp.clip(1 - eps, 1e-6, None)
    sig = jax.vmap(lambda w, q: xs.sigma(w, q, knobs, current))(W, Q2)
    W = jax.lax.stop_gradient(W); Q2 = jax.lax.stop_gradient(Q2)
    return W, Q2, Gamma * sig


def _hist1d(vals, w, edges):
    idx = jax.lax.stop_gradient(jnp.clip(jnp.searchsorted(edges, vals) - 1, 0, edges.shape[0] - 2))
    h = jax.ops.segment_sum(w, idx, num_segments=edges.shape[0] - 1)
    return h / jnp.diff(edges)


def fold_dsigma_dW(knobs: DCCKnobs, W_edges, key, n=200000, sf="pke12p_tot.data",
                   xs: DCCCrossSection | None = None, mode="EM", e_beam=E_BEAM,
                   ep_lo=EP_LO, ep_hi=EP_HI, theta_max_deg=None):
    """Differentiable nuclear dsigma/dW histogram (thin wrapper over fold_events)."""
    W, Q2, w = fold_events(knobs, key, n, sf, xs, mode, e_beam, ep_lo, ep_hi, theta_max_deg)
    return _hist1d(W, w, W_edges)


def fold_hist2d(knobs: DCCKnobs, W_edges, Q2_edges, key, n=200000, sf="pke12p_tot.data",
                xs: DCCCrossSection | None = None, mode="CC", e_beam=E_BEAM,
                ep_lo=EP_LO, ep_hi=EP_HI, theta_max_deg=None):
    """Differentiable joint (W, Q2) histogram -- both knobs identifiable (pw_norm
    localises in W, M_A shapes Q2). Returns (nW, nQ2) array of summed weights."""
    W, Q2, w = fold_events(knobs, key, n, sf, xs, mode, e_beam, ep_lo, ep_hi, theta_max_deg)
    iw = jnp.clip(jnp.searchsorted(W_edges, W) - 1, 0, W_edges.shape[0] - 2)
    iq = jnp.clip(jnp.searchsorted(Q2_edges, Q2) - 1, 0, Q2_edges.shape[0] - 2)
    flat = jax.lax.stop_gradient(iw * (Q2_edges.shape[0] - 1) + iq)
    nb = (W_edges.shape[0] - 1) * (Q2_edges.shape[0] - 1)
    return jax.ops.segment_sum(w, flat, num_segments=nb).reshape(
        W_edges.shape[0] - 1, Q2_edges.shape[0] - 1)

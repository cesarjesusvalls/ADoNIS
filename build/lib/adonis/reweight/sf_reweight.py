"""Differentiable spectral-function (initial-state) reweight knobs.

The struck nucleon (|p|, E_removal) is SAMPLED from the tabulated spectral function S(p,E); a
deformation of S reweights each event by the density ratio w = S_theta(p,E) / S_0(p,E) at the
recorded sampled point.  Knobs (all nominal = no-op):
  kF_sf     : Fermi-momentum / |p|-axis SCALE    -> S(p/kF, .)        (nominal 1.0)
  Eb_shift  : binding/removal-energy SHIFT [MeV] -> S(., E - Eb)      (nominal 0.0)
  sf_norm   : overall normalization              -> * sf_norm        (nominal 1.0)
  src_tail  : high-|p| (short-range-correlation) tail scale          (nominal 1.0)

Interpolation is FAITHFUL to the upstream SpectralFunction (spectral.py Interp2D, order (3,1)):
cubic in p, linear in E, clamped >=0 and 0 outside the grid.  The p-axis is a uniform cubic B-spline
(coefficients prefiltered along p with scipy.ndimage.spline_filter1d, mode='mirror'), so kF_sf keeps
well-defined C2 derivatives; the E-axis is 2-tap linear, matching the ACHILLES SF resolution.  D2/D3
wrt Eb_shift (and kF_sf near the SF support edge) are non-smooth at S=0 -- physically real, since the
tabulated SF has a hard zero edge.  The variation plots use the exact reweight (D0), not a Taylor
expansion, so they are unaffected there.  w == 1 at nominal (numerator == denominator).
"""
from __future__ import annotations

import numpy as np
import os

import jax.numpy as jnp
from scipy.ndimage import spline_filter1d

from adonis.nuclear.spectral import SpectralFunction
from adonis.channels import constants as C


def sf_grids(sf: SpectralFunction):
    """Cubic-B-spline prefiltered coefficients + uniform-grid metadata (one-time, NumPy)."""
    spec2d = np.asarray(sf.spec, float).reshape(sf.np, sf.ne)
    mom = np.asarray(sf.mom, float); energy = np.asarray(sf.energy, float)
    coef = spline_filter1d(spec2d, axis=0, order=3, mode="mirror")
    return dict(coef=jnp.asarray(coef), nm=int(sf.np), ne=int(sf.ne),
                m0=float(mom[0]), hm=float(mom[1] - mom[0]), e0=float(energy[0]), he=float(energy[1] - energy[0]),
                m_lo=float(mom[0]), m_hi=float(mom[-1]), e_lo=float(energy[0]), e_hi=float(energy[-1]),
                norm=float(sf.norm))


def _bw(f):
    """Uniform cubic B-spline weights for the 4 taps (i-1,i,i+1,i+2) at fractional offset f in [0,1)."""
    return ((1.0 - f) ** 3 / 6.0,
            (4.0 - 6.0 * f ** 2 + 3.0 * f ** 3) / 6.0,
            (1.0 + 3.0 * f + 3.0 * f ** 2 - 3.0 * f ** 3) / 6.0,
            f ** 3 / 6.0)


def _mirror(idx, n):
    """scipy.ndimage 'mirror' boundary (reflect without repeating the edge sample)."""
    per = 2 * (n - 1)
    idx = jnp.mod(idx, per)
    return jnp.where(idx >= n, per - idx, idx)


def _bspline2d(g, p, E):
    """S(p,E), faithful to the upstream SpectralFunction: cubic B-spline in p (C2) x linear in E
    (2-tap), clamped >=0, 0 outside the grid.  Linear-in-E matches the ACHILLES SF (Interp2D order
    (3,1)).  kF_sf scales the p-axis (cubic), so its D2/D3 stay well-defined in the bulk."""
    u = (p - g["m0"]) / g["hm"]; v = (E - g["e0"]) / g["he"]
    iu = jnp.floor(u).astype(jnp.int32); iv = jnp.floor(v).astype(jnp.int32)
    wu = _bw(u - iu); fv = v - iv
    coef = g["coef"]; nm = g["nm"]; ne = g["ne"]
    j0 = jnp.clip(iv, 0, ne - 1); j1 = jnp.clip(iv + 1, 0, ne - 1)
    val = 0.0
    for a in range(4):
        ia = _mirror(iu - 1 + a, nm)
        col = coef[ia, j0] * (1.0 - fv) + coef[ia, j1] * fv
        val = val + col * wu[a]
    in_grid = (p >= g["m_lo"]) & (p <= g["m_hi"]) & (E >= g["e_lo"]) & (E <= g["e_hi"])
    return jnp.where(in_grid, jnp.maximum(val, 0.0), 0.0)



from adonis.constants import EB_MIRROR as _EB_MIRROR


def sf_reweight(grids, p_mag, E_removal, *, kF_sf=1.0, Eb_shift=0.0, sf_norm=1.0, src_tail=1.0,
                p_src=300.0, w_src=80.0):
    """Per-event spectral-function reweight w = sf_norm * tail * S(p/kF, E-Eb)/S(p,E).
    grids = sf_grids(SpectralFunction).  p_mag,E_removal (N,) the recorded sampled struck (|p|, removal).
    tail = 1 + (src_tail-1)*sigmoid((|p|-p_src)/w_src) enhances the high-|p| (SRC) region.  == 1 at nominal
    (kF=1,Eb=0,norm=1,src_tail=1).  Differentiable in every knob; C2 in (kF_sf,Eb_shift) -> valid D2/D3."""
    Eb_shift = jnp.abs(Eb_shift) if _EB_MIRROR else jnp.maximum(Eb_shift, 0.0)
    s0 = _bspline2d(grids, p_mag, E_removal)
    sθ = _bspline2d(grids, p_mag / kF_sf, E_removal - Eb_shift)
    ratio = jnp.where(s0 > 0, sθ / jnp.where(s0 > 0, s0, 1.0), 1.0)
    tail = 1.0 + (src_tail - 1.0) / (1.0 + jnp.exp(-(p_mag - p_src) / w_src))
    return sf_norm * tail * ratio


def removal_from_struck(p_struck):
    """(|p|, E_removal) from the struck-nucleon 4-vector(s): removal = mqe - E_in  (QESpectral)."""
    p_struck = jnp.asarray(p_struck)
    pmag = jnp.linalg.norm(p_struck[..., 1:], axis=-1)
    return pmag, (C.mN - p_struck[..., 0])

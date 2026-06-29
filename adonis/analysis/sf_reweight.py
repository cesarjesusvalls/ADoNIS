"""Differentiable spectral-function (initial-state) reweight knobs (differentiable_knobs.md Group E).

The struck nucleon (|p|, E_removal) is SAMPLED from the tabulated spectral function S(p,E); a deformation of
S reweights each event by the density ratio  w = S_theta(p,E) / S_0(p,E)  at the recorded sampled point.
Knobs (all nominal = no-op):
  kF_sf     : Fermi-momentum / |p|-axis SCALE   -> S(p/kF, .)        (nominal 1.0)
  Eb_shift  : binding/removal-energy SHIFT [MeV] -> S(., E - Eb)     (nominal 0.0)
  sf_norm   : overall normalization              -> * sf_norm        (nominal 1.0)
  src_tail  : high-|p| (short-range-correlation) tail scale          (nominal 1.0)

INTERPOLATION: a C2 uniform-grid CUBIC B-SPLINE (coefficients prefiltered once with scipy.ndimage.
spline_filter, mode='mirror'; evaluated in JAX with the cubic B-spline basis).  C2 is required so the
2nd/3rd derivatives wrt kF_sf/Eb_shift are well-defined -- a bilinear (C0) interp gives correct gradients
but garbage curvature (its 2nd derivative is 0 within a cell, delta at edges), which broke the kF_sf/Eb_shift
D2/D3.  The reweight is exactly 1 at nominal for ANY interp (numerator==denominator at the same point), so
this change does NOT touch the nominal forward prediction -- only the derivative quality.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.ndimage import spline_filter

from adonis.xsec.spectral import SpectralFunction
from adonis.xsec import constants as C


def sf_grids(sf: SpectralFunction):
    """Cubic-B-spline prefiltered coefficients + uniform-grid metadata (one-time, NumPy)."""
    spec2d = np.asarray(sf.spec, float).reshape(sf.np, sf.ne)          # spec[j*ne+i] -> (mom, energy)
    mom = np.asarray(sf.mom, float); energy = np.asarray(sf.energy, float)
    coef = spline_filter(spec2d, order=3, mode="mirror")              # B-spline coeffs (deconvolution)
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
    """C2 cubic B-spline S(p,E) on the uniform (mom,energy) grid; clipped >=0; 0 outside the grid."""
    u = (p - g["m0"]) / g["hm"]; v = (E - g["e0"]) / g["he"]
    iu = jnp.floor(u).astype(jnp.int32); iv = jnp.floor(v).astype(jnp.int32)
    wu = _bw(u - iu); wv = _bw(v - iv)
    coef = g["coef"]; nm = g["nm"]; ne = g["ne"]
    val = 0.0
    for a in range(4):
        ia = _mirror(iu - 1 + a, nm)
        col = 0.0
        for b in range(4):
            ib = _mirror(iv - 1 + b, ne)
            col = col + coef[ia, ib] * wv[b]
        val = val + col * wu[a]
    in_grid = (p >= g["m_lo"]) & (p <= g["m_hi"]) & (E >= g["e_lo"]) & (E <= g["e_hi"])
    return jnp.where(in_grid, val, 0.0)        # NO clip: it kinks the C2 spline where it overshoots <0



def sf_reweight(grids, p_mag, E_removal, *, kF_sf=1.0, Eb_shift=0.0, sf_norm=1.0, src_tail=1.0,
                p_src=300.0, w_src=80.0):
    """Per-event spectral-function reweight w = sf_norm * tail * S(p/kF, E-Eb)/S(p,E).
    grids = sf_grids(SpectralFunction).  p_mag,E_removal (N,) the recorded sampled struck (|p|, removal).
    tail = 1 + (src_tail-1)*sigmoid((|p|-p_src)/w_src) enhances the high-|p| (SRC) region.  == 1 at nominal
    (kF=1,Eb=0,norm=1,src_tail=1).  Differentiable in every knob; C2 in (kF_sf,Eb_shift) -> valid D2/D3."""
    s0 = _bspline2d(grids, p_mag, E_removal)
    sθ = _bspline2d(grids, p_mag / kF_sf, E_removal - Eb_shift)
    ratio = jnp.where(s0 > 0, sθ / jnp.where(s0 > 0, s0, 1.0), 1.0)    # 0/0 -> 1 (no-op for off-grid)
    tail = 1.0 + (src_tail - 1.0) / (1.0 + jnp.exp(-(p_mag - p_src) / w_src))
    return sf_norm * tail * ratio


def removal_from_struck(p_struck):
    """(|p|, E_removal) from the struck-nucleon 4-vector(s): removal = mqe - E_in  (QESpectral)."""
    p_struck = jnp.asarray(p_struck)
    pmag = jnp.linalg.norm(p_struck[..., 1:], axis=-1)
    return pmag, (C.mN - p_struck[..., 0])

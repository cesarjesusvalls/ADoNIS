"""Differentiable spectral-function (initial-state) reweight knobs (differentiable_knobs.md Group E).

The struck nucleon (|p|, E_removal) is SAMPLED from the tabulated spectral function S(p,E); a deformation of
S reweights each event by the density ratio  w = S_theta(p,E) / S_0(p,E)  at the recorded sampled point.
Knobs (all nominal = no-op):
  kF_sf     : Fermi-momentum / |p|-axis SCALE   -> S(p/kF, .)        (nominal 1.0)
  Eb_shift  : binding/removal-energy SHIFT [MeV] -> S(., E - Eb)     (nominal 0.0)
  sf_norm   : overall normalization              -> * sf_norm        (nominal 1.0)
  src_tail  : high-|p| (short-range-correlation) tail scale          (nominal 1.0)

INTERPOLATION: FAITHFUL to the upstream SpectralFunction (spectral.py Interp2D, polynomial order (3,1)):
CUBIC in p, LINEAR in E, with S clamped >=0 and 0 outside the grid.  The p-axis is a uniform cubic
B-spline (coefficients prefiltered along p only with scipy.ndimage.spline_filter1d, mode='mirror'), so
kF_sf (which scales the p-axis) keeps well-defined C2 derivatives; the E-axis is 2-tap linear, matching
the ACHILLES SF resolution.  An earlier bicubic (C2-in-E) B-spline overshot the steep low-E rise of S(p,E)
-> NEGATIVE densities (~14% of events at Eb=+4 MeV) and a ~12% biased Eb_shift reweight; that was an
undiscussed divergence from ACHILLES.  Consequence of the faithful choice: D2/D3 wrt Eb_shift (and wrt
kF_sf near the SF support edge) are non-smooth at S=0 -- physically real (the tabulated SF has a hard zero
edge).  The arrow/variation plots use the EXACT reweight (D0), not a Taylor expansion, so they are
unaffected.  The reweight is exactly 1 at nominal (numerator==denominator), so the nominal forward
prediction is untouched.
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
    spec2d = np.asarray(sf.spec, float).reshape(sf.np, sf.ne)          # spec[j*ne+i] -> (mom, energy)
    mom = np.asarray(sf.mom, float); energy = np.asarray(sf.energy, float)
    coef = spline_filter1d(spec2d, axis=0, order=3, mode="mirror")    # prefilter the p-axis ONLY (cubic-p; E stays linear)
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
    """S(p,E) FAITHFUL to the upstream SpectralFunction: cubic B-spline in p (C2) x LINEAR in E (2-tap),
    clamped >=0, 0 outside the grid.  Linear-in-E matches the ACHILLES SF (Interp2D order (3,1)); a bicubic
    (C2-in-E) overshoots the steep low-E rise -> negative S + ~12% biased Eb_shift reweight.  kF_sf scales
    the p-axis (cubic) so its D2/D3 stay well-defined in the bulk."""
    u = (p - g["m0"]) / g["hm"]; v = (E - g["e0"]) / g["he"]
    iu = jnp.floor(u).astype(jnp.int32); iv = jnp.floor(v).astype(jnp.int32)
    wu = _bw(u - iu); fv = v - iv
    coef = g["coef"]; nm = g["nm"]; ne = g["ne"]
    j0 = jnp.clip(iv, 0, ne - 1); j1 = jnp.clip(iv + 1, 0, ne - 1)
    val = 0.0
    for a in range(4):                                            # cubic B-spline in p
        ia = _mirror(iu - 1 + a, nm)
        col = coef[ia, j0] * (1.0 - fv) + coef[ia, j1] * fv       # linear in E on the p-filtered coeffs
        val = val + col * wu[a]
    in_grid = (p >= g["m_lo"]) & (p <= g["m_hi"]) & (E >= g["e_lo"]) & (E <= g["e_hi"])
    # density: clamp >=0 and 0 off-grid (upstream `where(in_grid & (r>0), r, 0)`).  Unclamped, the spline
    # overshoots <0 near the steep low-E rise -> NEGATIVE reweights for Eb_shift>0 (~14% of events at +4 MeV,
    # min ratio -2600).  Kink only at the S=0 hard edge (physically real).
    return jnp.where(in_grid, jnp.maximum(val, 0.0), 0.0)



_EB_MIRROR = os.environ.get("S4_EB_MIRROR", "") == "1"   # see the note inside sf_reweight()


def sf_reweight(grids, p_mag, E_removal, *, kF_sf=1.0, Eb_shift=0.0, sf_norm=1.0, src_tail=1.0,
                p_src=300.0, w_src=80.0):
    """Per-event spectral-function reweight w = sf_norm * tail * S(p/kF, E-Eb)/S(p,E).
    grids = sf_grids(SpectralFunction).  p_mag,E_removal (N,) the recorded sampled struck (|p|, removal).
    tail = 1 + (src_tail-1)*sigmoid((|p|-p_src)/w_src) enhances the high-|p| (SRC) region.  == 1 at nominal
    (kF=1,Eb=0,norm=1,src_tail=1).  Differentiable in every knob; C2 in (kF_sf,Eb_shift) -> valid D2/D3."""
    # ONE-SIDED KNOB, two ways of enforcing it:
    #   clamp  (default)   S(., E - max(Eb,0)) -- the prediction is CONSTANT for Eb<0, so chi2 is exactly
    #                      flat there and the gradient vanishes: an absorbing region a minimiser cannot
    #                      climb out of.  Since Eb_shift's nominal IS its floor (_EB_EPS = 0.01), every
    #                      fit starts on that edge, which is the documented failure mode.
    #   mirror (S4_EB_MIRROR=1)  S(., E - |Eb|) -- the response is EVEN about zero, so below the boundary
    #                      the gradient points back toward it with the right magnitude and the minimiser
    #                      is pushed out instead of stalling.  This is what T2K does for parameters whose
    #                      prior central value sits on a physical boundary (arXiv:2606.14015): "the
    #                      response functions ... were mirrored across the boundary".  The fit then runs
    #                      UNBOUNDED in Eb and the physical quantity is |Eb|; note the likelihood is even,
    #                      so it carries twin minima at +-Eb, and Eb=0 is a stationary point by symmetry.
    #                      Mirroring fixes the MINIMISATION, not the statistics -- the interval near the
    #                      boundary still needs an FC-style construction.
    Eb_shift = jnp.abs(Eb_shift) if _EB_MIRROR else jnp.maximum(Eb_shift, 0.0)
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

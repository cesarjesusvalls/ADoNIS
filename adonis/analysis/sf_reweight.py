"""Differentiable spectral-function (initial-state) reweight knobs (differentiable_knobs.md Group E).

The struck nucleon (|p|, E_removal) is SAMPLED from the tabulated spectral function S(p,E); a deformation of
S reweights each event by the density ratio  w = S_theta(p,E) / S_0(p,E)  at the recorded sampled point.
Knobs (all nominal = no-op):
  kF_sf     : Fermi-momentum / |p|-axis SCALE   -> S(p/kF, .)        (nominal 1.0)
  Eb_shift  : binding/removal-energy SHIFT [MeV] -> S(., E - Eb)     (nominal 0.0)
  sf_norm   : overall normalization              -> * sf_norm        (nominal 1.0)
  src_tail  : high-|p| (short-range-correlation) tail scale          (nominal 1.0)

The reweight is exactly 1 at nominal for ANY interpolation (numerator==denominator at the same point), so a
smooth JAX BILINEAR interp of the table suffices -- its accuracy only sets the GRADIENT quality, not the
nominal identity.  S_0 is the same bilinear (not the NumPy cubic) on both sides, so the ratio is exact.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from adonis.xsec.spectral import SpectralFunction
from adonis.xsec import constants as C


def sf_grids(sf: SpectralFunction):
    """Return (mom (np,), energy (ne,), spec2d (np,ne), norm) as jnp arrays for the JAX interp."""
    spec2d = np.asarray(sf.spec, float).reshape(sf.np, sf.ne)          # spec[j*ne+i] -> (mom, energy)
    return (jnp.asarray(sf.mom), jnp.asarray(sf.energy), jnp.asarray(spec2d), float(sf.norm))


def _bilinear(mom, energy, spec2d, p, E):
    """Differentiable bilinear S(p,E) on the (mom,energy) grid; 0 outside.  p,E (N,)."""
    nx = mom.shape[0]; ny = energy.shape[0]
    ix = jnp.clip(jnp.searchsorted(mom, p) - 1, 0, nx - 2)
    iy = jnp.clip(jnp.searchsorted(energy, E) - 1, 0, ny - 2)
    x0 = mom[ix]; x1 = mom[ix + 1]; y0 = energy[iy]; y1 = energy[iy + 1]
    tx = jnp.clip((p - x0) / (x1 - x0), 0.0, 1.0); ty = jnp.clip((E - y0) / (y1 - y0), 0.0, 1.0)
    z00 = spec2d[ix, iy]; z10 = spec2d[ix + 1, iy]; z01 = spec2d[ix, iy + 1]; z11 = spec2d[ix + 1, iy + 1]
    val = (z00 * (1 - tx) * (1 - ty) + z10 * tx * (1 - ty) + z01 * (1 - tx) * ty + z11 * tx * ty)
    in_grid = (p >= mom[0]) & (p <= mom[-1]) & (E >= energy[0]) & (E <= energy[-1])
    return jnp.where(in_grid, jnp.clip(val, 0.0, None), 0.0)


def sf_reweight(grids, p_mag, E_removal, *, kF_sf=1.0, Eb_shift=0.0, sf_norm=1.0, src_tail=1.0,
                p_src=300.0, w_src=80.0):
    """Per-event spectral-function reweight w = sf_norm * tail * S(p/kF, E-Eb)/S(p,E).
    grids = sf_grids(SpectralFunction).  p_mag,E_removal (N,) the recorded sampled struck (|p|, removal).
    tail = 1 + (src_tail-1)*sigmoid((|p|-p_src)/w_src) enhances the high-|p| (SRC) region.  == 1 at nominal
    (kF=1,Eb=0,norm=1,src_tail=1).  Differentiable in every knob (autodiff)."""
    mom, energy, spec2d, _ = grids
    s0 = _bilinear(mom, energy, spec2d, p_mag, E_removal)
    sθ = _bilinear(mom, energy, spec2d, p_mag / kF_sf, E_removal - Eb_shift)
    ratio = jnp.where(s0 > 0, sθ / jnp.where(s0 > 0, s0, 1.0), 1.0)    # 0/0 -> 1 (no-op for off-grid)
    tail = 1.0 + (src_tail - 1.0) / (1.0 + jnp.exp(-(p_mag - p_src) / w_src))
    return sf_norm * tail * ratio


def removal_from_struck(p_struck):
    """(|p|, E_removal) from the struck-nucleon 4-vector(s): removal = mqe - E_in  (QESpectral)."""
    p_struck = jnp.asarray(p_struck)
    pmag = jnp.linalg.norm(p_struck[..., 1:], axis=-1)
    return pmag, (C.mN - p_struck[..., 0])

"""The reweight parameter vector: which knobs exist, their order, and where they are physical.

Two different things live here, and only one of them belongs to the generator:

  THE PARAMETERISATION -- the knob order (PNAMES, NPAR), the theta <-> PhysicsParams maps, and the
  physical bounds (PHYS_BOUND, phys_lo/phys_hi/clip_phys).  These are properties of the model: a
  negative axial mass is unphysical no matter who is fitting it.  The knob set itself comes from
  adonis.reweight.reweight_model.knob_specs; this module fixes the ORDER, which the Jacobian, the
  covariance and every saved npz index by.

  PRIOR -- how well an analyst claims to know each knob, defaulting to 20% multiplicative with
  natural units where that is meaningless (Eb_shift 4 MeV, f_NN_cex 0.1).  That is an analysis
  choice, not a property of the generator, and callers should pass their own: UnfoldEngine takes
  knob_prior for exactly this reason.  It stays here only because it must line up with the knob
  order above, and splitting the two invites them to disagree.
"""
import os

import numpy as np
import jax.numpy as jnp

from adonis.reweight.reweight_model import knob_specs as _knob_specs, nominal_knobs

_PRIOR_POLICY = {"Eb_shift": 4.0, "f_NN_cex": 0.10}     # everything else: 20% multiplicative
SPEC = [(name, idx, lab, _PRIOR_POLICY.get(name, 0.20))
        for (name, idx, lab, _nom) in _knob_specs(nominal_knobs())]
NPAR = len(SPEC)
PNAMES = [f"{k}[{i}]" if i is not None else k for k, i, *_ in SPEC]
PRIOR = np.array([s[3] for s in SPEC])


def theta_nominal(nom):
    """The SPEC theta vector at the nominal PhysicsParams `nom`."""
    th = np.zeros(NPAR)
    for j, (key, idx, *_) in enumerate(SPEC):
        v = getattr(nom, key)
        th[j] = float(v[idx]) if idx is not None else float(v)
    return th


def knobs_of(theta, nom):
    """Full PhysicsParams from the SPEC theta vector (tuple components rebuilt)."""
    upd = {}
    tup = {}
    for j, (key, idx, *_) in enumerate(SPEC):
        if idx is None:
            upd[key] = theta[j]
        else:
            tup.setdefault(key, list(np.asarray(getattr(nom, key))))[idx] = theta[j]
    for key, lst in tup.items():
        upd[key] = jnp.asarray(lst)
    return nom._replace(**upd)


# --------------------------------------------------------------------------- physical boundaries
# Dials with a hard physical boundary.  The model clamps outside these (e.g. sf_reweight flattens for
# E_b < 0), so the likelihood is exactly flat there: a fit step, a profile node or a toy truth placed
# outside the bound is unrecoverable, and one placed exactly on it is degenerate.  Everything that
# generates a theta value must go through `phys_lo` / `clip_phys` rather than hardcoding a floor.
# These are CODE-VALIDITY ranges: outside them the model stops being meaningful (negative weights,
# mirror minima, singular reweights), not merely "physically disfavoured".  Tighter physics-motivated
# priors are a separate question and deliberately not encoded here.
#   * M_A_* enter the axial dipole only as M_A^2 (F_A = -g_A/(1+Q2/M_A^2)^2, form_factors.py), so the
#     likelihood is exactly symmetric under M_A -> -M_A: an unbounded fit has a mirror minimum at
#     negative M_A.
#   * f_NN_cex is a fraction: nncex_slot_factor returns f/0.5 (swapped) and (1-f)/0.5 (not), so outside
#     [0,1] it produces negative event weights.
#   * every other dial is a multiplicative scale with nominal 1.0; <= 0 means a negative cross-section
#     contribution, and the cascade rate scales additionally appear as exp(-a/s) (s<=0 is singular).
#   * Eb_shift: the SF reweight clamps Eb<0, so the likelihood is exactly flat below zero.
_POS = (0.0, None)                          # strictly positive scale (FLOOR_EPS keeps it off zero)
PHYS_BOUND = {
    "M_A_qe": _POS, "M_A_res": _POS,                          # mirror minimum at -M_A
    "f_NN_cex": (0.0, 1.0),                                   # a fraction; negative weights outside
    "Eb_shift": (0.0, None),                                  # model clamps below zero
    "axial_strength": _POS, "vector_strength": _POS, "res_axial_strength": _POS,
    "delta_strength": _POS, "pion_pole": _POS,
    "mu_p": _POS, "mu_n": _POS, "gep": _POS, "gen": _POS,
    "sabs": _POS, "s_piN_elastic": _POS, "s_piN_cex": _POS, "s_conv": _POS,
    "s_NN_elastic": _POS, "s_NN_inelastic": _POS,             # tuple knobs: bound applies per component
    "kF_sf": _POS, "sf_norm": _POS, "src_tail": _POS,
    "qe_norm": _POS, "res_norm": _POS,
}
FLOOR_EPS = 1e-2                            # stay strictly INSIDE the bound, never on the clamp itself


def _base(name):
    """'s_NN_elastic[pn]' -> 's_NN_elastic' so tuple components inherit the knob's bound."""
    return name.split("[", 1)[0]


# With ADONIS_EB_MIRROR=1 the SF response is EVEN in Eb_shift (see sf_reweight), so negative values are
# meaningful -- they denote the same physical shift |Eb|.  Bounding the fit at zero would then re-impose
# the very wall the mirroring exists to remove, so Eb_shift becomes unbounded in that mode.
# This file decides whether Eb_shift is bounded at zero; adonis/reweight/sf_reweight.py decides whether
# the SF response is mirrored about zero.  Both must read the same flag, or the fit ends up bounded
# away from values the physics treats as meaningful (or unbounded into values it does not).
from adonis.constants import EB_MIRROR
_MIRRORED = {"Eb_shift"} if EB_MIRROR else set()


def phys_lo(name):
    """Smallest value a dial may take, offset off the clamp by FLOOR_EPS.  None if unbounded below."""
    if _base(name) in _MIRRORED:
        return None
    b = PHYS_BOUND.get(_base(name))
    return None if b is None or b[0] is None else b[0] + FLOOR_EPS


def phys_hi(name):
    b = PHYS_BOUND.get(_base(name))
    return None if b is None or b[1] is None else b[1] - FLOOR_EPS


def clip_phys(name, v):
    """Clip a value into the dial's physical range (no-op for unbounded dials)."""
    lo, hi = phys_lo(name), phys_hi(name)
    if lo is not None:
        v = max(v, lo)
    if hi is not None:
        v = min(v, hi)
    return v

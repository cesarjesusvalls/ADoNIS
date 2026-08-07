"""The 28-knob reweight basis shared by every Gate-I gradient / fit.

The SINGLE source of the knob ORDER, the fit-local PRIOR policy, and the theta<->PhysicsParams maps.
Relocated from analysis/paper/physical_fit.py into the reusable core so AnaSample, the reweight engine,
and the section drivers all import ONE definition instead of re-deriving it.  The knob set itself comes
from adonis.reweight.reweight_model.knob_specs (which drops the dead sscat / dormant pw_norm knobs and
expands the tuple knobs); here we only attach the fit priors: 20% multiplicative by default, natural
units for the non-multiplicative knobs (Eb_shift +/-4 MeV, f_NN_cex +/-0.1).
"""
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
# Dials with a HARD PHYSICAL boundary.  The model CLAMPS outside these (e.g. sf_reweight flattens for
# E_b < 0), so the likelihood is exactly flat there: a fit step, a profile node or a toy truth placed
# outside the bound is unrecoverable, and one placed exactly ON it is degenerate.  Everything that
# generates a theta value must go through `phys_lo` / `clip_phys` rather than hardcoding a floor --
# the same 1e-2 was previously copy-pasted into six call sites, free to drift apart.
# CODE-VALIDITY ranges: outside these the MODEL stops being meaningful (negative weights, mirror minima,
# singular reweights) -- not merely "physically disfavoured".  Tighter physics-motivated priors are a
# separate question and deliberately NOT encoded here.
#   * M_A_* enter the axial dipole ONLY as M_A^2 (F_A = -g_A/(1+Q2/M_A^2)^2, form_factors.py), so the
#     likelihood is exactly symmetric under M_A -> -M_A: an unbounded fit has a MIRROR MINIMUM at negative
#     M_A.  Observed: M_A_qe = -1.55, M_A_res = -0.50 in the prior-thrown coverage ensemble.
#   * f_NN_cex is a FRACTION: nncex_slot_factor returns f/0.5 (swapped) and (1-f)/0.5 (not), so outside
#     [0,1] it produces NEGATIVE event weights.
#   * every other dial is a multiplicative SCALE with nominal 1.0; <= 0 means a negative cross-section
#     contribution, and the cascade rate scales additionally appear as exp(-a/s) (s<=0 is singular).
#   * Eb_shift: the SF reweight clamps Eb<0, so the likelihood is exactly FLAT below zero.
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


def phys_lo(name):
    """Smallest value a dial may take, offset off the clamp by FLOOR_EPS.  None if unbounded below."""
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

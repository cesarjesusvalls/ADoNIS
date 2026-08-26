"""The reweight parameter vector: which knobs exist, their order, and where they are physical.

Two different things live here, and only one of them belongs to the generator:

  PARAMETERISATION -- the knob order (PNAMES, NPAR), the theta <-> PhysicsParams maps, and the
  physical bounds (PHYS_BOUND, phys_lo/phys_hi/clip_phys). These are properties of the model: a
  negative axial mass is unphysical no matter who is fitting it. The knob set itself comes from
  adonis.reweight.reweight_model.knob_specs; this module fixes the ORDER, which the Jacobian, the
  covariance, and every saved npz index by.

  PRIOR -- how well an analyst claims to know each knob (a default multiplicative fraction, with
  natural-unit overrides for a few knobs). This is an analysis choice, not a property of the
  generator: callers may pass their own (see UnfoldEngine's knob_prior argument). It stays in this
  module only because it must remain index-aligned with the parameterisation above.
"""
import os

import numpy as np
import jax.numpy as jnp

from adonis.reweight.reweight_model import knob_specs as _knob_specs, nominal_knobs

_PRIOR_POLICY = {"Eb_shift": 4.0, "f_NN_cex": 0.10}
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


_POS = (0.0, None)
PHYS_BOUND = {
    "M_A_qe": _POS, "M_A_res": _POS,
    "f_NN_cex": (0.0, 1.0),
    "Eb_shift": (0.0, None),
    "axial_strength": _POS, "vector_strength": _POS, "res_axial_strength": _POS,
    "delta_strength": _POS, "pion_pole": _POS,
    "mu_p": _POS, "mu_n": _POS, "gep": _POS, "gen": _POS,
    "sabs": _POS, "s_piN_elastic": _POS, "s_piN_cex": _POS, "s_conv": _POS,
    "s_NN_elastic": _POS, "s_NN_inelastic": _POS,
    "kF_sf": _POS, "sf_norm": _POS, "src_tail": _POS,
    "qe_norm": _POS, "res_norm": _POS,
}
FLOOR_EPS = 1e-2


def _base(name):
    """'s_NN_elastic[pn]' -> 's_NN_elastic' so tuple components inherit the knob's bound."""
    return name.split("[", 1)[0]


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

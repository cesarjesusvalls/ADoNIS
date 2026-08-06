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

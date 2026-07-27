"""Shared four-vector / phase-space algebra.

The home for kinematic helpers that were reinvented across the generators.  JAX-first by preference;
numpy `_np`-suffixed variants exist where a helper feeds the bit-pinned numpy production-sampling
paths (the PRNG-stream-identity reason numpy stays there).

Metric signature (+,-,-,-); 4-vectors are (...,4) = (E, px, py, pz) in MeV.

NOTE (deliberately NOT consolidated here): the jax Minkowski dot `_mink_dot` and `boost_to_rest` live
in adonis/channels/dcc/lepton.py because they are woven through the DCC stack + the (golden-pinned)
observables/kinematics.py; the numpy oracle boost (oracle/finalstate._boost_to_rest) and the
momentum-scaled isotropic-direction inlines (qe/res/spectral) reorder float ops and are not
bit-identical -- they stay put with cross-references.
"""
from __future__ import annotations

import numpy as np


def boost_np(p4, beta):
    """Boost 4-vector(s) p4 (...,4) by beta (...,3) (CM->lab when beta = P/E of total).  NumPy, batched
    -- verbatim the sampler-path boost formerly copied byte-for-byte in channels/qe.py + channels/ee.py."""
    b2 = np.sum(beta ** 2, axis=-1, keepdims=True)
    g = 1.0 / np.sqrt(np.clip(1 - b2, 1e-15, None))
    bp = np.sum(beta * p4[..., 1:], axis=-1, keepdims=True)
    E = (g[..., 0] * (p4[..., 0] + bp[..., 0]))
    fac = (g - 1.0) * np.where(b2 > 1e-15, bp / np.clip(b2, 1e-15, None), 0.0) + g * p4[..., :1]
    vec = p4[..., 1:] + fac * beta
    return np.concatenate([E[..., None], vec], axis=-1)


def mass2_np(p):
    """Minkowski invariant p.p = E^2 - |p|^2 over a batch (N,4) (numpy)."""
    return p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=1)


def kallen(s, s1, s2):
    """Kallen triangle function lambda(s,s1,s2) = (s - s1 - s2)^2 - 4 s1 s2 (backend-neutral)."""
    return (s - s1 - s2) ** 2 - 4 * s1 * s2

"""Free-nucleon initial state: a single nucleon at rest (Phase A3).

The elementary-target limit of a NuclearModel -- no Fermi motion, no removal
energy.  `sample_nucleon` returns p_vec = 0 and E_removal = 0, so the struck
nucleon is on-shell at rest, p_struck = (M_N, 0, 0, 0), and the DCC vertex sees
free-nucleon kinematics.  This is the target for the ANL/BNL bubble-chamber
comparison (paper Fig. anl_bnl): total pion-production cross sections for
nu_mu p -> mu- p pi+, nu_mu n -> mu- n pi+, nu_mu n -> mu- p pi0 as sigma(E_nu).

It plugs into exactly the same sample/reweight contract as SpectralFunction, so
the differentiable weight (and its M_A gradient) is unchanged -- only the initial
state is swapped.  Being a detached, parameter-free sampler, its closure_test is
skipped (no differentiable knob) and there is no spectral-table oracle; the
physics oracle for the free nucleon lives at the cross-section level (sigma(E_nu)
vs ACHILLES on a stationary nucleon), gated in the A3 cross-section module.
"""
from __future__ import annotations

import jax.numpy as jnp

from adonis.nuclear.base import NuclearModel


class FreeNucleon(NuclearModel):
    """A nucleon at rest: p = 0, E_removal = 0 (elementary target)."""

    def __init__(self, name: str = "free"):
        self.name = name

    def sample_nucleon(self, key, n):
        p_vec = jnp.zeros((n, 3))
        E_rm = jnp.zeros((n,))
        return p_vec, E_rm


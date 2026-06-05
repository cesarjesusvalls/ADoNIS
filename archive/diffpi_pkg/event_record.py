"""EventRecord: the full lab-frame final state of one batch of CC single-pion events.

A JAX-friendly NamedTuple of arrays (leading axis = event).  Carries the lab 4-momenta
of every external particle, the per-event weight, and the channel / PDG identities, so
any observable (observables.py) or signal definition can be computed downstream.  The
weight is differentiable in the physics knobs; the kinematics are detached samples.
"""
from __future__ import annotations

from typing import NamedTuple

import jax


class EventRecord(NamedTuple):
    k: jax.Array          # (N,4) incoming neutrino   (lab)
    kp: jax.Array         # (N,4) outgoing lepton     (lab)
    p_struck: jax.Array   # (N,4) initial struck nucleon, off-shell (lab)
    p_pi: jax.Array       # (N,4) outgoing pion       (lab, on-shell)
    p_N: jax.Array        # (N,4) outgoing nucleon    (lab, on-shell)
    w: jax.Array          # (N,)  weight (differentiable in knobs)
    channel: jax.Array    # (N,)  index into CC_CHANNELS
    pid_pi: jax.Array     # (N,)  final pion PDG  (111 pi0, 211 pi+)
    pid_N: jax.Array      # (N,)  final nucleon PDG (2212 p, 2112 n)
    pid_Ni: jax.Array     # (N,)  initial nucleon PDG
    W: jax.Array          # (N,)  hadronic invariant mass used for the decay [MeV]
    Q2_adj: jax.Array     # (N,)  on-shell-rebalanced Q^2 fed to the amplitude [MeV^2]

    def filter(self, keep):
        """Return a new EventRecord with only the events where `keep` (bool (N,)) is True."""
        import jax.numpy as jnp
        idx = jnp.nonzero(keep)[0]
        return EventRecord(*[jnp.take(x, idx, axis=0) for x in self])

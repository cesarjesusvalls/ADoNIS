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
    k: jax.Array
    kp: jax.Array
    p_struck: jax.Array
    p_pi: jax.Array
    p_N: jax.Array
    w: jax.Array
    channel: jax.Array
    pid_pi: jax.Array
    pid_N: jax.Array
    pid_Ni: jax.Array
    W: jax.Array
    Q2_adj: jax.Array

    def filter(self, keep):
        """Return a new EventRecord with only the events where `keep` (bool (N,)) is True."""
        import jax.numpy as jnp
        idx = jnp.nonzero(keep)[0]
        return EventRecord(*[jnp.take(x, idx, axis=0) for x in self])

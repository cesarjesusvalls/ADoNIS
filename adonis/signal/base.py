"""Signal definitions: a boolean selection on an EventRecord, based on the (post-FSI)
particle content and/or true kinematics.  `select(event) -> (N,) bool mask`; downstream the
mask multiplies the weight (so predictions are restricted to the signal, differentiably).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp

from adonis import observables as obs


class SignalDef(ABC):
    @abstractmethod
    def select(self, event):
        """Return a boolean (N,) mask of events passing the signal definition."""

    def weight(self, event):
        """Signal-restricted weight = event weight where selected, else 0 (differentiable)."""
        return jnp.where(self.select(event), event.w, 0.0)


class AllEvents(SignalDef):
    def select(self, event):
        return jnp.ones(event.w.shape, dtype=bool)


class HasPion(SignalDef):
    """Final-state pion of a given PDG (e.g. 211 pi+, 111 pi0); None = any pion."""
    def __init__(self, pid=None):
        self.pid = pid

    def select(self, event):
        if self.pid is None:
            return jnp.ones(event.w.shape, dtype=bool)
        return event.pid_pi == self.pid


class KinematicCut(SignalDef):
    """Cut an observable into [lo, hi]. observable: name in observables.OBSERVABLES."""
    def __init__(self, observable, lo=-jnp.inf, hi=jnp.inf):
        self.fn = obs.OBSERVABLES[observable]; self.lo = lo; self.hi = hi

    def select(self, event):
        v = self.fn(event)
        return (v >= self.lo) & (v <= self.hi)

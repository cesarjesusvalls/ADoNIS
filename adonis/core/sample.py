"""The Sample bundle and Sampler interface.

A `Sample` is the fixed, DETACHED proposal of the kind-1 estimator: everything that does
NOT depend on the physics knobs (kinematics, precomputed angular kernels, lepton tensor,
cuts, phase-space prefactors, lab final-state momenta).  Currently a plain dict keyed by
the fields the channel needs; promoted to a typed container only if it earns its keep.

A `Sampler` produces a Sample from a PRNG key.  Composable samplers (flux, nuclear,
leptonic, decay-angle) contribute their detached draws into one bundle.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

Sample = dict


class Sampler(ABC):
    @abstractmethod
    def propose(self, key, n) -> Sample:
        """Return a detached Sample bundle of `n` events."""

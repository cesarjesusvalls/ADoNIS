"""Final-state-interaction model interface.

An FSIModel is a DIFFERENTIABLE transform of the produced EventRecord (propagate /
absorb / scatter the hadrons).  It must stay differentiable in `params` so FSI parameters
can be fit jointly with the production knobs.  `NoFSI` is the identity (bare production).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class FSIModel(ABC):
    @abstractmethod
    def apply(self, params, event):
        """EventRecord -> EventRecord (differentiable in params)."""

"""Final-state-interaction model interface.

An FSIModel is a DIFFERENTIABLE transform of the produced EventRecord (propagate /
absorb / scatter the hadrons).  It must stay differentiable in `params` so FSI parameters
can be fit jointly with the production knobs.  `NoFSI` is the identity (bare production).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from adonis.core.validation import SelfTestMixin


class FSIModel(SelfTestMixin, ABC):
    """FSI ABC for the differentiable-chain (core.chain.Generator) path; NoFSI is the concrete
    implementor.  NOTE: the production cascade (adonis.fsi.cascade) is invoked directly by
    workflow/reweight, NOT wrapped in this interface -- production bypasses this ABC by design."""

    @abstractmethod
    def apply(self, params, event):
        """EventRecord -> EventRecord (differentiable in params)."""

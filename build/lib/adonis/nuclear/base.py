"""Nuclear initial-state model interface.

A NuclearModel provides the struck-nucleon initial state: a DETACHED sample of the
nucleon 3-momentum and removal energy (the fixed proposal of the kind-1 estimator).
Swap implementations (spectral function, Fermi gas, LFG, ...) without touching the
primary process.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from adonis.core.validation import SelfTestMixin


class NuclearModel(SelfTestMixin, ABC):
    @abstractmethod
    def sample_nucleon(self, key, n):
        """Return (p_vec [n,3] MeV, E_removal [n] MeV) -- detached draws."""
        raise NotImplementedError


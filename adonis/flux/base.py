"""Neutrino flux model interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from adonis.core.validation import SelfTestMixin


class FluxModel(SelfTestMixin, ABC):
    """Flux ABC for the differentiable-chain (core.chain.Generator) path; Monochromatic is the
    concrete implementor.  NOTE: the config-driven production generators use SpectrumFlux/HadronBeam
    (adonis.flux) directly, which do NOT subclass this -- production bypasses this ABC by design."""

    @property
    @abstractmethod
    def e_nu_nominal(self) -> float:
        """Representative beam energy [MeV] (used for the monochromatic scalar path)."""

    @abstractmethod
    def sample_enu(self, key, n):
        """Return per-event neutrino energy [MeV] (n,) -- detached."""

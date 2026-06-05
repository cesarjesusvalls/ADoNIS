"""Neutrino flux model interface."""
from __future__ import annotations

from abc import ABC, abstractmethod


class FluxModel(ABC):
    @property
    @abstractmethod
    def e_nu_nominal(self) -> float:
        """Representative beam energy [MeV] (used for the monochromatic scalar path)."""

    @abstractmethod
    def sample_enu(self, key, n):
        """Return per-event neutrino energy [MeV] (n,) -- detached."""

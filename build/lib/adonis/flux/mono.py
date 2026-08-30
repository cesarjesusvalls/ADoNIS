"""Monochromatic neutrino beam."""
from __future__ import annotations

import jax.numpy as jnp

from adonis.flux.base import FluxModel


class Monochromatic(FluxModel):
    def __init__(self, energy: float = 1500.0):
        self.energy = float(energy)

    @property
    def e_nu_nominal(self) -> float:
        return self.energy

    def sample_enu(self, key, n):
        return jnp.full((n,), self.energy)

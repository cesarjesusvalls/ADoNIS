"""Centralised parameters for ADoNIS.

`PhysicsParams` holds the TUNABLE physics knobs that are differentiated and fit via
gradient information.  It is a `typing.NamedTuple`, so JAX automatically registers it as a
pytree (its fields are the leaves) -- `jax.grad`/`jax.jvp` flow through it cleanly, and
`fit` can differentiate w.r.t. any subset.  Keep ONLY differentiable knobs here.

`GenConfig` holds the STATIC (non-differentiated) configuration of a generation run -- the
nuclear-input file, beam energy, sampling ranges, interpolation fidelity, masses.  It is a
frozen dataclass (hashable, usable as a default / cache key).

`DCCKnobs` is kept as an alias of `PhysicsParams` so the migrated DCC code (which referred
to `DCCKnobs`) keeps working during/after the cutover.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from adonis.constants import MQE, M_PI


class PhysicsParams(NamedTuple):
    """Tunable, differentiable physics knobs (pytree leaves)."""
    axial_MA: float = 1.000        # axial mass [GeV]; Q^2-dependent reweight of the axial block
    axial_strength: float = 1.0    # overall axial-current scale
    pw_norm: tuple = ()            # () = no rescale, else length-14 per-partial-wave (1+pw_norm)
    # --- FSI (cascade) knobs: total interaction rates of the in-medium pion [1/fm] ---
    fsi_sigma_scatter: float = 0.30  # pion-nucleon (quasi-elastic) scatter rate
    fsi_sigma_abs: float = 0.20      # pion absorption rate (-> CC0pi)
    # room for future knobs -- add as leaves here.


# Back-compat alias: the migrated DCC modules import/accept `DCCKnobs`.
DCCKnobs = PhysicsParams


@dataclass(frozen=True)
class GenConfig:
    """Static configuration of a generation run (not differentiated)."""
    sf: str = "pke12p_tot.data"       # spectral-function table (nuclear model input)
    e_nu: float = 1500.0              # monochromatic neutrino energy [MeV]
    ep_lo: float = 50.0               # outgoing-lepton energy proposal range [MeV]
    ep_hi: float = 1480.0
    theta_max_deg: float = 180.0      # lepton polar-angle proposal cap [deg]
    m_pi: float = M_PI                # final-state pion mass [MeV]
    m_N: float = MQE                  # final-state nucleon mass [MeV]
    m_lep: float = 0.0                # outgoing charged-lepton mass [MeV] (0=massless e/nu_e; muon=105.658)
    spline: bool = True               # amp interp: True=FMM spline (faithful, DEFAULT). False=bilinear:
                                      # NOT W-faithful (>1% dsigma/dW tail) -- explicit-awareness only
    n_theta: int = 16                 # angular-quadrature grid (for the integrated reference)
    n_phi: int = 16

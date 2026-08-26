"""Centralised parameters for ADoNIS: the physics knobs and the static run configuration.

`PhysicsParams` holds all tunable physics knobs that are differentiated and fit via gradient
information.  It is a `typing.NamedTuple`, so JAX registers it as a pytree (its fields are the
leaves) -- `jax.grad`/`jax.jvp` flow through it cleanly.  Every consumer (reweight, DCC
amplitudes, cascade FSI, the fit) reads these fields by name; there is no parallel knob dict.
`nominal_knobs()` returns the nominal instance; `knob_specs()` is the ordered metadata every
bank label / scan / fit enumerates through.

`ChainConfig` holds the static (non-differentiated) configuration of a generation run.
`DCCKnobs` is an alias of `PhysicsParams` (the DCC modules refer to it by that name).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from adonis.constants import MQE, M_PI

_NPW = 14
_EB_EPS = 1e-2
_ISO = {0: "pp", 1: "pn", 2: "nn"}


class PhysicsParams(NamedTuple):
    """All tunable, differentiable physics knobs (pytree leaves).  Defaults are the nominal values
    (so `PhysicsParams()` is nominal, except pw_norm -- see below).  Grouped by the reweight stage
    each knob enters."""
    M_A_qe: float = 1.0
    M_A_res: float = 1.0
    axial_strength: float = 1.0
    vector_strength: float = 1.0
    mu_p: float = 1.0
    mu_n: float = 1.0
    gep: float = 1.0
    gen: float = 1.0
    res_axial_strength: float = 1.0
    pion_pole: float = 1.0
    delta_strength: float = 1.0
    pw_norm: tuple = ()
    sabs: float = 1.0
    sscat: float = 1.0
    s_piN_elastic: float = 1.0
    s_piN_cex: float = 1.0
    s_conv: float = 1.0
    s_NN_elastic: tuple = (1.0, 1.0, 1.0)
    s_NN_inelastic: tuple = (1.0, 1.0, 1.0)
    f_NN_cex: float = 0.5
    kF_sf: float = 1.0
    Eb_shift: float = _EB_EPS
    sf_norm: float = 1.0
    src_tail: float = 1.0
    qe_norm: float = 1.0
    res_norm: float = 1.0


DCCKnobs = PhysicsParams


def nominal_knobs() -> PhysicsParams:
    """The nominal knob instance.  Identical to the bare default except pw_norm is materialised to 14
    zeros (the fit/reweight path indexes all 14 partial waves; the DCC generation default keeps pw_norm=())."""
    return PhysicsParams(pw_norm=tuple([0.0] * _NPW))


def knob_specs(NOM: PhysicsParams):
    """Ordered (knob_name, component_idx|None, display_label, nominal_value) for the plotted/fitted knobs.
    pw_norm is excluded (cost); sscat is excluded (superseded by the granular s_piN_*/s_NN_* knobs).
    Tuple knobs are expanded per component (s_NN_elastic -> [pp]/[pn]/[nn]).  The single source of
    knob metadata -- bank labels, scan grids, and fits all enumerate knobs through this."""
    out = []
    for name, val in NOM._asdict().items():
        if name in ("pw_norm", "sscat"):
            continue
        if isinstance(val, tuple):
            for i, vi in enumerate(val):
                lab = f"{name}[{_ISO[i]}]" if name.startswith("s_NN") else f"{name}[{i}]"
                out.append((name, i, lab, float(vi)))
        else:
            out.append((name, None, name, float(val)))
    return out


@dataclass(frozen=True)
class ChainConfig:
    """Static configuration of a generation run (not differentiated)."""
    sf: str = "pke12p_tot.data"
    e_nu: float = 1500.0
    ep_lo: float = 50.0
    ep_hi: float = 1480.0
    theta_max_deg: float = 180.0
    m_pi: float = M_PI
    m_N: float = MQE
    m_lep: float = 0.0
    spline: bool = True
    n_theta: int = 16
    n_phi: int = 16

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

_NPW = 14                # DCC partial waves (pw_norm length when engaged)
# Eb_shift nominal: a negligible epsilon (NOT 0) -- the SF removal-energy reweight S(p,E-Eb)/S(p,E)
# has a non-differentiable corner at Eb=0; anchoring nominal at a small positive Eb keeps the gradient
# on the smooth one-sided branch (see reweight.full_knobs / differentiable_knobs.md).
_EB_EPS = 1e-2
_ISO = {0: "pp", 1: "pn", 2: "nn"}     # s_NN_* tuple-component labels


class PhysicsParams(NamedTuple):
    """All tunable, differentiable physics knobs (pytree leaves).  Defaults are the nominal values
    (so `PhysicsParams()` is nominal, except pw_norm -- see below).  Grouped by the reweight stage
    each knob enters."""
    # --- hard vertex (amps2 quadratic): QE + RES leptonic/hadronic current knobs ---
    M_A_qe: float = 1.0            # QE axial dipole mass [GeV] (reweights the QE amps2 record)
    M_A_res: float = 1.0           # RES/DCC axial dipole mass [GeV]
    axial_strength: float = 1.0    # QE axial-current scale
    vector_strength: float = 1.0   # QE vector-current scale
    mu_p: float = 1.0              # proton magnetic form-factor (G_Mp) scale
    mu_n: float = 1.0              # neutron magnetic form-factor (G_Mn) scale
    gep: float = 1.0               # G_Ep scale
    gen: float = 1.0               # G_En scale
    res_axial_strength: float = 1.0  # RES axial-current (C5A) scale
    pion_pole: float = 1.0         # RES pion-pole term scale
    delta_strength: float = 1.0    # P33 Delta(1232) partial-wave strength
    pw_norm: tuple = ()            # DCC per-partial-wave rescale: () = no rescale (the generation
    #                                default), else length-14 (1+pw_norm).  nominal_knobs() sets 14
    #                                zeros so the fit path can index all 14; the bare default stays ()
    #                                so the DCC generation skip-branch (`if pw_norm != ()`) still fires.
    # --- FSI (kind-1 cascade record) reweight knobs ---
    sabs: float = 1.0              # pion absorption rate (s_pi_abs)
    sscat: float = 1.0             # total pi-N scatter rate (superseded by the granular s_piN_* below)
    s_piN_elastic: float = 1.0     # pi-N elastic rate
    s_piN_cex: float = 1.0         # pi-N charge-exchange rate
    s_conv: float = 1.0            # piN -> eta N conversion rate
    s_NN_elastic: tuple = (1.0, 1.0, 1.0)     # NN elastic rate (pp, pn, nn)
    s_NN_inelastic: tuple = (1.0, 1.0, 1.0)   # NN -> N Delta -> NN pi inelastic rate (pp, pn, nn)
    f_NN_cex: float = 0.5          # NN charge-exchange fraction
    # --- spectral function (density-ratio) reweight knobs ---
    kF_sf: float = 1.0             # Fermi-momentum scale of the SF
    Eb_shift: float = _EB_EPS      # removal-energy shift [MeV] (nominal = small +epsilon, see above)
    sf_norm: float = 1.0           # SF overall normalisation
    src_tail: float = 1.0          # short-range-correlation high-|p| tail scale
    # --- channel-level normalisations ---
    qe_norm: float = 1.0
    res_norm: float = 1.0


# Alias: the DCC modules import/accept `DCCKnobs`.
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
    sf: str = "pke12p_tot.data"       # spectral-function table (nuclear model input)
    e_nu: float = 1500.0              # monochromatic neutrino energy [MeV]
    ep_lo: float = 50.0               # outgoing-lepton energy proposal range [MeV]
    ep_hi: float = 1480.0
    theta_max_deg: float = 180.0      # lepton polar-angle proposal cap [deg]
    m_pi: float = M_PI                # final-state pion mass [MeV]
    m_N: float = MQE                  # final-state nucleon mass [MeV]
    m_lep: float = 0.0                # outgoing charged-lepton mass [MeV] (0=massless e/nu_e; muon=105.658)
    spline: bool = True               # amp interp: True=FMM spline (faithful, default). False=bilinear,
                                      # not W-faithful (>1% dsigma/dW tail) -- use only knowingly
    n_theta: int = 16                 # angular-quadrature grid (for the integrated reference)
    n_phi: int = 16

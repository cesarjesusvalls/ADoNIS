"""The single registry every probe-dependent dispatch reads: what dcc_mode, leptonic kind, and
spin-average a probe implies, and whether it is implemented.  Every field is explicit, and an
unrecognised or unimplemented probe raises rather than falling through to a default -- there is no
implicit "same as CC" behavior anywhere here.

Deliberately dependency-light (no jax, no table loads) so any module can import it.  Numeric
constants that need `constants`/`conventions` -- the DCC amplitude normalisations -- are derived
and looked up by probe name in dcc/current.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeSpec:
    """One probe's dispatch data.  Every field is explicit; there is no implicit 'same as CC'
    default."""
    name: str
    dcc_mode: int
    lep_kind: str
    spin_avg: float
    em_propagator: bool
    implemented: bool


_SPECS = (
    ProbeSpec(name="CC", dcc_mode=1, lep_kind="CC_nu", spin_avg=0.5,
              em_propagator=False, implemented=True),
    ProbeSpec(name="EM", dcc_mode=10, lep_kind="EM", spin_avg=0.25,
              em_propagator=True, implemented=True),
    ProbeSpec(name="NC", dcc_mode=-1, lep_kind="NC_nu", spin_avg=0.5,
              em_propagator=False, implemented=True),
)

PROBES = {s.name: s for s in _SPECS}
IMPLEMENTED = tuple(s.name for s in _SPECS if s.implemented)


def probe_spec(probe):
    """The ProbeSpec for `probe`; never returns a default.

    Raises ValueError for an unknown probe (typo, or a stale config value) and NotImplementedError
    for a probe that is registered but has no physics behind it yet -- distinguishing a mistake from
    a roadmap gap."""
    spec = PROBES.get(probe)
    if spec is None:
        raise ValueError(f"unknown probe {probe!r}; known probes are {sorted(PROBES)} "
                         f"(implemented: {list(IMPLEMENTED)})")
    if not spec.implemented:
        raise NotImplementedError(
            f"probe {probe!r} is registered but not implemented -- it would otherwise fall through "
            f"to CC and return charged-current numbers silently. "
            f"Implemented probes: {list(IMPLEMENTED)}.")
    return spec


def dcc_mode(probe):
    """DCC amplitude `mode` for `probe`."""
    return probe_spec(probe).dcc_mode


def probe_for_mode(mode):
    """Inverse lookup: the ProbeSpec whose dcc_mode is `mode`, or raise.  Used by the assembly
    layer, which is parameterised by `mode` rather than by probe name."""
    for s in _SPECS:
        if s.dcc_mode == mode:
            if not s.implemented:
                raise NotImplementedError(
                    f"DCC mode {mode} is probe {s.name!r}, which is registered but not implemented "
                    f"-- `mode < 10` would otherwise route it through the CC branch silently.")
            return s
    raise ValueError(f"unknown DCC mode {mode!r}; known modes are "
                     f"{ {s.dcc_mode: s.name for s in _SPECS} }")

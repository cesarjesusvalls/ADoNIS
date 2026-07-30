"""What a probe IS -- the single registry every probe-dependent dispatch reads.

Before this module, four independent sites decided what a probe meant by testing for ONE probe and
falling through to CC:

    dcc/current.py:208     _mode     = 10 if probe == "EM" else 1
    currents/matrix_element.py:53   lep_kind  = "EM" if probe == "EM" else "CC_nu"
    dcc/channel.py:117     is_em     = (current == "EM")
    dcc/assembly.py:126    if mode < 10:  ...            # mode=-1 (NC) takes the CC branch

Every one of them answers "not EM" with "then it's CC", so a probe nobody has implemented yet --
`"NC"`, or a typo -- produces charged-current numbers **silently**.  `HadronStructure(channels=
NC_CHANNELS)` runs today and returns CC.  That is the failure mode this module exists to remove:
an unknown probe now raises, everywhere, at the point of dispatch.

`"NC"` is registered as KNOWN BUT NOT IMPLEMENTED on purpose.  It raises a different, informative
error than a typo does, and the phase that implements it flips one flag here rather than hunting
for `else`-branches.  See docs/nc_implementation_plan.md.

Deliberately dependency-light (no jax, no table loads) so any module can import it.  Numeric
constants that need `constants`/`conventions` -- the DCC amplitude normalisations -- stay where they
are derived, in dcc/current.py, and are looked up by probe name there.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeSpec:
    """One probe's dispatch data.  Every field is EXPLICIT -- there is no 'same as CC' default,
    because that is exactly how the silent fallbacks got written."""
    name: str
    dcc_mode: int          # DCC amplitude `mode` (amp_dcc_sl_module.f): 1 CC, 10 EM, -1 NC
    lep_kind: str          # kind= for currents/leptonic.py::lepton_current
    spin_avg: float        # initial-state spin/helicity average folded into the cross section
    em_propagator: bool    # photon 1/q^2 (EM) vs a ~constant W/Z propagator (weak)
    implemented: bool      # False = registered so the error message is useful, not usable


_SPECS = (
    # nu N -> l N.  spin_avg 1/2 = the single neutrino helicity averaged over 2 nucleon spins.
    ProbeSpec(name="CC", dcc_mode=1, lep_kind="CC_nu", spin_avg=0.5,
              em_propagator=False, implemented=True),
    # e N -> e' N inclusive (e,e').  spin_avg 1/4 = 2 electron helicities x 2 nucleon spins.
    ProbeSpec(name="EM", dcc_mode=10, lep_kind="EM", spin_avg=0.25,
              em_propagator=True, implemented=True),
    # nu N -> nu N.  mode=-1 for BOTH nu and nubar (currents_pi_dcc.f90:71-101), so the hadronic side
    # is identical between them and only leptonic.py's `anti` differs.  spin_avg 1/2 as CC.
    # RES is implemented (leptonic NC_nu + the literal Fortran vector current + _NORM_NC).
    # NC QE is NOT: dirac.py has no NC branch yet, so the QE path still raises -- see P6.
    ProbeSpec(name="NC", dcc_mode=-1, lep_kind="NC_nu", spin_avg=0.5,
              em_propagator=False, implemented=True),
)

PROBES = {s.name: s for s in _SPECS}
IMPLEMENTED = tuple(s.name for s in _SPECS if s.implemented)


def probe_spec(probe):
    """The ProbeSpec for `probe`, or raise.  NEVER returns a default.

    Raises ValueError on an unknown probe (typo, or a value from a stale config) and
    NotImplementedError on a probe that is registered but has no physics behind it yet.  The two
    are different errors on purpose: one is a mistake, the other is a roadmap item."""
    spec = PROBES.get(probe)
    if spec is None:
        raise ValueError(f"unknown probe {probe!r}; known probes are {sorted(PROBES)} "
                         f"(implemented: {list(IMPLEMENTED)})")
    if not spec.implemented:
        raise NotImplementedError(
            f"probe {probe!r} is registered but not implemented -- it would otherwise fall through "
            f"to CC and return charged-current numbers silently. "
            f"Implemented probes: {list(IMPLEMENTED)}. See docs/nc_implementation_plan.md.")
    return spec


def dcc_mode(probe):
    """DCC amplitude `mode` for `probe`.  Replaces `10 if probe == 'EM' else 1`."""
    return probe_spec(probe).dcc_mode


def probe_for_mode(mode):
    """Inverse lookup: the ProbeSpec whose dcc_mode is `mode`, or raise.  Used by the assembly
    layer, which is parameterised by `mode` rather than by probe name."""
    for s in _SPECS:
        if s.dcc_mode == mode:
            if not s.implemented:
                raise NotImplementedError(
                    f"DCC mode {mode} is probe {s.name!r}, which is registered but not implemented "
                    f"-- `mode < 10` would otherwise route it through the CC branch silently. "
                    f"See docs/nc_implementation_plan.md.")
            return s
    raise ValueError(f"unknown DCC mode {mode!r}; known modes are "
                     f"{ {s.dcc_mode: s.name for s in _SPECS} }")

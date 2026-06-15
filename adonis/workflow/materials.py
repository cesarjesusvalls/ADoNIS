"""Material / target resolution for the ADoNIS workflow API.

A generation config names a target by chemical formula (e.g. "C", "CH", "H2O").  This module parses
the formula into element stoichiometry and resolves each element against a REGISTRY of nuclei for
which ADoNIS actually has the nuclear inputs to run.

PHYSICS LIMIT (honest by construction): only carbon-12 and the free proton are runnable today --
  C : density c12_density.txt + QMC configs (A=12) + spectral pke12{n,p}  -> full cascade
  H : free proton, struck at rest, handled at the PRIMARY level (no cascade), combined as a
      separate bank at analysis time
pke40{n,p} exist for argon but there is NO Ar density / QMC configuration, so the cascade cannot run
on it.  An unsupported element raises UnsupportedMaterial -- we NEVER silently fall back to carbon.
Adding a new nucleus = add its NuclearTarget (density, spectral n/p, qmc, A, Z) once its inputs exist.
"""
from __future__ import annotations
import re
from dataclasses import dataclass


class UnsupportedMaterial(ValueError):
    """Raised when a formula references an element ADoNIS has no nuclear inputs for."""


@dataclass(frozen=True)
class NuclearTarget:
    symbol: str
    A: int
    Z: int
    density: str | None         # DiscreteCascadeConfig.nucleus token; None = free nucleon (no cascade)
    spectral_n: str | None      # neutron spectral-function path; None for free proton
    spectral_p: str | None      # proton spectral-function path
    qmc: str | None             # QMC configuration file; None = no cascade
    free_nucleon: bool          # True -> primary-level only, separate bank, struck nucleon at rest

    @property
    def runs_cascade(self) -> bool:
        return self.density is not None and self.qmc is not None


# Only nuclei with the FULL set of inputs present in the repo.
REGISTRY: dict[str, NuclearTarget] = {
    "C": NuclearTarget("C", 12, 6, "c12_density.txt",
                       "data/Spectral_Functions/pke12n_tot.data",
                       "data/Spectral_Functions/pke12p_tot.data",
                       "QMC_configs.out.gz", free_nucleon=False),
    "H": NuclearTarget("H", 1, 1, None, None, None, None, free_nucleon=True),
}

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    """'CH' -> {'C':1,'H':1} ; 'H2O' -> {'H':2,'O':1} ; 'C8H8' -> {'C':8,'H':8}.  Pure string parse;
    does NOT check the registry (use resolve_targets for that)."""
    s = formula.strip()
    if not s:
        raise ValueError("empty material formula")
    counts: dict[str, int] = {}
    pos = 0
    for m in _TOKEN.finditer(s):
        if m.start() != pos:
            raise ValueError(f"cannot parse material formula {formula!r} (stuck at {s[pos:]!r})")
        el, num = m.group(1), m.group(2)
        counts[el] = counts.get(el, 0) + (int(num) if num else 1)
        pos = m.end()
    if pos != len(s):
        raise ValueError(f"cannot parse material formula {formula!r} (trailing {s[pos:]!r})")
    return counts


def resolve_targets(formula: str) -> list[tuple[NuclearTarget, int]]:
    """Parse + resolve against the REGISTRY.  Returns [(NuclearTarget, stoichiometric_count), ...].
    Raises UnsupportedMaterial for any element ADoNIS cannot run (no silent carbon fallback)."""
    counts = parse_formula(formula)
    out = []
    for el, n in counts.items():
        if el not in REGISTRY:
            raise UnsupportedMaterial(
                f"element {el!r} (in material {formula!r}) is not supported -- ADoNIS has nuclear "
                f"inputs only for {sorted(REGISTRY)}.  Add a NuclearTarget once its density/QMC/"
                f"spectral inputs exist.")
        out.append((REGISTRY[el], n))
    return out


def stoichiometric_weights(formula: str) -> dict[str, float]:
    """Per-element multiplicities as floats, e.g. 'CH' -> {'C':1.0,'H':1.0}.  These weight the
    per-element banks when combining a compound target at analysis time."""
    return {el: float(n) for el, n in parse_formula(formula).items()}

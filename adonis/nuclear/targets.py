"""Nuclear targets: which nuclei ADoNIS has inputs for, and how a chemical formula resolves to them.

A generation config names a target by chemical formula (e.g. "C", "CH", "H2O").  This module parses
the formula into element stoichiometry and resolves each element against a registry of nuclei ADoNIS
has the nuclear inputs to run.  Every per-nucleus input is carried here (the single source of truth)
and threaded into the generators + cascade; nothing is hardcoded downstream.

Each NuclearTarget carries, mirroring ACHILLES's per-nucleus inputs (Nucleus.cc / Configuration.cc /
SpectralFunction.cc):
  density_p / density_n : proton / neutron number-density files (data/nuclear/).  ACHILLES ALWAYS
      reads SEPARATE p and n densities (Nucleus.cc:36-80); for N=Z nuclei (C) both point to the same
      file.  Radius = first r where rho_proton < 1e-6 fm^-3 (Nucleus.cc:49-51); local Fermi momentum
      is PER-SPECIES k_F^s = cbrt(3 pi^2 rho_s) hbarc (Nucleus.cc:212-238).
  configs : nucleon-configuration file (Achilles/data/configurations/).  QMC (C) and RMF (Ar) share
      one on-disk format: header [A Nconfigs maxWgt minWgt] then per config A lines of "isospin x y z",
      a weight line, a blank line (Configuration.cc:19-65).  A is read FROM THE HEADER, not hardcoded.
  spectral_n / spectral_p : neutron / proton spectral-function files (resolved relative to Achilles/).
      pke40 (Ar) has the same on-disk format as pke12 (C), only a larger (p,E) grid.
  free_nucleon : True -> primary-level only (struck nucleon at rest), no cascade, separate bank.

Runnable today: C-12, Ar-40 (full cascade), free-proton H.  O-16 has densities+configs but NO
spectral function (no pke16) -> QE/RES cannot run, so it is intentionally NOT registered.  An
unsupported element raises UnsupportedMaterial -- we NEVER silently fall back to carbon.  Adding a
nucleus = add its NuclearTarget once density_p/n + configs + spectral_n/p all exist.
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
    density_p: str | None
    density_n: str | None
    spectral_n: str | None
    spectral_p: str | None
    configs: str | None
    free_nucleon: bool

    @property
    def runs_cascade(self) -> bool:
        return self.density_p is not None and self.configs is not None


REGISTRY: dict[str, NuclearTarget] = {
    "C": NuclearTarget("C", 12, 6, "c12_density.txt", "c12_density.txt",
                       "data/Spectral_Functions/pke12n_tot.data",
                       "data/Spectral_Functions/pke12p_tot.data",
                       "QMC_configs.out.gz", free_nucleon=False),
    "Ar": NuclearTarget("Ar", 40, 18, "rho_Ar_p.txt", "rho_Ar_n.txt",
                        "data/Spectral_Functions/pke40n_tot.data",
                        "data/Spectral_Functions/pke40p_tot.data",
                        "AR40_configs_RMF_achilles.out.gz", free_nucleon=False),
    "H": NuclearTarget("H", 1, 1, None, None, None, None, None, free_nucleon=True),
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


def spectral_inputs(material: str) -> tuple[int, int, str, str]:
    """(Z, N, proton-SF path, neutron-SF path) for a single-nucleus target.

    The hard-vertex channels each carried their own copy of this table.  Four copies of two nuclei is
    survivable; the failure mode is that adding a nucleus means finding all four, and a channel that
    was missed accepts the material name and then reads the wrong spectral function.
    """
    t = REGISTRY.get(material)
    if t is None or t.free_nucleon:
        raise UnsupportedMaterial(
            f"{material!r} has no spectral-function inputs; known: "
            + ", ".join(k for k, v in REGISTRY.items() if not v.free_nucleon))
    return t.Z, t.A - t.Z, t.spectral_p, t.spectral_n

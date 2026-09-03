"""Guardrail: every physical constant in ADoNIS must match ACHILLES, and the "average nucleon mass
used at a threshold / clip-floor / on-shell energy" bug class must never silently reappear.

Motivated by a constants audit which found that
repeated symbol-scoped mass audits kept missing masses hidden inside `jnp.clip(...)` floors and
thresholds.  This test is use-site aware, not just value aware.

Two independent checks:
  1. test_constant_values_match_achilles  -- every constant value == ACHILLES Constants.hh / Particles.yml.
  2. test_no_avg_mass_at_threshold        -- no NEW `(2*M_N)**2` / `M_N**2` clip-floor/threshold in the
                                             FSI cascade beyond an explicit whitelist of sites verified to
                                             match ACHILLES's OWN average-mass choice.
"""
import re
import pathlib
import pytest

import adonis.constants as C

REPO = pathlib.Path(__file__).resolve().parents[1]

ACHILLES = {
    "mp": 938.27208816, "mn": 939.56542054, "mN": (938.27208816 + 939.56542054) / 2.0,
    "mpip": 139.57018, "mpi0": 134.9764, "mdelta": 1232.25,
    "HBARC": 197.3269804,
    "alpha": 1.0 / 137.035999084, "GF_GEV": 1.1663787e-5,
    "sin2w": 0.23129, "Vud": 0.97367, "Vus": 0.225,
    "MASS_PDG_PROTON": 938.27, "MASS_PDG_NEUTRON": 939.57,
    "MASS_PDG_PIP": 139.571, "MASS_PDG_PI0": 134.977, "MASS_PDG_MUON": 105.7,
}
ACHILLES_ELSEWHERE = {"meta": 548.0, "rho0": 0.17, "AMU": 931.49410248}

REL_TOL = 1e-9


@pytest.mark.parametrize("name,ref", sorted(ACHILLES.items()))
def test_constant_values_match_achilles(name, ref):
    got = getattr(C, name)
    assert abs(got - ref) <= REL_TOL * abs(ref) + 1e-12, f"{name}: ADoNIS {got!r} != ACHILLES {ref!r}"


def test_gf_unit_conversion():
    assert abs(C.GF - C.GF_GEV / 1e6) <= 1e-12 * abs(C.GF)


def test_decentralized_constants_still_match():
    txt = (REPO / "adonis/fsi/cascade.py").read_text()
    assert "_M_ETA_PHYS = 547.862" in txt, "eta mass constant missing/changed in cascade.py"
    oset = (REPO / "adonis/fsi/pion_nuclear_xsec.py").read_text()
    assert "0.17" in oset, "normal density 0.17 (Constant::rho0) missing from pion_nuclear_xsec"


_DANGER = re.compile(r"\(\s*2\s*\*\s*M_N\s*\)\s*\*\*\s*2|/\s*4\.?0?\s*-\s*M_N\s*\*\*\s*2")
_WHITELIST_SUBSTR = (
    "M_N ** 2 - dot4",
    "(_e_phys - M_N)", "[M_N, 0.0, 0.0, 0.0]",
)
_GUARDED_FILES = ["adonis/fsi/cascade.py"]


@pytest.mark.parametrize("relpath", _GUARDED_FILES)
def test_no_avg_mass_at_threshold(relpath):
    offenders = []
    for i, line in enumerate((REPO / relpath).read_text().splitlines(), 1):
        code = line.split("#", 1)[0]
        if _DANGER.search(code) and not any(w in code for w in _WHITELIST_SUBSTR):
            offenders.append(f"{relpath}:{i}: {line.strip()}")
    assert not offenders, (
        "avg nucleon mass used at a threshold/clip-floor (the recurring bug class). Use the PHYSICAL "
        "per-pair mass, or add to the whitelist only after verifying ACHILLES itself uses the average "
        "there.\n" + "\n".join(offenders))

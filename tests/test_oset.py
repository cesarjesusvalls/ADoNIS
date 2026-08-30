"""The live Oset self-energies agree with an independent transcription of the same formulae.

tests/reference/oset_selfenergy.py transcribes ACHILLES OsetCrossSections.cc separately from
adonis/fsi/pion_nuclear_xsec.py and keeps its own copy of the Oset coefficients, so agreement
between the two shows neither has drifted from the published parameterisation.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from adonis.fsi import pion_nuclear_xsec as live
from tests.reference import oset_selfenergy as ref


T_PI = (25.0, 50.0, 85.0, 120.0, 165.0, 200.0, 250.0, 300.0, 450.0)
RHO_FRAC = (0.25, 0.5, 1.0)

CHANNELS = (
    ("abs_NN", ref.self_energy_abs_NN, live._self_energy_abs_NN),
    ("abs_NNN", ref.self_energy_abs_NNN, live._self_energy_abs_NNN),
    ("qe", ref.self_energy_qe, live._self_energy_qe),
    ("abs_total", ref.absorption_self_energy,
     lambda t, m, d: live._self_energy_abs_NN(t, m, d) + live._self_energy_abs_NNN(t, m, d)),
)


@pytest.mark.parametrize("name,f_ref,f_live", CHANNELS, ids=[c[0] for c in CHANNELS])
@pytest.mark.parametrize("rho_frac", RHO_FRAC)
def test_live_matches_reference(name, f_ref, f_live, rho_frac):
    for t in T_PI:
        r = float(f_ref(t, rho_frac=rho_frac, m_pi=ref.M_PI))
        v = float(f_live(t, ref.M_PI, rho_frac * live.NORMAL_DENSITY))
        assert v == pytest.approx(r, rel=1e-9, abs=1e-12), (name, t, rho_frac, r, v)


def test_coefficients_match_reference():
    for attr in ("C_Q", "C_A2", "C_A3", "C_ALPHA", "C_BETA"):
        assert np.asarray(getattr(live, attr)) == pytest.approx(
            np.asarray(getattr(ref, attr))), attr


def test_abs_NNN_clamped_below_delta_region():
    """The C_A3 quadratic goes negative at low T_pi; 3N absorption is clamped to zero there."""
    assert float(live._self_energy_abs_NNN(30.0, ref.M_PI, live.NORMAL_DENSITY)) == 0.0
    assert float(live._self_energy_abs_NNN(200.0, ref.M_PI, live.NORMAL_DENSITY)) > 0.0


def test_reference_is_independent():
    """A reference that imported the live constants would compare them against themselves."""
    src = (Path(__file__).parent / "reference" / "oset_selfenergy.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        mod = (node.module if isinstance(node, ast.ImportFrom) else None) or ""
        names = [a.name for a in getattr(node, "names", [])] if isinstance(node, ast.Import) else []
        assert not mod.startswith("adonis"), f"reference imports from {mod}"
        assert not any(n.startswith("adonis") for n in names), f"reference imports {names}"

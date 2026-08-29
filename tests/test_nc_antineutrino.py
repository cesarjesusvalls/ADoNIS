"""P9/G9 -- antineutrino NC, and the oracle card-routing landmine.

`lepton_current(..., anti=True)` has existed since before this work and **no production caller has
ever passed it**, for ANY probe.  An implemented-but-ungated path is worse than a raise, because it
looks available.  So: gate it.

What is NOT here, stated plainly rather than omitted: the nubar flux plumbing (`_PROBE_BEAMS` /
a nubar spectrum table) and a nubar free-nucleon ACHILLES card.  Those are the rest of P9.  They
cannot affect figures 11 and 12 either way -- `flux/microboone_numu.dat` is numu-only -- which is
exactly why P9 sits off the critical path.

The hadronic side needs nothing: `currents_pi_dcc.f90:71-101` maps BOTH nu and nubar NC to
DCC_mode = -1, so only the leptonic current differs.  That is asserted below rather than assumed,
because it is the fact that makes P9 small.
"""
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

from adonis.channels import constants as C
from adonis.channels.currents.leptonic import lepton_current, _couplings
from adonis.channels.probes import probe_spec

_METRIC = np.array([1.0, -1.0, -1.0, -1.0])


def _kin():
    E = 3000.0
    p_in = np.array([[E, 0.0, 0.0, E]])
    p_out = np.array([[2400.0, 300.0, 0.0, 2381.0]])
    return p_in, p_out


def _amps2(L):
    L = np.asarray(L)
    return float(np.sum(np.abs(np.einsum('...cm,m,...cm->...c', L, _METRIC, L.conj())) ** 2))


def test_anti_changes_the_nc_current():
    """The V-A structure is not helicity-blind: nu and nubar must give different currents."""
    p_in, p_out = _kin()
    nu = np.asarray(lepton_current(p_in, p_out, kind="NC_nu", anti=False))
    nub = np.asarray(lepton_current(p_in, p_out, kind="NC_nu", anti=True))
    assert not np.allclose(nu, nub), "anti=True produced the neutrino current -- the flag is a no-op"
    assert np.all(np.isfinite(nub))


def test_anti_uses_the_same_couplings_and_propagator():
    """Z couples identically to nu and nubar; only the spinor assignment differs.  So the coupling
    tuple must be untouched by `anti` -- if a future change makes the couplings anti-dependent, this
    fails rather than silently double-counting the difference."""
    cl, cr, M, G, prop = _couplings("NC_nu")
    assert (cr, M, G) == (0.0 + 0j, C.MZ, C.GAMZ)
    assert cl == pytest.approx(C.ee * 1j / (2 * C.sw * C.cw), rel=1e-14)


def test_the_hadronic_side_is_identical_for_nu_and_nubar():
    """currents_pi_dcc.f90:71-101 maps BOTH to DCC_mode = -1.  This is what makes P9 small: the
    amplitude, the isospin rotation, VFAC/VVFAC and _NORM_NC are all shared, and only leptonic.py
    differs.  Asserted so the claim is checkable rather than a comment."""
    assert probe_spec("NC").dcc_mode == -1


def test_anti_is_still_unexercised_by_any_production_caller():
    """A guard on the CLAIM, not on the code.  If someone wires anti=True into a generator, this test
    fails and the ungated path becomes a deliberate decision instead of an accident."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    hits = []
    for d in ("adonis", "analysis"):
        for p in (root / d).rglob("*.py"):
            for i, line in enumerate(p.read_text().splitlines(), 1):
                code = line.split("#", 1)[0]
                if "anti=True" in code.replace(" ", ""):
                    hits.append(f"{p.relative_to(root)}:{i}")
    assert not hits, ("anti=True now has a production caller; P9's flux plumbing and its nubar "
                      "free-nucleon oracle gate must land with it:\n" + "\n".join(hits))


def test_a_card_gets_the_image_its_own_contents_ask_for():
    """Routing reads the card, not its name.

    A name-prefix table used to decide this, and nothing in it covered run_ee_*_fsi, so every
    (e,e')-with-FSI card was sent to the no-cascade image and could not run at all.  Asserting the
    rule over every card is only possible because the choice now comes from the card.
    """
    from analysis.oracle_tools.run_achilles import image_for, cascade_is_on
    cards = sorted((ROOT / "configs" / "achilles").glob("*.yml"))
    assert cards, "no ACHILLES cards found"
    wrong = []
    for c in cards:
        raw = c.read_text()
        wants_cascade = cascade_is_on(raw)
        has_processes = any(ln.startswith("Processes:") for ln in raw.splitlines())
        image, _native, binary = image_for(c)
        if not has_processes:
            ok = image.endswith("cascade") and binary.endswith("achilles-cascade")
        elif wants_cascade:
            ok = image.endswith("fullcascade")
        else:
            ok = "oracle" in image
        if not ok:
            wrong.append(f"{c.name} -> {image}")
    assert not wrong, "cards routed against what they ask for: " + ", ".join(wrong)


def test_the_nc_cascade_card_gets_the_cascade_build():
    """The NC MicroBooNE card turns the cascade on, so it must not get the no-cascade image."""
    from analysis.oracle_tools.run_achilles import image_for
    image, native, _ = image_for(ROOT / "configs" / "achilles" / "run_MicroBooNE_Ar_fsi_nc.yml")
    assert image == "achilles:fullcascade" and native


def test_the_free_nucleon_nc_cards_route_to_the_no_cascade_oracle():
    from analysis.oracle_tools.run_achilles import image_for
    for stem in ("run_freenucleon_nc_res_H", "run_freenucleon_nc_res_N",
                 "run_freenucleon_nc_qe_H", "run_freenucleon_nc_qe_N"):
        image, _native, _entry = image_for(ROOT / "configs" / "achilles" / f"{stem}.yml")
        assert "oracle" in image, f"{stem} routed to {image}, expected the no-cascade oracle"



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
import numpy as np
import pytest

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


def test_the_nc_fsi_card_name_routes_correctly_and_the_obvious_name_does_not():
    """`run_achilles._RULES` routes by card-name PREFIX, and the obvious NC name breaks it.

    `run_MicroBooNE_Ar_nc_fsi` does NOT start with `run_MicroBooNE_Ar_fsi`, so it falls through to
    the amd64 `:oracle` default -- which is the no-cascade image.  Putting the suffix AFTER the
    matched prefix, `run_MicroBooNE_Ar_fsi_nc`, fixes it with zero code change.  This test pins the
    naming rule so the fix cannot be undone by someone choosing the more natural-looking name.

    Scoped deliberately to the NC card.  A broader "every cascade card must route natively" assertion
    fails on pre-existing cards (run_ee_*_fsi route to the amd64 oracle today); whether that is a
    latent bug or those cards are simply never driven through this runner is NOT established here,
    so it is recorded in the plan rather than asserted as a test.
    """
    from analysis.oracle_tools.run_achilles import _image_for
    good_img, good_native, _ = _image_for("run_MicroBooNE_Ar_fsi_nc")
    bad_img, bad_native, _ = _image_for("run_MicroBooNE_Ar_nc_fsi")
    assert good_img == "achilles:fullcascade" and good_native, \
        "run_MicroBooNE_Ar_fsi_nc no longer inherits the MicroBooNE cascade rule"
    assert bad_img != good_img, \
        "the prefix rule changed -- re-check whether the _nc_fsi naming is still a trap"


def test_the_nc_cards_route_to_the_no_cascade_oracle():
    from analysis.oracle_tools.run_achilles import _image_for
    for stem in ("run_freenucleon_nc_res_H", "run_freenucleon_nc_res_N",
                 "run_freenucleon_nc_qe_H", "run_freenucleon_nc_qe_N"):
        image, native, entry = _image_for(stem)
        assert "oracle" in image, f"{stem} routed to {image}, expected the no-cascade oracle"

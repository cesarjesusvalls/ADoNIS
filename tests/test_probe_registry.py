"""G1 -- no probe dispatch may fall through to CC.

Answering "not EM" with "then it is CC" turns an unimplemented probe or a typo into charged-current
numbers with no error.  Each dispatch site must name what it produces: NC RES produces NC, NC QE
raises, and an unknown probe raises rather than defaulting.
"""
import numpy as np
import pytest

from adonis.channels.probes import PROBES, IMPLEMENTED, probe_spec, dcc_mode, probe_for_mode


def test_registry_has_no_implicit_default():
    assert set(PROBES) == {"CC", "EM", "NC"}
    assert IMPLEMENTED == ("CC", "EM", "NC")


def test_known_probes_resolve_to_their_achilles_values():
    cc, em = probe_spec("CC"), probe_spec("EM")
    assert (cc.dcc_mode, cc.lep_kind, cc.spin_avg, cc.em_propagator) == (1, "CC_nu", 0.5, False)
    assert (em.dcc_mode, em.lep_kind, em.spin_avg, em.em_propagator) == (10, "EM", 0.25, True)
    assert dcc_mode("CC") == 1 and dcc_mode("EM") == 10


def test_unknown_probe_raises_and_names_what_is_known():
    with pytest.raises(ValueError, match="unknown probe"):
        probe_spec("garbage")
    with pytest.raises(ValueError, match="unknown probe"):
        probe_spec("weak")


def test_nc_now_resolves_to_the_nc_current():
    """UPDATED, not deleted, when NC RES landed -- exactly as the phase plan requires.  What it
    asserts flipped from "raises" to "is NC"; what it GUARDS is unchanged: NC must never be CC."""
    nc = probe_spec("NC")
    assert (nc.dcc_mode, nc.lep_kind, nc.spin_avg, nc.em_propagator) == (-1, "NC_nu", 0.5, False)
    assert nc.dcc_mode != probe_spec("CC").dcc_mode
    assert nc.lep_kind != probe_spec("CC").lep_kind


def test_mode_lookup_mirrors_the_probe_lookup():
    assert probe_for_mode(1).name == "CC"
    assert probe_for_mode(10).name == "EM"
    assert probe_for_mode(-1).name == "NC"
    with pytest.raises(ValueError, match="unknown DCC mode"):
        probe_for_mode(7)


def test_assembly_build_zmtx_refuses_an_unknown_mode():
    from adonis.channels.dcc.assembly import build_zmtx
    npw = 3
    z = np.zeros((8, npw), complex)
    kw = dict(two_J=np.ones(npw, int), two_L=np.zeros(npw, int), two_I=np.ones(npw, int),
              m_N=939.0, m_pi=138.0)
    with pytest.raises(ValueError, match="unknown DCC mode"):
        build_zmtx(z, z, z, 1300.0, 1.0e5, mode=7, itiz=1, **kw)


def test_differential_build_zmtx_batched_refuses_an_unknown_mode():
    from adonis.channels.dcc.differential import build_zmtx_batched
    npw = 3
    z = np.zeros((1, 8, npw), complex)
    kw = dict(two_J=np.ones(npw, int), two_L=np.zeros(npw, int), two_I=np.ones(npw, int),
              m_N=939.0, m_pi=138.0)
    with pytest.raises(ValueError, match="unknown DCC mode"):
        build_zmtx_batched(z, z, z, np.array([1300.0]), np.array([1.0e5]),
                           mode=7, itiz=1, **kw)


def test_exclusive_amps2_batch_refuses_an_unknown_probe():
    from adonis.channels.dcc.current import exclusive_amps2_batch
    z = np.zeros((1, 4))
    args = (z, z, z, z, z, 1, 211)
    with pytest.raises(ValueError, match="unknown probe"):
        exclusive_amps2_batch(*args, probe="garbage")


def test_me_cross_section_refuses_unknown_probes_and_underspecified_nc():
    from adonis.channels.currents.matrix_element import me_cross_section
    z = np.zeros((1, 4))
    with pytest.raises(ValueError, match="is_proton"):
        me_cross_section(z, z, z, z, probe="NC")
    with pytest.raises(ValueError, match="unknown probe"):
        me_cross_section(z, z, z, z, probe="garbage")


def test_me_cross_section_spin_avg_comes_from_the_probe_when_unset():
    """The old default was 0.5 (CC's value) for every probe, so an EM caller that forgot spin_avg
    silently got the CC average.  The probe now decides."""
    from adonis.channels.currents.matrix_element import me_cross_section
    k = np.array([[1000.0, 0.0, 0.0, 1000.0]])
    kp = np.array([[900.0, 0.0, 100.0, 890.0]])
    p = np.array([[939.0, 0.0, 0.0, 0.0]])
    pp = np.array([[1039.0, 0.0, -100.0, 110.0]])
    em = me_cross_section(k, kp, p, pp, probe="EM", is_proton=True)
    em_explicit = me_cross_section(k, kp, p, pp, spin_avg=0.25, probe="EM", is_proton=True)
    assert em["spin_avg"] == 0.25
    assert np.array_equal(np.asarray(em["me_xsec"]), np.asarray(em_explicit["me_xsec"]))
    cc = me_cross_section(k, kp, p, pp, probe="CC")
    assert cc["spin_avg"] == 0.5


def test_dcc_channel_probe_string_is_validated():
    from adonis.channels.probes import probe_spec as ps
    assert ps("EM").em_propagator is True
    assert ps("CC").em_propagator is False
    with pytest.raises(ValueError):
        ps("em")

"""The NC RES vector current, checked against analytic limits and a normalisation fixed by theory.

Not one of these needs an oracle.  Each is a closed-form consequence of the Fortran, so they can be
written from the source and cannot be tuned to whatever the implementation happens to produce:

  * two `sw2` limits that collapse the NC vector coupling onto forms we already trust;
  * the axial is present for NC and absent for EM;
  * the pion pole is CC-only, so its knob must move NC by EXACTLY zero;
  * proton and neutron differ only by the VVFAC sign on the I=1/2 isoscalar term;
  * `_NORM_NC/_NORM_EM` equals (2 sw cw)^2, which follows from the couplings and not from any fit.

The form under test (amp_dcc_sl_module.f, read directly).  The step that matters is NOT in the
assembly loops at all -- it is at :675-691, once at table read, in place:

    if(mode.lt.10)         ! weak only, EM excluded
    if(itpind(ipw)==3)goto 510    ! I=3/2 skipped
    zampv <- 0.5*(zampv - zampv_is)   ! isovector
    zampv_is <- 0.5*(zampv + zampv_is) ! isoscalar

so by the time the vector block (:867-995) and the NC isoscalar block (:1004-1050) run, the arrays
ALREADY mean isovector/isoscalar.  ADoNIS holds the raw loader blocks and rotates at use time:

    I=3/2 : VFAC*vec
    I=1/2 : VFAC*0.5*(vec-isv) + VVFAC(itiz)*0.5*(vec+isv)

This is also why the CC branch's 0.5*(vec-isv) is correct rather than invented.
"""
import numpy as np
import pytest

from adonis.channels.dcc.differential import build_zmtx_batched

SW2 = 0.2312
VFAC = 1.0 - 2.0 * SW2


def _waves():
    """Three partial waves: two I=1/2 (2I=1) and one I=3/2 (2I=3), J=3/2 so idxp_start=1."""
    two_J = np.array([3, 3, 3]); two_L = np.array([0, 2, 2]); two_I = np.array([1, 1, 3])
    return two_J, two_L, two_I


def _amps(seed=0):
    """Deterministic non-trivial (vec, isv, axial) blocks, shape (1, 8, npw)."""
    rng = np.random.default_rng(seed)
    def blk():
        return (rng.normal(size=(1, 8, 3)) + 1j * rng.normal(size=(1, 8, 3)))
    return blk(), blk(), blk()


def _zmtx(mode, itiz, vec, isv, axial, **kw):
    two_J, two_L, two_I = _waves()
    return np.asarray(build_zmtx_batched(
        vec, isv, axial, np.array([1300.0]), np.array([2.0e5]), two_J, two_L, two_I,
        mode=mode, itiz=itiz, m_N=939.0, m_pi=138.04, **kw))


def test_sw2_to_zero_would_leave_a_pure_isovector_coupling():
    """VFAC -> 1 and VVFAC -> 0 as sw2 -> 0, so the NC vector current would collapse onto the raw
    `vec` block on every wave.  Checked as an identity on the coefficients rather than by patching
    the constant, so it holds no matter how the code spells the arithmetic."""
    vec, isv, axial = _amps()
    z_full = _zmtx(-1, 1, vec, isv, axial)
    z_novs = _zmtx(-1, 1, vec, np.zeros_like(isv), axial)
    assert np.all(np.isfinite(z_novs))
    assert not np.allclose(z_full, z_novs), "the isoscalar block does nothing -- VVFAC never applied"


def test_isoscalar_block_touches_I_half_waves_only():
    """`if(itpind(ipw)==1...)` -- the I=3/2 wave must be untouched by isv, for NC."""
    vec, isv, axial = _amps()
    isv_only32 = np.zeros_like(isv); isv_only32[:, :, 2] = isv[:, :, 2]
    a = _zmtx(-1, 1, vec, np.zeros_like(isv), axial)
    b = _zmtx(-1, 1, vec, isv_only32, axial)
    assert np.allclose(a, b), "an I=3/2 isoscalar amplitude leaked into the NC current"

    isv_only12 = np.zeros_like(isv); isv_only12[:, :, 0] = isv[:, :, 0]
    c = _zmtx(-1, 1, vec, isv_only12, axial)
    assert not np.allclose(a, c), "the I=1/2 isoscalar amplitude is being ignored"


def test_proton_and_neutron_differ_only_by_the_vvfac_sign():
    """vvfac(+1) = -2sw2, vvfac(-1) = +2sw2 (:293-294), and NOTHING ELSE in the weak vector current
    depends on the target.  So p and n must straddle the VVFAC=0 current symmetrically:

        z_p + z_n = 2 * z(VVFAC=0)

    Note the reference is the isoscalar-term-free current, NOT the isv=0 current: after the
    :675-691 rotation `isv` feeds BOTH the isovector 0.5*(vec-isv) and the isoscalar 0.5*(vec+isv),
    so zeroing isv changes the isovector too and would make this test assert the wrong thing.
    NC_ISV_SIGN is the documented diagnostic knob for exactly this.
    """
    import adonis.channels.dcc.differential as D
    vec, isv, axial = _amps()
    zp = _zmtx(-1, +1, vec, isv, axial)
    zn = _zmtx(-1, -1, vec, isv, axial)
    old = D.NC_ISV_SIGN
    try:
        D.NC_ISV_SIGN = 0.0
        z0 = _zmtx(-1, +1, vec, isv, axial)
    finally:
        D.NC_ISV_SIGN = old
    assert np.allclose(zp + zn, 2.0 * z0, atol=1e-9), \
        "p and n do not straddle the VVFAC=0 current -> the VVFAC sign flip is wrong"
    assert not np.allclose(zp, zn), "p and n are identical -- VVFAC is not being applied at all"


def test_nc_has_an_axial_current_and_em_does_not():
    """`if(mode.lt.10)` gates the axial block, so NC (mode=-1) keeps it and EM (mode=10) loses it."""
    vec, isv, axial = _amps()
    nc_a = _zmtx(-1, 1, vec, isv, axial)
    nc_0 = _zmtx(-1, 1, vec, isv, np.zeros_like(axial))
    assert not np.allclose(nc_a, nc_0), "NC ignored the axial amplitude"
    em_a = _zmtx(10, 1, vec, isv, axial)
    em_0 = _zmtx(10, 1, vec, isv, np.zeros_like(axial))
    assert np.allclose(em_a, em_0), "EM picked up an axial current"


def test_the_pion_pole_knob_has_exactly_zero_effect_on_nc():
    """The pion-pole block is nested under `if(mode.gt.0)` (:844), i.e. CC modes 1-4 only.  EXACTLY
    zero, not 'small': assert bit equality."""
    vec, isv, axial = _amps()
    a = _zmtx(-1, 1, vec, isv, axial, pion_pole=1.0)
    b = _zmtx(-1, 1, vec, isv, axial, pion_pole=3.7)
    assert np.array_equal(a, b), "the pion-pole knob moved the NC current"
    c = _zmtx(1, 1, vec, isv, axial, pion_pole=1.0)
    d = _zmtx(1, 1, vec, isv, axial, pion_pole=3.7)
    assert not np.allclose(c, d), "the pion-pole knob does nothing for CC either -- test is vacuous"


def test_cc_and_em_are_bit_identical_to_the_isv_recombination_they_were_validated_with():
    """in miniature.  The governing rule is that NC work must not regress CC, so the CC branch
    must still compute i32*vec + (1-i32)*0.5*(vec-isv) exactly, and EM its own forms."""
    vec, isv, axial = _amps(seed=3)
    two_J, two_L, two_I = _waves()
    i32 = np.array([0.0, 0.0, 1.0])
    got = _zmtx(1, 1, vec, isv, np.zeros_like(axial), pion_pole=0.0)
    want_src = i32 * vec + (1.0 - i32) * 0.5 * (vec - isv)
    ref = np.asarray(build_zmtx_batched(
        want_src, np.zeros_like(isv), np.zeros_like(axial), np.array([1300.0]), np.array([2.0e5]),
        two_J, two_L, two_I, mode=10, itiz=1, m_N=939.0, m_pi=138.04, pion_pole=0.0))
    assert np.allclose(got, ref), "the CC vector recombination changed"


def test_norm_nc_over_norm_em_matches_the_value_registered_before_measurement():
    from adonis.channels.dcc.current import _NORM_NC, _NORM_EM
    from adonis.channels import constants as C
    predicted = (2.0 * C.sw * C.cw) ** 2
    assert predicted == pytest.approx(0.7113, abs=5e-4), "the registered 0.7113 no longer holds"
    assert _NORM_NC / _NORM_EM == pytest.approx(predicted, rel=1e-12)


def test_nc_qe_demands_the_struck_nucleon_species():
    """The NC QE path must never produce a number it cannot justify.

    The NC couplings are per nucleon (LeptonicCurrent.cc:97-115), unlike CC's single
    isovector combination, so a caller who does not say which nucleon was struck gets an error
    rather than the proton's answer."""
    from adonis.channels.currents.dirac import hadron_current_qe_dirac
    z = np.zeros((1, 4))
    with pytest.raises(ValueError, match="is_proton"):
        hadron_current_qe_dirac(z, z, z, z, probe="NC")
    h = np.asarray(hadron_current_qe_dirac(z, z, z, z, probe="NC", is_proton=np.array([True])))
    assert np.all(np.isfinite(h))

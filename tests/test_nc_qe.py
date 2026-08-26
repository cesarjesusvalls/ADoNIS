"""G6 -- NC QE: the coupl1 two-sided gate, the strange omission, and the p/n differential.

The centrepiece is the D1 gate.  Whether ACHILLES's `coupl1` is a typo is an INFERENCE, not a fact,
so ADoNIS implements BOTH conventions and MEASURES the difference instead of betting on one:

  use_achilles_nc_coupling=True  must reproduce ACHILLES verbatim;
  use_achilles_nc_coupling=False must differ from it by EXACTLY 1.0396 on the F1/F2 terms of BOTH nucleons, and by
              nothing else -- FA/FAP untouched, CC and EM untouched.

Two-sided is strictly stronger than either single choice: it pins the bug reproduction AND the
correct physics at once, and neither branch can later be "fixed" silently.
"""
import numpy as np
import pytest

from adonis.channels import constants as C
from adonis.channels.currents.dirac import hadron_current_qe_dirac, nc_coupl1
from adonis.channels.currents.form_factors import nucleon_ff

QUIRK_FACTOR = 1.0 / (2.0 * C.sw)


def _kin(n=4):
    rng = np.random.default_rng(11)
    k = np.zeros((n, 4)); k[:, 0] = 1500.0; k[:, 3] = 1500.0
    kp = np.column_stack([np.full(n, 1200.0), rng.normal(0, 200, n), rng.normal(0, 200, n),
                          np.full(n, 1100.0)])
    p = np.column_stack([np.full(n, 939.0), rng.normal(0, 100, n), rng.normal(0, 100, n),
                         rng.normal(0, 100, n)])
    po = p + (k - kp)
    return k, kp, p, po


def _H(is_proton, use_achilles_nc_coupling=False, **kw):
    k, kp, p, po = _kin()
    return np.asarray(hadron_current_qe_dirac(k, kp, p, po, probe="NC",
                                              is_proton=np.full(len(k), is_proton),
                                              use_achilles_nc_coupling=use_achilles_nc_coupling, **kw))


def test_the_quirk_factor_is_exactly_1_0396():
    assert nc_coupl1(use_achilles_nc_coupling=True) / nc_coupl1(use_achilles_nc_coupling=False) == pytest.approx(QUIRK_FACTOR, rel=1e-12)
    assert QUIRK_FACTOR == pytest.approx(1.0396, abs=5e-4)


def test_quirk_true_is_the_achilles_expression_verbatim():
    """LeptonicCurrent.cc:94 -- (ee*i/(4*sin2w*cw)) * (0.5 - 2*sin2w)."""
    want = (C.ee * 1j / (4 * C.sin2w * C.cw)) * (0.5 - 2 * C.sin2w)
    assert nc_coupl1(use_achilles_nc_coupling=True) == pytest.approx(want, rel=1e-14)


def test_quirk_false_is_the_standard_model_coupling():
    """coupl2 = ee*i/(4*sw*cw) fixes the normalisation at g/(2c_W) = ee/(2*sw*cw), so the SM vector
    coupling is ee*i/(2*sw*cw) * (0.5 - 2*sin2w).  Guards the factor-2 error that a single-expression
    switch invites: writing X=sw instead of X=sw/2 would make the DEFAULT half the right coupling."""
    want = (C.ee * 1j / (2 * C.sw * C.cw)) * (0.5 - 2 * C.sin2w)
    assert nc_coupl1(use_achilles_nc_coupling=False) == pytest.approx(want, rel=1e-14)
    assert nc_coupl1(use_achilles_nc_coupling=False) != pytest.approx(
        (C.ee * 1j / (4 * C.sw * C.cw)) * (0.5 - 2 * C.sin2w), rel=1e-6), \
        "coupl1 is half the SM value -- the X=sw factor-2 error"


@pytest.mark.parametrize("is_proton", [True, False])
def test_the_quirk_scales_BOTH_nucleons_F1F2_and_nothing_else(is_proton):
    """The gate as it must be stated.  coupl1 multiplies the STRUCK nucleon's own F1/F2 -- for the
    proton (LeptonicCurrent.cc:97-100) AND for the neutron (:110-111).  An earlier version of this
    gate said "proton only, neutron unchanged", which asserts something false and could never close.

    Isolating the coupl1 term: with FA and the -coupl2 partner both switched off (axial_scale=0 and a
    form-factor scale zeroing the OTHER nucleon's F1/F2), the current is pure coupl1, so the two
    branches must differ by exactly the scalar QUIRK_FACTOR.
    """
    ff_off = {"gen": 0.0, "gmn": 0.0} if is_proton else {"gep": 0.0, "gmp": 0.0}
    a = _H(is_proton, use_achilles_nc_coupling=False, axial_scale=0.0, ff_scale=ff_off)
    b = _H(is_proton, use_achilles_nc_coupling=True, axial_scale=0.0, ff_scale=ff_off)
    nz = np.abs(a) > 1e-12
    assert nz.any(), "the isolated coupl1 current is identically zero -- test is vacuous"
    ratio = b[nz] / a[nz]
    assert np.allclose(ratio, QUIRK_FACTOR, rtol=1e-10), \
        f"{'proton' if is_proton else 'neutron'} F1/F2 did not scale by {QUIRK_FACTOR}"


def test_the_quirk_leaves_the_axial_alone():
    """FA carries coupl2, which is identical in both branches, so a pure-axial current must not move."""
    ff_zero_vec = {"gep": 0.0, "gen": 0.0, "gmp": 0.0, "gmn": 0.0}
    a = _H(True, use_achilles_nc_coupling=False, ff_scale=ff_zero_vec)
    b = _H(True, use_achilles_nc_coupling=True, ff_scale=ff_zero_vec)
    assert np.allclose(a, b, atol=0), "the use_achilles_nc_coupling leaked into the axial current"


def test_the_quirk_cannot_touch_cc_or_em():
    k, kp, p, po = _kin()
    for probe, kw in (("CC", {}), ("EM", dict(is_proton=np.full(len(k), True)))):
        a = np.asarray(hadron_current_qe_dirac(k, kp, p, po, probe=probe, use_achilles_nc_coupling=False, **kw))
        b = np.asarray(hadron_current_qe_dirac(k, kp, p, po, probe=probe, use_achilles_nc_coupling=True, **kw))
        assert np.array_equal(a, b), f"{probe} moved when the NC use_achilles_nc_coupling flag changed"


def test_proton_and_neutron_are_genuinely_different_currents():
    """D3: NC elastic on a NEUTRON has no CC analogue in this repo, so exercise it explicitly."""
    assert not np.allclose(_H(True), _H(False)), "NC QE gives the same current for p and n"


def test_fa_flips_sign_between_proton_and_neutron():
    """FA <- +coupl2 on a proton (:101), -coupl2 on a neutron (:112)."""
    ff_zero_vec = {"gep": 0.0, "gen": 0.0, "gmp": 0.0, "gmn": 0.0}
    hp = _H(True, ff_scale=ff_zero_vec)
    hn = _H(False, ff_scale=ff_zero_vec)
    assert np.allclose(hp, -hn, atol=1e-10), "the NC axial did not flip sign between p and n"


def test_no_strange_form_factors_reach_the_nc_current():
    """ACHILLES computes FormFactors::FAs (FormFactor.cc:92,123) and NEVER consumes it:
    FormFactorInfo::Type (FormFactor.hh:22-48) has no strange entry, so CouplingsFF cannot dispatch
    on one.  Including F1s/F2s/G_A^s would make ADoNIS more physically complete and fail every gate
    here by construction, because every gate is an ACHILLES comparison.  Pinned, not left to a
    comment, so a future contributor cannot 'complete the physics' and silently break every figure."""
    ff = nucleon_ff(np.array([0.3]))
    strange = [k for k in ff if "s" in k.lower() and k not in ("F1p", "F1n", "F2p", "F2n")]
    assert not strange, f"nucleon_ff grew a strange form factor: {strange}"


def test_the_induced_pseudoscalar_DOES_contribute_via_the_shifted_transfer(monkeypatch):
    """Pins a fact that contradicts the obvious argument, so nobody re-derives the wrong one.

    The tempting reasoning is: FAP enters as FAP*(q/mN)*gamma5, i.e. along q^mu; the outgoing NC
    lepton is a massless neutrino; so q.j_lep = 0 and the term drops out.  That is FALSE here.  The
    hadron current rides on `qsh`, the DE FOREST-SHIFTED transfer (qsh[0] = q[0] + p_in_nuc[0] -
    E_in_on), which is NOT the leptonic q -- so the contraction does not vanish.

    Measured: scaling FAP by 137 moves NC amps2 by roughly 600x.  Asserting that it MOVES (rather
    than that it does not) keeps the open question visible: LeptonicCurrent.cc lists no FAP coupling
    for either current, but the paper runs use the FortranModel QE path, which is the authority.
    Until the P6 free-nucleon oracle settles it, NC mirrors CC's validated FAP treatment.
    """
    import adonis.channels.currents.dirac as D
    from adonis.channels.currents.matrix_element import me_cross_section
    k, kp, p, po = _kin()
    mag = np.sqrt(np.sum(kp[:, 1:] ** 2, axis=1))
    kp = np.column_stack([mag, kp[:, 1:]])
    isp = np.full(len(k), True)
    base = np.asarray(me_cross_section(k, kp, p, po, probe="NC", is_proton=isp)["amps2"])

    real_ff = D.nucleon_ff
    def big_fap(Q2, ff_scale=None):
        d = dict(real_ff(Q2, ff_scale=ff_scale)); d["FAP"] = d["FAP"] * 137.0
        return d
    monkeypatch.setattr(D, "nucleon_ff", big_fap)
    bumped = np.asarray(me_cross_section(k, kp, p, po, probe="NC", is_proton=isp)["amps2"])
    assert not np.allclose(base, bumped, rtol=1e-6), \
        "FAP no longer reaches the NC current -- if this is deliberate, update the P6 note too"
    assert np.all(bumped > base), "the FAP contribution changed sign"


def test_nc_qe_requires_is_proton():
    """The NC couplings are PER NUCLEON, so a caller that forgets which nucleon was struck must not
    silently get the proton's."""
    k, kp, p, po = _kin()
    with pytest.raises(ValueError, match="is_proton"):
        hadron_current_qe_dirac(k, kp, p, po, probe="NC")

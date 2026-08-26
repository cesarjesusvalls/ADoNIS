"""Gate: the transcribed pion-absorption charge modes reproduce ACHILLES's isospin partition
(Nucl.Phys. A568, DeltaInteractions.cc).  These proton-count distributions are the physics ADoNIS
must sample to get ejected-proton multiplicity right for any Z/N (the carbon-vs-argon fix)."""
from adonis.fsi.absorption_modes import proton_count_dist, PIP, PIM, PI0, PROTON, NEUTRON

F56, F16 = 5.0 / 6.0, 1.0 / 6.0


def test_proton_count_both_partners_present():
    expect = {
        (PIP, PROTON):  (0.0, 0.0, 1.0),
        (PIP, NEUTRON): (0.0, F16, F56),
        (PIM, PROTON):  (F56, F16, 0.0),
        (PIM, NEUTRON): (1.0, 0.0, 0.0),
        (PI0, PROTON):  (0.0, F56, F16),
        (PI0, NEUTRON): (F16, F56, 0.0),
    }
    for key, exp in expect.items():
        got = proton_count_dist(*key)
        assert all(abs(a - b) < 1e-12 for a, b in zip(got, exp)), (key, got, exp)


def test_normalized():
    for pion in (PIP, PIM, PI0):
        for struck in (PROTON, NEUTRON):
            assert abs(sum(proton_count_dist(pion, struck)) - 1.0) < 1e-12


def test_partner_absent_renormalizes():
    assert proton_count_dist(PIP, NEUTRON, has_p=False, has_n=True) == (0.0, 1.0, 0.0)
    assert proton_count_dist(PIP, NEUTRON, has_p=True, has_n=False) == (0.0, 0.0, 1.0)
    assert proton_count_dist(PIP, NEUTRON, has_p=False, has_n=False) == (0.0, 0.0, 0.0)

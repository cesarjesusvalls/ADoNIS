"""Gate: the transcribed pion-absorption charge modes reproduce ACHILLES's isospin partition
(Nucl.Phys. A568, DeltaInteractions.cc).  These proton-count distributions are the physics ADoNIS
must sample to get ejected-proton multiplicity right for any Z/N (the carbon-vs-argon fix)."""
from adonis.fsi.absorption_modes import proton_count_dist, PIP, PIM, PI0, PROTON, NEUTRON

F56, F16 = 5.0 / 6.0, 1.0 / 6.0


def test_proton_count_both_partners_present():
    # (P0, P1, P2), both p and n partners locally available
    expect = {
        (PIP, PROTON):  (0.0, 0.0, 1.0),       # pi+ p -> pp always
        (PIP, NEUTRON): (0.0, F16, F56),       # pi+ n -> 2p w.p. 5/6 (the Ar-critical case)
        (PIM, PROTON):  (F56, F16, 0.0),       # pi- p -> 0p w.p. 5/6
        (PIM, NEUTRON): (1.0, 0.0, 0.0),       # pi- n -> nn always
        (PI0, PROTON):  (0.0, F56, F16),       # pi0 p -> 1p w.p. 5/6
        (PI0, NEUTRON): (F16, F56, 0.0),       # pi0 n -> 1p w.p. 5/6
    }
    for key, exp in expect.items():
        got = proton_count_dist(*key)
        assert all(abs(a - b) < 1e-12 for a, b in zip(got, exp)), (key, got, exp)


def test_normalized():
    for pion in (PIP, PIM, PI0):
        for struck in (PROTON, NEUTRON):
            assert abs(sum(proton_count_dist(pion, struck)) - 1.0) < 1e-12


def test_partner_absent_renormalizes():
    # pi+ n with NO proton partner nearby -> the 2p (opposite-isospin p partner) mode drops -> 1p only
    assert proton_count_dist(PIP, NEUTRON, has_p=False, has_n=True) == (0.0, 1.0, 0.0)
    # pi+ n with NO neutron partner -> only the 2p (p partner) mode survives
    assert proton_count_dist(PIP, NEUTRON, has_p=True, has_n=False) == (0.0, 0.0, 1.0)
    # no partner at all -> no realizable absorption
    assert proton_count_dist(PIP, NEUTRON, has_p=False, has_n=False) == (0.0, 0.0, 0.0)

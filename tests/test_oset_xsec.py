"""F2 gate: the exact Oset in-medium pion cross sections (port of ACHILLES OsetCrossSections.cc).

These are the REAL absolute cross sections (mb) the ACHILLES cascade uses -- no tuned knobs.
End-to-end validation is the cascade-level pion-spectrum/CC0pi match (test_cascade_real);
here we gate the physics (Delta-peaked, right magnitude, charge-exchange structure) and the
differentiability in the Oset coefficients.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.fsi import pion_nuclear_xsec as ox

M_PI = ox.M_PIP
KF = float(np.cbrt(0.085 * 3 * np.pi ** 2) * ox.HBARC)   # symmetric matter, one species
RHO = 0.17


def _sa(T):
    E = T + M_PI; p = np.sqrt(E * E - M_PI * M_PI)
    return float(ox.abs_cross_section(E, M_PI, p, p / E, KF, RHO))


def _qe(T):
    E = T + M_PI; p = np.sqrt(E * E - M_PI * M_PI)
    return ox.qe_cross_sections(E, M_PI, p, KF, RHO, 0.0)


def test_absorption_delta_peaked():
    """Absorption peaks in the Delta region (T_pi ~ 130-200 MeV) at tens of mb, falls off."""
    Ts = np.array([30, 85, 150, 200, 300, 450])
    sa = np.array([_sa(T) for T in Ts])
    assert np.all(sa > 0)
    pk = Ts[int(sa.argmax())]
    assert 120 <= pk <= 220, pk
    assert 25 < sa.max() < 70, sa.max()                 # in-medium abs ~ tens of mb at the Delta
    assert _sa(450) < 0.3 * sa.max()                    # falls off above the Delta


def test_qe_charge_exchange_structure():
    """The QE channel map includes charge exchange (pi- p -> pi0 n etc.) and forbids the
    double charge flip (pi+ -> pi-)."""
    q = _qe(180)
    assert q[(211, -211)] == 0.0 and q[(-211, 211)] == 0.0   # no double charge exchange
    assert float(q[(-211, 111)]) > 0.0                       # pi- -> pi0 charge exchange present
    assert float(q[(111, 211)]) > 0.0                        # pi0 -> pi+ present
    tot = sum(float(v) for v in q.values())
    assert 40 < tot < 200, tot                               # total QE ~ 100 mb at the Delta


def test_differentiable_in_oset_coeff():
    """The absorption cross section is differentiable in the Oset 2N-absorption coefficient."""
    def f(c0):
        import adonis.fsi.pion_nuclear_xsec as o
        ca2 = o.C_A2.at[0].set(c0)
        x = (200.0) / M_PI; beta = o._quad(x, o.C_BETA)
        # rebuild via the public path: perturb the module constant through a closure
        E = 200.0 + M_PI; p = np.sqrt(E * E - M_PI * M_PI)
        save = o.C_A2; o.C_A2 = ca2
        try:
            return o.abs_cross_section(E, M_PI, p, p / E, KF, RHO)
        finally:
            o.C_A2 = save
    g = float(jax.grad(f)(1.06))
    assert np.isfinite(g) and g != 0.0

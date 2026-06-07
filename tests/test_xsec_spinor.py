"""Self-validation of the Weyl-basis spinor port (no ACHILLES needed): the Dirac identities
  ubar_h(p) u_h(p) = 2m   and   sum_h u_h(p) ubar_h(p) = pslash + m
must hold bit-exact for the ACHILLES spinor construction.  This pins the basis/normalisation
before the spinors feed the leptonic/hadronic currents.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.xsec import spinor as sp


def _pslash(p):
    g = np.asarray(sp.GAMMA)
    return p[0] * g[0] - p[1] * g[1] - p[2] * g[2] - p[3] * g[3]   # E g0 - px g1 - ...


def test_ubar_u_equals_2m():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rng.uniform(100, 1200)
        p3 = rng.uniform(-800, 800, 3)
        E = np.sqrt(m ** 2 + p3 @ p3)
        mom = jnp.asarray([E, *p3])
        for hel in (-1, +1):
            ub = sp.ubar(hel, mom); u = sp.uspinor(hel, mom)
            val = complex(jnp.sum(ub * u))
            assert abs(val - 2 * m) < 1e-6 * m, (val, 2 * m, hel)


def test_completeness_sum_u_ubar():
    rng = np.random.default_rng(1)
    for _ in range(20):
        m = rng.uniform(100, 1200)
        p3 = rng.uniform(-800, 800, 3)
        E = np.sqrt(m ** 2 + p3 @ p3)
        mom = jnp.asarray([E, *p3])
        acc = np.zeros((4, 4), complex)
        for hel in (-1, +1):
            u = np.asarray(sp.uspinor(hel, mom)); ub = np.asarray(sp.ubar(hel, mom))
            acc += np.outer(u, ub)
        target = _pslash(np.asarray([E, *p3])) + m * np.eye(4)
        assert np.max(np.abs(acc - target)) < 1e-6 * m, np.max(np.abs(acc - target))


if __name__ == "__main__":
    test_ubar_u_equals_2m(); test_completeness_sum_u_ubar()
    print("spinor identities OK")

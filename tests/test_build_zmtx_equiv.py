"""Stage-1 unification guard: the scalar assembly.build_zmtx (vmapped, used by the numpy
generator) and the batched differential.build_zmtx_batched (used by the differentiable path)
must produce IDENTICAL zmtx for the CC case with default DBG.  This is the equivalence that
licenses collapsing the generator onto the batched implementation."""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from adonis.primary.dcc.assembly import build_zmtx as build_scalar
from adonis.primary.dcc.differential import build_zmtx_batched
from adonis.primary.dcc.loader import load_cached


def test_build_zmtx_scalar_equals_batched_cc():
    T = load_cached()
    two_J, two_L, two_I = np.asarray(T.pw_2J), np.asarray(T.pw_2L), np.asarray(T.pw_2I)
    npw = len(two_J)
    rng = np.random.default_rng(0)
    N = 64
    def rc(shape): return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    vec = jnp.asarray(rc((N, 8, npw))); isv = jnp.asarray(rc((N, 8, npw))); axial = jnp.asarray(rc((N, 8, npw)))
    W = jnp.asarray(1100.0 + 800.0 * rng.random(N)); Q2 = jnp.asarray(1.0e5 * rng.random(N))
    for itiz in (+1, -1):
        f = lambda v, i, a, w, q: build_scalar(v, i, a, w, q, two_J, two_L, two_I,
                                               mode=1, itiz=itiz, m_N=938.919, m_pi=138.039)
        z_scalar = jax.vmap(f)(vec, isv, axial, W, Q2)
        z_batch = build_zmtx_batched(vec, isv, axial, W, Q2, two_J, two_L, two_I,
                                     mode=1, itiz=itiz, m_N=938.919, m_pi=138.039)
        d = float(jnp.max(jnp.abs(z_scalar - z_batch)))
        scale = float(jnp.max(jnp.abs(z_scalar)))
        assert d / scale < 1e-12, f"itiz={itiz}: max rel diff {d/scale:.2e}"


if __name__ == "__main__":
    test_build_zmtx_scalar_equals_batched_cc()
    print("OK: scalar build_zmtx == batched build_zmtx_batched (CC, both itiz)")

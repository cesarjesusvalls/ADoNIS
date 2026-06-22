"""Pool kind-1 FSI reweight (pool_fsi_reweight): nominal identity, autodiff==FD, and consistency with
the legacy reweight primitives.  Synthetic representative records (no pool compile -> fast); the real
pool record is gated separately by scripts/_pool_rec_check.py (Stage 1/2)."""
import numpy as np
import jax, jax.numpy as jnp
from adonis.fsi.cascade_full import pool_fsi_reweight
from adonis.fsi.cascade_discrete import pion_branch_reweight, nucleon_scat_reweight

N, KP, KN = 200, 16, 64


def _record(seed=0):
    rng = np.random.default_rng(seed)
    nh = rng.integers(0, 6, N)                                   # pion hits per event
    ns = rng.integers(0, 30, N)                                  # nucleon candidate steps per event
    bc = rng.integers(0, 3, (N, KP)).astype(np.int32)           # branch 0/1/2
    sa = rng.uniform(0.2, 3.0, (N, KP)); ss = rng.uniform(0.2, 3.0, (N, KP)); si = rng.uniform(0.0, 0.5, (N, KP))
    hh = rng.random((N, KN)) < 0.3                              # scattered?
    a = rng.uniform(0.05, 3.0, (N, KN))                        # a_nom = pi b^2 / sigma
    return dict(bc=jnp.asarray(bc), sa=jnp.asarray(sa), ss=jnp.asarray(ss), si=jnp.asarray(si),
                nh=jnp.asarray(nh.astype(np.int32)), hh=jnp.asarray(hh), a=jnp.asarray(a),
                ns=jnp.asarray(ns.astype(np.int32)))


def test_nominal_identity():
    w = np.asarray(pool_fsi_reweight(_record(1), 1.0, 1.0))
    assert np.abs(w - 1.0).max() < 1e-12


def test_factorization_matches_legacy():
    r = _record(2)
    w = np.asarray(pool_fsi_reweight(r, 1.3, 0.7))
    wp = np.asarray(pion_branch_reweight((r["bc"], r["sa"], r["ss"], r["si"], r["nh"]), 1.3, 0.7))
    wn = np.asarray(nucleon_scat_reweight((r["hh"], r["a"], r["ns"]), 0.7))
    assert np.allclose(w, wp * wn, atol=1e-12)


def test_autodiff_equals_fd():
    r = _record(3)
    def loss(theta):
        return jnp.sum(pool_fsi_reweight(r, theta[0], theta[1]))
    th0 = jnp.array([1.15, 0.85])
    g_ad = np.asarray(jax.grad(loss)(th0))
    eps = 1e-4
    g_fd = np.array([float((loss(th0.at[d].add(eps)) - loss(th0.at[d].add(-eps))) / (2 * eps)) for d in (0, 1)])
    assert np.allclose(g_ad, g_fd, rtol=1e-4, atol=1e-6), f"ad={g_ad} fd={g_fd}"

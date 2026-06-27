"""Pool kind-1 FSI reweight (pool_fsi_reweight): nominal identity, autodiff==FD, backward-compat, and the
GRANULAR pion knobs (s_piN_elastic / s_piN_cex / s_conv).  Synthetic representative records (no pool
compile -> fast); the real pool record is gated separately by scripts/_pool_rec_check.py."""
import numpy as np
import jax, jax.numpy as jnp
from adonis.fsi.cascade_full import pool_fsi_reweight
from adonis.fsi.cascade_discrete import fsi_pion_reweight, nucleon_scat_reweight, fsi_nucleon_reweight

N, KP, KN = 200, 16, 64


def _record(seed=0):
    rng = np.random.default_rng(seed)
    nh = rng.integers(0, 6, N)                                   # pion hits per event
    ns = rng.integers(0, 30, N)                                  # nucleon candidate steps per event
    bc = rng.integers(0, 4, (N, KP)).astype(np.int32)           # granular channel 0 el/1 cex/2 abs/3 conv
    sa = rng.uniform(0.2, 3.0, (N, KP)); ss = rng.uniform(0.2, 3.0, (N, KP)); si = rng.uniform(0.0, 0.5, (N, KP))
    ss_el = ss * rng.uniform(0.0, 1.0, (N, KP))                 # elastic part (<= total scatter)
    hh = rng.random((N, KN)) < 0.3                              # scattered?
    a = rng.uniform(0.05, 3.0, (N, KN))                        # a_nom = pi b^2 / sigma_tot
    iso = rng.integers(0, 3, (N, KN)).astype(np.int32)          # nucleon pair-iso pp/pn/nn
    finel = rng.uniform(0.0, 0.4, (N, KN))                      # inelastic fraction
    inel = rng.random((N, KN)) < finel                         # realized inelastic (only matters at a hit)
    return dict(bc=jnp.asarray(bc), sa=jnp.asarray(sa), ss_el=jnp.asarray(ss_el), ss=jnp.asarray(ss),
                si=jnp.asarray(si), nh=jnp.asarray(nh.astype(np.int32)), hh=jnp.asarray(hh),
                a=jnp.asarray(a), iso=jnp.asarray(iso), finel=jnp.asarray(finel), inel=jnp.asarray(inel),
                ns=jnp.asarray(ns.astype(np.int32)))


def test_nominal_identity():
    w = np.asarray(pool_fsi_reweight(_record(1), 1.0, 1.0))
    assert np.abs(w - 1.0).max() < 1e-12


def test_granular_nominal_identity():
    w = np.asarray(pool_fsi_reweight(_record(1), 1.0, 1.0, s_piN_elastic=1.0, s_piN_cex=1.0, s_conv=1.0))
    assert np.abs(w - 1.0).max() < 1e-12


def test_factorization():
    r = _record(2)
    w = np.asarray(pool_fsi_reweight(r, 1.3, 0.7))
    wp = np.asarray(fsi_pion_reweight((r["bc"], r["sa"], r["ss_el"], r["ss"], r["si"], r["nh"]),
                                      1.3, 0.7, 0.7, 1.0))     # back-compat: s_el=s_cex=sscat, s_conv=1
    wn = np.asarray(nucleon_scat_reweight((r["hh"], r["a"], r["ns"]), 0.7))
    assert np.allclose(w, wp * wn, atol=1e-12)


def test_nucleon_granular_backcompat_equals_legacy():
    """fsi_nucleon_reweight with s_el=s_inel=sscat (all iso) == legacy nucleon_scat_reweight(., sscat)."""
    r = _record(7)
    srec = (r["hh"], r["a"], r["iso"], r["finel"], r["inel"], r["ns"])
    s3 = jnp.full((3,), 0.75)
    wg = np.asarray(fsi_nucleon_reweight(srec, s3, s3))
    wl = np.asarray(nucleon_scat_reweight((r["hh"], r["a"], r["ns"]), 0.75))
    assert np.allclose(wg, wl, atol=1e-12)


def test_nucleon_granular_nominal_identity():
    r = _record(8)
    w = np.asarray(pool_fsi_reweight(r, 1.0, 1.0, s_NN_elastic=(1., 1., 1.), s_NN_inelastic=(1., 1., 1.)))
    assert np.abs(w - 1.0).max() < 1e-12


def test_nucleon_granular_autodiff_equals_fd():
    r = _record(9)
    def loss(theta):
        return jnp.sum(pool_fsi_reweight(r, 1.0, 1.0,
                       s_NN_elastic=theta[:3], s_NN_inelastic=theta[3:]))
    th0 = jnp.array([1.1, 0.9, 1.2, 0.8, 1.15, 0.95])
    g_ad = np.asarray(jax.grad(loss)(th0))
    eps = 1e-4
    g_fd = np.array([float((loss(th0.at[d].add(eps)) - loss(th0.at[d].add(-eps))) / (2 * eps)) for d in range(6)])
    assert np.allclose(g_ad, g_fd, rtol=2e-4, atol=1e-6), f"ad={g_ad} fd={g_fd}"


def test_backcompat_default_equals_explicit():
    """pool_fsi_reweight(r,sabs,sscat) == explicit s_piN_elastic=s_piN_cex=sscat (defaults reproduce legacy)."""
    r = _record(4)
    w0 = np.asarray(pool_fsi_reweight(r, 1.2, 0.8))
    w1 = np.asarray(pool_fsi_reweight(r, 1.2, 0.8, s_piN_elastic=0.8, s_piN_cex=0.8, s_conv=1.0))
    assert np.allclose(w0, w1, atol=1e-12)


def test_cex_knob_affects_hits_not_nohits():
    """s_piN_cex enters the total-sigma normalization D at EVERY hit (correct likelihood-ratio behavior),
    so it changes any event WITH a pion hit, but leaves no-hit events (nh==0) exactly unchanged."""
    r = _record(5)
    base = np.asarray(pool_fsi_reweight(r, 1.0, 1.0))
    bumped = np.asarray(pool_fsi_reweight(r, 1.0, 1.0, s_piN_cex=1.5))
    nh = np.asarray(r["nh"])
    assert np.all(np.abs(bumped - base)[nh == 0] < 1e-12)       # no pion hit -> identity
    assert np.any(np.abs(bumped - base)[nh > 0] > 1e-9)         # any hit -> changed (D coupling)


def test_autodiff_equals_fd():
    r = _record(3)
    def loss(theta):
        return jnp.sum(pool_fsi_reweight(r, theta[0], theta[1]))
    th0 = jnp.array([1.15, 0.85])
    g_ad = np.asarray(jax.grad(loss)(th0))
    eps = 1e-4
    g_fd = np.array([float((loss(th0.at[d].add(eps)) - loss(th0.at[d].add(-eps))) / (2 * eps)) for d in (0, 1)])
    assert np.allclose(g_ad, g_fd, rtol=1e-4, atol=1e-6), f"ad={g_ad} fd={g_fd}"


def test_granular_autodiff_equals_fd():
    r = _record(6)
    def loss(theta):
        return jnp.sum(pool_fsi_reweight(r, 1.0, 1.0, s_piN_elastic=theta[0], s_piN_cex=theta[1], s_conv=theta[2]))
    th0 = jnp.array([1.1, 0.9, 1.2])
    g_ad = np.asarray(jax.grad(loss)(th0))
    eps = 1e-4
    g_fd = np.array([float((loss(th0.at[d].add(eps)) - loss(th0.at[d].add(-eps))) / (2 * eps)) for d in range(3)])
    assert np.allclose(g_ad, g_fd, rtol=1e-4, atol=1e-6), f"ad={g_ad} fd={g_fd}"

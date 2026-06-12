"""Gates for the cascade walk/weight split (kind-1 records):
1. the reweight from compressed walk records equals the in-propagation weight (any theta);
2. the (sabs, sscat) gradient through the records is exact (autodiff == central FD);
3. records never overflow their slot capacity.
The walk is theta-independent, so these records let a fit run each cascade ONCE and
re-evaluate/differentiate the weights cheaply (see scripts/cc0pi_tune_adonis.py)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from adonis.core.event import EventRecord
from adonis.fsi.cascade_discrete import (DiscreteCascadeFSI, DiscreteNucleonFSI,
                                         DiscreteCascadeConfig, pion_branch_reweight,
                                         nucleon_scat_reweight)

N = 1500
CFG = DiscreteCascadeConfig(cylinder=False, step=0.04, max_steps=260)


def _events(n=N):
    kp, kd, ki = jax.random.split(jax.random.PRNGKey(7), 3)
    pm = 200.0 + 200.0 * jax.random.uniform(kp, (n,))
    d = jax.random.normal(kd, (n, 3)); d = d / jnp.linalg.norm(d, axis=1, keepdims=True)
    E = jnp.sqrt(pm ** 2 + 139.57018 ** 2)
    p_pi = jnp.concatenate([E[:, None], d * pm[:, None]], axis=1)
    p_N = jnp.concatenate([jnp.full((n, 1), 1000.0), jnp.zeros((n, 2)), jnp.full((n, 1), 350.0)], axis=1)
    return EventRecord(k=jnp.zeros((n, 4)), kp=jnp.zeros((n, 4)), p_struck=jnp.zeros((n, 4)),
                       p_pi=p_pi, p_N=p_N, w=jnp.ones(n), channel=jnp.zeros(n, jnp.int32),
                       pid_pi=jnp.full(n, 211, jnp.int32), pid_N=jnp.full(n, 2212, jnp.int32),
                       pid_Ni=jnp.where(jax.random.uniform(ki, (n,)) < 0.5, 2112, 2212).astype(jnp.int32),
                       W=jnp.zeros(n), Q2_adj=jnp.zeros(n))


def _walk():
    ev = _events()
    pion = DiscreteCascadeFSI(CFG); pion.apply(None, ev, key=jax.random.PRNGKey(11), sabs=1.3, sscat=0.7)
    nuc = DiscreteNucleonFSI(CFG); nuc.apply(None, ev, key=jax.random.PRNGKey(12), sscat=0.7)
    return pion, nuc


_PION, _NUC = _walk()


def test_records_match_propagation():
    w_rec = np.asarray(pion_branch_reweight(_PION.last_brec, 1.3, 0.7))
    assert np.allclose(w_rec, np.asarray(_PION.last_w_fsi), rtol=1e-12, atol=0)
    s1, s2, has_ko = _NUC.last_srec
    w_rec2 = np.asarray(nucleon_scat_reweight(s1, 0.7)
                        * jnp.where(has_ko, nucleon_scat_reweight(s2, 0.7), 1.0))
    assert np.allclose(w_rec2, np.asarray(_NUC.last_w_scat), rtol=1e-12, atol=0)
    # nominal theta -> all weights exactly 1
    assert np.allclose(np.asarray(pion_branch_reweight(_PION.last_brec, 1.0, 1.0)), 1.0, atol=1e-12)
    assert np.allclose(np.asarray(nucleon_scat_reweight(s1, 1.0)), 1.0, atol=1e-12)


def test_records_capacity():
    assert int(jnp.max(_PION.last_brec[4])) <= _PION.last_brec[1].shape[1]
    s1, s2, _ = _NUC.last_srec
    assert int(jnp.max(s1[2])) <= s1[1].shape[1]
    assert int(jnp.max(s2[2])) <= s2[1].shape[1]


def test_reweight_grad_closure():
    """autodiff == central FD for sum-of-weights in (sabs, sscat) through the records."""
    s1, s2, has_ko = _NUC.last_srec

    def f(th):
        return (jnp.sum(pion_branch_reweight(_PION.last_brec, th[0], th[1]))
                + jnp.sum(nucleon_scat_reweight(s1, th[1])
                          * jnp.where(has_ko, nucleon_scat_reweight(s2, th[1]), 1.0)))

    th0 = jnp.array([1.1, 0.9]); eps = 1e-4
    g_ad = np.asarray(jax.grad(f)(th0))
    g_fd = np.array([(float(f(th0 + eps * e)) - float(f(th0 - eps * e))) / (2 * eps)
                     for e in (jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))])
    rel = np.abs(g_ad - g_fd) / np.clip(np.abs(g_fd), 1e-12, None)
    assert rel.max() < 1e-5, (g_ad, g_fd)

"""Gate for the early-exit cascade walk (DiscreteCascadeConfig.early_exit):
the while_loop walk (stop when no particle can interact again) must be BIT-EXACT equal to
the reference lax.scan in every returned output -- same per-step keys, and skipped steps
are identity operations (dead particles, or pions outside the radius moving outward).
n_trunc is excluded for the pion (its early-exit semantics deliberately exclude inert
walkers: it counts only particles that could still interact at the cap)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import dataclasses
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from adonis.core.event import EventRecord
from adonis.fsi.cascade_discrete import DiscreteCascadeFSI, DiscreteNucleonFSI, DiscreteCascadeConfig

N = 3000


def _events(n=N):
    kp, kd, ki = jax.random.split(jax.random.PRNGKey(7), 3)
    pm = 200.0 + 200.0 * jax.random.uniform(kp, (n,))
    d = jax.random.normal(kd, (n, 3)); d = d / jnp.linalg.norm(d, axis=1, keepdims=True)
    E = jnp.sqrt(pm ** 2 + 139.57018 ** 2)
    p_pi = jnp.concatenate([E[:, None], d * pm[:, None]], axis=1)
    p_N = jnp.concatenate([jnp.full((n, 1), 1100.0), jnp.zeros((n, 2)), jnp.full((n, 1), 600.0)], axis=1)
    return EventRecord(k=jnp.zeros((n, 4)), kp=jnp.zeros((n, 4)), p_struck=jnp.zeros((n, 4)),
                       p_pi=p_pi, p_N=p_N, w=jnp.ones(n), channel=jnp.zeros(n, jnp.int32),
                       pid_pi=jnp.full(n, 211, jnp.int32), pid_N=jnp.full(n, 2212, jnp.int32),
                       pid_Ni=jnp.where(jax.random.uniform(ki, (n,)) < 0.5, 2112, 2212).astype(jnp.int32),
                       W=jnp.zeros(n), Q2_adj=jnp.zeros(n))


_EV = _events()


def _pair(cyl):
    base = DiscreteCascadeConfig(cylinder=cyl, step=0.04, max_steps=300)
    return (dataclasses.replace(base, early_exit=False), dataclasses.replace(base, early_exit=True))


def _eq(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def test_pion_early_exit_bitexact():
    for cyl in (False, True):
        outs = []
        for cfg in _pair(cyl):
            o = DiscreteCascadeFSI(cfg)
            ev = o.apply(None, _EV, key=jax.random.PRNGKey(11), sabs=1.3, sscat=0.8)
            outs.append((ev.p_pi, ev.pid_pi, o.last_absorbed, o.last_abs_proton, o.last_w_fsi,
                         o.last_nseg, o.last_nsc, *o.last_brec))
        for a, b in zip(*outs):
            assert _eq(a, b), f"pion mismatch (cylinder={cyl})"


def test_nucleon_early_exit_bitexact():
    for cyl in (False, True):
        outs = []
        for cfg in _pair(cyl):
            o = DiscreteNucleonFSI(cfg)
            ev = o.apply(None, _EV, key=jax.random.PRNGKey(12), sscat=0.8)
            s1, s2, has_ko = o.last_srec
            outs.append((ev.p_N, o.last_w_scat, has_ko, *s1, *s2))
        for a, b in zip(*outs):
            assert _eq(a, b), f"nucleon mismatch (cylinder={cyl})"

"""End-to-end regression of the ADoNIS high-level API: recover M_A from a small
pseudo-dataset using DCCSinglePion (sample/weight contract) + analysis.fit, on the
integrated Q^2 distribution.  Exercises PhysicsParams, the channel, observables, and the
forward-mode fitter."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis import ChainConfig, DCCSinglePion, PhysicsParams, observables as obs
from adonis.core.event import EventRecord
from adonis.reweight.fit import fit_scalar

cfg = ChainConfig(spline=False)
ch = DCCSinglePion(cfg)
N = 20_000 if _os.environ.get("ADONIS_CI_FAST") else 40_000
MA_TRUE, MA_INIT = 1.20, 0.90
edges = np.linspace(0, 1.6e6, 25)
nb = len(edges) - 1


def ev_of(S):
    z = jnp.zeros(S["n"])
    return EventRecord(k=S["k_lab"], kp=S["kp_lab"], p_struck=S["p_struck"], p_pi=S["p_pi"],
                       p_N=S["p_N"], w=z, channel=z, pid_pi=z, pid_N=z, pid_Ni=z,
                       W=S["W"], Q2_adj=S["Q2_adj"])


def q2_idx(S):
    v = obs.Q2(ev_of(S))
    return jax.lax.stop_gradient(jnp.clip(jnp.searchsorted(jnp.asarray(edges), v) - 1, 0, nb - 1))


# data target at MA_TRUE
Sd = ch.sample(jax.random.PRNGKey(101), N)
wd, _ = ch.weight(PhysicsParams(M_A_res=MA_TRUE), Sd); wd = np.asarray(wd)
idd = np.asarray(q2_idx(Sd))
d = np.bincount(idd, weights=wd, minlength=nb) / N
derr = np.sqrt(np.bincount(idd, weights=wd ** 2, minlength=nb)) / N + 1e-15

# model (two-replica via halves)
Sm = ch.sample(jax.random.PRNGKey(202), N); H = N // 2
idm = q2_idx(Sm)
dj, ej = jnp.asarray(d), jnp.asarray(derr)


def loss(MA):
    w, _ = ch.weight(PhysicsParams(M_A_res=MA), Sm)
    mA = jax.ops.segment_sum(w[:H], idm[:H], num_segments=nb) / H
    mB = jax.ops.segment_sum(w[H:], idm[H:], num_segments=nb) / H
    return jnp.sum((mA - dj) * (mB - dj) / ej ** 2) / nb


MAf, hist, losses, rel = fit_scalar(loss, MA_INIT, lr=0.05, iters=60, label="Q2")
print(f"M_A_true={MA_TRUE}  init={MA_INIT}  ->  recovered {MAf:.4f}  (grad AD/FD rel {rel:.1e})")
ok = abs(MAf - MA_TRUE) < 0.1 and rel < 1e-3
print("PASS" if ok else "FAIL")


def test_ma_recovered():
    assert abs(MAf - MA_TRUE) < 0.1, f"recovered {MAf}"
    assert rel < 1e-3, f"grad AD/FD rel {rel}"

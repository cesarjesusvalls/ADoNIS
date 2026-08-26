"""Gate: fold_final_state reproduces dcc_fold_full's dsigma/dW, dsigma/dQ2 and total
weight in expectation (the un-integration is unbiased), and produces sane final states."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.channels.dcc.amplitudes import DCCKnobs
from adonis.channels.dcc.structure import HadronStructure
from adonis.channels.dcc.fold_integrated import fold_full_events
from adonis.channels.dcc.channel import fold_final_state
from adonis import kinematics as obs

import jax.numpy as jnp
_FAST = bool(_os.environ.get("ADONIS_CI_FAST"))
hs = HadronStructure(n_theta=16, n_phi=16, spline=False)
n = 100_000 if _FAST else 200_000
CHUNK = 5_000
key = jax.random.PRNGKey(7)


def _chunks(total, k0):
    k = k0
    for c in range(0, total, CHUNK):
        k, sub = jax.random.split(k)
        yield min(CHUNK, total - c), sub


_W, _Q, _w = [], [], []
for nc, sub in _chunks(n, key):
    W, Q2, w = fold_full_events(DCCKnobs(), sub, n=nc, hs=hs)
    _W.append(np.asarray(W)); _Q.append(np.asarray(Q2)); _w.append(np.asarray(w))
Wf, Q2f, wf = np.concatenate(_W), np.concatenate(_Q), np.concatenate(_w)

_evs = [fold_final_state(DCCKnobs(), sub, n=nc, hs=hs) for nc, sub in _chunks(n, jax.random.PRNGKey(8))]
ev = type(_evs[0])(**{f: jnp.concatenate([getattr(e, f) for e in _evs]) for f in _evs[0]._fields})
Wm, Q2m, wm = np.asarray(obs.W(ev)), np.asarray(obs.Q2(ev)), np.asarray(ev.w)

print(f"total weight: integrated {wf.sum():.6e}   final-state {wm.sum():.6e}   "
      f"ratio {wm.sum()/wf.sum():.4f}")
print(f"kept (w!=0): integrated {(wf!=0).sum()}   final-state {(wm!=0).sum()}")

def norm_hist(v, w, edges):
    h = np.histogram(v, bins=edges, weights=w)[0]
    return h / np.diff(edges) / h.sum()

We = np.linspace(1080, 1650, 40)
Q2e = np.linspace(0, 1.5e6, 40)
dWf, dWm = norm_hist(Wf, wf, We), norm_hist(Wm, wm, We)
dQf, dQm = norm_hist(Q2f, wf, Q2e), norm_hist(Q2m, wm, Q2e)
print(f"dsigma/dW   max rel diff (bins>1% peak): {np.max(np.abs(dWm-dWf)[dWf>0.01*dWf.max()]/dWf[dWf>0.01*dWf.max()]):.3f}")
print(f"dsigma/dQ2  max rel diff (bins>1% peak): {np.max(np.abs(dQm-dQf)[dQf>0.01*dQf.max()]/dQf[dQf>0.01*dQf.max()]):.3f}")

from adonis.constants import M_PI, MQE
from adonis.channels.dcc.conventions import kin_m_pi
keep = wm != 0
mpi2 = obs.mink_dot(ev.p_pi, ev.p_pi)
mN2 = obs.mink_dot(ev.p_N, ev.p_N)
print(f"pion mass  recon: {np.sqrt(np.asarray(mpi2)[keep]).mean():.3f}  (M_PI={M_PI:.3f})")
print(f"nucl mass  recon: {np.sqrt(np.asarray(mN2)[keep]).mean():.3f}  (MQE={MQE:.3f})")
W_had = np.asarray(obs.W(ev))
W_true = np.asarray(ev.W)
print(f"W(p_pi+p_N) vs W(q+p_struck) max|d| (kept): {np.max(np.abs(W_had-W_true)[keep]):.3e} MeV")

cts = np.asarray(obs.cos_theta_star(ev))[keep]
print(f"cos_theta* range [{cts.min():.3f},{cts.max():.3f}]  mean {np.average(cts, weights=wm[keep]):.3f}")
print(f"channel fractions (weighted): {[float(np.sum(wm[(np.asarray(ev.channel)==c)&keep])/wm[keep].sum()) for c in range(3)]}")


def test_unintegration_unbiased():
    assert 0.95 < wm.sum() / wf.sum() < 1.05
    for dm, df in ((dWm, dWf), (dQm, dQf)):
        sel = df > 0.10 * df.max()
        rel = np.abs(dm - df)[sel] / df[sel]
        assert np.median(rel) < 0.08, np.median(rel)
        assert np.percentile(rel, 90) < 0.20, np.percentile(rel, 90)


def test_final_state_onshell():
    assert abs(np.sqrt(np.asarray(mpi2)[keep]).mean() - kin_m_pi(M_PI)) < 1.0
    assert abs(np.sqrt(np.asarray(mN2)[keep]).mean() - MQE) < 1.0
    assert np.max(np.abs(W_had - W_true)[keep]) < 1e-3

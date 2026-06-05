"""Gate: fold_final_state reproduces dcc_fold_full's dsigma/dW, dsigma/dQ2 and total
weight in expectation (the un-integration is unbiased), and produces sane final states."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.primary.dcc.amplitudes import DCCKnobs
from adonis.primary.dcc.structure import HadronStructure
from adonis.primary.dcc.fold_integrated import fold_full_events
from adonis.primary.dcc.channel import fold_final_state
from adonis import observables as obs

hs = HadronStructure(n_theta=16, n_phi=16, spline=False)   # bilinear = faster for the test
n = 200_000
key = jax.random.PRNGKey(7)

# integrated fold (reference)
Wf, Q2f, wf = fold_full_events(DCCKnobs(), key, n=n, hs=hs)
Wf, Q2f, wf = map(np.asarray, (Wf, Q2f, wf))

# full-final-state fold
ev = fold_final_state(DCCKnobs(), key, n=n, hs=hs)
Wm, Q2m, wm = np.asarray(obs.W(ev)), np.asarray(obs.Q2(ev)), np.asarray(ev.w)

print(f"total weight: integrated {wf.sum():.6e}   final-state {wm.sum():.6e}   "
      f"ratio {wm.sum()/wf.sum():.4f}")
print(f"kept (w!=0): integrated {(wf!=0).sum()}   final-state {(wm!=0).sum()}")

# distribution agreement (normalized), shared binning
def norm_hist(v, w, edges):
    h = np.histogram(v, bins=edges, weights=w)[0]
    return h / np.diff(edges) / h.sum()

We = np.linspace(1080, 1650, 40)
Q2e = np.linspace(0, 1.5e6, 40)
dWf, dWm = norm_hist(Wf, wf, We), norm_hist(Wm, wm, We)
dQf, dQm = norm_hist(Q2f, wf, Q2e), norm_hist(Q2m, wm, Q2e)
print(f"dsigma/dW   max rel diff (bins>1% peak): {np.max(np.abs(dWm-dWf)[dWf>0.01*dWf.max()]/dWf[dWf>0.01*dWf.max()]):.3f}")
print(f"dsigma/dQ2  max rel diff (bins>1% peak): {np.max(np.abs(dQm-dQf)[dQf>0.01*dQf.max()]/dQf[dQf>0.01*dQf.max()]):.3f}")

# final-state sanity: pion + nucleon on-shell, W consistency
from adonis.constants import M_PI, MQE
keep = wm != 0
mpi2 = obs._mink_dot(ev.p_pi, ev.p_pi)
mN2 = obs._mink_dot(ev.p_N, ev.p_N)
print(f"pion mass  recon: {np.sqrt(np.asarray(mpi2)[keep]).mean():.3f}  (M_PI={M_PI:.3f})")
print(f"nucl mass  recon: {np.sqrt(np.asarray(mN2)[keep]).mean():.3f}  (MQE={MQE:.3f})")
W_had = np.asarray(obs.W(ev))
W_true = np.asarray(ev.W)
print(f"W(p_pi+p_N) vs W(q+p_struck) max|d| (kept): {np.max(np.abs(W_had-W_true)[keep]):.3e} MeV")

# cos theta* should be roughly uniform-ish weighted; just check range + finite
cts = np.asarray(obs.cos_theta_star(ev))[keep]
print(f"cos_theta* range [{cts.min():.3f},{cts.max():.3f}]  mean {np.average(cts, weights=wm[keep]):.3f}")
print(f"channel fractions (weighted): {[float(np.sum(wm[(np.asarray(ev.channel)==c)&keep])/wm[keep].sum()) for c in range(3)]}")

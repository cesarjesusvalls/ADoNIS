"""Differentiability gate: the full-final-state weight is differentiable in the physics
knob M_A, with autodiff == central finite-difference (kind-1 reweighting -> exact)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.primary.dcc.amplitudes import DCCKnobs
from adonis.primary.dcc.structure import HadronStructure
from adonis.primary.dcc.channel import fold_final_state
from adonis import observables as obs

hs = HadronStructure(n_theta=12, n_phi=12, spline=False)
N = 30_000
key = jax.random.PRNGKey(3)


def total_xsec(MA):
    ev = fold_final_state(DCCKnobs(axial_MA=MA), key, n=N, hs=hs)
    return jnp.sum(ev.w)


def dsig_dW_bin(MA, lo=1180.0, hi=1240.0):
    ev = fold_final_state(DCCKnobs(axial_MA=MA), key, n=N, hs=hs)
    Wv = obs.W(ev)
    insel = (Wv > lo) & (Wv < hi)
    return jnp.sum(jnp.where(insel, ev.w, 0.0))


for name, f in [("total xsec", total_xsec), ("dsig/dW peak bin", dsig_dW_bin)]:
    g_ad = float(jax.grad(f)(1.0))
    eps = 2e-3
    g_fd = float((f(1.0 + eps) - f(1.0 - eps)) / (2 * eps))
    rel = abs(g_ad - g_fd) / (abs(g_ad) + abs(g_fd) + 1e-30)
    print(f"{name:18s}  d/dMA: autodiff {g_ad:+.6e}  finite-diff {g_fd:+.6e}  rel {rel:.2e}  "
          f"{'OK' if rel < 1e-3 else 'CHECK'}")

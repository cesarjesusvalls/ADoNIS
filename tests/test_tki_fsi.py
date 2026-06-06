"""Phase C gates: TKI observables + topology signals under the cascade FSIModel.

Ties Phase C (observables/signals) to Phase D (FSI): the cascade must (a) partition events
into CC1pi (surviving pion) and CC0pi (absorbed) topologies, (b) drag the single-transverse
imbalance delta_pT into a high-imbalance tail (the canonical FSI signature), and (c) keep
the whole post-FSI observable differentiable in the FSI knobs.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis import GenConfig, DCCSinglePion, PhysicsParams, observables as obs
from adonis.primary.dcc.channel import assemble_event
from adonis.fsi.cascade import ToyCascadeFSI, CascadeConfig
from adonis.signal import CC1Pi, CC0Pi

FAST = bool(os.environ.get("ADONIS_CI_FAST"))
N = 8_000 if FAST else 30_000
ch = DCCSinglePion(GenConfig(spline=False))
fsi = ToyCascadeFSI(CascadeConfig(seed=4))
params = PhysicsParams(fsi_sigma_scatter=0.35, fsi_sigma_abs=0.22)

S = ch.sample(jax.random.PRNGKey(1), N)
w, LWc = ch.weight(params, S)
pre = assemble_event(S, w, LWc)
post = fsi.apply(params, pre, key=jax.random.PRNGKey(2))


def test_topology_partition():
    """CC1pi and CC0pi exactly partition the events; FSI absorption creates a CC0pi rate."""
    s1 = np.asarray(CC1Pi().select(post))
    s0 = np.asarray(CC0Pi().select(post))
    assert np.all(s1 ^ s0)                                  # mutually exclusive + exhaustive
    cc0 = float(np.sum(np.asarray(post.w)[s0]) / np.sum(np.asarray(pre.w)))
    assert 0.2 < cc0 < 0.8, cc0                              # a sizable absorption rate


def test_fsi_broadens_delta_pT():
    """The cascade drags CC1pi delta_pT into a high-imbalance tail (FSI signature)."""
    ctr_cut = 300.0
    dpre = np.asarray(obs.delta_pT(pre)); wpre = np.asarray(pre.w)
    s1 = np.asarray(CC1Pi().select(post))
    dpost = np.asarray(obs.delta_pT(post))[s1]; wpost = np.asarray(post.w)[s1]
    tail_pre = wpre[dpre > ctr_cut].sum() / wpre.sum()
    tail_post = wpost[dpost > ctr_cut].sum() / wpost.sum()
    assert tail_post > 3 * tail_pre, (tail_pre, tail_post)   # FSI strongly enhances the tail
    assert tail_post > 0.08, tail_post


def test_post_fsi_observable_differentiable():
    """A post-FSI TKI observable (CC1pi mean delta_pT) is differentiable in sigma_scatter:
    autodiff vs finite difference with a frozen proposal (exact kind-1 reweighting)."""
    base = PhysicsParams(fsi_sigma_scatter=0.35, fsi_sigma_abs=0.22)
    key = jax.random.PRNGKey(5)

    def mean_dpt(sc):
        p = base._replace(fsi_sigma_scatter=sc)
        ev = fsi.apply(p, pre, key=key, proposal=base)
        sel = ev.pid_pi != 0
        wsel = ev.w * sel
        return jnp.sum(wsel * obs.delta_pT(ev)) / (jnp.sum(wsel) + 1e-9)

    g_ad = float(jax.grad(mean_dpt)(0.35))
    eps = 1e-3
    g_fd = float((mean_dpt(0.35 + eps) - mean_dpt(0.35 - eps)) / (2 * eps))
    rel = abs(g_ad - g_fd) / (abs(g_ad) + abs(g_fd) + 1e-30)
    assert rel < 1e-3, (g_ad, g_fd, rel)

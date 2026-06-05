"""Phase D gate: the toy cascade's differentiable machinery (recovered from f60947a).

Proves the score-function / expected-value-deposit machinery survives a stochastic FSI
transform on toy physics (the plan's "most important early checkpoint"): the
differentiable (weighted) histogram is UNBIASED vs the hard (sampled) one, and gradients
flow through the whole cascade to all five physics knobs (M_A, m_Δ, Γ_Δ, σ_scatter,
σ_abs). Toy physics -> closure gate only, no oracle. See docs/phases/PHASE_D.md.
"""
import numpy as np
import jax
import jax.numpy as jnp

from adonis.fsi.toy.integrate_full import weighted_histogram, sampled_histogram, ConfigFull


def _truth(cfg):
    return (cfg.true_MA, cfg.true_mDelta, cfg.true_GammaDelta,
            cfg.true_sigma_scatter, cfg.true_sigma_abs)


def test_toy_forward_unbiased():
    """weighted (differentiable) == sampled (hard) histogram in expectation."""
    cfg = ConfigFull()
    truth = _truth(cfg)
    ks = jax.random.split(jax.random.PRNGKey(0), 16)
    samp = np.mean([np.asarray(sampled_histogram(truth, k, cfg, 40_000)) for k in ks], axis=0)
    wt = np.mean([np.asarray(weighted_histogram(truth, k, cfg, 40_000)) for k in ks], axis=0)
    relL2 = np.linalg.norm(wt - samp) / np.linalg.norm(samp)
    assert relL2 < 0.05, relL2


def test_toy_cascade_5param_differentiable():
    """jax.grad of a χ² loss flows through the stochastic cascade to ALL five knobs,
    finite and non-zero (the headline Phase-D differentiability proof)."""
    cfg = ConfigFull()
    truth = _truth(cfg)
    key = jax.random.PRNGKey(0)
    data = jnp.asarray(np.mean(
        [np.asarray(sampled_histogram(truth, k, cfg, 40_000))
         for k in jax.random.split(key, 8)], axis=0))

    def loss(p):
        return jnp.sum((weighted_histogram(p, key, cfg, 60_000) - data) ** 2)

    g = jax.grad(loss)(truth)
    g = np.array([float(x) for x in g])
    assert np.all(np.isfinite(g)), g
    assert np.all(np.abs(g) > 0), g                 # every knob has a live gradient

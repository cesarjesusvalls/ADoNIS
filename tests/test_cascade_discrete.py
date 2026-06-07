"""Gate: the differentiable JAX discrete-Glauber cascade reproduces the ACHILLES pi+-12C
transparency oracle (correct radius=10 fm normalisation).  This is the production cascade that
replaces the continuum (which was ~2x low for missing the transverse exp(-pi b^2/sigma) reach).
"""
import numpy as np
import jax
import pytest
from pathlib import Path

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
_QMC = ROOT.parent / "Achilles" / "data" / "configurations" / "QMC_configs.out.gz"
_ORC = ROOT / "data" / "oracle" / "cascade_pip_c12_virt_abs.csv"
pytestmark = pytest.mark.skipif(not (_QMC.exists() and _ORC.exists()),
                                reason="QMC configs or oracle not present")

BMAX = 10.0   # ACHILLES beam-disk radius -> piR2 = pi*10^2*10 = 3141.6 mb


def _sig(plab, n=15000, seed=0):
    import jax.numpy as jnp
    from adonis.fsi.cascade_discrete import propagate_discrete, sample_nucleons, DiscreteCascadeConfig
    import adonis.fsi.oset_xsec as ox
    cfg = DiscreteCascadeConfig(step=0.05, max_steps=300, seed=seed)
    ka, kb, kn, kp = jax.random.split(jax.random.PRNGKey(seed), 4)
    b = BMAX * jnp.sqrt(jax.random.uniform(ka, (n,))); ang = 2 * jnp.pi * jax.random.uniform(kb, (n,))
    Epi = np.sqrt(ox.M_PIP ** 2 + plab ** 2)
    pos0 = jnp.stack([b * jnp.cos(ang), b * jnp.sin(ang), jnp.full(n, -6.5)], 1)
    p0 = jnp.tile(jnp.array([Epi, 0., 0., plab]), (n, 1)); ch = jnp.zeros(n, jnp.int32)
    npos, nmom, nisp = sample_nucleons(kn, n, cfg)
    _, _, absb, nsc, _abs_lead = propagate_discrete(pos0, p0, ch, npos, nmom, nisp, cfg, kp)
    geo = np.pi * BMAX ** 2 * 10.0
    absb = np.asarray(absb)
    return geo * absb.mean(), geo * ((np.asarray(nsc) > 0) | absb).mean()


def test_discrete_matches_oracle_at_delta():
    orc = np.loadtxt(_ORC)
    for p in (215.0, 275.0):
        sa, sr = _sig(p)
        oa = np.interp(p, orc[:, 0], orc[:, 2]); orr = np.interp(p, orc[:, 0], orc[:, 1])
        assert 0.85 < sa / oa < 1.20, (p, "abs", sa, oa)        # absolute, within oracle stats
        assert 0.85 < sr / orr < 1.20, (p, "reac", sr, orr)


def test_discrete_absorption_fraction():
    """p-averaged-ish absorption fraction at the Delta ~ 0.3 (the ACHILLES VirtRes value)."""
    sa, sr = _sig(275.0)
    assert 0.25 < sa / sr < 0.38, sa / sr

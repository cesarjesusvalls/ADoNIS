"""FLAT/streaming FSI record == DENSE per-event record, through the ACTUAL pool cascade.

The flat layout (adonis/fsi/cascade.py, docs/logbook/fsi_record_cap_techdebt.md) streams the FSI
reweight record into one flat (TOTAL,) buffer sized by n*E[interactions] (tail-free), instead of the
legacy per-event (n, K_max) dense buffer.  The reweight is a scatter-add by event index, so the two must
agree to float64 precision at any knob vector.  This runs a real QE cascade both ways and compares.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


def _run(flat, n=300, seed=0):
    import sys
    import adonis.fsi.cascade as CF
    from adonis.channels import qe as qe_xsec
    from analysis.campaign import tune as T
    CF.FLAT_FSI_REC = flat
    caps = (n * 8, n * 24) if flat else T.REC_CAPS
    qe = qe_xsec.sample_importance(n, seed=seed)
    res = CF.cascade_nucleus(jnp.zeros((n, 4)), jnp.asarray(qe["p_out"]), jnp.zeros(n, jnp.int32),
                             jnp.full(n, 2112, jnp.int32), jnp.full(n, 2212, jnp.int32),
                             T.POOLCFG(seed=2), jax.random.PRNGKey(1), channel="qe", rec_caps=caps)
    rec = res[4]
    comp = CF.compact_fsi_record({**rec, "n_events": n}); comp["n_events"] = n
    ws = [np.asarray(CF.pool_fsi_reweight(comp, a, b)) for a, b in [(1.0, 1.0), (1.3, 0.8), (0.7, 1.4)]]
    return np.stack(ws), (len(comp["p_eidx"]), len(comp["n_eidx"]))


def test_flat_equals_dense_qe_cascade():
    wf, sf = _run(flat=True)
    wd, sd = _run(flat=False)
    assert sf == sd, f"slot counts differ: flat {sf} vs dense {sd}"
    assert np.array_equal(wf[0], wd[0]), "nominal reweight not bit-identical"
    assert np.allclose(wf, wd, rtol=1e-12, atol=0), f"max|dw|={np.abs(wf-wd).max():.2e}"


def test_flat_overflow_raises():
    """Undersized flat budget must fail LOUD (not silently truncate) -- via compact_fsi_record AND the
    direct reweight path (pool_fsi_reweight on a raw flat record)."""
    import sys
    import adonis.fsi.cascade as CF
    from adonis.channels import qe as qe_xsec
    from analysis.campaign import tune as T
    CF.FLAT_FSI_REC = True
    n = 300
    qe = qe_xsec.sample_importance(n, seed=0)
    res = CF.cascade_nucleus(jnp.zeros((n, 4)), jnp.asarray(qe["p_out"]), jnp.zeros(n, jnp.int32),
                             jnp.full(n, 2112, jnp.int32), jnp.full(n, 2212, jnp.int32),
                             T.POOLCFG(seed=2), jax.random.PRNGKey(1), channel="qe", rec_caps=(10, 10))
    rec = res[4]
    with pytest.raises(ValueError, match="overflow"):
        CF.compact_fsi_record({**rec, "n_events": n})
    with pytest.raises(ValueError, match="overflow"):
        CF.pool_fsi_reweight({**rec, "n_events": n}, 1.2, 0.9)

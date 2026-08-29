"""Every name adonis/__init__ advertises resolves, and the composed generator runs.

The exports are lazy, so a stale entry raises only on attribute access; instantiating Generator with a
primary channel is the composition the package documents.
"""
from __future__ import annotations

import pytest

import adonis


def test_every_advertised_name_resolves():
    """Each entry in the lazy table imports.  A stale entry raises only on attribute access."""
    bad = {}
    for name in adonis._LAZY:
        try:
            getattr(adonis, name)
        except Exception as e:                       # noqa: BLE001 - report all, not the first
            bad[name] = f"{type(e).__name__}: {e}"
    assert not bad, f"advertised names that do not resolve: {bad}"


def test_composed_generator_produces_events():
    """Generator(channel) draws a proposal and builds a lab final state.

    This is the composition the package advertises: a primary Channel wired to a flux and nuclear
    model, with FSI defaulting to the identity.
    """
    jax = pytest.importorskip("jax")
    gen = adonis.Generator(adonis.DCCSinglePion(adonis.ChainConfig()))
    ev = gen.generate(jax.random.PRNGKey(0), 256)

    import numpy as np
    w = np.asarray(ev.w)
    assert w.shape == (256,)
    assert np.isfinite(w).all(), "non-finite event weights"
    assert (w >= 0).all(), "negative event weights"
    for field in ("k", "kp", "p_struck", "p_pi", "p_N"):
        v = np.asarray(getattr(ev, field))
        assert v.shape == (256, 4), f"{field}: expected (n,4) four-vectors, got {v.shape}"
        assert np.isfinite(v).all(), f"{field}: non-finite components"


def test_generator_default_fsi_is_the_identity():
    """FSI defaults to NoFSI, so the final state equals the primary one."""
    jax = pytest.importorskip("jax")
    import numpy as np
    cfg = adonis.ChainConfig()
    a = adonis.Generator(adonis.DCCSinglePion(cfg)).generate(jax.random.PRNGKey(1), 128)
    b = adonis.Generator(adonis.DCCSinglePion(cfg), fsi=adonis.NoFSI()).generate(jax.random.PRNGKey(1), 128)
    assert np.array_equal(np.asarray(a.p_pi), np.asarray(b.p_pi))
    assert np.array_equal(np.asarray(a.w), np.asarray(b.w))

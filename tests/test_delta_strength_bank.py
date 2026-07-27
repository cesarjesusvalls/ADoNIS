"""delta_strength (P33 Delta-1232 strength) bank-level validation: builds a tiny event bank carrying
the hv_res_delta record and gates nominal-identity (forward untouched) + autodiff==FD + channel
correctness.  The underlying amps2 reweight is gated separately in test_res_strength_reweight (wave 5)."""
import os, glob, subprocess, sys, tempfile
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def tiny_bank():
    d = tempfile.mkdtemp(prefix="delta_bank_")
    env = {**os.environ, "CC0PI_N": "600", "CHUNK": "600"}
    subprocess.run([sys.executable, "analysis/t2k/differentiability/event_bank.py", d],
                   cwd=REPO, env=env, check=True, capture_output=True)
    B = {}
    for f in sorted(glob.glob(f"{d}/chunk_*.npz")):
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            B[k] = z[k]
    return B


def test_delta_strength_bank(tiny_bank):
    import jax; jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from adonis.reweight import bank_reweight as BR
    from adonis.reweight.full_knobs import nominal_knobs
    B = tiny_bank
    assert "hv_res_delta_a" in B and "res_p_N" in B
    JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    w0 = np.asarray(BR.weight_jit(JB, nom, grids))
    w1 = np.asarray(BR.weight_jit(JB, {**nom, "delta_strength": 1.0}, grids))
    assert np.max(np.abs(w1 - w0)) == 0.0                       # nominal identity: forward untouched
    ch = np.asarray(B["channel"])
    w12 = np.asarray(BR.weight_jit(JB, {**nom, "delta_strength": 1.2}, grids))
    assert np.max(np.abs(w12[ch == 0] - w0[ch == 0])) == 0.0    # QE never touched
    assert np.mean(np.abs(w12[ch == 1] - w0[ch == 1]) > 1e-9) > 0.05   # some RES (Delta) rows move
    f = lambda ds: jnp.sum(BR.bank_weight(JB, {**nom, "delta_strength": ds}, grids))
    g_ad = float(jax.grad(f)(1.0)); eps = 1e-4
    g_fd = (float(f(1.0 + eps)) - float(f(1.0 - eps))) / (2 * eps)
    assert abs(g_ad) > 0 and abs(g_ad - g_fd) / abs(g_fd) < 1e-3   # clean, nonzero gradient

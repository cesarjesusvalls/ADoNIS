"""Validate the differentiable DCC amplitude interpolator (diffpi/dcc.py).

Checks: grid-point exactness, interior bilinear vs NumPy, vmap batching, the knob
layer (axial_strength, pw_norm), and differentiability w.r.t. both the knobs and the
kinematics (W) via finite differences.

Run:  python validate_dcc_interp.py
"""
import numpy as np
import jax
import jax.numpy as jnp

from diffpi.dcc_loader import load_cached
from diffpi.dcc import DCCAmplitudes, DCCKnobs, numpy_bilinear

t = load_cached()
D = DCCAmplitudes(t)
full_vec = t.vec.sum(axis=-1)[:, :, :, :, 0]      # NumPy reference (n_q2,n_w,n_idx,n_pw)
ok = True

def chk(name, cond):
    global ok; ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

# 1) grid point -> exact table value
iw, iq = 5, 3
vec, _, _ = D.amplitudes(float(t.W[iw]), float(t.Q2[iq]))
chk("grid-point matches table", np.allclose(np.asarray(vec), full_vec[iq, iw], atol=1e-9))

# 2) interior point matches NumPy bilinear
W1, Q21 = 1232.0, 300000.0
vec1, _, ax1 = D.amplitudes(W1, Q21)
chk("interior matches numpy bilinear",
    np.allclose(np.asarray(vec1), numpy_bilinear(t, full_vec, W1, Q21), rtol=1e-6, atol=1e-9))

# 3) vmap batching
Ws, Qs = jnp.array([1150., 1232., 1300.]), jnp.array([0., 1e5, 3e5])
vb = jax.vmap(lambda w, q: D.amplitudes(w, q)[0])(Ws, Qs)
chk("vmap batched shape (3,n_idx,n_pw)", vb.shape == (3, vec1.shape[0], vec1.shape[1]))

# 4) axial_strength scales axial only
v_a, _, ax_a = D.amplitudes(W1, Q21, DCCKnobs(axial_strength=2.0))
chk("axial_strength scales axial only",
    np.allclose(np.asarray(ax_a), 2.0 * np.asarray(ax1)) and np.allclose(np.asarray(v_a), np.asarray(vec1)))

# 5) grad w.r.t. axial_strength == finite diff
def loss_as(a):
    _, _, ax = D.amplitudes(W1, Q21, DCCKnobs(axial_strength=a))
    return jnp.sum(jnp.abs(ax) ** 2)
chk("grad wrt axial_strength == finite diff",
    np.isclose(float(jax.grad(loss_as)(1.0)), float((loss_as(1.0 + 1e-4) - loss_as(1.0 - 1e-4)) / 2e-4), rtol=1e-4))

# 6) grad w.r.t. W (reparameterised kinematics) == finite diff
def loss_W(w):
    v, _, _ = D.amplitudes(w, Q21); return jnp.sum(jnp.abs(v) ** 2)
chk("grad wrt W == finite diff",
    np.isclose(float(jax.grad(loss_W)(W1)), float((loss_W(W1 + 1e-2) - loss_W(W1 - 1e-2)) / 2e-2), rtol=1e-3))

# 7) per-partial-wave norm knob scales p33 column
i33 = D.labels.index("p33")
pw = tuple(0.1 if i == i33 else 0.0 for i in range(D.n_pw))
v_pw, _, _ = D.amplitudes(W1, Q21, DCCKnobs(pw_norm=pw))
chk("pw_norm scales p33 column by 1.1", np.allclose(np.asarray(v_pw[:, i33]), 1.1 * np.asarray(vec1[:, i33])))

print(f"\nDCC interpolator: {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)

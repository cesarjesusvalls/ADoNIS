"""Consistency gate for angular_diff: (1) batched build_zmtx == scalar build_zmtx;
(2) grid-summed differential tensor == current_and_tensor integrated tensor."""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from diffpi.dcc_loader import load_cached
from diffpi.dcc import DCCAmplitudes, DCCKnobs
from diffpi.hadron_assembly import build_zmtx, angular_kernel, current_and_tensor
from diffpi.hadron_xsec import CC_CHANNELS
from diffpi.achilles_const import MQE as M_N, M_PI
from diffpi import angular_diff as ad

t = load_cached()
amp = DCCAmplitudes(t)
twoJ, twoL, twoI = np.asarray(t.pw_2J), np.asarray(t.pw_2L), np.asarray(t.pw_2I)

n_theta = n_phi = 16

# common quadrature grid (same g-order as angular_kernel)
ct, wct = np.polynomial.legendre.leggauss(n_theta)
phi = 2 * np.pi * np.arange(n_phi) / n_phi
wphi = 2 * np.pi / n_phi
theta_g, phi_g, w_g = [], [], []
for it in range(n_theta):
    for ip in range(n_phi):
        theta_g.append(np.arccos(ct[it])); phi_g.append(phi[ip]); w_g.append(wct[it] * wphi)
theta_g, phi_g, w_g = map(np.asarray, (theta_g, phi_g, w_g))
G = theta_g.shape[0]

max_zmtx, max_rel = 0.0, 0.0
for (W0, Q20) in [(1100.0, 0.05e6), (1232.0, 0.2e6), (1500.0, 1.0e6)]:
    vec_s, isv_s, axial_s = amp.amplitudes(W0, Q20, DCCKnobs())
    for ci, ch in enumerate(CC_CHANNELS):
        zmtx_s = build_zmtx(vec_s, isv_s, axial_s, W0, Q20, twoJ, twoL, twoI,
                            mode=ch.mode, itiz=ch.itiz, m_N=M_N, m_pi=M_PI)
        zmtx_b = ad.build_zmtx_batched(vec_s[None], isv_s[None], axial_s[None],
                                       jnp.array([W0]), jnp.array([Q20]), twoJ, twoL, twoI,
                                       mode=ch.mode, itiz=ch.itiz, m_N=M_N, m_pi=M_PI)[0]
        max_zmtx = max(max_zmtx, float(jnp.max(jnp.abs(zmtx_b - zmtx_s))))

        ker = angular_kernel(twoJ, twoL, twoI, tcrz=ch.tcrz, tiz=ch.tiz, tpinz=ch.tpinz,
                             tpiz=ch.tpiz, n_theta=n_theta, n_phi=n_phi)
        W_int_ref = current_and_tensor(zmtx_s, ker)
        pre = ad.precompute_diff_coeffs(twoJ, twoL, twoI, tcrz=ch.tcrz, tiz=ch.tiz,
                                        tpinz=ch.tpinz, tpiz=ch.tpiz)
        assert np.array_equal(pre["ixi1_map"], ker["ixi1_map"])
        Kfac = ad.angular_factor(theta_g, phi_g, pre)
        zmtx_grid = jnp.broadcast_to(zmtx_b, (G, 8, zmtx_b.shape[-1]))
        zj = ad.differential_current(zmtx_grid, Kfac, pre["ixi1_map"])
        W_int_diff = jnp.einsum("g,gmn->mn", jnp.asarray(w_g), ad.differential_tensor(zj))
        rel = float(jnp.max(jnp.abs(W_int_diff - W_int_ref)) / (jnp.max(jnp.abs(W_int_ref)) + 1e-30))
        max_rel = max(max_rel, rel)
        print(f"  W={W0:.0f} Q2={Q20/1e6:.2f} ch{ci}: build_zmtx d={float(jnp.max(jnp.abs(zmtx_b-zmtx_s))):.1e}  intg rel={rel:.1e}")

print(f"[1] build_zmtx batched vs scalar  max|d| = {max_zmtx:.3e}")
print(f"[2] grid-summed differential vs integrated  max rel = {max_rel:.3e}")
ok = max_zmtx < 1e-10 and max_rel < 1e-9
print("PASS" if ok else "FAIL")

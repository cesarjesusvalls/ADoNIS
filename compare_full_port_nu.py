"""Phase-2.5 physics test (step 2 of 2): does the full hadron-tensor port move the
weak dsigma/dW toward the neutrino oracle?

Reuses the SAME event sampling as the validated CC fold (so the proposal + spectral
fold + flux are held fixed), and swaps ONLY the hadronic weight:
  (a) diagonal  -- old dcc_xsec sigma(W,Q2) [transverse-only, no multipole interference]
  (b) full W_T  -- full-hadron-tensor transverse response (exact multipole norms,
                   per-channel isospin CGs, same-L interference restored)
  (c) full T+L  -- W_T + eps*W_L  (longitudinal response added; eps from the EM-like
                   virtual-boson polarisation as a PLACEHOLDER until the real CC
                   lepton tensor / V-A contraction lands in step 1->option 1)
all folded through the identical Gamma flux + spectral function, normalised to unit
area, overlaid on the oracle.  This isolates the hadronic improvement from the flux.

Run:  python compare_full_port_nu.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffpi.dcc import DCCKnobs
from diffpi.dcc_xsec import DCCCrossSection, M_N
from diffpi.spectral import load_spectral, SpectralSampler
from diffpi.hadron_xsec import HadronStructure

jax.config.update("jax_enable_x64", True)

E_NU = 1500.0
N = 400000
EP_LO, EP_HI, THETA_MAX = 150.0, 1450.0, 60.0

# ---- sample events (identical proposal to dcc_fold CC mode) ----------------- #
key = jax.random.PRNGKey(1)
klep, ksf, kth = jax.random.split(key, 3)
Ep = jax.lax.stop_gradient(EP_LO + (EP_HI - EP_LO) * jax.random.uniform(klep, (N,)))
theta = jax.lax.stop_gradient(jnp.deg2rad(THETA_MAX) * jax.random.uniform(kth, (N,)))
s2 = jnp.sin(theta / 2) ** 2
Q2 = 4.0 * E_NU * Ep * s2
omega = E_NU - Ep
q3 = jnp.sqrt(Q2 + omega ** 2)
qx = -Ep * jnp.sin(theta); qz = E_NU - Ep * jnp.cos(theta)
q4 = jnp.stack([omega, qx, jnp.zeros_like(qx), qz], axis=-1)
sampler = SpectralSampler(load_spectral("pke12p_tot.data"))
p_vec, E_rm = sampler.sample(ksf, N)
pi4 = jnp.concatenate([(M_N - E_rm)[:, None], p_vec], axis=1)
tot = q4 + pi4
W2 = tot[:, 0] ** 2 - jnp.sum(tot[:, 1:] ** 2, axis=1)
W = jnp.sqrt(jnp.clip(W2, 1.0, None))
Gamma = (Ep / E_NU) * jnp.sin(theta)              # flat W-propagator flux (as in fold)
# EM-like longitudinal polarisation (PLACEHOLDER eps for curve c)
eps = 1.0 / (1.0 + 2.0 * (q3 ** 2 / Q2) * (s2 / jnp.clip(1 - s2, 1e-9, None)))

# ---- three hadronic weights ------------------------------------------------- #
xs = DCCCrossSection()
sig_diag = jax.vmap(lambda w, q: xs.sigma(w, q, DCCKnobs(), "all"))(W, Q2)

hs = HadronStructure(n_theta=12, n_phi=12)
WT, WL = hs.structures_at(W, Q2, DCCKnobs())

w_a = np.asarray(Gamma * sig_diag)
w_b = np.asarray(Gamma * WT)
w_c = np.asarray(Gamma * (WT + eps * WL))
Wn = np.asarray(W)

# ---- histogram in the oracle W binning + normalise -------------------------- #
o = np.load("oracle/oracle_distributions_nu.npz")
We = o["W_edges"]; Wc = 0.5 * (We[:-1] + We[1:]); dW_or = o["dsig_W"]


def hist_norm(w):
    h, _ = np.histogram(Wn, bins=We, weights=w)
    h = h / np.diff(We)
    a = np.trapz(h, Wc)
    return h / a if a > 0 else h


def unit(v):
    a = np.trapz(v, Wc); return v / a if a > 0 else v


ha, hb, hc, ho = hist_norm(w_a), hist_norm(w_b), hist_norm(w_c), unit(dW_or)

pk = lambda h: Wc[np.argmax(h)]
chi2 = lambda h: float(np.sum((h - ho) ** 2) / np.sum(ho ** 2))   # relative L2 to oracle
print(f"oracle peak           : {pk(ho):.0f} MeV")
print(f"(a) diagonal   peak={pk(ha):.0f}  relL2={chi2(ha):.4f}")
print(f"(b) full W_T   peak={pk(hb):.0f}  relL2={chi2(hb):.4f}")
print(f"(c) full T+L   peak={pk(hc):.0f}  relL2={chi2(hc):.4f}  (eps placeholder)")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Wc, ho, "o-", ms=4, color="C1", label=f"oracle (peak {pk(ho):.0f})")
ax.plot(Wc, ha, "-", color="C7", label=f"(a) diagonal dcc_xsec (peak {pk(ha):.0f})")
ax.plot(Wc, hb, "-", lw=2, color="C0", label=f"(b) full W_T (peak {pk(hb):.0f})")
ax.plot(Wc, hc, "--", color="C2", label=f"(c) full W_T+eps*W_L (peak {pk(hc):.0f})")
ax.axvline(1232, ls=":", color="k", lw=0.8)
ax.set_xlabel("W [MeV]"); ax.set_ylabel("norm d$\\sigma$/dW")
ax.set_title("Full hadron-tensor port vs neutrino oracle (unit area)")
ax.legend()
fig.tight_layout(); fig.savefig("full_port_nu_dsigmadW.png", dpi=120)
print("\nsaved -> full_port_nu_dsigmadW.png")

"""Phase-2.5 physics test: does the full hadron-tensor port reproduce the weak
dsigma/dW of the neutrino oracle?

Reuses the SAME event sampling as the validated CC fold (proposal + spectral fold held
fixed), and compares hadronic weights of increasing fidelity:
  (a) diagonal  -- old dcc_xsec sigma(W,Q2) [transverse-only, no multipole interference]
  (b) full W_T  -- full-hadron-tensor transverse response (exact multipole norms,
                   per-channel isospin CGs, same-L interference restored)
  (c) full T+L  -- W_T + eps*W_L  (EM-eps placeholder for the longitudinal weight)
  (d) full L.W  -- the FAITHFUL contraction L_{mu,nu} W^{mu,nu} (real CC lepton tensor,
                   V-A interference) times the piN 2-body phase space k_pi(W)/W.

Curve (d) is the physically complete object: it reproduces the oracle dsigma/dW to
relL2 ~ 4e-4 (peak position exact). The decisive fix over (a)-(c) was restoring the
hadronic phase-space factor k_pi(W)/W (= fnuc*k_pi/(16 pi^3 W) in amp_dcc_sl.f), which
rises from threshold and had been dropped when the hadron tensor was built directly;
with the real lepton tensor carrying the flux, the hadronic factor is the BARE 2-body
phase space k_pi/W (NOT k_pi/(E_gamma W) -- that double-counts the EM flux).

NOTE (pending exactness, see achilles_const): the kinematic cuts here are still the
approximate ones; the exact ACHILLES cuts (W in [1076.957, 2000], Q2_adj in [0,5 GeV^2])
and the on-shell-rebalanced Q2_adj for the amplitude are the next refinement.

Run:  python compare_full_port_nu.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffpi.dcc import DCCKnobs
from diffpi.dcc_xsec import DCCCrossSection, M_N, pion_cm_momentum
from diffpi.spectral import load_spectral, SpectralSampler
from diffpi.hadron_xsec import HadronStructure
from diffpi.lepton_tensor import cm_lepton_momenta, lepton_tensor_cc, contract

jax.config.update("jax_enable_x64", True)

E_NU = 1500.0
N = 800000
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

# ---- (d) faithful contraction L_{mu,nu} W^{mu,nu} (real CC lepton tensor) ---- #
# lab momenta: nu along +z, lepton in x-z plane, struck nucleon from spectral fn.
k_lab = jnp.stack([jnp.full((N,), E_NU), jnp.zeros((N,)), jnp.zeros((N,)),
                   jnp.full((N,), E_NU)], axis=-1)
kp_lab = jnp.stack([Ep, Ep * jnp.sin(theta), jnp.zeros((N,)), Ep * jnp.cos(theta)], axis=-1)
p_st = pi4                                          # (M_N - E_rm, p_vec)
k_cm, kp_cm = cm_lepton_momenta(k_lab, kp_lab, p_st)
Lmn = lepton_tensor_cc(k_cm, kp_cm)
Wmn = hs.tensor_at(W, Q2, DCCKnobs())
LW = contract(Lmn, Wmn)
# physical single-pion phase space: W above piN threshold and positive energy transfer
# (sub-threshold Fermi configs give a spacelike q+p_struck -> unphysical boost; the
#  diagonal sigma killed them via phase_space=0, here we mask them explicitly).
Wthr = M_N + 138.0
phys = (W2 > Wthr ** 2) & (omega > 0)
# leptonic phase space/flux: d2sigma/dEp dOmega ~ (Ep/E) L.W ; dOmega = 2pi sin th dth
# hadronic piN 2-body phase space: k_pi(W)/W  (= fnuc*k_pi/(16 pi^3 W) in amp_dcc_sl.f,
# constants absorbed in the unit-area normalisation). This is THE fix that aligns the
# peak; without it the missing-with-W k_pi pulls strength to low W (peak ~10 MeV low).
kpi = pion_cm_momentum(W)
w_d = np.asarray(jnp.where(phys, (Ep / E_NU) * jnp.sin(theta) * LW * kpi / W, 0.0))
Wn = np.asarray(W)

# ---- fine W binning, oracle re-histogrammed from event-level data ----------- #
NBINS = 60
oe = np.load("oracle/oracle_nu_events.npz")
W_or, w_or = oe["W"], oe["w_nb"]
We = np.linspace(1080.0, 1650.0, NBINS + 1)
Wc = 0.5 * (We[:-1] + We[1:])
mor = np.isfinite(W_or)
dW_or, _ = np.histogram(W_or[mor], bins=We, weights=w_or[mor])
dW_or = dW_or / np.diff(We)


def hist_norm(w):
    h, _ = np.histogram(Wn, bins=We, weights=w)
    h = h / np.diff(We)
    a = np.trapz(h, Wc)
    return h / a if a > 0 else h


def unit(v):
    a = np.trapz(v, Wc); return v / a if a > 0 else v


ha, hb, hc, hd, ho = (hist_norm(w_a), hist_norm(w_b), hist_norm(w_c),
                      hist_norm(w_d), unit(dW_or))

pk = lambda h: Wc[np.argmax(h)]
chi2 = lambda h: float(np.sum((h - ho) ** 2) / np.sum(ho ** 2))   # relative L2 to oracle
print(f"oracle peak           : {pk(ho):.0f} MeV")
print(f"(a) diagonal   peak={pk(ha):.0f}  relL2={chi2(ha):.4f}")
print(f"(b) full W_T   peak={pk(hb):.0f}  relL2={chi2(hb):.4f}")
print(f"(c) full T+L   peak={pk(hc):.0f}  relL2={chi2(hc):.4f}  (eps placeholder)")
print(f"(d) full L.W   peak={pk(hd):.0f}  relL2={chi2(hd):.4f}  (real CC lepton tensor)")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Wc, ho, "o-", ms=4, color="C1", label=f"oracle (peak {pk(ho):.0f})")
ax.plot(Wc, ha, "-", color="C7", label=f"(a) diagonal dcc_xsec (peak {pk(ha):.0f})")
ax.plot(Wc, hb, "-", lw=1, color="C0", label=f"(b) full W_T (peak {pk(hb):.0f})")
ax.plot(Wc, hc, "--", color="C2", lw=1, label=f"(c) full W_T+eps*W_L (peak {pk(hc):.0f})")
ax.plot(Wc, hd, "-", lw=2.5, color="C3", label=f"(d) full L.W (peak {pk(hd):.0f})")
ax.axvline(1232, ls=":", color="k", lw=0.8)
ax.set_xlabel("W [MeV]"); ax.set_ylabel("norm d$\\sigma$/dW")
ax.set_title("Full hadron-tensor port vs neutrino oracle (unit area)")
ax.legend()
fig.tight_layout(); fig.savefig("full_port_nu_dsigmadW.png", dpi=120)
print("\nsaved -> full_port_nu_dsigmadW.png")

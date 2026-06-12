"""Loss evolution starting at the NOMINAL point, vs the REAL ADoNIS nominal prediction.

The tune (cc0pi_tune_t2k) uses the structural toy model -- whose nominal (sigma=1,1) is a poor delta_pT
SHAPE.  This script (a) computes the chi^2 of the REAL ADoNIS nominal prediction (the ACHILLES-
mirroring disaggregation, NO fitting) vs the T2K delta_pT data with the full covariance, and (b) runs
the toy optimizer from nominal and plots its loss vs iteration against that reference -- showing the
real nominal already agrees and the toy model is the limitation.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, uproot
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scripts.cc0pi_closure_tki import proposal, _hist, DPT_EDGES, SIG_NN0, SIG_ABS0, MB_FM2, RHO, DX
from scripts.cc0pi_tune_t2k import model, COVINV, DATA, _edges, BW, d_val, d_err, COV

NDF = 8 - 1     # 8 bins, allow an overall normalization (nominal still has the shape fixed)

# ---- REAL ADoNIS nominal delta_pT prediction (Cylinder disaggregation) in the data units --------- #
dis = np.load("data/oracle/cc0pi_disaggregated.npz")
ad_dpt = np.concatenate([dis[f"{c}_True_dpt"] for c in ("QE-C", "RES-C")])     # MeV
ad_w = np.concatenate([dis[f"{c}_True_w"] for c in ("QE-C", "RES-C")])         # nb (per 12C)
h_nb, _ = np.histogram(ad_dpt, bins=DPT_EDGES, weights=ad_w)
dpt_dsig_nb_per_MeV = h_nb / np.diff(DPT_EDGES)                                # nb/MeV per 12C
# nb/MeV per-12C -> 1e-38 cm^2/(GeV/c)/nucleon  (1nb=1e-33cm^2; /12 nucleons; *1000 MeV->GeV; *1e38)
CONV = 1e-33 / 12.0 * 1000.0 * 1e38
m_nom = jnp.asarray(dpt_dsig_nb_per_MeV * CONV)
# allow the overall normalization to float (the nominal shape is what is being judged)
A_nom = float((m_nom @ COVINV @ DATA) / (m_nom @ COVINV @ m_nom))
r = A_nom * m_nom - DATA
chi2_nom = float(r @ COVINV @ r)
print(f"ADoNIS NOMINAL delta_pT: chi2/ndf = {chi2_nom/NDF:.2f}  (norm A_nom={A_nom:.3f}, integral match)")

# ---- toy optimizer starting at nominal (sigma_abs=sigma_scatter=1); record loss vs iteration ---- #
N = 400_000
S0 = proposal(jax.random.PRNGKey(0), N)
A0 = float(jnp.sum(DATA * jnp.asarray(BW)) / jnp.sum(model(jnp.array([1.0, 1.0, 0.0]), S0) * jnp.asarray(BW)))
theta = jnp.array([1.0, 1.0, np.log(A0)])               # NOMINAL start


def loss(theta, key):
    k1, k2 = jax.random.split(key)
    r1 = model(theta, proposal(k1, N)) - DATA; r2 = model(theta, proposal(k2, N)) - DATA
    return r1 @ COVINV @ r2


vg = jax.jit(jax.value_and_grad(loss)); m = jnp.zeros(3); v = jnp.zeros(3); lr = 0.02
hist = []
for it in range(500):
    l, g = vg(theta, jax.random.PRNGKey(10 + it))
    hist.append(float(l) / NDF)
    m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
    mh = m / (1 - 0.9 ** (it + 1)); vh = v / (1 - 0.999 ** (it + 1))
    theta = theta - lr * mh / (jnp.sqrt(vh) + 1e-8)
fit = np.asarray(theta)
print(f"toy: nominal chi2/ndf={hist[0]:.2f} -> best {hist[-1]:.2f}  (s_abs={fit[0]:.3f} s_sc={fit[1]:.3f})")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
ax[0].plot(hist, color="C0", lw=1.8, label="toy model fit (from nominal)")
ax[0].axhline(chi2_nom / NDF, color="C2", ls="--", lw=2, label=f"REAL ADoNIS nominal = {chi2_nom/NDF:.2f}")
ax[0].axhline(1.0, color="0.6", ls=":", lw=1, label="$\\chi^2$/ndf = 1")
ax[0].set(xlabel="iteration", ylabel="$\\chi^2$/ndf", title="loss vs iteration (start = nominal $\\sigma$=1,1)")
ax[0].set_yscale("log"); ax[0].legend(fontsize=9)
ctr = 0.5 * (_edges[1:] + _edges[:-1]); mb = np.asarray(model(jnp.asarray(fit), S0))
ax[1].errorbar(ctr, d_val, yerr=d_err, fmt="o", color="k", capsize=3, label="T2K data")
ax[1].step(_edges, np.append(np.asarray(A_nom * m_nom), float(A_nom * m_nom[-1])), where="post",
           color="C2", lw=2, label=f"ADoNIS nominal ($\\chi^2$/ndf {chi2_nom/NDF:.1f})")
ax[1].step(_edges, np.append(mb, mb[-1]), where="post", color="C0", lw=1.6, ls="--",
           label=f"toy best fit ($\\chi^2$/ndf {hist[-1]:.1f})")
ax[1].set(xlabel=r"$\delta p_T$ [GeV/c]", ylabel=r"d$\sigma$/d$\delta p_T$ [$10^{-38}$cm$^2$/(GeV/c)/nuc]",
          title="$\\delta p_T$: data vs nominal vs toy fit"); ax[1].legend(fontsize=8); ax[1].set_ylim(bottom=0)
fig.suptitle("ADoNIS NOMINAL (ACHILLES-mirroring) vs the toy tune on T2K $\\delta p_T$")
fig.tight_layout(); fig.savefig("paper_figures/cc0pi_tune_nominal.png", dpi=120); print("wrote paper_figures/cc0pi_tune_nominal.png")

"""Stochastic neutrino closure: recover (pw_norm[p33], axial mass M_A) from NOISY
Monte-Carlo CC-fold pseudo-data -- the realistic-conditions analogue of the
deterministic run_dcc_closure.py.

Unlike the deterministic grid closure (exact, noise-free), here the observable is
a finite-sample CC fold (diffpi/dcc_fold.fold_hist2d), so the gradient is a
reweighting estimator through a STOCHASTIC histogram.  This stress-tests the
Phase-1 machinery on the real DCC physics: the unbiased TWO-REPLICA loss
mean((m1-data)*(m2-data)) (independent keys cancel model-variance bias) + gradient
clipping + LR annealing.

Identifiability: pw_norm[p33] scales the Delta -> localises in W; M_A shapes the
axial Q^2 falloff -> the joint (W,Q^2) histogram separates them.

Run:  python run_dcc_nu_closure.py
"""
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffpi.dcc import DCCKnobs
from diffpi.dcc_xsec import DCCCrossSection
from diffpi.dcc_fold import fold_hist2d
from diffpi.harness import run_closure

X = DCCCrossSection()
i33 = X.D.labels.index("p33")
n_pw = len(X.D.labels)

E_NU = 1500.0
W_edges = jnp.linspace(1100.0, 1420.0, 17)
Q2_edges = jnp.linspace(0.0, 1.1e6, 16)          # 0 - 1.1 GeV^2
FOLD_KW = dict(mode="CC", e_beam=E_NU, ep_lo=150.0, ep_hi=1450.0, theta_max_deg=60.0)

N_DATA = 1_500_000
N_MODEL = 150_000


def knobs(d_p33, MA):
    pw = tuple(d_p33 if i == i33 else 0.0 for i in range(n_pw))
    return DCCKnobs(pw_norm=pw, axial_MA=MA)


# ---- pseudo-data at injected truth (fixed, high-stats) ---------------------- #
# NB: fold_hist2d returns sum-of-weights, which scales with the event count n, so
# the histograms MUST be divided by n to be sample-size-independent cross-section
# estimates -- otherwise the optimizer mis-tunes the xsec knobs to match the raw
# normalization (data 1.5M events vs model 150k) instead of the physics.
D_TRUE, MA_TRUE = 0.15, 1.25
data = fold_hist2d(knobs(D_TRUE, MA_TRUE), W_edges, Q2_edges,
                   jax.random.PRNGKey(20240603), n=N_DATA, **FOLD_KW) / N_DATA
inv = 1.0 / (data + 0.02 * jnp.max(data))         # chi^2-style weighting


def to_params(u):
    return jnp.array([u[0], 1.0 * jnp.exp(u[1])])   # d_p33 free; M_A = exp(u) > 0


def loss_fn(u, key):
    p = to_params(u)
    k1, k2 = jax.random.split(key)
    m1 = fold_hist2d(knobs(p[0], p[1]), W_edges, Q2_edges, k1, n=N_MODEL, **FOLD_KW) / N_MODEL
    m2 = fold_hist2d(knobs(p[0], p[1]), W_edges, Q2_edges, k2, n=N_MODEL, **FOLD_KW) / N_MODEL
    # unbiased two-replica chi^2 (independent keys cancel the model-variance bias)
    return jnp.mean((m1 - data) * (m2 - data) * inv)


theta_true = np.array([D_TRUE, MA_TRUE])
init = jnp.array([0.0, 0.0])
res = run_closure(loss_fn, to_params, init, jax.random.PRNGKey(1),
                  iterations=400, learning_rate=0.02, theta_true=theta_true,
                  clip_norm=10.0, final_lr_frac=0.03)

dp_fit, ma_fit = float(res.theta_fit[0]), float(res.theta_fit[1])
print("\n=== recovered vs truth (stochastic CC fold) ===")
print(f"  pw_norm[p33]:  {dp_fit:.4f}   (truth {D_TRUE})")
print(f"  M_A [GeV]   :  {ma_fit:.4f}   (truth {MA_TRUE})")
# tolerance is looser than the deterministic closure (MC noise floor)
ok = abs(dp_fit - D_TRUE) < 0.03 and abs(ma_fit - MA_TRUE) < 0.05
print(f"\nStochastic neutrino M_A closure: {'PASS' if ok else 'FAIL'}")

its = np.arange(len(res.loss_history))
ph = np.asarray(res.param_history)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(its, res.loss_history, color="C0", lw=0.8)
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("two-replica chi$^2$ loss")
ax[0].set_title("stochastic closure loss (MC fold)")
ax[1].plot(its, ph[:, 0], color="C0", lw=0.8, label="pw_norm[p33]")
ax[1].axhline(D_TRUE, ls="--", color="C0", alpha=0.5)
ax[1].plot(its, ph[:, 1], color="C3", lw=0.8, label="M_A [GeV]")
ax[1].axhline(MA_TRUE, ls="--", color="C3", alpha=0.5)
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("parameter")
ax[1].set_title(f"recovery: p33={dp_fit:.3f}/{D_TRUE}, M_A={ma_fit:.3f}/{MA_TRUE}")
ax[1].legend()
fig.tight_layout()
fig.savefig("dcc_nu_closure.png", dpi=110)
print("saved -> dcc_nu_closure.png")
raise SystemExit(0 if ok else 1)

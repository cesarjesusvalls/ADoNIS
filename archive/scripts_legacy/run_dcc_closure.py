"""Phase-2 capstone closure: recover REAL DCC physics parameters by gradient descent.

Inject truth knobs (pw_norm[p33], axial mass M_A), build the 2-D (W, Q^2) weak
single-pion cross section as pseudo-data, then fit the knobs back from a nominal
start with Adam on the differentiable assembly (diffpi/dcc_xsec.py).  This is the
end-to-end demonstration of the whole project on the actual DCC amplitudes:
real physics parameters -> differentiable observable -> recovered by gradients.

The forward model is DETERMINISTIC (grid + quadrature, no MC) so gradients are
exact and the closure is clean (no two-replica / annealing needed).

Identifiability: pw_norm[p33] scales the Delta (P33) -> localises to the W peak;
M_A shapes the axial Q^2 falloff -> the (W,Q^2) shape separates them.

Run:  python run_dcc_closure.py
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
from diffpi.harness import run_closure

X = DCCCrossSection()
i33 = X.D.labels.index("p33")
n_pw = len(X.D.labels)

# observable grid
W_grid = jnp.linspace(1100.0, 1400.0, 40)
Q2_grid = jnp.linspace(20000.0, 500000.0, 22)   # 0.02 - 0.5 GeV^2
WW, QQ = jnp.meshgrid(W_grid, Q2_grid, indexing="ij")
Wf, Qf = WW.reshape(-1), QQ.reshape(-1)


def model_grid(d_p33, MA):
    """2-D weak (vec+axial) sigma(W,Q2) flattened; knob-dependent, differentiable."""
    pw = tuple(d_p33 if i == i33 else 0.0 for i in range(n_pw))
    k = DCCKnobs(pw_norm=pw, axial_MA=MA)
    return jax.vmap(lambda w, q: X.sigma(w, q, k, "all"))(Wf, Qf)


# truth + pseudo-data
D_TRUE, MA_TRUE = 0.20, 1.20
data = model_grid(D_TRUE, MA_TRUE)
scale = float(jnp.mean(data))


def to_params(u):
    # physical params [d_p33, M_A]; d_p33 free, M_A = exp(u) > 0 (=1 at u=0)
    return jnp.array([u[0], 1.0 * jnp.exp(u[1])])


def loss_fn(u, key):
    p = to_params(u)
    m = model_grid(p[0], p[1])
    return jnp.mean(((m - data) / scale) ** 2)


theta_true = np.array([D_TRUE, MA_TRUE])
init = jnp.array([0.0, 0.0])      # nominal: d_p33=0, M_A=1.0
res = run_closure(loss_fn, to_params, init, jax.random.PRNGKey(0),
                  iterations=400, learning_rate=0.05, theta_true=theta_true,
                  clip_norm=None, final_lr_frac=0.05)

dp_fit, ma_fit = float(res.theta_fit[0]), float(res.theta_fit[1])
print("\n=== recovered vs truth ===")
print(f"  pw_norm[p33]:  {dp_fit:.4f}   (truth {D_TRUE})")
print(f"  M_A [GeV]   :  {ma_fit:.4f}   (truth {MA_TRUE})")
ok = abs(dp_fit - D_TRUE) < 0.01 and abs(ma_fit - MA_TRUE) < 0.01
print(f"\nDCC physics closure: {'PASS' if ok else 'FAIL'}")

# evolution figure
its = np.arange(len(res.loss_history))
ph = np.asarray(res.param_history)     # (iters, 2)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].semilogy(its, res.loss_history, color="C0")
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("loss (rel MSE)"); ax[0].set_title("closure loss")
ax[1].plot(its, ph[:, 0], color="C0", label="pw_norm[p33]")
ax[1].axhline(D_TRUE, ls="--", color="C0", alpha=0.5)
ax[1].plot(its, ph[:, 1], color="C3", label="M_A [GeV]")
ax[1].axhline(MA_TRUE, ls="--", color="C3", alpha=0.5)
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("parameter")
ax[1].set_title("parameter recovery (dashed = truth)"); ax[1].legend()
fig.tight_layout()
fig.savefig("dcc_closure.png", dpi=110)
print("saved -> dcc_closure.png")
raise SystemExit(0 if ok else 1)

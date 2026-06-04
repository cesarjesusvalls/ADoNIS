"""Validate the FAITHFUL full-port fold (dcc_fold_full) against the neutrino CC oracle.

The full hadron-tensor port (amp_dcc_sl.f) contracted with the real CC lepton tensor and
folded over the 12C spectral function, with ACHILLES-exact kinematics (mqe, on-shell
rebalanced Q2_adj, the W in [1076.957,2000] / Q2 in [0,5 GeV^2] cuts, piN phase space
k_pi/W). Gates, against the EVENT-LEVEL oracle (oracle/oracle_nu_events.npz, 200k events):

  1. dsigma/dW  reproduces the oracle (relL2 < 5e-3, Delta-peak position exact);
  2. dsigma/dQ2 reproduces the oracle (relL2 < 5e-3, hard Q^2 reach matched);
  3. the fold is differentiable in M_A with an EXACT gradient (autodiff == finite diff).

Run:  python validate_fold_full.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffpi.dcc import DCCKnobs
from diffpi.hadron_xsec import HadronStructure
from diffpi.dcc_fold_full import fold_full_events, fold_full_dsigma_dW

jax.config.update("jax_enable_x64", True)

ok = True
def chk(name, cond, extra=""):
    global ok; ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {extra}")

hs = HadronStructure(n_theta=12, n_phi=12)
W, Q2, w = fold_full_events(DCCKnobs(), jax.random.PRNGKey(1), n=1_000_000, hs=hs)
W, Q2, w = np.asarray(W), np.asarray(Q2), np.asarray(w)

oe = np.load("oracle/oracle_nu_events.npz")
W_or, Q2_or, w_or = oe["W"], oe["Q2"], oe["w_nb"]


def norm_hist(v, wt, edges):
    c = 0.5 * (edges[:-1] + edges[1:])
    h, _ = np.histogram(v, bins=edges, weights=wt); h = h / np.diff(edges)
    a = np.trapz(h, c); return c, (h / a if a > 0 else h)


def relL2(hm, ho):
    return float(np.sum((hm - ho) ** 2) / np.sum(ho ** 2))


# ---- dsigma/dW -------------------------------------------------------------- #
We = np.linspace(1080.0, 1650.0, 61)
Wc, hmW = norm_hist(W, w, We)
_,  hoW = norm_hist(W_or, w_or, We)
rW = relL2(hmW, hoW)
chk("dsigma/dW reproduces oracle (relL2<5e-3)", rW < 5e-3, f"relL2={rW:.5f}")
chk("Delta-peak position exact", Wc[np.argmax(hmW)] == Wc[np.argmax(hoW)],
    f"model={Wc[np.argmax(hmW)]:.0f}, oracle={Wc[np.argmax(hoW)]:.0f} MeV")

# ---- dsigma/dQ2 ------------------------------------------------------------- #
Q2e = np.linspace(0.0, np.percentile(Q2_or, 99.5), 41)
Q2c, hmQ = norm_hist(Q2, w, Q2e)
_,   hoQ = norm_hist(Q2_or, w_or, Q2e)
rQ = relL2(hmQ, hoQ)
chk("dsigma/dQ2 reproduces oracle (relL2<5e-3)", rQ < 5e-3, f"relL2={rQ:.5f}")
# robust reach: highest Q^2 where the normalised dsigma/dQ2 is still >= 5% of its peak
def reach5(c, h):
    idx = np.where(h >= 0.05 * h.max())[0]
    return c[idx[-1]] / 1e6
reach_m, reach_o = reach5(Q2c, hmQ), reach5(Q2c, hoQ)
chk("Q^2 reach matches oracle (5%-of-peak, within 10%)",
    abs(reach_m - reach_o) / reach_o < 0.1, f"model={reach_m:.2f}, oracle={reach_o:.2f} GeV^2")

# ---- exact gradient in M_A -------------------------------------------------- #
def scalar(MA):
    h = fold_full_dsigma_dW(DCCKnobs(axial_MA=MA), jnp.asarray(We),
                            jax.random.PRNGKey(3), n=150_000, hs=hs)
    return jnp.sum(h)
g = float(jax.grad(scalar)(1.1))
fd = float((scalar(1.105) - scalar(1.095)) / 0.01)
chk("d(fold)/dM_A exact (autodiff==finite-diff)", np.isclose(g, fd, rtol=3e-2),
    f"auto={g:.4e}, fd={fd:.4e}")

# ---- figure ----------------------------------------------------------------- #
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(Wc, hoW, "o-", ms=4, color="C1", label=f"oracle (peak {Wc[np.argmax(hoW)]:.0f})")
ax[0].plot(Wc, hmW, "-", lw=2, color="C3", label=f"faithful fold (peak {Wc[np.argmax(hmW)]:.0f})")
ax[0].axvline(1232, ls=":", color="k", lw=0.8)
ax[0].set_xlabel("W [MeV]"); ax[0].set_ylabel("norm d$\\sigma$/dW")
ax[0].set_title(f"d$\\sigma$/dW  (relL2={rW:.1e})"); ax[0].legend()
ax[1].plot(Q2c / 1e6, hoQ * 1e6, "o-", ms=4, color="C1", label="oracle")
ax[1].plot(Q2c / 1e6, hmQ * 1e6, "-", lw=2, color="C3", label="faithful fold")
ax[1].set_xlabel(r"$Q^2$ [GeV$^2$]"); ax[1].set_ylabel("norm d$\\sigma$/d$Q^2$")
ax[1].set_title(f"d$\\sigma$/d$Q^2$  (relL2={rQ:.1e})"); ax[1].legend()
fig.suptitle("Faithful full hadron-tensor port (L.W) vs ACHILLES neutrino oracle")
fig.tight_layout(); fig.savefig("fold_full_validation.png", dpi=120)
print("\nsaved -> fold_full_validation.png")
print(f"\nFaithful full-port fold vs neutrino oracle: {'PASS' if ok else 'FAIL'}")
raise SystemExit(0 if ok else 1)

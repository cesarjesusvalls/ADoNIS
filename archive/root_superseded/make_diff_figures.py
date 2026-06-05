"""Demonstrate that the EXCLUSIVE (full-final-state) predictions are differentiable in the
physics knob M_A, with an EXACT gradient.

Uses the kind-1 split: sample the fixed detached proposal ONCE (sample_final_state), then
re-weight at any M_A (weight_from_sample) -- so the per-bin gradient of every exclusive
distribution is one cheap reverse/forward-mode pass on the SAME sample.  For each
observable x:
  * top:    dsigma/dx at M_A = 1.00 and 1.20  -> the knob MOVES the exclusive prediction;
  * bottom: d(dsigma/dx)/dM_A from autodiff (line) vs central finite difference (points)
            -> coincide => differentiable + exact.

Run:  python make_diff_figures.py
"""
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffpi.dcc import DCCKnobs
from diffpi.hadron_xsec import HadronStructure
from diffpi.fold_final_state import sample_final_state, weight_from_sample
from diffpi.event_record import EventRecord
from diffpi import observables as obs

os.makedirs("figures", exist_ok=True)
# N kept modest: the autodiff-vs-FD match is per-bin EXACT on the shared sample regardless
# of statistics, and the spline amplitude interp under jacfwd is memory-heavy at large N.
hs = HadronStructure(n_theta=16, n_phi=16, spline=False)
N = 50_000
MA0, EPS, MA_HI = 1.0, 5e-3, 1.2

print("sampling fixed proposal once ...", flush=True)
S = sample_final_state(jax.random.PRNGKey(20), n=N, hs=hs)
z = jnp.zeros(N)
ev0 = EventRecord(k=S["k_lab"], kp=S["kp_lab"], p_struck=S["p_struck"], p_pi=S["p_pi"],
                  p_N=S["p_N"], w=z, channel=z, pid_pi=z, pid_N=z, pid_Ni=z,
                  W=S["W"], Q2_adj=S["Q2_adj"])

PANELS = [("W", "W [MeV]", np.linspace(1080, 1640, 33), 1.0),
          ("Q2", "$Q^2$ [GeV$^2$]", np.linspace(0, 1.6e6, 33), 1e-6),
          ("cos_theta_star", r"$\cos\theta^*_\pi$", np.linspace(-1, 1, 25), 1.0),
          ("ppi_mag", r"$|p_\pi|$ [MeV]", np.linspace(0, 700, 29), 1.0),
          ("lepton_energy", r"$E_\ell$ [MeV]", np.linspace(0, 1400, 29), 1.0),
          ("lepton_costheta", r"$\cos\theta_\ell$ (lab)", np.linspace(0.2, 1, 29), 1.0)]


def make_dsig(name, edges):
    vals = obs.OBSERVABLES[name](ev0)
    nb = edges.shape[0] - 1
    idx = jax.lax.stop_gradient(jnp.clip(jnp.searchsorted(jnp.asarray(edges), vals) - 1, 0, nb - 1))
    dx = jnp.diff(jnp.asarray(edges))
    def dsig(MA):
        w, _ = weight_from_sample(DCCKnobs(axial_MA=MA), S)
        return jax.ops.segment_sum(w, idx, num_segments=nb) / dx
    return dsig


fig = plt.figure(figsize=(15, 8))
outer = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.28)
slots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

for (name, xlabel, edges, sc), (r, col) in zip(PANELS, slots):
    x = 0.5 * (edges[:-1] + edges[1:]) * sc
    dsig = make_dsig(name, edges)
    h0 = np.asarray(dsig(MA0)); hi = np.asarray(dsig(MA_HI))
    g_ad = np.asarray(jax.jacfwd(dsig)(MA0))
    g_fd = (np.asarray(dsig(MA0 + EPS)) - np.asarray(dsig(MA0 - EPS))) / (2 * EPS)
    S_disp = h0.sum() * (edges[1] - edges[0])
    big = np.abs(g_ad) > 0.02 * np.abs(g_ad).max()
    relmax = float(np.max(np.abs(g_ad - g_fd)[big] / (np.abs(g_ad)[big] + 1e-30)))

    inner = outer[r, col].subgridspec(2, 1, height_ratios=[3, 2], hspace=0.05)
    ax = fig.add_subplot(inner[0]); axg = fig.add_subplot(inner[1], sharex=ax)
    ax.plot(x, h0 / S_disp, "-", color="cornflowerblue", lw=1.5, label="$M_A=1.00$")
    ax.plot(x, hi / S_disp, "--", color="darkorange", lw=1.5, label="$M_A=1.20$")
    ax.set_ylabel("norm. d$\\sigma$/dx"); ax.set_title(name); ax.legend(fontsize=8)
    ax.tick_params(labelbottom=False)
    axg.axhline(0, color="0.7", lw=0.7)
    axg.plot(x, g_ad / S_disp, "-", color="cornflowerblue", lw=2.0, label="autodiff $\\partial_{M_A}$")
    axg.plot(x, g_fd / S_disp, "o", color="darkorange", ms=3.5, label="finite diff")
    axg.set_xlabel(xlabel); axg.set_ylabel(r"$\partial_{M_A}$ d$\sigma$/dx")
    axg.legend(fontsize=7, loc="best")
    axg.text(0.02, 0.06, f"max rel(AD,FD) = {relmax:.1e}", transform=axg.transAxes, fontsize=7, va="bottom")
    print(f"{name:16s} autodiff vs FD  max rel err = {relmax:.2e}", flush=True)

fig.savefig("figures/exclusive_differentiability.png", dpi=120, bbox_inches="tight")
print("saved -> figures/exclusive_differentiability.png")

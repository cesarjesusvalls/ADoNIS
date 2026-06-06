"""Phase D1 figure: the intranuclear cascade's distortion of the produced pion spectrum.

Generates CC single-pion events with the real DCC production vertex, then applies the
concrete cascade FSIModel (`ToyCascadeFSI`).  Plots the pion-momentum spectrum before vs
after FSI (energy-loss softening + absorption depletion) and annotates the CC0pi
(absorbed) fraction -- the qualitative signature FSI imprints on the final state.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adonis import GenConfig, DCCSinglePion, PhysicsParams
from adonis.primary.dcc.channel import assemble_event
from adonis.fsi.cascade import ToyCascadeFSI, CascadeConfig

N = int(os.environ.get("ADONIS_FSI_N", 60_000))
ch = DCCSinglePion(GenConfig(spline=False))
fsi = ToyCascadeFSI(CascadeConfig(seed=5))
params = PhysicsParams(fsi_sigma_scatter=0.35, fsi_sigma_abs=0.22)

S = ch.sample(jax.random.PRNGKey(1), N)
w, LWc = ch.weight(params, S)
pre = assemble_event(S, w, LWc)
post = fsi.apply(params, pre, key=jax.random.PRNGKey(2))


def ppi(ev):
    return np.asarray(jnp.linalg.norm(ev.p_pi[:, 1:], axis=1))


edges = np.linspace(0, 500, 41)
wpre = np.asarray(pre.w)
wpost = np.asarray(post.w)
absorbed = np.asarray(post.pid_pi == 0)
cc0pi_frac = float(np.sum(wpost[absorbed]) / np.sum(wpre))

h_pre, _ = np.histogram(ppi(pre), bins=edges, weights=wpre)
h_post, _ = np.histogram(ppi(post)[~absorbed], bins=edges, weights=wpost[~absorbed])
ctr = 0.5 * (edges[1:] + edges[:-1])

fig, ax = plt.subplots(figsize=(7, 5))
ax.step(ctr, h_pre, where="mid", lw=2, label="produced (pre-FSI)")
ax.step(ctr, h_post, where="mid", lw=2, label="escaped (post-FSI)")
ax.set_xlabel(r"$|p_\pi|$  [MeV]")
ax.set_ylabel("weighted events / bin")
ax.set_title("Cascade FSI distortion of the pion spectrum  (¹²C, DCC CC1π)")
ax.legend(title=f"absorbed (CC0π): {cc0pi_frac:.0%}\n"
                rf"$\sigma_{{sc}}$={float(params.fsi_sigma_scatter):.2f}, "
                rf"$\sigma_{{abs}}$={float(params.fsi_sigma_abs):.2f} fm$^{{-1}}$")
ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures", "fsi_cascade_c12.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=130)
print(f"wrote {out}")
print(f"pre  sum={h_pre.sum():.3e}   post(escaped) sum={h_post.sum():.3e}   "
      f"CC0pi absorbed fraction={cc0pi_frac:.3f}")

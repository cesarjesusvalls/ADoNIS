"""Phase C figure: the FSI cascade's distortion of a single-transverse kinematic imbalance.

delta_pT (the transverse momentum imbalance of the visible final state) is the canonical
FSI-sensitive observable in modern neutrino experiments: with no FSI it reflects only the
initial-nucleon Fermi motion; the intranuclear cascade (scatter + absorption) drags strength
into a high-delta_pT tail and converts CC1pi -> CC0pi by absorption.  This plots the CC1pi
delta_pT before vs after the cascade FSIModel, and reports the absorption-driven CC0pi rate.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from adonis import ChainConfig, DCCSinglePion, PhysicsParams, observables as obs
from adonis.primary.dcc.channel import assemble_event
from adonis.fsi.cascade import ToyCascadeFSI, CascadeConfig
from adonis.signal import CC1Pi, CC0Pi
ROOT = Path(__file__).resolve().parents[1]

N = int(os.environ.get("ADONIS_TKI_N", 60_000))
ch = DCCSinglePion(ChainConfig(spline=False))
fsi = ToyCascadeFSI(CascadeConfig(seed=4))
params = PhysicsParams(fsi_sigma_scatter=0.35, fsi_sigma_abs=0.22)

S = ch.sample(jax.random.PRNGKey(1), N)
w, LWc = ch.weight(params, S)
pre = assemble_event(S, w, LWc)
post = fsi.apply(params, pre, key=jax.random.PRNGKey(2))

dpt_pre = np.asarray(obs.delta_pT(pre))
dpt_post = np.asarray(obs.delta_pT(post))
wpre = np.asarray(pre.w)
sel1 = np.asarray(CC1Pi().select(post))           # surviving-pion CC1pi
sel0 = np.asarray(CC0Pi().select(post))           # absorbed -> CC0pi
cc0pi_frac = float(np.sum(post.w[sel0]) / np.sum(wpre))

edges = np.linspace(0, 700, 36)
h_pre, _ = np.histogram(dpt_pre, bins=edges, weights=wpre)
h_post, _ = np.histogram(dpt_post[sel1], bins=edges, weights=np.asarray(post.w)[sel1])
ctr = 0.5 * (edges[1:] + edges[:-1])

# tail fraction (delta_pT > 300 MeV): the FSI signature
def tail(h):
    return h[ctr > 300].sum() / h.sum()

fig, ax = plt.subplots(figsize=(7, 5))
ax.step(ctr, h_pre / h_pre.sum(), where="mid", lw=2,
        label=f"produced CC1π (pre-FSI)  tail={tail(h_pre):.0%}")
ax.step(ctr, h_post / h_post.sum(), where="mid", lw=2,
        label=f"surviving CC1π (post-FSI)  tail={tail(h_post):.0%}")
ax.set_xlabel(r"$\delta p_T$  [MeV]")
ax.set_ylabel("normalised CC1π events / bin")
ax.set_title("FSI distortion of the TKI imbalance $\\delta p_T$  (¹²C, DCC CC1π)")
ax.legend(title=f"absorbed → CC0π: {cc0pi_frac:.0%}\n"
                rf"$\sigma_{{sc}}$={float(params.fsi_sigma_scatter):.2f},"
                rf" $\sigma_{{abs}}$={float(params.fsi_sigma_abs):.2f} fm$^{{-1}}$")
ax.grid(alpha=0.3)
fig.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "tki_fsi_dpt_c12.png", dpi=130)
print(f"wrote {out / 'tki_fsi_dpt_c12.png'}")
print(f"delta_pT tail (>300 MeV): pre {tail(h_pre):.3f} -> post {tail(h_post):.3f};  "
      f"CC0pi(absorbed) frac {cc0pi_frac:.3f}")

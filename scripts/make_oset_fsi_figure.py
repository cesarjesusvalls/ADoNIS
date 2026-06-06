"""F->D figure: the Oset-shaped, momentum-dependent FSI absorption in the cascade.

Runs the cascade FSIModel in `oset_shape` mode on mono-energetic pion batches and plots the
absorption fraction vs pion kinetic energy T_pi, overlaid with the Oset absorption shape
that drives it.  Absorption peaks in the Delta region (T_pi ~ 180 MeV) with the low-T_pi
s-wave tail -- the physical pion-absorption profile, now feeding the differentiable cascade.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from adonis.core.event import EventRecord
from adonis.params import PhysicsParams
from adonis.fsi.cascade import ToyCascadeFSI, CascadeConfig
from adonis.fsi.mb.oset import absorption_rate_shape
ROOT = Path(__file__).resolve().parents[1]
M_PI = 139.57

m = ToyCascadeFSI(CascadeConfig(oset_shape=True, seed=2))
params = PhysicsParams(fsi_sigma_scatter=0.05, fsi_sigma_abs=0.30)   # low scatter -> isolate absorption


def mono(p_mev, n=8000):
    key = jax.random.PRNGKey(int(p_mev)); k1, k2 = jax.random.split(key)
    ct = jax.random.uniform(k1, (n,), minval=-1, maxval=1); ph = jax.random.uniform(k2, (n,)) * 2 * np.pi
    st = jnp.sqrt(1 - ct ** 2)
    d = jnp.stack([st * jnp.cos(ph), st * jnp.sin(ph), ct], 1) * p_mev
    E = jnp.sqrt(p_mev ** 2 + M_PI ** 2); z = jnp.zeros((n, 4)); o = jnp.ones(n)
    p_pi = jnp.concatenate([jnp.full((n, 1), E), d], 1)
    return EventRecord(k=z, kp=z, p_struck=z, p_pi=p_pi, p_N=z, w=o,
                       channel=jnp.zeros(n, jnp.int32), pid_pi=jnp.full(n, 211, jnp.int32),
                       pid_N=jnp.full(n, 2212, jnp.int32), pid_Ni=jnp.full(n, 2112, jnp.int32),
                       W=jnp.full(n, 1232.0), Q2_adj=jnp.full(n, 1.0e5))


pmom = np.arange(70, 430, 25.0)
Tpi = np.sqrt(pmom ** 2 + M_PI ** 2) - M_PI
af = np.array([float(np.mean(np.asarray(m.apply(params, mono(int(pp)), key=jax.random.PRNGKey(7)).pid_pi == 0)))
               for pp in pmom])
Tgrid = np.linspace(5, Tpi.max(), 200)
shape = np.asarray(absorption_rate_shape(jnp.asarray(Tgrid)))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Tpi, af, "o-", color="tab:red", label="cascade absorption fraction")
ax2 = ax.twinx()
ax2.plot(Tgrid, shape, color="tab:blue", lw=2, alpha=0.7, label="Oset abs. shape (norm@180)")
ax.axvline(180, ls=":", color="gray"); ax.set_xlabel(r"$T_\pi$  [MeV]")
ax.set_ylabel("absorbed fraction", color="tab:red")
ax2.set_ylabel("Oset absorption shape", color="tab:blue")
ax.set_title("Oset-shaped momentum-dependent FSI absorption (peaks at the Δ)")
l1, lab1 = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lab1 + lab2, loc="lower right")
fig.tight_layout()
out = ROOT / "figures"; out.mkdir(exist_ok=True)
fig.savefig(out / "oset_fsi_absorption_c12.png", dpi=130)
print("wrote", out / "oset_fsi_absorption_c12.png")
print("absorbed fraction peaks at T_pi =", Tpi[int(np.argmax(af))], "MeV")

"""Test the 'near-surface degrading scatter -> capture' hypothesis: iterate the ADoNIS single-pass
nucleon step on carbon QE protons, record per proton the RADIUS of its last elastic scatter and its
fate, then histogram last-scatter-radius for CAPTURED vs ESCAPED protons (overlaid with the local
Fermi momentum k_F(r) and the proton density)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ADONIS_N_RECOIL"] = "6"
import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.workflow.materials import resolve_targets
from adonis.workflow.generate import gen_events
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _nucleon_step, _load_density, _kf_local, _rho_species
from adonis.fsi.cascade_full import setup_carbon
from adonis.xsec.spectral import SpectralFunction
from adonis.fsi import oset_xsec as ox
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
M_N = float(ox.M_N); MS = 600

tg = resolve_targets("C")[0][0]
cfg = DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=MS, seed=1, nn_inelastic=True,
                            pauli=True, nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
sf_n = SpectralFunction(tg.spectral_n)
a = gen_events("qe", 40000, 0, sf_n=sf_n, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); s = w > 0; m = int(s.sum()); w = w[s]
pN = jnp.asarray(a["p_N"][s]); ipid = jnp.asarray(a["ipid"][s])
su = setup_carbon(pN, jnp.zeros(m, jnp.int32), ipid.astype(jnp.int32), cfg, jax.random.PRNGKey(11))
rgrid, rhoP, rhoN, radius = _load_density(tg.density_p, tg.density_n)
keys = jax.random.split(jax.random.PRNGKey(99), MS)
p4 = pN; pos = su["pos0"]; d3 = p4[:, 1:]
dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
fz = jnp.zeros(m); alive = jnp.ones(m, bool); consumed = su["consumed0"]
is_p = jnp.ones(m, bool)
last_scat_r = jnp.full(m, -1.0); nsc = jnp.zeros(m, jnp.int32)
for i in range(MS):
    (p4, pos, dhat, fz, alive), esc, recap, do, ko, pin, consumed, _ = _nucleon_step(
        p4, pos, dhat, fz, is_p, alive, su["npos"], su["nmom"], su["nisp"], consumed,
        rgrid, rhoP, rhoN, radius, cfg, keys[i])
    r = jnp.linalg.norm(pos, axis=1)
    last_scat_r = jnp.where(do > 0, r, last_scat_r)        # radius of the most-recent elastic scatter
    nsc = nsc + do
p_fin = np.linalg.norm(np.asarray(p4)[:, 1:], axis=1)
lsr = np.asarray(last_scat_r); nsc = np.asarray(nsc)
captured = p_fin < 1.0
escaped_scat = (~captured) & (nsc > 0)                      # scattered but escaped
print(f"[C proton] captured={np.average(captured,weights=w):.4f}  radius={radius:.2f} fm  "
      f"scattered-and-captured frac with a recorded last-scatter: {(captured&(lsr>0)).sum()}/{captured.sum()}")

# histogram last-scatter radius for captured vs escaped-scattered
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
edges = np.linspace(0, radius * 1.05, 31)
for mask, lbl, c in [(captured & (lsr > 0), "captured", "crimson"), (escaped_scat, "escaped (scattered)", "steelblue")]:
    ax[0].hist(lsr[mask], bins=edges, weights=w[mask], density=True, histtype="step", lw=2, color=c, label=lbl)
ax[0].set_xlabel("radius of last elastic scatter [fm]"); ax[0].set_ylabel("normalized"); ax[0].legend()
ax[0].set_title(f"C single-pass: last-scatter radius by fate (R={radius:.2f} fm)")
ax[0].axvline(radius, ls="--", c="k", lw=1)
# overlay k_F(r) and rho_p(r)
rr = np.linspace(0, radius * 1.05, 200)
kf = np.asarray(_kf_local(_rho_species(jnp.asarray(rr), rgrid, rhoP)))
rho = np.asarray(_rho_species(jnp.asarray(rr), rgrid, rhoP))
ax2 = ax[1]; ax2.plot(rr, kf, "g-", lw=2, label="$k_F(r)$ [MeV]")
ax2.axhline(137, ls=":", c="crimson", label="KE=10 MeV ($|p|$=137)")
ax2.set_xlabel("radius [fm]"); ax2.set_ylabel("$k_F$ [MeV]"); ax2.legend(loc="upper right")
ax3 = ax2.twinx(); ax3.plot(rr, rho, "b--", lw=1.5, alpha=0.6); ax3.set_ylabel(r"$\rho_p(r)$ [fm$^{-3}$]", color="b")
ax2.set_title("local $k_F(r)$ vs the 137 MeV capture threshold")
ax2.axvline(radius, ls="--", c="k", lw=1)
plt.tight_layout(); out = "paper_figures/capture_vs_radius_C.png"; plt.savefig(out, dpi=110)
print(f"wrote {out}")
# the smoking gun: where k_F drops below 137 (capture can happen via scatter beyond this radius)
r_cross = rr[np.argmax(kf < 137)] if (kf < 137).any() else radius
print(f"k_F(r) drops below 137 MeV (capture threshold) at r={r_cross:.2f} fm  (=> scatters beyond here can degrade to capture)")
print(f"fraction of captured protons whose last scatter was beyond r={r_cross:.2f}: "
      f"{np.average((lsr[captured&(lsr>0)]>r_cross), weights=w[captured&(lsr>0)]):.3f}")

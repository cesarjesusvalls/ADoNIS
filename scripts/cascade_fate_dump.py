"""Single-pass primary-nucleon FATE dump for the ADoNIS cascade (NO daughter re-cascade).

Methodical cascade diagnostic: take the pre-FSI primary nucleons, propagate each ONCE through the
Argon background (the leading-nucleon walk = the pool `_nucleon_step`, primary alone: its spawns are
recorded but NOT re-cascaded), and classify its fate + its produced knockouts -- granular, per species,
per initial-momentum bin, so it can be compared EXACTLY against an ACHILLES single-FSI-pass dump.

Fate categories (mutually exclusive, primary only):
  ESCAPED_FREE : never interacted (nsc==0), left the nucleus           (transparency)
  ELASTIC      : >=1 NN-elastic scatter, no pion, escaped (degraded |p|)
  INELASTIC    : produced a pion (NN->NDelta->NN'pi)
  RECAPTURED   : reached the boundary with KE<10 MeV -> bound (#8)      (absorbed)

Per primary we also record: initial |p|, final |p|, #proton-knockouts, #neutron-knockouts, and the
leading knockout |p|.  Saved to an npz for the ADoNIS-vs-ACHILLES comparison.

Run: python -u scripts/cascade_fate_dump.py [Ar|C] [proton|neutron] [n_events] [out.npz]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ADONIS_N_RECOIL", "6")          # generous so knockout counting is not truncated
import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
from adonis.workflow.materials import resolve_targets
from adonis.workflow.generate import gen_events
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _nucleon_step, _load_density
from adonis.fsi.cascade_full import setup_carbon
from adonis.xsec.spectral import SpectralFunction
import adonis.xsec.flux as flux
from adonis.fsi import oset_xsec as ox

M_N = float(ox.M_N)
mat = sys.argv[1] if len(sys.argv) > 1 else "Ar"
species = sys.argv[2] if len(sys.argv) > 2 else "proton"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
out = sys.argv[4] if len(sys.argv) > 4 else f"/tmp/ado_fate_{mat}_{species}.npz"
pauli = (sys.argv[5].lower() not in ("0", "false", "nopauli")) if len(sys.argv) > 5 else True
max_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 260
flux.BEAM_MODE = "is"

tg = resolve_targets(mat)[0][0]
cfg = DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=max_steps, seed=1, nn_inelastic=True,
                            pauli=pauli, nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
print(f"[cfg] pauli={pauli} max_steps={max_steps}")
sf_n = SpectralFunction(tg.spectral_n)

# Primary nucleons: use the QE struck->outgoing proton momenta as the realistic primary |p| spectrum.
# For the neutron-primary characterization we keep the SAME momentum spectrum but set species=neutron,
# isolating the cascade's species dependence at matched kinematics.
a = gen_events("qe", N, 0, sf_n=sf_n, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); sel = w > 0
pN = jnp.asarray(a["p_N"][sel]); ipid = jnp.asarray(a["ipid"][sel]); m = int(sel.sum()); w = w[sel]
is_p = jnp.full(m, species == "proton")

su = setup_carbon(pN, jnp.zeros(m, jnp.int32), ipid.astype(jnp.int32), cfg, jax.random.PRNGKey(11))
# POOL single-pass: iterate _nucleon_step on the PRIMARY ALONE -- spawns (knockouts/created pion) are
# NOT inserted, so the primary shares the consumed mask with nothing (faithful isolated single-pass in
# the pool's per-step physics).  We still RECORD the spawns it WOULD emit (knockout counts, made_pi).
rgrid, rhoP, rhoN, radius = _load_density(tg.density_p, tg.density_n)
keys = jax.random.split(jax.random.PRNGKey(99), max_steps)
p4 = pN; pos = su["pos0"]; d3 = p4[:, 1:]
dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
fz = jnp.zeros(m); alive = jnp.ones(m, bool); consumed = su["consumed0"]
nsc = jnp.zeros(m, jnp.int32); made_pi = jnp.zeros(m, bool)
n_ko_p = jnp.zeros(m, jnp.int32); n_ko_n = jnp.zeros(m, jnp.int32); lead_ko = jnp.zeros(m)
for i in range(max_steps):
    (p4, pos, dhat, fz, alive), esc, recap, do, ko, pin, consumed, _ = _nucleon_step(
        p4, pos, dhat, fz, is_p, alive, su["npos"], su["nmom"], su["nisp"], consumed,
        rgrid, rhoP, rhoN, radius, cfg, keys[i])
    nsc = nsc + do                                             # elastic-scatter count
    ko_p4, ko_pos, ko_fz, ko_q, ko_al = ko
    made_pi = made_pi | pin[4]                                 # created-pion (NN->NDelta->Npi) emitted
    n_ko_p = n_ko_p + (ko_al & (ko_q == 1)).astype(jnp.int32)
    n_ko_n = n_ko_n + (ko_al & (ko_q == 0)).astype(jnp.int32)
    lead_ko = jnp.maximum(lead_ko, jnp.linalg.norm(ko_p4[:, 1:], axis=1) * ko_al)
pN_out = np.asarray(p4); nsc = np.asarray(nsc); made_pi = np.asarray(made_pi)
n_ko_p = np.asarray(n_ko_p); n_ko_n = np.asarray(n_ko_n); lead_ko = np.asarray(lead_ko)

p_init = np.linalg.norm(np.asarray(pN)[:, 1:], axis=1)
p_fin = np.linalg.norm(pN_out[:, 1:], axis=1)
ke_fin = pN_out[:, 0] - M_N

# fate codes: 0 ESCAPED_FREE, 1 ELASTIC, 2 INELASTIC, 3 RECAPTURED
fate = np.where(made_pi, 2, np.where(p_fin < 1.0, 3, np.where(nsc == 0, 0, 1)))
names = ["ESCAPED_FREE", "ELASTIC", "INELASTIC", "RECAPTURED"]

def wfrac(mask): return float((w * mask).sum() / w.sum())
print(f"=== ADoNIS single-pass primary fate: {mat} {species}  (N={m}, weighted) ===")
print(f"primary <|p|>={(w*p_init).sum()/w.sum():.1f} MeV")
print(f"{'fate':14s} {'frac':>8s}  {'<|p|_in>':>9s} {'<|p|_fin>':>10s}")
for c, nm in enumerate(names):
    mk = fate == c
    fr = wfrac(mk)
    pin = (w[mk]*p_init[mk]).sum()/max((w[mk]).sum(),1e-30) if mk.any() else 0
    pfn = (w[mk]*p_fin[mk]).sum()/max((w[mk]).sum(),1e-30) if mk.any() else 0
    print(f"{nm:14s} {fr:8.4f}  {pin:9.1f} {pfn:10.1f}")
print(f"\nknockouts/primary:  proton={ (w*n_ko_p).sum()/w.sum():.3f}  neutron={ (w*n_ko_n).sum()/w.sum():.3f}")
print(f"mean elastic scatters nsc = {(w*nsc).sum()/w.sum():.3f}")
# fate fractions by initial-|p| bin
print(f"\n{'p_in bin':>14s} | " + " ".join(f"{n:>12s}" for n in names))
edges = [0, 200, 350, 500, 700, 1000, 5000]
for lo, hi in zip(edges[:-1], edges[1:]):
    b = (p_init >= lo) & (p_init < hi)
    if not b.any(): continue
    wb = w[b]; row = [f"[{lo},{hi})"]
    for c in range(4):
        row.append(f"{(wb*(fate[b]==c)).sum()/wb.sum():12.4f}")
    print(" ".join(f"{x:>14s}" if i == 0 else x for i, x in enumerate(row)))

np.savez(out, p_init=p_init, p_fin=p_fin, ke_fin=ke_fin, nsc=nsc, made_pi=made_pi,
         fate=fate, n_ko_p=n_ko_p, n_ko_n=n_ko_n, lead_ko=lead_ko, w=w,
         species=species, material=mat)
print(f"\nwrote {out}")

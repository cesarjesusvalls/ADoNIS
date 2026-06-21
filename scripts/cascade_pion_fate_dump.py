"""Single-pass primary-PION fate dump for the ADoNIS cascade -- the pion analog of
scripts/cascade_fate_dump.py (which does p/n).  Take the pre-FSI primary RES pions, seed the nucleus
EXACTLY as the production cascade (setup_carbon), propagate each pion ONCE by iterating the validated
`_pion_step` (the same per-step physics the pooled production engine runs), and classify its fate +
record initial |p|, so it can be compared per-momentum-bin against an ACHILLES achilles:fatedump.

Fate categories (mutually exclusive, primary pion only):
  TRANSMITTED  : never interacted (nsc==0), escaped                       (transparency)
  QUASI_ELASTIC: >=1 scatter, pion survives with SAME charge
  CHARGE_EX    : pion survives but charge changed (pi+ -> pi0 etc.)       (SCX)
  ABSORBED     : piNN->NN, pion removed                                   (absorption)
  CONVERTED    : eta-N' conversion, pion removed

Run: python -u scripts/cascade_pion_fate_dump.py [C|Ar] [n_res] [out.npz] [max_steps]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ADONIS_N_RECOIL", "6")
import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
from adonis.workflow.materials import resolve_targets
from adonis.xsec import res_xsec
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _pion_step, _load_density
from adonis.fsi.cascade_full import setup_carbon
from adonis.xsec.spectral import SpectralFunction
import adonis.xsec.flux as flux

mat = sys.argv[1] if len(sys.argv) > 1 else "C"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 120000
out = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/ado_pionfate_{mat}.npz"
max_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 260
flux.BEAM_MODE = "is"

tg = resolve_targets(mat)[0][0]
cfg = DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=max_steps, seed=1, nn_inelastic=True,
                            pauli=True, nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
assert cfg.algo == "step"
sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)

# primary RES pions (pre-cascade) -- the realistic primary-pion |p| spectrum, exactly as production
e = res_xsec.generate(N, seed=0, return_events=True, sf_n=sf_n, sf_p=sf_p,
                      n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
w = np.asarray(e["w"]); sel = w > 0; w = w[sel]
ppi = jnp.asarray(np.asarray(e["p_pi"])[sel])
ppid = jnp.asarray(np.asarray(e["ppid"])[sel].astype(np.int32))
ipid = jnp.asarray(np.asarray(e["ipid"])[sel].astype(np.int32))
m = int(sel.sum())

su = setup_carbon(ppi, ppid, ipid, cfg, jax.random.PRNGKey(11))
npos, nmom, nisp = su["npos"], su["nmom"], su["nisp"]
rgrid, rhoP, rhoN, radius = _load_density(tg.density_p, tg.density_n)
keys = jax.random.split(jax.random.PRNGKey(99), max_steps)

p4 = ppi; pos = su["pos0"]; ch = su["ch0"]; ch_init = np.asarray(su["ch0"])
nsc = jnp.zeros(m, jnp.int32); alive = jnp.ones(m, bool); consumed = su["consumed0"]
absorbed = jnp.zeros(m, bool); converted = jnp.zeros(m, bool)
# absorption-product (piNN->NN) nucleon momenta + charges, captured the step the pion is absorbed
ap1 = jnp.zeros(m); ap2 = jnp.zeros(m); aq1 = jnp.zeros(m, jnp.int32); aq2 = jnp.zeros(m, jnp.int32)
for i in range(max_steps):
    d3 = p4[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
    (p4, pos, _dp, ch, nsc, alive), esc, is_abs, is_conv, s1, s2, consumed, srec = _pion_step(
        p4, pos, dhat, ch, nsc, alive, npos, nmom, nisp, consumed, rgrid, rhoP, rhoN, radius, cfg, keys[i])
    new_abs = is_abs & ~absorbed
    ap1 = jnp.where(new_abs, jnp.linalg.norm(s1[0][:, 1:], axis=1), ap1)
    ap2 = jnp.where(new_abs, jnp.linalg.norm(s2[0][:, 1:], axis=1), ap2)
    aq1 = jnp.where(new_abs, s1[3].astype(jnp.int32), aq1)
    aq2 = jnp.where(new_abs, s2[3].astype(jnp.int32), aq2)
    absorbed = absorbed | is_abs; converted = converted | is_conv
ap1 = np.asarray(ap1); ap2 = np.asarray(ap2); aq1 = np.asarray(aq1); aq2 = np.asarray(aq2)

p_init = np.linalg.norm(np.asarray(ppi)[:, 1:], axis=1)
p_fin = np.linalg.norm(np.asarray(p4)[:, 1:], axis=1)
nsc = np.asarray(nsc); absorbed = np.asarray(absorbed); converted = np.asarray(converted)
ch_fin = np.asarray(ch)

# fate codes: 0 TRANSMITTED, 1 QUASI_ELASTIC, 2 CHARGE_EX, 3 ABSORBED, 4 CONVERTED
fate = np.where(absorbed, 3, np.where(converted, 4,
       np.where(ch_fin != ch_init, 2, np.where(nsc > 0, 1, 0))))
names = ["TRANSMITTED", "QUASI_ELASTIC", "CHARGE_EX", "ABSORBED", "CONVERTED"]

def wf(mask): return float((w * mask).sum() / w.sum())
print(f"=== ADoNIS single-pass primary-PION fate: {mat}  (N={m}, weighted) ===")
print(f"primary <|p_pi|>={(w*p_init).sum()/w.sum():.1f} MeV  pi+ frac={wf(ch_init==0):.3f} pi0 frac={wf(ch_init==1):.3f}")
print(f"{'fate':14s} {'frac':>8s}  {'<|p|_in>':>9s}")
for c, nm in enumerate(names):
    mk = fate == c
    pin = (w[mk]*p_init[mk]).sum()/max(w[mk].sum(), 1e-30) if mk.any() else 0
    print(f"{nm:14s} {wf(mk):8.4f}  {pin:9.1f}")
print(f"\n{'p_in bin':>13s} | " + " ".join(f"{n:>13s}" for n in names))
edges = [0, 100, 150, 200, 250, 300, 400, 500, 700, 2000]
for lo, hi in zip(edges[:-1], edges[1:]):
    b = (p_init >= lo) & (p_init < hi)
    if not b.any(): continue
    wb = w[b]; row = " ".join(f"{(wb*(fate[b]==c)).sum()/wb.sum():13.4f}" for c in range(5))
    print(f"{f'[{lo},{hi})':>13s} | {row}")

# absorption-product proton spectrum (the two piNN->NN nucleons), per primary-pion |p| bin
THR = 250.0
ab = (fate == 3)
print(f"\n=== absorption-product nucleons (piNN->NN), pi+ absorbed events, ejection thr={THR:.0f} MeV ===")
print(f"{'pi |p| bin':>13s} {'Nabs':>7s} {'<p1>':>7s} {'<p2>':>7s} {'<#p>250>':>9s} {'frac 0 above':>12s}")
mpip = ab & (ch_init == 0)
for lo, hi in zip([0, 150, 200, 250, 300, 400, 700], [150, 200, 250, 300, 400, 700, 2000]):
    b = mpip & (p_init >= lo) & (p_init < hi)
    if not b.any(): continue
    wb = w[b]; n_above = ((ap1[b] > THR).astype(float) + (ap2[b] > THR).astype(float))
    # protons only (charge==1); count protons above threshold
    np_above = ((ap1[b] > THR) & (aq1[b] == 1)).astype(float) + ((ap2[b] > THR) & (aq2[b] == 1)).astype(float)
    f0 = float((wb * (np_above == 0)).sum() / wb.sum())
    print(f"{f'[{lo},{hi})':>13s} {int(b.sum()):7d} {(wb*ap1[b]).sum()/wb.sum():7.1f} "
          f"{(wb*ap2[b]).sum()/wb.sum():7.1f} {(wb*np_above).sum()/wb.sum():9.3f} {f0:12.3f}")
np.savez(out, p_init=p_init, p_fin=p_fin, nsc=nsc, absorbed=absorbed, converted=converted,
         ch_init=ch_init, ch_fin=ch_fin, fate=fate, w=w, material=mat,
         abs_p1=ap1, abs_p2=ap2, abs_q1=aq1, abs_q2=aq2)
print(f"\nwrote {out}")

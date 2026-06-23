"""STANDALONE Cat-3 buffer disentangle: separate the pool-STACK overflow (sofl, controlled by P) from
the OUT-buffer overflow (oofl, controlled by M_out) -- cascade_nucleus combines them, so this calls
run_cascade_pool directly and sweeps P and M_out independently.  Ar (A=40) worst case.

Also flags max_steps truncation (particles still alive when the step loop ends -> silently lost).

Run: python -u scripts/test_cascade_buffers.py [Ar|C] [N]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAT = sys.argv[1] if len(sys.argv) > 1 else "Ar"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
os.environ.setdefault("ADONIS_N_RECOIL", "8")
import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
from adonis.workflow.materials import resolve_targets
from adonis.xsec import res_xsec
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec.spectral import SpectralFunction
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density
from adonis.fsi.cascade_full import (setup_nucleus, make_pool_stepper, run_cascade_pool, empty_batch,
                                     PION, NUCLEON, _ORIG_PRIM_PI)


def g0_res(ppi, pN, ppid, Npid, ipid, cfg, key):
    n = pN.shape[0]
    su = setup_nucleus(ppi, jnp.asarray(ppid, jnp.int32), jnp.asarray(ipid, jnp.int32), cfg, key)
    g0 = empty_batch(n, 2)
    g0["alive"] = jnp.ones((n, 2), bool)
    g0["species"] = jnp.array([PION, NUCLEON], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)
    g0["charge"] = jnp.stack([su["ch0"], (Npid == 2212).astype(jnp.int32)], axis=1)
    g0["p4"] = jnp.stack([ppi, pN], axis=1)
    g0["pos"] = jnp.broadcast_to(su["pos0"][:, None, :], (n, 2, 3))
    g0["origin"] = jnp.array([_ORIG_PRIM_PI, 0], jnp.int32)[None, :] * jnp.ones((n, 1), jnp.int32)
    return su, g0


def main():
    tg = resolve_targets(MAT)[0][0]
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(1000, int(np.ceil(3.0 * radius / 0.04)))
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True, pauli=True,
                                nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    e = res_xsec.generate(N, seed=0, return_events=True, sf_n=SpectralFunction(tg.spectral_n),
                          sf_p=SpectralFunction(tg.spectral_p), n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
    rw = np.asarray(e["w"]); rs = rw > 0
    ppi = jnp.asarray(np.asarray(e["p_pi"])[rs]); ppid = np.asarray(e["ppid"])[rs]
    pN = jnp.asarray(np.asarray(e["p_N"])[rs]); Npid = jnp.asarray(np.asarray(e["Npid"])[rs], jnp.int32)
    ipid = np.asarray(e["ipid"])[rs]; n = int(rs.sum())
    k = jax.random.PRNGKey(31)
    su, g0 = g0_res(ppi, pN, ppid, Npid, ipid, cfg, k)
    stepper = make_pool_stepper(su, cfg)
    print(f"[buffers] {MAT} A={tg.A} radius={radius:.2f} max_steps={ms} N={n}  (RES; sofl=stack/P, oofl=out/M_out)")
    print(f"{'P':>4} {'M_out':>6} | {'sofl(stack)':>12} {'oofl(out)':>12}   ({'% of N':>8})")
    for P, Mo in [(12, 24), (16, 24), (16, 48)]:                  # baseline / P-only / +M_out -> disentangle
        out, sofl, oofl, prim = run_cascade_pool(g0, stepper, jax.random.fold_in(k, 1), su["consumed0"],
                                                 M=P, max_steps=cfg.max_steps, M_out=Mo, prim_origin=_ORIG_PRIM_PI)
        s, o = int(sofl), int(oofl)
        print(f"{P:>4} {Mo:>6} | {s:>12} {o:>12}   ({100*(s+o)/n:>7.2f}%)", flush=True)
    print("  (production was P=12,M_out=24; new default P=16.  M_out is HARDCODED 24 in _cascade_pool.)")


if __name__ == "__main__":
    main()

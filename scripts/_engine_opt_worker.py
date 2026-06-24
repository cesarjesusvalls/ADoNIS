"""One-config cascade throughput measurer (for the (n_workers, n_w, P) study).  Runs cascade_nucleus on a
fixed C RES sample of N events at (P, n_w, q_cap); times the 1st call (compile+run = JIT) and the 2nd
(pure run); prints ONE JSON line.  Spawned in parallel by _engine_opt_study.py to measure worker scaling.

Run: python -u scripts/_engine_opt_worker.py N P n_w q_cap [seed]
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ADONIS_N_RECOIL", "8")
import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
from adonis.workflow.materials import resolve_targets
from adonis.xsec import res_xsec
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec.spectral import SpectralFunction
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density
from adonis.fsi.cascade_full import cascade_nucleus

N = int(sys.argv[1]); P = int(sys.argv[2])
n_w = None if sys.argv[3] in ("none", "None") else int(sys.argv[3])
q_cap = None if sys.argv[4] in ("none", "None") else int(sys.argv[4])
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0


def main():
    tg = resolve_targets("C")[0][0]
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(1000, int(np.ceil(3.0 * radius / 0.04)))
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True, pauli=True,
                                nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    e = res_xsec.generate(N, seed=SEED, return_events=True, sf_n=SpectralFunction(tg.spectral_n),
                          sf_p=SpectralFunction(tg.spectral_p), n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
    rs = np.asarray(e["w"]) > 0
    ppi = jnp.asarray(np.asarray(e["p_pi"])[rs]); ppid = jnp.asarray(np.asarray(e["ppid"])[rs], jnp.int32)
    pN = jnp.asarray(np.asarray(e["p_N"])[rs]); Npid = jnp.asarray(np.asarray(e["Npid"])[rs], jnp.int32)
    ipid = jnp.asarray(np.asarray(e["ipid"])[rs], jnp.int32)
    n = ppi.shape[0]
    a = (ppi, pN, ppid, ipid, Npid)
    t0 = time.time()
    r = cascade_nucleus(*a, cfg, KEY := jax.random.PRNGKey(20240623 + SEED), P=P, channel="res", n_w=n_w, q_cap=q_cap)
    jax.block_until_ready(r[0]["p4"]); t1 = time.time()
    r = cascade_nucleus(*a, cfg, KEY, P=P, channel="res", n_w=n_w, q_cap=q_cap)
    jax.block_until_ready(r[0]["p4"]); t2 = time.time()
    run_s = t2 - t1
    out = dict(N=n, P=P, n_w=sys.argv[3], q_cap=sys.argv[4], compile_run_s=round(t1 - t0, 2),
               jit_s=round((t1 - t0) - run_s, 2), run_s=round(run_s, 3), evps=round(n / run_s, 1),
               ofl_pct=round(100.0 * float(np.asarray(r[2])) / n, 4))
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()

"""One full-batch worker for the 1-vs-2 worker comparison.  Generates N C-RES events, runs ONE full-batch
cascade (n_w=0 = whole batch, queue ON) compile+run, prints JSON {n_eff, P, jit_s, run_s, evps, ofl_pct}.
Launched 1x (64k) or 2x-in-parallel (32k each) by the orchestrator; concurrent run phases reflect core
contention.  Run: python -u scripts/_engine_wcmp_worker.py N P seed
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

N = int(sys.argv[1]); P = int(sys.argv[2]); SEED = int(sys.argv[3])
KEY = jax.random.PRNGKey(20240623 + SEED)


def main():
    tg = resolve_targets("C")[0][0]
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(1000, int(np.ceil(3.0 * radius / 0.04)))
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=100000, seed=1, nn_inelastic=True, pauli=True,
                                nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    ngen = int(N / 2.4) + 4000
    e = res_xsec.generate(ngen, seed=SEED, return_events=True, sf_n=SpectralFunction(tg.spectral_n),
                          sf_p=SpectralFunction(tg.spectral_p), n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
    rs = np.asarray(e["w"]) > 0
    ppi = jnp.asarray(np.asarray(e["p_pi"])[rs][:N]); ppid = jnp.asarray(np.asarray(e["ppid"])[rs][:N], jnp.int32)
    pN = jnp.asarray(np.asarray(e["p_N"])[rs][:N]); Npid = jnp.asarray(np.asarray(e["Npid"])[rs][:N], jnp.int32)
    ipid = jnp.asarray(np.asarray(e["ipid"])[rs][:N], jnp.int32)
    n = ppi.shape[0]; a = (ppi, pN, ppid, ipid, Npid)
    t = time.time(); r = cascade_nucleus(*a, cfg, KEY, channel="res", n_w=0, q_cap=64)
    jax.block_until_ready(r[0]["p4"]); t_c = time.time() - t
    t = time.time(); r = cascade_nucleus(*a, cfg, KEY, channel="res", n_w=0, q_cap=64)
    jax.block_until_ready(r[0]["p4"]); t_r = time.time() - t
    print("RESULT " + json.dumps(dict(n_eff=n, P=P, jit_s=round(t_c - t_r, 1), run_s=round(t_r, 1),
          evps=round(n / t_r, 1), ofl_pct=round(100.0 * float(np.asarray(r[2])) / n, 4))), flush=True)


if __name__ == "__main__":
    main()

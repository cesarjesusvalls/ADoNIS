"""One worker for the (n_workers, total_events, P) study.  Generates N_gen C-RES events, processes them
through the PRODUCTION default engine (auto-refill + waiting-queue) in fixed-size chunks, and reports its
JIT time + steady throughput + cascade wall (generation excluded) as one JSON line.

JIT vs run separation: the first chunk is run twice (compile, then pure run); jit_s = compile - run,
steady_evps = chunk/run.  Then ALL chunks are processed once under a timer = cascade_wall (compiled).

Run: python -u scripts/_engine_opt_worker.py N_gen P seed [chunk=16000]
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

N_GEN = int(sys.argv[1]); P = int(sys.argv[2]); SEED = int(sys.argv[3])
CHUNK = int(sys.argv[4]) if len(sys.argv) > 4 else 16000
KEY = jax.random.PRNGKey(20240623 + SEED)


def main():
    tg = resolve_targets("C")[0][0]
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(1000, int(np.ceil(3.0 * radius / 0.04)))
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True, pauli=True,
                                nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    e = res_xsec.generate(N_GEN, seed=SEED, return_events=True, sf_n=SpectralFunction(tg.spectral_n),
                          sf_p=SpectralFunction(tg.spectral_p), n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
    rs = np.asarray(e["w"]) > 0
    ppi = jnp.asarray(np.asarray(e["p_pi"])[rs]); ppid = jnp.asarray(np.asarray(e["ppid"])[rs], jnp.int32)
    pN = jnp.asarray(np.asarray(e["p_N"])[rs]); Npid = jnp.asarray(np.asarray(e["Npid"])[rs], jnp.int32)
    ipid = jnp.asarray(np.asarray(e["ipid"])[rs], jnp.int32)
    n_all = ppi.shape[0]
    nchunk = max(1, n_all // CHUNK); n_use = nchunk * CHUNK            # equal chunks -> single compile
    def call(sl):
        r = cascade_nucleus(ppi[sl], pN[sl], ppid[sl], ipid[sl], Npid[sl], cfg, KEY, P=P, channel="res")
        jax.block_until_ready(r[0]["p4"]); return float(np.asarray(r[2]))
    c0 = slice(0, CHUNK)
    t = time.time(); call(c0); t_c = time.time() - t                  # compile + run
    t = time.time(); call(c0); t_s = time.time() - t                  # pure run
    jit_s = max(0.0, t_c - t_s); steady = CHUNK / t_s
    t = time.time(); ofl = 0.0
    for i in range(nchunk):
        ofl += call(slice(i * CHUNK, (i + 1) * CHUNK))
    wall = time.time() - t
    print("RESULT " + json.dumps(dict(N_gen=N_GEN, n_eff=n_use, P=P, jit_s=round(jit_s, 1),
          steady_evps=round(steady, 1), cascade_wall_s=round(wall, 1),
          ofl_pct=round(100.0 * ofl / n_use, 4))), flush=True)


if __name__ == "__main__":
    main()

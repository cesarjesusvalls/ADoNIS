"""Cascade-engine timing/utilization study (single process).  Sweeps P (stack width) x N (events per
kernel call) and reports, per cell: pure RUN time (2nd call, compile excluded), throughput (events/s),
and the overflow rate sofl+oofl (the speed-vs-correctness tradeoff -- small P drops particles).  Used to
find the best (N, P) and to test the underutilization hypothesis (small N/P -> small tensors -> idle
cores).  Pair with _engine_workers.sh for the worker-count axis.

Run: python -u scripts/_engine_study.py [Ngrid e.g. 2000,8000,32000] [Pgrid e.g. 8,16,24] [channel res|qe]
"""
import sys, os, time, threading, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ADONIS_N_RECOIL", "8")
import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
from adonis.workflow.materials import resolve_targets
from adonis.xsec import res_xsec, qe_xsec
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec.spectral import SpectralFunction
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density
from adonis.fsi.cascade_full import cascade_nucleus

NG = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2000, 8000, 32000]
PG = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [8, 16, 24]
CH = sys.argv[3] if len(sys.argv) > 3 else "res"
KEY = jax.random.PRNGKey(20240623)


@contextlib.contextmanager
def ticker(label, every=15):
    """Heartbeat thread: prints '<label> ... <s>s elapsed' every `every` seconds so the opaque XLA
    compile shows live progress even when a cell takes longer than expected."""
    t0 = time.time(); stop = threading.Event()

    def run():
        while not stop.wait(every):
            print(f"    ... {label}: {time.time()-t0:.0f}s elapsed", flush=True)
    th = threading.Thread(target=run, daemon=True); th.start()
    try:
        yield
    finally:
        stop.set(); th.join(timeout=1)


def main():
    tg = resolve_targets("C")[0][0]
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(1000, int(np.ceil(3.0 * radius / 0.04)))
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True, pauli=True,
                                nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
    Nmax = max(NG)
    if CH == "res":
        e = res_xsec.generate(Nmax, seed=0, return_events=True, sf_n=sf_n, sf_p=sf_p,
                              n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
        rs = np.asarray(e["w"]) > 0
        ppi = jnp.asarray(np.asarray(e["p_pi"])[rs]); ppid = jnp.asarray(np.asarray(e["ppid"])[rs], jnp.int32)
        pN = jnp.asarray(np.asarray(e["p_N"])[rs]); Npid = jnp.asarray(np.asarray(e["Npid"])[rs], jnp.int32)
        ipid = jnp.asarray(np.asarray(e["ipid"])[rs], jnp.int32)
    else:
        q = qe_xsec.sample_importance(Nmax, seed=0, sf=sf_n, n_neutron=tg.A - tg.Z)
        m = np.asarray(q["w"]) != 0
        pN = jnp.asarray(np.asarray(q["p_out"])[m]); Npid = jnp.full(int(m.sum()), 2212, jnp.int32)
        ppi = jnp.zeros((pN.shape[0], 4)); ppid = jnp.zeros(pN.shape[0], jnp.int32); ipid = jnp.full(pN.shape[0], 2112, jnp.int32)
    navail = pN.shape[0]
    print(f"[study] C {CH}  max_steps={ms}  available={navail}  cores={os.cpu_count()}", flush=True)
    print(f"{'P':>4} {'N':>8} {'run(s)':>9} {'ev/s':>10} {'ofl%':>7}", flush=True)
    cells = [(P, N) for P in PG for N in NG]
    for ci, (P, N) in enumerate(cells):
        if True:
            n = min(N, navail)
            args = (ppi[:n], pN[:n], ppid[:n], ipid[:n], Npid[:n])
            tc = time.time()
            print(f"  [{ci+1}/{len(cells)}] compiling P={P} N={n} ...", flush=True)
            with ticker(f"cell {ci+1}/{len(cells)} P={P} N={n} compiling"):
                r = cascade_nucleus(*args, cfg, KEY, P=P, channel=CH)   # compile+run
                jax.block_until_ready(r[0]["p4"])
            print(f"  [{ci+1}/{len(cells)}] compile+run {time.time()-tc:.1f}s; timing pure run ...", flush=True)
            t0 = time.time()
            r = cascade_nucleus(*args, cfg, KEY, P=P, channel=CH)   # pure run
            jax.block_until_ready(r[0]["p4"]); dt = time.time() - t0
            ofl = float(np.asarray(r[2])); oflpct = 100.0 * ofl / n
            print(f"{P:>4} {n:>8} {dt:>9.3f} {n/dt:>10.0f} {oflpct:>7.3f}", flush=True)


if __name__ == "__main__":
    main()

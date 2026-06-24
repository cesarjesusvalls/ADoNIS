"""Refill-engine throughput study: fixed total N events, sweep the working-set width n_w (and the
lock-step baseline n_w=None).  Reports pure RUN time (compile excluded), throughput (ev/s), overflow.
The lock-step baseline runs `while any(alive)` across ALL N (one long event makes everyone step to
max_steps); refill lets finished events vacate, so a SMALL n_w should be much faster while a too-small
n_w underutilizes the cores.  Picks the overnight n_w.

Run: python -u scripts/_refill_study.py [N=12000] [nw_list=none,1024,4096,12000] [channel=res]
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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
NWL = sys.argv[2].split(",") if len(sys.argv) > 2 else ["none", "1024", "4096", "12000"]
CH = sys.argv[3] if len(sys.argv) > 3 else "res"
P = 16; KEY = jax.random.PRNGKey(20240623)


@contextlib.contextmanager
def ticker(label, every=15):
    t0 = time.time(); stop = threading.Event()
    th = threading.Thread(target=lambda: [print(f"    ... {label}: {time.time()-t0:.0f}s", flush=True)
                                          for _ in iter(lambda: stop.wait(every) or stop.is_set(), True)], daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()


def main():
    tg = resolve_targets("C")[0][0]
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(1000, int(np.ceil(3.0 * radius / 0.04)))
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True, pauli=True,
                                nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
    if CH == "res":
        e = res_xsec.generate(N, seed=0, return_events=True, sf_n=sf_n, sf_p=sf_p,
                              n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
        rs = np.asarray(e["w"]) > 0
        ppi = jnp.asarray(np.asarray(e["p_pi"])[rs]); ppid = jnp.asarray(np.asarray(e["ppid"])[rs], jnp.int32)
        pN = jnp.asarray(np.asarray(e["p_N"])[rs]); Npid = jnp.asarray(np.asarray(e["Npid"])[rs], jnp.int32)
        ipid = jnp.asarray(np.asarray(e["ipid"])[rs], jnp.int32)
    else:
        q = qe_xsec.sample_importance(N, seed=0, sf=sf_n, n_neutron=tg.A - tg.Z); m = np.asarray(q["w"]) != 0
        pN = jnp.asarray(np.asarray(q["p_out"])[m]); Npid = jnp.full(int(m.sum()), 2212, jnp.int32)
        ppi = jnp.zeros((pN.shape[0], 4)); ppid = jnp.zeros(pN.shape[0], jnp.int32); ipid = jnp.full(pN.shape[0], 2112, jnp.int32)
    n = pN.shape[0]
    print(f"[refill-study] C {CH}  N={n}  P={P}  max_steps={ms}  cores={os.cpu_count()}", flush=True)
    print(f"{'n_w':>8} {'run(s)':>9} {'ev/s':>10} {'ofl%':>7} {'speedup':>8}", flush=True)
    base = None
    for tok in NWL:
        nw = None if tok == "none" else min(int(tok), n)
        args = (ppi, pN, ppid, ipid, Npid)
        with ticker(f"n_w={tok} compile+run"):
            r = cascade_nucleus(*args, cfg, KEY, P=P, channel=CH, n_w=nw)
            jax.block_until_ready(r[0]["p4"])
        t0 = time.time()
        r = cascade_nucleus(*args, cfg, KEY, P=P, channel=CH, n_w=nw)
        jax.block_until_ready(r[0]["p4"]); dt = time.time() - t0
        ofl = float(np.asarray(r[2])); evps = n / dt
        if base is None:
            base = evps
        print(f"{tok:>8} {dt:>9.3f} {evps:>10.0f} {100*ofl/n:>7.3f} {evps/base:>8.2f}x", flush=True)


if __name__ == "__main__":
    main()

"""Full-batch P-scan: 256k C-RES events in ONE worker / ONE tensor (n_w = whole batch == lock-step; no
refill streaming), queue ON so small P keeps (not drops) overflow -> fair throughput.  Scans P and reports
per P: total JIT time, pure-run ev/s, overflow%.  P ascending so partial results survive an OOM at large P.

Run: python -u scripts/_engine_fullbatch_scan.py [N=256000] [Pgrid=2,4,6,8] [q_cap=64]
"""
import sys, os, time, threading, contextlib
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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256000
PG = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2,4,6,8").split(",")]
QCAP = int(sys.argv[3]) if len(sys.argv) > 3 else 64
KEY = jax.random.PRNGKey(20240623)


@contextlib.contextmanager
def ticker(label, every=20):
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
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=100000, seed=1, nn_inelastic=True, pauli=True,
                                nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    ngen = int(N / 2.4) + 5000                                        # res yields ~2.6x after the w>0 cut
    print(f"[fullbatch] generating ~{N} C-RES events (ngen={ngen}) ...", flush=True)
    e = res_xsec.generate(ngen, seed=0, return_events=True, sf_n=SpectralFunction(tg.spectral_n),
                          sf_p=SpectralFunction(tg.spectral_p), n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
    rs = np.asarray(e["w"]) > 0
    ppi = np.asarray(e["p_pi"])[rs][:N]; ppid = np.asarray(e["ppid"])[rs][:N].astype(np.int32)
    pN = np.asarray(e["p_N"])[rs][:N]; Npid = np.asarray(e["Npid"])[rs][:N].astype(np.int32)
    ipid = np.asarray(e["ipid"])[rs][:N].astype(np.int32)
    n = ppi.shape[0]
    a = (jnp.asarray(ppi), jnp.asarray(pN), jnp.asarray(ppid), jnp.asarray(ipid), jnp.asarray(Npid))
    print(f"[fullbatch] N={n}  n_w=full(lock-step)  q_cap={QCAP}  max_steps={ms}  cores={os.cpu_count()}", flush=True)
    print(f"{'P':>4} {'JIT(s)':>8} {'run(s)':>9} {'ev/s':>9} {'ofl%':>8}", flush=True)
    for P in PG:
        try:
            with ticker(f"P={P} compile"):
                t = time.time(); r = cascade_nucleus(*a, cfg, KEY, channel="res", n_w=0, q_cap=QCAP)
                jax.block_until_ready(r[0]["p4"]); t_c = time.time() - t
            t = time.time(); r = cascade_nucleus(*a, cfg, KEY, channel="res", n_w=0, q_cap=QCAP)
            jax.block_until_ready(r[0]["p4"]); t_r = time.time() - t
            ofl = float(np.asarray(r[2]))
            print(f"{P:>4} {t_c-t_r:>8.1f} {t_r:>9.2f} {n/t_r:>9.0f} {100*ofl/n:>8.4f}", flush=True)
        except Exception as ex:
            print(f"{P:>4}  FAILED: {type(ex).__name__}: {str(ex)[:120]}", flush=True)


if __name__ == "__main__":
    main()

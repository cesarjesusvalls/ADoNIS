"""Runtime timing for the cascade engine (separates COMPILE from RUN).  Times cascade_nucleus on a fixed
C RES sample: 1st call = compile+run, 2nd call = pure run (cache hit).  Reports events/sec at a few
(N, P) so we can tell whether the per-event RNG / refill changes the RUNTIME vs the old engine and whether
low-N/low-P underutilizes (user's hypothesis).  Run on HEAD (new) and on the pre-change commit (old) to
compare.

Run: python -u scripts/_engine_timing.py [N1,N2,...] [P]
"""
import sys, os, time
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

NS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [3000, 12000]
P = int(sys.argv[2]) if len(sys.argv) > 2 else 16
KEY = jax.random.PRNGKey(20240623)


def main():
    tg = resolve_targets("C")[0][0]
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(1000, int(np.ceil(3.0 * radius / 0.04)))
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True, pauli=True,
                                nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
    Nmax = max(NS)
    e = res_xsec.generate(Nmax, seed=0, return_events=True, sf_n=sf_n, sf_p=sf_p,
                          n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
    rs = np.asarray(e["w"]) > 0
    ppi = jnp.asarray(np.asarray(e["p_pi"])[rs]); ppid = jnp.asarray(np.asarray(e["ppid"])[rs], jnp.int32)
    pN = jnp.asarray(np.asarray(e["p_N"])[rs]); Npid = jnp.asarray(np.asarray(e["Npid"])[rs], jnp.int32)
    ipid = jnp.asarray(np.asarray(e["ipid"])[rs], jnp.int32)
    navail = ppi.shape[0]
    print(f"[timing] C RES  P={P}  max_steps={ms}  available events={navail}", flush=True)
    print(f"{'N':>8} {'compile+run(s)':>15} {'run(s)':>10} {'ev/s':>12}", flush=True)
    for N in NS:
        n = min(N, navail)
        a = (ppi[:n], pN[:n], ppid[:n], ipid[:n], Npid[:n])
        t0 = time.time()
        out = cascade_nucleus(a[0], a[1], a[2], a[3], a[4], cfg, KEY, P=P, channel="res")
        jax.block_until_ready(out[0]["p4"]); t1 = time.time()
        out = cascade_nucleus(a[0], a[1], a[2], a[3], a[4], cfg, KEY, P=P, channel="res")
        jax.block_until_ready(out[0]["p4"]); t2 = time.time()
        print(f"{n:>8} {t1-t0:>15.2f} {t2-t1:>10.3f} {n/(t2-t1):>12.0f}", flush=True)


if __name__ == "__main__":
    main()

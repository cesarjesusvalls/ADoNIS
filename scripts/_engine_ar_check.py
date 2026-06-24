"""Ar validation of the unified default engine (Ar is where overflow actually happens, unlike C).
Three runs on the SAME fixed Ar sample (RES+QE, P=16):
  LOCK   n_w=0,    q_cap=0   -- lock-step, legacy drop-on-overflow (reference)
  REFILL n_w=512,  q_cap=0   -- refill, still drop-on-overflow -> must be BIT-EXACT to LOCK (refill faithful on Ar)
  DEFAULT n_w=None,q_cap=None -- production default (auto-refill + waiting-queue) -> sofl MUST be 0 (queue keeps
                                 the overflow particles) and total escaped finals >= LOCK (the kept particles).
Confirms: (1) refill is bit-exact on Ar too, (2) the queue eliminates stack overflow (the plan's allowed change).

Run: python -u scripts/_engine_ar_check.py
"""
import sys, os
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

N = 6000; KEY = jax.random.PRNGKey(20240623); P = 16


def main():
    tg = resolve_targets("Ar")[0][0]
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(1000, int(np.ceil(3.0 * radius / 0.04)))
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True, pauli=True,
                                nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    e = res_xsec.generate(N, seed=0, return_events=True, sf_n=SpectralFunction(tg.spectral_n),
                          sf_p=SpectralFunction(tg.spectral_p), n_neutron=tg.A - tg.Z, n_proton=tg.Z)["events"]
    rs = np.asarray(e["w"]) > 0
    ppi = jnp.asarray(np.asarray(e["p_pi"])[rs]); ppid = jnp.asarray(np.asarray(e["ppid"])[rs], jnp.int32)
    pN = jnp.asarray(np.asarray(e["p_N"])[rs]); Npid = jnp.asarray(np.asarray(e["Npid"])[rs], jnp.int32)
    ipid = jnp.asarray(np.asarray(e["ipid"])[rs], jnp.int32)
    n = ppi.shape[0]
    print(f"[Ar check] N={n} RES  P={P}  max_steps={ms}", flush=True)

    def run(n_w, q_cap, tag):
        pt, nt, ofl, cr = cascade_nucleus(ppi, pN, ppid, ipid, Npid, cfg, KEY, P=P, channel="res",
                                          n_w=n_w, q_cap=q_cap)
        finals = int(np.asarray(nt[0]["alive"]).sum())
        print(f"  {tag:<8} n_w={str(n_w):<5} q={str(q_cap):<5} overflow(sofl+oofl)={int(ofl):<6} escaped_nucleons={finals}", flush=True)
        return dict(pt=pt, nt=nt, ofl=int(ofl), finals=finals)

    lock = run(0, 0, "LOCK")
    refill = run(512, 0, "REFILL")
    default = run(None, None, "DEFAULT")

    # (1) refill (q=0) bit-exact to lock-step
    d = float(np.abs(np.asarray(lock["nt"][0]["p4"]).astype(np.float64)
                     - np.asarray(refill["nt"][0]["p4"]).astype(np.float64)).max())
    same = (d < 1e-9) and np.array_equal(np.asarray(lock["nt"][0]["alive"]), np.asarray(refill["nt"][0]["alive"])) \
        and lock["ofl"] == refill["ofl"]
    print(f"\n(1) REFILL(q=0) == LOCK bit-exact: {same}  (max|d p4|={d:.2e}, ofl {lock['ofl']}=={refill['ofl']})")

    # (2) queue eliminates overflow + keeps particles
    print(f"(2) overflow: LOCK={lock['ofl']}  DEFAULT(queue)={default['ofl']}  -> queue drives overflow to "
          f"{'ZERO' if default['ofl'] == 0 else default['ofl']}")
    print(f"    escaped nucleons: LOCK={lock['finals']}  DEFAULT={default['finals']}  "
          f"(+{default['finals']-lock['finals']} kept by the queue)")
    ok = same and default["ofl"] == 0 and default["finals"] >= lock["finals"]
    print("\nRESULT:", "PASS" if ok else "FAIL (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

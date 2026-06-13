"""BANK the pion FSI fate, ONE generation -> all survival/redistribution analysis is then free.
Per event (carbon RES, t-channel sampler, spline amps2): vertex W, the PRE-cascade (primary) pion
4-vector + pid, the POST-cascade pion 4-vector + pid (DiscreteCascadeFSI, the SAME pion cascade as
res_C), the muon, and the absolute weight w.  Only the PION cascade is run (the proton side is
settled).  Output: data/oracle/t2k_cc1pi_pion_fate.npz.

Usage: python scripts/gen_pion_fate.py [NRES=120000] [NSEED=4]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from adonis.xsec import res_xsec
from adonis.core.event import EventRecord
import scripts.cc1pi_fig_tki as F        # pins spline; provides _CFG / cascade

NRES = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def one(seed):
    e = res_xsec.generate(NRES, seed=seed, return_events=True)["events"]
    knu, kmu, pstr = (np.asarray(e[k]) for k in ("k_nu", "k_mu", "p_struck"))
    ppi, pN, w = np.asarray(e["p_pi"]), np.asarray(e["p_N"]), np.asarray(e["w"])
    ppid = np.asarray(e["ppid"]); m = len(w)
    ev = EventRecord(k=jnp.asarray(knu), kp=jnp.asarray(kmu), p_struck=jnp.asarray(pstr),
                     p_pi=jnp.asarray(ppi), p_N=jnp.asarray(pN), w=jnp.asarray(w),
                     channel=jnp.zeros(m, jnp.int32), pid_pi=jnp.asarray(ppid, jnp.int32),
                     pid_N=jnp.full((m,), 2212, jnp.int32), pid_Ni=jnp.asarray(e["ipid"], jnp.int32),
                     W=jnp.zeros(m), Q2_adj=jnp.zeros(m))
    pion = F.DiscreteCascadeFSI(F._CFG(seed=1)); ev = pion.apply(None, ev, key=jax.random.PRNGKey(seed + 11))
    qv = knu - kmu; tot = qv + pstr
    W = np.sqrt(np.clip(tot[:, 0] ** 2 - np.sum(tot[:, 1:] ** 2, axis=1), 0, None))
    return dict(W=W, k_mu=kmu, pi_pre=ppi, pid_pre=ppid,
                pi_post=np.asarray(ev.p_pi), pid_post=np.asarray(ev.pid_pi), w=w / NSEED)


def main():
    parts = [one(sd) for sd in range(NSEED)]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez("data/oracle/t2k_cc1pi_pion_fate.npz", **out)
    n = len(out["w"])
    pre = out["pid_pre"] == 211
    surv = (out["pid_post"] == 211)
    print(f"banked {n} carbon RES events  sigma={out['w'].sum():.4e} nb", flush=True)
    print(f"  primary pi+: {int(pre.sum())}  ;  post-cascade still pi+: {int((pre&surv).sum())}  "
          f"(overall pi+ survival {(out['w'][pre&surv].sum()/out['w'][pre].sum()):.3f})", flush=True)
    print("wrote data/oracle/t2k_cc1pi_pion_fate.npz", flush=True)


if __name__ == "__main__":
    main()

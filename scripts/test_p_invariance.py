"""Phase 0 oracle: P-invariance test for the pool cascade.

Runs the SAME primaries + SAME frozen nucleus + SAME base RNG key through cascade_nucleus at several P
(stack widths) and compares the per-event final state.  By the engine's design ("P = pure capacity dial")
the result should be IDENTICAL for every P >= the event's concurrent occupancy.  This harness measures the
actual P-dependence: per-event exact-match fraction vs the P=12 reference + the weighted N(p)/N(n)
distribution per P.  PASS (post-fix) = 100% per-event match across all P.

Use small n (~5k).  Run:  CHANNEL=res N=5000 python -u scripts/test_p_invariance.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from adonis.workflow.generate import gen_events
from adonis.workflow.materials import resolve_targets
from adonis.fsi.cascade_real import _load_density
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
import adonis.fsi.cascade_full as CF

CH = os.environ.get("CHANNEL", "res")
N = int(os.environ.get("N", "5000"))
PS = [int(x) for x in os.environ.get("PS", "1,2,4,12").split(",")]
REF_P = max(PS)


def build_cfg():
    tg = resolve_targets("C")[0][0]
    _, _, _, r = _load_density(tg.density_p, tg.density_n)
    ms = max(600, int(np.ceil(3.0 * r / 0.04)))
    ms = max(ms, int(os.environ.get("MAXSTEPS", "0")))        # override to test step-cap truncation at small P
    return DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True,
                                 nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)


def mult(nterms):
    """per-event (n_p, n_n) ejected nucleon counts (|p|>0; charge 1=p,0=n) from gen-0 escaped finals."""
    g0 = nterms[0]
    sp = np.asarray(g0["species"]); ch = np.asarray(g0["charge"]); al = np.asarray(g0["alive"])
    al = al & (np.linalg.norm(np.asarray(g0["p4"])[:, :, 1:], axis=2) > 0.0)
    n_p = ((sp == CF.NUCLEON) & (ch == 1) & al).sum(1)
    n_n = ((sp == CF.NUCLEON) & (ch == 0) & al).sum(1)
    return n_p.astype(np.int64), n_n.astype(np.int64)


def wdist(c, w, nmax=6):
    c = np.clip(c, 0, nmax)
    return np.array([w[c == k].sum() for k in range(nmax + 1)])


def main():
    cfg = build_cfg()
    a = gen_events(CH, N, seed=0)                              # grid=None -> bit-identical primaries
    w = np.asarray(a["w"]); ne = len(w)
    p_pi = jnp.asarray(a["p_pi"]) if "p_pi" in a else jnp.asarray(a["p_N"])
    p_N = jnp.asarray(a["p_N"]); ppid = jnp.asarray(a["ppid"], jnp.int32)
    ipid = jnp.asarray(a["ipid"], jnp.int32); Npid = jnp.asarray(a["Npid"], jnp.int32)
    key = jax.random.PRNGKey(0 + 11)                          # SAME key for every P
    print(f"channel={CH}  n_events={ne} (n={N})  Ps={PS}  ref=P{REF_P}", flush=True)
    res = {}
    for P in PS:
        out = CF.cascade_nucleus(p_pi, p_N, ppid, ipid, Npid, cfg, key, P=P, channel=CH, n_w=0)
        res[P] = mult(out[1])
    rp, rn = res[REF_P]
    print(f"\n{'P':>3} {'meanN(p)':>9} {'meanN(n)':>9} {'match%(p)':>10} {'match%(n)':>10} {'match%(both)':>12}", flush=True)
    for P in PS:
        np_, nn_ = res[P]
        mp = float((np_ == rp).mean() * 100); mn = float((nn_ == rn).mean() * 100)
        mb = float(((np_ == rp) & (nn_ == rn)).mean() * 100)
        print(f"{P:>3} {(np_*w).sum()/w.sum():>9.4f} {(nn_*w).sum()/w.sum():>9.4f} {mp:>10.2f} {mn:>10.2f} {mb:>12.2f}", flush=True)
    print(f"\nweighted N(p) distribution (k=0..6) per P:", flush=True)
    for P in PS:
        print(f"  P={P:>3}: {wdist(res[P][0], w)}", flush=True)
    print(f"weighted N(n) distribution (k=0..6) per P:", flush=True)
    for P in PS:
        print(f"  P={P:>3}: {wdist(res[P][1], w)}", flush=True)
    allmatch = all(float(((res[P][0] == rp) & (res[P][1] == rn)).mean()) == 1.0 for P in PS)
    print(f"\nP-INVARIANT (per-event exact match across all P): {allmatch}", flush=True)


if __name__ == "__main__":
    main()

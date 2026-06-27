"""Engine throughput timing: time cascade_nucleus at a fixed P on a fixed batch, post-JIT.  Used to A/B
the P-invariant engine vs the pre-fix engine (swap adonis/fsi/cascade_full.py) -- confirms the sorted
reconcile + extra fields didn't degrade production-P throughput.

Run: CHANNEL=res N=10000 P=12 python -u scripts/_pinv_timing.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from adonis.workflow.generate import gen_events
from adonis.workflow.materials import resolve_targets
from adonis.fsi.cascade_real import _load_density
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
import adonis.fsi.cascade_full as CF

CH = os.environ.get("CHANNEL", "res"); N = int(os.environ.get("N", "10000")); P = int(os.environ.get("P", "12"))


def build_cfg():
    tg = resolve_targets("C")[0][0]
    _, _, _, r = _load_density(tg.density_p, tg.density_n)
    ms = max(600, int(np.ceil(3.0 * r / 0.04)))
    return DiscreteCascadeConfig(step=0.04, max_steps=100000, seed=1, nn_inelastic=True,
                                 nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)


def main():
    cfg = build_cfg()
    a = gen_events(CH, N, seed=0)
    w = np.asarray(a["w"]); ne = len(w)
    p_pi = jnp.asarray(a["p_pi"]) if "p_pi" in a else jnp.asarray(a["p_N"])
    p_N = jnp.asarray(a["p_N"]); ppid = jnp.asarray(a["ppid"], jnp.int32)
    ipid = jnp.asarray(a["ipid"], jnp.int32); Npid = jnp.asarray(a["Npid"], jnp.int32)
    key = jax.random.PRNGKey(11)

    def run():
        out = CF.cascade_nucleus(p_pi, p_N, ppid, ipid, Npid, cfg, key, channel=CH, n_w=0)
        out[1][0]["alive"].block_until_ready()
        return out

    t0 = time.time(); run(); t_c = time.time() - t0          # compile + run
    t0 = time.time(); run(); t_r = time.time() - t0          # post-JIT run
    t0 = time.time(); run(); t_r2 = time.time() - t0         # post-JIT run (2nd, confirm stable)
    tr = min(t_r, t_r2)
    print(f"CH={CH} N={N} events={ne} P={P}: compile+run={t_c:.1f}s  run={tr:.2f}s  {ne/tr:.0f} ev/s", flush=True)


if __name__ == "__main__":
    main()

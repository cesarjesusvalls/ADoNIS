"""Fast (no-physics) test of the persistent-refill BOOKKEEPING in run_cascade_pool: a toy stepper over N
events, comparing (i) refill n_w=N == no-refill BIT-EXACT (flush-to-global reproduces lock-step), and
(ii) refill n_w<N collects the SAME per-event terminals by evt_id (real slot reuse).  Seconds, not minutes.
Run: python -u scripts/_refill_toy_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from adonis.fsi.cascade_full import empty_batch, run_cascade_pool

N, M, A = 40, 6, 4


def toy(stk, key, state, step=0, bg=None):
    """Each live slot: tag decremented per step, escapes (terminal) at tag<=0.5; spawns ONE child (tag 1)
    while tag>=3.  Ignores bg/state (refill swaps them but the toy physics doesn't read them)."""
    n = stk["alive"].shape[0]
    tag = stk["p4"][..., 0]; alive = stk["alive"]
    terminal = alive & (tag <= 0.5)
    stk2 = {**stk, "p4": stk["p4"].at[..., 0].add(jnp.where(alive, -1.0, 0.0))}
    spawn_here = alive & ~terminal & (tag >= 3.0)
    spawn = empty_batch(n, M); spawn["alive"] = spawn_here
    spawn["p4"] = spawn["p4"].at[..., 0].set(jnp.where(spawn_here, 1.0, 0.0))
    return stk2, terminal, spawn, state


def main():
    rng = np.random.default_rng(0)
    tags = rng.integers(1, 8, size=N).astype(float)              # per-event initial tag (walk length)
    g0 = empty_batch(N, M)
    g0["alive"] = g0["alive"].at[:, 0].set(True)
    g0["p4"] = g0["p4"].at[:, 0, 0].set(jnp.asarray(tags))
    pending = dict(stack=g0, consumed0=jnp.zeros((N, A), bool),
                   npos=jnp.zeros((N, A, 3)), nmom=jnp.zeros((N, A, 4)), nisp=jnp.zeros((N, A), bool))
    KEY = jax.random.PRNGKey(0); cons0 = jnp.zeros((N, A), bool)

    # reference: no-refill lock-step
    outR, soflR, ooflR, primR = run_cascade_pool(g0, toy, KEY, cons0, M, max_steps=50, M_out=24)
    aR = np.asarray(outR["alive"]); tR = np.asarray(outR["p4"][..., 0])
    nR = aR.sum(axis=1)                                          # per-event terminal count

    # MODE A: refill n_w=N (no actual refill) -> must be BIT-EXACT
    outA, soflA, ooflA, primA = run_cascade_pool(g0, toy, KEY, cons0, M, max_steps=50, M_out=24,
                                                 pending=pending, n_w=N, per_event_cap=50)
    aA = np.asarray(outA["alive"])
    okA = np.array_equal(aR, aA) and int(soflA) == int(soflR) and int(ooflA) == int(ooflR)
    print(f"MODE A (n_w=N): alive==ref {np.array_equal(aR, aA)}  sofl {int(soflR)}=={int(soflA)}  "
          f"oofl {int(ooflR)}=={int(ooflA)}  -> {'PASS' if okA else 'FAIL'}")

    # MODE B: refill n_w=8 (real slot reuse) -> same per-event terminal COUNT by evt_id
    outB, *_ = run_cascade_pool(g0, toy, KEY, cons0, M, max_steps=50, M_out=24,
                                pending=pending, n_w=8, per_event_cap=50)
    nB = np.asarray(outB["alive"]).sum(axis=1)
    okB = np.array_equal(nR, nB)
    print(f"MODE B (n_w=8): per-event terminal counts == ref  -> {'PASS' if okB else 'FAIL'}")
    if not okB:
        bad = np.where(nR != nB)[0][:10]
        print(f"   mismatched events {bad}: ref {nR[bad]} vs refill {nB[bad]}")
    print("RESULT:", "PASS" if (okA and okB) else "FAIL")
    sys.exit(0 if (okA and okB) else 1)


if __name__ == "__main__":
    main()

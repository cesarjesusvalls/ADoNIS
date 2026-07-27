"""Pooled-engine S2: the per-step in/out reconcile (pool_reconcile) and the generic loop
(run_cascade_pool).  Validates the stack bookkeeping -- drop terminals, keep survivors, insert created
particles, count overflow, preserve identity -- independently of the (S2b) physics stepper.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp
from adonis.fsi.cascade import empty_batch, pool_reconcile, run_cascade_pool


def _batch(n, M, tags, alive):
    """A (n,M) ParticleBatch tagged by p4[...,0] = tags so identity is checkable; alive from `alive`."""
    b = empty_batch(n, M)
    b["p4"] = b["p4"].at[..., 0].set(jnp.asarray(tags, float))
    b["alive"] = jnp.asarray(alive, bool)
    return b


def test_reconcile_drop_keep_insert():
    # M=4 stack: slots tagged 10,20,30,0 ; slot3 already dead. terminal kills slot1 (tag 20).
    # spawn (K=2): tags 50,60 both valid.  survivors {10,30} + spawn {50,60} = 4 -> fits, ofl=0.
    M = 4
    stack = _batch(1, M, [[10, 20, 30, 0]], [[True, True, True, False]])
    terminal = jnp.asarray([[False, True, False, False]])
    spawn = _batch(1, 2, [[50, 60]], [[True, True]])
    new, _, ofl = pool_reconcile(stack, terminal, spawn, M)
    tags = set(np.asarray(new["p4"][0, :, 0])[np.asarray(new["alive"][0])].tolist())
    assert int(ofl) == 0
    assert tags == {10, 30, 50, 60}, tags          # dropped 20, kept 10/30, inserted 50/60


def test_reconcile_overflow():
    # M=4: 4 alive survivors (none terminal) + 2 spawn = 6 -> 2 overflow; stack stays full at 4.
    M = 4
    stack = _batch(1, M, [[10, 20, 30, 40]], [[True, True, True, True]])
    terminal = jnp.zeros((1, M), bool)
    spawn = _batch(1, 2, [[50, 60]], [[True, True]])
    new, _, ofl = pool_reconcile(stack, terminal, spawn, M)
    assert int(ofl) == 2
    assert int(new["alive"].sum()) == 4            # packed full


def test_loop_drains_and_collects_output():
    # Toy stepper: each live slot escapes when its tag reaches 0 (decremented each step); a slot with
    # tag>=3 spawns ONE child (tag 1).  Original tag 5 -> escapes after 5 steps + spawns 3 children
    # (at tag 5,4,3), each escaping shortly after -> 4 terminals collected in the output buffer.
    M = 6
    init = _batch(1, M, [[5, 0, 0, 0, 0, 0]], [[True, False, False, False, False, False]])

    def stepper(stk, key, state, step=0, dt_evt=None):   # step/dt_evt: real-stepper contract; toy ignores them
        tag = stk["p4"][..., 0]
        alive = stk["alive"]
        terminal = alive & (tag <= 0.5)                       # escape when tag hits 0
        stk2 = {**stk, "p4": stk["p4"].at[..., 0].add(jnp.where(alive, -1.0, 0.0))}
        spawn_here = alive & ~terminal & (tag >= 3.0)         # spawn while energetic
        spawn = empty_batch(1, M)
        spawn["alive"] = spawn_here
        spawn["p4"] = spawn["p4"].at[..., 0].set(jnp.where(spawn_here, 1.0, 0.0))
        return stk2, terminal, spawn, state

    out, sofl, oofl, _ = run_cascade_pool(init, stepper, jax.random.PRNGKey(0), jnp.int32(0),
                                          M, max_steps=50, M_out=24)
    assert int(out["alive"].sum()) == 4            # 1 original + 3 children collected
    assert int(sofl) == 0 and int(oofl) == 0


# NOTE: the per-step bit-exactness cross-checks (pool _nucleon_step/_pion_step vs the BFS
# _propagate_*_discrete reference) were removed with the BFS engine in the pool-unification cleanup.
# The pool per-step physics is validated via the kind-1 reweight (tests/test_pool_fsi_reweight.py)
# and end-to-end against ACHILLES in the analysis/paper high-statistics pipeline (not a unit test).


if __name__ == "__main__":
    test_reconcile_drop_keep_insert(); test_reconcile_overflow(); test_loop_drains_and_collects_output()
    print("pool reconcile + loop OK")

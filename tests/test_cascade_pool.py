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
    M = 4
    stack = _batch(1, M, [[10, 20, 30, 0]], [[True, True, True, False]])
    terminal = jnp.asarray([[False, True, False, False]])
    spawn = _batch(1, 2, [[50, 60]], [[True, True]])
    new, _, ofl = pool_reconcile(stack, terminal, spawn, M)
    tags = set(np.asarray(new["p4"][0, :, 0])[np.asarray(new["alive"][0])].tolist())
    assert int(ofl) == 0
    assert tags == {10, 30, 50, 60}, tags


def test_reconcile_overflow():
    M = 4
    stack = _batch(1, M, [[10, 20, 30, 40]], [[True, True, True, True]])
    terminal = jnp.zeros((1, M), bool)
    spawn = _batch(1, 2, [[50, 60]], [[True, True]])
    new, _, ofl = pool_reconcile(stack, terminal, spawn, M)
    assert int(ofl) == 2
    assert int(new["alive"].sum()) == 4


def test_loop_drains_and_collects_output():
    M = 6
    init = _batch(1, M, [[5, 0, 0, 0, 0, 0]], [[True, False, False, False, False, False]])

    def stepper(stk, key, state, step=0, dt_evt=None):
        tag = stk["p4"][..., 0]
        alive = stk["alive"]
        terminal = alive & (tag <= 0.5)
        stk2 = {**stk, "p4": stk["p4"].at[..., 0].add(jnp.where(alive, -1.0, 0.0))}
        spawn_here = alive & ~terminal & (tag >= 3.0)
        spawn = empty_batch(1, M)
        spawn["alive"] = spawn_here
        spawn["p4"] = spawn["p4"].at[..., 0].set(jnp.where(spawn_here, 1.0, 0.0))
        return stk2, terminal, spawn, state

    out, sofl, oofl, _ = run_cascade_pool(init, stepper, jax.random.PRNGKey(0), jnp.int32(0),
                                          M, max_steps=50, M_out=24)
    assert int(out["alive"].sum()) == 4
    assert int(sofl) == 0 and int(oofl) == 0




if __name__ == "__main__":
    test_reconcile_drop_keep_insert(); test_reconcile_overflow(); test_loop_drains_and_collects_output()
    print("pool reconcile + loop OK")

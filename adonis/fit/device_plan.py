"""How the fused kernel is sized on whatever GPU it lands on.

Every memory knob in this codebase has, at some point, been a hardcoded number that happened to work on
one card.  `S4_JAC_BATCH=16` against 17 dials; a 0.25 pool fraction tuned on an A100; a dial batch chosen
because 11 GB was what turing had.  Each one silently changed the arithmetic, the dispatch count, or the
failure mode on a different device.  This module makes the sizing an explicit, logged, overridable PLAN.

WHAT LIMITS THE KERNEL.  Two different resources, and they are easy to confuse:

  * BFC POOL (`XLA_PYTHON_CLIENT_MEM_FRACTION`, default 0.75 of the card).  Holds the resident banks and
    the program's tensors.  Wants to be LARGE.
  * EVERYTHING OUTSIDE THE POOL -- the compiled CUBIN, CUDA graphs, cuBLAS/cuDNN workspaces, and the
    compiler's own scratch.  Wants the pool to be SMALL, because preallocation takes that memory away.

Measured on turing (11 GB) at 250k events/sample: an 8.07 GB pool failed loading the chi2 program; 4.84
reached residuals; 2.69 reached the jacobian.  One stage further per halving -- the classic signature of
a resource that lives outside the pool.  So the default here is a pool just large enough for the data,
not the stock 75%.

WHAT SETS THE CEILING.  A vmapped JVP carries an (n_dials x n_events) tangent through every intermediate,
so the working set scales as the PRODUCT.  Calibration points, all measured:

    device      events/sample   dials   result
    turing 11GB     20,000        17    ok
    turing 11GB     60,000        17    OOM
    ampere 40GB    125,000        17    ok
    ampere 40GB    250,000        17    OOM  (even at dial batch 1)

which puts the ok/fail boundary near 50,000 dial-events per GB of device memory.  `BUDGET_PER_GB` is set
below that, and everything else follows from it.

THE THREE DIALS, in the order they should be reached for:

  event_chunk  split the EVENT axis (see BinSpec.chunks).  The model is a sum over events, so this is
               exact and costs NO extra passes -- only per-chunk dispatch overhead.  The right answer.
  jac_batch    split the DIAL axis.  Also exact, but each extra dispatch costs a full extra PRIMAL pass,
               so it makes Gauss-Newton look more expensive for a reason that is memory, not algorithm.
               Use only when chunking is unavailable.
  mem_fraction move the pool/outside-pool split.  Changes no arithmetic at all.

Config (all optional; anything omitted is auto-sized and logged):

    compute:
      mem_fraction: 0.25      # XLA_PYTHON_CLIENT_MEM_FRACTION; must be set before jax is imported
      event_chunk:  200000    # 0 = no chunking
      jac_batch:    0         # 0 = all dials in one dispatch
      hvp_batch:    4
      budget_per_gb: 40000    # dial-events per GB before chunking kicks in
"""
from __future__ import annotations

import os

# Below the ~50k dial-events/GB where the measurements above cross from ok to OOM.
BUDGET_PER_GB = 40_000
HVP_BATCH_DEFAULT = 4          # the hessian program is built ON TOP of residuals+jacobian; keep it small
POOL_HEADROOM = 2.5            # pool = this x the resident bank bytes, so the rest is free for CUBINs


def _cfg(cfg):
    c = getattr(cfg, "compute", None)
    return c if isinstance(c, dict) else {}


def apply_env(cfg, log=print):
    """Set the XLA pool fraction.  MUST be called before jax is imported anywhere in the process.

    Returns the fraction applied, or None if left to XLA's default.  An explicit value in the config
    always wins; otherwise the environment is left alone here and sized later by `resolve`, which can
    only advise because by then jax is already up.
    """
    f = _cfg(cfg).get("mem_fraction")
    if f is None:
        return None
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(float(f))
    log(f"[compute] XLA_PYTHON_CLIENT_MEM_FRACTION={float(f):g} (from config)")
    return float(f)


def device_gb():
    """Total device memory in GB, or None on CPU."""
    try:
        import jax
        d = jax.local_devices()[0]
        if d.platform != "gpu":
            return None
        m = d.memory_stats()
        # bytes_limit is the POOL, not the card; divide by the fraction actually in force to recover it
        frac = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", 0.75))
        return float(m["bytes_limit"]) / max(frac, 1e-6) / 2 ** 30
    except Exception:                                        # noqa: BLE001
        return None


def resolve(cfg, n_dials, n_events_max, log=print):
    """Return the memory plan {event_chunk, jac_batch, hvp_batch} for this device and problem.

    Explicit config values are honoured verbatim -- this never silently overrides a stated choice.  What
    is auto-sized is auto-sized from the measured budget, and the whole plan is LOGGED, because the
    single most expensive lesson in this codebase is that a number which changes the dispatch count must
    never be invisible in the output.
    """
    c = _cfg(cfg)
    gb = device_gb()
    plan = dict(event_chunk=int(c.get("event_chunk", 0) or 0),
                jac_batch=int(c.get("jac_batch", 0) or 0),
                hvp_batch=int(c.get("hvp_batch", HVP_BATCH_DEFAULT)),
                device_gb=gb, auto=False)

    if plan["event_chunk"] or gb is None:
        log(f"[compute] plan {plan} (explicit or non-GPU)")
        return plan

    budget = int(c.get("budget_per_gb", BUDGET_PER_GB)) * gb        # dial-events the device can hold
    need = n_dials * n_events_max
    if need <= budget:
        log(f"[compute] {gb:.0f} GB device, {n_dials} dials x {n_events_max:,} events "
            f"= {need/1e6:.2f}M dial-events, budget {budget/1e6:.2f}M -> no chunking")
        return plan

    chunk = max(1024, int(budget // max(n_dials, 1)))
    plan["event_chunk"], plan["auto"] = chunk, True
    log(f"[compute] {gb:.0f} GB device, {n_dials} dials x {n_events_max:,} events "
        f"= {need/1e6:.2f}M dial-events EXCEEDS budget {budget/1e6:.2f}M "
        f"-> event_chunk={chunk:,} ({-(-n_events_max // chunk)} windows/sample). "
        f"Chunking the EVENT axis is exact and costs no extra passes; the dial axis is not touched.")
    return plan

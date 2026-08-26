"""Sizes the fused kernel's memory footprint (event chunk, dial batch, XLA pool fraction) for
whatever GPU it lands on.

Explicit config values are honoured verbatim; anything else is auto-sized from a measured per-GB
budget and logged, since a change to dispatch count must be visible.

Three knobs: `event_chunk` splits the event axis (exact, no extra passes); `jac_batch` splits the
dial axis (exact, but each batch costs a full extra primal pass); `mem_fraction` moves the
pool/outside-pool split and changes no arithmetic.

Config (all optional):

    compute:
      mem_fraction: 0.25      # XLA_PYTHON_CLIENT_MEM_FRACTION; must be set before jax is imported
      event_chunk:  200000    # 0 = no chunking
      jac_batch:    0         # 0 = all dials in one dispatch
      hvp_batch:    4
      budget_per_gb: 40000    # dial-events per GB before chunking kicks in
"""
from __future__ import annotations

import os

BUDGET_PER_GB = 40_000
HVP_BATCH_DEFAULT = 4
POOL_HEADROOM = 2.5


def _cfg(cfg):
    c = getattr(cfg, "compute", None)
    return c if isinstance(c, dict) else {}


def apply_env(cfg, log=print):
    """Set the XLA pool fraction.  Must be called before jax is imported anywhere in the process.

    Returns the fraction applied, or None if left to XLA's default.  An explicit config value always
    wins; otherwise the environment is left untouched here for `resolve` to advise on later.
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
        frac = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", 0.75))
        return float(m["bytes_limit"]) / max(frac, 1e-6) / 2 ** 30
    except Exception:
        return None


def resolve(cfg, n_dials, n_events_max, log=print):
    """Return the memory plan {event_chunk, jac_batch, hvp_batch} for this device and problem.

    Explicit config values are honoured verbatim; anything auto-sized is logged, since a change to
    dispatch count must never be invisible in the output.
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

    budget = int(c.get("budget_per_gb", BUDGET_PER_GB)) * gb
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

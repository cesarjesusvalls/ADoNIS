# FSI reweight-record cap: nucleus-scaling is a PATCH (computational tech debt)

**Status:** works, physics-exact. To be fixed for COMPUTATIONAL reasons only — NOT a fidelity issue.

## What it is
The differentiable event/beam banks store a per-cascade-step FSI reweight record. Because the cascade
runs inside a **jitted `jax.lax.while_loop`/scan** (static shapes required), the in-loop log is a
**fixed-width dense `(n_events, K)` buffer**, `K = rec_caps`. After the run, `compact_fsi_record`
(numpy) drops the padding into the stored **ragged** layout (flat slot arrays + per-slot event index).

`K` must bound the **maximum** per-event step count. That max scales with the nucleus (bigger nucleus =
longer cascades), so we scale the cap: `_sc = ceil(A/12)`, `rec_caps = (96*_sc, 256*_sc)` (¹²C→(96,256),
⁴⁰Ar→(384,1024)). Mirrors `analysis/beams/beam_bank.py` (which learned this after losing 112 Ar shards).

## Why it's a patch (the computational waste)
The per-event step-count distribution is a **long tail**: the vast majority of events use *far* fewer
slots than `K` (the dense record is ~97% padding for a neutrino event; pion occupancy ~2.3/96, nucleon
~10.6/64 for ¹²C). Sizing `K` for the worst-case tail means the **transient dense buffer over-allocates
massively** — e.g. Ar at CHUNK=10k allocates `10k × 1024` nucleon slots but the *average* event uses
~tens. This wastes:
- **transient RAM** during generation (grows as `CHUNK × K × n_fields`), and
- some **compute** touching the padding.

The STORED bank is unaffected (ragged/compact), and the reweight physics is exact regardless of `K`. So
this is purely a generation-time memory/compute inefficiency, not a correctness problem.

## Why it exists (and why it's NOT fundamental)
The record is **write-only OUTPUT** — the cascade physics never reads it back. It is frozen only because
it lives in the **`while_loop` carry** (loop state), and XLA requires every carry array to keep a
**constant shape across iterations**. So `K` is static purely as an artifact of where the log lives, not
because the computation needs it. A fixed `K` (+ the loud-overflow guard in `compact_fsi_record`) is the
current trade.

You *can* stream the log out on the fly (`jax.experimental.io_callback`) instead of carrying it — but a
callback fires **inside** the multi-thousand-step hot loop, forcing a **device→host sync every step**,
which serializes the vectorized cascade and is orders of magnitude slower. So streaming is possible but
the slowest option; the two-pass/spill fixes below keep on-device speed while removing the padding.

## Proper fixes (options, unimplemented)
1. **Auto-size `K`**: a cheap pre-pass (or the first chunk) measures the true max occupancy, then allocate
   exactly that (+small margin) instead of a nucleus-scaled guess. Cheapest to do.
2. **Small `K` + graceful overflow spill**: keep a tight `K` for the common case; the rare high-occupancy
   events spill to a slow numpy fallback path (no shard loss, no over-allocation).
3. **Segment/compress the log**: store one row per cascade *segment* (already exists as the `log_cap`
   segment-logger, `cascade_full.py`) rather than per candidate step — far fewer slots.
4. **Move the per-step logging out of the jitted region** (host callback / two-pass) — removes the
   static-shape constraint entirely, at a speed cost.

Guard that catches undersizing loudly (already in place): `adonis/fsi/cascade_full.py:compact_fsi_record`
raises with the exact `K` needed.

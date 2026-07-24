# FSI reweight-record cap: nucleus-scaling is a PATCH (computational tech debt)

**Status:** RESOLVED for `event_bank` (flat streaming record + auto-sized cap, default on; see "Resolution"
below). The dense per-event `(n,K)` buffer + nucleus-scaled guess below is retained only as the fallback
(`ADONIS_FLAT_FSI=0`) and for other callers (`beam_bank`/`tune`) not yet migrated. History kept for context.

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

## Resolution (implemented in `event_bank`, default on)
The flat/streaming record (single flat `(TOTAL,)` buffer per species + global cursor, sized by the
tail-insensitive TOTAL interaction count) replaces the per-event `(n,K)` dense buffer. `TOTAL` is
**auto-sized**, not guessed:
- A **100-event calibration pass** runs the QE and RES cascades with a throwaway `(8,8)` buffer purely as a
  COUNTER — the flat cursor `gc` advances by the true interaction count regardless of buffer size
  (`cascade_full.py:456`), so the tiny buffer reads the exact per-event rate. Budget = `rate * CHUNK * 1.5`.
- **Symmetric `max` cap**: both the pion and nucleon buffers are sized to `max(pion, nucleon)` of the
  calibrated estimate. Reason: the pion slot-count is **zero-inflated / unmeasurable at 100 events** — pions
  come only from rare threshold `NN->NNpi` (`cascade_full.py:300`), each then logging a BURST of ~20 in-slab
  slots, so a 100-event sample sees 0 ~2/3 of the time. We never size the pion buffer from its own count; the
  reliable, tail-insensitive nucleon count covers it (pion slots <= nucleon slots always; `max` self-corrects
  if a config inverts that). Knobs: `ADONIS_REC_MARGIN` (1.5), `ADONIS_REC_NCAL` (100), `ADONIS_REC_CAPS`
  (explicit override). The loud overflow guard remains as the backstop.

## Proper fixes (options — #1 now done for event_bank; others still open for beam_bank/tune)
1. **Auto-size `K`** — DONE for `event_bank` (see Resolution above): a 100-event pre-pass measures the true
   rate, then allocate `rate * CHUNK * margin` instead of a nucleus-scaled guess.
2. **Small `K` + graceful overflow spill**: keep a tight `K` for the common case; the rare high-occupancy
   events spill to a slow numpy fallback path (no shard loss, no over-allocation).
3. **Segment/compress the log**: store one row per cascade *segment* (already exists as the `log_cap`
   segment-logger, `cascade_full.py`) rather than per candidate step — far fewer slots.
4. **Move the per-step logging out of the jitted region** (host callback / two-pass) — removes the
   static-shape constraint entirely, at a speed cost.

Guard that catches undersizing loudly (already in place): `adonis/fsi/cascade_full.py:compact_fsi_record`
raises with the exact `K` needed.

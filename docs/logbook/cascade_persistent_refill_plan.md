# Persistent-refill cascade engine — implementation plan

Supersede the cascade driver IN PLACE (no `_v2`/versioned names) with a persistent-working-set engine:
- **(a) particle waiting-queue** — overflow particles (>P alive at once) go to a fixed FIFO buffer `Q`
  (by creation order) and drain into freed slots instead of being dropped.  Decouples per-step compute
  (`P`) from concurrent capacity (`P+Q`).
- **(b) per-event refill + per-event step caps** — the `(n,P)` working set holds `n` events; when an
  event's particles are all done (or it hits its OWN step cap) its slot is flushed to per-event output
  buffers and refilled from a pending-event pool.  Removes the global `max_steps` lock-step.
- **(c) multi-worker** — shard the working set across devices/replicas; a shared cursor (or host
  scheduler) feeds pending events to whichever worker has a free slot.

**Hard constraints**
- Preserve ALL functionality: forward rich schema (`pterm`, `nterms`, `created`), the kind-1 FSI reweight
  records (`rec`/`rofl`), the segment logger (`log`/`wptr`/`logofl`), the RES primary-pion fate latch
  (`prim`).  All consumers (cascade_carbon/cascade_nucleus/run_fsi/generate.py/gen_cascade_segments) keep
  working with the same return contracts.
- **Modify NO physics.**  The ONLY allowed behavioural change: particles that the current code silently
  drops on overflow are now kept (a correctness improvement).  Everything else bit-identical.
- No versioning names.  Edit `run_cascade_pool` / `make_pool_stepper` / `pool_reconcile` / `_cascade_pool`
  in place.

## Current architecture (must be preserved)
- `make_pool_stepper(su, cfg, with_rec, with_seg)` closes over the per-event background `su` =
  {npos,nmom,nisp (n,A), pos0, consumed0, ch0}.  `stepper(stack,key,consumed[,step]) -> stack2,
  terminal(n,M), spawn(n,3M), consumed2 [, rec][, seg]`.
- `run_cascade_pool` `lax.while_loop` over steps; carry = (i, stack(n,M), consumed, out(n,M_out), sofl,
  oofl, prim(n), rb(rec), rofl, log(n,L), wptr(n), logofl).  Per step: stepper -> accumulate out
  (compact to M_out), prim latch, rec (`_rec_scatter`), seg (scatter-append), then `pool_reconcile`
  (compact survivors+spawn to M, sofl).  Cond: `i<max_steps & any(alive)`.
- `_cascade_pool` builds g0 (QE: 1 nucleon; RES: pion+recoil, origin-tagged), calls run_cascade_pool,
  maps `out` -> rich schema.

**The blocker for refill:** the background is CLOSED OVER (one fixed `su` for all `n`).  To refill an
event slot with a NEW event we must make the background (npos,nmom,nisp,consumed) WORKING-SET STATE that
the stepper takes as an argument and that refill swaps.

## Stages (each ends with a bit-exact gate + commit)

### Stage 0 — golden reference (no code change)
Capture, from the CURRENT engine on a fixed C seed (small N, both QE & RES), the full outputs:
`pterm,nterms,created` + `rec` (with_rec) + `log,wptr,logofl` (with_seg) + `sofl,oofl,prim`.  Save to
`/tmp/refeng_C.npz`.  This is the bit-exactness oracle for the whole rewrite.  Script:
`scripts/_engine_golden.py` (kept for re-use).

### Stage 1 — background as an explicit stepper argument (refactor, bit-exact)
`make_pool_stepper(cfg, with_rec, with_seg)` no longer closes over `su`; `stepper(stack, bg, consumed,
key[, step])` where `bg=(npos,nmom,nisp)`.  `run_cascade_pool` carries `bg` (constant for now) and passes
it.  Update `_cascade_pool`/callers.  **Gate:** output == `/tmp/refeng_C.npz` bit-for-bit.

### Stage 2 — particle waiting queue Q (FIFO by creation), bit-exact at Q=0
`pool_reconcile(stack, terminal, spawn, wait, M, Q)` -> (new_stack(n,M), new_wait(n,Q), sofl).  Logic:
combined = survivors ++ wait ++ spawn, ordered by creation (track_id ascending, stable); first M ->
active, next Q -> wait, rest -> sofl.  (Needs track_id on every particle incl. spawns — already added for
the logger; make it unconditional & monotonic.)  `run_cascade_pool` carries `wait`; cond also requires
`any(wait alive)` empty.  **Gates:** (i) Q=0 -> == golden (drops identical); (ii) Q large -> identical
EXCEPT the ~0.01-0.5% overflow events now keep their particles (sofl drops, more terminals); verify
those events differ ONLY by the kept particles (no other change).

### Stage 3 — persistent per-event refill + per-event step caps (the core)
Working set = `n_w` event-slots.  State adds: per-slot `bg`, `consumed`, `evt_id`, `nstep` (per-event
counter), and a PENDING pool (host arrays of all N_total events' g0+bg) + `cursor`.  Per-event output
buffers become global `(N_total, …)` indexed by `evt_id`.
Loop (`while cursor<N_total OR any slot active`): step all slots; accumulate per-slot out/rec/seg/prim;
`nstep++`; a slot is DONE if (no alive particles & empty wait) OR `nstep==per_event_cap` -> FLUSH its
out/rec/seg/prim/sofl/oofl into the global buffers at `evt_id`, then REFILL from `cursor` (load next
event's g0+bg, reset nstep/consumed/accumulators) or mark slot idle if pending exhausted.
`per_event_cap` replaces the global `max_steps`.
**Gate:** set `n_w = N_total`, `per_event_cap = max_steps`, refill OFF (cursor pre-exhausted) -> ==
golden.  Then `n_w < N_total` with refill ON -> SAME per-event outputs (compare by evt_id to golden);
only the wall-clock/occupancy changes, not the physics.

### Stage 4 — multi-worker sharding
`shard_map`/`pmap` the Stage-3 loop across W workers, each an `(n_w,P)` set, sharing the pending pool via
a partitioned cursor (worker w takes events `w::W`, or an atomic host cursor).  Per-event outputs gather
to `(N_total,…)`.  **Gate:** W=1 == Stage 3; W>1 == W=1 (same per-event outputs, faster).

### Validation (final)
Run 1 seed (C) through the new engine via `gen_cascade_segments` + `cascade_vertex_matrix` vs the
existing ACHILLES segment dump; confirm the matrix is unchanged (≤ the Stage-2 kept-particle delta) — i.e.
no physics moved.  Also: `cc0pi_tune` smoke (reweight path) + a forward `adonis_generate` smoke.

## Risks / notes
- Differentiability: the reweight `rec` flush-on-completion into `(N_total,…)` must stay a pure scatter
  (no data-dependent shape) -> keep `evt_id` scatter with `mode='drop'`, like the segment logger.
- Determinism: refill strictly by `cursor` order; per-event RNG keyed by `evt_id` (not loop step) so a
  refilled event is reproducible regardless of which slot/worker runs it.
- `prim` latch, formation-zone clock, consumed: all become per-slot, reset on refill.
- Memory: pending pool holds all N_total events' g0+bg -> chunk N_total if large (driver loops chunks,
  each chunk a persistent run).
- Stage 1+2 are prerequisites and low-risk; Stage 3 is the bulk; Stage 4 is the scale-out layer.

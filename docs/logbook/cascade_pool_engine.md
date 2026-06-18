# Cascade pooled engine — single per-step-reconciled propagation loop

## Goal
Replace the generation-synchronized BFS (`cascade_full.run_cascade`) with the ORIGINAL design intent:
a single fixed-size particle stack stepped once per step, with particles moved in/out by status at the
end of each step (ACHILLES's evolving-list, vectorized over events). Same physics goal, much faster,
and a more faithful (true step-order) consumption of the background.

## Why (the BFS waste)
Current `run_cascade`: `for g in range(max_gen): kernel(cur)  # full max_steps loop on P slots`. So:
- `max_gen` separate full-`max_steps` passes; reconciliation (`compact`) only BETWEEN generations.
- A recoil created at step k is held (`best_ko`) and only propagates in generation g+1 (creation latency).
- Later generations' P-buffers are almost empty but fully processed.

Cost ≈ `max_gen × P × n × max_steps × A`.

## Measurement (Ar QE, P=16/max_gen=5, nothing truncated; scripts ad hoc)
particles/event across all generations: **mean 1.84, median 2, 90%=3, 99%=5, max 10**, overflow 0.
per-generation mean alive: gen0 1.00, gen1 0.66, gen2 0.158, gen3 0.021, gen4 0.002 (~4x falloff/gen).
max alive in any single gen: 90%=2, 99%=3, max 4.

Implications:
- **Concurrent occupancy ≤ total ≤ 10** (99% ≤ 5) -> the existing **P=12 covers the whole concurrent
  stack** (P maps to the pool size M; no need to grow it; overflow counted as today).
- **Dead-slot waste ~10–24×**: unified cost ≈ `M × n × max_steps × A` -> ratio `max_gen·P/M` ≈
  (5·6)/3 ≈ 10x (dev), (6·12)/3 ≈ 24x (production). gen-2/3/4 buffers are 97–99.8% empty.

## Design
State: an `(n, M)` stack; per slot `{p4, pos, species(0=π,1=N), charge, fz, alive}` + per-slot kind-1
record accumulators. Init = gen-0 particles (QE: the struck->p; RES: recoil N + pion + π-knockouts).

Loop `while jnp.any(alive)` (global `max_steps` cap):
1. **Propagate one step**, per slot dispatched by species to the existing per-step bodies
   (`_propagate_discrete` / `_propagate_nucleon_discrete` step logic), vectorized to `(n, M)`.
2. **Reconcile (in/out)**: free slots that escaped / were captured / absorbed / converted; scatter the
   particles created this step (recoils, NN-inelastic 2nd nucleon, created pions) into freed slots;
   count overflow (the `compact` primitive, moved inside the step body).
3. Accumulate per-slot kind-1 records.

## Subtleties / risks (must handle)
1. **RNG positionality** — the bodies use `jax.random.uniform(key, (n, A))` (positional). Any
   event/particle reordering or compaction shifts the stream. The pool must key per particle via
   `fold_in(base_key, persistent_particle_id)` so randomness is independent of slot position.
2. **Shared consumed mask races** — M stack particles can strike the same background nucleon in one
   step. Reuse the intra-generation slot-serialized depletion (a nucleon consumed by an earlier slot is
   unavailable to later ones in the same step).
3. **Mixed-species stack** — per-slot species dispatch (π vs N bodies), as the v1 `make_kernel` did.
4. **kind-1 records** (`sigma_scatter` / branch reweight -> the differentiable weights) must accumulate
   per slot through the unified loop; re-pin the reweight closures.
5. **Not bit-identical to the BFS** — true step-order consumption differs from generation order. This is
   a faithfulness IMPROVEMENT; validate against ACHILLES (transparency / single-pass fate / multiplicity
   / topology), NOT against old BFS banks.

## Staged plan (each a clean commit, behind `DiscreteCascadeConfig.engine = "bfs" | "pool"`; default
## stays "bfs" until validated)
- S1: config switch + pooled `run_cascade_pool` skeleton (data structures, the loop shell, overflow
  accounting) wired so `engine="pool"` is selectable but a no-op-ish stub.
- S2: per-step body extraction/vectorization to `(n, M)` with per-particle keys + slot-serialized
  consumed depletion (nucleon first, then pion).  **DONE (loop+reconcile), partial.**
  - S2 delivered the generic loop + in/out reconcile (`run_cascade_pool` / `pool_reconcile` =
    `compact(concat(survivors, spawned), M)`), stepper-pluggable, unit-tested. bfs bit-exact.
  - **S2b finding — re-implementation, NOT extraction:** the validated `_propagate_nucleon_discrete`
    body is built on the BFS knockout model (`best_ko` top-K slots; recoils REGISTERED and deferred to
    the next generation; the leading continues in place).  The pool's model is opposite — a recoil
    created at a step is an IMMEDIATE new stack particle (a `spawn`), no `best_ko`.  So the pool stepper
    must RE-EXPRESS the per-step orchestration in spawn-per-step form, reusing the physics PRIMITIVES
    (escape/recapture, formation zone, in-slab geometry, elastic scatter + per-species Pauli k_F, the
    NN->NDelta->NN'pi inelastic branch + channel charges) rather than copy the body.  This is the large
    focused block; do it incrementally (nucleon step -> validate vs a single-particle bfs segment ->
    pion step -> mixed dispatch).
- S3: reconcile (drop terminal, insert created, overflow) inside the step loop.  **DONE in S2** (it IS
  `pool_reconcile`).  Remaining: per-step TERMINAL/output collection (escaped particles = the final
  state) into a fixed output buffer (scatter each step).
- S4: kind-1 record accumulation + reweight closures.
- S5: validation vs ACHILLES (transparency, fate, multiplicity, topology) + speedup measurement.
- S6: flip default to "pool" once green; keep "bfs" available.

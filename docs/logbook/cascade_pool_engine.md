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
  - **S2b-1 DONE** (commit 1ce28bc): `_nucleon_step` per-step physics, spawn-emitting; bit-exact vs
    bfs leading trajectory (`test_nucleon_step_matches_bfs_segment`, max|Δp4|=6e-12, nsc exact).
  - **S2b-2 DONE** (commit 96acde4): `run_cascade_pool` state-threading + terminal/output collection.
  - **S2b-3 DONE** (this commit): `make_pool_stepper` ((n,M) slot-serial scan, per-slot keys, consumed
    threaded) + output-collection fix (collect `term_batch alive = terminal` directly; the stepper
    already sets escaped slots `alive=False`, so the old `alive & terminal` mask dropped every escape ->
    0 protons).  **Validated**: pool reproduces the bfs nucleon-elastic avalanche multiplicity
    (`scripts/cascade_pool_validate.py 0`, Ar QE, 7537 ev, M=12, ms683, nn_inelastic=False):
    BFS proton(>250)/ev=1.2716, POOL=1.2721, **ratio 1.0004**, stack/out overflow 0.  Pool is not
    bit-identical (true step-order consumption + slot-positional RNG) so ~1.0 is the expected agreement.
  - **S2b-pion DONE** (this commit): `_pion_step` = `_propagate_discrete.body` (algo="step") re-expressed
    spawn-per-step (absorption piNN->NN up to 2 protons, scatter recoil, eta-N' conversion baryon emitted
    as IMMEDIATE nucleon spawns; scattered pion continues in place, charge oscillates; abs/conv pion
    removed).  Bit-exact vs bfs leading-pion trajectory (`test_pion_step_matches_bfs_segment`: p_pi, ch,
    nsc, absorbed, conv, pos all exact).  `make_pool_stepper` now does MIXED dispatch: both bodies run on
    every slot, selected by species (the v1 2x-eval tradeoff; the dominant dead-slot 10-24x waste is gone).
    Each slot emits up to 2 NUCLEON spawns + 1 PION spawn (nucleon slot -> 1 knockout N + 1 NN-created pi;
    pion slot -> up to 2 abs/recoil N).  Stack gains an `nsc` field (pion beam-vs-internal escape).
  - **Created-pion divergence from bfs (expected, subtlety #5)**: bfs re-cascades only the SINGLE leading
    created pion per event, from the QE vertex with a FRESH consumed mask (`cascade_carbon_v2` re-entry).
    The pool propagates EVERY created pion from its TRUE creation point with the running consumed mask ->
    MORE faithful, NOT bit-identical.  Created pions start `nsc=0` (beam/plane escape) matching the bfs
    `pion_segment` re-entry convention.  Validate inelastic vs ACHILLES, not vs bfs.
  - **S2b multiplicity validation** (`scripts/cascade_pool_validate.py`, Ar QE, 7537 ev, ms683):
    - elastic (nn_inel=False, M=12): BFS proton(>250)/ev=1.2716, POOL=1.2707, **ratio 0.9993**, ovf 0.
    - inelastic (nn_inel=True, M=16): BFS=1.2850, POOL=1.3099, **ratio 1.0194** (+1.9%), stack_ofl=2.
      The +1.9% is the created-pion faithfulness divergence above (pool propagates ALL created pions);
      direction-vs-ACHILLES is an OPEN S5 question.  M=16 already overflows twice -> re-measure
      concurrent occupancy for production sizing (the old P=12 measurement predated all-created-pion
      propagation).  carbon bit-exact BFS suite (10 tests) still green after the nsc/compact additions.
  - **RNG note (for S4)**: slot keys are `split(key,M)[m]` (slot-positional).  After a compaction a
    particle changes slot -> its RNG stream shifts.  Valid randomness, but not reproducible-per-particle;
    S4 should re-key via `fold_in(base, persistent_track_id)` (subtlety #1) for determinism.
- S3: reconcile (drop terminal, insert created, overflow) inside the step loop.  **DONE in S2** (it IS
  `pool_reconcile`).  Per-step TERMINAL/output collection **DONE in S2b-2/3** (escaped particles
  scattered into the fixed (n,M_out) output buffer each step).
- **S2b-integrate (QE) DONE** (this commit): `cascade_carbon_v2(engine="pool", channel="qe")` runs
  make_pool_stepper + run_cascade_pool and maps the flat (n,M_out) terminal buffer back to the rich
  schema -- nterms[0] = the NUCLEON terminals (pid from charge), `created` = the leading surviving pion
  (CC0pi veto pion), `pterm` = QE "none" sentinel.  End-to-end QE CC0pi topology vs BFS (Ar, 7537 ev,
  P=16, ms683, weighted; `scripts/cascade_pool_topology.py`):
    <Np> bfs 1.2370 / pool 1.2435 (+0.5%); 0p .0480->.0460, 1p .7168->.7221, 2p+ .2353->.2320 -- ALL
    within ~1 sigma; pi_surv .0056->.0044 (pool propagates created pions further -> more absorbed).
  RES pool (primary pion as a gen-0 PION stack slot + pterm/created schema split) is the next step.
  NOTE: the generate.py driver does not yet thread `engine` through GenConfig -> a bank generation still
  uses bfs; engine is selectable only at the cascade_carbon_v2 call (S2b-integrate-driver follow-up).
- **Speedup (the payoff)** `scripts/cascade_pool_timing.py` (Ar QE, 3775 ev, P=16, ms683,
  nn_inelastic, post-JIT run-only): **bfs 94.53 ms/ev -> pool 33.76 ms/ev = 2.8x faster.**  Confirms
  the dead-slot elimination beats the dual-body 2x-eval cost.
- **S5 vs ACHILLES (QE CC0pi Ar) DONE.**  Generated a 5-seed pool QE Ar bank
  (`configs/gen_ar_pool_qe.yaml` -> `t2k_cc0pi_engine_rich_arpool.npz`, M=16, ms683) and ran the CC0pi
  analysis vs the ACHILLES Ar reference (`configs/ana_cc0pi_arpool_nopcut.yaml`, pool QE + BFS
  RES-absorbed; only the QE engine differs from `ana_cc0pi_ar_nopcut`).
    sigma ACH/ADO: **BFS 1.020 -> POOL 1.014** (toward 1.000).  chi2/ndf (POOL vs BFS):
    dpt 1.15/2.15, dalphat 1.86/2.76, Q2 1.54/1.51, W 2.11/2.52, p_mu 1.98/1.71, cos_mu 0.96/1.45,
    lp_p 2.41/2.85 -> POOL lower on 5/7 (Q2 flat, p_mu up).  Figure cc0pi_ENGINE_argon_POOL_nopcut.png.
  - **CAVEAT (buffer confound, honest):** the committed BFS `_ar` bank is the DEV config P=6/max_gen=3
    (gen_ar_ms600), the pool is M=16.  A live event can need up to 16 concurrent particles
    (overflow-check below), which P=6x3 truncates -> the BFS `_ar` loses high-multiplicity protons.
    So the 1.020->1.014 gain is the pool's wide single-stack CAPACITY (+ all-created-pion faithfulness),
    NOT the engine algorithm.  At EQUAL buffers (P=16 both) the engines agree within ~1 sigma
    (the cascade_pool_topology test).  The pool's value: it makes high-capacity cascades AFFORDABLE
    (2.8x faster), so production can run M=16 cheaply.
  - **Overflow is on DEAD (w=0) events only** (`scripts/cascade_pool_overflow_check.py`, 30000 ev, 94%
    live): <nterm>_live = 1.986 IDENTICAL at M=16 and M=32 (max live 16 both); only dead-event avalanches
    grow (2.645->3.083, max 21->36).  out_ofl=0 at M_out=24.  So the ~556/30k stack overflow per seed is
    zero-weight runaway events; the bank's live physics is unbiased and **M=16 is sufficient**.
- S4: kind-1 record accumulation + reweight closures (differentiability through the pool).
- S6: flip default to "pool" once RES integrated + S4 done; keep "bfs" available.
- **RES pool integration** (CC1pi): primary pion as a gen-0 PION stack slot + the pterm/created schema
  split -- the remaining channel.

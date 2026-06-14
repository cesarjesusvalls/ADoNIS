# Logbook: faithful multi-particle differentiable cascade engine

## Goal

Replace the factorized, leading-only FSI surrogate with a cascade that tracks **every** particle of the
cascade tree faithfully (one shared nucleus, all secondaries co-propagated), while staying
**differentiable** (kind-1: frozen walk + per-particle reweights) and **JIT/vmap-able** (fixed shapes).
This closes the cross-cascade 2nd-order effects the current chain misses (logbook cc1pi_t2k #30-33):
- nucleon-FSI-created pions rescuing an event whose primary pion died (currently veto-only),
- the in-window-proton yield deficit (ACHILLES co-evolves all kicked nucleons; we track ~5 leading),
- any "one cascade's product becomes the other's signal object" path (factorization breaks these).

Differentiability is NOT the blocker (agreed): freeze the walk (sample every discrete choice at
nominal), each particle is a node carrying weight = product of theta-dependent step likelihood ratios
along its Markov path x parent spawn-probability; observables mask nodes/events and sum weights ->
gradient flows through the weights, not the frozen branching. This is the existing kind-1 trick
(pion_branch_reweight / nucleon_scat_reweight) extended to the full spawning tree. The real problem is
purely #2: a faithful cascade is variable particle-count + variable depth, vs JAX fixed shapes.

Prior art: the shared-state event cascade (cc1pi_t2k #10-16) was built and REVERTED -- it was
nominal-forward only (no per-branch reweight records -> non-differentiable) and slot-capped/non-faithful.
This engine differs: (a) kind-1 per-particle reweight records from the start, (b) BFS-generational reusing
the validated single-particle kernels, (c) standalone + validated against the rich banks before any wiring.

## Design (BFS generations over a shared nucleus; refines the "N-pass / stack" idea)

- **One nucleus per event** (shared): positions (n,A,3), isospin (n,A), and a SHARED consumed mask (n,A)
  updated as particles deplete the Fermi sea (Pauli / no double-use).
- **Particle = one Markov walk segment with fixed identity**: it propagates multi-step (elastic scatters
  keep identity) until a terminal (escape / absorption) or a TRANSMUTING interaction (charge-exchange,
  inelastic production) -> it ends and emits its outgoing products. Knockout nucleons are emitted as
  secondaries whenever an interaction recoils a nucleon. (So pi+ -> pi0 -> pi+ becomes gen1 pi+, gen2 pi0,
  gen3 pi+ -- exactly the user's reframe; physically equivalent since propagation is Markovian.)
- **Unified species-tagged single-particle kernel**: one kernel propagates a {pion|nucleon} through the
  shared nucleus and returns (terminal fate, terminal 4-vec/pos, ALL spawned secondaries + their
  production vertex/momentum/formation-zone, kind-1 walk record). Built from the existing
  _propagate_discrete (pion) and _propagate_nucleon_discrete (nucleon) step logic, species-branched with
  jnp.where; the NEW part is emitting *all* secondaries (not just the leading) and the spawn records.
- **BFS generation loop**: fixed-capacity buffer (n, P, state) per generation; state = {species, p4, pos,
  fz, alive, parent_weight_idx}. Gen 0 = pre-FSI interaction products (pion + recoil nucleon). Each
  generation: vmap the kernel over P slots -> record terminals (for observables) + scatter secondaries
  into the gen N+1 buffer. lax.scan / unroll over MAX_GEN. Empty slots masked (jnp.where no-op, weight 0).
- **Differentiability**: each particle records walk sufficient statistics; reweight(theta) = parent
  weight x spawn prob x prod(step likelihood ratios), a pure function of records. Forward at nominal.
- **No silent caps**: P (max particles/gen) and MAX_GEN are fixed; overflow (dropped particles) is COUNTED
  and log()'d, never silently truncated.

## Compute (the spline is the cost)

The DCC/MB spline + xsec eval per step dominates. v1: padded per-gen buffers -> empty slots still eval
(masked) = waste ~ empties/P. Mitigate by sizing P to the real per-gen width (cascades are narrow). v2
(optimization, deferred): compacted worklist/stack -- gather live particles, refill free slots with new
spawns to maximize lane utilization; investigate whether dead lanes can skip the spline (likely not
per-lane in XLA, but compaction shrinks empties). Decide v1-vs-v2 after measuring the v1 waste.

## Open questions / risks (validate, don't assume)

1. **Consumption ordering**: ACHILLES co-evolves in TIME with a shared consumed set; BFS is generation-
   ordered (gen N consumes, gen N+1 sees the update). Within a generation, parallel particles consuming
   the same background can conflict. v1 = generation-ordered approximation; validate vs ACHILLES; escalate
   to time-stepped lockstep only if it matters.
2. **Buffer sizing** (P, MAX_GEN): measure from ACHILLES (status-29 cascade-participant multiplicity ~<=8;
   existing banks) before fixing.
3. **Reweight variance**: more branches -> more weight factors -> variance growth away from nominal
   (standard kind-1). Exact at nominal; check the usable theta range.

## Plan (phased, each gated before the next)

- **P0 Scoping**: measure per-generation width + depth from ACHILLES (status-29 multiplicity) and the rich
  banks -> fix P, MAX_GEN. Standalone module `adonis/fsi/cascade_full.py` + a dev driver.
- **P1 Unified kernel**: species-tagged single-particle propagator on the shared nucleus, emitting all
  secondaries + records. GATE: single-particle limit (one initial particle, no secondary re-propagation)
  == existing _propagate_discrete / _propagate_nucleon_discrete, bit-exact.
- **P2 BFS loop**: fixed buffers + lax.scan over MAX_GEN + overflow logging. GATE: pi+-12C transparency
  oracle reproduced (the existing validation).
- **P3 kind-1 records + reweight(theta)**: per-particle records; reweight pure function. GATES: nominal
  identity (weights==forward), reweight==in-walk (bit-exact), autodiff==central-FD closure.
- **P4 Validation vs the rich banks** (cc1pi_signal sigdef framework, no regeneration of ACHILLES):
  pion fate decomposition (abs/cex/scatter vs ACHILLES -- check the over-absorption #33), CC1pi inclusive
  (target the 6% pion-FSI gap) and full signal (target the proton-leg yield deficit), high-W region,
  and the created-pion-rescue 2nd-order effect. Compare to t2k_cc1pi_rich_{adonis,ach_FSI,ach_nofsi}.npz +
  the transparency oracles.
- **P5 (deferred)**: compute optimization (compacted stack) and/or time-ordered consumption if P4 demands.

## Status

- Plan written; git clean at 273b0de (rich-bank/sigdef system committed).
- **P0 DONE** (200k ACHILLES T2K_CH_virt events, cascade-hadron multiplicity):
    status29 (intermediates): mean 1.41  max 22  p99 10  p99.9 12
    final-state hadrons:      mean 1.99  max 12  p99 7   p99.9 8
    TOTAL cascade hadrons:    mean 3.40  max 33  p99 17  p99.9 21   (final pions p99.9=2, nucleons p99.9=8)
  => budget mean ~3.4, p99.9 ~21 over a shallow tree. CHOSEN: P=10 particles/generation, MAX_GEN=6
     (60-slot budget; covers p99.9 even if bunched). Overflow COUNTED+logged in the engine to verify/refine.
     v1 waste is large (mean 3.4 vs 60 slots) -> confirms the v2 compacted-stack optimization will matter.
  Next: P1 unified species-tagged single-particle kernel (gate: single-particle limit == existing kernels).
- **P1a DONE** (adonis/fsi/cascade_full.py): the fixed-shape BFS MECHANISM (problem #2) built + validated,
  with a PLUGGABLE single-particle kernel (physics dropped in next). ParticleBatch dict (species, charge,
  p4, pos, fz, alive, w, fate); compact() packs live particles to a fixed-width buffer + COUNTS overflow
  (never silent); run_cascade() does BFS over MAX_GEN generations. Tests (toy kernels): nspawn=0 ->
  gen0-only 8 terminals; nspawn=1 -> width-stable 48; nspawn=2 -> [8,16,32,40,40] doubling then capped at
  P*events with overflow=104 counted; jit OK with a pure-JAX kernel. The mechanism (variable particle
  count in fixed shapes, spawning, compaction, overflow) is the part the reverted event-cascade lacked.
  Next P1b: the unified species-tagged PHYSICS kernel (pion: oset abs / MB scatter+charge-exchange /
  conversion; nucleon: NN elastic / NN->NDelta->NNpi), emitting ALL secondaries, reusing the primitives
  from cascade_discrete (_xsec, _two_body_cm_scatter, abs/inelastic splits, formation zones). GATE:
  single-particle limit (one initial particle, secondaries discarded) == _propagate_discrete /
  _propagate_nucleon_discrete, bit-exact.

## Design decision (after reading both kernels in full): BFS-by-generation, reuse the segment kernels

Two architectures considered for the engine loop:
  (a) WORKLIST / interaction-stepped co-evolution: one evolving live-set; every particle advances ONE
      interaction in lockstep; dead leave, secondaries refill (the user's stack). MOST faithful (shared
      consumed set updated in TIME), bounds secondaries/step to <=2. But it can't reuse the existing
      multi-step propagators and needs ~30 outer steps x compaction.
  (b) BFS-by-generation: each particle propagates its FULL multi-step segment in one kernel call (REUSE
      _propagate_discrete / _propagate_nucleon_discrete verbatim -> single-particle-limit gate is
      automatic); the pion's charge oscillation (pi+ ->pi0 ->pi+) stays WITHIN its segment (already
      faithful); only RECOIL nucleons + inelastic/abs product pions spawn the next generation.
CHOSEN v1 = (b): smallest faithful step from the current code, reuses validated physics, gate for free.
  - Secondaries per segment capped at SPAWN (v1: the leading recoil, as today, but now RE-CASCADED across
    generations -- the new capability vs the current 2-deep chain); extend to top-N next. Overflow counted.
  - Approximations (flagged, validate in P4): generation-ordered consumed set (not time-ordered);
    parallel consumption within a generation (P slots consume the shared background independently).
  - Escalate to (a) only if P4 shows the time-ordering / all-secondaries matter beyond the cap.

- **P1b (pion) DONE**: cascade_full.pion_segment wraps _propagate_discrete -> (terminal pion + fate,
  leading-recoil proton secondary); setup_carbon replicates DiscreteCascadeFSI's nucleus/struck-vertex
  setup. GATE (4k events, same key): survived-pi+ p4 max|diff| = 0.0; pid match exact; survival
  1035==1035; leading-recoil p4 == pion.last_scat_ko (0.0). Bit-exact reproduction of the production
  chain. NEXT P1b: nucleon_segment (wrap _propagate_nucleon_discrete -> terminal + leading knockout;
  inelastic pion currently veto-flag only -> extend the kernel to RETURN _pPiX so it spawns faithfully),
  then species-dispatch both into run_cascade, then top-N secondaries.

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

- (this entry) Plan written; git clean at 273b0de (rich-bank/sigdef system committed). Next: P0 scoping.

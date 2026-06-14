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
- **P1b (nucleon) DONE**: cascade_full.nucleon_segment wraps _propagate_nucleon_discrete -> (terminal
  nucleon + leading knockout proton; made_pi flag). GATE (4k): term p_N max|diff| 0.0, knockout p4
  max|diff| 0.0 vs the direct propagator. Both species kernels now bit-exact.
  NEXT (integration): species-dispatched mixed-buffer kernel in run_cascade (run both segment fns per
  slot, select by species; shared nucleus + consumed across the P slots -- v1 parallel-consumption
  approx). Then faithful extensions: emit the NN-inelastic pion 4-vec (extend _propagate_nucleon_discrete
  to return _pPiX) and top-N secondaries. Then P2 transparency, P3 records/autodiff, P4 validation.
- **P1b INTEGRATION DONE** (engine alive end-to-end): make_kernel (species-dispatched mixed buffer: each
  slot runs both segment fns, selects by species; shared nucleus+consumed; 1 leading secondary/slot) +
  cascade_carbon (gen-0 = primary pion + recoil nucleon at the struck vertex, BFS via run_cascade).
  TEST (n~1361 carbon RES events, P=8, max_gen=5): surviving pi+ engine 782 vs production 789 (~0.25sigma;
  not bit-exact only because each gen/slot gets distinct keys by design -- the bit-exact gate is the
  standalone pion_segment, passed). Generations decay [2722,512,42,0,0] (gen0=2*1361), protons tracked
  1468 (vs the current 2-deep chain's few), overflow 0. The faithful multi-particle BFS cascade RUNS with
  validated physics.
  NEXT (faithful extensions, each gated): (1) top-N secondaries per segment (not leading-only) -- extend
  the segment kernels to emit all recoils/products; (2) emit the NN-inelastic pion 4-vec (extend
  _propagate_nucleon_discrete to return _pPiX) so the created-pion-rescue effect is modeled; (3) P3 kind-1
  per-particle records + reweight(theta) (gates: nominal identity, reweight==in-walk, autodiff==FD);
  (4) P2 pi+-12C transparency; (5) P4 validate vs the rich banks (over-absorption, 6% pion gap, proton
  yield, created-pion rescue). Also: v2 compute (species-partition instead of 2x-run-both; compacted stack).
- **DIFFERENTIABILITY PROVEN on the engine** (problem 1, on the real BFS engine): threaded sabs/sscat
  through cascade_carbon -> make_kernel -> segments. O(sabs) = engine surviving-pi+ xsec; autodiff
  dO/dsabs = -1.4845e-6 vs central FD -1.4845e-6, rel 1.4e-7. Gradient negative (more abs -> fewer
  surviving pi+), as expected. The frozen-walk + per-particle kind-1 weight product down the tree IS
  differentiable, confirmed numerically. BUG FOUND+FIXED by this test: make_kernel set the TERMINAL
  weight = parent weight only, dropping the segment reweight seg_w -> surviving-pi+ was constant in sabs
  (FD=0); fixed to w = parent_w * seg_w. (Secondaries already carried it.)
  STANDING: faithful multi-particle cascade engine runs end-to-end, bit-exact single-particle gates,
  differentiable (autodiff==FD 1e-7). Foundation complete. Remaining = faithful-secondary extensions
  (top-N + inelastic pion) then P2/P4 validation vs the banks.

## Timing + device (first engine run, 30k->13769 carbon RES events)

- res_xsec.generate: 32.3s (the PRIMARY sampling -- a real cost, separate from the cascade).
- engine P=10 mg=6: FIRST (compile+run) 251.3s ; WARM 238.7s -> RUNTIME-bound (compile only ~13s).
  => ~58 ev/s, ~20-40x slower than the factorized chain.  The cost is algorithmic: the v1 kernel runs
     BOTH segment fns on ALL P slots and pads each generation; P0 said mean 3.4 particles vs 10x6=60
     slot-gens -> ~2x(60/3.4) ~ 35x wasted work on empty/masked slots.
- DEVICE: CPU (jax.devices()=[CpuDevice], backend cpu); float64 (jax_enable_x64=True, physics needs it).
  Apple Silicon GPU is NOT viable: the only path is jax-metal (experimental), which has poor/no float64
  (Metal is float32-centric) -> would break the bit-exact physics, plus gaps in while_loop/scan/scatter/
  top_k that the cascade uses.  And it's runtime-not-compile bound, so even a working GPU helps only the
  raw flops, not the wasted-slot overhead.
- CONCLUSION: v2 (compacted live-set stack + species-partition, no 2x-run-both, no per-gen padding) is
  ESSENTIAL, not optional -- it's the ~10-35x lever that makes the engine practical, on CPU.
- SCALING CONFIRMED linear in P x MAX_GEN: P=10 mg=6 (60 slot-gens) warm 238.7/238.8s; P=6 mg=4
  (24 slot-gens) warm 95.1/95.7s; 239/95 = 2.5 = 60/24 exactly. Cost ~ P*MAX_GEN*2species*n_events.
  v2 target: compact to ~mean 3.4 live + species-partition -> ~5-10x -> ~58 ev/s to ~300-600 ev/s.

## P4-lite PHYSICS (engine vs ACHILLES proton-leg, the localized deficit) -- FIRST RESULT, POSITIVE

scripts/cascade_engine_protoncheck.py, 25k->2138 carbon signal events (1 seed, P=10 mg=6).
P(in-window proton | pi_p) ENG/ACH per bin (225..1125): 1.13 1.15 0.83 1.08 0.77 1.13 0.91 (mean ~1.00).
Compare: BUGGY chain 1.0->1.9 (too high, the tail); FIXED current chain 0.88->0.70 (systematically too
LOW, the deficit). => the engine's deeper shared-nucleus re-cascade REMOVES the systematic proton-leg
deficit -- it scatters around 1.0 with no monotonic bias. The recoils re-cascade + repopulate the
[450,1200] window like ACHILLES's full cascade, vs the factorized leading-only chain's under-production.
CAVEAT: 1 seed / 2138 events -> +-15-23% bin scatter is partly statistical; cannot claim a tight %
agreement yet, and v1 is leading-secondary only (top-N may refine). But the trend is unambiguous and
positive: the monotonic 0.70-0.88 deficit is gone. The faithful engine does the right proton-leg physics.
NEXT: (v2 speedup to afford high stats) -> confirm the agreement at low MC error; then top-N secondaries
+ inelastic-pion emission; then the full CC1pi signal + pion over-absorption checks vs the rich banks.

## FULL CC1pi SIGNAL: faithful engine vs ACHILLES -- ~1% across all variables (4 seeds)

scripts/cascade_engine_figure.py (engine signal = surviving pi+ gen-0 + leading in-window proton across
ALL gens + muon; observables via cc1pi_fig_tki.observables; vs ach_select fixed sigdef).  4 seeds, ~2614
signal events.  ENGINE chi2/ndf | (factorized fixed chain in parens):
  integral ACH/ADO 0.996 (1.163)   <-- the 16% proton-leg deficit CLOSED to 0.4%
  W 1.28 (2.75)  pi_p 0.86 (5.06)  pn 3.23 (12.71)  dptt 2.72 (6.17)  daT 0.51 (9.61)  Q2 1.36 (8.12)  lp_p 1.39 (3.77)
=> the faithful BFS engine (v1, leading-secondary, differentiable autodiff==FD) reproduces ACHILLES's
   full CC1pi signal to ~1% integral with all shapes agreeing (chi2 0.5-3.2). Both residuals of the
   factorized chain -- the W/pi_p TAIL and the ~16% norm DEFICIT -- are gone. This is the project goal
   (<=1% ADoNIS-vs-ACHILLES) MET for CC1pi, with differentiability preserved.
CAVEATS (honest): 4 seeds / 2614 events -> large bin errors; pn 3.2, dptt 2.7 slightly elevated (stats or
   a small residual -- the missing inelastic-pion veto + leading-only secondaries). Needs the v2 speedup
   for high-stats confirmation; top-N + inelastic-pion are the remaining faithfulness items. Engine is
   ~20-40x slower than the chain (v2 compaction pending).

## HIGH-STATS CONFIRMATION (v2, 8 seeds, padded -> 1 compile) + the N_eff finding

8 seeds (pre-generated, padded to max=13861 -> single compile, 0 dropped), 5324 signal events.
ENGINE chi2/ndf | ACH/ADO:  pn 1.01/0.972  dptt 1.30/0.973  daT 1.27/0.973  W 2.53/0.943
  Q2 2.13/0.963  pi_p 1.03/0.973  lp_p 0.48/0.973.  integral ACH/ADO 0.973.
=> CONFIRMED: the faithful engine reproduces ACHILLES's full CC1pi signal; all chi2 <= 2.5; integral
   0.973 +-6% consistent with 1.0. The low-stats pn 3.2/dptt 2.7 were STATS (now 1.0/1.3). v2 has no bug
   (the 0.88/1.0/1.5 integral scatter was N_eff noise; converges to 0.973).
N_eff FINDING (important, generator-level): ADoNIS res_xsec SIGNAL weights are spiky -- N_eff/N ~ 0.04-0.07
   (engine 1-seed 664->N_eff 27; factorized cc1pi_ratios 12833->N_eff 932; ACHILLES tki 54790->N_eff 50104,
   i.e. ACHILLES is ~UNWEIGHTED, N_eff/N 0.91). So ADoNIS needs ~13x more events than ACHILLES for the same
   precision. The Gaussian weighted-error chi2 (ed=sqrt(Sum w^2)) propagates N_eff correctly (ed/dd=1/sqrt(N_eff,bin))
   but DEGRADES at low N_eff/bin (~1): Sum w^2 underestimates the variance for sparse bins + the bin sum is
   non-Gaussian -> chi2 inflated (the NRES=3000 smoke chi2=27 was this, not a bug). Fix = high stats /
   unweighting / better importance proposal for the signal corner.
v2 differentiability CONFIRMED both knobs: sabs rel 1.2e-7, sscat rel 4.6e-7.
PADDING: pre-generate all seeds, N_PAD=max count, pad short seeds w=0 (excluded by w>0 cut), no truncation,
   single compile (no per-seed re-JIT). (one big seed instead is memory-blocked: 17GB RAM, ~6.5GB/13.8k ev.)
v2a SPEED: 141s vs v1 239s (~1.7x); the bigger v2b worklist/compaction still pending.

## TOP-N secondaries DONE (kernels + engine), production bit-exact

- Kernels: _propagate_discrete (pion recoils) and _propagate_nucleon_discrete (nucleon knockouts) now
  track the TOP-K proton recoils (_N_RECOIL=4) in the carry via min-slot insertion (was leading-only).
  The max-momentum slot = the old single best_rec/best_ko bit-exactly, so the production return (which
  uses only the leading) is UNCHANGED. GATES: prodgate DiscreteCascadeFSI pid/p_pi/w diff 0.0 (leading
  recoil 2e-13 float-reorder); full res_C chain count/sigma identical, observables <=1e-12. scat_ko_all /
  ko_all (n,K,...) appended for the engine; all callers (DiscreteCascadeFSI/DiscreteNucleonFSI/res_C)
  unpack the new return.
- Engine: pion top-K recoils -> gen0 nucleon buffer; nucleon kernel emits P*K knockouts/gen (compacted).
  MEASURED (1 seed, 13769 ev): protons tracked 14662 -> 15477 (+5.6%); surviving pi+ unchanged (8036);
  159s vs 141s (~13%, NOT Kx -- compaction bounds it); overflow 0 at P=12 (P=24 IDENTICAL -> fully
  captured, no drops). So top-K is cheap and complete.
- TODO verify: re-run the AD==FD differentiability with top-K on (same w_fsi/w_scat mechanism -> expected
  to hold) and the 8-seed engine figure with top-K (does +5.6% protons move the 0.973 integral?).

## REMAINING fidelity: inelastic-pion emission (created-pion-rescue) -- the bigger structural item

NN->NDelta->NNpi currently DISCARDS the created pion (_pPiX) and only sets made_pi (veto flag).  To model
the created-pion-rescue (primary pi+ dies, a nucleon-FSI pi+ becomes the signal pion):
1. extend _propagate_nucleon_discrete to RETURN the created pion 4-vec + CHARGE.  Charge needs the
   NN->NDelta->Npi isospin/Clebsch-Gordan (Delta charge -> pi charge) -- NEW physics (current code uses
   isospin-avg mass 138.04, no charge).  Read ACHILLES NucleonNucleon/Delta-decay charge handling.
2. engine: the created pion RE-ENTERS the cascade as a pion -> breaks v2a's "pion-once" -> need a pion
   stream back in the BFS (pion buffer alongside the nucleon buffer).
3. no-other-meson veto becomes exact: signal = exactly one surviving pi+ (currently approximate).
This is a fresh-context sub-task (new charge physics + engine restructure).  top-N is the cheap win;
this is the structural one.

## Inelastic-pion sub-task (a): ACHILLES charge physics READ (NucleonNucleon.cc + data/decays.yml)

NN pair -> Delta states by total charge (AllowedResonanceStates), split by isofactor (Delta++/- =1,
Delta+/0 =1/3):
  pp(q=2): Delta++ n (3/4) | Delta+ p (1/4)
  pn(q=1): Delta+ n (1/2)  | Delta0 p (1/2)
  nn(q=0): Delta0 n (1/4)  | Delta- p (3/4)
Delta -> N pi branching (decays.yml; gamma channels 0.55% dropped+renormalized):
  Delta++ -> pi+ p (1) ; Delta+ -> pi+ n (1/3), pi0 p (2/3) ; Delta0 -> pi0 n (2/3), pi- p (1/3) ;
  Delta- -> pi- n (1).
=> created-pion pi+ fraction by pair: pp 5/6, pn 1/6, nn 0.  (kinematics already done in
_propagate_nucleon_discrete inelastic branch with isospin-avg masses; only the CHARGE/pid is new.)
PLAN: (i) kernel -- sample Delta state (u107) + decay (u108) via NEW fold_in keys (existing inelastic
streams + made_pi stay bit-exact), track the leading made-pion (4-vec + charge + pos + fz), RETURN it
appended; gate made_pi + existing outputs unchanged.  (ii) engine -- spawn the made pion into a PION
buffer that re-enters the BFS (pion segment) -- breaks v2a pion-once.  (iii) no-other-meson veto exact.

## INELASTIC-PION DONE -- full-fidelity engine reproduces ACHILLES (~2%)

- (i) kernel: _propagate_nucleon_discrete samples the Delta charge state + Delta->Npi decay (fold_in
  107/108) and tracks the LEADING created pion (4-vec, charge idx, vertex, fz), returned appended.
  Carry surgery (15-element carry) GATED: res_C all observables 0.0 diff, count/sigma identical -> the
  production chain is BIT-EXACT (the new keys don't perturb the elastic/inelastic streams or made_pi).
- (ii) engine (cascade_carbon_v2): gather the leading NN-created pion per event across the nucleon BFS,
  re-cascade it via pion_segment; returns `created` (surviving pid/p4).  Smoke (9120 ev): 8.6% make a
  pion, 163 RESCUE (primary died + created pi+ survives).
- (iii) figure: combine {primary, created} pions with the EXACT one-pi+/no-other-meson veto (conv -1
  counts as other meson) -- apples-to-apples with ach_select.
- FULL RESULT (8 seeds, top-N + inelastic-pion + rescue + exact veto, N_eff 283): integral ACH/ADO 0.983;
  chi2/ndf pn 1.29 dptt 1.45 daT 1.25 W 2.55 Q2 2.40 pi_p 1.28 lp_p 0.57.  All <=2.55; integral 0.983
  +-6% consistent with 1.0 (top-K-only was 0.973).  => the fully-faithful, differentiable engine
  reproduces ACHILLES's CC1pi signal to ~2%.
- TIMING: 26.0 min/8 seeds (vs ~23 top-K-only; created-pion pass ~+13%).  CPU ~3.3/8 cores during run
  (XLA not saturating; headroom -> parallel seeds or v2b).  RAM ~2.2 GB/run.
- Residual leads (small, stats-limited): W 2.55, Q2 2.40 slightly elevated -- a possible mild Q2 shape
  diff (ADoNIS high low-Q2, low high-Q2) worth a high-stats look (needs v2b speedup / unweighting for N_eff).
DONE for this phase: top-N + inelastic-pion + rescue + exact veto, production bit-exact, differentiable.
Remaining (optimization/validation, not new physics): v2b worklist speedup; unweighting (N_eff/N~0.05);
consumption time-ordering validation; M_A/fit records; transparency + 6% pion check + CC0pi + wire into prod.

## OVERNIGHT high-stat reusable RICH banks (running) + offline consumer

- Two long runs from gen_cc_engine_rich.py, parallel, checkpoint every seed (partial always usable):
  - CC1pi (res) NSEED=50 -> data/oracle/t2k_cc1pi_engine_rich.npz  (~4.4 min/seed -> ~3.7 h)
  - CC0pi (qe)  NSEED=60 -> data/oracle/t2k_cc0pi_engine_rich.npz  (~5.3 min/seed -> ~5.3 h)
  Bank fields (per event): mu,nu,struck,pid_Ni; pi_post+pid_pi+pi_nsc (primary pion post-FSI);
  cr_p4+cr_pid (NN-created pion, 0 if dead); prot(top-6,4)+prot_origin(0=RES/QE-chain|1=pion-KO|-1=empty)
  +prot_gen(BFS depth); w; ipid/Npid.  Any signal definition is then pure re-binning, no rerun.
- NORM source-of-truth fix: gen now stores RAW per-event weight and divides by the ACTUAL number of
  seeds banked at each save (`out["w"]/len(parts)`), NOT the planned NSEED.  => every checkpoint, partial
  or complete, is correctly normalized; NSEED only sets the stopping point.  (In-flight runs predate the
  fix and divide by NSEED in-memory -> correct only on FULL completion; a complete bank is identical.)
- Offline consumer scripts/cc1pi_engine_plot.py: reads the rich bank, applies the DEFAULT sigdef
  (pi+ = primary OR NN-created, exactly-one-pi+/no-other-meson; leading in-window proton among top-6),
  observables via cc1pi_fig_tki.observables, overlays vs ACHILLES rich bank + ratio/chi2.  Dry-run on the
  2-seed partial: shape flat (ACH/ADO ~const across all 7 vars) -> selection logic correct; absolute norm
  off by ~NSEED/n_seeds (partial, pre-fix in-flight bank) as expected.  Full bank decides the absolute.

## HIGH-STAT CC1pi result (50-seed bank, 692,979 ev -> 31,194 signal) -- EVIDENCE
- cc1pi_engine_plot.py on the FULL bank:  selected sigma ENGINE 1.1319e-6 / ACH 1.1846e-6 nb.
  ACH/ADO = 1.047  (engine ~4.7% LOW; near-FLAT across observables, ratio points cluster 1.05-1.10).
- chi2/ndf:  pn 4.85  dptt 2.82  dalphat 1.19  W 1.44  Q2 1.95  pi_p 1.74  lp_p 1.43.
- pn is the worst: high-pn tail (600-1500 bin) ratio ~1.2 -> engine under-produces high inferred
  nucleon momentum.  The 8-seed 0.983 was within noise (N_eff~283, +-6%); 1.047 is the reliable value.
- => at high stats a ~5% near-flat normalization deficit + a pn high-tail shape deficit emerge (were
  stat-masked at 8 seeds).  NOT yet at the 1% goal.  To investigate (read code first): top-6 proton
  truncation bias (does the in-window leader ever fall past rank 6?), created-pion/veto bookkeeping,
  cascade survival vs ACHILLES.

## CC1pi residual DECOMPOSED (high-stat evidence)
- Significance: ACH/ADO = 1.0466 +- 0.0231 -> 2.0 sigma.  Engine signal N=31194 but N_eff=2182 (7.0%,
  spiky res_xsec weights) -> 2.14% stat err; ACHILLES N_eff 88.7% -> 0.55%.  Engine stats limit precision.
- Localize (pre/post-FSI on matched banks):
    PRE-FSI (generation): ACH/ADO 1.0277   (res_xsec gens ~2.8% fewer signal events, T2K fiducial)
    cascade survival post/pre: ACH 0.5023 vs ENGINE 0.4932 (engine over-removes ~1.8%)
    1.0277 x (0.5023/0.4932) = 1.0466  == measured.  => 4.7% = ~2.8% generation + ~1.8% cascade.
  Factorized chain for contrast: post-FSI ACH/ADO 1.163, survival 0.444 (16% low).  => the ENGINE
  (top-N + inelastic-pion + created-pion-rescue) is the fix: survival 0.493 ~ ACH 0.502.
- pn high-tail (600-1500): ratio ~1.2, i.e. ~1.15 even after dividing out the flat 4.7% -> a real
  high-pn SHAPE deficit (engine under-produces high transverse-momentum recoils, a cascade-transport
  feature).  Per-bin N_eff 367-990 (stat 3-5%) so partly stat-inflated, but visually consistent.
- NEXT (for the 1% goal): (a) chase the 2.8% pre-FSI res_xsec signal-region deficit -- CHEAP, no cascade,
  high-stat pre-FSI-only run; relates to historical sigma_RES residual but here it's the T2K-fiducial
  SHAPE.  (b) raise engine N_eff (more seeds / unweighting, N_eff/N~0.07) to turn the 2 sigma into a clear
  measurement.  (c) the pn high-tail ties to the parked cascade transport residual.

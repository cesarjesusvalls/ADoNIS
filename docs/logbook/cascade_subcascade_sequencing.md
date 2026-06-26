# QE/RES multiplicity vs ACHILLES: step-resolution + the refill recoil-drop bug

## RESOLUTION (supersedes the "sub-cascade sequencing" framing below)
The investigation below started from an apparent QE multi-knockout-neutron deficit and pursued a
stepping-clock (time-sync) explanation. The de-noised conclusion is different and is recorded here:

1. **The forward dσ-vs-multiplicity prediction agrees with ACHILLES to ~1%** for both QE and RES, and did
   so *before* any of this session's changes. The "deficit" was never visible in the actual forward
   observable at the level first claimed.
2. **QE N(n) high-k tail** (k=2 ~−7%, k=3 ~−30% on rare ~1e-8 bins) is a **step-resolution** effect:
   distance-sync at step 0.04 is mildly under-converged; finer step (0.02) or time-sync's β-weighting
   shifts it, and ADoNIS converges to ~0.8% of ACHILLES. The gen-resolved ablation (sub-cascade gen≥1
   ~−22%) is a *sensitive transport probe* that barely moves the forward observable — not a physics bug.
3. **The RES N(p) "halving"** (mean N(p) 1.78→0.80) seen when generation switched to the refill engine was
   a **refill BUG, not physics**: the refill path (`run_cascade_pool`) compacted g0 to width M and dropped
   primaries beyond M — for RES that is the **recoil nucleon** (2nd primary at P=1), and the refill init +
   slot-refill never routed it to the wait queue (the no-refill path was fixed for this; refill was not;
   the P-invariance gates only used lockstep, so it was never caught). FIXED: route the overflow primary to
   the wait queue at init (`compact(g0, M+Q)`) and on slot-refill. RES N(p)/N(n)/N(π) then match ACHILLES
   to ~1% (verified: refill n_p 0.77 → 1.751 == lockstep == cv5 1.78).
4. **time-sync (adaptive ACHILLES AdaptiveStep, `cfg.time_step`)** is a **wash** on forward observables vs
   distance-sync. De-noised at 250k–1M, distance / adaptive / decoupled (`Δt=step`) all match ACHILLES to
   ~1–2% on every species in QE and RES (an earlier "decoupled over-produces pions ~2.7x" claim was a QE
   rare-bin (~1e-7) statistical fluke -- it does NOT survive the de-noised RES, where pions are the primary
   and decoupled matches to ~1%). The three differ only in cost (distance < adaptive << decoupled ~3x) and
   P-invariance (distance & decoupled exact; adaptive a small ~99.9% break at intermediate P). time-sync is
   kept **behind `cfg.time_step` (default False = distance-sync, the committed engine)** as experimental/
   reference -- it confirms, rather than improves on, the default.
5. **Engine physics is faithful**: post-fix lockstep reproduces the pre-fix engine exactly; the per-scatter
   pn-recoil path was verified line-by-line vs ACHILLES (see the table below).

Net: the real, validated win this session is the **refill recoil-drop fix**. The distance-sync default is
correct; the stepping detour did not change the forward prediction.

## Question (original framing — kept for the record)
The QE CC0π final-state neutron multiplicity sits ~2.7% low vs ACHILLES (ACH/ADO mean N(n) 1.027),
and the deficit GROWS with knockout rank (rank-2 ~1.06, rank-3 ~1.45, rank-4 ~2.9). The leading proton
agrees at rank-0 (~1.3σ); the neutron does not (~5σ at rank-0). Where exactly is the neutron lost, and
is it a per-scatter physics bug or a transport effect?

## Method
All comparisons are on **identical input**: ADoNIS's pool cascade is run on ACHILLES's exact per-event
nucleus (background config + Fermi momenta + struck vertex) parsed from an `ACHILLES_CASCADEDUMP` stream
(`su_external`), so any difference is **pure transport** — input generation is removed.
- `scripts/cascade_ablation.py` — final-state N(p)/N(n) + ejected-neutron |p| spectrum.
- `scripts/cascade_seq_test.py` — cascade INTENSITY (successful NN scatters/event, neutron-ejecting
  scatters, ADoNIS per-generation breakdown, scatter-multiplicity distribution) via the in-engine
  segment logger (`cascade_nucleus(..., log_cap=L)`), on the same input.
- Pauli toggle: `cascade_ablation.py --no-pauli`.

## Per-scatter physics is FAITHFUL (line-by-line, both codebases)
Every ingredient of the pn-recoil / cascade-step path matches ACHILLES, verified at file:line on both
sides and numerically on identical inputs:

| Ingredient | ADoNIS | ACHILLES | Match |
|---|---|---|---|
| Formation zone | `_formation_zone(p4_lead, recoil)` (cascade_discrete.py:356) | `SetFormationZone(particle1.Mom, part.Mom)` (Particle.cc:9-11, Cascade.cc:971) | ✓ |
| k_F formula | `cbrt(ρ·3π²)·ħc` (cascade_real.py:97) | `cbrt(rho·3·π²)·HBARC` (Nucleus.cc:226) | ✓ |
| k_F density | per-species ρ_p/ρ_n (cascade_discrete.py:259) | `ProtonRho`/`NeutronRho` per PID (Nucleus.cc:214-217) | ✓ |
| Pauli predicate | `|p|<k_F` hard cut (line 281) | `|p|<k_F` hard cut (Cascade.cc:1006-7) | ✓ |
| Pauli — both outgoing | leading + recoil | loop over `particles_out` (Cascade.cc:815-6) | ✓ |
| Leading k_F position | leading's own `|pos|` (lines 269-271) | `particle1.Position()` = projectile (NN.cc:177) | ✓ |
| Recoil k_F position | struck vertex `kf_j` (line 260) | `particle2.Position()` = struck (NN.cc:178) | ✓ |
| Struck momentum | `nmom[j]`, no resample (line 257) | background config, unmodified | ✓ |
| σ (elastic per-pair mass + inelastic) | lines 238-243 | NNElastic.cc:184 | ✓ |
| Interaction prob | `exp(-π·b²/σ)` (line 248) | `exp(-π·b²/σ)` Gaussian default (Cascade.cc:39) | ✓ |
| Target selection | min-b² among prob-passers (lines 251-2) | first passer in b²-ascending order (Cascade.cc:782) — *provably identical distribution* | ✓ |
| 2→2 recoil kinematics | `_two_body_cm_scatter` | — | ✓ numerically (median 245 vs 247, frac>700 0.026 vs 0.027) |

**No per-scatter formula bug exists.**

## Pauli toggle (50k, identical input) — pins the soft shoulder
```
Pauli ON:   ADoNIS N(n)=0.2169  ACH=0.2283  ACH/ADO 1.053  (150-250 MeV ≈ k_F: 1.09-1.13, +2-4σ)
Pauli OFF:  ADoNIS N(n)=0.4008  ACH=0.2283  ACH/ADO 0.570  (massive over-production)
```
The soft-shoulder deficit (100-250 MeV ≈ k_F) is Pauli-sensitive. The hard tail (>550 MeV) is NOT Pauli
(far above k_F). Since per-scatter Pauli is bit-identical, this is not a Pauli-formula bug.

## Sequencing test (100k QE, identical input) — localizes the deficit to the sub-cascade
```
                              ADoNIS    ACHILLES   ACH/ADO
successful NN scatters/evt    0.4093    0.4163     1.017
scatters ejecting a neutron   0.2210    0.2243     1.015
ACHILLES attempts (incl blkd) 1.1069  → blocked frac 0.624

successful-scatter multiplicity:   k=0  0.995   (match)
                                   k=1  1.001   (match)
                                   k=2  1.030
                                   k=3  1.074
                                   k=4  1.069
                                   k=5  1.545
ADoNIS by generation:  gen 0 (primary) 0.3593   gen≥1 (sub-cascade) 0.0500
```
**Single-scatter events (k≤1) match to <0.5%. The deficit appears only at k≥2 and grows with
multiplicity (3% → 7% → 55%)** — i.e. it is entirely in events that require a SECONDARY (gen≥1) scatter.
This maps directly onto the rank-2/3/4 neutron multi-knockout deficit. The chain:
- per-scatter physics → faithful
- first scatter (gen 0) intensity → matches <0.5%
- **secondary scatters (gen≥1) → ADoNIS ~3-7% low → the multi-knockout-neutron deficit**

(N(n) final 2.7% low vs neutron-ejecting-scatters 1.5% low: the extra ~1% is created neutrons lost in
post-scatter propagation/capture — a smaller, second-order piece of the same sub-cascade story.)

## Structural cause: distance-sync vs time-sync stepping
Both engines are **breadth-first / time-synchronized** at the high level — all active particles advance
together each global step, and scatter products are picked up the NEXT step, never run to completion
first (ADoNIS pool; ACHILLES `Cascade::Evolve` Cascade.cc:344-383, products via `UpdateKicked` at :376).
That rule matches.

The divergence is the STEP SIZE:
- **ADoNIS**: every particle advances a FIXED DISTANCE `cfg.step = 0.04 fm` per global step
  (cascade_discrete.py:369); formation zone decremented by each particle's own `step/β` (line 226/366).
  → **distance-synchronized**.
- **ACHILLES**: ADAPTIVE GLOBAL timestep set by the fastest active particle
  `timeStep = stepDistance / max_β(all kicked)` (Cascade.cc:655-662); each particle advances
  `dist = β·timeStep`, so the fastest moves 0.04 fm and slower particles move LESS; formation zone
  decremented by the SHARED timeStep. → **time-synchronized**.

Properties:
- **Identical for a lone particle** (max_β = its own β → dist = 0.04). The schemes diverge ONLY when
  particles of different speeds coexist — exactly the sub-cascade (fast leading + slow near-k_F recoil).
- **Formation-zone expiry DISTANCE is invariant** to the scheme (≈ `fz·β` either way), so this is an
  ordering/competition effect on the shared "already-hit" (`consumed`) mask and on re-scatter timing,
  not a per-particle physics change.

Within-step tie-break also differs (ACHILLES ascending particle-index `std::set`; ADoNIS pool slot order
by `gtime` then `sid`) — a secondary candidate.

## Status
- The deficit is a **transport-sequencing residual localized to the sub-cascade (gen≥1, k≥2)**, ~1.5%
  on neutron-ejecting scatters / ~2.7% on final N(n), rare-event-dominated. Headline QE multiplicity
  agreement is ~1%.
- The leading structural candidate is the distance-sync vs adaptive-time-sync stepping; **not yet proven**
  vs the within-step tie-break order. The decisive test is to implement ACHILLES's adaptive global
  timestep in the pool and re-run `cascade_seq_test.py`; if k≥2 closes, the stepping scheme is confirmed.
- Tension to reconcile: an adaptive global timestep COUPLES coexisting particles via `max_β`, which
  interacts with the P=1 serialization / P-invariance design (pool_p_invariance.md). Design TBD.

## Scripts / evidence
- `scripts/cascade_ablation.py` (final state + |p| spectrum, `--no-pauli`)
- `scripts/cascade_seq_test.py` (intensity + per-generation + multiplicity)
- `scripts/_scatter_kin_compare.py` (2→2 recoil kinematics on ACHILLES inputs)
- dumps: `_oracle_out/QE_cascdump.cdump` (50k), `_oracle_out/QE_cascdump_big.cdump` (500k)

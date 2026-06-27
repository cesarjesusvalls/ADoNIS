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

## 2026-06-27: fix P=1 + commit to ACHILLES within-step ORDER (user decision)
Decision (user): stop allowing P to vary (P=1 is the validated fastest config), and commit the within-step
processing order to ACHILLES's. So the (gtime, sid=RNG-key) P-invariant tiebreak — kept only to make
results P-independent — is no longer needed; replace it with ACHILLES's order and hardcode the stack at M=1.

ACHILLES order (verified, Cascade.cc): `kickedIdxs` is a `std::set<size_t>` iterated **ascending particle
index = creation order** (Cascade.cc:353); `FinalizeMomentum` appends products (higher index), picked up
the NEXT timestep (`UpdateKicked` at :376). Maps onto the pool's gtime cohort + within-cohort tiebreak.

Changes (adonis/fsi/cascade_full.py):
- `compact(sort_priority=True)` lexsort tiebreak: `pkey` (sid) -> `track_id` (creation index).
- `track_id` now ALWAYS populated in the stepper spawn (was `with_seg`-only); `step_i` passed to the
  stepper in every path (forward too) so track_id is monotone in creation time. pkey is KEPT for the
  physics RNG (step_key), only the ORDERING key changed.
- `P` removed from `cascade_nucleus`/`_cascade_pool` (hardcoded M=1); `CascadeHyperparams.P` field, the
  CLI `--P`, and `pool_fsi.run_fsi(P=)` removed. P-study/timing dev scripts now obsolete (flagged).
- NOTE on caps: with M=1 the per-event `nstep` is the SUM of all particles' steps (serial), so dev cfgs
  with small `max_steps` trip the runaway guard; production already uses `max_steps=100000` (path-budget
  is the physics bound) and runs clean (overflow 0).

Evidence — forward QE CC0π on C, muon-accepted, 120k (_ord) vs same ACHILLES bank vs the pre-change bank:
```
                 ACH      ADO_old  ACH/old   ADO_new  ACH/new
mean N(n)        0.2132   0.2087   1.022     0.2079   1.026      <- headline FLAT (slightly worse)
mean N(p)        1.1233   1.1231   1.000     1.1294   0.995
N(n) k=2         0.01591  0.01460  1.090     0.01490  1.068
N(n) k=3         0.00132  0.00094  1.406     0.00102  1.290      <- moved a little, but ~60-event bin
N(n) k=4         0.00013  0.00003  4.60      0.00007  1.85       (noise-dominated)
```
The ACHILLES-order change moves the rare high-k tail slightly toward ACHILLES but **does NOT close the
headline mean-N(n) residual** (1.022 -> 1.026, flat within sampling noise); N(p) is unchanged within noise.
=> within-step ordering is NOT the dominant lever for the QE neutron-multiplicity residual. (An earlier raw
smoke test WITHOUT the muon cut showed a large k=2 jump; that was the no-cut selection, not the ordering.)
The change is kept anyway: it is the FAITHFUL ACHILLES order (removes the arbitrary RNG tiebreak) and is
not a regression.

### Identical-input ablation (DEFINITIVE, transport-only) — ordering REFUTED
Rebuilt an ACHILLES image with the CASCADEDUMP instrumentation (no existing image had it; the
`achilles:cascadedump` tag had been repurposed for a RESDUMP build). Two gotchas fixed during the rebuild:
(a) with `BUILD_SHARED_LIBS=ON` the dump strings live in `libphysics.so`, not the exe; (b) the RESDUMP
instrumentation (XSecBackend.cc) was ALWAYS-ON and flooded the CASCADEDUMP stderr — gated it behind
`ACHILLES_RESDUMP` (Achilles fork). New image `achilles:cascadedump2` (Dockerfile.vtxw + `-Wl,--no-as-needed`).

ADoNIS NEW-ordering engine run on ACHILLES's EXACT per-event input (5725 QE events, pure transport):
```
              ADoNIS   ACHILLES  ACH/ADO        (OLD engine, logbook)
mean N(p)     1.1572   1.1474    0.991
mean N(n)     0.2164   0.2235    1.033          <- SAME as old engine's 1.033 (NO change)
N(n) k=2                         1.045
N(n) k=3                         1.289
ejected-n |p| p50 337/336  p90 608/618  p99 917/1159   <- ADoNIS still misses the high-|p| tail
```
The transport-only mean N(n) ACH/ADO = **1.033 is IDENTICAL to the old (RNG-order) engine's 1.033**. The
ACHILLES within-step ordering change does NOT move the transport residual at all. Combined with the flat
forward result, this is conclusive: **within-step processing order / seeds is NOT the cause** of the QE
neutron-multiplicity deficit. The residual is a genuine sub-cascade transport effect — ADoNIS
under-produces secondary neutrons at k≥2 AND at high |p| (p99 917 vs 1159 MeV) on identical input, with
bit-faithful per-scatter physics. Next candidates (NOT ordering): formation-zone / re-scatter count limiting
the high-|p| secondary tail; recapture in the sub-cascade; or the gtime-cohort vs ACHILLES adaptive-timestep
coupling (earlier ruled a wash on forward, but the high-|p| tail was not isolated then).

### 2026-06-27 (cont.): localize the neutron loss — it is POST-ejection, high-|p|, multi-scatter
Full-stats identical-input ablation (~50k QE, new image `achilles:cascadedump2`):
- mean N(p) 1.000 (perfect), mean N(n) ACH/ADO **1.045**; N(n) k=2 ACH/ADO **1.177** (now solid, ~900 evts).
- `--seq`: successful NN scatters 1.002, **scatters ejecting a neutron 0.999** (PERFECT). So ADoNIS ejects
  exactly the ACHILLES number of neutrons at the scatter level — the entire deficit is POST-ejection.
- ejected-neutron |p| spectrum (ablation_qe_neutron_p.png): matches <600 MeV (so NOT low-|p| recapture),
  ADoNIS progressively LOW above ~650 MeV (p99 984 vs 1065). The loss is the HIGH-|p| neutron tail.
- N(p) "perfect" is DILUTION: mean N(p)=1 primary proton + ~0.16 knockout; a generic ~5% sub-cascade
  knockout deficit shows as 5% in N(n) (all knockout) but ~0.7% in N(p). So NOT necessarily neutron-specific.

Concrete ACHILLES/ADoNIS DIFFERENCE found while chasing this (NucleonNucleon.cc:62-65): ACHILLES NN
ELASTIC offers TWO final states at HALF-σ each -- `{id1,id2}` and the **id-SWAPPED `{id2,id1}`** -- i.e. a
50% charge-exchange (for pn: 50% neutron-forward / 50% neutron-recoil). ADoNIS has NO swap (leading always
keeps the incident species, recoil the struck). Per single scatter under ISOTROPIC CM (both codes isotropic)
the swap is count- and spectrum-NEUTRAL (neutron is high-|p| half the time either way), so it cannot explain
the single-scatter match-but-cascade-deficit by itself. BUT it changes WHICH SPECIES is the fast penetrating
particle in the SUB-CASCADE (ACHILLES: a fast forward NEUTRON 50% of pn; ADoNIS: always a fast proton),
which can change downstream multi-scatter neutron production -> a viable multi-scatter candidate. NOT YET
implemented/measured. (kin mode earlier matched the RECOIL only; it never tested the swapped forward neutron.)
Status: ordering REFUTED; tracking-drop REFUTED (ejection matches, no structural drop); recapture REFUTED
(low-|p| matches). Open: high-|p| multi-scatter secondary-neutron deficit; the NN-elastic charge-exchange
swap is the leading concrete difference to implement + measure next.

### 2026-06-27: IMPLEMENTED NN-elastic charge exchange -> high-|p| tail FIXED
Added the 50% id-swap to ADoNIS `_nucleon_step` elastic branch (cascade_discrete.py): a per-event coin
(fold 109) swaps the outgoing isospin of the continuing (a-role) and recoil (b-role) nucleons -> threaded
through the 2->2 masses, the Pauli k_F species (leading at its own pos, recoil at the struck vertex), the
recoil charge, AND a NEW leading-charge update (the engine never flipped a nucleon's charge before; also
fixed the latent inelastic leading-charge = channel charge).  `make_pool_stepper` applies `qln` so a
nucleon's charge can now change.  No-op for pp/nn.

Identical-input ablation (same 108k-QE dump), BEFORE -> AFTER charge exchange:
```
                    BEFORE   AFTER   ACHILLES
mean N(n) ACH/ADO   1.045    1.028
mean N(p) ACH/ADO   1.000    1.003
N(n) k=2 ACH/ADO    1.177    1.123
N(n) k=3 ACH/ADO    1.289    1.017     <- fixed
ejected-n p99 |p|   984      1059      1065   <- HIGH-|p| TAIL FIXED
ejected-n p90 |p|   603      618       619
ejected-n count     23785    24089     24181
```
The missing NN-elastic charge exchange WAS the high-|p| neutron-tail deficit (p99 984->1059≈1065) and ~40%
of the mean-N(n) residual (1.045->1.028).  Mechanism: ACHILLES puts the fast forward particle as a NEUTRON
in 50% of pn elastic; ADoNIS always kept it a proton, so its neutrons were systematically the slow recoil.
RESIDUAL after the fix: mean N(n) ~1.028, k=2 ~1.123 -- a smaller multi-neutron piece still open (next lead).
ejected-neutron |p| spectrum now matches across the full range (ablation_qe_neutron_p.png, ratio chi2/ndf 1.1).

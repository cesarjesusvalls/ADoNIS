# Cascade divergence registry — ADoNIS vs ACHILLES

Living document. One row per divergence between the ADoNIS JAX cascade and the ACHILLES C++ reference
on the **active execution path** for the tagged-beam validation
(`Mode: CrossSection`, `Algorithm: Base`, `Probability: Gaussian`, `InMedium: None`,
`PotentialProp: False`, `VirtResInteractions.yml` with `ResonanceMode: Decay`).

Edit in place as items are resolved. Do not delete rows — move them to **RESOLVED** with the evidence.

## RESULT — Round 1 (D1+D2+D7, commit 44cec6d), full regeneration, both-sides errors

Prot-beam π-production, aggregated over the three open bins, ADoNIS/ACHILLES:

| target | pre-fix | **Round 1 (D1+D2+D7)** |
|---|---|---|
| ¹²C  | 0.9707 | **1.0064 ± 0.0129 (0.5σ)** |
| ⁴⁰Ar | 0.9546 | **0.9977 ± 0.0098 (0.2σ)** |

**The Ar residual is CLOSED.** The 2.6σ deficit is gone; both targets now agree with ACHILLES at <0.5σ.
All 6 configs regenerated at ~2M events single-core; neut and pip banks are now real (not baseline
fallbacks).

**Round 2 (D3/D6/D8/D10, commit 801855c)** implemented and committed after reading the ACHILLES source
for each. Smoke clean (reacted within baseline noise, no NaN). Full regeneration done, banks
`beam_<b>_<t>_v2` (~1.7–2.0M ev each).

**Round 2 RESULT — v2 slightly OVER-corrects; v1 remains the best-agreement configuration.**

prot π-production, aggregate, ADoNIS/ACHILLES (both-sides errors):

| target | v1 (D1+D2+D7) | v2 (+D3+D6+D8+D10) |
|---|---|---|
| ¹²C  | 1.0064 | 1.0120 ± 0.0138 (0.9σ) |
| ⁴⁰Ar | **0.9977** | **1.0188 ± 0.0101 (1.9σ high)** |

neut π-production rose consistently (+0.7% C, +1.1% Ar); pip **unchanged** (0.0% — the pion beam has no
NN→NΔ→NNπ channel), which localizes the shift to Δ-decay pion production exactly as expected.

**Interpretation (honest).** My a-priori guess that D6+D8 would *reduce* pions was WRONG. The (1+3cos²θ)
law is peaked at cosθ=±1, so more decay pions go forward along the Δ (≈beam) direction and **escape more
easily** → *more* surviving pions. This dominates D8's ~0.5% γ-branch removal. Net: v2 raises π
production ~+2% on Ar, moving it from 0.2σ low to **1.9σ high**. Each of D3/D6/D8/D10 is individually
faithful to the ACHILLES source, but the *bundle* over-corrects Ar.

**Recommendation.** D1+D2+D7 (v1, commit 44cec6d) is the best-agreement production configuration — both
targets <0.5σ. D3/D6/D8/D10 (v2, commit 801855c) are committed and correct-in-principle but net over-
correct Ar to 1.9σ. Before adopting v2 for production, D6 deserves a closer look (is the anisotropy axis
— Δ-momentum vs the mother-nucleon direction — exactly ACHILLES's? the Δ has no Mothers set so the code
uses its own momentum, but this is the single most likely spot for a residual sign/axis subtlety). This
is a **user decision**, deferred: keep v2, revert to v1, or investigate D6 further. Not claimed as an
improvement.

### D3, D6, D8, D10 — IMPLEMENTED (commit 801855c), faithfulness caveat above
- **D3** advance along pre-scatter direction (Cascade.cc:705) — both step fns.
- **D6** Δ→Nπ (1+3cos²θ)/4 about the Δ-momentum axis (DecayHandler.cc:126-131,76-79). Verified sampler.
- **D10** per-draw ZXZ Euler rotation of configs (Configuration.cc:76-79).
- **D8** Δ⁺/Δ⁰→Nγ BR 0.0055 (decays.yml).

## Symptom being chased

| Observable | Status |
|---|---|
| Total reaction rate (ADoNIS/ACHILLES) | **~1.00 everywhere** — agrees |
| π-production σ, nucleon beam | **2–5% LOW** (C 0.963±0.011, Ar 0.955±0.009 at high p; ~3–5σ) |
| Realized inelastic vertices per NN collision | 1.02–1.12 (slightly high) |
| Created-pion fates | 54–66% elastic, 8–14% CEX, 25–33% absorption, 0% conversion |

Any candidate must move the **π numerator** while leaving the **reaction denominator** intact.

---

## OPEN — confirmed code divergences

Verified by reading both sources; the ACHILLES path is confirmed active for this run card.

### D1. Escape test carries an `& outward` gate that ACHILLES does not have
- **ADoNIS**: `cascade_discrete.py:368` (`_nucleon_step`, `esc_sphere = (|pos|>radius) & outward`),
  `:615` (`_pion_step`, `inert = (|pos|>radius) & outward`)
- **ACHILLES**: `Cascade.cc:640` — `if(particle.Position().Magnitude2() > pow(radius,2))`, a **pure
  position check, no directional gate**. Called unconditionally every step (`Cascade.cc:382`) for every
  propagating particle.
- **Effect**: ADoNIS's escape set is a strict subset of ACHILLES's. A particle beyond `radius` but
  momentarily moving inward/tangentially (exactly the surface competition zone for a created pion) is
  retired by ACHILLES but stays interaction-eligible in ADoNIS → extra reabsorption chances.
- **Sign**: fewer surviving pions. **Matches the deficit.**
- **Provenance**: predates the `external_test` work — introduced in `1f2fd01` (original cascade).
- **Status**: NOT FIXED. Awaiting go-ahead.

### D2. `_pion_step` still uses the `nsc==0` proxy instead of the `external_test` flag
- **ADoNIS**: `cascade_discrete.py:614` `ext = nsc == 0`; `_pion_step` (`:591`) never receives an
  `is_beam` argument. Call site `cascade_full.py:192` passes nothing, while `:190` passes
  `is_beam=stack["external_test"][:,m]` to `_nucleon_step`.
- **ACHILLES**: only `ParticleStatus::external_test` gets the z-plane rule (`Cascade.cc:628-651`);
  secondaries are pushed as `Status::propagating` (`Cascade.cc:965-973`).
- **Effect**: created pions (also `nsc==0` at birth) wrongly get the beam's lenient plane escape.
- **Sign**: *opposite* to the deficit (bonus escape route), and smaller than D1 since
  `z>=R ⟹ |pos|>=R`. Net escape set is still a subset of ACHILLES's.
- **Note**: the flag itself is correct and is properly cleared on pion interactions
  (`cascade_full.py:237`). Only the plumbing into `_pion_step` is missing — one argument.
- **Status**: NOT FIXED.

### D3. Displacement direction on an interacting step
- **ADoNIS**: `cascade_discrete.py:579,583-584` — `p4` set to the **post**-interaction momentum,
  `dhat` recomputed from it, then `pos += dstep*dhat_NEW`. Same pattern for pions at `:857-858,863-864`.
- **ACHILLES**: `Cascade.cc:705` calls `Propagate` with the **pre**-interaction momentum, before
  `FinalizeMomentum` runs; `Particle.cc:13-27` uses `momentum.Theta()/Phi()` at call time. Outgoing
  particles are placed at that already-advanced point (`DeltaInteractions.cc:316-317`).
- **Effect**: up to ~2×step displacement error per wide-angle scatter; compounds over rescatters.
- **Sign**: not analytically obvious.
- **Status**: NOT FIXED.

### D6. Δ→Nπ decay angular distribution
- **ADoNIS**: isotropic.
- **ACHILLES**: `AngularMom: 2` → w(cosθ) ∝ (1+3cos²θ)/4.
- **Status**: NOT FIXED. Affects pion angular distribution → escape probability.

### D7. Pion birth-position coin
- **ADoNIS**: independent coin (`_ev_fold_uniform(sk,110)`) choosing slot a/b for the product pion.
- **ACHILLES**: a **single** 50/50 coin that co-locates the pion with its partner nucleon N2. Note the
  C++ dead-code trap: `for(auto &decay: decays_a) decay.Position()=...` runs *after*
  `decays_out.insert(...)`, so slot-a Δ products keep `particle1.Position()`.
- **Status**: PARTIAL fix uncommitted in the working tree. **NOT INERT — it closes most of the C
  deficit.** First result (¹²C, 71 chunks, 1.775M events, 2026-07-22):

  | bin [MeV/c] | POST-FIX | pre/ACH | **post/ACH** | pull |
  |---|---|---|---|---|
  | [800,1000)  | 0.00094 ± 0.00005 | 0.969 | 0.955 ± 0.055 | −0.8σ |
  | [1000,1200) | 0.00673 ± 0.00014 | 0.991 | 1.038 ± 0.022 | +1.7σ |
  | [1200,1400) | 0.01724 ± 0.00023 | 0.963 | 0.987 ± 0.013 | −1.0σ |

  Aggregate over the three bins: **0.971 → 0.9993 ± 0.0111**, a 2.6σ shift. No bin is now
  individually significant. Note this is the *partially* correct fix (independent coin, not
  ACHILLES's single coin co-locating the pion with partner nucleon N2), so the remaining
  correction may move it further.

  **⁴⁰Ar** (76 chunks, 1.900M events — near-final statistics):

  | bin [MeV/c] | POST-FIX | pre/ACH | **post/ACH** | pull |
  |---|---|---|---|---|
  | [800,1000)  | 0.00209 ± 0.00008 | 0.945 | 0.965 ± 0.036 | −1.0σ |
  | [1000,1200) | 0.01375 ± 0.00020 | 0.955 | 0.989 ± 0.014 | −0.8σ |
  | [1200,1400) | 0.03534 ± 0.00031 | 0.955 | 0.971 ± 0.009 | **−3.4σ** |

  Aggregate: **0.9546 → 0.9752 ± 0.0073**. The improvement is real, but Ar retains a **2.5% deficit at
  3.4σ**, concentrated in the highest-momentum bin — unlike C, which closed to 0.9993 ± 0.0111.

  **Interpretation**: the birth-position fix is real and helps both targets, but leaves an
  **A-dependent residual**. C closed; Ar did not. This points the remaining search at mechanisms that
  scale with nucleus size / number of rescatters — primarily **D1** (`& outward` escape gate: more
  surface re-entry chances in a bigger nucleus), **D10** (missing config Euler rotation), and
  **D5 on Ar specifically** (still open, see below).

### D8. Δ⁺/Δ⁰ → Nγ radiative branch (BR 0.55%) missing in ADoNIS
- **Status**: NOT FIXED. Small; would *reduce* pions if added, i.e. wrong sign.

### D10. Per-draw Euler rotation of nucleon configurations
- **ACHILLES**: `Configuration.cc:71-94` applies a random Euler rotation to each drawn config.
- **ADoNIS**: not applied. Relevant here because the beam is a **fixed-z** beam, so an unrotated
  config library is not averaged over orientation by the beam geometry.
- **Status**: NOT FIXED.

---

## OPEN — challenged by the user, under test

Ablation job **32744294** (4 variants × {C, Ar}, N=20k, seed 0, `abl_caps.py`) toggles each knob in
isolation. `base` reproduces `beam_bank.py:114` exactly.

### D4. `path_budget_R = 3.0` — silent particle kill with no ACHILLES counterpart
- **ADoNIS**: `cascade_full.py:253` — `al_2 &= (_lpath_2 < cfg.path_budget_R * radius)`. A particle
  exceeding the budget is dropped with **no terminal flag**: not escaped, not absorbed, never enters the
  output buffer `O`. Uncounted.
- **ACHILLES**: no path-length budget anywhere. Only bound is `cMaxSteps=100000`, and reaching it
  **throws** (`Validate()`, `Cascade.cc:210-220`) rather than silently dropping.
- **Budget size**: `radius` here is the density-tail radius, **not** the card's `radius: 10`. So the
  budget is ≈19.7 fm (¹²C, R≈6.57) and ≈27.3 fm (⁴⁰Ar, R≈9.1).
- **Prior**: genuinely uncertain. Escaping needs ~R of net displacement, but a multiply-rescattering
  pion random-walks; 19.7 fm is ~4–8 mean free paths. Rare, not obviously zero. Naturally worse for Ar.
- **Test**: `pbudget` variant sets `path_budget_R=100`. Same seed → if the fingerprints
  (`fp_alive`/`fp_nsc`/`fp_pi`) are **identical**, the mechanism never fires and D4 is discarded.
- **RESULT** (¹²C, N=20k, seed 0, job 32746309):

  | fingerprint | `base` (budget=3.0) | `pbudget` (budget=100) | Δ |
  |---|---|---|---|
  | `fp_alive` | 22627 | 22629 | **+2** (0.009%) |
  | `fp_nsc` | 2470 | 2477 | +7 |
  | `reacted` | 1658 | 1660 | +2 |
  | `fp_pi` | 90 | 90 | **0** |
  | frac_pi, all 3 bins | 3/3576, 23/3641, 64/3675 | *identical* | **0** |

- **VERDICT: the mechanism DOES fire, but is negligible — and it touches ZERO pions.** 2 particles in
  22 627 (0.009%); every pion observable is bit-identical. It cannot contribute to a 4–5% π deficit.
  **Discarded as a cause of the symptom.** It remains a genuine code divergence (silent, uncounted kill
  with no ACHILLES counterpart) and should still be removed for faithfulness, but at zero priority.
  *The user's prior was correct in practice.* ⁴⁰Ar confirmation pending (job 32749315).

### D5. `beam_bank` runs `M=12` with `q_cap=0` (drop-on-overflow)
- **ADoNIS**: `beam_bank.py:113-115` calls `run_cascade_pool` **directly** with `M=12` and no `q_cap`
  → falls to the `q_cap=0` default (`cascade_full.py:547`) = legacy drop-on-overflow. The returned
  `_sofl`/`_oofl` counters are **discarded**, while `rofl` on the same line *is* asserted.
- Every other consumer (`cascade_nucleus`, `run_fsi`, `gen_cc_engine_rich`) resolves to
  `P=16, q_cap=64` via `run_cascade_nucleus` (`cascade_full.py:918`). `beam_bank` was never migrated.
- **Prior evidence, and why it may not transfer**: `docs/TODO.md:26-29` and
  `cascade_refill_audit.md:27-28` measured **0.66% stack overflow on Ar at P=12** — but for **RES
  neutrino events**, which inject many primaries. A tagged beam injects **one** particle, so peak
  occupancy should be far lower. The 0.66% figure likely does **not** transfer.
- A separate agent measured `sofl+oofl = 0/2608` for Ar RES at `P=16, q_cap=64` — a *different*
  configuration; it does not bear on `M=12, q_cap=0`.
- **Test**: `base` prints `sofl`/`oofl`/`occ_max`/`occ_mean` directly; `cap` variant reruns at
  `M=16, q_cap=64`. If `sofl==0` and fingerprints match, D5 is discarded.
- **RESULT** (¹²C, N=20k, seed 0, job 32746309): `base` reports
  **`sofl = 0`, `oofl = 0`, `occ_max = 10`, `occ_mean = 1.131`.**
- **VERDICT: DISCARDED.** Peak stack occupancy is 10 against `M=12` — the stack never overflows, so
  `q_cap=0` has nothing to drop. The 0.66% figure from `docs/TODO.md:26-29` came from **RES neutrino
  events** (many injected primaries) and does **not** transfer to a single-particle tagged beam, exactly
  as the user argued. `M=12` is adequate here. *The user was right; I over-weighted a number measured
  on a different configuration.*
- **¹²C fully confirmed**: the `cap` variant (`M=16, q_cap=64`) returns fingerprints **bit-identical**
  to `base` (`fp_alive` 22627, `fp_nsc` 2470, `fp_pi` 90, all three frac_pi bins identical). With no
  overflow there is nothing for the queue to change. **D5 closed for ¹²C.**
- **⁴⁰Ar also confirmed — D5 FULLY CLOSED.** `base`/Ar (`M=12, q_cap=0`) reports
  **`sofl = 0`, `oofl = 0`**, and its fingerprints (`fp_alive` 29096, `fp_nsc` 8641, `fp_pi` 210, all
  three frac_pi bins) are **bit-identical** to `cap`/Ar (`M=16, q_cap=64`). `M=12` never overflows on
  either target.
- **Recorded false alarm**: an intermediate reading of `occ_max = 14 > M = 12` on Ar was briefly
  escalated as evidence that `base`/Ar might drop particles. It was wrong: `occ_max` in `abl_caps.py`
  is `O["alive"].sum(axis=1).max()` — **output-buffer** occupancy against `M_out=24`, not peak *stack*
  occupancy. It has no bearing on stack overflow. Lesson: state which buffer a counter measures before
  drawing a conclusion from it.
- **D5 closed on both targets. The user's prior was correct throughout.**

### D9. `time_step=False` (distance-sync) vs ACHILLES's always-on `AdaptiveStep`
- **ACHILLES**: `Cascade.cc:349,655-662` — `AdaptiveStep` (β_max time-sync) called unconditionally;
  nothing in this run card disables it.
- **ADoNIS**: `DiscreteCascadeConfig.time_step` defaults `False`; `beam_bank` does not override.
- **Prior**: a 1M-event study (`docs/logbook/cascade_subcascade_sequencing.md`) found this a ~1–2%
  wash on forward observables. That study did **not** target π production specifically, which is the
  weakness in calling it settled.
- **Test**: `tstep` variant. Not a deterministic A/B (stepping changes everything), so compare
  `frac_pi` statistically.
- **RESULT** (¹²C, N=20k, seed 0): `tstep` gives `fp_alive` 22595 / `fp_nsc` 2447 / `fp_pi` **91**
  against `base`'s 22627 / 2470 / **90**. Summed over the three bins:
  **ratio tstep/base = 1.011 ± 0.150 (0.07σ)**.
- **VERDICT: NO EVIDENCE OF AN EFFECT — but the test is UNDERPOWERED and does not settle D9.**
  At N=20k there are only ~90 produced pions on ¹²C, i.e. **10.5% 1σ precision** on the pion count.
  This can exclude effects larger than ~21% (2σ); it has **no power at all** to resolve the 2–5% scale
  we care about. Reporting "consistent with no effect" here would be misleading without this caveat.
  Settling D9 at the 2% level needs ~100× the statistics (a full ~2M-event bank pair), which has not
  been run. ⁴⁰Ar `tstep` still running.
- **Status**: NOT SETTLED (no evidence, insufficient sensitivity). Prior 1M-event study still the best
  available evidence that it is a ~1–2% wash. *The user's prior is plausible but not yet confirmed.*

---

## RESOLVED

### S1. 1.0 MeV/c spawn floor on created particles — **DISCARDED**
- **Claim**: `cascade_discrete.py:567` discards any spawned particle with lab |p| < 1.0 MeV/c —
  including the NN→NΔ→NNπ product pion — without decrementing the reaction count. Symptom shape
  matched exactly (π numerator down, reaction denominator untouched).
- **Evidence against**: histogrammed all final-state pions in three ACHILLES
  `run_cascade_prot_C_*.hepmc` files (2047 pions). Lowest populated bin is **[10,20) MeV/c with 1
  pion (0.05%)**; bins [0,1), [1,2), [2,5), [5,10) are **all empty**. ACHILLES produces no
  sub-MeV final-state pions, so a 1 MeV/c floor removes an empty population.
- **Verdict**: cannot account for any part of a 4–5% deficit. Closed.

### S2. `M_out=24` output-buffer overflow — **DISCARDED**
- Occupancy 1.1–1.6 of 24, max 16, fraction ≥23 = 0.0000.

### S3. `max_steps=2000` truncation — **DISCARDED**
- Inert on the `pending=None` path; the real bound is `_HARD_STEPS=100000`, matching ACHILLES's
  `cMaxSteps`, and it *raises* rather than truncating (`cascade_full.py:685-688`).

### S4. Conversion sink — **DISCARDED**
- σ_conv ≈ 0 below W≈1700; measured conversion branch fraction 0.0000.

### S5. Formation zone — **DISCARDED**
- ACHILLES exempts pions; ADoNIS matches. Decrement/reset cadence verified identical
  (`Particle.cc:9-11,25`, `Cascade.cc:974-975`).

### S6. Inelastic 2-nucleon Pauli check — **DISCARDED**
- Under `ResonanceMode: Decay`, `NucleonNucleon::GenerateMomentum` returns already-decayed {N,N,π} and
  `PauliBlocking` is applied to all `particles_out`, so both nucleons are checked on both sides.
  An earlier "ACHILLES checks only one nucleon" claim came from misreading `Cascade.cc:232`, which is
  the Δ *escape* force-decay path, not the vertex. The corresponding code change was reverted.

### S7. Nucleus / density / k_F / Pauli layer — **DISCARDED**
- Density and QMC config files byte-identical on disk; Ar cutoff effect <0.01% of nucleons;
  local Fermi gas on both sides.

### S8. σ_reaction normalization (`sigma_of_p` sub-range edges) — **NOT TRIPPED**
- `achilles_beam.py:90` normalizes as `n_tried * diff(edges)/(edges[-1]-edges[0])`, valid **only** for
  full-range edges. `make_figs.py:35` builds edges from the ADoNIS manifest `pmin/pmax`, which match the
  ACHILLES cards (`[300,1400]` p/n, `[50,1000]` π⁺) exactly. Safe today, **not enforced in code** —
  landmine for any future re-run with zoomed bins. This bug class has already bitten twice.

### S9. Observable definitions (`reacted`, `has_pi`) — **MATCH**
- Both sides require survival-to-escape, exclude η, include π⁰. ACHILLES beam particle correctly
  identified by `status==29 && px==py==0` (verified against raw hepmc bytes).

### S10. Legacy `cascade_real.py` σ divergences (missing 5/6 isospin split, proton-density k_F for both
  species) — **NOT APPLICABLE**. Dead code; the pool engine is the sole production path.

---

## Fixed and committed

| Item | Commit |
|---|---|
| Mass conventions (avg vs physical) — 6+ real bugs, plus guardrail test + `constants_audit.md` | — |
| Deviation 1 (η) | — |
| Deviation 2: beam z-plane escape via explicit `external_test` flag | `4ad217a`, `3dfabd8` |
| FSI `rec_caps` scaling by `ceil(A/12)` | — |

Deviation 2 result: reacted rate 0.913→1.021; χ²/ndf prot reaction 4.07→0.86, neut 2.17→0.40.

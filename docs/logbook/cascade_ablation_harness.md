# Cascade ablation harness — hybrid ACHILLES↔ADoNIS transport comparison

## Goal
Run ADoNIS's pool cascade on ACHILLES's **exact per-event input nucleus** (propagating primary +
full background config) and compare final states + per-interaction traces, isolating cascade-**transport**
implementation deviations from input-generation differences. Motivation: the QE FSI proton/neutron
tension — mean N(p) 1.007 (exact) but mean N(n) 1.121 (12% low), traced to ~9% fewer secondary nucleon
knockouts overall + slight proton-bias in isospin (n-fraction 0.622 vs 0.635). Code-reading found the
mechanism faithfully ported (same GiBUU isospin-resolved σ, same Gaussian exp(−πb²/σ) selection, same
formation-zone formula E/|m²−p₁·p₂|·ℏc, same any-outgoing<k_F Pauli, same QMC background). So the ~9%
is quantitative — needs an ablation to localize.

User decision (2026-06-24): build the harness with a **per-interaction trace** (not just input→output
endpoints), so we compare scatter-by-scatter. Precedent: an early-project ablation caught an obscure
implementation diff invisible to code-reading.

## Design (4 parts)
1. **ACHILLES CASCADEDUMP (C++ fork)** — behind `ACHILLES_CASCADEDUMP`, in `Cascade::Evolve`/
   `FinalizeMomentum`, per event dump:
   - INPUT: event id; propagating primary nucleon(s) p4+pos+pid; EVERY background nucleon p4+pos+isospin
     (struck already removed → for QE the struck neutron is the propagating proton).
   - TRACE: each NN interaction — target bg index, both isospins, in/out p4s, Pauli pass/fail,
     elastic/inelastic. (Hook near existing NNDUMP/NSCATR, Cascade.cc ~753-942.)
   - OUTPUT: final-state nucleons p4+pid.
   - Push to remote `fork` (NEVER origin); rebuild native arm64 `achilles:fullcascade`.
2. **Parser** (`adonis/data/oracle/parse_cascadedump.py`) → per-event arrays: prim_p4/pos, npos/nmom/nisp,
   consumed0, ACHILLES trace, ACHILLES final-state nucleons.
3. **ADoNIS ingestion** — `cascade_nucleus(..., su_external=)` bypasses `setup_nucleus` (cascade_full.py:631,
   the single injection point) and builds `su` from ACHILLES arrays. Pool already supports per-event bg.
4. **Harness** (`scripts/cascade_ablation.py`) — run ADoNIS pool on ACHILLES's identical inputs; compare
   N(p)/N(n) + momenta endpoint-wise AND trace statistics (scatters/event, target-isospin dist, Pauli
   reject rate, per-scatter energy transfer). Toggle ADoNIS Pauli/formation/σ (↔ ACHILLES NO_PAULI build)
   to bisect the ~9%.

## Validation ladder
- Round-trip identity: feed ADoNIS's OWN sampled config through `su_external` → bit-identical to the
  normal path (gate the ingestion before involving ACHILLES).
- Input check: ADoNIS-on-ACHILLES-input primary momentum == ACHILLES primary (parser sanity).
- Localization: the trace stat that diverges (scatter count / Pauli reject / isospin) names the mechanism.

## Log
- 2026-06-24: plan set; building ADoNIS `su_external` first (self-contained, round-trip gateable).
- 2026-06-24: harness complete + round-trip bit-exact. ACHILLES `ACHILLES_CASCADEDUMP` (Cascade.cc/hh)
  dumps per-event input nucleus + per-NN-scatter trace + final state; built `achilles:cascadedump`.
  **ROOT CAUSE FOUND — interaction-probability model mismatch.** The production oracles were generated
  with `Probability: Cylinder`; ADoNIS is Gaussian-only. So all prior QE ADoNIS-vs-ACHILLES multiplicity
  comparisons were Gaussian-vs-Cylinder.
  - Ablation (ADoNIS pool on ACHILLES's EXACT 50k Gaussian input nuclei): mean N(n) ACH/ADO 1.033,
    secondary nucleons 0.993 — transport faithful to <1% (Pauli-block frac 0.62, 1.11 NN attempts/evt).
  - Fresh Gaussian-vs-Gaussian, independent input, 1M each side: total σ 0.994; mean N(n) ACH/ADO ~1.014
    (was 1.121 vs Cylinder); dσ/dp(p) χ²/ndf 1.7; dσ/dp(n) χ²/ndf 5.5 (was 16); near-k_F neutron deficit
    closed. QE clean to ~1%.
  - Action taken: ALL run cards switched Cylinder→Gaussian (Achilles + ADoNIS, grep-verified 0 remain).
    Cylinder-era output (33G hepmc+banks) deleted; regenerated 1M QE Gaussian both sides.
  - Residuals (≪1% of σ): N(π⁺)=1 QE ADoNIS ~1.5× (FSI NN→NΔ→NNπ rate); neutron dσ/dp >700 MeV low.
  - Harness reusable: `scripts/cascade_ablation.py <dump> C [--no-pauli]`; dump via `ACHILLES_CASCADEDUMP=1`.

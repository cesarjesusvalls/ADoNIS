# CC1pi cascade tension: ADoNIS vs ACHILLES per-bracket fate study (C, Gaussian)

Goal: localize the CC1pi ADoNIS-vs-ACHILLES matrix tension in the cascade, in detailed initial-pion-|p|
brackets, significance-weighted (chi2/pull, NOT ratio).

## Method
- ADoNIS: FULL pooled cascade (scripts/cascade_debug_dump.py) on C RES, ~555k pi+ events, LARGE buffers
  P=16, M_out=48, N_RECOIL=8, max_steps=1000 (march 40 fm).  **overflow stack=0 out=0 throughout** — buffers
  are NOT a factor.  Per event: primary-pion init |p|, full final-state pion multiplicity npip/npi0/npim,
  proton leg.
- ACHILLES: achilles:fatepion2 (ACHILLES_FATEDUMP=1) on the Gaussian C fate cards s1-s4 = 2.0M events,
  549,422 RES primary-pion events.  Per event: primary-pion FATE pin + FATE_FS npip/npi0/npim.
- Both sides classified IDENTICALLY from final-state pion multiplicity: survived-pi+ = npip>=1;
  absorbed = no final pion; charge-ex = no pi+ but >=1 pi0/pi-.  Comparison scripts/cascade_debug_compare.py.

## RESULT (decisive trend, huge significance; ndf=10 brackets per channel)
| channel        | chi2/ndf | trend |
|----------------|---------:|-------|
| **absorbed**   | **2.78** | ADoNIS == ACHILLES across ALL brackets (pulls mostly <2sigma) — absorption CORRECT |
| survived-pi+   | 657      | ADoNIS systematically HIGH, +10 to +37 sigma every bracket |
| charge-exchange| 1591     | ADoNIS systematically LOW, -15 to -61 sigma every bracket |

The survival-excess and charge-ex-deficit are COMPLEMENTARY per bracket (absorption matches), e.g.
[0,100): surv +0.219 / cex -0.209; [500,700): surv +0.194 / cex -0.187.  Final-state proton multiplicity
<np> matches ACHILLES to ~1% (only the [700,2000) bracket ~10% off).

ACHILLES charge-ex is ~flat 0.19-0.30 vs |p_pi|; ADoNIS charge-ex rises 0.012 (|p|<100) -> 0.12 (Delta
region ~250-350) -> 0.03 (high p).  So ADoNIS has almost NO low-energy charge-exchange and too little
everywhere; the deficit is WORST at low pion momentum (~18x too low) and least bad near the Delta.

## Hypothesis
The cascade pion-SCATTER charge-channel partition is the culprit: absorption (Oset) and kinematics/proton
leg are correct, but the pi+ that scatters stays pi+ instead of charge-exchanging to pi0.  In
`adonis/fsi/cascade_discrete._pion_step` the outgoing pion charge is sampled from
`sig_io = cascade_mb.jax_channel_sigmas_resolved(...)` (3 outgoing-charge channels) via
`probs = sig_io_j/sum; out_ch = ...`.  The charge-exchange channel sigma appears far too low relative to
elastic, especially at low energy (suggesting the low-energy/s-wave charge-exchange is missing or
mis-weighted while the Delta-region piece is partially present).  NEXT: read cascade_mb.jax_channel_sigmas
+ the ACHILLES MesonBaryonInteraction charge channels to pin whether the channel sigmas or the isospin
partition are wrong.  (Held as hypothesis — absorption-correct + complementary surv/cex is strong but the
exact sigma source must be confirmed in code.)

## Refinement 1 (code-read + cross-checks): culprit narrowed, candidates eliminated
- **Not a counting bug.** ADoNIS dump cross-tab: escaped primaries = 89% stay pi+, 10.1% pi0, 0.6% pi-;
  event-cex 0.085 == escaped-primary-cex 0.085; of cex'd primaries only 0.3% are mislabeled npip>=1.
  So the low cex is REAL, not a charge-recording artifact. Absorption 0.206 (== ACHILLES ~0.20).
- **Selection algorithm is faithfully mirrored.** ACHILLES `Cascade::Interacted` (Cascade.cc) iterates
  nucleons NEAREST-FIRST and returns the first whose `probability(b2, xsec/10)=exp(-pi b2/(sigma/10))`
  roll passes -- i.e. "closest passer", IDENTICAL to ADoNIS `_pion_step` (`passes=u<prob; j=argmin perp2`).
  Both use the charge-resolved per-nucleon sigma. So the partner-selection is NOT the divergence.
- **Channel sigmas have cex.** cascade_mb: on a NEUTRON, pi+n->pi0p (cex) sigma DOMINATES elastic
  (cex fraction 0.45-0.80 over W); pi+p is pure elastic (198 mb at Delta). sigma(pi+p)/sigma(pi+n)~2.9
  (correct Delta isospin). Pure sigma-weighting => struck-neutron fraction ~0.26 => per-scatter cex
  ~0.26*0.68 ~ 0.17 (~ACHILLES 0.20).  But ADoNIS event-cex is 0.085 => inferred struck-neutron
  fraction ~0.16 < 0.26.  ACHILLES (cex 0.20) => inferred ~0.37 > 0.26.
- So ADoNIS under-charge-exchanges and ACHILLES OVER-cex's vs naive single-scatter sigma-weighting.
  The lever that reconciles both is MULTI-SCATTER: more reactions per pion => more neutron-scatter
  chances => more cex.  ACHILLES pi+ overall reaction (status==29) = 0.479 (Delta-peaked: 0.13 at
  |p|<100 -> 0.68 at 250-300 -> 0.24 at high p).  HYPOTHESIS (refined): ADoNIS pions react/scatter
  FEWER times than ACHILLES, so they cex less (each elastic-on-proton keeps pi+; cex needs a neutron
  hit, which is rarer per-scatter and ADoNIS gets fewer scatters).  Test: compare per-bracket pion
  SCATTER MULTIPLICITY / reaction fraction (ADoNIS nsc vs ACHILLES status).  [ADoNIS pion nsc added to
  cascade_debug_dump for this test.]  Alternative still open: pi0 re-conversion asymmetry.

## Artifacts
- ADoNIS dump: /tmp/ado_cascade_debug_C.npz ; ACHILLES: /tmp/ach_fatepion_C_gauss.fate
- scripts/cascade_debug_dump.py (full-pool momentum-bracketed dump) + scripts/cascade_debug_compare.py (chi2/pull)

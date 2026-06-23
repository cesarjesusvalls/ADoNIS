# CC1pi cascade tension: ADoNIS vs ACHILLES per-bracket fate study (C, Gaussian)

Goal: localize the CC1pi ADoNIS-vs-ACHILLES matrix tension in the cascade, in detailed initial-pion-|p|
brackets, significance-weighted (chi2/pull, NOT ratio).

## !!! CORRECTION / RETRACTION (supersedes everything below) !!!
The "pion charge-exchange ~2.4x too low" trend reported below was a **comparison-script bug, NOT a
physics divergence**.  `cascade_debug_compare.py` set the per-bracket momentum from ANY pion FATE line
(`pidin in {211,111,-211}`); RES also produces pi0/pi- PRIMARIES (which charge-exchange readily), so those
events were folded into the "ACHILLES" sample and compared against ADoNIS's pi+-ONLY dump -> a spurious
cex excess on the ACHILLES side -> fake cex deficit on ADoNIS.
**Fix:** bracket ACHILLES by a pi+ primary only (`pidin==211`), matching the ADoNIS dump.
**Corrected result (apples-to-apples, pi+ primary both sides):** the cascade pion fate MATCHES ACHILLES:
survived-pi+ chi2/ndf=1.86, charge-exchange chi2/ndf=1.22, absorbed chi2/ndf=2.47 (pulls within ~+-2sigma).
Integrated multiplicities are identical: ADoNIS/ACHILLES <npip>=0.712/0.711, <npi0>=0.083/0.084,
<npim>=0.006/0.007, absorption 0.205/0.205.  **There is NO pion charge-exchange divergence.**
Scripts (compare/plot/complementarity) fixed to pi+-primary bracketing.  Corrected figure:
paper_figures/cc1pi_cascade_fate_corrected.png.  Lesson: a "divergence with ACHILLES" must first be
checked against the comparison code itself (apples-to-apples) before being read as physics --
exactly the validation-not-diagnostic discipline.

## --- ORIGINAL (WRONG) ANALYSIS BELOW, kept for the record; superseded by the correction above ---

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

## Refinement 2: scatter-multiplicity ruled out; narrowed to per-scatter charge outcome (NOT root-caused)
- **Reaction/transmission rate MATCHES** ACHILLES per bracket (ADoNIS transmitted 0.525 vs ACHILLES
  status==1 0.521; e.g. Delta-region 0.316 vs 0.316).  So pions interact equally often -> NOT a
  multiplicity/scatter-rate issue.  Mean scatter count among escaped-scatterers ~1.33.
- Combined with absorption matching, the SCATTER total sigma matches too.  Reduced to: among
  non-absorption reactions (scatters), ACHILLES converts pi+ -> pi0 ~72%, ADoNIS ~31%.
- **ADoNIS event-cex is 99% PRIMARY charge-exchange** (0.0855 of 0.0864); secondary-pi0 production
  (primary absorbed + NN->NDelta->Npi0) is negligible (0.0009).  <npi0>/event = 0.084.
- ADoNIS per-(escaped-scattered) cex = 0.316 -> implied per-single-scatter cex ~0.25 (ABOVE the naive
  sigma-weighted neutron expectation 0.17 -> ADoNIS is NOT over-selecting protons).
- **Verified CORRECT/faithful in ADoNIS** (none is the bug): channel table is the FULL piN isospin set
  (pi0 HAS elastic channels pi0p->pi0p / pi0n->pi0n, lines 26/28); sigma(pi+p)/sigma(pi+n)=2.9 and
  cex/elastic on neutron ~2 both MATCH piN data; selection algorithm == ACHILLES Cascade::Interacted
  (nearest-first first-passer); charge counting (no mislabel); reaction rate; absorption; buffers (0/0).
- **OPEN PUZZLE (not resolved):** ACHILLES ~72% cex-among-scatters is hard to reconcile with piN isospin
  (pi+p has NO cex; even 50% neutron x 100% cex gives 0.50).  So either (a) a subtlety in the ACHILLES
  FATE_FS cex definition I'm parsing, or (b) ACHILLES's in-medium piN charge outcome genuinely differs
  from the isospin-pure ANL-Osaka table ADoNIS uses.  The decisive measurement -- ACHILLES sigma(pi+p)
  vs sigma(pi+n) -- is BLOCKED: GETXSEC dumps total sigma + nucleon pid but NOT the incoming pion charge,
  so the p/n ratio (1.62 mixed) is contaminated by pi-/pi0 (sigma(pi-n)~sigma(pi+p)).

## STATUS: tension precisely LOCALIZED, not root-caused
The CC1pi cascade tension is the pion CHARGE-EXCHANGE yield (ADoNIS too low by ~2.4x at event level);
absorption, reaction rate, proton leg, channel sigmas-vs-data, selection algorithm, and buffers are all
verified correct.  Held as an open, well-bounded discrepancy (NOT a declared root cause).

## Recommended next steps (need user direction; any cascade-physics change must be discussed per CLAUDE.md)
1. Add an incoming-pion-charge tag to ACHILLES GETXSEC (1-line) -> cleanly measure ACHILLES sigma(pi+p)
   vs sigma(pi+n) and compare to ADoNIS 2.9 (tests the species-sigma-split / proton-over-selection).
2. Instrument per-scatter (struck species + in/out pion charge) on BOTH sides for a direct
   cex-per-scatter comparison (ADoNIS: expose nisp[j]+out_ch from _pion_step; ACHILLES: FinalizeMomentum).
3. Re-derive the ACHILLES FATE_FS-based cex definition to rule out a parsing/bookkeeping subtlety.

## Artifacts
- ADoNIS dump: /tmp/ado_cascade_debug_C.npz ; ACHILLES: /tmp/ach_fatepion_C_gauss.fate
- scripts/cascade_debug_dump.py (full-pool momentum-bracketed dump) + scripts/cascade_debug_compare.py (chi2/pull)

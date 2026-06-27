# Differentiable knobs — plan, registry, and validation

## Goal
Expose **every physics parameter that affects the pre-FSI and post-FSI prediction** as a differentiable
tuning knob θ, fitted via the blueprint's exact-**reweight-of-a-frozen-walk** design (θ-independent walk +
per-event reweight). Add the **"easy" (reweightable) knobs first**, each with an **individual closure test**.
Defer the deterministic-threshold knobs (cascade k_F, recapture, formation zone) to a soft-relaxation
extension (last section).

Today only 3 knobs are exposed: `sabs`, `sscat`, `M_A` (analysis/t2k/differentiability/{tune,closure}.py).
This plan extends that to ~20–25 reweightable knobs.

## Organizing principle: reweightable (easy) vs walk-changing (hard)
The frozen-walk reweight works **iff the parameter sets the probability/density of a SAMPLED quantity**
(discrete branch or continuous draw) — then reweight by the likelihood/density ratio at the *recorded*
sampled value. It does NOT matter that two branches lead to different downstream kinematics: both branches
are realized across the ensemble; the ratio just shifts their relative weights.

The dividing line is **sampled-probability/density (easy)** vs **deterministic threshold/geometry (hard)** —
NOT "initial vs final state" and NOT "affects downstream".

Four reweight sub-mechanisms (all "easy"):
- **R1 σ-scale** — scales an interaction cross section; the hit is a Bernoulli with smooth prob
  `exp(−π b²/σ)`. Likelihood ratio over the recorded per-step (b², σ-components). [kind-1 FSI record:
  cascade_full.py:`pool_fsi_reweight` (385), cascade_discrete.py:`pion_branch_reweight` (88) /
  `nucleon_scat_reweight` (106)]
- **R2 amps² decomposition** — enters the hard amplitude polynomially; exact per-event
  `w(θ)=(a+b·r+c·r²)/(a+b+c)` from a 3-point amplitude eval. [amps² record, pattern:
  adonis/analysis/ma_records.py:`ma_reweight` (60), form_factors.py:`axial_reweight_dipole` (102)]
- **R3 density ratio** — parameterizes a continuous sampling density `p(x;θ)` (spectral function p/E,
  Δ Breit–Wigner mass); `w = p(x;θ)/p(x;θ₀)` at the recorded x.
- **R4 branch fraction** — a sampled discrete branch (NN CEX swap, π-abs isospin partition, Δ-decay CG);
  `w = f_new/f_old` for the realized branch.

**Hard (deferred):** the parameter is a DETERMINISTIC threshold/geometry (`1[|p|<k_F]`, recapture cut,
formation-zone formula, density grid) applied after sampling — no smooth density to ratio. Needs a soft
relaxation to become reweightable+differentiable (see last section).

## Validation ladder (run for EVERY knob)
1. **Nominal identity** — `w(θ₀) == 1` bitwise (reweight is exactly 1 at nominal; the bit-exact gate).
2. **reweight == in-walk** — a forward generation AT θ equals the nominal walk reweighted to θ (for R1/R4
   where a forward-at-θ generation is meaningful).
3. **autodiff == FD** — `dO/dθ` from autodiff matches finite difference (per-knob).
4. **injected-value closure** — fit recovers a θ* injected into pseudo-data (independent walks), within
   Hessian errors; no clip-rail pathology.

## Knob registry
Nominal = the value at which the reweight is identically 1. "EXISTS" = already implemented.
(File:line for already-verified items are exact; survey-sourced locations marked "≈" — verify at impl.)

### Group A — FSI cross-section scales  (R1, kind-1 record + channel tag)
Split the single `sscat` into per-channel scales (each a likelihood ratio; extend the kind-1 record to tag
the channel per scatter).
| knob | scales | nominal | status |
|---|---|---|---|
| `s_pi_abs` | π absorption σ (Oset) | 1.0 | EXISTS (=sabs) |
| `s_piN_elastic` | πN elastic σ | 1.0 | split out of sscat |
| `s_piN_cex` | πN charge-exchange σ | 1.0 | split out of sscat |
| `s_NN_elastic` | NN elastic σ (opt. pp/pn/nn split) | 1.0 | split out of sscat |
| `s_NN_inelastic` | NN→NΔ→NNπ σ | 1.0 | split out of sscat |

### Group B — FSI branch fractions  (R4, branch-bit record)
| knob | branch | nominal | location |
|---|---|---|---|
| `f_NN_cex` | NN-elastic charge-exchange (id-swap) | 0.5 | cascade_discrete.py `_swap_cx` (fold 109) |
| `f_abs_isospin` | π-absorption p/n partition (Oset 5/6) | 5/6 | ≈cascade_discrete.py abs partition |
| `f_delta_decay` | Δ→Nπ Clebsch–Gordan split | CG | ≈cascade_discrete.py inelastic dch/pi_q |

### Group C — Δ resonance (cascade)  (R3 density ratio)
| knob | param | nominal | note |
|---|---|---|---|
| `M_delta` | Δ Breit–Wigner mass (NN→NΔ sampling) | 1232 MeV | record sampled Δ mass; w=BW(m;θ)/BW(m;θ₀) |
| `Gamma_delta` | Δ width | running | same record |

### Group D — Hard-vertex amplitude / form factors  (R2 amps² decomposition)
| knob | enters | nominal | status |
|---|---|---|---|
| `axial_MA` | QE+RES axial FF (dipole) | 1.0 GeV | EXISTS |
| `axial_strength` | overall axial current (linear) | 1.0 | EXISTS (code path; UNTESTED) |
| `vec_MV` / Kelly G_E,G_M | QE vector FF | Kelly | ≈primary/dcc/form_factors.py 36–42 |
| `mu_p`, `mu_n` | nucleon magnetic moments | 2.793 / −1.913 | ≈form_factors.py 37–38 |
| `F_P_scale` | pseudoscalar / pion-pole | PCAC | ≈assembly.py:48 debug hook → promote |
| `pw_norm[14]` | DCC per-partial-wave norm | 0 (off) | EXISTS (code path; UNTESTED) |
| `MA_res` / `C5A` | RES axial mass / N-Δ axial coupling | 1.0 | separate from QE M_A |
| `bkg_scale` | DCC non-resonant amplitude | 1.0 | ≈loader.py za-component |
| `vec_axial_split` | vector-vs-axial block scales | 1.0 | amps² blocks |

### Group E — Initial-state spectral function  (R3 density ratio)
Parametric deformation over the tabulated `S(|p|,E)` (adonis/xsec/spectral.py); record the sampled (p,E),
`w = S_θ(p,E)/S_0(p,E)`.
| knob | deformation | nominal |
|---|---|---|
| `kF_sf` | Fermi-momentum / |p|-axis scale | 1.0 |
| `Eb_shift` | binding/removal-energy shift | 0 |
| `sf_norm` | overall normalization | 1.0 |
| `src_tail` | high-|p| (SRC) tail scale | 1.0 |

### Group F — overall normalizations  (trivial R)
`qe_norm`, `res_norm`, `H_norm` — global σ multipliers (nominal 1.0); included for completeness.

## PROGRESS (2026-06-27)
- **Group D DONE (10 hard-vertex knobs, all closure-tested + committed):** QE `M_A` (pre-existing),
  `axial_strength`, `vector_strength`, `mu_p`(gmp), `mu_n`(gmn), `gep`, `gen`; RES `res_axial_strength`(C5A),
  `pw_norm[14]`, `pion_pole`(F_P).  Mechanism: amps2 quadratic in the scale -> 3-eval (a,b,c) records
  (adonis/analysis/ma_records.py) + `strength_reweight`/`ma_reweight`.  Hooks: `me_cross_section`
  vector_scale/ff_scale; `exclusive_amps2_batch` knobs=/pion_pole=.  Tests: test_strength_reweight.py,
  test_res_strength_reweight.py (+ all nominal bit-exact gates still pass).
  - `bkg_scale` DEFERRED: the DCC table sums bare+dressed+nonres at load (loader.py); needs the loader to
    keep the 3 components separate to scale the non-resonant one.  (axial_z/time, vec_cc_z are assembly
    DIAGNOSTICS, not physical knobs -> not exposed.)
- **Group A scoped (NEXT, core-record surgery):** the per-channel sigma split needs the kind-1 record
  (emitted in _pion_step/_nucleon_step + cascade_full make_pool_stepper with_rec + _rec_scatter) extended to
  carry per-channel sigma + the realized sub-channel.  In-engine the components ALREADY exist: pion `sio`
  (n,A,3 per-out-channel -> elastic=sio[ch_in], cex=sum others) and `sa` (abs); nucleon `sig_el(same_iso)`
  (pp/nn vs pn) + `sig_in`.  Granular reweight generalizes pion_branch_reweight/nucleon_scat_reweight; GATE
  with nominal-identity (granular==current sabs/sscat==1) + reweight==in-walk before trusting.
- **Group A PION side DONE (3 knobs):** `s_piN_elastic`, `s_piN_cex`, `s_conv` (+ `s_pi_abs`=sabs).
  Kind-1 pion record extended (branch 3->4 channels {el,cex,abs,conv} + ss_el); `fsi_pion_reweight`
  per-hit = s_realized*D0/D; `pool_fsi_reweight` backward-compatible + keyword granular knobs.  GATED:
  real-record nominal==1 (4e-16), forward identical w/wo records, autodiff==FD, 16/16 cascade tests.
  Commit ed2059b.  NOTE: granular reweight==in-walk for s_el!=s_cex is not forward-gated (the forward
  cascade uses nominal sigma); it is the exact analytic likelihood ratio, validated by nominal-identity +
  back-compat (which IS in-walk-gated) + autodiff==FD.
- Group A NUCLEON side NEXT (same pattern): split nucleon sigma into pp/pn/nn x elastic/inelastic.  Record
  needs per-candidate sig_el,sig_in + pair-iso + realized el/inel; reweight is the Gaussian hit/no-hit
  (a_nom = G/sigma_tot, sigma_tot(s)=s_el*sig_el+s_inel*sig_in) x el/inel sub-branch.
- Groups B/C/E/F still pending (B branch-fractions share the record surgery; E SF event-level; F norms).

## FINAL STATE THIS SESSION — 20 knobs added, validated, committed
- **Group D (10):** M_A*, axial_strength, vector_strength, mu_p, mu_n, gep, gen, res_axial_strength(C5A),
  pw_norm[14], pion_pole(F_P).  (* pre-existing.)  amps2 3-eval decomposition + strength/ma reweight.
- **Group A (9):** pion s_piN_elastic, s_piN_cex, s_conv (+ s_pi_abs=sabs) ; nucleon
  s_NN_elastic{pp,pn,nn}, s_NN_inelastic{pp,pn,nn}.  Core kind-1 record surgery; fsi_pion_reweight +
  fsi_nucleon_reweight; pool_fsi_reweight backward-compatible + granular keywords.  Real-record gates:
  nominal==1 (4e-16), forward identical with/without records, autodiff==FD (3e-8).
- **Group B (1):** f_NN_cex (NN-elastic charge-exchange fraction, nominal 0.5).
- Tests: test_strength_reweight.py, test_res_strength_reweight.py, test_pool_fsi_reweight.py (all green);
  cascade regression suite + all pre-existing bit-exact/oracle gates still pass.

## UPDATE 2026-06-28: Group E DONE + nothing-broke confirmed
- **Group E (4 knobs) DONE:** kF_sf, Eb_shift, sf_norm, src_tail (adonis/analysis/sf_reweight.py) -- JAX
  bilinear S(p,E) interp + density-ratio reweight on the recorded struck (|p|,removal).  Closure green
  (nominal identity, sf_norm multiplier, autodiff==FD x4, kF moves weights).  Commit 7a7197a.
- **NOTHING BROKE (verified):** full suite 86 passed / 4 skipped / 0 failed; and a full-config 1M QE bank
  regenerated with the NEW code is **byte-for-byte identical** to the committed reference (w,n_n,n_p,mu,
  struck all array_equal, maxabsdiff 0).  Every knob is a true nominal no-op on the forward prediction.
- **RUNNING TOTAL: 24 differentiable knobs** (D 10 + A 9 + B 1 + E 4), each gated, all committed.

## UPDATE 2026-06-28: Phase 4 — multi-mechanism JOINT CLOSURE via model_hist_full
The end-to-end differentiable predictor `full_knobs.model_hist_full(knobs, R, HV, SF, edges)` threads ALL
24 knobs into the T2K CC0pi dsigma/dx; `tests/test_full_knobs_grad.py` gates nominal==legacy model_hist +
autodiff==FD on real bins (7/7).  Capstone: a **joint closure across all three reweight mechanisms at once**
(`analysis/t2k/differentiability/closure_full.py`, mirrors closure.py) — perturb+recover
`M_A` (hard-vertex amps2, R2) + `sabs` (FSI kind-1, R1) + `kF_sf` (spectral-function density, R3); pseudo-data
from INDEPENDENT walks; two-replica covariant chi2; Hessian errors.  Reduced stats (CC0PI_N=12000, NREP=8,
NPSE=4).
- **ABSOLUTE norm (`--noA`, the blueprint default):** chi2/ndf 9.83 -> 0.54; M_A 1.090+/-0.106 (true 1.10,
  **-0.10 sig**), kF_sf 1.117+/-0.026 (true 1.10, +0.65 sig), sabs 0.627+/-0.440 (true 1.30, -1.53 sig;
  weakly constrained on dpt — large error, within 2 sig).  All three recovered within errors.
- **PROFILED A:** chi2/ndf 4.51 -> 0.62 but M_A **-4.70 sig** biased — profiling A absorbs M_A's QE-norm
  handle, leaving only its shape signature (degenerate with kF_sf).  This is exactly the blueprint
  normalization stance: fit ABSOLUTELY by default; profile a norm only with an explicit stated reason.
  The profiled-vs-absolute contrast (M_A -4.70 sig -> -0.10 sig) is the diagnostic, not a bug.
- Figures: output/figures/cc0pi_tune_closure_full_dpt{,_abs}.png; history npz in /tmp/adonis_tune_runs.
- NOTE on sabs: dpt has weak FSI sensitivity (the FSI knob error dwarfs the others); dat (delta_alphaT) is
  the FSI-sensitive observable — a dat closure would tighten sabs.

## REMAINING (need new infrastructure, NOT the record-extension pattern) — deliberately deferred
- **f_abs_isospin, f_delta_decay (Group B):** these are multi-component isospin CLEBSCH-GORDAN structure,
  not well-defined single-scalar tunables; their physical content is already captured by the per-channel
  sigma knobs (Group A).  Expose only if a specific multi-outcome parameterization is wanted.
- **M_delta, Gamma_delta (Group C):** the cascade Delta mass is sampled from the FULL dsigma/dm
  (nn_inelastic.dsigma_dm: relativistic BW x matrix element x phase space), not a bare BW.  Reweight needs
  recording the sampled m_d + the density ratio dsigma_dm(m_d;theta)/dsigma_dm(m_d;theta0) -> re-evaluate
  dsigma_dm at two pole masses.  Doable, but new (density-ratio) machinery.
- **Spectral function (Group E):** sf_norm + src_tail are trivial event-weight multipliers (no interp);
  kF_sf + Eb_shift need a JAX-DIFFERENTIABLE SF interpolation (current SpectralFunction is NumPy/detached) to
  evaluate S at scaled/shifted (p,E).  The differentiable interp is the real new piece.
- **bkg_scale (Group D):** needs the DCC loader to keep the 3 table components separate (currently summed).
- **Group F norms:** trivial global event-weight multipliers; fold in with the tune-theta integration.
- **cascade k_F:** deferred by design (soft-Pauli relaxation; see last section).

## Implementation phases (each ends green-tested + committed)
- **Phase 0 — scaffolding.** A single knob registry (`name → default → record → reweight_fn`) + a per-event
  record container; generalize the closure-test harness to be parameterized by knob (nominal-identity +
  autodiff==FD + injected-recovery). Reuse the PhysicsParams + ma_records patterns.
- **Phase 1 — Group A + B.** Extend the kind-1 record with channel/branch tags; per-channel σ reweights +
  branch-fraction reweights; closure test each knob.
- **Phase 2 — Group D.** Generalize `ma_records` to a per-knob amps² (a,b,c) decomposition for each
  vector/axial/pw/background/pion-pole knob; closure each.
- **Phase 3 — Group E + C.** SF deformation density-ratio reweights + Δ Breit–Wigner density ratio; record
  sampled (p,E) and Δ mass; closure each.
- **Phase 4 — Group F + integration.** Norm knobs; fold all knobs into the tune θ vector; joint closure.

## Future extension — cascade k_F (and the other hard thresholds)  [NOT in the easy set]
The cascade Pauli block is a **hard cut** `blocked = 1[|p_out| < k_F]` (deterministic; matches ACHILLES
`Cascade.cc`). It is NOT reweightable from a frozen walk: flipping k_F requires the **un-sampled branch**
(the rejected scatter, or the not-taken straight line), which the frozen walk does not contain, and the
likelihood of the realized outcome under a new k_F is a 0/1 indicator (no smooth density).

Difference vs the SF k_F: the SF k_F parameterizes a **smooth sampling density** `S(p)` (→ smooth ratio,
reweightable); the cascade k_F is a **hard accept/reject threshold** applied after sampling.

**Plan to expose it:** soften the block to a smooth blocking probability
`P_block(|p|;k_F) = σ_sig((k_F − |p|)/δ)` (Fermi/sigmoid of width δ). The block becomes a sampled Bernoulli
with a smooth probability → reweightable (R1/R4) AND pathwise-differentiable in k_F.
- **Caveat:** soft Pauli ≠ ACHILLES hard Pauli — a deliberate physics relaxation. Validate that δ→0
  reproduces the hard-cut prediction within tolerance; the gradient is the (correct) derivative of the
  relaxed model (→ distributional derivative of the hard cut as δ→0). Discuss before adoption.
- The same recipe makes the **recapture-KE threshold** (10 MeV `Escaped`) and **formation-zone gating**
  knobs, if wanted later.

## Notes / discipline
- Every reweight MUST satisfy `w(θ₀)==1` exactly (nominal-identity gate).
- R3 density ratios (SF, Δ BW) have importance-weight variance that grows in low-density tails → large
  pulls get noisy (sampling limit, not a correctness issue); keep pulls modest or resample.
- No fitted constants / no hardcoded ACHILLES values: knobs are **multipliers/shifts on the first-principles
  values threaded from source**, nominal-identity by construction. ([[no-hardcoded-achilles-constants]])
- The NN charge-exchange (Group B `f_NN_cex`) is the 50% id-swap added with the cascade fix
  ([[nn-elastic-charge-exchange]]); record its swap bit so `sscat`/`s_NN_elastic` reweights also flow
  through the charge-exchanged scatters.

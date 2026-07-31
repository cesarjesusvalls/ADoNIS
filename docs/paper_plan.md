# ADoNIS methodology paper — plan, status, and handoff

Working plan for the paper built on ADoNIS (the differentiable ACHILLES). This file is the single
place to track what the paper needs, what already exists, and what is still to do. Evidence trail for
the analysis lives in `docs/logbook/altgen_fakedata.md` (§1–19).

## Thesis (one paragraph)
A **differentiable** neutrino-interaction generator (ADoNIS: exact per-event reweights of a frozen
walk, autodiff through the cascade) lets you decide **from the data itself** which cross-section
parameters a sample can measure, and lets you **learn physics without being biased by model
incompleteness**. The core result is the **manifold taxonomy** of "unknown unknowns": mismodeling
that lies *inside* the model's reachable manifold vs *outside* it needs opposite defenses — a flexible
kinematic-habitat nuisance (inside) vs coherence gating + region excision (outside) — and the
regime is **precision-relative** (an unknown-unknown is only "outside" when its per-bin distortion is
resolvable above the errors).

## The five-part arc (from the project owner) — status

| # | Section | Status | Assets / pointers | To do |
|---|---|---|---|---|
| 1 | **Validation: ADoNIS reproduces ACHILLES** — the paper's non-NC figures (inclusive e,e'; free-nucleon & π-nucleus σ; e4ν; T2K/MINERvA/MicroBooNE TKI; DCC σ(W)) | ✅ figure-complete | `analysis/paper/sec1_validation/make.py` → `fig01_ee_domega`, `fig02_anl_sigma`, `fig03_pi_nucleus_sigma`, `fig456_e4nu`, `fig07/08/09`, `fig10_uboone_cc1p0pi`, `fig13_piN_sigma` | fold into the paper |
| 2 | **Compact "we have gradient information for all knobs"** | ✅ figure-complete | `analysis/paper/sec2_gradients/make.py` → `sec2_gradients_shape` (27×188 per-knob-normalized pull heatmap), `sec2_gradient_reach` | fold into the paper |
| 3 | **Fisher information per observable subset** → what is worth fitting | ✅ figure-complete | `analysis/paper/sec3_fisher/make.py` → `sec3_shrinkage_subsets`, `sec3_failure_modes`, `sec3_degeneracy`; engine = `physical_fit.py` Gate I with `PHYSFIT_OBS` | fold into the paper |
| 4 | **Closures** with Fisher-selected parameters | ✅ figure-complete | `physfit_fig7_closure5.png` (5-param, ≤0.5σ), 7-param on the 9-obs suite (`p9_closure.npz`, ≤0.4σ); `scripts/altgen/physfit_closure5_fig.py` | fold into the paper; optionally a fluctuated closure (null calibration) |
| 5 | **Fitting data not described by the model** (unknown unknowns) | ✅ figure-complete | `physfit_fig1–10`, `output/reports/physical_fit_report.pdf`; scripts in `scripts/altgen/physical_fit_run.py` + fig scripts | assemble into the paper narrative (taxonomy: inside=Q²-nuisance, outside=coherence/excise; precision frontier) |

**All five sections are figure-complete.** 4–5 still need paper write-up.

## Section 1 figures (2026-07-15)

`analysis/paper/sec1_validation/make.py` — builds every non-NC ACHILLES-paper (arXiv:2508.19213) figure as
an ADoNIS-vs-ACHILLES overlay + ratio, from the live paper_banks banks (ADoNIS) and fs_rich oracles
(ACHILLES), through the one validated `plotting.chi2_ratio_panel` under the paper style. Config-driven
per-event TKI figures come from `configs/analysis/*.yaml` via `adonis.workflow.analyze.run_analysis`; the
non-histogram figures (σ(E), σ(p), σ(W), dσ/dω, angular) are standalone drivers in the same package.
`python -m analysis.paper.sec1_validation.make [--light]` builds them all (--light skips the heavy amps2/MC
drivers fig2/fig13).

- `fig01_ee_domega` — inclusive (e,e') dσ/dω on ⁴⁰Ar/¹²C, QE/RES/total (χ²/ndf 1.04 / 1.56)
- `fig02_anl_sigma` — free-nucleon RES σ(E_ν): pπ⁺ / nπ⁺ / pπ⁰ (χ²/ndf 32.8 / 3.4 / 0.9; pπ⁺ driven by the near-threshold point)
- `fig03_pi_nucleus_sigma` — π⁺ absorption + reaction σ(p) on ¹²C/⁴⁰Ar (χ²/ndf 0.58–1.3)
- `fig456_e4nu` — e4ν (e,e') E_QE / E_cal / P_T (χ²/ndf 1.83 / 2.14 / 1.16)
- `fig07 / fig08 / fig09` — T2K CC0π / T2K CC1π⁺ / MINERvA CC0π TKI (χ²/ndf 0.7–1.7)
- `fig10_uboone_cc1p0pi` — MicroBooNE CC1p0π δp_T in δα_T slices (χ²/ndf ~1)
- `fig13_piN_sigma` — meson-baryon DCC σ(W) + angular, ADoNIS MC vs ANL-Osaka

Every ratio sits within a few percent; the outliers are documented near-threshold analytic features (fig2
pπ⁺ E=400, fig13 ηN cusp). NC figures are intentionally out of scope.

## Section 3 result (2026-07-14): Gate I per observable subset

One bank pass produces `J` (27 knobs × 188 bins, 11 datasets) → `output/altgen/physfit_gate1_full.npz`;
every subset is a **row slice** of it, so the subset study is exact and free. Sections 2 and 3 share this
one Jacobian. Observable subsets live in `physical_fit.OBS_SUBSETS` (single source of truth, also used to
pick the released knobs in a subset-restricted fit).

**Which knobs each observable class measures** (shrinkage σ_post/σ_prior < 0.5, syst 5%).
Bank: `output/event_bank_v2` (1,874,385 ev, ragged record, pion SURVIVAL factor); npz `physfit_gate1_full_v2`.

| subset | #FIT | knobs |
|---|---|---|
| lepton (p_μ, cosθ_μ) | 2 | `kF_sf`, `Eb_shift` |
| lepton + pion kin | 3 | + `M_A_res` |
| TKI | 5 | `M_A_res`, `kF_sf`, `Eb_shift`, `s_NN_elastic[pn]`, `f_NN_cex` |
| multiplicities alone | 1 | `Eb_shift` |
| all 9 kinematic | 8 | + `M_A_qe`, `res_axial_strength`, `s_piN_elastic` |
| ALL 11 (kin + mult) | **10** | + `sabs`, `s_NN_elastic[pp]` |

- A **lepton-only** analysis measures *only the initial state*: the entire hard vertex and all of FSI are
  invisible to it. Pion kinematics unlock the RES vertex; TKI unlocks nucleon FSI; `M_A_qe` needs the
  full combination.
- The **multiplicities measure almost nothing on their own** but are not redundant: added to the 9
  kinematic observables they push `sabs` (0.52 → 0.38) and `s_NN_elastic[pp]` (0.51 → 0.47) over the line
  and sharpen the nucleon-FSI block (`f_NN_cex` 0.37 → 0.27, `s_NN_elastic[pn]` 0.36 → 0.29). They buy
  *FSI*, as expected from counting knocked-out nucleons and surviving pions.

> **The 8 → 10 change (2026-07-14) is a BUG FIX, not a re-tune.** The pion FSI reweight used to be
> branch-only: it could not express a change in the pion *interaction probability*, so `sabs`,
> `s_piN_elastic`, `s_piN_cex` and `s_conv` could only redistribute channels among interactions that had
> already happened, and a common rescale of the four was an EXACT per-event identity (the "pion-FSI flat
> direction" of `docs/logbook/info_content.md` — an artifact of the record, not physics). With the
> survival factor added (commit e656504, gated by `scripts/_pion_inwalk_gate.py`: reweight-vs-in-walk
> +11.6% → +0.45% ± 0.37%):
>
> | knob | raw (before → after) | marginalized (before → after) | |
> |---|---|---|---|
> | `sabs` | 0.331 → 0.208 | 0.68 → **0.38** | freeze → **FIT** |
> | `s_piN_elastic` | 0.441 → 0.347 | 0.78 → **0.42** | freeze → **FIT** |
> | `s_piN_cex` | 0.449 → 0.372 | 0.85 → 0.85 | still degenerate |
> | `s_conv` | 13.6 → 11.4 | 1.00 → 1.00 | still invisible |
>
> The raw sensitivities rose (the knobs now move the rate, not just the branch), but the *marginalized*
> ones rose far more — that is the flat direction dying. Nothing outside the pion block moved (all within
> noise), so the fix is surgical. Pion absorption and π–N elastic scattering are now measurable at T2K.

**"Freeze" hides two physically opposite failure modes** — reported separately (`sec3_failure_modes`),
since they demand opposite responses:
- **DEGENERATE** (raw shrinkage < 0.5, marginalized > 0.5): the data sees the knob clearly, another knob
  spends the sensitivity. `qe_norm` is the extreme case — raw **0.034** (one of the most sensitive knobs
  in the registry) → marginalized **0.86**, a 25× penalty, because `axial_strength`/`vector_strength`/
  `sf_norm` can each spell "scale the QE rate" (posterior corr. −0.50/−0.31/−0.22). `res_norm` likewise
  (raw 0.052 → 0.71) against `sf_norm` (corr. **−0.78**). *A better observable can recover these.*
- **INVISIBLE** (raw > 1): no information at this precision, and no re-parameterization helps.
  `s_conv` (raw 13.8), `pion_pole` (4.0), `gen` (2.6), `s_NN_inelastic[pn/nn]`.

Precision-relative by construction: at `ADONIS_SYST=0.15` only 4 knobs pass instead of 8
(`physfit_gate1_s15.npz`) — measurability is a property of the knob **and** the dataset's precision.

### Why five knobs carry *no* information (INVISIBLE), measured

Gate I is given nothing but the Jacobian, and it independently reproduces known physics: every
zero-information knob multiplies a term that is negligible or kinematically shut at T2K. Two mechanisms.

**(a) Amplitude terms that barely contribute.** `amps2(s) = a + b·s + c·s²` exactly, so the fractional
sensitivity is `(b+2c)/(a+b+c)` — computed per event, weighted over the bank:

| knob | \|∂ln amps²/∂θ\| | a 20% prior move buys | for scale |
|---|---|---|---|
| `gen` (G_E^n) | **0.012** | 0.25% on σ | `gep` 0.226, `mu_p` 0.441 |
| `pion_pole` | **0.045** | 0.9% on σ | `M_A_res` 1.086, `vector_strength` 0.955 |

Against a 5% systematic those are ≲0.05σ per bin. Interpretation (standard, not separately verified here):
G_E^n is small (Galster) *and* enters τ-suppressed relative to G_M at T2K Q²; the pseudoscalar/pion-pole
term contracts with the lepton tensor ∝ m_ℓ², i.e. helicity-suppressed for ν_μ (it would matter for ν_τ).

**(b) FSI channels that barely fire.** The knob's leverage is proportional to the channel's share of the
total cross section, and that share is set by a threshold the T2K final state does not reach. (NB: `sabs`
and `s_piN_elastic` are NOT in this list any more — they were only ever "degenerate" because the reweight
was branch-only; see the 8 → 10 note above. `s_conv` and the NN-inelastic knobs remain genuinely invisible,
and no reweight fix can change that: a closed channel is a closed channel.)

| knob | channel | measured in the 1.87M bank |
|---|---|---|
| `s_conv` | πN→ηN′ (m_η = 548 MeV) | **430 / 624,652 pion hits** (0.07%); mean σ_conv/σ_tot = 7×10⁻⁴ |
| `s_NN_inelastic[nn]` | nn→NΔ→NNπ | mean `finel` = σ_inel/σ_tot = **1.5%**; 1,906 realized |
| `s_NN_inelastic[pn]` | pn→NΔ→NNπ | mean `finel` = **1.3%**; 8,178 realized |
| `s_NN_inelastic[pp]` | pp→NΔ→NNπ | mean `finel` = **2.6%**; 19,157 realized |

- πN→ηN′ needs √s ≥ m_η + m_N = 1487 MeV ⇒ **p_π ≳ 680 MeV/c** on a nucleon at rest; T2K pions are
  mostly < 400 MeV/c, so the channel is closed except in the rare tail.
- `s_NN_inelastic[iso]` enters *only* through `g = s_el·(1−finel) + s_inel·finel` (`fsi_nucleon_reweight`),
  so its leverage **is** `finel` ≈ 1–3%: NN→NNπ needs ~290 MeV of nucleon KE and the FSI nucleons are
  mostly below it. A 20% scale moves σ_tot by ~0.3–0.5%.
- The measured realized-inelastic ordering (pp 19157 > pn 8178 > nn 1906) reproduces the failure ordering
  exactly (raw shrinkage 1.00 < 2.51 < 3.87).

These are structural, not fixable by a better observable *of this sample*: unlike the DEGENERATE knobs, no
re-parameterization recovers them at T2K kinematics.

### Knob × sample: complementary tagged beams (2026-07-15)

The T2K verdicts are conditional on the T2K observable set. Two failure modes are curable by a different
**sample**: DEGENERATE knobs (a pure-FSI hadron beam has no competing hard-vertex/SF knobs) and INVISIBLE
knobs (the beam energy opens a channel that is shut at T2K). Fisher is additive, so each beam is just extra
rows of J: `F = F_T2K + Σ_beam F_beam` (`analysis/beams/beam_fisher.py`).

Three tagged beams on ¹²C (`analysis/beams/beam_bank.py`, 500k each; ACHILLES oracles
`configs/achilles/run_cascade_{pip_C_broad,prot_C,neut_C}.yml`), validated as pure transport
(`analysis/beams/make_figs.py` → `output/paper/beams_validation.png`): reaction χ²/ndf π⁺ 0.67 / p 0.69 /
n 1.29, all within ±1%; the pion-production observable matches ACHILLES through the NN→NNπ threshold
turn-on (p 0.45, n 1.23). π⁺ absorption drifts ~10–15% high above 700 MeV/c (thin stats; the known
cascade absorption residual).

| sample | #FIT | knob gained vs T2K | why |
|---|---|---|---|
| T2K alone | 10 | — | |
| T2K + π⁺ | 11 | **`s_conv`** (1.00 → 0.49) | p_π up to 1 GeV/c opens πN→ηN′ |
| T2K + p | 11 | **`s_NN_inelastic[pp]`** (0.75 → 0.50) | KE > 290 MeV opens NN→NNπ (pp) |
| T2K + n | 12 | **`s_NN_elastic[nn]`** (0.78→0.48), **`s_NN_inelastic[nn]`** (0.93→0.47) | the nn isospin channel |
| **ALL** | **14** | + all of the above; `sabs` 0.38→**0.13**, `s_piN_elastic` 0.42→**0.16** | |

Each beam rescues exactly the knob its kinematics targets, and nothing else moves. What stays INVISIBLE
even with all three beams is the right set — every hard-vertex / vector knob (`gen`, `pion_pole`, `mu_p`,
`mu_n`, `vector_strength`, `qe_norm`, `sf_norm`, `axial_strength`): a hadron beam has no leptonic probe, so
it cannot touch them. That is precisely the gap **(e,e')** fills (`docs/logbook/electron_scattering.md`) —
a purely-vector photon probe for the QE vector current + spectral function.

This whole result exists only because of the pion **survival** fix (commit e656504): with the old
branch-only reweight, a common pion-σ rescale left w ≡ 1, so a π–C reaction sample could not constrain the
σ magnitudes at all.

## The 27 fittable knobs (canonical registry = `full_knobs.knob_specs`)
Also a standalone table in the paper. All are multiplicative scale = 1.0 nominal unless noted;
reweight mechanism in the last column (R1 σ-scale, R2 amps² quadratic, R3 density-ratio, R4 branch).

### Initial state — spectral function S(p,E) (R3)
| knob | tunes |
|---|---|
| kF_sf | Fermi-momentum / \|p\|-axis scale of S(p,E) (momentum-distribution width) |
| Eb_shift (≈0 MeV) | binding / removal-energy shift (slides S in E) |
| sf_norm | overall normalization of S(p,E) |
| src_tail | high-\|p\| short-range-correlation tail strength |

### QE hard vertex — nucleon form factors (R2)
| knob | tunes |
|---|---|
| M_A_qe | QE axial mass — Q²-shape of F_A(Q²) |
| axial_strength | overall QE axial current strength (g_A) |
| vector_strength | overall QE vector current strength |
| mu_p / mu_n | proton / neutron magnetic form factors (G_M) |
| gep / gen | proton / neutron electric form factors (G_E) |

### RES (single-π) hard vertex — DCC amplitude (R2)
| knob | tunes |
|---|---|
| M_A_res | RES axial mass — Q²-shape of the RES axial FF |
| res_axial_strength | N–Δ axial coupling C5A (overall RES axial strength) |
| pion_pole | pseudoscalar / pion-pole (PCAC) term strength |

### Channel normalizations (trivial multiplier)
| knob | tunes |
|---|---|
| qe_norm | overall QE cross-section normalization |
| res_norm | overall RES cross-section normalization |

### Pion FSI — cross-section magnitudes (R1)
| knob | tunes |
|---|---|
| sabs | pion absorption σ (Oset, πNN→NN) |
| s_piN_elastic | π–N elastic scattering σ |
| s_piN_cex | π–N charge-exchange σ |
| s_conv | pion conversion σ (πN→ηN′) |

### Nucleon FSI — cross sections + branching (R1 + R4)
| knob | tunes |
|---|---|
| s_NN_elastic[pp/pn/nn] | N–N elastic σ per isospin pair (3) |
| s_NN_inelastic[pp/pn/nn] | N–N inelastic σ (NN→NΔ→NNπ) per isospin pair (3) |
| f_NN_cex (0.5) | N–N elastic charge-exchange fraction (isospin id-swap probability) |

**Count:** 4 + 7 + 3 + 2 + 4 + 7 = **27**.

## Decisions we converged on (2026-07-14)
- **No new knobs are being added.** The exact/easy reweightable additions we scoped
  (`delta_strength` = P33 Δ strength; Oset `M_δ`/`Γ_δ` = in-medium Δ in absorption) turn out to be
  **~degenerate at T2K kinematics** with existing knobs (`delta_strength`≈`res_norm`; `M_δ`≈`sabs`,
  since T2K pions are soft and sample only a slice of the Δ energy dependence). The genuinely
  *distinct* additions are the **geometric FSI** dials (cascade Pauli `k_F`, formation zone,
  recapture), which are **not cleanly reweightable** (hard thresholds → need a soft-sigmoid
  relaxation = a physics approximation, diverges from ACHILLES). Net tension: **exact ⟂ informative**
  for the amplitude/propagator-shape knobs. → Build knobs only to a **Fisher-measured** gap (section 3),
  not a suspected one.
- **`delta_strength` is implemented as dormant, validated capability** (commit 57a12f8): exact,
  forward-preserving (nominal identity max\|Δw\|=0), clean gradient (AD/FD 0.99983), gated by
  `tests/test_delta_strength_bank.py`. It is **NOT** in `nominal_knobs`/`knob_specs`, so it does not
  affect any fit; the current bank lacks its record so `bank_reweight` guards it. To make it fittable
  would require a bank regen — deferred pending a Fisher-shown need (likely: drop it).
- **Coverage gaps (honest):** no RES lineshape (Δ W-shape; baked into DCC dressed tables, not
  reweightable) and no geometric FSI. Everything present is a magnitude, a Q²/W amplitude shape, or an
  SF density deformation.
- **sscat, pw_norm** were removed from the fittable registry (dead / dormant-DCC); the registry is now
  a single source of truth (`knob_specs`), consumed by `physical_fit.SPEC`.

## Figure & artifact inventory (already produced)
Output artifacts live under `output/` (gitignored, on-machine) and are regenerable from committed
scripts. Persisted npz carry binned data/σ/model curves so figures re-render without refits.
- `output/reports/physical_fit_report.pdf` — 6-page methodology report (`scripts/altgen/physfit_report.py`).
- `physfit_fig1_gate1` (Gate I shrinkage), `fig2_closure`, `fig3_inject2x` (×2 artifact: M0 −27σ /
  M1 exact+localized), `fig4_genie` (GENIE as data + excision), `fig5_m0_vs_m2`, `fig6_genie2x`,
  `fig7_closure5` (5-param closure), `fig8_q2mod_data`, `fig9_q2nuis` (Q²-nuisance closure +
  reconstruction), `fig10_mecmix` (2p2h admixture: M1 recovers + maps habitat).
- Run npz: `output/altgen/physfit_*.npz`, `p9_*.npz` (persisted BFPs + binned curves).

## Immediate next steps (suggested order for the next session)
1. **Section 3 — Fisher per observable subset.** Run `physical_fit.py` Gate I with the observable
   subsets defined above (add a subset selector to `build_physfit_datasets`); output a
   knob × subset shrinkage table/heatmap + the degeneracy (Fisher eigen) structure. This is the
   analytical heart and its inputs are ready.
2. **Section 2 — compact gradient figure** from `grad_arrows`/`grad_all` (all 27 knobs).
3. **Section 1 — ADoNIS-vs-ACHILLES validation** figures (CC0π/CC1π/multiplicities + ratio + χ²/ndf).
4. **Assemble** sections 4–5 from the existing physfit figures + report.
5. Optional: revisit the geometric-FSI (soft-k_F) knob **only if** section 3 shows an unconstrained
   FSI direction the current dials can't fill.

## Reproducibility notes
- Worktree needs `ACHILLES_DATA=<repo>/data/achilles` and `data/{nuclear,achilles,experiment}`
  (symlinked from the main checkout in this worktree).
- Event bank: `ADONIS_EVENT_BANK` → the frozen 1.87M bank (`output/event_bank`). Do NOT overwrite it;
  build test/new banks to separate dirs.
- Knob registry is `full_knobs.knob_specs(nominal_knobs())`; fits enumerate through it.

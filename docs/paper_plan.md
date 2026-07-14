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
| 1 | **Validation: ADoNIS reproduces ACHILLES** — T2K CC0π, CC1π, particle multiplicities | machinery exists; figures to assemble | `analysis/t2k/differentiability/make_plots.py`, `bank_matrix.py` (matrix + multiplicity + nucleon-momentum from the 1.87M bank) | pick the informative subset; produce clean side-by-side ADoNIS-vs-ACHILLES + ratio panels with χ²/ndf |
| 2 | **Compact "we have gradient information for all knobs"** | machinery exists | `grad_arrows.py`, `grad_all.py` (exact per-bin ∂(dσ/dx)/∂θ for all 27 knobs, autodiff) | one compact figure (e.g. per-knob gradient-arrow grid, or a 27×N_bins sensitivity heatmap) |
| 3 | **Fisher information per observable subset** → what is worth fitting | table ready + Fisher engine ready | 27-knob table (below); `physical_fit.py` Gate I (Asimov Fisher) on the 9-observable suite; `info_content.py` | run Gate I on subsets: **lepton-only** (p_μ, cosθ_μ), **lepton+hadron kin**, **TKI** (δp_T/δα_T/pN/δp_TT/δα_T), **multiplicities**, **all combined**; show which knobs pass (shrinkage<0.5) per subset + the degeneracy structure |
| 4 | **Closures** with Fisher-selected parameters | ✅ figure-complete | `physfit_fig7_closure5.png` (5-param, ≤0.5σ), 7-param on the 9-obs suite (`p9_closure.npz`, ≤0.4σ); `scripts/altgen/physfit_closure5_fig.py` | fold into the paper; optionally a fluctuated closure (null calibration) |
| 5 | **Fitting data not described by the model** (unknown unknowns) | ✅ figure-complete | `physfit_fig1–10`, `output/reports/physical_fit_report.pdf`; scripts in `scripts/altgen/physical_fit_run.py` + fig scripts | assemble into the paper narrative (taxonomy: inside=Q²-nuisance, outside=coherence/excise; precision frontier) |

**Gap = sections 1–2** (validation + the compact gradient figure). Sections 3–5 are analytically done;
3 needs the per-subset Fisher run, 4–5 need write-up.

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

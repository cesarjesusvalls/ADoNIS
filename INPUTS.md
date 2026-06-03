# External inputs needed to go from toy to real

The differentiable chain (`diffpi/`) is complete in toy form: every physics piece
recovers its parameters from a histogram, and the full A→(C+B+D)→G chain tunes
vertex + FSI knobs jointly. To make it *real* (Phase 2) and fit *data* (Phase 3) we
need the external inputs below. Each entry lists: **what** it is, the **form** we
need it in, the **source** (paper ref / who holds it), and what it **unlocks**.

Legend: ⭐ = gating (unlocks the most, hardest to substitute) · ○ = substitutable
with a public/parameterized stand-in or digitization.

---

## A. Hard interaction vertex (electroweak single-pion production)

1. ⭐ **ANL-Osaka DCC electroweak amplitudes.** The tabulated partial-wave
   amplitudes for `γ*/W*/Z* N → π N` in the helicity–LSJ representation
   (Kamano:2013iva), for EM, CC, and NC, as functions of (W, Q²) and partial wave
   (L, J, I), per meson-baryon channel.
   - Form: complex amplitude grids vs (W, Q²); the resolution/axes of the tables,
     plus the convention doc to reconstruct ⟨p_π p|j^μ|N⟩.
   - Source: ANL-Osaka group (Kamano, Nakamura, Lee, Sato); in ACHILLES these are
     read via the Fortran90 wrapper (Isaacson:2022cwh). Likely bundled with the
     ACHILLES distribution or obtainable from the authors.
   - Unlocks: the real hard vertex (replaces our toy F_A²·BW). The single most
     important input.

2. ○ **Vector & axial form-factor inputs.** Vector transition form factors from
   pion electroproduction (DCC1, DCC2); axial part from PCAC. If the DCC tables
   already include these, this is subsumed by (A1); otherwise we need the form
   factors separately.
   - Unlocks: the M_A / form-factor knobs attaching to the real amplitudes.

---

## B. Initial state (nuclear structure)

3. ⭐ **Hole spectral functions S_t(k, E).** Probability of removing a nucleon of
   momentum k and removal energy E, normalized to Z (protons) / N (neutrons).
   - Form: 2-D tabulated grids per nucleus per isospin.
   - Source: ¹²C — Rocco:2019gfb (correlated-basis) and/or Benhar CBF; ⁴⁰Ar
     proton+neutron — Nikolakopoulos:2024mjj (from JLab (e,e′p)).
   - Unlocks: realistic lepton kinematics / the convolution in the hadron tensor;
     currently absent in the toy.

4. ○ **Nuclear density / configuration inputs for the cascade geometry.** Correlated
   proton/neutron spatial distributions: ¹²C from GFMC (Carlson:2014vla); ⁴⁰Ar from
   single-proton/neutron densities (Isaacson:2020wlx).
   - Form: sampled configurations or density profiles ρ_p(r), ρ_n(r).
   - Unlocks: the impact-parameter interaction-probability model (our toy uses a
     uniform sphere / fixed bump).

---

## C. Final-state interactions (the cascade)

5. ⭐ **DCC meson-baryon scattering amplitudes.** Same DCC source as (A1), but the
   strong τ^{L,±,I}(s) for πN → {πN, ηN, KΛ, KΣ}, used for total + angular cross
   sections (App. "Meson-baryon scattering amplitudes").
   - Unlocks: the real Component-B angular distributions in the cascade.

6. ○ **Oset absorption parameterization.** Coefficient functions C_Q, C_A2, C_A3 and
   exponents α, β, γ vs pion kinetic energy T_π (Oset:1987re; Salcedo:1987md), plus
   the s-wave piece and the isospin factors for 2N absorption (VicenteVacas:1993bk).
   - Form: the published fit functions / tables (valid to 350 MeV).
   - Unlocks: real Component-D absorption (our toy uses ad-hoc g2/g3 shapes).

7. ○ **Propagating-Δ (GiBUU) model constants.** One-pion-exchange matrix elements
   for NN→NΔ (Dmitriev:1986st) and NΔ→NΔ; couplings f_NNπ, f_NΔπ, f_ΔΔπ; form-factor
   cutoff Λ=0.63 GeV; κ=0.2 GeV; Δ width Γ=112 MeV; NΔ→NΔ isospin factors (paper
   Tab.); detailed-balance relation; density-suppression α (Song:2014xza). Branching
   ratios for decays.
   - Form: all published numbers/formulas — mostly transcription, no data files.
   - Unlocks: real Component-E propagating-Δ mode.

---

## D. Kinematics & constants (mostly transcription, not data files)

8. ○ **Particle masses, widths, couplings** (m_N, m_π, m_η, m_K, m_Λ, m_Σ, m_Δ,
   lepton masses; Δ width). From the PDG / the paper.
9. ○ **Phase-space & boost machinery.** Höche/Byckling 2→2 / 1→2 building blocks,
   Källén functions, Lorentz boosts, Wigner spin rotations, Clebsch-Gordan/isospin
   tables. All formulas — we implement, no external files.

---

## E. Experimental data (Phase 3 — the actual fit)

For each target observable we need three things together: **central values +
covariance + the neutrino flux** (for flux folding). Per the paper's comparisons:

10. ⭐ **One experiment's data release to start** (recommend MINERvA CC0π STV or T2K
    CC1π⁺): differential cross sections (e.g. dσ/dδp_T, dσ/dp_π, dσ/dθ),
    the **full covariance matrix**, and the **flux**.
    - Source: MINERvA:2018hba; T2K:2021naz / T2K:2018rnz — public data releases
      (experiment data portals / arXiv ancillary / NUISANCE).
11. ○ **Additional experiments** for the joint fit: e4ν (CLAS), MicroBooNE,
    more T2K/MINERvA channels; JLab inclusive (e,e′) (Murphy:2019wed).
12. ○ **FSI-validation data** (not a fit target, but for forward fidelity): π-nucleus
    scattering cross sections; πN→πN SAID database (cns_dac); NN→NNπ cross sections
    (GiBUU parameterizations, Buss:2011mx).

---

## F. Forward-fidelity validation references

13. ⭐ **ACHILLES forward outputs at published parameters** — histograms from the
    actual generator (or the digitized paper figures) for the same observables, so
    we can prove the differentiable forward model reproduces ACHILLES *before*
    tuning. ACHILLES is a public C++ generator; ideally we run it or get its output.
    - Unlocks: the non-gradient forward-fidelity gate (Strategy §4).

---

## G. Parameter priors (for regularized fits)

14. ○ **Published central values + uncertainties** for every tunable knob (M_A,
    couplings, resonance masses/widths, Oset coefficients, α, Λ, κ) — used as priors
    / regularization in Phase 3. From the respective references.

---

## Minimal gating subset (to make real progress on ONE channel)

To stand up the first *real* end-to-end fit on a single experiment, the smallest
sufficient set is:

- **A1** DCC electroweak amplitude tables (hard vertex),
- **B3** one spectral function (e.g. ¹²C or ⁴⁰Ar),
- **C5** DCC meson-baryon amplitudes + **C6** Oset parameterization (FSI),
- **E10** one experiment's data + covariance + flux,
- **F13** ACHILLES output (or digitized figures) for the forward-fidelity check.

Everything in groups **D** and **G**, and the formulas in **C7**, are transcription
we can do without external files. The hard dependencies are the **DCC amplitude
tables (A1/C5)**, the **spectral functions (B3)**, and a **data+covariance+flux
release (E10)** — all three are held by the ACHILLES authors / ANL-Osaka group /
the experiments, and are the things to request first.

# External inputs — re-evaluated against the ACHILLES clone

ACHILLES is cloned at `/Users/cjesus/Software/DiffSinglePiProd/Achilles`. After
inspecting it, **almost every physics input is already present locally** — the
DCC amplitude tables, form factors, spectral functions, meson-baryon amplitudes,
the Oset coefficients (hardcoded), the propagating-Δ couplings, and the fluxes.
The *only* genuinely external dependency is the **experimental measurement data
(central values + covariance)** used as fit targets in Phase 3.

Legend: ✅ in the clone (path given) · 🔨 obtainable locally (build/run ACHILLES) ·
❌ external (must be fetched).

---

## A. Hard interaction vertex

1. ✅ **DCC electroweak amplitudes** — `Achilles/data/dcc_EW.dat` (40 MB, **ASCII**).
   - Format: header `njLs` (14 partial waves) with quantum numbers per wave
     `jpind=2J, Lpind=2L, ispind=2Iπ, itpind=2It`; then `maxw,mxq2` (67×28) grids
     for W (1076.957–1480 MeV) and Q² (MeV²); then amplitude blocks
     `ie iq idx ipw igmb ils  za(1) za(2) za(3)` where `za` = (bare, dressed,
     non-resonant) complex amplitudes. Indices `idx`=current component (1–7
     vector/axial), `ipw`=partial wave (1–14), `igmb`=meson-baryon channel.
   - Reference parser: `src/Achilles/fortran/amp_dcc_sl.f` (`amplitude()`,
     `interpolate_amp()`). We re-implement parsing + bilinear (W,Q²) interpolation
     in JAX. **The keystone input — and it's text we can parse.**

2. ✅ **Form factors** — `Achilles/FormFactors.yml` + `src/Achilles/FormFactor.cc`.
   - Vector = Kelly; Axial = AxialZExpansion with **`MA: 1.000`**, `tcut=9mπ²`,
     `t0=-0.28`, and the z-expansion `CC Params` (9 coeffs). These are exactly the
     tunable knobs (axial mass / z-expansion coefficients, vector params).

---

## B. Initial state

3. ✅ **Spectral functions** — `Achilles/data/Spectral_Functions/pke{12,16,40}{p,n}_{tot,MF,bg,asym}.data`
   (¹²C, ¹⁶O, ⁴⁰Ar; proton & neutron). Format: header `ne np` (e.g. 200×40), norm
   constant, then per momentum p a block of `(E, S(p,E))` pairs. Reader:
   `src/Achilles/SpectralFunction.cc` (Interp2D). Covered.

4. ✅ **Nuclear densities / configurations** — `Achilles/data/densities/`,
   `Achilles/densities/`, plus `data/configurations/` (correlated configs).

---

## C. Final-state interactions

5. ✅ **DCC meson-baryon amplitudes** — `Achilles/data/MesonBaryonAmplitudes/ANL/ANL_{i}-{f}.dat`
   (16 files for the 4×4 channels πN=0, ηN=1, KΛ=2, KΣ=3), plus `pwa-piDelta-pin.dat`.
   - Format: comment lines, then `W[MeV]` + 40 columns = 20 partial waves ×
     (Re,Im). Wave order S11,S31,P11,P13,P31,P33,D13,D15,D33,D35,F15,F17,F35,F37,
     G17,G19,G37,G39,H19,H39 with `twoJ_vec/L_vec/twoI_vec` given in
     `src/Achilles/MesonBaryonAmplitudes.cc`. Cross sections via partial-wave sum +
     Legendre (matches paper App. Eq. for dσ/dΩ). Covered.

6. ✅ **Oset absorption coefficients** — hardcoded in `src/Achilles/OsetCrossSections.cc`:
   `fCoefCQ={-5.19,15.35,2.06}`, `fCoefCA2={1.06,-6.64,22.66}`,
   `fCoefCA3={-13.46,46.17,-20.34}`, `fCoefAlpha={0.382,-1.322,1.466}`,
   `fCoefBeta={-0.038,0.204,0.613}`, `ImB0=0.035`, plus s-wave QE coeffs. Each is a
   quadratic in `x=T_π/mπ`. Direct transcription — no file fetch.

7. ✅ **Propagating-Δ (GiBUU) constants** — `src/Achilles/ResonanceHelper.cc` /
   `CascadeInteractions/DeltaInteractions.cc`: `fps=2.202, fp=1.008, λ²=0.63², κ²=0.2²,
   gA=1.267, fπ=92.4 MeV`; Dmitriev-Sushkov NN→NΔ matrix element; Blatt-Weisskopf
   energy-dependent width; Δ spectral BW. Decays: `data/decays.yml`. Transcription.

---

## D. Kinematics & constants

8. ✅ Masses/PIDs/widths — `Achilles/data/Particles.yml`, `parameters.dat`,
   `data/decays.yml`. Phase-space/boost formulas we implement ourselves.

---

## E. Experimental data (Phase 3 — the fit targets)

9. ✅ **Fluxes** — `Achilles/flux/`: DUNE (`flux_dune_neutrino_ND.root`),
   MicroBooNE (`microboone.root`, yaml), MINERvA (`MINERvA_ME_Flux_*.root`, dat).
   Flux folding inputs are covered.

10. ✅ **Measured differential cross sections + covariance** — **in the NUISANCE clone**
    at `/Users/cjesus/Software/DiffSinglePiProd/nuisance`. The NUISANCE sample names
    match the ACHILLES paper figures exactly (ACHILLES uses NUISANCE sample defs):
    - **MINERvA CC0π STV** (MINERvA:2018hba): `data/MINERvA/CC0pi/CC0pi_STV/MINERvA_DataRelease_Updated.root`
      — directories `muonmomentum/muontheta/protonmomentum/protontheta/dalphat/dpt/dphit`,
      each a TList with `xsec_with_total_errors` (TH1D data) + a `TMatrixT<double>`
      (covariance). Class: `src/MINERvA/MINERvA_CC0pinp_STV_XSec_1D_nu.cxx`.
    - **T2K CC1π⁺ STV** (T2K:2021naz): `data/T2K/CC1pipNp_STV/xsec_daT.txt` (+ dpTT, pN)
      — **plain text** with bin edges, data xsec, full + shape covariance, AND the flux.
    - **T2K CC0π STV** (T2K:2018rnz): `T2K_CC0pinp_STV_XSec_1Ddphit/dat/dpt_nu` classes + data.
    - **e4ν**: `data/Electron/12C.dat`, `16O.dat`. **MicroBooNE**: `data/MicroBooNE/CC1Mu*`.
    - Extraction: plain-text parse (T2K STV, e4ν) or `uproot` (MINERvA/MicroBooNE ROOT —
      verified `uproot 5.6.2` reads the data hist + TMatrix covariance). What we still
      transcribe from the NUISANCE C++ sample classes: the **signal definition** (cuts/
      topology) and the **binning/flux-folding** recipe, so our model prediction is
      computed comparably. Flux is bundled (in the T2K txt; `flux/` ROOT for others).

---

## F. Forward-fidelity validation

11. 🔨 **ACHILLES reference output** — the generator itself is the clone; we can
    **build and run it** (see `examples/run_*.yml`, `cascade.yml`, `run.yml`) to
    produce reference histograms at published parameters and validate that the
    differentiable forward model reproduces ACHILLES *before* tuning. No external
    fetch — just a build (CMake) + run.

---

## G. Parameter priors

12. ✅ Published central values for every knob are in `FormFactors.yml`,
    `parameters.dat`, and the hardcoded constants above; uncertainties from the
    cited papers when we add priors.

---

## Bottom line (revised)

The previously-"gating" inputs (DCC amplitudes, spectral functions, MB amplitudes,
Oset, fluxes) are **all in the clone**. The build order is now unblocked:

1. **Differentiable DCC loader** (A1) — parse `dcc_EW.dat`, JAX interpolation in
   (W,Q²), with knobs; mirror `amp_dcc_sl.f`. *The keystone, fully local.*
2. **Differentiable form factors** (A2) — port `FormFactor.cc` Kelly + axial
   z-expansion with `MA` and z-coeffs as the tunable parameters.
3. **MB amplitudes + Oset** (C5/C6) — port the ANL partial-wave → cross-section
   path and the Oset quadratics into the cascade.
4. **Forward-fidelity gate** (F11) — build/run ACHILLES, match histograms.
5. **Only then** fetch one **data+covariance** release (E10) via NUISANCE and fit.

**There are now ZERO external dependencies.** The measured cross sections +
covariance (E10) are in the NUISANCE clone, extractable with `uproot` (verified) or
plain-text parsing. Every physics input, every data target, the fluxes, and the
generator itself are all local. The whole Phase-2/3 program can proceed offline.

What we still *write* (not fetch): the JAX loaders for the ACHILLES data files, the
transcription of NUISANCE **signal definitions + binning** (from the C++ sample
classes) so our model prediction matches theirs, and the differentiable χ² against
the extracted data+covariance.

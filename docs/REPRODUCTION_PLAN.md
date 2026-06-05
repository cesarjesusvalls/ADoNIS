# Reproduction plan — arXiv:2508.19213 in ADoNIS

Goal: **fully reproduce** "Single pion-production and pion propagation in ACHILLES"
(Isaacson et al., arXiv:2508.19213v2) with ADoNIS — every figure / demonstration —
while keeping the chain **differentiable end-to-end** and **cross-validated against
the ACHILLES oracle** at every step.

> The paper does **no** data-driven parameter tuning: it fixes all parameters from
> external theory/fits and compares predictions to data as-is (it even sets its one
> cascade knob, the density-suppression α, to 0 by hand, and defers axial-form-factor
> studies to future work). So *reproduction = forward modeling + validation*.
> **Gradient-based tuning is a later program, started only once reproduction is complete.**
> The differentiable machinery is still built and gradient-checked at every step — it is
> exercised by the closure gate, just not used for fitting yet.

See `docs/STRATEGY.md` for the overall project strategy, `docs/CONTAINER.md` for the
oracle image, and `docs/STATUS.md` for history.

## Governing discipline (applied to every module)

1. **Differentiability gate** — each module ships `closure_test()`: autodiff == central
   finite-difference on its differentiable output (weight/observable w.r.t. that module's
   physics knobs). A module is not done until its closure passes **and** the end-to-end
   gradient still matches FD after it is wired into the chain.
2. **Oracle gate** — each module ships `oracle_test()`: generate matching ACHILLES events
   with the same config via `ghcr.io/cesarjesusvalls/achilles:oracle` (extend
   `.github/workflows/oracle.yml`), parse to an oracle npz, compare χ²/ndf ≈ 1.
3. **No tuning yet** — all parameters fixed at the paper's external values.
4. Both gates run **per file** in CI (`.github/workflows/ci.yml`); figures upload as artifacts.

Legend: ☐ todo · ◐ partial · ☑ done.

## Current state

- ☑ CC single-pion ANL-Osaka DCC electroweak vertex, full exclusive final state (lepton + π + N) on ¹²C
- ☑ Spectral-function nuclear model (¹²C), Monochromatic flux, `NoFSI`
- ☑ Per-module `closure_test`/`oracle_test` contract; CI green; oracle published as the `oracle-data` release
- ☑ Differentiable cascade **proof-of-concept** exists in git history (Phase-1 toy `component_c/d/e`, `integrate_full`; removed in commit `f60947a`, recoverable) — joint closure recovered M_A + m_Δ + Γ + σ_sc + σ_abs. Real-physics FSI port is Phases D–H.

---

## Phase 0 — Inventory & oracle-matrix infrastructure (prerequisite)

**See `docs/phases/PHASE0.md` for the full inventory + run-mode verification matrix.**
Headline: every *physics* input (incl. ⁴⁰Ar SF, DCC MB PWA tables, densities/configs,
all fluxes) is already in the oracle image; only the NUISANCE data releases (Phase-3
fit targets) are external. The image binary does **event generation only** — the
standalone π–nucleus cascade (`achilles-cascade`, behind `ACHILLES_ENABLE_CASCADE_TEST`,
OFF) is **not** in it, so Figs c12_ar40 (G2/H2) need a cascade-enabled build later.

- ☑ Inventory every external input the paper needs (present-in-image / NUISANCE-clone / missing) — done in `docs/phases/PHASE0.md §1`.
- ☑ Verify the image actually runs each needed mode (EM/CC/free-nucleon ✅ ran; NC to verify; standalone cascade ❌ needs a rebuild) — `PHASE0.md §2–3`.
- ☐ Generalize `oracle.yml` into a **config matrix**: one ACHILLES run card per experiment/observable → per-target oracle npz published as release assets (same image, same parse pipeline). *(design note in `PHASE0.md §5`)*
- ☐ Decide native-vs-NUISANCE for signal/binning: **reimplement signal defs + binning natively** (for differentiability); use NUISANCE/HepData **data releases only as comparison targets**.

---

## Phase A — Currents & free-nucleon vertex (no nucleus, no cascade)

Reuses the proven differentiable vertex machinery → lowest risk.

- ☐ **A1 — EM 1π current** (electron probe). closure: grad wrt vector-FF knobs. oracle: vs ACHILLES EM 1π. *(prereq for Fig 1, e4ν)*
- ☐ **A2 — NC 1π current**. closure + oracle vs ACHILLES NC 1π. *(prereq for MicroBooNE NC π⁰)*
- ◐ **A3 — Free-nucleon target** (bare nucleon, no SF), 3 isospin channels νp→μ⁻pπ⁺, νn→μ⁻nπ⁺, νn→μ⁻pπ⁰ as σ(E_ν). closure: grad wrt M_A. oracle: vs ACHILLES nucleon run + ANL/BNL reanalyzed data. **→ Fig 2 (ANL/BNL)** — *vertex DONE: `FreeNucleon` model + σ(E_ν) scan, closure dσ/dM_A exact, and σ(E_ν) for all 3 channels reproduces ACHILLES (1H+1N, ν_e) to ~1% (single universal constant, 27 cells). Remaining: muon mass (ν_μ) + physical units + ANL/BNL data overlay. See `docs/phases/PHASE_A3.md`.*

## Phase B — QE channel + inclusive (e,e′)

- ☐ **B1 — QE 1-nucleon vertex** W^{μν}_{1N} (+ SF fold). closure: grad wrt QE FF. oracle: inclusive QE peak.
- ☐ **B2 — EM inclusive fold** dσ/dω = QE + 1π. oracle: vs JLab-config ACHILLES, ¹²C. **→ Fig 1 (¹²C)**
- ☐ **B3 — ⁴⁰Ar spectral function** (NuclearModel swap). closure: sampler check. oracle: Ar inclusive. **→ Fig 1 (⁴⁰Ar)**
  - external: ⁴⁰Ar p/n spectral functions.

## Phase C — Flux folding + signal/observables infrastructure (bare production)

- ☐ **C1 — FluxModel**: T2K / MINERvA / MicroBooNE-BNB histogram fluxes + electron beams (per-event E_ν). closure: weight still diff in M_A under flux.
- ☐ **C2 — SignalDef**: 0π, 1p0π, CC0π, CC1π⁺, NC1π⁰ with detection thresholds (detached cuts). closure: grads flow through selected-event weights.
- ☐ **C3 — Observables**: TKI (δp_T, δα_T, p^N, δp_TT, p_n^recon) + estimators (E_QE, E_cal). closure: autodiff==FD on each. oracle: vs ACHILLES event-level.
- Gate: flux-folded bare-production prediction for a signal reproduces ACHILLES **with FSI off**.

## Phase D — FSI scaffold & differentiability proof (de-risk early)

- ☐ **D1 — Port the toy cascade** (`component_c/d/e`, `integrate_full`) into a concrete `FSIModel` acting on the real `EventRecord`. closure: **revive the 5-param joint closure** (M_A, m_Δ, Γ, σ_sc, σ_abs) on real final states — proves differentiability survives a stochastic FSI transform *before* investing in real physics. (Toy physics; no oracle.)
- This is the single most important early checkpoint (kind-1/kind-3 score weights, two-replica loss, expected-Jacobian gate, gradient-SNR control all already validated).

## Phase E — Real meson-baryon scattering (DCC PWA)

- ☐ **E1 — PWA total σ(W)** for πN, ηN, KΛ, KΣ. closure: grad wrt a σ-norm knob. oracle: vs ACHILLES INC. **→ Fig DCC_total**
- ☐ **E2 — Angular dσ/dΩ** (Legendre P_L, P_L′, full interference) + inverse-CDF angle sampling. closure: diff in σ-knobs (angle detached). oracle. **→ Fig DCC_angular**
  - external: DCC meson-baryon PWA tables.

## Phase F — Real Oset absorption

- ☐ **F1 — Oset Im Σ_Δ** (C_Q/C_A2/C_A3 vs T_π, s-wave) → absorption σ. closure: grad wrt absorption strength. oracle: Oset parameterization values.

## Phase G — Cascade engine + Virtual Resonances mode (highest risk)

- ☐ **G1 — Impact-parameter propagation**: nuclear configs (GFMC ¹²C, densities ⁴⁰Ar), time steps, Gaussian/cylinder probability, Pauli blocking. closure: **gradient-SNR gate** (cascade weight differentiable, SNR > 1).
  - external: GFMC ¹²C configurations, ⁴⁰Ar densities.
- ☐ **G2 — Virtual Resonances mode** = DCC πN (E) + Oset (F) + NN→NNπ (immediate-decay Δ), wired as an `FSIModel`. closure: end-to-end grad vs FD. oracle: π⁺–C & π⁺–Ar **reaction + absorption** cross sections (Virtual). **→ Fig c12_ar40 (Virtual)**
- *Differentiability is genuinely at risk here* (discrete channel selection, variable interaction count); Phase D front-loads the proof.

## Phase H — GiBUU Δ model + Propagating Resonances mode

- ☐ **H1 — GiBUU Δ physics**: NN↔NΔ (detailed balance), NΔ→NΔ, one-pion-exchange MEs (Λ, κ, z form factors), Δ spectral function, density-suppression option. closure: grad wrt Δ params. oracle. **→ Fig pion_production / delta_compare**
- ☐ **H2 — Propagating-Δ mode**: Δ as dynamical d.o.f. (decay P(t)=1−e^{−δt/τ}, branchings, Pauli-blocked decays) + Oset s-wave. closure: end-to-end grad. oracle: π⁺–C & π⁺–Ar (Propagating). **→ Fig c12_ar40 (Propagating)**

## Phase I — Full exclusive e/ν comparisons (vertex + cascade + flux + signal)

Each gate: full-chain closure (M_A + an FSI knob, autodiff==FD) preserved; oracle χ²/ndf vs ACHILLES (both cascade modes) at the level achieved so far.

- ☐ **I1 — e4ν** electron-C: 0π dσ/dE_QE, 1p0π dσ/dE_cal, 1p0π dσ/dP_T. **→ Figs e4v***
- ☐ **I2 — T2K**: CC0π (δp_T, δα_T), CC1π⁺ (p^N, δp_TT). **→ Figs T2K***
- ☐ **I3 — MINERvA**: CC0π (δα_T, p_n^recon). **→ Figs MINERvA***
- ☐ **I4 — MicroBooNE** (needs ⁴⁰Ar + NC + A_C smearing matrix): CC1p0π double-diff (δp_T,δα_T); NC1π⁰Xp double/single-diff. **→ Figs 1mu1p0pi, NCpi0***

---

## Cross-cutting infrastructure

- **Oracle at scale**: `oracle.yml` config matrix → per-experiment oracle npz release assets.
- **NUISANCE**: native signal/binning (differentiable); NUISANCE/HepData data releases as targets only.
- **Per-module CI**: every module's closure + oracle gates added to the per-file pytest run; figures uploaded.

## Principal risks

1. **Cascade differentiability (G/H)** — discrete branching + variable interaction counts. Mitigation: Phase D proves the architecture on toy physics first; gradient-SNR is a hard gate in G1.
2. **Scope** — this is re-implementing a multi-author generator paper; delivered phase-by-phase, each independently gated and useful.
3. **External inputs** — inventoried in Phase 0 (most in the image/clone; data releases in the NUISANCE clone).

## Execution order

Phase 0 → **A → B → C** (reuse the proven vertex; reproduce **Fig 1 + Fig 2**, build e/ν infra, zero cascade risk) → **D** (prove FSI differentiability) → **E → F → G → H** (real cascade, both modes; pion-nucleus figs) → **I** (exclusive e/ν figs).

Suggested first build slice: **A3** (free-nucleon 1π → Fig 2) — the smallest self-contained, fully-differentiable, oracle-checkable unit.

## What "fully reproduced" means (definition of done)

All paper figures reproduced by ADoNIS within the χ²/ndf tolerance used for the existing
final-state oracle gates, every contributing module passing both its closure and oracle
gates in CI, and the full chain differentiable end-to-end (verified by closure on M_A and
at least one FSI knob through production + cascade + flux + signal). Only then does the
tuning program begin.

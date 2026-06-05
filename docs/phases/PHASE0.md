# Phase 0 — Inventory & oracle-mode verification

Prerequisite phase from `docs/REPRODUCTION_PLAN.md`. Goal: (a) inventory every
external input the paper needs and mark it present/missing, and (b) **verify which
ACHILLES run modes the published oracle image can actually run** before building any
module that depends on that mode (the "verify image modes first" discipline).

Image under test: `ghcr.io/cesarjesusvalls/achilles:oracle` (commit `a4f761ea`,
`v0.3.0-1-ga4f761ea`). All `docker` probes below ran on Apple Silicon under
`--platform linux/amd64` (emulated; ~10–15 s warm-up per run).

---

## 1. Input inventory — everything the paper needs is already in the image

Listed from the image's `/achilles/data`, `/achilles/flux`, `/achilles/examples`.
`ACHILLES_DATA` for the model side defaults to `./achilles_data/` (only the two
runtime tables are fetched there; the rest live in the image and are read by the
ACHILLES binary itself when generating oracles).

| Input (plan/INPUTS.md ref) | Location in image | Status | Needed by |
|---|---|---|---|
| DCC electroweak amplitudes | `data/dcc_EW.dat` (40 MB) | ✅ (fetched to `achilles_data/`) | A1/A2/A3, B, I |
| Form factors | `FormFactors.yml` (`MA: 1.000`, Kelly, axial z-exp) | ✅ | A, B |
| ¹²C spectral fn (p) | `data/Spectral_Functions/pke12p_tot.data` | ✅ (fetched) | B1/B2 |
| ¹²C spectral fn (n) + MF/bg | `pke12n_{tot,MF,bg}.data` | ✅ in image | B, isospin |
| ¹⁶O spectral fn | `pke16{p,n}_tot.data` | ✅ in image | (e,e′) cross-checks |
| **⁴⁰Ar spectral fn (p/n)** | `pke40{p,n}_{tot,MF,bg}.data`, `pke40{p,n}_asym_tot.data` | ✅ in image | **B3, I4 (MicroBooNE)** |
| DCC meson-baryon PWA tables | `data/MesonBaryonAmplitudes/ANL/ANL_{0..3}-{0..3}.dat` (16) + `pwa-piDelta-pin.dat` | ✅ in image | **E (DCC σ/angular)** |
| Oset absorption coeffs | hardcoded in `src/.../OsetCrossSections.cc` (transcribe) | ✅ (source in image `/achilles/src`) | F |
| GiBUU Δ constants | `src/.../ResonanceHelper.cc`, `CascadeInteractions/DeltaInteractions.cc` | ✅ (source) | H |
| Nuclear densities / configs | `data/densities/`, `data/configurations/QMC_configs.out.gz`, `data/realOP_12C_EDAI.dat`, `data/rho_0p5.dat` | ✅ in image | G |
| Decays | `data/decays.yml` | ✅ | H |
| Particles / masses | `data/Particles.yml`, `parameters.dat` | ✅ | all |
| Fluxes | `flux/` — T2K (`T2K_nu.dat`, `t2k_flux.csv`), MINERvA (`MINERvA_ME_Flux_*.root`, `minerva_*.dat`), MicroBooNE (`microboone.root`, `microboone_flux_numu.yaml`), DUNE, MiniBooNE | ✅ in image | **C1, I2/I3/I4** |
| GEANT interaction data | `data/GeantData.hdf5` | ✅ | G (hN model) |

**The full ACHILLES source tree is in the image at `/achilles/src`** (it is a build
dir, not a stripped install), so the C++/Fortran constants that the plan transcribes
(Oset, GiBUU) can be read straight from the image as well as from the sibling
`../Achilles` clone. The git history is excluded (`[warning] This is not a git
repository version of Achilles`).

**External (must fetch) — Phase-3 fit targets only:** measured differential cross
sections + covariance. These live in the sibling **`../nuisance`** clone (NUISANCE
data releases) per `docs/INPUTS.md §E`; not needed until the tuning program.

**Conclusion:** zero missing *physics* inputs for Phases A–H. The only fetch-step is
the NUISANCE data releases for the eventual fit (Phase 3 / plan Phase I targets).

---

## 2. Run-mode verification matrix (the load-bearing Phase-0 result)

The image ships a **single** binary, `/achilles/bin/achilles`, whose entrypoint is the
**event generator** (`EventGen`, `main.cc::GenerateEvents`). Its docopt usage is just
`achilles [<input>]` + `--display-*` flags — **there is no cascade/transparency
subcommand.** This is the key constraint that shapes which oracles we can produce.

| Mode | How invoked | Verified | Plan phases | Notes |
|---|---|---|---|---|
| **CC ν 1π on ¹²C** | `achilles run_nue_12C…` / `res_1pi_12C_nu.yml` | ✅ (existing `oracle_finalstate.npz`, gates pass) | I, baseline | the established oracle |
| **EM e⁻ 1π** (electron probe) | `examples/run_electron_{1N,1H,12C}_SF_*.yml` | ✅ **ran** (1N, 10 evt, 12 s, σ≈676 nb, mult-4 final state) | **A1**, B2, I1 | example cards present |
| **Free-nucleon target** (`Name: 1N`) | `examples/run_*_1N_SF_*.yml` | ✅ **ran** (same run as above) | **A3** | ⚠️ `1N` example still uses `QESpectral` + a ¹²C density → it is a *single nucleon with spectral/Fermi motion*, **not a static free nucleon**. For Fig 2 (ANL/BNL bare-nucleon σ(Eν)) we need a nucleon at rest — resolve in Phase A (try a no-motion nuclear model / momentum-zeroed SF, or compare at the SF level). |
| **ν̄ CC 1π** | `examples/run_nuebar_*` | ◐ card present, not run | (ν̄ cross-checks) | |
| **NC ν 1π** | construct `Processes: Leptons: [12,[12]]` (ν in/out) | ☐ **not verified** — no example card; must build one and run to confirm the DCC backend emits NC | **A2**, I4 (NC π⁰) | verify before building A2 |
| **Cascade as FSI inside event gen** | `Cascade: { Run: True, … }` block in a `run_*.yml` | ◐ supported by the generator (the block exists in every card; set `Run: True`); not yet run | **I** (exclusive e/ν with FSI on) | this is how FSI-on observables get their oracle |
| **Standalone π–nucleus σ** (reaction/absorption, transparency) | `achilles-cascade` (`CascadeMain.cc`+`RunCascade.cc`) | ❌ **NOT in the image** | **G2, H2 (Fig c12_ar40)** | see §3 |

### Free-nucleon EM smoke run (recorded)
```
docker run --rm --platform linux/amd64 -v "$PWD/_oracle_out":/out \
  ghcr.io/cesarjesusvalls/achilles:oracle /out/elec_1N_smoke.yml
# -> Total xsec 676.19 nb (0.03%), Generated 10/10 events, "Success!", 12 s
```
(The smoke config is `examples/run_electron_1N_SF_500.yml` with `Output.Name`
redirected to `/out/…`; the bundled cards write to a relative path that isn't mounted.)

---

## 3. The cascade gap (Phases G2/H2) — known, deferred, with a clear fix

The standalone pion-nucleus cross-section runs that produce the paper's **π⁺–C & π⁺–Ar
reaction + absorption cross-section figure (Fig c12_ar40)** use the **`achilles-cascade`**
executable, which is gated behind a CMake option:

```cmake
# src/Achilles/CMakeLists.txt
add_executable(achilles main.cc)                 # <- the only target built in the image
if(ACHILLES_ENABLE_CASCADE_TEST)
    add_executable(achilles-cascade CascadeMain.cc RunCascade.cc)   # <- NOT built (option OFF)
endif()
```

The image's stale top-level `cascade.yml` is *also* out of schema for the current
binary (the generator rejects it: requires `Main/NEvents`; the cascade `Target`
`NucleonNucleon` now requires a `ResonanceMode` option). So even pointing the
generator at it fails — it was a card for the separate cascade binary.

**The two cascade modes the paper compares are real and configured** — see the
ACHILLES defaults (both in the image at `data/default/`):

- **Virtual Resonances (Phase G)** — `data/default/VirtResInteractions.yml`:
  `NucleonNucleon {Mode: GiBUU, ResonanceMode: Decay}` + `PionInteraction
  {HardScatter: MesonBaryonInteraction, Absorption: PionAbsorptionOneStep}`
  → DCC πN rescattering (E) + Oset absorption (F) + immediate-decay Δ.
- **Propagating Resonances (Phase H)** — `data/default/PropResInteractions.yml`:
  `NucleonNucleon {Mode: GiBUU, ResonanceMode: Propagate}` + `DeltaInteraction
  {Mode: GiBUU, SWaveAbsorption: True}` → Δ as a dynamical d.o.f.

**Fix when we reach G/H (pick one):**
1. **Local build** of ACHILLES with `-DACHILLES_ENABLE_CASCADE_TEST=ON` (the macOS
   recipe in `docs/INPUTS.md` works; add the flag) → produces `achilles-cascade`,
   run π⁺–C/π⁺–Ar locally, parse to an oracle npz, publish as a release asset. **Cleanest;
   no registry-push needed.**
2. **Rebuild + republish the image** with the flag on (needs `packages:write`; the
   current CI token is `packages:read`, so this likely needs the human).
3. **Indirect validation**: exercise the cascade only as FSI inside full event
   generation (`Cascade: Run: True`) and gate FSI-on *production* observables; does
   not reproduce the standalone π-A σ figure directly.

Recommendation: option 1 (local cascade build → release asset), built when Phase G
starts. Until then, the cascade FSI differentiability is de-risked on **toy physics**
in Phase D (recoverable from commit `f60947a`, `archive/diffpi_legacy/component_{c,d,e}.py`,
`integrate_full.py`) with **no oracle needed**, exactly as the plan sequences it.

---

## 4. What this unblocks / next steps

- **A3 (free-nucleon CC 1π → Fig 2)** — unblocked; mode runs. First build slice.
  Open detail: get a genuinely static nucleon for the ANL/BNL comparison (see §2 note).
- **A1 (EM 1π)** — unblocked; example cards run.
- **A2 (NC 1π)** — build an NC runcard and **run it once to confirm** the DCC backend
  emits NC before writing the module.
- **B3 (⁴⁰Ar SF)** — inputs present (`pke40*`); just a NuclearModel swap.
- **C1 (fluxes)** — all flux files present in the image.
- **E (DCC MB)** — PWA tables present (`MesonBaryonAmplitudes/ANL`).
- **G/H standalone cascade oracle** — needs a cascade-enabled build (§3); deferred.

## 5. Oracle-matrix infrastructure (plan Phase 0 item 2) — design note

`scripts/write_oracle_config.py` + `.github/workflows/oracle.yml` currently bake one
config (`data/oracle/res_1pi_12C_nu.yml`). Generalize to a **matrix**: a table of
`(probe, target, energy/flux, signal)` → runcard → `oracle_<key>.npz`, each published
as a release asset under the same `oracle-data` release, reusing
`scripts/oracle_from_hepmc.py`. Targets for the matrix as phases land: `e_12C`,
`e_1N`, `nu_1N` (A3), `nu_12C` (have), `nc_12C` (A2), `nu_40Ar` (B3/I4). See task #2.

> Practical note for autonomous runs: oracle generation on this machine is **emulated
> amd64** (~125 s fixed warm-up + per-event cost). Generate oracles locally at modest
> statistics for gating; reserve high-statistics regeneration for the CI `oracle.yml`
> (native amd64 runners) when a phase is otherwise green.

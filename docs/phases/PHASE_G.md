# Phase G — Cascade engine + Virtual Resonances mode

Plan ref: `docs/REPRODUCTION_PLAN.md` Phase G. The intranuclear cascade (Virtual
Resonances mode) and the standalone π–nucleus reaction/absorption cross sections
(**Fig c12_ar40**). This was the one hard showstopper (the published `:oracle` image
lacks the `achilles-cascade` binary, and this environment has no cmake/gfortran to build
it). **Resolved by building a cascade-enabled image inside Docker** (the toolchain lives
in the container, so the host's missing toolchain is irrelevant).

## The cascade-enabled image (`docker/Dockerfile.cascade`)
Builds ACHILLES (commit `a4f761ea`) on `alpine` with **`-DACHILLES_ENABLE_CASCADE_TEST=ON`**
(→ the standalone `achilles-cascade` = `CascadeMain.cc` + `RunCascade.cc`) and
**`-DCMAKE_CXX_FLAGS="-fno-visibility-inlines-hidden"`** (the factory-registry coalescing
flag `docs/INPUTS.md` flags as "REQUIRED to run").

```bash
# build context = the ACHILLES source tree
docker build -f docker/Dockerfile.cascade -t achilles:cascade /path/to/Achilles
docker run --rm --entrypoint /achilles/bin/achilles-cascade -v "$PWD/out":/out \
  achilles:cascade /out/cascade_virt_c12.yml
```

Verified: the image contains **both** `bin/achilles` (event generator — runs, 676 nb on the
1N smoke test) **and** `bin/achilles-cascade` (the previously-missing cascade driver).

## Running the cascade oracle
`achilles-cascade` reads a **self-contained** runcard (raw `YAML::LoadFile` → **no
`!include`**; inline the `data/default/VirtResInteractions.yml` interactions). CrossSection
mode (`RunCascade.cc::InitCrossSection`): each event is a test pion at a random impact
parameter in a disc of radius R; event weight = πR² [nb] if it interacts (a "hit"), else 0.
So **σ_reaction(p_π) = mean event weight per momentum bin** — parsed by
`scripts/cascade_xsec_from_hepmc.py`. Config recipe in `_oracle_out/cascade_virt_c12.yml`
(π⁺ on ¹²C, KickMomentum [80,500] MeV, Virtual mode).

The two modes: **Virtual (G)** = `VirtResInteractions.yml` (NucleonNucleon GiBUU
ResonanceMode:Decay + PionInteraction[MesonBaryon + PionAbsorptionOneStep]); **Propagating
(H)** = `PropResInteractions.yml` (ResonanceMode:Propagate + DeltaInteraction SWaveAbsorption).

## The cascade segfault — diagnosed & fixed (an upstream ACHILLES build bug)
The first cascade runs segfaulted (SIGSEGV) during the `Cascade` interaction parse. gdb
backtrace pinned it: `GetSuggestion()` (`Utilities.cc:274`, the "did you mean?" helper)
called from `InteractionHandler::decode` — i.e. the interaction name (even `NucleonNucleon`)
**was not in the registry**, and the empty-registry error path crashes.

**Root cause:** `achilles-cascade` links `event_gen docopt cmake_git_version_tracking` but
**not** `AchillesCascadeInteractions` — the shared lib whose static initializers self-register
the cascade interactions (`src/Achilles/CMakeLists.txt:172`, which has a literal TODO about
this). The event-gen `achilles` binary works because it links `plugin_manager` (+`dl`); the
cascade binary doesn't, so no registrations run. (It is **not** the visibility flag — adding
`-fno-visibility-inlines-hidden` did not fix it.)

**Fix** (`docker/Dockerfile.cascade`, a Python patch before cmake): (a) instantiate a
`Plugin::Manager` in `CascadeMain.cc` (as `main.cc` does) + link `plugin_manager`/`dl`;
(b) **force-link `AchillesCascadeInteractions` with `--no-as-needed`** (the actual fix —
the interactions self-register via that shared lib's static initializers, which only run if
the lib is a NEEDED dependency). After this the registry is populated and the cascade runs.

## Status — cascade oracle GENERATED (Fig c12_ar40 reproduced)
Full oracle done: pi+ on 12C (Virtual + Propagating) AND pi+ on 40Ar (Virtual), all
reproduce the Delta(1232) reaction peak (p_pi 275-305 MeV). data/oracle/cascade_pip_*.csv;
scripts/gen_cascade_oracle.py (batches over seeds to dodge the sporadic crash);
figures/cascade_pip_c12.png; gates in tests/test_cascade_oracle.py.

## How it got working
With the patched image, `achilles-cascade` on the Virtual π⁺–¹²C config prints
**"Cascade running in CrossSection mode"**, propagates pions through ¹²C, and **writes a
NuHepMC** (verified: 1182 events in a partial run). The showstopper is resolved — a
cascade-enabled image that *runs the intranuclear cascade* now exists (`docker/Dockerfile.cascade`).

Two refinements remain before the Fig-c12_ar40 oracle is final:
1. **A sporadic mid-run SIGSEGV** (~500 hits in) — a specific event/kinematic edge case in
   the cascade physics; work around with smaller `NEvents` batches (the achieved 1182-event
   file is already usable) or bisect the offending event.
2. **σ(p) extraction semantics**: `InitCrossSection` weights a "hit" by πR² [nb] and a miss
   by 0, so σ_reaction(p)=⟨weight⟩ needs BOTH hits and misses; the partial file shows only
   non-zero (hit) weights, so confirm whether misses are written (then ⟨w⟩ is direct) or the
   miss count must come from `generated_events` vs total attempts. Parser:
   `scripts/cascade_xsec_from_hepmc.py` (reads the named `W CV`/`W <val>` weight + the
   status-29 incoming test pion).

The model-side cascade FSIModel (whose differentiability Phase D de-risks on toy physics)
is the parallel build.

## In-event ν cascade — FIXED (`docker/Dockerfile.fullcascade` → `achilles:fullcascade`)
The *standalone* `achilles-cascade` tool gives the π-beam reaction σ; the **paper's central
"pion propagation" result** needs the in-event cascade on a ν final state
(`Cascade: Run: True` in the main generator). Both the `:oracle` and `:cascade` images
**SIGSEGV** there — same empty-registry root cause, but the earlier fix patched only the
`achilles-cascade` target. The main `achilles` executable links `event_gen`, which links
`AchillesCascadeInteractions` **PRIVATE** (CMakeLists line ~138), so it is not propagated to
`achilles` and the linker drops it (no referenced symbols → static initializers never run).
**Fix** (`Dockerfile.fullcascade`): force-link `AchillesCascadeInteractions` into the
`achilles` target with `LINKER:--no-as-needed`. The in-event cascade now runs cleanly
("Generated N/N events, Success", no SIGSEGV).

**Pion-propagation result** (`scripts/make_cascade_effect.py` →
`figures/cascade_effect_nue_c12.png`): ν_e RES single-π on ¹²C, cascade OFF vs ON
(E=2.222 GeV). The cascade **softens** the pion spectrum (peak ~250 → ~170 MeV, ⟨|p_π|⟩
314 → 281) and **absorbs 22%** of pions (CC1π → CC0π). The differentiable `ToyCascadeFSI`
reproduces the same softening + absorption and is **tuned to the real ACHILLES cascade**
(σ_sc≈0.17, σ_abs≈0.10 fm⁻¹ → CC0π 0.21, softening ratio 0.90) — the toy FSI's two physical
knobs match the gross cascade effect.

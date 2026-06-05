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

**Fix** (`docker/Dockerfile.cascade`, a `sed` patch before cmake): link
`AchillesCascadeInteractions` into `achilles-cascade` with `-Wl,--no-as-needed` (so the
shared lib's registrations run even though no symbols are referenced), plus `dl plugin_manager`.

## Status
☑ Showstopper resolved: cascade-enabled image **built**; the missing-registry segfault
**diagnosed and patched** in the build recipe (`docker/Dockerfile.cascade`). σ(p) parser +
self-contained config recipe ready. Cascade-oracle generation (π⁺–C/π⁺–Ar reaction σ →
Fig c12_ar40) + the model-side cascade FSIModel (Phase D de-risks the differentiability on
toy physics) are the remaining steps.

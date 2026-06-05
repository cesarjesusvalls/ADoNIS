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

## Status
☑ Showstopper resolved: cascade-enabled image **built** and the `achilles-cascade` binary
runs (version, full settings validation, interaction/decay setup). Build recipe committed
(`docker/Dockerfile.cascade`); the σ(p) parser + config recipe are ready.
_(The first build, without `-fno-visibility-inlines-hidden`, segfaulted in the cascade
interaction setup even on a minimal NucleonNucleon config — the documented factory-registry
issue; the rebuild adds the flag. Cascade-oracle generation + the model-side cascade
FSIModel (which Phase D de-risks on toy physics) follow.)_

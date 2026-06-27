# ACHILLES oracle container

ADoNIS validates its differentiable model against **ACHILLES**, the reference
neutrino event generator. To make this reproducible from a fresh clone on any
machine — and in CI — ACHILLES is published as a public container image:

```
ghcr.io/cesarjesusvalls/achilles:oracle
```

It is the **single source of truth**: it provides both the `achilles` binary that
generates the oracle events *and* the two data tables the ADoNIS model side reads
(`dcc_EW.dat`, the DCC electroweak amplitudes; `Spectral_Functions/pke12p_tot.data`,
the ¹²C spectral function). The binary, the tables, and the oracle therefore can
never drift apart.

- **Base / build:** `debian:bookworm-slim` (glibc), `linux/amd64`, Release, with
  Sherpa/ROOT off and GZIP on. HepMC3 + deps are fetched via CPM at build time.
- **Provenance:** built from ACHILLES commit **`a4f761ea`** (`git describe`
  → `v0.3.0-1-ga4f761ea`, "feat: Begin move from NuHepMC-v0.9 to NuHepMC-v1.0").
  `.git/` is excluded from the image, so the commit is only recorded here — update
  it whenever the image is rebuilt (and bump `DATA_CACHE_KEY` in `ci.yml`).
- **Size:** ~425 MB. Verified to run the neutrino-CC single-pion config end-to-end.

## Use the image

GitHub Actions runners are amd64, so the image runs natively there (no `--platform`).
On Apple Silicon it runs under emulation (slower); pass `--platform linux/amd64`.

```bash
docker pull ghcr.io/cesarjesusvalls/achilles:oracle

# generate events: mount an output dir and point the config's Output->Name at it
docker run --rm -v "$PWD/out":/out ghcr.io/cesarjesusvalls/achilles:oracle /out/run.yml
# -> ./out/<name>.hepmc   (the config's data/... includes resolve from /achilles)
```

The working directory inside the image is `/achilles`; a run config's `data/...`
includes resolve relative to it, so keep the workdir at `/achilles` (the default)
and write output to a **mounted** directory via an **absolute** path under `/out`.

## ⚠️ Cascade/FSI runs: use the NATIVE arm64 image, NOT the amd64 `:oracle` under `--platform`

The published `:oracle` image is **amd64**. On Apple Silicon, `--platform linux/amd64` runs it under
**Rosetta emulation**, and the **intra-nuclear cascade (`Cascade: Run: True`) SIGSEGVs at setup** —
the self-registering interaction-factory maps (NucleonNucleon/PionInteraction) don't survive
emulation. No-cascade runs (e.g. electron QESpectral) *do* survive emulation, which is misleading.
This is invocation, **not** an Ar/host/data problem (carbon crashes identically under emulation+cascade).

Run FSI / cascade neutrino generation on the **locally-built native arm64** images instead, with **no
`--platform`** (native arm64 on the M-series host → no emulation → cascade works):

```bash
# full neutrino + cascade generator (QE_Spectral_Func + RES_Spectral_Func + Cascade): use :fullcascade
docker run --rm -v "$PWD/_oracle_out":/out --entrypoint /achilles/bin/achilles \
  achilles:fullcascade /out/<run_card>.yml          # NO --platform

# pion+nucleus cascade-only transparency: achilles:cascade / :scatrec, entrypoint achilles-cascade
```
Verified end-to-end (carbon + Argon, QE+RES+FSI, T2K flux): "Event Run Concluded - Success!", valid
hepmc.  These native images carry both `achilles` and `achilles-cascade` and the full `data/`
(incl. Ar: `40Ar.yml`, `rho_Ar_{p,n}.txt`, `AR40_configs_RMF_achilles.out.gz`, `pke40{p,n}_tot.data`).
Note: a sporadic SIGSEGV-on-exit (exit 139) can still occur even natively, but the hepmc written
before it stays valid — batch over seeds if it bites.  Other local arm64 images: `achilles:scatrec`
(instrumented stderr dumps), `achilles:cascade*` (cascade-only).  The amd64 `:oracle` image is fine
for **no-cascade** runs (electron (e,e'), no-FSI references) even under emulation.

## Get the data tables locally

The two tables are **not** committed to this repo (one is ~38 MB). ADoNIS resolves
their location from the `ACHILLES_DATA` environment variable, which defaults to
`./achilles_data/` at the repo root (git-ignored). Populate it once:

```bash
python scripts/fetch_achilles_data.py        # docker-cp the tables from the image
# or, if you have a local ACHILLES checkout/build:
export ACHILLES_DATA=/path/to/Achilles/data
```

After that, everything (closure tests, figures, fits) runs from the clone alone.

## How CI uses it

- `.github/workflows/ci.yml` extracts the two tables from the image (cached on
  `DATA_CACHE_KEY`) into `achilles_data/`, sets `ACHILLES_DATA`, runs the
  per-module closure + oracle gates, and regenerates the figures. It fetches the
  oracle histograms from the `oracle-data` GitHub Release if present (the oracle
  gates skip otherwise).
- `.github/workflows/oracle.yml` runs the image to generate neutrino-CC hepmc,
  parses it to `oracle_finalstate.npz` (`scripts/oracle_from_hepmc.py`), validates
  the model against it, and publishes the npz as the `oracle-data` release asset
  (overwritten in place, so it is never versioned in git).

## Rebuilding the image

The Dockerfile (`Dockerfile.debian`) and the full build recipe live in the
ACHILLES source tree, not in this repo. Flags required/defensive:
`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` (docopt declares an ancient
`cmake_minimum_required`); `-DCMAKE_CXX_FLAGS="-fno-visibility-inlines-hidden"`
(keeps the self-registering factory maps coalescing across the shared libs);
**`-DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed"`** (CRITICAL for the cascade
executables: `libAchillesCascadeInteractions.so` is only transitively linked and
its interactions self-register, so the default `--as-needed` drops it from the
binary's `NEEDED` → no interactions register → `InteractionFactory::List()` empty →
`GetSuggestion(empty,…)` does `opts[0]` on an empty vector → SIGSEGV at cascade
setup, *after* the DecayHandler warnings, before "Cascade running").  Also build
SHARED (`-DBUILD_SHARED_LIBS=ON`), `-DACHILLES_ENABLE_CASCADE_TEST=ON` (gives
`bin/achilles-cascade`, the CrossSection-mode runner), `-DACHILLES_ENABLE_GZIP=ON`,
`-DCMAKE_BUILD_TYPE=RelWithDebInfo`.  Verify the fix with
`readelf -d bin/achilles-cascade | grep CascadeInteractions` (must be present).
Add a `.dockerignore` (`_dockerbuild`, `_resrun_out`, `_drvout`, `_zmtxout`,
`*.hepmc`, `.git`, …) — the source tree accumulates multi-GB run outputs that
otherwise bloat the build context to >10 GB.  After a rebuild, update the
provenance commit above and bump `DATA_CACHE_KEY` in `ci.yml`.

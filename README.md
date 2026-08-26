# ADoNIS

**A Differentiable generatOr of Neutrino Interaction Samples** — a modular, differentiable
Monte-Carlo surrogate for ACHILLES neutrino interactions, built so the underlying physics
parameters (e.g. the axial mass M_A) can be tuned against data by gradient descent.

The chain is composed from swappable components (flux, nuclear model, primary channel, FSI)
around the **kind-1 sample/reweight contract**: a fixed *detached* proposal is sampled once,
and only a pure-JAX, differentiable weight carries the physics knobs — so gradients are exact
and one sample can be reweighted/differentiated over many parameter values.

## Quick start

```bash
pip install -r requirements.txt
```

ADoNIS reads two ACHILLES tables that are **not** committed here (the DCC electroweak amplitudes,
~38 MB, and the spectral functions).  Resolve them with `ACHILLES_DATA`, which defaults to
`data/achilles/` at the repo root:

```bash
export ACHILLES_DATA=/path/to/Achilles/data      # a local ACHILLES checkout or build
```

or extract them from the public oracle image `ghcr.io/cesarjesusvalls/achilles:oracle`
(see `docs/CONTAINER.md`).  ACHILLES itself is never invoked at run time -- ADoNIS reimplements
the physics differentiably and uses ACHILLES only as a validation oracle -- but it does consume
those tabulated inputs.

## Layout

```
adonis/                 THE package -- everything reusable, and it stands alone
  constants.py          ACHILLES physical constants
  core/                 sample/reweight contract, typed PhysicsParams, EventRecord, autodiff
  flux/                 neutrino / electron / tagged-hadron beams
  nuclear/              spectral functions, densities, nuclear targets
  channels/             hard vertex: QE (CC/NC/EM), RES via the ANL-Osaka DCC tables, currents
  fsi/                  ONE jitted, differentiable intranuclear cascade + interaction models
  reweight/             the kind-1 weight: exact reduced-quadratic hard-vertex + FSI reweight
  workflow/             bank generation (one generator, every probe), selection, plotting
  detector/             smearing / efficiency models
  measurements/         published data releases (T2K, MINERvA, ...) as loaders
  analysis/             sample layer: binning, beams, Gate-I information gating, caching
  stats/                Fisher / Gaussian covariance machinery
  fit/                  fitters, minimisers, NUTS, the staged fit runner, benchmarks
  unfold/               response, templates, flux/detector priors, the unfolding run
  oracle/               ACHILLES parsers + runner (used to VALIDATE, never called at run time)
configs/                the run definitions -- banks/, samples/, fits/, achilles/
analysis/paper/         thin consumers of the above; one sub-package per topic (see its __init__)
tests/                  closure (autodiff==FD), oracle gates, layering
docs/                   design notes, plans, container + input provenance
```

`adonis/` never imports `analysis/`; `tests/test_layering.py` enforces it.

## Building the paper figures

Every topic exposes the same entry point, so there is exactly one way to build a figure:

```bash
python -m analysis.paper.validation.make            # ADoNIS vs ACHILLES, one YAML spec per figure
python -m analysis.paper.grad_info.make             # Fisher information + per-bin gradient shapes
python -m analysis.paper.inference.make  --label sec4_P2    # closure, rates, corner
python -m analysis.paper.unfolding.make  --label sec5f10    # unfolding example, budget, correlations
python -m analysis.paper.performance.make           # minimiser scaling; GPU vs CPU generation
```

`--label` selects which run under `output/altgen/` to read. **Pass it.** Several runs of the same
study sit side by side there and the module defaults are historical, so a stale label renders a
perfectly healthy figure of the wrong thing.

The inputs these consume -- event banks, fit artefacts -- are produced by the package:

```bash
python -m adonis.workflow.cli configs/banks/nu_T2K_C.yaml --out $OUT/nu_T2K_C
python -m adonis.analysis.gate1                                     # the 28-knob Jacobian
python -m adonis.fit configs/fits/sec4_P1.yaml --stage closure
python -m adonis.unfold.run configs/fits/sec5_unfold.yaml --label sec5cfg
```

## Status

The full differentiable final state is validated against the ACHILLES neutrino-CC oracle
(per-observable χ²/ndf ≈ 1 for the exclusive distributions; cosθ*_π, |p_π|, lepton/nucleon
kinematics), the predictions are differentiable in M_A with an exact gradient (autodiff vs
finite-difference ~1e-5), and M_A is recovered by gradient descent (high-stats closure lands
on the truth). See `docs/STATUS.md` for the full history and `docs/STRATEGY.md` for the plan.

Each module carries a closure test (standalone differentiability) and, where applicable, an
oracle test (physics validity), exposed uniformly as `module.closure_test()` /
`module.oracle_test()` (defaults skip where not applicable). New channels / nuclear models /
FSI subclass the respective ABCs (`adonis.core.process.Channel`,
`adonis.nuclear.base.NuclearModel`, `adonis.fsi.base.FSIModel`).

## Continuous integration

The ACHILLES oracle is published as a public container image,
`ghcr.io/cesarjesusvalls/achilles:oracle` (built from a pinned ACHILLES commit; see
`docs/CONTAINER.md`). It is the single source of truth for CI — both the `achilles`
binary that generates the oracle and the data tables (`dcc_EW.dat`, the spectral function)
the ADoNIS model side reads. The loaders resolve those via the `ACHILLES_DATA` env var.

- **`.github/workflows/ci.yml`** (push / PR): runs the per-module closure + oracle pytest
  gates and regenerates the verification figures (uploaded as build artifacts). It extracts
  the data tables from the image (cached) and fetches the oracle from the `oracle-data`
  release if present (oracle gates skip otherwise).
- **`.github/workflows/oracle.yml`** (manual / monthly): runs the image to generate
  neutrino-CC hepmc, parses it to `oracle_finalstate.npz`, validates the model against it,
  and publishes it as the `oracle-data` release asset (overwritten in place — kept out of
  git history). Trigger from the Actions tab; for a quick check use small `nevents`/`nbatches`.

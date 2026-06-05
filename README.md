# ADoNIS

**A Differentiable generatOr of Neutrino Interaction Samples** — a modular, differentiable
Monte-Carlo surrogate for ACHILLES neutrino interactions, built so the underlying physics
parameters (e.g. the axial mass M_A) can be tuned against data by gradient descent.

The chain is composed from swappable components (flux, nuclear model, primary channel, FSI)
around the **kind-1 sample/reweight contract**: a fixed *detached* proposal is sampled once,
and only a pure-JAX, differentiable weight carries the physics knobs — so gradients are exact
and one sample can be reweighted/differentiated over many parameter values.

## Layout

```
adonis/                 the package
  constants.py          ACHILLES physical constants
  params.py             PhysicsParams (tunable pytree knobs) + GenConfig (static config)
  core/                 autodiff, sample/Channel contract, EventRecord, Generator,
                        histogram utils, validation harness
  flux/                 FluxModel + Monochromatic
  nuclear/              NuclearModel + SpectralFunction
  primary/dcc/          the ANL-Osaka DCC single-pion channel (amplitudes, hadron/lepton
                        tensors, differential current, two-body decay, DCCSinglePion)
  fsi/                  FSIModel + NoFSI
  observables/          W, Q2, cos(theta*), |p_pi|, lepton/nucleon kin, TKI + registry
  signal/               SignalDef (particle-content / kinematic selections)
  analysis/             fit (forward-mode), comparison utilities
  data/oracle/          ACHILLES hepmc parsers (event_kin_full)
scripts/                thin drivers (validate / make figures / generate oracle)
tests/                  per-module closure (autodiff==FD) + oracle gates
data/                   oracle/ (targets+inputs), model/ (events), cache/ (fit/plot caches)
docs/                   STATUS.md, INPUTS.md
archive/                legacy Phase-1 toy code + the pre-refactor diffpi package
```

## Quick start

```bash
pip install -r requirements.txt          # jax, numpy, matplotlib
```
```python
import jax
from adonis import GenConfig, DCCSinglePion, Generator, PhysicsParams, observables as obs

gen = Generator(DCCSinglePion(GenConfig(spline=False)))
ev = gen.generate(jax.random.PRNGKey(0), 100_000)     # EventRecord (full lab final state)
W = obs.W(ev)                                          # any observable, differentiable in knobs
```

Run from the repo root (scripts add the root to `sys.path`):
```bash
python scripts/validate_final_state.py    # final state vs ACHILLES oracle (-> figures/)
python scripts/make_ma_fit.py             # M_A closure (cached; ADONIS_NDATA/NMODEL/ITERS to override)
python scripts/make_diff_figures.py       # exact M_A gradients of exclusive predictions
python -m pytest tests/ -q                # or run individual tests/*.py
```

## Status

The full differentiable final state is validated against the ACHILLES neutrino-CC oracle
(per-observable χ²/ndf ≈ 1 for the exclusive distributions; cosθ*_π, |p_π|, lepton/nucleon
kinematics), the predictions are differentiable in M_A with an exact gradient (autodiff vs
finite-difference ~1e-5), and M_A is recovered by gradient descent (high-stats closure lands
on the truth). See `docs/STATUS.md` for the full history and `../STRATEGY.md` for the plan.

Each module carries a closure test (standalone differentiability) and, where applicable, an
oracle test (physics validity). New channels / nuclear models / FSI subclass the respective
ABCs (`adonis.core.process.Channel`, `adonis.nuclear.base.NuclearModel`,
`adonis.fsi.base.FSIModel`).

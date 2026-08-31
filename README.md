# ADoNIS

<img src="assets/adonis_logo.png" alt="" width="150" align="right">

A differentiable generator of neutrino interaction samples.

ADoNIS generates neutrino-nucleus interaction samples in JAX, so the derivative of any observable
with respect to any physics parameter is available by automatic differentiation rather than by
finite differences. A generated sample is a *bank*: a set of events with the per-event quantities
needed to reweight them to new parameter values, which makes a fit a function evaluation over a
fixed sample rather than a regeneration loop.

The physics is a modular chain -- flux, nuclear model, primary interaction channel, final-state
cascade -- assembled from interchangeable components and validated against
[ACHILLES](https://arxiv.org/abs/2205.06378).

## What is in this repository

    adonis/     the library: generation, reweighting, the intranuclear cascade, fitting,
                unfolding.  Installable, and independent of any particular analysis.
    analysis/   the application layer for one paper: its fit campaign, benchmarks and figures.
                Deliberately not packaged -- it encodes one set of study choices.
    configs/    YAML for everything the code runs: samples, fits, generation, paths.
    tests/      the test suite.
    docker/     container recipes for the ACHILLES reference generator.
    jobs/       SLURM helpers for running the heavy stages on a cluster.  Optional; every
                stage also runs directly.

The split matters: `adonis/` never imports `analysis/`, and a test enforces it. If you want to
generate and fit neutrino samples, you need `adonis/` alone. If you want to reproduce the paper,
you need the repository.

## Install

    git clone https://github.com/cesarjesusvalls/ADoNIS.git && cd ADoNIS
    python -m venv .venv && . .venv/bin/activate
    pip install -e ".[plots,dev]"

For GPU, install the matching JAX wheel afterwards:

    pip install "jax[cuda12]"

A GPU is not required. It is roughly an order of magnitude faster for bank generation and for the
cascade, and the fits are practical without one at reduced statistics.

## Quick start

ADoNIS reads tabulated inputs that ship with the ACHILLES reference image:

    scripts/fetch_achilles_data.sh
    export ACHILLES_DATA=$PWD/achilles_data
    export JAX_ENABLE_X64=1

Generate a small bank and select a signal over it.  `configs/banks/getting_started.yaml` is the
smallest config that still runs the whole chain; the physics samples are the `nu_*`, `nc_*` and
`beam_*` configs, and each carries the event count its target needs:

    python -m adonis.workflow.cli configs/banks/getting_started.yaml \
        --out output/banks/getting_started/part_0 --seed0 0 --n-seeds 1
    python -m adonis.workflow.merge_bank \
        --parts output/banks/getting_started --out output/banks/getting_started/merged

Generation first sizes its final-state-interaction buffers from a calibration pass over a fixed
number of events, per channel.  For a bank this small that calibration, not the bank, is most of
the work; it is a fixed cost that disappears at production sizes.

The full inference chain at its smallest size, to check an installation end to end, is
`configs/fits/minimal.yaml` -- see the reproduction instructions below.

## Reproducing the paper

See [analysis/paper/REPRODUCE.md](analysis/paper/REPRODUCE.md). Every step runs at two statistics:

  - **low** -- confirms the whole chain runs, on one machine.  The figures come out, with large
    error bars.  Run this first; it catches a broken setup in minutes rather than after a long job.
  - **published** -- the statistics behind the paper.  Same commands, larger event counts, sharded
    across a cluster.  This is what reproduces the figures as printed.

## Comparing against ACHILLES

The ADoNIS-vs-ACHILLES comparisons run ACHILLES in a container, so no local build is needed:

    python -m analysis.oracle_tools.run_achilles configs/achilles/<card>.yml --nevents 1000

See REPRODUCE.md for which images are needed and where they come from.

## Citing

If you use this code, please cite:

```bibtex
@article{Jesus-Valls:2026nnj,
    author = "Jes{\'u}s-Valls, C{\'e}sar",
    title = "{ADoNIS: A Differentiable generatOr of Neutrino Interaction Samples}",
    eprint = "2608.25107",
    archivePrefix = "arXiv",
    primaryClass = "hep-ex",
    month = "8",
    year = "2026"
}
```

## License

MIT.  See LICENSE.

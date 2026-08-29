"""Computational-performance figures.

    python -m analysis.paper.performance.make                  # every figure
    python -m analysis.paper.performance.make minimizers       # only figures whose key contains this

  minimizers   gradients vs no gradients in the optimiser.  Reads <results>/bench_fair_*.npz,
               produced by `python -m analysis.benchmarks.bench_fair`.
  generation   CPU vs GPU event-generation throughput.  Reads the `stage_seconds` block
               adonis.workflow.generate_bank writes into every bank manifest.

The generation figure needs its bank directories named explicitly: set ADONIS_PERF_BANKS to a
semicolon-separated list of `Label=path`, or pass them after `generation`:

    python -m analysis.paper.performance.make generation \
        "EPYC 7713 (1 core)=output/bench/gen_cpu1" "A100-SXM4=output/bench/gen_ampere"

A bank with no `stage_seconds` (pre-dates per-stage timing) is rejected rather than plotted as zero.
"""
import os

from analysis.paper._driver import run


def bank_specs(args):
    """`Label=path` pairs for the generation figure, from the command line or the environment."""
    return args.extra or [s for s in os.environ.get("ADONIS_PERF_BANKS", "").split(";") if s.strip()]


FIGURES = {
    "minimizers": ("analysis.paper.performance.fig_minimizers", "main", lambda a: ()),
    "generation": ("analysis.paper.performance.fig_generation", "main",
                   lambda a: tuple(bank_specs(a))),
}


def _skip(key, args):
    if key == "generation" and not bank_specs(args):
        return "no bank directories given (pass 'Label=path' args or set ADONIS_PERF_BANKS)"
    return None


if __name__ == "__main__":
    run("analysis.paper.performance.make", FIGURES, skip=_skip)

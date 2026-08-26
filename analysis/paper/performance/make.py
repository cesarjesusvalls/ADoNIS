"""THE single entry point for the computational-performance figures.

    python -m analysis.paper.performance.make                  # every figure
    python -m analysis.paper.performance.make minimizers       # only figures whose key contains this

Two claims, two figures, both from artefacts the core package writes:

  minimizers   what gradients buy the OPTIMISER.  Reads output/altgen/bench_fair_*.npz, produced by
               `python -m analysis.benchmarks.bench_fair` -- Gauss-Newton against MIGRAD with and without a
               supplied gradient, over a grid of dial counts and statistical realisations.
  generation   what a GPU buys EVENT GENERATION.  Reads the `stage_seconds` block that
               adonis.workflow.generate_bank writes into every bank manifest, so the inputs are real
               production banks rather than a separate benchmark path.

BANK DIRECTORIES ARE NOT DISCOVERABLE, so the generation figure needs them named: set ADONIS_PERF_BANKS
to a semicolon-separated list of `Label=path` (one bank per device), or pass them after `generation`:

    python -m analysis.paper.performance.make generation \
        "EPYC 7713 (1 core)=output/bench/gen_cpu1" "A100-SXM4=output/bench/gen_ampere"

Any bank generated before per-stage timing existed has no `stage_seconds` and is rejected loudly rather
than plotted as zero.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "adonis").is_dir())
sys.path.insert(0, str(ROOT))

FIGURES = {
    "minimizers": "analysis.paper.performance.fig_minimizers",
    "generation": "analysis.paper.performance.fig_generation",
}


def _selected(names):
    return list(FIGURES) if not names else [k for k in FIGURES if any(n in k for n in names)]


def _bank_specs(extra):
    if extra:
        return extra
    env = os.environ.get("ADONIS_PERF_BANKS", "")
    return [s for s in env.split(";") if s.strip()]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    one = "--one" in argv
    args = [a for a in argv if not a.startswith("--")]
    # a token containing '=' is a bank spec for the generation figure, not a figure name
    names = [a for a in args if "=" not in a]
    banks = [a for a in args if "=" in a]
    keys = _selected(names)
    if not keys:
        raise SystemExit(f"[figures] no figure matches {names}; known: {sorted(FIGURES)}")

    if one:
        import importlib
        from analysis.paper import style
        style.use()
        for k in keys:
            mod = importlib.import_module(FIGURES[k])
            mod.main(*(_bank_specs(banks) if k == "generation" else ()))
        return

    ok = 0
    for k in keys:
        spec = _bank_specs(banks) if k == "generation" else []
        if k == "generation" and not spec:
            print("  [skip generation] no bank directories given "
                  "(pass 'Label=path' args or set ADONIS_PERF_BANKS)", flush=True)
            continue
        print(f"\n=== {k} ===", flush=True)
        cmd = [sys.executable, "-m", "analysis.paper.performance.make", "--one", k, *spec]
        ok += subprocess.run(cmd, cwd=str(ROOT)).returncode == 0
    print(f"\n{ok} figures built", flush=True)


if __name__ == "__main__":
    main()

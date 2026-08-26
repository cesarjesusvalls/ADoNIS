"""THE single entry point for the inference (closure + uncertainty) figures.

    python -m analysis.paper.inference.make                    # every figure
    python -m analysis.paper.inference.make corner             # only figures whose key contains this
    python -m analysis.paper.inference.make --one rates        # render one figure IN-PROCESS (no subprocess)
    python -m analysis.paper.inference.make --label sec4_P2    # which fit the figures are built from

Same shape as analysis/paper/validation/make.py: one command, one figure per subprocess (each figure
imports jax and mutates global matplotlib state, so sharing a process makes the later ones depend on the
earlier ones).  What differs is that these figures are driven by a fit LABEL rather than by a YAML spec --
the fit definition lives in configs/fits/, and the label selects which run under output/altgen/ to read.

THE LABEL IS NOT OPTIONAL IN PRACTICE.  Each figure module carries its own default, and those defaults
were the labels of whatever run was current when the module was written -- fig_rates defaulted to
sec4_P1 while the published figure came from sec4_P2, which is a difference of MC-error cut and live-bin
count that a reader cannot see.  Passing --label here sets ONE label for every figure in the set, so a
rebuilt figure set is internally consistent by construction.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "adonis").is_dir())
sys.path.insert(0, str(ROOT))

FIGURES = {
    "rates":   ("analysis.paper.inference.fig_rates", "main", lambda L: (L,)),
    "closure": ("analysis.paper.inference.fig_closure_summary", "main", lambda L: (L, f"{L}_ens")),
    "corner":  ("analysis.paper.inference.fig_corner_all", "main", lambda L: (L, L)),
}

DEFAULT_LABEL = "sec4_P2"


def _selected(names):
    return list(FIGURES) if not names else [k for k in FIGURES if any(n in k for n in names)]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    one = "--one" in argv
    label = DEFAULT_LABEL
    if "--label" in argv:
        i = argv.index("--label")
        label = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    names = [a for a in argv if not a.startswith("--")]
    keys = _selected(names)
    if not keys:
        raise SystemExit(f"[figures] no figure matches {names}; known: {sorted(FIGURES)}")

    if one:
        import importlib
        from analysis.paper import style
        style.use()
        for k in keys:
            mod, fn, args_of = FIGURES[k]
            getattr(importlib.import_module(mod), fn)(*args_of(label))
        return

    ok = 0
    for k in keys:
        print(f"\n=== {k}  (label={label}) ===", flush=True)
        cmd = [sys.executable, "-m", "analysis.paper.inference.make", "--one", "--label", label, k]
        ok += subprocess.run(cmd, cwd=str(ROOT)).returncode == 0
    print(f"\n{ok}/{len(keys)} figures built", flush=True)


if __name__ == "__main__":
    main()

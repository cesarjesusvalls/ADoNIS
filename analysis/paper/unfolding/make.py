"""Entry point for the unfolding figures.

    python -m analysis.paper.unfolding.make                     # every figure
    python -m analysis.paper.unfolding.make corr                # only figures whose key contains this
    python -m analysis.paper.unfolding.make --label sec5f10     # which unfolding run to read

The run itself is produced by the core package:

    python -m analysis.campaign.unfold_run configs/fits/sec5_unfold.yaml --label sec5cfg

which writes output/altgen/<label>_unfold.npz.  These figures are pure consumers of that npz; every
physics choice is in the config, stamped into the npz.  Pass --label explicitly -- output/altgen holds
multiple unfolding runs and the module defaults may not match the one you want.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "adonis").is_dir())
sys.path.insert(0, str(ROOT))

FIGURES = {
    "unfolded": ("analysis.paper.unfolding.fig_unfolded_lep", "main"),
    "budget":   ("analysis.paper.unfolding.fig_budget", "main"),
    "corr":     ("analysis.paper.unfolding.fig_correlation", "main"),
}

DEFAULT_LABEL = "sec5f10"


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
            mod, fn = FIGURES[k]
            getattr(importlib.import_module(mod), fn)(label)
        return

    ok = 0
    for k in keys:
        print(f"\n=== {k}  (label={label}) ===", flush=True)
        cmd = [sys.executable, "-m", "analysis.paper.unfolding.make", "--one", "--label", label, k]
        ok += subprocess.run(cmd, cwd=str(ROOT)).returncode == 0
    print(f"\n{ok}/{len(keys)} figures built", flush=True)


if __name__ == "__main__":
    main()

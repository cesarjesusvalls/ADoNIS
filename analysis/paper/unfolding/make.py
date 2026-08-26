"""THE single entry point for the unfolding figures.

    python -m analysis.paper.unfolding.make                     # every figure
    python -m analysis.paper.unfolding.make corr                # only figures whose key contains this
    python -m analysis.paper.unfolding.make --label sec5f10     # which unfolding run to read

The run itself is produced by the core package, not here:

    python -m analysis.campaign.unfold_run configs/fits/sec5_unfold.yaml --label sec5cfg

which writes output/altgen/<label>_unfold.npz.  These figures are pure consumers of that npz -- every
physics choice (signal definition, binning, detector model, priors, studies) is in the config, and the
resolved config is stamped into the npz so a figure can state the run it came from.

WHY --label MATTERS HERE.  output/altgen holds several unfolding runs that differ in signal definition
and binning: a stale sec5_unfold.npz from a 9-template STV scan sits next to the 58-template lepton runs.
The figure modules default to label="sec5", which matches the stale one, and a 90-parameter correlation
matrix renders perfectly happily.  Pass the label.
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

DEFAULT_LABEL = "sec5f10"          # the run behind the published figures


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

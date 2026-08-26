"""THE single entry point for the gradient-information figures (paper sections 2 and 3).

    python -m analysis.paper.grad_info.make                          # both figures
    python -m analysis.paper.grad_info.make fisher                   # only one
    python -m analysis.paper.grad_info.make --label multisample_carbon

Both figures are views of ONE object: the 28-knob Jacobian in output/altgen/<label>.npz, produced by

    python -m adonis.analysis.gate1

`fisher` (constraints_per_subset) asks which knobs the data can constrain, per sample subset;
`gradients` (multisample_grad_per_bin) shows where in the binned spectra each constrainable knob pulls.
They were two directories built from the same npz by two commands with two different default labels,
which is exactly how the two halves of one figure pair drift apart.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "adonis").is_dir())
sys.path.insert(0, str(ROOT))

FIGURES = {
    "fisher":    ("analysis.paper.grad_info.fisher", "main"),
    "gradients": ("analysis.paper.grad_info.gradients", "main"),
}

DEFAULT_LABEL = "multisample_carbon"          # the Jacobian behind the published figures


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
        cmd = [sys.executable, "-m", "analysis.paper.grad_info.make", "--one", "--label", label, k]
        ok += subprocess.run(cmd, cwd=str(ROOT)).returncode == 0
    print(f"\n{ok}/{len(keys)} figures built", flush=True)


if __name__ == "__main__":
    main()

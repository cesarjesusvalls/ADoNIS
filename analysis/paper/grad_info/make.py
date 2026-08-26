"""Entry point for the gradient-information figures.

    python -m analysis.paper.grad_info.make                          # both figures
    python -m analysis.paper.grad_info.make fisher                   # only one
    python -m analysis.paper.grad_info.make --label multisample_carbon

Reads the Jacobian npz in output/altgen/<label>.npz, produced by `python -m analysis.campaign.gate1`.
`fisher` = Fisher info per subset; `gradients` = per-bin gradient shapes.
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

DEFAULT_LABEL = "multisample_carbon"


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

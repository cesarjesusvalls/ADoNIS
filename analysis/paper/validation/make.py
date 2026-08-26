"""THE single entry point for every ADoNIS-vs-ACHILLES paper figure (arXiv:2508.19213).

Every figure is one YAML spec in this directory; helper.render(spec) draws it (its `render:` key selects
the render function).  Nothing else generates these figures -- there is exactly one way to run them.

    python -m analysis.paper.validation.make                 # every figure
    python -m analysis.paper.validation.make fig07 fig11      # only specs whose stem contains these
    python -m analysis.paper.validation.make --no-ratio       # drop the ACH/ADO ratio strip; write *_noratio

Each figure renders in its own subprocess (isolates its jax import + matplotlib state); --one <stem>
renders a single spec in-process
(used internally by the orchestrator).
"""
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent          # .../analysis/paper/validation
# Repo root = the first ancestor that contains the `adonis` package.  Marker-based, not a hand-counted
# .parents[N] (which silently breaks if this file moves or if you count from the dir vs the file).
ROOT = next(p for p in HERE.parents if (p / "adonis").is_dir())
sys.path.insert(0, str(ROOT))
from analysis.paper import style                                 # noqa: E402
from analysis.paper.validation import helper                # noqa: E402


def specs_for(names):
    allspecs = sorted(HERE.glob("*.yaml"))
    return allspecs if not names else [p for p in allspecs if any(n in p.stem for n in names)]


def _load(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw["_path"] = str(Path(path).resolve().relative_to(ROOT))
    raw.setdefault("name", Path(path).stem)
    return raw


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    one = "--one" in argv
    no_ratio = "--no-ratio" in argv          # drop the ACH/ADO ratio strip -> *_noratio files (paper figs untouched)
    names = [a for a in argv if not a.startswith("--")]
    specs = specs_for(names)
    if not specs:
        raise SystemExit(f"[figures] no spec matches {names or '<all>'} in {HERE}")

    if one:
        style.use()
        for p in specs:
            helper.render(_load(p), show_ratio=not no_ratio)
        return

    ok = 0
    for p in specs:
        print(f"\n=== {p.stem} ===", flush=True)
        cmd = [sys.executable, "-m", "analysis.paper.validation.make", "--one", p.stem]
        if no_ratio:
            cmd.append("--no-ratio")
        ok += subprocess.run(cmd, cwd=str(ROOT)).returncode == 0
    print(f"\n{ok}/{len(specs)} figures built", flush=True)


if __name__ == "__main__":
    main()

"""THE single entry point for every ADoNIS-vs-ACHILLES paper figure (arXiv:2508.19213).

Every figure is one YAML spec in this directory; helper.render(spec) draws it (its `render:` key selects
the render function).  Nothing else generates these figures -- there is exactly one way to run them.

    python -m analysis.paper.sec1_validation.make                # ALL figures
    python -m analysis.paper.sec1_validation.make fig07 fig11     # only specs whose stem contains these
    python -m analysis.paper.sec1_validation.make --light         # skip the heavy MC figures (fig02, fig13)

Each figure renders in its own subprocess (isolates its jax import + matplotlib state), as the pipeline
always did; --one <stem> renders a single spec in-process (used internally by the orchestrator).
"""
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from analysis.paper import style                                 # noqa: E402
from analysis.paper.sec1_validation import helper                # noqa: E402


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
    light = "--light" in argv
    one = "--one" in argv
    names = [a for a in argv if not a.startswith("--")]
    specs = specs_for(names)
    if not specs:
        raise SystemExit(f"[figures] no spec matches {names or '<all>'} in {HERE}")

    if one:
        style.use()
        for p in specs:
            helper.render(_load(p))
        return

    ok = 0
    for p in specs:
        if light and (yaml.safe_load(p.read_text()) or {}).get("heavy"):
            print(f"  [skip heavy] {p.stem}", flush=True)
            continue
        print(f"\n=== {p.stem} ===", flush=True)
        cmd = [sys.executable, "-m", "analysis.paper.sec1_validation.make", "--one", p.stem]
        ok += subprocess.run(cmd, cwd=str(ROOT)).returncode == 0
    print(f"\n{ok} figures built" + ("  (heavy fig02/fig13 skipped: --light)" if light else ""), flush=True)


if __name__ == "__main__":
    main()

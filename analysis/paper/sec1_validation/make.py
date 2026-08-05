"""THE single entry point for every ADoNIS-vs-ACHILLES paper figure (arXiv:2508.19213).

Every figure is one YAML spec in this directory; helper.render(spec) draws it (its `render:` key selects
the render function).  Nothing else generates these figures -- there is exactly one way to run them.

    python -m analysis.paper.sec1_validation.make                # every figure EXCEPT the heavy opt-outs
    python -m analysis.paper.sec1_validation.make fig07 fig11     # only specs whose stem contains these
    python -m analysis.paper.sec1_validation.make --all           # include the heavy opt-outs too (fig13)

A spec tagged `heavy: true` (fig13: live INC MC + a 200k-sample angular draw every run, no cache) is
OPT-OUT of the default full build -- it is built only when named explicitly (`make fig13`) or with `--all`.
Everything else, incl. the cache/scan-backed fig02/fig03, always builds.  Each figure renders in its own
subprocess (isolates its jax import + matplotlib state); --one <stem> renders a single spec in-process
(used internally by the orchestrator).
"""
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent          # .../analysis/paper/sec1_validation
# Repo root = the first ancestor that contains the `adonis` package.  Marker-based, not a hand-counted
# .parents[N] (which silently breaks if this file moves or if you count from the dir vs the file).
ROOT = next(p for p in HERE.parents if (p / "adonis").is_dir())
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
    all_ = "--all" in argv or "--heavy" in argv
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

    # heavy specs are opt-out of the default full build: skipped unless --all, or the stem was named
    # explicitly (naming a fig is an explicit opt-in).  `--light` is accepted for back-compat (no-op now).
    include_heavy = all_ or bool(names)
    ok = skipped = 0
    for p in specs:
        if not include_heavy and (yaml.safe_load(p.read_text()) or {}).get("heavy"):
            print(f"  [skip heavy, opt-out] {p.stem}  (run `make {p.stem}` or `make --all`)", flush=True)
            skipped += 1
            continue
        print(f"\n=== {p.stem} ===", flush=True)
        cmd = [sys.executable, "-m", "analysis.paper.sec1_validation.make", "--one", p.stem]
        ok += subprocess.run(cmd, cwd=str(ROOT)).returncode == 0
    print(f"\n{ok} figures built" + (f"  ({skipped} heavy opt-out skipped; --all to include)" if skipped else ""), flush=True)


if __name__ == "__main__":
    main()

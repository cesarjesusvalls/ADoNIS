"""THE single entry point for every ADoNIS-vs-ACHILLES paper figure (arXiv:2508.19213).

Every figure is ONE YAML spec in this directory, rendered THE SAME WAY -- one command, one paper style.
A spec is dispatched by whether it carries a `render:` key:

  * pure config  (no `render:`)      -> adonis.workflow.analyze.run_analysis, under the paper style
                                        (style.paper_run_analysis_kw()).  figs 7/8/9/12 + appendix A1/A2.
  * driver       (`render: <module>`) -> that module's render(spec); the module self-styles via
                                        style.use() + the style.* palette.  figs 1/2/3/4-6/10/11/13.

Each figure renders in its OWN subprocess (isolates the per-figure jax import + matplotlib state, as the
former sec1_validation/make.py did for its drivers), so one figure crashing never takes the batch down.

    python -m analysis.paper.figures.make                 # ALL figures
    python -m analysis.paper.figures.make fig07 fig11      # only specs whose stem contains these
    python -m analysis.paper.figures.make --light          # skip the heavy MC drivers (fig02, fig13)
"""
import importlib
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from adonis.workflow.config import load_analysis_config     # noqa: E402
from adonis.workflow.analyze import run_analysis            # noqa: E402
from analysis.paper import style                            # noqa: E402


def specs_for(names):
    """All figure specs (sorted), or the subset whose file stem contains any of `names`."""
    allspecs = sorted(HERE.glob("*.yaml"))
    return allspecs if not names else [p for p in allspecs if any(n in p.stem for n in names)]


def render_one(path, light=False):
    """Render ONE figure spec IN-PROCESS.  Config specs go through run_analysis under the paper style;
    driver specs call their module's render(spec).  Returns the run_analysis result (or None)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if "render" in raw:
        if light and raw.get("heavy"):
            print(f"  [skip heavy] {Path(path).stem}", flush=True)
            return None
        return importlib.import_module(raw["render"]).render(raw)
    return run_analysis(load_analysis_config(str(path)), **style.paper_run_analysis_kw())


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    light = "--light" in argv
    one = "--one" in argv                                    # internal: render the (single) spec in-process
    names = [a for a in argv if not a.startswith("--")]
    specs = specs_for(names)
    if not specs:
        raise SystemExit(f"[figures] no spec matches {names or '<all>'} in {HERE}")

    if one:
        style.use()
        for p in specs:
            render_one(p, light=light)
        return

    ok = 0
    for p in specs:
        print(f"\n=== {p.stem} ===", flush=True)
        cmd = [sys.executable, "-m", "analysis.paper.figures.make", "--one", p.stem]
        if light:
            cmd.append("--light")
        if subprocess.run(cmd, cwd=str(ROOT)).returncode == 0:
            ok += 1
        else:
            print(f"[FAIL] {p.stem}", flush=True)
    print(f"\n{ok}/{len(specs)} figures built" + ("  (heavy fig02/fig13 skipped: --light)" if light else ""),
          flush=True)


if __name__ == "__main__":
    main()

"""Entry point for every ADoNIS-vs-ACHILLES paper figure (arXiv:2508.19213).

Each figure is one sample config in configs/samples/; helper.render draws it, choosing the render
function from the config's `render:` key.  FIGURES names which configs are paper figures -- the
directory holds measurements that have no figure.

    python -m analysis.paper.validation.make                  # every figure
    python -m analysis.paper.validation.make fig07 fig11      # only these
    python -m analysis.paper.validation.make --no-ratio       # drop the ACH/ADO ratio strip

One subprocess per figure isolates the jax import and matplotlib state; --one renders in-process.
"""
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "adonis").is_dir())
sys.path.insert(0, str(ROOT))
from analysis.paper import style
from analysis.paper.validation import helper


SAMPLES = ROOT / "configs" / "samples"

FIGURES = {
    "fig01_ee_domega":                "fig01_ee_domega",
    "fig03_pi_nucleus_sigma":         "fig03_pi_nucleus_sigma",
    "fig0456_e4nu":                   "fig0456_e4nu",
    "fig07_t2k_cc0pi":                "t2k_cc0pi",
    "fig08_t2k_cc1pi":                "t2k_cc1pi",
    "fig09_minerva_cc0pi":            "minerva_stv",
    "fig10_uboone_cc1p0pi":           "fig10_uboone_cc1p0pi",
    "fig11_uboone_nc1pi0_doublediff": "fig11_uboone_nc1pi0_doublediff",
}


def figures_for(names):
    """(figure key, sample config) for each selected figure."""
    keys = sorted(FIGURES) if not names else [k for k in sorted(FIGURES) if any(n in k for n in names)]
    return [(k, SAMPLES / f"{FIGURES[k]}.yaml") for k in keys]


def _load(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw["_path"] = str(Path(path).resolve().relative_to(ROOT))
    raw.setdefault("name", Path(path).stem)
    return raw


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    one = "--one" in argv
    no_ratio = "--no-ratio" in argv
    names = [a for a in argv if not a.startswith("--")]
    selected = figures_for(names)
    if not selected:
        raise SystemExit(f"[figures] no figure matches {names}; known: {sorted(FIGURES)}")

    if one:
        style.use()
        for key, path in selected:
            spec = _load(path)
            spec["name"] = key
            helper.render(spec, show_ratio=not no_ratio)
        return

    ok = 0
    for key, _path in selected:
        print(f"\n=== {key} ===", flush=True)
        cmd = [sys.executable, "-m", "analysis.paper.validation.make", "--one", key]
        if no_ratio:
            cmd.append("--no-ratio")
        ok += subprocess.run(cmd, cwd=str(ROOT)).returncode == 0
    print(f"\n{ok}/{len(selected)} figures built", flush=True)


if __name__ == "__main__":
    main()

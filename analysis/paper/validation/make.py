"""Every ADoNIS-vs-ACHILLES paper figure (arXiv:2508.19213).

Each figure is one sample config in configs/samples/; helper.render draws it, choosing the render
function from the config's `render:` key.  FIGURES names which configs are paper figures -- the
directory holds measurements that have no figure.

    python -m analysis.paper.validation.make                  # every figure
    python -m analysis.paper.validation.make fig07 fig11      # only these
    python -m analysis.paper.validation.make --no-ratio       # drop the ACH/ADO ratio strip
"""
from functools import partial
from pathlib import Path

import yaml

from analysis.paper._driver import ROOT, run

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


def _load(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw["_path"] = str(Path(path).resolve().relative_to(ROOT))
    raw.setdefault("name", Path(path).stem)
    return raw


def _render(key, args):
    from analysis.paper.validation import helper
    spec = _load(SAMPLES / f"{FIGURES[key]}.yaml")
    spec["name"] = key
    helper.render(spec, show_ratio="--no-ratio" not in args.flags)


if __name__ == "__main__":
    run("analysis.paper.validation.make", {k: partial(_render, k) for k in FIGURES},
        flags=("--no-ratio",))

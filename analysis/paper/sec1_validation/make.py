"""Build the ADoNIS-vs-ACHILLES validation figures from configs/analysis/*.yaml.

  python -m analysis.paper.sec1_validation.make [config.yaml ...]   # default: all configs/analysis/*.yaml

Each config -> one figure (dsigma/dx overlay + ratio panel) via adonis.workflow.analyze.run_analysis,
under analysis.paper.style (serif, PDF+PNG).  The driver stays application-style-agnostic; the paper
look is applied HERE (style.use()), keeping adonis/ free of any analysis/ dependency.
"""
import sys
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from adonis.workflow.config import load_analysis_config      # noqa: E402
from adonis.workflow.analyze import run_analysis             # noqa: E402
from analysis.paper import style                             # noqa: E402

CONFIG_DIR = ROOT / "configs" / "analysis"


def main(argv=None):
    style.use()
    argv = sys.argv[1:] if argv is None else argv
    cfgs = argv or sorted(str(p) for p in CONFIG_DIR.glob("*.yaml"))
    ok = 0
    for c in cfgs:
        print(f"\n=== {c} ===", flush=True)
        try:
            run_analysis(load_analysis_config(c)); ok += 1
        except Exception as e:
            print(f"[FAIL] {c}: {e}", flush=True)
    print(f"\n{ok}/{len(cfgs)} figures built", flush=True)


if __name__ == "__main__":
    main()

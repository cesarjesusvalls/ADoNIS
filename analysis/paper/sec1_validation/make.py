"""Build ALL the ADoNIS-vs-ACHILLES validation figures (paper arXiv:2508.19213, non-NC).

  python -m analysis.paper.sec1_validation.make            # everything: config figs + all standalone drivers
  python -m analysis.paper.sec1_validation.make --light    # skip the heavy amps2/MC drivers (fig2, fig13)
  python -m analysis.paper.sec1_validation.make a.yaml b.yaml   # ONLY those configs/analysis YAMLs

Two kinds of figure:
  * CONFIG-DRIVEN (per-event STV/TKI histograms): one AnalysisConfig YAML per figure in configs/analysis/
    (figs 7/8/9/10), run IN-PROCESS through adonis.workflow.analyze.run_analysis under the paper style.
  * STANDALONE DRIVERS (non-histogram observables -- sigma(E), sigma(p), sigma(W), dsigma/domega, angular):
    figs 1/2/3/4-6/13, each a module with its own main(); run here as SUBPROCESSES (isolates argv + the
    per-driver jax import).  fig2 (free-nucleon amps2) and fig13 (INC beam-test MC + angular sampler) are the
    HEAVY ones -> skipped by --light.
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from adonis.workflow.config import load_analysis_config      # noqa: E402
from adonis.workflow.analyze import run_analysis             # noqa: E402
from analysis.paper import style                             # noqa: E402

CONFIG_DIR = ROOT / "configs" / "analysis"
LIGHT_DRIVERS = ["fig1_ee_domega", "fig3_beams", "fig456_e4nu", "fig10_uboone"]
HEAVY_DRIVERS = ["fig2_anl_sigma", "fig13_piN_sigma"]        # amps2 / MC -> minutes; skipped by --light


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    light = "--light" in argv
    cfg_args = [a for a in argv if a != "--light"]
    style.use()
    cfgs = cfg_args or sorted(str(p) for p in CONFIG_DIR.glob("*.yaml"))
    ok = 0
    for c in cfgs:
        print(f"\n=== config: {c} ===", flush=True)
        try:
            # ticks spread across the config ratio_ylim (0.75-1.25); 0.9/1.0/1.1 crowd the short strip
            run_analysis(load_analysis_config(c), panel_w=2.4, fig_h=3.0,
                         panel_kw=style.panel_kw(ratio_yticks=[0.8, 1.0, 1.2]),
                         legend_fn=style.panel_legend,
                         label_as_xlabel=True,
                         title_kw={"fontsize": 9, "y": 0.955, "va": "top"},
                         rect_top=1.0); ok += 1
        except Exception as e:
            print(f"[FAIL] {c}: {e}", flush=True)
    print(f"\n{ok}/{len(cfgs)} config figures built", flush=True)

    # standalone drivers only when NOT running an explicit config subset
    if cfg_args:
        return
    drivers = LIGHT_DRIVERS + ([] if light else HEAVY_DRIVERS)
    dok = 0
    for mod in drivers:
        print(f"\n=== driver: {mod} ===", flush=True)
        r = subprocess.run([sys.executable, "-m", f"analysis.paper.sec1_validation.{mod}"], cwd=str(ROOT))
        if r.returncode == 0:
            dok += 1
        else:
            print(f"[FAIL] {mod}: exit {r.returncode}", flush=True)
    print(f"\n{dok}/{len(drivers)} standalone drivers built"
          + ("  (heavy fig2/fig13 skipped: --light)" if light else ""), flush=True)


if __name__ == "__main__":
    main()

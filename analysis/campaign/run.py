"""The single entry point for a fit run.

    python -m adonis.fit configs/fits/sec4_P1.yaml --stage profile2d --shard 3/36

One config, one command, one stage at a time.  This replaces "source the right eleven environment
variables, then run whichever of eleven modules corresponds to the stage you want" -- the arrangement in
which the 2-D corner ran MAP while every other stage ran MLE, because one variable was set in three job
scripts and missing from the fourth.

What it guarantees:

  * the stage sees the config and nothing else.  Per-job values (which shard, which chain) are passed
    explicitly and are the ONLY things the runner puts in the environment.
  * a stale S4_* left in the environment is a hard error, not a silent override.  That is the specific
    failure this whole layer exists to prevent, so it is checked before any work starts.
  * every stage writes a manifest beside its output recording the config digest, so a merged product can
    prove its shards came from one definition rather than from two half-edited runs.
"""
import argparse
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

from adonis.fit.config import FitConfig

STAGE_MODULE = {
    "closure":    "analysis.campaign.stages.multisample",
    "profile":    "analysis.campaign.stages.multisample_profile",
    "profile2d":  "analysis.campaign.stages.multisample_corner2d",
    "gradient2d": "analysis.campaign.stages.multisample_corner2d",
    "toys":       "analysis.campaign.stages.multisample_coverage",
    "nuts":       "analysis.campaign.nuts_run",
}

SHARD_ENV = {
    "profile":    "S4_PROF_BASE",
    "profile2d":  "S4_PAIR_BASE",
    "gradient2d": "S4_PAIR_BASE",
    "toys":       "S4_TOY_BASE",
    "nuts":       "NUTS_CHAIN",
}

_OWNED = {"S4_PROF_BASE", "S4_PROF_N", "S4_PAIR_BASE", "S4_NPAIR", "S4_ROW_BASE", "S4_NROW",
          "S4_TOY_BASE", "NUTS_CHAIN", "ADONIS_FIT_CONFIG", "ADONIS_FIT_STAGE",
          "S4_JAC_BATCH", "S4_JAX_BIN", "S4_GATE_NPZ"}


def _reject_stale_env():
    """A leftover S4_*/PHYSFIT_*/NUTS_* in the environment silently overrides the config.  Refuse."""
    bad = sorted(k for k in os.environ
                 if k.startswith(("S4_", "PHYSFIT_", "NUTS_", "ALTGEN_")) and k not in _OWNED)
    if bad:
        raise SystemExit(
            "refusing to run: these are set in the environment and would override the config:\n  "
            + "\n  ".join(f"{k}={os.environ[k]}" for k in bad)
            + "\n\nThe config is the only source of run parameters.  `unset` them and re-run.")


def _shard(stage, spec, cfg):
    """Translate --shard k/N into the per-job variables that stage understands."""
    if spec is None:
        return {}
    k, n = (int(x) for x in spec.split("/"))
    if not 0 <= k < n:
        raise SystemExit(f"--shard {spec}: expected 0 <= k < N")
    if stage in ("profile2d", "gradient2d"):
        st = cfg.stage(stage)
        npair = len(st["dials"]) * (len(st["dials"]) - 1) // 2
        rows_per = max(1, -(-int(st["n"]) // max(1, n // npair)))
        pair, blk = divmod(k, max(1, n // npair))
        return {"S4_PAIR_BASE": str(pair), "S4_NPAIR": "1",
                "S4_ROW_BASE": str(blk * rows_per), "S4_NROW": str(rows_per)}
    if stage == "profile":
        return {"S4_PROF_BASE": str(k), "S4_PROF_N": "1"}
    if stage == "toys":
        return {"S4_TOY_BASE": str(k * int(cfg.stage("toys").get("shard_size", 50)))}
    if stage == "nuts":
        return {"NUTS_CHAIN": str(k)}
    raise SystemExit(f"stage {stage!r} is not shardable")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="adonis.fit", description=__doc__.split("\n")[0])
    ap.add_argument("config", help="a fit config, e.g. configs/fits/sec4_P1.yaml")
    ap.add_argument("--stage", required=True, choices=sorted(STAGE_MODULE))
    ap.add_argument("--shard", default=None, metavar="k/N",
                    help="which slice of the work this process does")
    ap.add_argument("--dry-run", action="store_true", help="resolve and print, run nothing")
    a = ap.parse_args(argv)

    _reject_stale_env()
    cfg = FitConfig.load(a.config)
    if a.stage != "closure" and not cfg.has(a.stage):
        raise SystemExit(f"{cfg.name}: config declares no {a.stage!r} stage "
                         f"(has {[u['method'] for u in cfg.uncertainty]})")

    env = {"ADONIS_FIT_CONFIG": str(Path(a.config).resolve()), "ADONIS_FIT_STAGE": a.stage}
    env.update(_shard(a.stage, a.shard, cfg))
    os.environ.update(env)

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True).stdout.strip() or "unknown"
    except Exception:
        sha = "unknown"
    man = {"config": str(a.config), "digest": cfg.digest(), "stage": a.stage, "shard": a.shard,
           "git": sha, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": env}
    print(f"[adonis.fit] {cfg.name}  stage={a.stage}  shard={a.shard or 'all'}  "
          f"digest={cfg.digest()}  git={sha}", flush=True)
    for k, v in sorted(env.items()):
        print(f"             {k}={v}", flush=True)
    if a.dry_run:
        print(json.dumps(man, indent=2))
        return

    outdir = Path("output/altgen"); outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.name}_{a.stage}" + (f"_{a.shard.replace('/', 'of')}" if a.shard else "")
    (outdir / f"{tag}.manifest.json").write_text(json.dumps(man, indent=2))

    from adonis.fit.device_plan import apply_env
    apply_env(cfg, log=lambda m: print(f"             {m}", flush=True))

    sys.argv = [STAGE_MODULE[a.stage], a.config]
    runpy.run_module(STAGE_MODULE[a.stage], run_name="__main__")


if __name__ == "__main__":
    main()

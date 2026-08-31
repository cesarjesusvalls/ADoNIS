"""Entry point for a fit run.

    python -m analysis.campaign.run configs/fits/closure.yaml --stage profile2d --shard 3/36

One config, one command, one stage at a time.  Guarantees: the stage sees only the config (per-job values
like shard/chain are passed explicitly, not via ad hoc environment variables); a stale S4_*/PHYSFIT_*/
NUTS_* left in the environment is a hard error; every stage writes a manifest beside its output recording
the config digest.
"""
import argparse
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

from analysis._cli import results_dir

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
    "profile":    "ADONIS_PROFILE_BASE",
    "profile2d":  "ADONIS_PAIR_BASE",
    "gradient2d": "ADONIS_PAIR_BASE",
    "toys":       "ADONIS_TOY_BASE",
    "nuts":       "NUTS_CHAIN",
}

_OWNED = {"ADONIS_PROFILE_BASE", "ADONIS_PROFILE_COUNT", "ADONIS_PAIR_BASE", "ADONIS_PAIR_COUNT", "ADONIS_ROW_BASE", "ADONIS_ROW_COUNT",
          "ADONIS_TOY_BASE", "NUTS_CHAIN", "ADONIS_FIT_CONFIG", "ADONIS_FIT_STAGE",
          "ADONIS_JAC_BATCH", "ADONIS_JAX_BINNING", "ADONIS_JACOBIAN_NPZ"}


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
        return {"ADONIS_PAIR_BASE": str(pair), "ADONIS_PAIR_COUNT": "1",
                "ADONIS_ROW_BASE": str(blk * rows_per), "ADONIS_ROW_COUNT": str(rows_per)}
    if stage == "profile":
        return {"ADONIS_PROFILE_BASE": str(k), "ADONIS_PROFILE_COUNT": "1"}
    if stage == "toys":
        return {"ADONIS_TOY_BASE": str(k * int(cfg.stage("toys").get("shard_size", 50)))}
    if stage == "nuts":
        return {"NUTS_CHAIN": str(k)}
    raise SystemExit(f"stage {stage!r} is not shardable")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="analysis.campaign.run", description=__doc__.split("\n")[0])
    ap.add_argument("config", help="a fit config, e.g. configs/fits/closure.yaml")
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
    print(f"[fit] {cfg.name}  stage={a.stage}  shard={a.shard or 'all'}  "
          f"digest={cfg.digest()}  git={sha}", flush=True)
    for k, v in sorted(env.items()):
        print(f"             {k}={v}", flush=True)
    if a.dry_run:
        print(json.dumps(man, indent=2))
        return

    outdir = results_dir(); outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.name}_{a.stage}" + (f"_{a.shard.replace('/', 'of')}" if a.shard else "")
    (outdir / f"{tag}.manifest.json").write_text(json.dumps(man, indent=2))

    from adonis.fit.device_plan import apply_env
    apply_env(cfg, log=lambda m: print(f"             {m}", flush=True))

    sys.argv = [STAGE_MODULE[a.stage], a.config]
    runpy.run_module(STAGE_MODULE[a.stage], run_name="__main__")


if __name__ == "__main__":
    main()

"""ADoNIS bank-generation CLI -- THE single entry point for every probe.

  python -u -m adonis.workflow.cli <config.yaml> --out <outdir> [shard overrides]

There is exactly ONE generator, `generate_bank(cfg, outdir)`, which dispatches on `cfg.probe`
(weak | EM | hadron); ALL diversity lives in the config, not in the command.  Examples:

  # neutrino reweight bank (weak hard vertex, T2K flux, QE+RES)
  python -u -m adonis.workflow.cli configs/banks/nu_T2K_C.yaml --out $OUT/nu_T2K_C

  # electron (e,e') bank -- same command, probe=EM in the config (monochromatic e- beam)
  python -u -m adonis.workflow.cli configs/banks/beam_e_C.yaml --out $OUT/beam_e_C

  # tagged pi+ beam (pure-FSI cascade) -- same command, probe=hadron in the config
  python -u -m adonis.workflow.cli configs/banks/beam_pip_C.yaml --out $OUT/beam_pip_C

Production sharding is one independent SLURM array task per seed block: pass
`--seed0 $SLURM_ARRAY_TASK_ID` (+ --n-per-seed/--n-seeds) and a per-task `--out .../part_$TASK`.
Every bank is chunk_NNN.npz + manifest.json; read it back with generate_bank.load_bank().
"""
import argparse

from adonis.workflow.config import load_gen_config
from adonis.workflow.generate_bank import generate_bank


def _apply_overrides(gc, a):
    """Per-shard overrides onto a loaded GenConfig (the config carries the physics; these carry scale)."""
    if a.n_per_seed is not None: gc.n_per_seed = a.n_per_seed
    if a.n_seeds is not None:    gc.n_seeds = a.n_seeds
    if a.seed0 is not None:      gc.seed0 = a.seed0
    if a.chunk is not None:      gc.chunk = a.chunk
    if a.tag is not None:        gc.tag = a.tag
    return gc


def main(argv=None):
    ap = argparse.ArgumentParser(description="ADoNIS bank generation -- one backbone for weak/EM/hadron.")
    ap.add_argument("config", help="a GenConfig YAML (configs/*.yaml)")
    ap.add_argument("--out", default=None,
                    help="bank output dir (default: <config out_dir>/<prefix>_<material><tag>)")
    ap.add_argument("--n-per-seed", type=int, default=None, help="events per chunk (one seed = one chunk)")
    ap.add_argument("--n-seeds", type=int, default=None, help="number of chunks (seeds) this shard runs")
    ap.add_argument("--seed0", type=int, default=None, help="first seed (shard offset; e.g. $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--chunk", type=int, default=None, help="override events/chunk (dense FSI buffers scale w/ this)")
    ap.add_argument("--tag", type=str, default=None, help="bank-name suffix")
    a = ap.parse_args(argv)

    gc = _apply_overrides(load_gen_config(a.config), a)
    out = generate_bank(gc, a.out)
    print("DONE:", out, flush=True)


if __name__ == "__main__":
    main()

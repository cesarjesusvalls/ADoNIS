# jobs/ — S3DF pipeline submitters (canonical, version-controlled)

These are the reusable job-control scripts for running ADoNIS on SLAC S3DF. This repo copy is the
**edit source of truth**; the **runtime** copy lives at `/sdf/data/neutrino/cjesus/ADoNIS/jobs/`
(compute-node visible, where `generated/` scripts + logs are written). Keep them in sync (edit here,
copy to the runtime dir before submitting) — `/sdf/home` may not be mounted on compute nodes, so a
symlink is avoided.

- `submit.py` — generic SLURM submitter (milano CPU / turing GPU), sources `env.sh`.
- `merge_bank.py` — merge sharded chunk-banks into one bank dir.
- `combine_fsrich.py` / (`combine_oracle.py`) — combine per-shard ACHILLES oracle npzs.
- `achilles_run.py` — run an ACHILLES run card via Apptainer.
- `env.sh` — central S3DF environment (venv, ACHILLES_DATA, JAX float64, output root).

Not tracked: the ~40 one-shot `diag_*`/`abl_*` diagnostics + `generated/` + logs (runtime-only).

# jobs/

SLURM helpers. **Optional** — every stage they submit also runs directly, and the reproduction
instructions in `analysis/paper/REPRODUCE.md` never require them.

    source jobs/env.sh                 # local: repository paths, python from PATH, no batch system
    ADONIS_SITE=s3df source jobs/env.sh

`env.sh` holds the logic (GPU detection, JAX flags); a site file under `jobs/sites/` holds the
values. To run somewhere new, copy `jobs/sites/s3df.sh`, set the paths and SLURM names in it, and
select it with `ADONIS_SITE`. Nothing outside that file needs editing.

`submit.py` wraps a command in a batch script that sources `env.sh`, so a job runs in the same
environment as an interactive shell. Without `--submit` it writes the script and prints the command
instead of submitting.

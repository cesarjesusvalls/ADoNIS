"""Submit an ADoNIS pipeline job to SLURM.

The job body sources `jobs/env.sh`, so a batch job runs in the same environment as an interactive
one. Logs land in $ADONIS_LOGS/<name>_<timestamp>.log.  Site values -- partition, account, paths --
come from the site file env.sh selects; nothing here is specific to one cluster.

Usage:
    python jobs/submit.py --name event_bank --time 08:00:00 --cpus 32 --mem 128G \
        --cmd '$ADONIS_PY -u -m adonis.workflow.cli configs/banks/nu_T2K_C.yaml --out $ADONIS_OUT/nu_T2K_C --n-per-seed 250000 --n-seeds 4' \
        --submit

    # SLURM array (e.g. one independent shard per task; use $SLURM_ARRAY_TASK_ID in --cmd):
    python jobs/submit.py --name beam_pip --array 0-7 --time 06:00:00 --cpus 16 --mem 64G \
        --cmd '...--seed-offset $SLURM_ARRAY_TASK_ID ... --out $ADONIS_OUT/beam_pip_C/part_$SLURM_ARRAY_TASK_ID' --submit

Without --submit it writes the script and prints the sbatch command (dry run).
"""
import argparse
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

JOBS = Path(__file__).resolve().parent
ENV = str(JOBS / "env.sh")
LOGS = Path(os.environ.get("ADONIS_LOGS") or JOBS.parent / "logs")


def parse_args():
    p = argparse.ArgumentParser(description="Submit a CPU SLURM job for the ADoNIS pipeline")
    p.add_argument("--name", required=True, help="job name (also the log/script basename)")
    p.add_argument("--cmd", required=True, help="command to run after sourcing env.sh (env vars expanded IN the job)")
    p.add_argument("--time", default="08:00:00", help="wall time HH:MM:SS")
    p.add_argument("--cpus", type=int, default=32, help="cpus-per-task")
    p.add_argument("--mem", default="128G", help="memory (e.g. 128G)")
    p.add_argument("--partition", default=None,
                   help="override the partition the site file names")
    p.add_argument("--account", default=None,
                   help="override the account the site file names; omitted if neither is set")
    p.add_argument("--gpu", action="store_true", help="run on the site's GPU partition, with a GPU")
    p.add_argument("--gres", default=None, help="SLURM generic resource, e.g. 'gpu:1' (implied by --gpu)")
    p.add_argument("--array", default=None, help="SLURM array spec, e.g. '0-7' or '0-47%8' (optional)")
    p.add_argument("--dependency", default=None,
                   help="SLURM dependency spec, e.g. 'afterany:12345' (run after that job/array finishes)")
    p.add_argument("--submit", action="store_true", help="actually sbatch (default: dry run)")
    return p.parse_args()


def main():
    a = parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    (JOBS / "generated").mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    arr = "%A_%a" if a.array else "%j"
    log = LOGS / f"{a.name}_{ts}_{arr}.log"
    array_line = f"#SBATCH --array={a.array}\n" if a.array else ""
    dep_line = f"#SBATCH --dependency={a.dependency}\n" if a.dependency else ""
    var = "ADONIS_SLURM_GPU_PARTITION" if a.gpu else "ADONIS_SLURM_PARTITION"
    part = a.partition or os.environ.get(var)
    if not part:
        raise SystemExit(f"no partition: pass --partition, or set {var} in the site file "
                         f"jobs/env.sh selects (ADONIS_SITE).")
    acct = a.account or os.environ.get("ADONIS_SLURM_ACCOUNT")
    acct_line = f"#SBATCH --account={acct}\n" if acct else ""
    gres = a.gres or (os.environ.get("ADONIS_SLURM_GRES", "gpu:1") if a.gpu else None)
    gres_line = f"#SBATCH --gres={gres}\n" if gres else ""

    script = f"""#!/bin/bash
#SBATCH --job-name={a.name}
#SBATCH --output={log}
#SBATCH --error={log}
#SBATCH --partition={part}
{acct_line}#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={a.cpus}
#SBATCH --mem={a.mem}
#SBATCH --time={a.time}
#SBATCH --requeue
{gres_line}{array_line}{dep_line}
echo "=========================================="
echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME)  array_task=${{SLURM_ARRAY_TASK_ID:-none}}"
echo "node $SLURM_NODELIST  started $(date)"
echo "=========================================="
source {ENV}
echo "PY=$ADONIS_PY  OUT=$ADONIS_OUT  x64=$JAX_ENABLE_X64  cpus={a.cpus}"
export OMP_NUM_THREADS={a.cpus}
t0=$SECONDS
bash -c '{a.cmd}'
rc=$?
echo "=========================================="
echo "done $(date)  elapsed $((SECONDS-t0))s  exit=$rc"
exit $rc
"""
    script_file = JOBS / "generated" / f"{a.name}_{ts}.sh"
    script_file.write_text(script)
    script_file.chmod(0o755)

    print(f"script: {script_file}")
    print(f"log:    {log}")
    if a.submit:
        if shutil.which("sbatch") is None:
            raise SystemExit("sbatch not found: this host has no SLURM. The script was written; run "
                             "its command directly, or submit it where SLURM is available.")
        r = subprocess.run(["sbatch", str(script_file)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
        print((r.stdout or "").strip() or (r.stderr or "").strip())
        if r.returncode:
            raise SystemExit(f"sbatch failed: {r.stderr}")
    else:
        print(f"\ndry run — to submit:\n  sbatch {script_file}")


if __name__ == "__main__":
    main()

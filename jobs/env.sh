#!/bin/bash
# Environment for every ADoNIS job, batch or interactive.
#
#     source jobs/env.sh
#
# Site-specific values live in jobs/sites/<name>.sh; ADONIS_SITE selects one (default: local).
# To add a site, copy jobs/sites/s3df.sh and set the paths in it -- nothing below needs editing.

_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_site="${ADONIS_SITE:-local}"
_site_file="$_here/sites/$_site.sh"
if [ ! -f "$_site_file" ]; then
    echo "jobs/env.sh: no site file $_site_file (set ADONIS_SITE to one of: $(cd "$_here/sites" && ls *.sh | sed 's/\.sh$//' | tr '\n' ' '))" >&2
    return 1 2>/dev/null || exit 1
fi
. "$_site_file"

export ADONIS_REPO ADONIS_DATA
export ADONIS_OUT="${ADONIS_OUT:-$ADONIS_DATA/output}"
export ADONIS_LOGS="${ADONIS_LOGS:-$ADONIS_DATA/logs}"

# GPU where one is present, else CPU.  Never import jax to decide: on a login node that is both slow
# and wrong about what the job will get.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
    export ADONIS_PY="$ADONIS_PY_GPU"
    export JAX_PLATFORMS=cuda,cpu   # cpu stays available so host callbacks work
    export XLA_FLAGS=""
else
    export ADONIS_PY="$ADONIS_PY_CPU"
    export JAX_PLATFORMS=cpu
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
fi

[ -n "${ACHILLES_DATA:-}" ] && export ACHILLES_DATA
[ -n "${ACHILLES_FLUX:-}" ] && export ACHILLES_FLUX
[ -n "${ACHILLES_SIF:-}" ]  && export ACHILLES_SIF

# The FSI nominal is an exact identity only in float64.
export JAX_ENABLE_X64=1

for v in ADONIS_SLURM_PARTITION ADONIS_SLURM_ACCOUNT ADONIS_SLURM_GPU_PARTITION ADONIS_SLURM_GRES; do
    [ -n "${!v:-}" ] && export "$v"
done

cd "$ADONIS_REPO"

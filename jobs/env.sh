#!/bin/bash
# Central S3DF environment for the ADoNIS reproduce pipeline.
# Source this from every interactive test and every SLURM job:  source /sdf/data/neutrino/cjesus/ADoNIS/jobs/env.sh
#
# WHY a shared file: keeps the venv path, ACHILLES_DATA, the output root and the
# JAX flags identical between hand tests and batch jobs, so a job never behaves
# differently from what was smoke-tested.

# --- repo + data roots ------------------------------------------------------
export ADONIS_REPO=/sdf/home/c/cjesus/DIFFGEN/ADoNIS
export ADONIS_DATA=/sdf/data/neutrino/cjesus/ADoNIS           # all project output lives here
export ADONIS_OUT=$ADONIS_DATA/output
export ADONIS_LOGS=$ADONIS_DATA/logs

# --- python + JAX backend: GPU by default where a GPU is present, else CPU -----
# (DO NOT import jax on the login node.)  GPU nodes -> the jax[cuda12] venv + cuda backend;
# CPU nodes -> the lean CPU venv.  Auto-selected so bank generation lands on the GPU on turing
# without any per-job flag, while CPU-only jobs (merge, achilles parse, plotting) stay lean.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
    export ADONIS_PY=/sdf/data/neutrino/cjesus/software/venvs/adonis-cuda/bin/python
    export JAX_PLATFORMS=cuda,cpu   # cuda = default device; cpu kept so host callbacks (runaway guard) work
    export XLA_FLAGS=""
else
    export ADONIS_PY=/sdf/data/neutrino/cjesus/software/venvs/adonis/bin/python
    export JAX_PLATFORMS=cpu
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
fi

# --- ACHILLES reference inputs (extracted from achilles-oracle.sif) ----------
export ACHILLES_DATA=$ADONIS_DATA/software/achilles_data      # dcc_EW.dat, Spectral_Functions/, ...
export ACHILLES_FLUX=$ADONIS_DATA/software/achilles_flux      # T2K_nu.dat, ... (bound into cascade images)
export ACHILLES_SIF=/sdf/data/neutrino/cjesus/software/images/achilles-oracle.sif
# QMC nucleon configs are resolved by the code from a hardcoded sibling path
# <DIFFGEN>/Achilles/data/configurations/ -> we symlinked it to $ACHILLES_DATA/configurations.

# --- JAX: float64 is REQUIRED (FSI nominal identity is exact only in x64) -----
export JAX_ENABLE_X64=1                                       # (JAX_PLATFORMS/XLA_FLAGS set above)

# --- SLURM defaults --------------------------------------------------------
# CPU jobs default to milano; GPU jobs (bank generation) use turing via submit.py --gpu.
export ADONIS_SLURM_PARTITION=milano
export ADONIS_SLURM_ACCOUNT=neutrino:default
export ADONIS_SLURM_GPU_PARTITION=turing
export ADONIS_SLURM_GRES=gpu:1

cd "$ADONIS_REPO"

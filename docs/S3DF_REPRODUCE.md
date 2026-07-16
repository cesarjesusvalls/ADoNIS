# Reproducing the ADoNIS pipeline on S3DF (SLAC)

This is the **S3DF-specific** runbook that mirrors `REPRODUCE.md` but adapts it to the SLAC
Shared Data Facility: no Docker (Apptainer instead), a CPU SLURM cluster for large productions,
and all output on `/sdf/data/neutrino` instead of a laptop. Written and verified 2026-07-15.

If you are on a laptop with Docker, use `REPRODUCE.md`. If you are on S3DF, use this.

> **The one rule that explains everything below:** the S3DF *login* nodes cannot import JAX in
> reasonable time (the large `jaxlib` shared objects read cold over NFS and effectively hang —
> observed >5 min and counting). On a **compute node** the identical import is ~1.4 s. So *every*
> piece of real work — even a 200-event smoke test — runs on a compute node via `srun`/`sbatch`,
> never on the login node. This is the whole reason the pipeline is migrated to jobs.

---

## 0. Layout and where things live

Home (`/sdf/home/c/cjesus`) has a tight quota, so **all data, images, venvs and output live on
`/sdf/data/neutrino/cjesus`**. Only the git checkout stays in home.

```
/sdf/home/c/cjesus/DIFFGEN/
├── ADoNIS/                      # the git checkout (this repo). data/ is a symlink (see below)
├── Achilles/                    # sibling: symlinks into the extracted ACHILLES image tree (see §2)
│   ├── data -> .../achilles_data
│   └── flux -> .../achilles_flux
└── nuisance -> .../software/nuisance     # sibling: extracted NUISANCE data (T2K binning)

/sdf/data/neutrino/cjesus/ADoNIS/          # <-- ADONIS_DATA: ALL project output lives here
├── output/                      # productions: achilles/ adonis/ altgen/ paper/ figures/ event_bank/
├── logs/                        # SLURM job logs
├── jobs/                        # env.sh + SLURM submitters + generated scripts + job configs
├── repo_data/                   # the repo's gitignored data/ dir, redirected here (nuclear/ oracle/ cache/ …)
└── software/
    ├── achilles_data/           # extracted /achilles/data from the oracle .sif
    ├── achilles_flux/           # extracted /achilles/flux from the oracle .sif
    └── nuisance/data/           # extracted /opt/nuisance/data from the nuisance .sif

/sdf/data/neutrino/cjesus/software/         # shared, pre-existing
├── venvs/adonis/                # the Python env — Python 3.11.15, jax 0.10.1 CPU (built with uv)
└── images/
    ├── achilles-oracle.sif      # docker://ghcr.io/cesarjesusvalls/achilles:oracle (no-cascade)
    └── lucid-nuisance.sif       # NUISANCE build — source of the T2K data tables
```

**The repo `data/` dir is a symlink** → `/sdf/data/neutrino/cjesus/ADoNIS/repo_data`. `data/` is
gitignored in this repo (all inputs are fetched/regenerable), and the code reads *and writes* under
`data/` (`data/nuclear/`, `data/oracle/`, `data/cache/`). Redirecting it keeps home clean and puts
every repo-relative data path on the fast/large filesystem.

---

## 1. Environment — source `env.sh` for everything

One shared file pins the venv, the ACHILLES inputs, the output root, and the JAX flags so an
interactive test and a batch job behave identically:

```bash
source /sdf/data/neutrino/cjesus/ADoNIS/jobs/env.sh
```

It sets (see the file for the full list): `ADONIS_PY` (the venv python — **never** import jax with it
on the login node), `ACHILLES_DATA`, `ACHILLES_SIF`, `ADONIS_OUT`, `JAX_ENABLE_X64=1` (float64 is
**required** — the FSI nominal identity is exact only in x64), `JAX_PLATFORMS=cpu`, and the SLURM
partition/account defaults (`milano` / `neutrino:default`).

### Python deps
The venv is built with `uv` (no pip). `requirements.txt` deps `PyYAML` and `tqdm` were missing from
the pre-existing venv and were added once:
```bash
export UV_CACHE_DIR=/sdf/data/neutrino/cjesus/software/.uv_cache
export UV_PYTHON_INSTALL_DIR=/sdf/data/neutrino/cjesus/software/.uv_python
~/.local/bin/uv pip install --python $ADONIS_PY PyYAML tqdm
```

---

## 2. Data inputs — extracted from the .sif images (no Docker)

`scripts/fetch_achilles_data.py` needs Docker; on S3DF we extract the same files from the Apptainer
images instead. All of this is **already done** (2026-07-15); re-run only on a clean data area.

**ACHILLES inputs** (out of `achilles-oracle.sif`) — the image *is* the single source of truth:
```bash
SIF=/sdf/data/neutrino/cjesus/software/images/achilles-oracle.sif
DEST=/sdf/data/neutrino/cjesus/ADoNIS/software/achilles_data
apptainer exec --bind $DEST:/out $SIF cp -r /achilles/data/. /out/     # dcc_EW.dat, Spectral_Functions/, configurations/QMC_configs.out.gz, densities/, …
DESTF=/sdf/data/neutrino/cjesus/ADoNIS/software/achilles_flux
apptainer exec --bind $DESTF:/out $SIF cp -r /achilles/flux/. /out/    # T2K_nu.dat, …
```

**How the code finds them** — three different resolvers, all wired via symlink:
- `adonis/io.py` honours `ACHILLES_DATA` → set in `env.sh` to `…/achilles_data`.
- `adonis/xsec/spectral.py` and `adonis/xsec/flux.py` resolve **relative** paths (`data/…`,
  `flux/…`) against a hardcoded **sibling** `<DIFFGEN>/Achilles/` — so `Achilles/data` and
  `Achilles/flux` are symlinks into the extracted trees.
- `adonis/fsi/cascade_discrete.py` resolves QMC configs from `<DIFFGEN>/Achilles/data/configurations/`
  (hardcoded, no env override) → covered by the `Achilles/data` symlink.

```bash
mkdir -p /sdf/home/c/cjesus/DIFFGEN/Achilles
ln -sfn .../achilles_data /sdf/home/c/cjesus/DIFFGEN/Achilles/data     # (done as per-entry symlinks; a whole-dir symlink also works)
ln -sfn .../achilles_flux /sdf/home/c/cjesus/DIFFGEN/Achilles/flux
```

**Nuclear densities** (the repo's gitignored `data/nuclear/`) come from `achilles_data/densities/`:
carbon `c12.prova.txt` → copied to `data/nuclear/c12_density.txt`; argon `rho_Ar_{p,n}.txt` copied
as-is. (Loader wants columns `r[fm], rho[fm^-3]`; the ACHILLES files are 3-col and compatible.)

**NUISANCE T2K binning** (`tune.py`, `make_plots.py` read `../nuisance/data/T2K/…`) — extracted from
the nuisance image:
```bash
apptainer exec --bind .../software/nuisance/data:/out lucid-nuisance.sif cp -r /opt/nuisance/data/. /out/
ln -sfn .../software/nuisance /sdf/home/c/cjesus/DIFFGEN/nuisance
```
Provides `T2K/CC0pi/STV/{dpt,dat}Results.root`. (Note: `CC1pipNp_STV/xsec_*.txt`, used by one later
figure, is **not** in the image under that name — TODO if that figure is needed.)

---

## 3. Running the containers with Apptainer (not Docker)

`analysis/utils/run_achilles.py` shells out to `docker run`; S3DF has no Docker. The equivalent
Apptainer invocation for a **no-cascade oracle** run:
```bash
apptainer exec --fakeroot --writable-tmpfs --pwd /achilles \
  --bind <out_dir>:/out $ACHILLES_SIF \
  /achilles/bin/achilles /out/<card>.yml
```
- `--fakeroot --writable-tmpfs` — ACHILLES writes cache/log files under `/achilles` (read-only in
  the image); the tmpfs overlay makes that succeed. Both flags are required.
- `--pwd /achilles` — run cards use paths relative to `/achilles` (`data/…`, `flux/…`, `!include`s).
- `--bind <out>:/out` — the run card lives in `<out>` and its `Output: Name: /out/<x>.hepmc` writes back.
- native x86_64 image → **no `--platform`/emulation** (that was a laptop-on-Apple-Silicon concern).

The **cascade** images (`:cascade`, `:fullcascade`) are not yet on S3DF — see §6 TODO.

---

## 4. Small local tests (do these before ANY batch job)

Grab one interactive compute node and reuse it for fast iteration (no per-test queue wait):
```bash
salloc --no-shell --partition=milano --account=neutrino:default --nodes=1 --ntasks=1 \
       --cpus-per-task=32 --mem=128G --time=04:00:00 --job-name=adonis_interactive
squeue -u $USER          # note the JOBID, then srun into it:
srun --jobid=<JOBID> --ntasks=1 --cpus-per-task=16 bash -c '<cmd>'
```

**ADoNIS smoke** (exercises QE + RES + FSI, writes a bank) — verified ~95 s for 200 events
(dominated by fixed JIT/VEGAS warmup; amortizes at scale):
```bash
srun --jobid=<JOBID> --ntasks=1 --cpus-per-task=16 bash -c '
  source /sdf/data/neutrino/cjesus/ADoNIS/jobs/env.sh
  CC0PI_N=200 CHUNK=200 $ADONIS_PY -u analysis/t2k/differentiability/event_bank.py $ADONIS_OUT/smoke_bank'
# -> chunk_000.npz (200 QE + 172 RES, finite weights) + manifest.json
```

**ACHILLES smoke** (oracle container → hepmc → extracted npz) — verified ~50 s for 500 free-H events:
```bash
# 1) reduce a card to 500 events, place it in the /out dir:
python3 -c "raw=open('configs/achilles/run_T2K_H.yml').read(); \
  open('$ADONIS_OUT/achilles_smoke/run_T2K_H_smoke.yml','w').write( \
  raw.replace('NEvents: 100000','NEvents: 500').replace('/out/T2K_H.hepmc','/out/T2K_H_smoke.hepmc'))"
# 2) run the container on a compute node:
srun --jobid=<JOBID> --ntasks=1 --cpus-per-task=8 bash -c '
  source /sdf/data/neutrino/cjesus/ADoNIS/jobs/env.sh
  apptainer exec --fakeroot --writable-tmpfs --pwd /achilles --bind $ADONIS_OUT/achilles_smoke:/out \
    $ACHILLES_SIF /achilles/bin/achilles /out/run_T2K_H_smoke.yml'
# 3) extract observables:
srun --jobid=<JOBID> --ntasks=1 bash -c '
  source /sdf/data/neutrino/cjesus/ADoNIS/jobs/env.sh
  $ADONIS_PY -m analysis.utils.extract cc1pi $ADONIS_OUT/achilles_smoke/T2K_H_smoke.hepmc \
    $ADONIS_OUT/achilles_smoke/T2K_H_smoke_cc1pi.npz'
# -> 96 cc1pi events with dpt/dalphat/W/Q2/pn observables
```

---

## 5. Batch productions (SLURM)  — see `jobs/`

**Scale (per user, 2026-07-15): 10M events for every configuration, both ACHILLES and ADoNIS, CPU
only, shards round-robin across the `milano` and `roma` partitions** (account `neutrino:default`).
`--partition=milano,roma` as a list is REJECTED by the account validator, so round-robin = two
submissions per production (half the shard range to each partition; global shard id = seed, so the
two arrays use disjoint `--array` ranges to keep seeds distinct).

### Tooling in `jobs/`
- `env.sh` — sourced by every job (venv, ACHILLES_DATA/FLUX, JAX float64, output root, partition/account).
- `submit.py` — generic CPU submitter. `--name --cmd --time --cpus --mem --partition --array --submit`.
  Writes `jobs/generated/<name>_<ts>.sh`, logs to `$ADONIS_LOGS`. **Run it with `$ADONIS_PY`** (the
  login node's system python3.6 lacks `subprocess.capture_output`; the script itself is now 3.6-safe
  but the venv python is the habit). `$SLURM_ARRAY_TASK_ID` and `$ADONIS_*` expand at job runtime.
- `achilles_run.py` — the **Apptainer** ACHILLES runner (replaces run_achilles.py's `docker run`).
  Picks the image/binary from the card name (reusing `run_achilles._image_for`), overrides
  NEvents+Seed, writes a unique hepmc, and runs `apptainer exec --writable-tmpfs --pwd /achilles
  --bind out:/out --bind ACHILLES_DATA:/achilles/data --bind ACHILLES_FLUX:/achilles/flux`.
  **No `--fakeroot`** (the alpine cascade images lack `faked`; the data/flux binds feed every image
  the same inputs).
- `merge_bank.py` — merges sharded `part_*/chunk_*.npz` into one bank dir (unique chunk names +
  a manifest with total `n_chunks`), which is what `bank_plot.load_bank` expects.
- `test_all_achilles.sh` — small-tests every ACHILLES card (all images) → OK/FAIL summary.

### Sharding
- **ADoNIS event bank** — seeds by chunk index, so a `SEED0` env offset was added to
  `analysis/t2k/differentiability/event_bank.py` (default 0 = original behaviour). Each array task
  runs one 250k chunk with `SEED0=$SLURM_ARRAY_TASK_ID` into `event_bank_10M/part_<id>`, then
  `merge_bank.py` combines them. 40 shards × 250k = 10M.
- **ADoNIS beam banks** — `beam_bank` already takes `--seed`; shard = `--seed $SLURM_ARRAY_TASK_ID
  --out .../part_<id>`, then `merge_bank.py`.
- **ACHILLES oracles** — shard over seeds (`achilles_run.py --seed $SLURM_ARRAY_TASK_ID`), each a
  separate hepmc, then `analysis.utils.extract` per shard and combine.

### Launched (2026-07-15)
```bash
# 10M ADoNIS event bank — 40 shards, round-robin milano(0-19)/roma(20-39):
EBCMD='SEED0=$SLURM_ARRAY_TASK_ID CC0PI_N=250000 CHUNK=250000 $ADONIS_PY -u \
  analysis/t2k/differentiability/event_bank.py $ADONIS_OUT/event_bank_10M/part_$SLURM_ARRAY_TASK_ID'
$ADONIS_PY jobs/submit.py --name eb10M_milano --array 0-19  --partition milano --time 04:00:00 --cpus 16 --mem 48G --cmd "$EBCMD" --submit
$ADONIS_PY jobs/submit.py --name eb10M_roma   --array 20-39 --partition roma   --time 04:00:00 --cpus 16 --mem 48G --cmd "$EBCMD" --submit
# then: $ADONIS_PY jobs/merge_bank.py --parts $ADONIS_OUT/event_bank_10M --out $ADONIS_OUT/event_bank_10M/merged
```
Monitor: `squeue -u $USER`; `tail -f $ADONIS_LOGS/eb10M_*_<jobid>_<taskid>.log`.

Interactive testing uses a held node (re-allocate if it expires — `salloc --no-shell` allocations
can be reclaimed): `salloc --no-shell -p milano -A neutrino:default -N1 -n1 -c32 --mem 128G -t 05:00:00`,
then `srun --jobid=<ID> ...`.

---

## 6. Status / TODO

- [x] Environment, data wiring, both smoke tests verified (2026-07-15).
- [x] SLURM submission scaffolding (`jobs/submit.py`, `achilles_run.py`, `merge_bank.py`).
- [x] **Cascade ACHILLES images built on S3DF** via `apptainer build --fakeroot
  --ignore-fakeroot-command` from `docker/{cascade,fullcascade}.def` (translations of the Dockerfiles;
  build FROM alpine + the fork source at 7054913f). `achilles-fullcascade.sif` contains BOTH a working
  `achilles` (in-event cascade) and `achilles-cascade` (standalone), so `achilles-cascade.sif` is a
  symlink to it. Gotchas that cost time: (1) `--ignore-fakeroot-command` needed at *build* (alpine has
  no `faked`); (2) at *run* time drop `--fakeroot` for these alpine images and bind data/flux.
- [x] Small tests for ALL configurations pass (ADoNIS event bank/beams/Gate-I; ACHILLES every card
  across all three images). Gate-I `J` is finite; its Fisher is nan only at tiny stats (empty bins).
- [x] §4 ADoNIS banks at 10M — event bank (9.75M, merged), Gate-I Jacobian (finite 27×188),
  beam banks π⁺/p/n (~10M each, merged). `jobs/merge_bank.py` pools ALL chunks (incl. preempted
  shards without a manifest); `beam_fisher` run → `output/altgen/beam_fisher.npz` (ALL: 14/27 FIT).
- [x] §3 ACHILLES oracles at 10M — T2K_C_fsi (cc1pi_rich → `t2k_cc1pi_rich_ach_FSI_proc.npz`, 10M
  INDEPENDENT events after the seed fix) + T2K_H (cc1pi). `jobs/combine_oracle.py` merges shards.
- [x] §5 figures §1–3 + methods report — sec1 (corrected), sec2, sec3 → `output/paper/*.png`.
  **FINAL `methods_report.html` (3.38 MB, 2026-07-16): every §1–5 figure embedded, 0 missing.**
  NOTE: `physical_fit_run.py` and `physfit_report.py` were patched to honor `PHYSFIT_OBS`
  (upstream builds all 11 observables; the GENIE modes only fill a subset — see §4b notes).
- [x] §4b/§5 closures + unknown-unknowns (2026-07-16). The working recipe:
  - **GENIE fake data**: `jobs/run_genie_s3df.sh` — GENIE 3.4 (gevgen/gntpc + AR23 xsec splines) lives
    in `lucid.sif`; the generator-list configs and `make_flux_root.C` are in `ADoNIS/scripts/altgen/`
    and get bound to `/genie_cfg` + `/scripts`. **3M events is intractable** (~400 ev/s with the
    minimal splines → hours, and preemption kills it); **300k finishes in ~40 min** and is plenty for
    the closure demo. Keep the `_3M` output *name* (the fits' default `ADONIS_GST` paths expect it).
    Copy `T2K_nu.dat` to `output/altgen/` (build_fakedata `_flux()` reads it there).
  - **Fits** (`scripts/altgen/physical_fit_run.py`, one SLURM job each, 128G/16cpu, ~15-20 min on the
    10M bank): `build_physfit_datasets(obs=None)` builds ALL 11 observables, but genie mode fills
    only `tki` (5) and mecmix only `kin9` (9) → KeyError. Patched `physical_fit_run.py` to honor
    `PHYSFIT_OBS`. For report-consistency ALL report-input fits run with `PHYSFIT_OBS=tki`:
    `physfit_closure`, `physfit_closure_fluct` (`PHYSFIT_FLUCT=1`), `physfit_inject2x`,
    `physfit_inject2x_v2`, `physfit_genie` (`PHYSFIT_MODE=genie`); mecmix runs `PHYSFIT_OBS=kin9`
    (`ADONIS_LABEL=p9_mecmix`). q2-pair: `p9_closure_q2mod` (mode=closure_q2mod, default 0.2,0.3) and
    `p9_closq2_nuis` (+`PHYSFIT_Q2NUIS=0.02,0.08,0.18,0.45,1.2` — MUST match the hardcoded KNOTS in
    `physfit_q2nuis_fig.py`). closure5: `PHYSFIT_INJECT="M_A_res=0.85,kF_sf=1.10,Eb_shift=3.0,`
    `s_NN_elastic[pn]=1.25,f_NN_cex=0.40"` `ADONIS_LABEL=physfit_closure5`.
    **NEVER pass `PHYSFIT_2X=dpt>300` through bash -c unquoted** — the `>` becomes a shell redirect
    (the default is already `dpt>300`; just don't set it).
  - **Figures**: `physfit_closure5_fig.py` (fig7), `physfit_q2nuis_fig.py` (fig9),
    `physfit_mecmix_fig.py` (fig10), `physfit_report.py` (fig1–4 + PDF; needs `ADONIS_EVENT_BANK` +
    `PHYSFIT_OBS=tki`). All write `output/figures/physfit_fig*.png`.
- [ ] Other ACHILLES oracle cards at 10M (inclusive (e,e'), res1pi, free-nucleon, cascade beams, Ar)
  — validated by the small tests but not needed for §1–3/beam_fisher; generate if a figure requires.
- [ ] `CC1pipNp_STV` NUISANCE table for the CC1π STV figure (not in the nuisance image under that name).

### Gotchas discovered during the 10M run (READ THESE)
1. **ACHILLES RNG seed** — ACHILLES reads the seed from `Options/Initialize/Seed` (neutrino/EventGen)
   or top-level `Initialize/seed` (RunCascade), **NOT** `Main: Seed`. The cards pull Options from
   `!include data/default/OptionDefaults.yml` whose `Seed` defaults to a fixed `12345678`, so **every
   shard is byte-identical unless you override that key**. Symptom: the combined oracle's per-bin
   `sqrt(Σw²)` error is ~√(n_shards)× too small (e.g. δα_T fluctuates ~7% bin-to-bin but error bars are
   ~1%) because the "N×M events" are really M events copied N times. `jobs/achilles_run.py` now injects
   `Options.Initialize.Seed` (inlining OptionDefaults) + top-level `Initialize.seed`. ALWAYS verify two
   shards differ (compare a kinematic field) before trusting a combined oracle.
2. **ACHILLES log write permission** — ACHILLES writes `achilles.log`/`References.txt` to its CWD
   (`/achilles`, root-owned). Without `--fakeroot` (which the alpine cascade images can't use) those
   writes are intermittently `Permission denied` → SIGABRT (~50% of batch shards fail). Fix: bind
   writable host files over those two paths (`achilles_run.py` does this). Do NOT re-add `--fakeroot`.
3. **Preemption** — `neutrino:default` is preemptible on BOTH milano and roma; shards die mid-run.
   `submit.py` sets `#SBATCH --requeue`; `jobs/backfill.py` resubmits missing coverage (banks are
   reconciled by chunk-count, not shard-completion, since finished chunks persist).
4. **Interactive `salloc --no-shell` nodes get reclaimed** roughly hourly — re-allocate when `srun
   --jobid` says "expired". Batch jobs are unaffected.

### Key path wiring recap (all symlinks, to keep home light)
- repo `data/`   → `$ADONIS_DATA/repo_data`   (gitignored; holds `nuclear/` densities + `oracle/`, `cache/`)
- repo `output/` → `$ADONIS_DATA/output`      (gitignored; scripts write CWD-relative `output/...`)
- `../Achilles/data` + `../Achilles/flux` → extracted ACHILLES image trees (spectral fns, dcc_EW,
  QMC configs, T2K flux; resolved by `adonis/xsec/{spectral,flux}.py` + `fsi/cascade_discrete.py`)
- `../nuisance/data` → extracted NUISANCE tree (T2K CC0pi STV binning for `tune.py`)
- densities: ACHILLES `c12.prova.txt` → `data/nuclear/c12_density.txt`; `rho_Ar_{p,n}.txt` as-is.

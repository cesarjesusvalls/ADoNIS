# Regenerating everything: samples, banks, and the §1–§4 figures

The single runbook. It replaces `analysis/paper/REPRODUCE.md`, `docs/S3DF_REPRODUCE.md`,
`docs/RUNNING.md`, `docs/sec4_figures.md`, `docs/sec2_sec3_figures.md` and `docs/sec4_sec5_figures.md`,
and covers sections 1-4 (section 5 was removed),
which described overlapping and partly-stale versions of the same pipeline.

```
ACHILLES container ─▶ oracle npz  (reference side, §1 only)  ─┐
                                                               ├─▶ analysis/paper/*  ─▶ output/paper/*.png
configs/banks/*.yaml ─▶ ADoNIS banks ─▶ Gate-I Jacobian ──────┘
```

Two rules that are not optional on S3DF:

- **Never run on the login node.** `jax` import is NFS-throttled there and hangs for minutes; on a
  compute node it is ~1.4 s. Everything below assumes `srun`/`sbatch`.
- **Use the shared venv, not the repo `.venv`** — the latter is a stale macOS aarch64 tree.

```bash
export P=/sdf/data/neutrino/cjesus/software/venvs/adonis/bin/python        # CPU
export PCU=/sdf/data/neutrino/cjesus/software/venvs/adonis-cuda/bin/python # CUDA
export PYTHONPATH=$PWD                       # from the repo root
export ADONIS_OUT=/sdf/data/neutrino/cjesus/ADoNIS/output
```

`data/` and `output/` in the repo are symlinks into `/sdf/data/neutrino/cjesus/ADoNIS`. If you hit
`FileNotFoundError: data/achilles/dcc_EW.dat`, the `repo_data/achilles` symlink is missing — relink it to
`software/achilles_data`.

---

## 1. ACHILLES reference samples (§1 only)

Only section 1 compares against ACHILLES. Sections 2–4 use ADoNIS banks alone.

The container has no Docker on S3DF; use Apptainer, and note the three flags are all required:

```bash
SIF=/sdf/data/neutrino/cjesus/software/images/achilles-oracle.sif
apptainer run --fakeroot --writable-tmpfs --pwd /achilles --bind <out>:/out $SIF /out/run.yml
```

Driven from `configs/achilles/*.yml` through one entry point:

```bash
$P -m adonis.oracle.run_achilles configs/achilles/run_T2K_C_fsi_gauss.yml   --seeds 8 --extract cc1pi_rich
$P -m adonis.oracle.run_achilles configs/achilles/run_MINERvA_C_fsi.yml     --seeds 8 --extract cc0pi
$P -m adonis.oracle.run_achilles configs/achilles/run_cascade_pip_C_broad.yml  --seeds 48
$P -m adonis.oracle.run_achilles configs/achilles/run_cascade_prot_C.yml       --seeds 8
$P -m adonis.oracle.run_achilles configs/achilles/run_cascade_neut_C.yml       --seeds 8
```

ACHILLES ignores `Main: Seed` — the seed must be set under `Options/Initialize/Seed`, or every shard is
byte-identical and the errors come out √N too small.

---

## 2. ADoNIS banks

**One generator**, dispatching on `cfg.probe`, driven entirely by `configs/banks/*.yaml`:

```bash
$PCU -u -m adonis.workflow.cli configs/banks/nu_T2K_C.yaml    --out $ADONIS_OUT/nu_T2K_C
$PCU -u -m adonis.workflow.cli configs/banks/nu_MINERvA_C.yaml --out $ADONIS_OUT/nu_MINERvA_C
$PCU -u -m adonis.workflow.cli configs/banks/nu_MINERvA_H.yaml --out $ADONIS_OUT/nu_MINERvA_H   # free-H (CH)
$PCU -u -m adonis.workflow.cli configs/banks/beam_pip_C.yaml  --out $ADONIS_OUT/beam_pip_C
$PCU -u -m adonis.workflow.cli configs/banks/beam_e_C.yaml    --out $ADONIS_OUT/beam_e_C
```

Bank generation runs on the **turing GPU** by default (the jitted cascade is ~25×). Shard with
`--n-seeds/--seed0` across a SLURM array — one seed is one chunk — then merge and **rechunk**:

```bash
$P -m adonis.workflow.rechunk $ADONIS_OUT/<bank>/merged $ADONIS_OUT/<bank>/merged_rc --target-gb 0.5
```

Rechunking is exact and is not optional in practice: uniform ~0.5 GB chunks remove per-chunk streaming
overhead that otherwise dominates every downstream fit.

Sharding note: prefer many 1-core array tasks over one fat process (~13–18× per core).

---

## 3. The Gate-I Jacobian (input to §2, §3, §4)

One command builds the object all three sections consume, `output/altgen/multisample_carbon.npz`:

```bash
$PCU -u -m analysis.paper.gate1_multisample
```

It reads the sample definitions from `configs/samples/*.yaml` through `AnaSample`/`SampleSet` and appends
the FSI-only beam blocks. This is the *only* supported builder — a second one existed and was removed
(commit `13e8513`) after producing a silently broken 350-bin stack.

---

## 4. Figures

### §1 — validation against ACHILLES

Config-driven, one YAML per figure in `analysis/paper/sec1_validation/`:

```bash
$P -m analysis.paper.sec1_validation.make            # all figures
$P -m analysis.paper.sec1_validation.make fig07      # one
```

Needs `--mem=48G` for the heavy ones; `fig09` over the 20M bank OOMs silently without it (exit 0, stale
PNG).

### §2 — Fisher / Gate I, and §3 — gradients

Both read the Jacobian npz from step 3 and re-style instantly without refitting:

```bash
$P -m analysis.paper.sec2_fisher.make        # column set declared in configs/paper/sec2_subsets.yaml
$P -m analysis.paper.sec3_gradients.make
```

### §4 — closure, uncertainty, gradient flow, rates

The run parameters are recorded in `configs/fits/sec4_P1.yaml`. **That file is documentation, not yet an
input** — the stages are still driven by `S4_*` environment variables (P2 of
`docs/consolidation_plan.md`). Until then, source the environment from the config by hand.

Stages, in dependency order (all on GPU; `PHYSFIT_INJECT` = the `data.inject` block):

```bash
export S4_SIG_CAP=250000 S4_NU_CHUNKS=120 S4_E_CHUNKS=40 S4_BEAM_CHUNKS=10
export S4_SIGMA_SYST_ONLY=1 S4_MIN_MCFRAC=0.05 S4_PRIOR_SCALE=0.0
export PHYSFIT_INJECT="M_A_qe=1.12,M_A_res=0.85,...,src_tail=0.92"   # see configs/fits/sec4_P1.yaml
export ADONIS_LABEL=sec4_P1

$PCU -u -m analysis.paper.physfit.multisample                 # closure fit  -> sec4_P1.npz
ALTGEN_NIT=60  $PCU -u -m analysis.paper.physfit.multisample_profile        # 1-D profile (shard: S4_PROF_BASE)
$P             -m analysis.paper.physfit.multisample_profile_merge sec4_P1

MODE=prof S4_CORNER_N=41 S4_CORNER_FROM_PROFILE=1 \
  S4_CORNER_DIALS="M_A_res,delta_strength,res_axial_strength,Eb_shift" \
  $PCU -u -m analysis.paper.physfit.multisample_corner2d      # figure B  (shard: S4_PAIR_BASE/S4_ROW_BASE)

MODE=grad S4_CORNER_N=25 S4_CORNER_FROM_PROFILE=0 \
  $PCU -u -m analysis.paper.physfit.multisample_corner2d      # figure C

S4_FIXED_TRUTH="$PHYSFIT_INJECT" S4_FITTER=trf ALTGEN_NIT=200 \
  S4_TOY_BASE=0 S4_NTOYS=50 $PCU -u -m analysis.paper.physfit.multisample_coverage   # 2000 toys, 40 shards

NUTS_CHAIN=0 NUTS_SAMPLES=1500 NUTS_WARMUP=150 $PCU -u -m adonis.fit.nuts            # 16 chains
```

Then the figures, which only read persisted npz and never refit:

```bash
$P -m analysis.paper.sec4_closure.fig_closure_summary sec4_P1 sec4_P1_ens   # A
$P -m analysis.paper.sec4_closure.fig_corner_all      sec4_P1 sec4_P1       # B
FIELD=gn $P -m analysis.paper.sec4_closure.fig_corner_grad sec4_P1          # C
$P -m analysis.paper.sec4_closure.fig_rates           sec4_P1               # D
```

| figure | script | output |
|---|---|---|
| A | `fig_closure_summary` | `sec4_P1_figA.png` |
| B | `fig_corner_all` | `sec4_P1_figB.png` |
| C | `fig_corner_grad` | `sec4_corner_grad_gn.png` |
| D | `fig_rates` | `sec4_P1_figD.png` |

`fig_corner_prof.py` is **not** a figure; it owns `load_views`/`snap_axis`/`view_for`, which B imports.

Supporting analyses (not figures) are in `analysis/paper/sec4_closure/diagnostics/` — coverage, the
mirror scan, the toy restarts, and the rest. `adonis/fit/nuts.py --selftest` samples a 16-D Gaussian with
a −0.995 correlated pair and checks the recovered mean and covariance; run it after touching the sampler.

---

## SLURM accounts

| what | account / QOS |
|---|---|
| turing GPU, non-preemptable | `-A neutrino:cider-nu -q normal`, or `mli:nu-ml-dev`, `mli:cider-ml` |
| ampere GPU | same three accounts, `-q normal` |
| CPU (plots, merges, parsing) | `-A neutrino:default -q preemptable -p milano` |

Never use `neutrino:default`/`mli:default` at `normal` on turing — those are preemptable and get killed
(exit 143). All three turing accounts draw on one node pool, so a `normal`-QOS job in a *different*
account still contends; use a different **partition** (ampere) when you need to not compete with a
running campaign.

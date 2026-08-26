# Reproducing the ADoNIS paper

Every figure in the paper is built by one command in this directory. Each reads persisted `.npz`
files and does no physics of its own, so restyling a figure costs seconds and a figure can never
disagree with the run it shows.

The work is in producing those `.npz` files. There are four stages, in order:

1. **Event banks** — generate ADoNIS events once; everything downstream reweights them.
2. **ACHILLES reference** — the same physics from the generator ADoNIS is validated against.
3. **Analysis runs** — the Jacobian, the fits, the unfolding.
4. **Figures** — this directory.

Stages 1 and 2 are the expensive ones (hours on a cluster at paper statistics). Stages 3 and 4 take
minutes to hours from their output.

---

## 0. Setup

```bash
pip install -e .            # the adonis package
pip install matplotlib      # figures
export JAX_ENABLE_X64=1     # required: the FSI nominal reweight is exact only in double precision
```

ADoNIS reads ACHILLES' tabulated inputs (amplitudes, spectral functions, nucleon configurations,
flux). Point it at them:

```bash
export ACHILLES_DATA=/path/to/achilles/data
```

To also run ACHILLES itself you need its container images:

```bash
export ACHILLES_IMAGES=/path/to/images   # achilles-oracle.sif, achilles-fullcascade.sif, achilles-cascade.sif
export ADONIS_CONTAINER=apptainer        # or docker; omit to auto-detect
```

Figures are written to `output/paper/`. Set `ADONIS_PAPER_OUT` to send them elsewhere — useful for
comparing a rebuild against the published set without overwriting it.

### Check the installation

```bash
python -m analysis.campaign.gate1 --label smoke --fit-config configs/fits/smoke.yaml --max-chunks 4
python -m analysis.campaign.run configs/fits/smoke.yaml --stage closure
```

`configs/fits/smoke.yaml` is the full inference chain at the smallest size that still exercises it.
Every stage below accepts it in place of the paper config.

---

## 1. Event banks

One config per sample in `configs/banks/`. The config carries the physics (probe, beam, target,
channels, FSI); the command line carries only scale.

```bash
python -m adonis.workflow.cli configs/banks/nu_T2K_C.yaml --out output/banks/nu_T2K_C/part_0 \
    --n-per-seed 250000 --n-seeds 4 --seed0 0
```

`--n-per-seed` is **per channel**, so a two-channel config produces more events than that number.
One seed produces one chunk. Shards are independent — run them as a job array with a different
`--seed0` and `--out` per task, then merge:

```bash
python -m adonis.workflow.merge_bank --parts output/banks/nu_T2K_C --out output/banks/nu_T2K_C/merged
python -m adonis.workflow.rechunk output/banks/nu_T2K_C/merged output/banks/nu_T2K_C/merged --target-gb 0.5
```

The rechunk is optional and exact; it makes chunks uniform so streaming is not dominated by
per-chunk overhead.

The paper uses 10M events per neutrino sample. The banks the analysis configs expect are:

| Bank config | Used for |
|---|---|
| `nu_T2K_C`, `nu_T2K_H` | T2K CC0π and CC1π⁺ (H for the free-hydrogen part of hydrocarbon) |
| `nu_MINERvA_C`, `nu_MINERvA_H` | MINERvA CC0π, inclusive, CC1π⁺ |
| `nu_uBooNE_Ar`, `nc_uBooNE_Ar` | MicroBooNE CC1p0π and NC 1π⁰Xp |
| `beam_e_C`, `beam_e_Ar`, `beam_e_C_1159` | inclusive (e,e′) and e4ν |
| `beam_pip_{C,Ar}`, `beam_prot_{C,Ar}`, `beam_neut_{C,Ar}` | tagged hadron beams (cascade alone) |

A GPU is not required. The hard vertex runs at the same speed on a CPU core; the cascade is
10–13× faster on a GPU and is the bottleneck.

## 2. ACHILLES reference

One run card per sample in `configs/achilles/`. The runner picks the right image for the card and
runs it under docker or apptainer.

```bash
python -m analysis.oracle_tools.run_achilles configs/achilles/run_T2K_C_fsi_gauss.yml \
    --nevents 250000 --seeds 40 --out-dir output/achilles/nu_T2K_C --extract fs_rich
```

Each seed is a genuinely independent sample — the seed is written where each ACHILLES binary
actually reads it. ACHILLES can fault mid-run; partial output is kept.

`--extract fs_rich` turns each `.hepmc` into an observable npz. Combine the shards into the single
reference file the analysis configs name:

```bash
python -m analysis.oracle_tools.combine "output/achilles/nu_T2K_C/*.npz" \
    output/achilles/fsrich/nu_T2K_C.npz
```

## 3. Analysis runs

**The Jacobian** (Sec. 3, and the parameter selection used by Sec. 4):

```bash
python -m analysis.campaign.gate1 --label multisample_carbon
```

Writes `output/altgen/multisample_carbon.npz`: the stacked per-bin Jacobian over all samples and
beams, plus the Fisher matrix and the resulting σ_post/σ_prior per parameter.

**The fit** (Sec. 4). One stage per command; all read `configs/fits/sec4_P2.yaml`:

```bash
for stage in closure profile profile2d gradient2d toys nuts; do
    python -m analysis.campaign.run configs/fits/sec4_P2.yaml --stage $stage
done
```

The expensive stages shard. Pass `--shard k/N` to split the work across jobs — 40 shards for the
2000 toys, 20 chains for NUTS, one per parameter for the profile:

```bash
python -m analysis.campaign.run configs/fits/sec4_P2.yaml --stage toys --shard 7/40
```

**The unfolding** (Sec. 5):

```bash
python -m analysis.campaign.unfold_run configs/fits/sec5_unfold.yaml --label sec5f10
```

Runs every study in the config and writes `output/altgen/sec5f10_unfold.npz`.

## 4. Figures

```bash
python -m analysis.paper.validation.make --no-ratio      # ADoNIS vs ACHILLES, 8 figures
python -m analysis.paper.grad_info.make                  # gradient information, 2
python -m analysis.paper.inference.make --label sec4_P2  # fit and uncertainties, 3
python -m analysis.paper.unfolding.make --label sec5f10  # unfolding, 3
```

Each renders one figure per subprocess. Name a figure to build only that one
(`... validation.make fig09`), and pass `--label` to keep a rebuilt set reading the same run.

`--no-ratio` drops the ADoNIS/ACHILLES ratio strip and writes `*_noratio`; those are the variants
the paper uses.

| Figure in the paper | File | Command |
|---|---|---|
| MINERvA CC0π | `fig09_minerva_cc0pi_noratio` | `validation.make --no-ratio fig09` |
| MicroBooNE NC 1π⁰Xp | `fig11_uboone_nc1pi0_doublediff_noratio` | `validation.make --no-ratio fig11` |
| Inclusive (e,e′) | `fig01_ee_domega_noratio` | `validation.make --no-ratio fig01` |
| π⁺–nucleus cross sections | `fig03_pi_nucleus_sigma_noratio` | `validation.make --no-ratio fig03` |
| Per-bin parameter response | `multisample_grad_per_bin` | `grad_info.make gradients` |
| Constraints per probe | `constraints_per_subset` | `grad_info.make fisher` |
| Observables before and after the fit | `Asimov_xsec_samples` | `inference.make rates` |
| Parameter recovery and coverage | `closure_demo` | `inference.make closure` |
| Corner plot | `corner_plots` | `inference.make corner` |
| Unfolded cross section | `unfolding_example` | `unfolding.make unfolded` |
| Uncertainty budget | `unfolding_uncertainty_budget` | `unfolding.make budget` |
| Post-fit correlations | `unfolding_param_correlations` | `unfolding.make corr` |
| T2K CC0π (appendix) | `fig07_t2k_cc0pi_noratio` | `validation.make --no-ratio fig07` |
| T2K CC1π⁺ (appendix) | `fig08_t2k_cc1pi_noratio` | `validation.make --no-ratio fig08` |
| MicroBooNE CC1p0π (appendix) | `fig10_uboone_cc1p0pi_noratio` | `validation.make --no-ratio fig10` |
| e4ν (appendix) | `fig0456_e4nu_noratio` | `validation.make --no-ratio fig0456` |

The optimiser comparison quoted in Sec. 4.1 comes from `analysis.benchmarks.bench_fair`; render it
with `python -m analysis.paper.performance.make minimizers`.

---

## Where things are

```
configs/banks/      one per event bank
configs/achilles/   one per ACHILLES run card
configs/samples/    signal definition, observables and binning per measurement
configs/fits/       the fit and unfolding runs
analysis/campaign/  the runs: Jacobian, fit stages, unfolding
analysis/paper/     these figures
adonis/             the package: physics, reweighting, cascade, fitting, unfolding
```

A sample config is the single definition of a measurement. The same file drives the validation
figure, the Jacobian and the fit, so those three can never disagree about what the sample is.

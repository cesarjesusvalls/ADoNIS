# AnaSample migration — remaining plan

**As of 2026-08-06, `main` @ `56affe7` (branch `joint-amps2` == `main`).**

The sample-centric refactor is the idea that a **sample** (bank + signal + observables + real edges) is
the fundamental object, and the paper's sections are *verbs* over it: **plot** (sec1), **gradient/Gate-I**
(sec2/3), **fit** (sec4/5). The core object is `adonis/analysis/sample.py::AnaSample` (+ `SampleSet`),
built from an `adonis.workflow.config.AnalysisConfig`; samples live in `configs/samples/*.yaml`.

## Done (merged to main)

- **Core:** `adonis/analysis/{sample.py,knobs.py}` — `AnaSample` (`.selected`/`.datasets`/`.jacobian`/
  `.gate1`/`.plot`/`.dump_cache`), `SampleSet` (composition, additive Fisher), the 28-knob `SPEC/PNAMES/
  PRIOR` + `theta_nominal`/`knobs_of`.
- **5 samples** in `configs/samples/`: `t2k_cc0pi`, `t2k_cc1pi`(+`_ch`), `minerva_stv`, `minerva_ptpz`,
  `ee_omega` — each validated (gradient) on a compute node.
- **CH free-H** via `signal.target: CH` (streamed from `nu_T2K_H`); empty-bank FSI dtype fix in `bank_plot`.
- **Multisample:** `analysis/paper/gate1_multisample.py` (SampleSet + cached beams) → `multisample_carbon.npz`
  (303 bins, 19 obs, 16/28 FIT). sec2/sec3 rewired for namespaced `sample:obs` dskeys.
- **sec1 `.plot()`** reproduces ALL 8 sec1 figures; sec2/sec3 rendered from the new npz.

## Remaining — P2 cleanup (no decision needed; do any time)

1. **Delete the 4 satellite drivers** (`analysis/paper/{electron_fit,minerva_fit,minerva_ptpz_fit,
   t2k_pcos_fit}.py`) + `analysis/paper/sec3_gradients/build_multisample.py` — replaced by AnaSample /
   `gate1_multisample`. First `grep -rn` to confirm no other importers (note `minerva_ptpz_fit` imports
   `bin2d_dataset` from `t2k_pcos_fit`; both go together).
2. **Slim `analysis/paper/physical_fit.py`** to `from adonis.analysis.knobs import SPEC, NPAR, PNAMES,
   PRIOR, theta_nominal, knobs_of` (single source), KEEPING `build_physfit_datasets` / `OBS_SUBSETS` /
   `design_edges` / `DOMAINS` / `MULT_EDGES` — sec4/5 still import those. Do NOT delete physical_fit; it is
   a HUB.
3. **Repoint** `analysis/paper/beams/beam_fisher.py` `from analysis.paper import physical_fit as PF` →
   `from adonis.analysis import knobs as PF` (uses only `PF.{SPEC,NPAR,knobs_of,theta_nominal}`).
4. **Full sec1 migration:** move `fig01/fig03/fig0456/fig10/fig11` configs into `configs/samples/`, make
   `sec1_validation/make.py` a thin `AnaSample.plot()` loop over `configs/samples/`, delete the duplicate
   `sec1_validation/*.yaml`. (The 3 nu STV figs already live in `configs/samples/`.)
5. **Thin CLIs + SLURM launcher** + update `analysis/paper/REGENERATE.md` to the one-command flow (cache beams on CPU
   → `gate1_multisample` on turing GPU → render sec2/sec3). Delete the old `sec3_gradients/regenerate.sh`
   idea (never committed).

**Importer web to respect (why physical_fit can't just be deleted):** `analysis/paper/physfit/*`,
`sec4_closure/*`, `sec5_methods/*` import `SPEC/PNAMES/PRIOR/NPAR/theta_nominal/knobs_of/
build_physfit_datasets/OBS_SUBSETS` from `physical_fit`. Slimming (step 2) keeps them working.

## sec4 (4.1–4.4) migration — IN PROGRESS

**Done + committed (main):**
- The closure fit ENGINE is sourced from `configs/samples/` (`AnaSample.bin_datasets` builds the engine
  `ds`); `build_multisample_engine` builds 5 bank sub-engines from the configs + 3 `BeamSample`s; `lm_fit`
  / `MultiEngine` / `set_closure_data` + all 5 sec4 drivers unchanged. dskeys now match the new
  `multisample_carbon.npz`, so the 16-dial subset comes from it. `fig_dists` (4.2) repointed to namespaced
  keys; the other figs are per-dial (unaffected).
- **N_selected, not N_total** (user request): `bank_plot.filter_events` (gather a selected-event subset,
  remapping every ragged family — validated EXACT) + `selection.select_bank` (stream the full bank, filter
  each chunk, concat — validated EXACT: reweight/normalization/observables 0.0 rel). `BankSample(signal=)`
  caches only the compact signal, so the fit streams the full bank but resides `N_selected`. Closure smoke
  recovers all 16 dials (~0.5σ).

**Remaining for 4.1–4.4:** regenerate the 5 driver npzs at a consistent label + render:
- 4.1 recovery: `multisample_profile` -> `<L>_profile.npz` -> `fig_recovery.py`. 4.1b Eb-wall:
  `multisample_ebwall` -> `sec4_ebwall.npz` -> `fig_ebwall.py`.
- 4.2 dists: `multisample` (main closure) -> `<L>.npz` -> `fig_dists.py` (keys already fixed).
- 4.3 coverage: `multisample_coverage` (N toys, EXPENSIVE) -> `sec4_coverage_*.npz` -> `coverage_fig.py`.
- 4.4 corner: `multisample_corner` (120 pairs x 19^2, EXPENSIVE) -> `<L>_corner.npz` -> `corner_fig.py`.
  `profile`/`corner` read `<L>.npz` first, so run the main closure at `<L>` before them.
- Decisions kept from old sec4: carbon T2K CC1pi (t2k_cc1pi_ch keys, no free-H offset), scale=1/bw.

## Remaining — sec5 `.fit()` verb  ⛔ BLOCKED ON A DECISION

The sec4/5 fit machinery (`analysis/paper/physfit/*` = closures/coverage/methods, `sec4_closure/*`,
`sec5_methods/*`) is UNTOUCHED and still runs on the old `physical_fit` (`build_physfit_datasets` +
`OBS_SUBSETS` + the fit drivers + GENIE/NEUT fake data).

**Decision the user needs to make first:** *which* sec4/5 figures survive the paper reorg (closure5,
coverage, non-Gaussian, q2mod/mecmix method figs, timing/scaling, …). See the paper-status memories
`sec4-nongaussian-coverage` and `paper-sec5-diff-methods`.

**Once decided, the migration is:**
- Add `AnaSample.fit(data, ...)` (and `SampleSet.fit`) — a new verb reusing the existing fit engine
  (`analysis/paper/physfit/physical_fit_run.py`'s Newton/GN loop; see `physfit-lmfit-convergence`).
  Inputs come from `.datasets()` (central/sigma/Cinv) + the jvp Jacobian, exactly as `.gate1()` does.
- Point the KEPT sec4/5 figure drivers at `AnaSample`/`SampleSet` instead of `build_physfit_datasets`;
  retire the dropped ones + their `physical_fit` deps; then finish deleting/​slimming per P2.
- Fake data (GENIE/NEUT) stays external (`analysis/paper/physfit/run_genie.sh`).

## How to resume the full pipeline (reference)

```
# 1. beams (CPU, cached to data/cache/plot/beam_jac_*_n15_s0.05.npz)
srun -A neutrino:default -q preemptable -p milano --mem=64G -c 8 \
  bash -c "JAX_PLATFORMS=cpu PYTHONPATH=$PWD python -c 'from analysis.paper.beams import beam_fisher as B; [B.beam_jacobian(b) for b in (\"pip\",\"prot\",\"neut\")]'"
# 2. full multisample (turing GPU, non-preempt; samples stream on-device, beams from cache)
srun -A mli:nu-ml-dev -q normal -p turing --gres=gpu:1 --mem=64G -c 8 \
  bash -c "PYTHONPATH=$PWD /sdf/data/neutrino/cjesus/software/venvs/adonis-cuda/bin/python -m analysis.paper.gate1_multisample"
# 3. render (CPU)
python -m analysis.paper.sec2_fisher.make ; python -m analysis.paper.sec3_gradients.make
# sec1 (CPU): AnaSample.from_config("configs/samples/<x>.yaml").plot()  (or sec1_validation.make for the rest)
```

**Cosmetic TODO:** sec3 `multisample_carbon_shape` — the coarse T2K CC1π columns (3–5 bins) crowd their
2-line x-labels; needs a labels pass.

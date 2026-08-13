# Consolidation plan — one API + config files for generation, samples and fits

Baseline: `1cc7518` (sec4 figure set A–D finalised).

The audit found three layers in very different states. Generation is already right; sample definitions
are half-right; the fit layer has no configuration at all. The plotting layer is *already* what we want
(every `make.py` / `fig_*.py` reads a persisted npz and never re-fits) — it only needs relocating.

| layer | state | work |
|---|---|---|
| bank generation | one CLI + `configs/banks/*.yaml`, shared by all sections | none |
| analysis samples | one API (`AnaSample`) but three parallel regimes, sample lists hardcoded twice | P1 |
| fit machinery | 53 `S4_*` env vars, 11 entry points, no run manifest | P2–P3 |
| plotting | reads persisted npz, no re-fitting | P4 (relocate only) |

## Why this is worth doing

Every one of these cost real time in the sec4 campaign, and all trace to the missing run-definition layer:

- `multisample_corner2d` never read `S4_PRIOR_SCALE`, so its inner fits were MAP-regularised while the
  closure, 1-D profile and coverage ensemble were MLE. The 2-D contours were systematically tighter than
  the 1-D profile they must contain.
- The same script ignored `S4_PROF_FITTER` and hard-wired `lm_fit`, so the 1-D profile was not the
  minimum of the 2-D surface over the other axis.
- Run parameters lived in *filenames* (`n41_00_r014`). Two resolutions collided on disk, and a mid-flight
  rerun silently changed an already-published figure.
- Row shards each subtracted their own `nanmin`, so merged surfaces sat at different offsets.
- Adding the MINERvA CC1π⁺ samples required editing two hardcoded sample lists that can disagree.

## P0 — stop the bleeding (today, ~1 h)

1. **Vendor `nuts_own.py` into the repo.** It produced the 24,000 samples behind figure B and lives only
   in a session scratchpad, which is not backed up. Move to `adonis/fit/nuts.py` *verbatim first*
   (commit), then de-hardcode the truth path and `sys.path` insert in a second commit so the move is
   reviewable separately from the edit.
2. **Vendor the analysis one-offs that produced published numbers**: `toys_vs_profile.py`,
   `toy_restart.py`, `hist2d_raw.py`, `mirror_scan.py`, `central.py`, `slice_sdelta.py`, `avg_data.py`
   → `analysis/paper/sec4_closure/diagnostics/`. These are the evidence for the coverage argument.
3. Record the P1 run parameters (currently only in `$SC/*.sh`) as the first fit config, even before the
   loader exists — `configs/fits/sec4_P1.yaml`, written by hand, checked against the job scripts.

## P1 — one sample layer (~1 day)

4. **Promote beam samples to configs.** `BEAM_OBS` is a hardcoded dict duplicated in
   `gate1_multisample.py` and `sec3_gradients/build_multisample.py`. Add `configs/samples/beam_*.yaml`
   with the same schema (`nbins`, `syst`, observable keys) and have `beam_fisher.beam_jacobian` read them.
5. **Fold the sec1 figure configs into `configs/samples/`.** They already use the same
   `render: multiobs` schema — signal block, observables, edges — but sit in the section folder and are
   parsed by `sec1_validation/make.py` instead of `AnaSample`. One family, one reader. Keep the
   figure-only keys (`reference:`, `ratio_band:`, `out_path:`) as an optional `figure:` sub-block.
6. **Delete both hardcoded sample lists.** `SAMPLES` in `gate1_multisample.py` and the `_bank(...)` calls
   in `physfit/multisample.py` become one list in the fit config.
7. Guardrail test: build the sec4 engine from the config and assert `dskeys` and `row0` match the
   committed `multisample_carbon.npz` exactly. This is the test that would have caught the sample-list
   divergence.

## P2 — fit configuration (~2 days)

8. **`configs/fits/*.yaml`** as the single run definition:

```yaml
name: sec4_P1
samples: [t2k_cc0pi, t2k_cc1pi_ch, minerva_stv, minerva_ptpz, ee_omega,
          minerva_cc1pip_tpi, minerva_cc1pip_q2]
beams:   [pip, prot, neut]
data:
  mode: closure                      # closure | real | toy
  inject: {M_A_qe: 1.12, M_A_res: 0.85, Eb_shift: 0.5, ...}
  sigma:  {syst: 0.05, mc_term: false, mask_mcfrac: 0.05, freeze_at: nominal}
fit:
  dials: gate1                       # gate1 | explicit list
  estimator: mle                     # mle | map(prior_scale)
  minimizer: {method: trf, max_nfev: 200, gtol: 1e-8, xtol: 1e-14, ftol: 1e-14}
uncertainty:
  - {method: gaussian}
  - {method: profile,   n: 13, reach: adaptive}
  - {method: profile2d, dials: [M_A_res, delta_strength, res_axial_strength, Eb_shift], n: 41}
  - {method: nuts,      chains: 16, samples: 1500, warmup: 150}
  - {method: toys,      n: 2000, fixed_truth: true, seed0: 0}
```

9. **`python -m adonis.fit <config> --stage <name> --shard k/N`** as the single entry point, replacing the
   11 `multisample_*.py` mains. Each stage writes `output/altgen/<name>/<stage>/shard_k.npz` **plus a
   manifest** carrying the resolved config, git SHA, and shard identity — so shards cannot silently
   disagree and a rerun cannot half-overwrite a completed set.
10. **Uncertainty methods behind one interface** (`estimate(engine, theta_hat, cfg) -> npz`), so
    profile / profile2d / NUTS / FC / toys are selected by name rather than by which script you run.
11. Retire the `S4_*` surface. Keep a thin env override for genuinely per-job things only
    (`--shard`, chunk caps), and fail loudly on any unrecognised `S4_*` still set in the environment.

## P3 — sharding and merging (~half day) — **done**

12. Move sharding into the runner: `--shard k/N` decides pair/row/seed ranges from the config, rather
    than `S4_PAIR_BASE` × `S4_ROW_BASE` × filename conventions. ✔
13. One merge function per stage that validates the manifest set is complete and self-consistent before
    writing the merged product — no more "98% complete so it silently fell back to the coarse grid". ✔
    `adonis/fit/merge.py`, used by the corner and profile merges.
14. Store raw quantities, never per-shard-normalised ones (the `chi2_abs` lesson). ✔ (predates P3)

**Correction to item 13: not the manifests.** The runner's `.manifest.json` cannot be what a merge
validates against, because a manifest and its shard are associated only by filename convention — copy,
rename or hand-merge a shard and its claims are silently gone, which is precisely the situation the check
exists to detect. The provenance is therefore stamped INSIDE the npz (`adonis/fit/provenance.py`,
`prov_*` keys); manifests remain a run log. Shards also stamp *which slice of the work they were
assigned* (row block, dial block), which turns "is this complete?" from a NaN-fraction proxy into an
exact statement that can name the missing rows.

Grandfathered on purpose: every sec4_P1 product predates stamping, and re-running the campaign is out of
scope, so an unstamped shard warns and falls back to the NaN proxy. A shard that *carries* provenance and
disagrees with its siblings is a hard error.

## P4 — plotting relocation (~half day)

15. Move the sec4 figure scripts under `analysis/paper/sec4_closure/` (already true) and give each
    section a `make.py` that renders its whole set from configs + persisted npz. Figures A–D are already
    pure consumers; this is mostly deleting the remaining env reads (`FIELD`, `NUTS_SMOOTH`, `MASS`) in
    favour of a `figures:` block in the fit config.
16. `analysis/paper/README.md`: one table mapping figure → config → stage → npz.

## Sequencing

P0 is independent and should land first. P1 unblocks P2 (the fit config references sample names, so the
sample layer must be canonical first). P3 depends on P2's manifest. P4 is last and is the smallest.

Suggested first PR: **P0 + item 7** — vendoring plus the guardrail test — because it protects what exists
and adds the check that catches the class of bug that dominated this campaign, without changing any
interface.

## Explicitly out of scope

- Re-running any sec4 result. The consolidation must reproduce `sec4_P1`; that is the acceptance test,
  not an opportunity to change physics.

  **Correction to the standard.** "Bit-for-bit" is not achievable and was the wrong bar: the model
  evaluation is not deterministic on GPU. Measured, the engine fails to reproduce *itself* — the same
  code, twice, in one process, gives max|Δ| = 5.7e-14 on `data`, which is exactly the discrepancy against
  the committed npz. The achievable and verified standard is **equality to float64 round-off** (max
  relative error 3.6e-16, median 0), with the integer/structural quantities — `subset`, `dskeys`, `row0`,
  `sigma`, `truth` — exact.
- The `jobs/` SLURM submitters, which live outside the repo. Worth a separate decision: either vendor
  them or generate them from the fit config.

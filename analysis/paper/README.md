# `analysis/paper` — figures for the ADoNIS methodology paper

One sub-package per section of `docs/paper_plan.md`. Each is a **thin driver** over an already-validated
engine: no physics, no binning, no reweighting is re-implemented here. If a figure needs a number, it
comes from the engine that owns it.

Figures are written to `output/paper/` (gitignored) as `.png` + `.pdf`. Shared style: `style.py`.

| section | package | engine it drives |
|---|---|---|
| 1 — ADoNIS reproduces ACHILLES | `sec1_validation` | `analysis/t2k/make_plots.py`, `differentiability/bank_matrix.py` |
| 2 — Fisher / Gate I: constrainable knobs | `sec2_fisher` | `analysis/paper/physical_fit.py` (Gate I) |
| 3 — gradients of the fittable knobs | `sec3_gradients` | `full_knobs.knob_specs`, `bank_reweight.weight_jit` |
| 4 — fitting & statistical interpretability | `sec4_closure` | `physfit/multisample.py` (nonlinear multisample closure) |

Sections 5 (computational performance / novel methods) and 6 (unknown unknowns) were removed and are
not in the first version of the paper; see `git log -- analysis/paper/sec5_methods analysis/paper/sec6_unknowns`.

## Section 2 — Fisher / Gate I per data subset

The Gate-I Jacobian is computed **once** over the full observable set and persisted; every subset is a
**row slice** of it, so the subset study costs no bank pass.

```bash
# one bank pass: J (27 knobs x 188 bins), 11 datasets  -> output/altgen/physfit_gate1_full_v2.npz
ADONIS_EVENT_BANK=output/event_bank_v2 PHYSFIT_OBS=full ADONIS_LABEL=physfit_gate1_full_v2 \
  python -u scripts/altgen/physical_fit.py

# all subset figures from that one npz (seconds, no bank)
python -m analysis.paper.sec2_fisher.make
```

`OBS_SUBSETS` in `scripts/altgen/physical_fit.py` is the single source of truth for the subsets — it is
consumed by the Gate-I run, by any subset-restricted **fit** (`physical_fit_run.py` reads the Gate-I npz
to pick which knobs to release), and by the figures here.

### Observables

Nine kinematic (20 bins each) + two multiplicity (`kin9` + `mult` = `full`):

- **CC0π**: δp_T, δα_T, p_μ, cosθ_μ — 10⁻³⁸ cm²/nucleon
- **CC1π**: p_N, δp_TT, δα_T, p_π, cosθ_π — nb/CH (incl. the frozen free-H offset)
- **CC-inclusive**: `N_p` (protons above 300 MeV/c; 0,1,2,3,≥4) and `N_π±` (π⁺ and π⁻ together, no
  threshold; 0,1,≥2)

The multiplicities sit in a **CC-inclusive** sample under the **muon-only** T2K acceptance (p_μ > 250 MeV,
cosθ_μ > −0.6). They cannot live inside the exclusive selections: `bank_plot.acceptance()` is the CC0π
signal cut and carries a leading-proton window that makes `N_p = 0` unreachable, and `N_π±` is 0 by
construction in CC0π and 1 in CC1π. Proton threshold is `ADONIS_PMULT_THR` (default 300 MeV/c).

### Reading the output

A knob is **FIT** when shrinkage `σ_post/σ_prior < 0.5` — the data, not the prior, determines it. Because
`σ_post` is *marginalized*, failing has two physically opposite causes, and the figures separate them:

- **DEGENERATE** — low raw shrinkage (the data sees it clearly with other knobs fixed) but high
  marginalized shrinkage (another knob mimics it). A better observable can recover it.
  *e.g.* `qe_norm`: raw 0.034, marginalized 0.86 — a 25× penalty, from `axial_strength`/`vector_strength`/`sf_norm`
  all being able to spell "scale the QE rate".
- **INVISIBLE** — raw shrinkage > 1: the sample carries no information at this precision, and no
  re-parameterization helps. *e.g.* `s_conv` (raw 13.8), `pion_pole`, `gen`.

Both are reported: `sec2_shrinkage_subsets` (marginalized vs raw, per subset) and `sec2_failure_modes`
(the two-axis plane). `sec2_degeneracy` shows the prior-scaled Fisher eigen-spectrum.

Gate I is **precision-relative** by construction: `ADONIS_SYST` (default 0.05) sets the systematic that
defines "resolvable". At 15% only 4 knobs pass instead of 7 — measurability is a property of the knob
*and* the dataset's precision, which is the paper's thesis in miniature.

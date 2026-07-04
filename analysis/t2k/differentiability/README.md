# `analysis/t2k/differentiability` — the differentiable T2K analysis chain

Everything here runs on top of the frozen, validated ADoNIS pipeline (`adonis/`), exposing the physics as
**exact, JAX-differentiable per-event reweights**. Two generations of machinery coexist:

- **Event-bank era (preferred)**: primaries + cascades sampled ONCE into a persistent full-record bank;
  the exact reweight `w(θ)` for all knobs is a cheap differentiable re-sum at plot/fit time — no
  regeneration, no Taylor. Modules: `event_bank`, `bank_plot`, `bank_reweight`, `bank_arrows`,
  `bank_matrix`, `info_content`, `validate`.
- **Walk-time era**: proposal + pool-walk replicas sampled at run time, autodiff through
  `full_knobs.model_hist_full`. Use when the frozen bank does not cover what you need (different settings,
  independent pseudo-experiments). Modules: `tune`, `grad_arrows`, `closure`.

**Conventions (all scripts):** run from the **repo root** (`ADoNIS/`), with the venv active, and
`python -u ... > logfile 2>&1 &` for anything long. T2K data files are read from `../nuisance/data/...`
relative to the CWD. The 1M event bank lives in the MAIN checkout — worktrees must set

```bash
export ADONIS_EVENT_BANK=/Users/homelab/Lab/Playground/projects/DIFFGEN/ADoNIS/output/event_bank
```

Run histories/caches go to `/tmp/adonis_tune_runs/` (+ `output/jac_cache/`); figures to `output/figures/`.

## Core libraries (imported, not usually run)

| module | what it provides |
|---|---|
| `full_knobs.py` | `nominal_knobs()` (the knob dict: hard-vertex, FSI, SF, norms; `M_A_qe`/`M_A_res` split per channel), `knob_specs()` (THE knob enumeration: labels, tuple expansion — single source of truth), `model_hist_full` (walk-time differentiable histogram), `build_hv_sf` |
| `bank_reweight.py` | `bank_weight(JB, knobs, grids)` — EXACT per-event `w(θ)`, jax-differentiable in every knob; `to_jax` (one-time device transfer), `weight_jit`, `default_grids` |
| `bank_plot.py` | `load_bank(dir)`; signal selectors (`signal_cc0pi`, `signal_cc1pi`, `signal_cc1pi_stv` — T2K windows + cosθ>cos70°) and observables (`dpt`, `dat`, `pN_1pi`, `dptt_1pi`, `dat_1pi`, multiplicities, ranked momenta) |
| `tune.py` | the CC0π tune blueprint: frozen proposal (`build_proposal`), θ-independent pool walk (`build_walk`/`bin_walk`/`build_replica`), `model_hist`, T2K STV data + covariance loading (`EDGES/DATA/COV/CONV`, `_dpt/_dat/_sel`). Runnable: the original 2-knob data tune |

The standard bank recipe:

```python
from analysis.t2k.differentiability import bank_plot as BP, bank_reweight as BR
from analysis.t2k.differentiability.full_knobs import nominal_knobs
B     = BP.load_bank(os.environ["ADONIS_EVENT_BANK"])
JB    = BR.to_jax(B)                    # one-time device transfer
grids = BR.default_grids()
w     = BR.weight_jit(JB, nominal_knobs(), grids)   # (N,) exact weights, differentiable in the knobs
```

Memory note (16 GB machine): differentiate via **per-knob `jax.jvp`** (forward) loops, never a wide
`jacfwd`/`jax.hessian` over many knobs — the FSI records make wide-tangent memory explode. Reverse-mode
(`value_and_grad`) through the bank is ~20× slower than forward here; prefer jvp-based fits
(see `info_content`'s LM engine).

## Runnable tools

### `event_bank.py` — build the bank (expensive: ~21 h for 1 M; do NOT rebuild casually)
```bash
python -u analysis/t2k/differentiability/event_bank.py   # see its header for N/chunking env knobs
```

### `bank_arrows.py` — exact per-knob variation pages from the bank (+20 %/+50 %, no Taylor)
```bash
python analysis/t2k/differentiability/bank_arrows.py [bankdir]           # CC0pi STV + Q2/W/cos/pmu/plead
python analysis/t2k/differentiability/bank_arrows.py cc1pi [stv] [bankdir]  # CC1pi topological / T2K STV
python analysis/t2k/differentiability/bank_arrows.py incl [bankdir]      # multiplicities + ranked momenta
# -> output/figures/{cc0pi,cc1pi,incl}_arrows_variation*.pdf
```

### `bank_matrix.py` — ADoNIS-vs-ACHILLES T2K matrix sourced from the bank (used by `make_plots --matrix`)
```bash
python analysis/t2k/make_plots.py --matrix          # the usual entry point
```

### `info_content.py` — per-bin/per-dataset Fisher decomposition + joint fits to T2K CC0π+CC1π⁺ STV
```bash
python -u analysis/t2k/differentiability/info_content.py                 # 4-param axial set (default)
python -u analysis/t2k/differentiability/info_content.py --set full17    # 2 hard-vertex + 4 SF + 11 FSI
python -u analysis/t2k/differentiability/info_content.py --set full17 --skip-grad  # fits-only restart
python -u analysis/t2k/differentiability/info_content.py --plot-only /tmp/adonis_tune_runs/info_content_<set>.npz
```
Stages (each gated): nominal identity vs `weight_jit` → per-knob per-bin AD-vs-FD → Fisher
`JᵀC⁻¹J` per dataset/bin + eigen-spectrum (flat directions) → **checkpoint**
(`..._grad.npz`, restartable) → pseudo-data closure → absolute + profiled-norm joint LM fits
(Gauss–Newton errors). Figures: `info_content_{gradient_heatmaps,fisher_breakdown,joint_fit}[_full17].png`.
Findings/evidence: `docs/logbook/info_content.md` (incl. the exact pion-FSI flat direction:
a common rescale of `sabs/s_piN_elastic/s_piN_cex/s_conv` is per-event invariant by construction of
`fsi_pion_reweight` — only 3 of the 4 pion knobs are independent).

### `validate.py` — validation suites (evidence for the logbook)
```bash
python -u analysis/t2k/differentiability/validate.py --sf       # SF knobs: 8 checks + sf_knob_checks.png
python -u analysis/t2k/differentiability/validate.py --fisher   # Eb FD eps-scan + pion null direction
#   (--fisher needs the info_content --set full17 gradient checkpoint)
```

### `grad_arrows.py` — walk-time Jacobian figures (fresh proposal+walks, autodiff of model_hist_full)
```bash
CC0PI_N=30000 python analysis/t2k/differentiability/grad_arrows.py              # 1D arrow grids (dpt+dat)
CC0PI_N=30000 python analysis/t2k/differentiability/grad_arrows.py --2d         # 2D (dpt,dat) heatmaps
CC0PI_N=1000000 CHUNK=80000 NREP=1 python analysis/t2k/differentiability/grad_arrows.py --hi-stats  # chunked, cached
python analysis/t2k/differentiability/grad_arrows.py --hi-stats --replot        # re-render from cache
python analysis/t2k/differentiability/grad_arrows.py --plot-only <npz>          # re-render one 1D grid
```

### `closure.py` — pseudo-data parameter-recovery closures (independent walks; the blueprint gate)
```bash
CC0PI_N=40000 python analysis/t2k/differentiability/closure.py [dpt|dat] [sabs*] [sscat*] [MA*] [--noA]
CC0PI_N=40000 python analysis/t2k/differentiability/closure.py --full [dpt|dat] [--noA]   # FIT registry knobs
python analysis/t2k/differentiability/closure.py --plot-only /tmp/adonis_tune_runs/closure*.npz
```

### `tune.py` — the original CC0π 2-knob data tune (the blueprint reference)
```bash
CC0PI_N=40000 CC0PI_NREP=4 python -u analysis/t2k/differentiability/tune.py [dpt|dat]
```

## Removed in the 2026-07 consolidation
`bank_taylor.py`, `bank_showcase.py`, `bank_validate.py` (all depended on the old Taylor-bank `D1/D2/D3`
fields that the full-record bank no longer stores); `bank_arrows_{incl,obs}.py` → `bank_arrows.py` modes;
`grad_{2d,all}.py` → `grad_arrows.py` flags; `closure_full.py` → `closure.py --full`;
`scripts/{info_content,sf_knob_checks,fd_gate_diagnose}.py` → `info_content.py` / `validate.py` here.

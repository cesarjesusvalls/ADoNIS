# Reproducing the paper figures

End-to-end runbook to regenerate the methods-report figures (`analysis/paper/methods_report.py`).
Each figure is ADoNIS (this repo) vs ACHILLES (a reference container). Two independent sides feed
the figures:

```
ACHILLES images ──▶ oracle files (reference side)  ─┐
                                                     ├─▶ analysis/paper/*  ─▶ output/paper/*.png
ADoNIS env ──▶ banks + Gate-I Jacobian (model side) ─┘
```

> **Post-refactor note (2026-07-27).** The reusable engines now live under `adonis/`
> (`adonis.fsi.cascade`, `adonis.flux`, `adonis.reweight`, `adonis.oracle`); `analysis/paper/` is
> only the paper-specific drivers. Generation defaults to the **jitted** cascade and, on a GPU node,
> the **turing GPU** (auto-selected by `jobs/env.sh`; ~25× on the QE cascade). Paths below reflect the
> new layout.

---

## 0. Layout

Clone **two** repos side by side (the pool cascade reads `QMC_configs.out.gz` from the sibling
ACHILLES checkout):

```
DIFFGEN/
├── ADoNIS/            # this repo
└── Achilles/          # the fork, sibling checkout (see §2)
```

## 1. Prerequisites (ADoNIS side)

- Python 3.11; `pip install -r requirements.txt` (CPU) or `pip install "jax[cuda12]"` for GPU.
- **ACHILLES data tables** the model reads (spectral function + DCC amplitudes + ANL meson-baryon):
  extracted from the oracle image into `$ACHILLES_DATA` (see `.github/workflows/ci.yml` for the exact
  `docker cp` list, or point `ACHILLES_DATA` at an existing `Achilles/data`).
- **Nucleon configs**: `../Achilles/data/configurations/QMC_configs.out.gz` (sibling checkout, §2).
- Committed: `data/nuclear/`, `data/experiment/`. JAX float64 is required (scripts set it).

## 2. ACHILLES images
- **oracle** (no-cascade, PUBLIC): `docker pull ghcr.io/cesarjesusvalls/achilles:oracle`.
- **cascade / fullcascade** (BUILD from the fork, must run native): `docker build -f
  docker/Dockerfile.{cascade,fullcascade} -t achilles:{cascade,fullcascade} ../Achilles`.
- The runner `adonis/oracle/run_achilles.py` picks the right image per card.

## 3. ACHILLES oracles (reference side)  → `output/achilles/`
```bash
python -m adonis.oracle.run_achilles configs/achilles/run_T2K_C_fsi_gauss.yml --seeds 8 --extract cc1pi_rich
python -m adonis.oracle.run_achilles configs/achilles/run_T2K_H.yml           --seeds 4 --extract cc1pi
python -m adonis.oracle.run_achilles configs/achilles/run_cascade_pip_C_broad.yml --seeds 48
python -m adonis.oracle.run_achilles configs/achilles/run_cascade_prot_C.yml      --seeds 8
python -m adonis.oracle.run_achilles configs/achilles/run_cascade_neut_C.yml      --seeds 8
```

## 4. ADoNIS banks + Gate-I Jacobian (model side)  → `output/`
```bash
# reweight bank (QE+RES, kind-1 FSI records) -- the reweight-bank generator (fast on GPU/turing now)
CC0PI_N=1000000 CHUNK=250000 python -u -m adonis.reweight.event_bank output/event_bank_v2

# Gate-I Jacobian (27 knobs x bins; shared J for sec2 + sec3)
ADONIS_EVENT_BANK=output/event_bank_v2 PHYSFIT_OBS=full ADONIS_LABEL=physfit_gate1_full_v2 \
  python -u -m analysis.paper.physical_fit          # -> output/altgen/physfit_gate1_full_v2.npz

# tagged-beam cascade banks
python -u -m analysis.paper.beams.beam_bank pip  --n 500000 --pmin 50  --pmax 1000 --chunk 50000 --out output/beam_pip_C
python -u -m analysis.paper.beams.beam_bank prot --n 500000 --pmin 300 --pmax 1400 --chunk 50000 --out output/beam_prot_C
python -u -m analysis.paper.beams.beam_bank neut --n 500000 --pmin 300 --pmax 1400 --chunk 50000 --out output/beam_neut_C
```

On S3DF, submit generation to the GPU via `jobs/submit.py --gpu` (turing); env.sh auto-selects the
`adonis-cuda` venv + `JAX_PLATFORMS=cuda,cpu` there.

## 4b. Sections 4-5 (closures + unknown-unknowns)
These render from persisted npz under `output/altgen/`. To regenerate from scratch you also need
GENIE/NEUT fake data (a separate GENIE container, `analysis/paper/physfit/run_genie.sh`):
```bash
analysis/paper/physfit/run_genie.sh                                     # -> output/altgen/genie_*.gst.root
PHYSFIT_MODE=closure PHYSFIT_METHOD=all GATE1_NPZ=output/altgen/physfit_gate1_full_v2.npz \
  ADONIS_LABEL=physfit_closure5 python -u -m analysis.paper.physfit.physical_fit_run   # sec4
#  sec5 modes: PHYSFIT_MODE in {inject2x, genie, q2mod, closure_q2mod, mecmix}
```

## 5. Regenerate the figures
```bash
python -m analysis.paper.sec1_validation.make      # ADoNIS-vs-ACHILLES validation (neutrino) + beams
python -m analysis.paper.sec2_gradients.make       # gradients for all 27 knobs (reads the Gate-I npz)
python -m analysis.paper.sec3_fisher.make          # Fisher per observable subset
python -m analysis.paper.beams.beam_fisher --syst 0.05      # knob x sample Fisher -> output/altgen/beam_fisher.npz
# sec4-5 figures (from the §4b npz):
python -m analysis.paper.physfit.physfit_closure5_fig      # fig7 closure5   (sec4)
python -m analysis.paper.physfit.physfit_report           # fig1-4 + PDF
python -m analysis.paper.physfit.physfit_q2nuis_fig       # fig9  (sec5)
python -m analysis.paper.physfit.physfit_mecmix_fig       # fig10 (sec5)
python -m analysis.paper.methods_report                   # -> output/paper/methods_report.html
```

## Status (post-refactor)
Everything is reproducible; all generators were recovered into the refactored layout:
- reweight bank -> `adonis.reweight.event_bank`; ACHILLES oracles -> `adonis.oracle.*`
- beam studies -> `analysis.paper.beams.{beam_bank,beam_fisher}`
- fit engine + sec4/5 + report figs -> `analysis.paper.{physical_fit,info_content}` +
  `analysis.paper.physfit.{physical_fit_run, physfit_*_fig, physfit_report, build_fakedata}`
(GENIE/NEUT fake-data generation for some sec5 modes still needs the external GENIE container.)

Sanity after a rebuild: `python -m pytest tests/test_pool_fsi_reweight.py -q` (FSI reweight identity).

# Reproducing every figure from scratch

End-to-end runbook to regenerate all figures in the methods report
(`analysis/paper/methods_report.py`) on a clean machine. Each figure is
ADoNIS (this repo) vs ACHILLES (a reference container). The pipeline is:

```
ACHILLES images ──▶ oracle files (reference side)  ─┐
                                                     ├─▶ figure scripts ─▶ output/paper/*.png ─▶ report
ADoNIS env ──▶ banks + Gate-I Jacobian (model side) ─┘
```

Two independent sides feed the figures; do the ACHILLES oracles and the ADoNIS
banks in parallel, then the figure scripts consume both.

---

## 0. Layout assumed

Clone **two** repos side by side (the ADoNIS pool cascade reads `QMC_configs.out.gz`
from the sibling ACHILLES checkout, and the cascade images are built from it):

```
DIFFGEN/
├── ADoNIS/            # this repo
├── Achilles/          # the fork, sibling checkout  (see §2)
└── nuisance/          # NUISANCE, for the T2K STV binning data (optional; see §1)
```

---

## 1. Prerequisites (ADoNIS side)

- **Docker**, **git**, **Python 3.11**.
- Python env:
  ```bash
  cd ADoNIS
  python3.11 -m venv .venv && . .venv/bin/activate
  pip install -r requirements.txt          # jax>=0.6, jaxlib, numpy, matplotlib, PyYAML, tqdm, pytest
  ```
- **ACHILLES data tables** the ADoNIS model reads (spectral function + DCC amplitudes).
  These come from the oracle image and are the single source of truth:
  ```bash
  python scripts/fetch_achilles_data.py     # -> data/achilles/{dcc_EW.dat, Spectral_Functions/pke12p_tot.data}
  ```
  (Override the location with `ACHILLES_DATA=/path/to/Achilles/data`.)
- **Nucleon configurations** for the pool cascade: `QMC_configs.out.gz`, resolved from
  `../Achilles/data/configurations/QMC_configs.out.gz` (the sibling checkout in §2 provides it).
- **Committed data** (already in the repo): `data/nuclear/` (12C + Ar densities),
  `data/experiment/` (ANL/BNL CC1π, JLab (e,e′), T2K CC0π STV).
- **T2K STV binning data** for the Gate-I datasets: the physical-fit binning reads the T2K CC0π STV
  release. The committed `data/experiment/t2k_cc0pi_stv/` covers it; a full NUISANCE checkout at
  `../nuisance` is the upstream source if you need to regenerate that.

Everything runs with JAX float64 (`jax.config.update("jax_enable_x64", True)`); the scripts set it
themselves — don't disable it (the FSI reweight's nominal identity depends on it).

---

## 2. ACHILLES images

Two kinds of run need two kinds of image (see `docs/CONTAINER.md` for the full rationale).

### 2a. No-cascade oracle image — PUBLIC, just pull
Used for **(e,e′)** and any no-FSI reference. amd64; on Apple Silicon it runs under emulation, which is
fine for no-cascade runs.
```bash
docker pull ghcr.io/cesarjesusvalls/achilles:oracle          # ACHILLES commit a4f761ea (v0.3.0-1)
```

### 2b. Cascade images — BUILD from the fork (not published)
The intra-nuclear cascade must run **natively** (under Rosetta emulation it SIGSEGVs at setup). Build the
two images from the ACHILLES fork, which is also the source for the QMC configs above:
```bash
cd ..
git clone https://github.com/cesarjesusvalls/Achilles-adonis.git Achilles
cd Achilles && git checkout 7054913f        # v0.3.0-6-g7054913f — the commit these figures were made from
cd ../ADoNIS

# neutrino + cascade generator (QE+RES+FSI):
docker build -f docker/Dockerfile.fullcascade -t achilles:fullcascade ../Achilles
# hadron-beam cascade (pi+/p/n on 12C, CrossSection mode):
docker build -f docker/Dockerfile.cascade     -t achilles:cascade     ../Achilles
```
The Dockerfiles patch ACHILLES's CMake to force-link the self-registering cascade interactions with
`--no-as-needed` and enable `ACHILLES_ENABLE_CASCADE_TEST=ON` (without this the cascade factory is empty
and setup crashes). On an amd64 host these build and run natively; on Apple Silicon build them native
arm64 (no `--platform`).

The runner `analysis/utils/run_achilles.py` picks the right image per card automatically (cascade cards →
`achilles:cascade` / `:fullcascade` native; everything else → `:oracle`).

---

## 3. ACHILLES oracles (reference side)

All write to `output/achilles/`. ACHILLES SIGSEGVs sporadically mid-run, so every generation batches over
seeds (`--seeds N`) and the hepmc written before a crash stays valid.

```bash
# --- T2K neutrino rich oracle (fullcascade image) -> the 11-observable reference ---
python -m analysis.utils.run_achilles configs/achilles/run_T2K_C_fsi_gauss.yml --seeds 8 --extract cc1pi_rich
#   -> output/achilles/t2k_cc1pi_rich_ach_FSI_proc.npz   (muon/proton/pion 4-vecs, proc code, weight)
python -m analysis.utils.run_achilles configs/achilles/run_T2K_H.yml          --seeds 4 --extract cc1pi   # free-H (CH)

# --- tagged-beam cascade oracles (cascade image), broad threshold-spanning beams ---
python -m analysis.utils.run_achilles configs/achilles/run_cascade_pip_C_broad.yml --seeds 48   # pi+ 50-1000 MeV/c
python -m analysis.utils.run_achilles configs/achilles/run_cascade_prot_C.yml      --seeds 8    # p 300-1400
python -m analysis.utils.run_achilles configs/achilles/run_cascade_neut_C.yml      --seeds 8    # n 300-1400
#   (pi+ needs more seeds: each ACHILLES seed dies early, and pi+ has the lowest reacted fraction)

# --- inclusive (e,e') oracles (oracle image, no cascade) -- only needed if/when the e-C driver lands ---
python -m analysis.utils.run_achilles configs/achilles/run_inclusive_ee_C_qe.yml  --seeds 4
python -m analysis.utils.run_achilles configs/achilles/run_inclusive_ee_C_res.yml --seeds 4
```

Each cascade card is `NEvents: 500000`, `KickMomentum: [pmin,pmax]`; `NEvents` counts **written
(reacted)** events, and the per-bin normalization comes from the running `GenCrossSection` counter.

---

## 4. ADoNIS banks + Gate-I Jacobian (model side)

All write to `output/`. Run long jobs to a logfile (`python -u ... > log 2>&1 &`).

```bash
# --- the frozen 1.87M-event bank (QE + RES, ragged kind-1 FSI records) ~2.5-3 h, ~1 GB ---
CC0PI_N=1000000 CHUNK=250000 python -u -m analysis.t2k.differentiability.event_bank output/event_bank_v2

# --- Gate-I Jacobian: 27 knobs x 188 bins (the shared J for sec2 + sec3) ~75 s on the bank ---
ADONIS_EVENT_BANK=output/event_bank_v2 PHYSFIT_OBS=full ADONIS_LABEL=physfit_gate1_full_v2 \
  python -u scripts/altgen/physical_fit.py                 # -> output/altgen/physfit_gate1_full_v2.npz

# --- tagged-beam banks (500k tried each; chunk 50k to stay off swap) ~2.5 h total ---
python -u -m analysis.beams.beam_bank pip  --n 500000 --pmin 50  --pmax 1000 --chunk 50000 --out output/beam_pip_C
python -u -m analysis.beams.beam_bank prot --n 500000 --pmin 300 --pmax 1400 --chunk 50000 --out output/beam_prot_C
python -u -m analysis.beams.beam_bank neut --n 500000 --pmin 300 --pmax 1400 --chunk 50000 --out output/beam_neut_C
```

### 4b. Sections 4–5 (closures + unknown-unknowns) — heavier, optional
These figures render from persisted npz under `output/altgen/`. To regenerate the npz from scratch you
also need **GENIE / NEUT** fake data (a separate LUCiD/GENIE container, `scripts/altgen/run_genie.sh` /
`run_neut.sh`) and the physical-fit engine:
```bash
scripts/altgen/run_genie.sh                                 # -> output/altgen/genie_t2k_12C_ar23_*.gst.root (LUCiD container)
PHYSFIT_MODE=closure PHYSFIT_METHOD=all GATE1_NPZ=output/altgen/physfit_gate1_full_v2.npz \
  PHYSFIT_INJECT="M_A_res=0.85,kF_sf=1.10,Eb_shift=3.0,s_NN_elastic[pn]=1.25,f_NN_cex=0.40" \
  ADONIS_LABEL=physfit_closure5 python -u scripts/altgen/physical_fit_run.py     # sec4
#  sec5 modes: PHYSFIT_MODE in {inject2x, genie, q2mod, closure_q2mod, mecmix} -- see scripts/altgen/physical_fit_run.py
```

---

## 5. Regenerate the figures

Fast (seconds–minutes each; they read the banks/npz above, no re-generation):

```bash
# section 1 -- ADoNIS vs ACHILLES validation (neutrino) + calls the beam figure
python -m analysis.paper.sec1_validation.make            # sec1_cc0pi/cc1pi/multiplicity + beams_validation
# section 2 -- gradients for all 27 knobs (reads physfit_gate1_full_v2.npz)
python -m analysis.paper.sec2_gradients.make             # sec2_gradients_all27, sec2_gradient_reach
# section 3 -- Fisher per observable subset
python -m analysis.paper.sec3_fisher.make                # sec3_shrinkage_subsets, sec3_failure_modes, sec3_degeneracy
# knob x sample Fisher (table + npz; the beam banks + Gate-I npz)
python -m analysis.beams.beam_fisher --syst 0.05         # -> output/altgen/beam_fisher.npz

# section 4-5 (from the persisted npz of §4b) -- exactly the figures the report embeds:
python scripts/altgen/physfit_closure5_fig.py            # fig7_closure5            (§4)
python scripts/altgen/physfit_report.py                  # fig1-4 (incl. fig3_inject2x, fig4_genie) + PDF
python scripts/altgen/physfit_q2nuis_fig.py              # fig9_q2nuis             (§5, inside-manifold)
python scripts/altgen/physfit_mecmix_fig.py              # fig10_mecmix            (§5, outside-manifold)

# the report itself (embeds the figures above)
python -m analysis.paper.methods_report                  # -> output/paper/methods_report.html
```

Outputs land in `output/paper/` (`.png` + `.pdf`) and `output/figures/`. `output/` is gitignored;
everything is regenerable from committed scripts.

---

## 6. Gotchas & runtimes

| item | note |
|---|---|
| JAX float64 | required; scripts set `jax_enable_x64`. The FSI nominal identity is exact only in x64. |
| bank memory | `event_bank` peaks a few GB; beam banks use `--chunk 50000` (250k thrashes a 16 GB box). |
| ACHILLES SIGSEGV | expected, mid-run; batch over `--seeds`. π⁺ needs ~48 seeds to match p/n statistics. |
| cascade image platform | build/run native (no `--platform`); the amd64 `:oracle` under emulation SIGSEGVs the cascade. |
| determinism | banks are seeded; ADoNIS-vs-ACHILLES agreement is statistical (χ²/ndf), not bit-identical. |
| runtimes (M-series) | event bank ~2.7 h · each beam bank ~2 h · Gate I ~75 s · figures seconds–minutes. |

Sanity check after a rebuild: `python -m pytest tests/test_pool_fsi_reweight.py -q` (13 tests) confirms
the FSI reweight (nominal identity, autodiff==FD, ragged==dense, the pion survival factor).

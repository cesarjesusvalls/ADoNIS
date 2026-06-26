# Running ADoNIS: generation, multiplicity plots, and the CC0π/CC1π cross-section matrix

**One generation feeds every plot.** The forward generator writes "engine-rich" banks that carry both the
per-particle final state (muon/proton/pion 4-vectors + weights) *and* the per-event multiplicity counts
(`n_p/n_n/n_pip/n_pi0/n_pim`). The multiplicity figures and the CC0π/CC1π cross-section matrix read the
**same** banks — you do **not** regenerate per plot.

- QE / CC0π bank: `data/oracle/t2k_cc0pi_engine_rich<tag>_batch<NN>.npz`
- RES / CC1π bank: `data/oracle/t2k_cc1pi_engine_rich<tag>_batch<NN>.npz`

All commands use the project venv: `.venv/bin/python -u …` (the `-u` keeps logs unbuffered — always
stream long jobs to a logfile, never pipe through `tail`).

---

## 1. Generate ADoNIS banks (the batched driver)

`scripts/adonis_generate_batched.py` spawns K worker processes; each runs M seeds of N events in one
process (JIT-compiles once, reused across its seeds) and writes its own checkpointed bank
`…_batch<NN>.npz`. Total events = K × M × N, in K bank files. Workers checkpoint after every seed, so a
bank is a valid (correctly normalized) partial-σ estimate at all times — you can plot live as it runs.

```bash
# QE / CC0π — 6 workers × 5 seeds × 25k = 750k events, distance-sync (default), refill
.venv/bin/python -u scripts/adonis_generate_batched.py configs/gen_c_qe_cv5.yaml \
    --workers 6 --seeds-per-worker 5 --n-per-seed 25000 --n-w 2048 --single-thread --tag _myqe \
    > /tmp/gen_myqe.log 2>&1 &

# RES / CC1π — same; RES builds the VEGAS importance grid ONCE up front (cached), then cascades
.venv/bin/python -u scripts/adonis_generate_batched.py configs/gen_c_res_cv5.yaml \
    --workers 6 --seeds-per-worker 5 --n-per-seed 25000 --n-w 2048 --single-thread --tag _myres \
    > /tmp/gen_myres.log 2>&1 &
```

Key flags / env:

| knob | meaning |
|---|---|
| `--workers K` | parallel worker processes. **Use ≤ 6** on this 8-core machine (leave headroom). |
| `--seeds-per-worker M`, `--n-per-seed N` | total = K·M·N events, in K bank files of M·N each |
| `--n-w 2048` | **refill** working set (events in flight). **Always use refill.** `--n-w 0` is lock-step: one slow event stalls the whole seed (and time-sync slow events can grind for hours). |
| `--single-thread` | 1 thread/worker → K workers fit K cores without oversubscribing |
| `--tag _foo` | output bank suffix; lets one config produce several named bank sets (no throwaway configs) |
| `--no-fsi` | PRE-FSI banks (primary products, no cascade); tag → `_nofsi` |
| `ADONIS_TIMESTEP=0\|1` | stepping clock: **0 = distance-sync (default, committed engine)**; 1 = time-sync (ACHILLES AdaptiveStep). Time-sync is experimental — physics-equivalent on forward observables, ~3× slower for decoupled, small P-invariance break for adaptive. Default 0 is the right choice. |
| `ADONIS_DECOUPLED=1` | (only with `TIMESTEP=1`) the `Δt=step` decoupled variant — experimental, slowest. |

Configs: `configs/gen_c_qe_*.yaml` (CC0π/QE), `configs/gen_c_res_*.yaml` (CC1π/RES; has the `vegas:`
block). `material: C` (carbon) or `Ar`. The cascade block (`step`, `nn_inelastic`, `path_budget_R`,
`time_step`) maps to `CascadeHyperparams`; `--n-per-seed`/`--seeds-per-worker`/`--tag` override the YAML.

Roughly (8 workers, refill, time-sync; distance-sync is faster): **1M QE ≈ 18 min, 1M RES ≈ 37 min**.

---

## 2. Multiplicity plots (`dσ` vs final-state multiplicity)

`scripts/multiplicity_xsec.py` — absolute dσ vs N(p)/N(n)/N(π+)/N(π0)/N(π−), QE *or* RES, ADoNIS vs
ACHILLES (top panel + ACH/ADO ratio), under the matrix's muon-acceptance CC-inclusive cut. It **globs
`…_batch*`** banks for the given `TAG` and averages them — so it consumes the batched output directly.

```bash
# QE multiplicity
TAG=myqe CHANNEL=qe ACH_BANK=data/oracle/t2k_qe_gauss_ach_proc.npz \
    .venv/bin/python -u scripts/multiplicity_xsec.py /tmp/mult_qe.png

# RES multiplicity
TAG=myres CHANNEL=res ACH_BANK=data/oracle/t2k_C_res_gauss_proc.npz \
    .venv/bin/python -u scripts/multiplicity_xsec.py /tmp/mult_res.png
```

Env: `TAG` (bank tag), `CHANNEL` (`qe`|`res`), `ACH_BANK` (ACHILLES proc reference), `ADO_DIR` (bank dir,
default `data/oracle`), `MODE` (`fsi`|`nofsi`). The script prints total σ ACH/ADO and the per-bin
dσ table, and writes the PNG.

---

## 3. CC0π / CC1π cross-section matrix

`scripts/gen_cc_matrix.py` builds the full {CC0π,CC1π} × {QE,RES,QE+RES} × {incl,0p,1p,2p} × {C,Ar}
figure matrix (ratio + χ²) plus a summary table, ADoNIS vs ACHILLES, into `paper_figures/matrix/`.

```bash
.venv/bin/python -u scripts/gen_cc_matrix.py C fsi    # [C|Ar|all] [fsi|nofsi]
```

It reads **single-file** engine-rich banks (the `MATERIALS` table, e.g. `…_cv30.npz`/`…_cv12.npz`) or, to
use your own banks, the env overrides `ADONIS_QE_BANK` / `ADONIS_RES_BANK`. The analyzer **sums** `w`
across the banks it's given, so the batched `_batch*` files must first be **merged into one file**
(concatenate, divide `w` by the number of batches — each batch's `w` already sums to σ):

```bash
# merge batched output -> one bank per channel, for the matrix
.venv/bin/python - <<'PY'
import numpy as np, glob
for ch, tag in [("cc0pi", "myqe"), ("cc1pi", "myres")]:
    fs = sorted(glob.glob(f"data/oracle/t2k_{ch}_engine_rich_{tag}_batch*.npz"))
    keys = np.load(fs[0], allow_pickle=True).files
    d = {k: np.concatenate([np.load(f, allow_pickle=True)[k] for f in fs]) for k in keys}
    d["w"] = d["w"] / len(fs)                       # average the per-batch sigma estimates
    out = f"data/oracle/t2k_{ch}_engine_rich_{tag}_merged.npz"; np.savez(out, **d)
    print("wrote", out, len(d["w"]), "events")
PY

ADONIS_QE_BANK=data/oracle/t2k_cc0pi_engine_rich_myqe_merged.npz \
ADONIS_RES_BANK=data/oracle/t2k_cc1pi_engine_rich_myres_merged.npz \
    .venv/bin/python -u scripts/gen_cc_matrix.py C fsi
```

**ACHILLES reference for the matrix** is a *combined* QE+RES proc bank (`MATERIALS["C"][2] =
t2k_cc1pi_rich_ach_FSI_proc.npz`, filtered per cell by `ref_proc` = 200 QE / 401,402 RES). This is a
separate ACHILLES product (HepMC → proc-bank extraction; see `docs/CONTAINER.md` and the `extract_*` /
`gen_ach_*` scripts) and is **not** the per-channel `t2k_qe_gauss_ach_proc.npz` / `t2k_C_res_gauss_proc.npz`
banks used for the multiplicity plots. If it's absent, regenerate it before running the matrix (or point
`MATERIALS` / the ACH ref at an available combined bank).

---

## Notes / gotchas

- **Same banks, two plots.** Generate once (§1); make multiplicity (§2, globs `_batch*`) and the matrix
  (§3, merged single file). The engine-rich bank has both the 4-vectors and the multiplicity counts.
- **Always refill (`--n-w 2048`), never lock-step (`--n-w 0`).** Lock-step stalls on a single slow event
  (hours for time-sync). Refill is bit-equivalent and fast. (Multi-primary channels — RES has pion+recoil
  — depend on the refill wait-queue carrying the overflow primary; validated by `scripts/test_p_invariance.py`.)
- **Distance-sync is the default and the right choice.** `time_step`/`decoupled` are experimental and a
  wash on forward observables — see `docs/logbook/cascade_subcascade_sequencing.md`.
- **Validation harnesses:** `scripts/cascade_seq_test.py` (identical-input transport ablation vs an
  ACHILLES `CASCADEDUMP`), `scripts/cascade_ablation.py`, `scripts/test_p_invariance.py` (P=1≡P=12 gate).
- Banks live in `data/oracle/` (gitignored). Figures: multiplicity → wherever you point the output PNG;
  matrix → `paper_figures/matrix/` (+ `paper_figures/matrix_nofsi/`).

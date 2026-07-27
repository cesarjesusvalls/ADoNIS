# Running ADoNIS: generation, multiplicity plots, and the CC0π/CC1π cross-section matrix

**One generation feeds every plot.** The forward generator writes per-event banks that carry both the
per-particle final state (muon/proton/pion 4-vectors + weights) *and* the per-event multiplicity counts
(`n_p/n_n/n_pip/n_pi0/n_pim`). The multiplicity figures and the CC0π/CC1π cross-section matrix read the
**same** banks — you do **not** regenerate per plot.

Bank name = `{flux}_{material}_{chanout}{tag}.npz` (every part threaded from the config; `tag` empty for
the canonical run), written to `output/adonis/`:

- QE / CC0π bank: `output/adonis/t2k_C_cc0pi<tag>_batch<NN>.npz`
- RES / CC1π bank: `output/adonis/t2k_C_cc1pi<tag>_batch<NN>.npz`

All commands use the project venv: `.venv/bin/python -u …` (the `-u` keeps logs unbuffered — always
stream long jobs to a logfile, never pipe through `tail`).

---

## 1. Generate ADoNIS banks (one CLI, single process or K workers)

`python -m adonis.workflow.cli <config>` is the single entry point (the generation logic lives in
`adonis.workflow.generate`). `--workers <= 1` runs in-process; `--workers K` re-invokes the CLI once per
shard as a subprocess — each runs M seeds of N events in one process (JIT once, reused across its seeds)
and writes its own checkpointed `…_batch<NN>.npz`. Total events = K × M × N. Workers checkpoint after
every seed, so a bank is a valid (correctly normalized) partial-σ estimate at all times — plot live as it
runs. RES warms up the VEGAS grid ONCE up front (cached) before the workers fan out.

```bash
# QE / CC0π — 6 workers × 5 seeds × 25k = 750k events, distance-sync (default), refill
.venv/bin/python -u -m adonis.workflow.cli configs/gen_c_qe_t2k.yaml \
    --workers 6 --seeds-per-worker 5 --n-per-seed 25000 --n-w 2048 --single-thread \
    > /tmp/gen_qe.log 2>&1 &

# RES / CC1π — same; builds + caches the VEGAS importance grid up front, then cascades
.venv/bin/python -u -m adonis.workflow.cli configs/gen_c_res_t2k.yaml \
    --workers 6 --seeds-per-worker 5 --n-per-seed 25000 --n-w 2048 --single-thread \
    > /tmp/gen_res.log 2>&1 &

# one process (in-process), e.g. a quick smoke run
.venv/bin/python -u -m adonis.workflow.cli configs/gen_c_qe_t2k.yaml --n-seeds 1 --n-per-seed 1000 --n-w 256
```

Key flags / env:

| knob | meaning |
|---|---|
| `--workers K` | parallel worker processes (`>1` ⇒ batched). **Use ≤ 6** on this 8-core machine. `1` (default) ⇒ in-process. |
| `--seeds-per-worker M`, `--n-per-seed N` | total = K·M·N events, in K bank files of M·N each (single process: `--n-seeds`/`--n-per-seed`). |
| `--n-w 2048` | **refill** working set (events in flight). **Always use refill.** `--n-w 0` is lock-step: one slow event stalls the whole seed. |
| `--single-thread` | 1 thread/worker → K workers fit K cores without oversubscribing |
| `--tag _foo` | extra bank suffix; lets one config produce several named bank sets (no throwaway configs). Canonical run: omit (empty). |
| `--no-fsi` | PRE-FSI banks (primary products, no cascade); tag → `_nofsi` |
| `ADONIS_TIMESTEP=0\|1` | stepping clock: **0 = distance-sync (default, committed engine)**; 1 = time-sync (experimental, physics-equivalent on forward observables). Default 0 is the right choice. |

Configs: `configs/gen_{c,ar}_{qe,res}.yaml`. Each sets `flux` (t2k; only flux wired today), `material`
(`C`/`Ar`), `channels`, the `cascade` block (`step`, `nn_inelastic`, `mprot` → `CascadeHyperparams`),
`vegas:` (RES only), `out_dir: output/adonis`, and `tag` (empty). CLI flags override the YAML.

Roughly (6 workers, refill): **1M QE ≈ 18 min, 1M RES ≈ 37 min**.

---

## 2. Multiplicity plots (`dσ` vs final-state multiplicity)

`scripts/multiplicity_xsec.py` — absolute dσ vs N(p)/N(n)/N(π+)/N(π0)/N(π−), QE *or* RES, ADoNIS vs
ACHILLES (top panel + ACH/ADO ratio), under the matrix's muon-acceptance CC-inclusive cut. It **globs
`…_batch*`** banks and averages them — so it consumes the batched output directly.

```bash
# QE multiplicity (carbon)
MAT=C CHANNEL=qe ACH_BANK=output/achilles/t2k_qe_gauss_ach_proc.npz \
    .venv/bin/python -u scripts/multiplicity_xsec.py /tmp/mult_qe.png

# RES multiplicity (carbon)
MAT=C CHANNEL=res ACH_BANK=output/achilles/t2k_C_res_gauss_proc.npz \
    .venv/bin/python -u scripts/multiplicity_xsec.py /tmp/mult_res.png
```

Env: `MAT` (`C`|`Ar`, material in the bank name), `FLUX` (`t2k`), `TAG` (extra variant suffix; empty for
canonical), `CHANNEL` (`qe`|`res`), `ACH_BANK` (ACHILLES proc reference), `ADO_DIR` (bank dir, default
`output/adonis`), `MODE` (`fsi`|`nofsi`). Prints total σ ACH/ADO + the per-bin dσ table, writes the PNG.

---

## 3. CC0π / CC1π cross-section matrix

`scripts/gen_cc_matrix.py` builds the full {CC0π,CC1π} × {QE,RES,QE+RES} × {incl,0p,1p,2p} × {C,Ar}
figure matrix (ratio + χ²) plus a summary table, ADoNIS vs ACHILLES, into `paper_figures/matrix/`.

```bash
.venv/bin/python -u scripts/gen_cc_matrix.py C fsi    # [C|Ar|all] [fsi|nofsi]
```

It reads **single-file** banks (the `MATERIALS` table → `output/adonis/t2k_C_cc0pi.npz` etc.) or, to use
your own, the env overrides `ADONIS_QE_BANK` / `ADONIS_RES_BANK`. The analyzer **sums** `w` across the
banks it's given, so the batched `_batch*` files must first be **merged into one file** (concatenate,
divide `w` by the number of batches — each batch's `w` already sums to σ):

```bash
# merge batched output -> one bank per channel, for the matrix
.venv/bin/python - <<'PY'
import numpy as np, glob
for ch in ("cc0pi", "cc1pi"):
    fs = sorted(glob.glob(f"output/adonis/t2k_C_{ch}_batch*.npz"))
    keys = np.load(fs[0], allow_pickle=True).files
    d = {k: np.concatenate([np.load(f, allow_pickle=True)[k] for f in fs]) for k in keys}
    d["w"] = d["w"] / len(fs)                       # average the per-batch sigma estimates
    out = f"output/adonis/t2k_C_{ch}.npz"; np.savez(out, **d)
    print("wrote", out, len(d["w"]), "events")
PY

.venv/bin/python -u scripts/gen_cc_matrix.py C fsi
```

**ACHILLES reference for the matrix** is a *combined* QE+RES proc bank (`MATERIALS["C"][2]`, filtered per
cell by `ref_proc` = 200 QE / 401,402 RES). This is a separate ACHILLES product (HepMC → proc-bank
extraction; see `docs/CONTAINER.md` and the `extract_*` / `gen_ach_*` scripts). The ACHILLES bank
filenames/location are pending the separate ACHILLES-output naming cleanup.

---

## Notes / gotchas

- **Same banks, two plots.** Generate once (§1); make multiplicity (§2, globs `_batch*`) and the matrix
  (§3, merged single file). The bank has both the 4-vectors and the multiplicity counts.
- **Always refill (`--n-w 2048`), never lock-step (`--n-w 0`).** Lock-step stalls on a single slow event.
  (Multi-primary channels — RES has pion+recoil — depend on the refill wait-queue carrying the overflow
  primary; validated by `scripts/test_p_invariance.py`.)
- **Distance-sync is the default and the right choice.** `time_step` is experimental and a wash on forward
  observables — see `docs/logbook/cascade_subcascade_sequencing.md`.
- **Validation harnesses:** `scripts/cascade_seq_test.py` (identical-input transport ablation vs an
  ACHILLES `CASCADEDUMP`), `scripts/cascade_ablation.py`, `scripts/test_p_invariance.py` (P=1≡P=12 gate).
- Banks live in `output/adonis/` (gitignored). ACHILLES products in `output/achilles/`. Figures: multiplicity
  → wherever you point the output PNG; matrix → `paper_figures/matrix/` (+ `paper_figures/matrix_nofsi/`).

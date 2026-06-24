# Cascade refill rewrite — consistency audit (2026-06-24)

Audit of commits `be75d63`→`af879d9` (Stages 0–3b + gen wiring) against the plan
(`cascade_persistent_refill_plan.md`).  The plan's stated goal: **"supersede the cascade driver IN PLACE
with a persistent-working-set engine"** + **"the ONLY allowed behavioural change: particles silently
dropped on overflow are now KEPT."**  Method: grep every caller of `cascade_nucleus`/`run_fsi` and check
which mechanisms each actually uses.

## Headline: mechanisms were BUILT + VALIDATED but not WIRED into the default path
Every production consumer still runs the **lock-step, no-queue, drop-on-overflow** path.  The new
machinery only runs in test/study scripts and the one segment generator.

| mechanism | built? | gated? | wired into default path? |
|---|---|---|---|
| per-event RNG (3a) | yes | yes | **YES** (global default) — consistent |
| particle queue Q (Stage 2) | yes | yes (stress) | **NO** — `q_cap` defaults 0; `_cascade_pool` never forwards it; no consumer passes it |
| refill / n_w (3b) | yes | yes (C) | **NO** — only `gen_cascade_segments` (via `ADONIS_NW`); everything else lock-step |
| per_event_cap (3b) | yes | implied | **NO** — no consumer sets it; default == global max_steps |
| multi-worker (Stage 4) | — | — | DIVERGED — process-level seed sharding, not the in-engine cursor shard the plan describes |

## Findings (by severity)

### A. Correctness — the plan's ONE allowed change is OFF everywhere
- **A1. Q never wired.** `run_cascade_pool(q_cap=0)`; the two `_cascade_pool` calls omit it;
  `cascade_nucleus`/`run_fsi`/`generate.py` have no `q_cap` param.  ⇒ "overflow particles are now KEPT"
  — the correctness improvement that justified the rewrite — is **inactive in all production runs**.
  (For C@P=16 overflow is incidentally 0, so it didn't bite; for Ar@P=16 it was ~0.66% = real dropped
  particles, still being dropped.)
- **A2. `M_out=24` is a hardcoded context-property literal** (appears as `24` in both `_cascade_pool`
  calls + the default).  It caps escaped finals/event → **out-overflow (`oofl`)**, which the queue does
  NOT address (Q only feeds the active stack).  An independent silent-drop cap with a magic number; a
  bug-in-waiting per the coding-discipline rule. Not derived from anything.

### B. Consistency — refill is opt-in, so the headline workloads never get it
The 1.3× refill + per-event caps reach **only the cascade-segment matrix**.  Lock-step is still used by:
- `adonis/workflow/generate.py` (main forward generator → **cross-section banks**),
- `scripts/gen_cc_engine_rich.py` (the **cross-section matrix** banks the user actually wanted),
- `adonis/fsi/pool_fsi.run_fsi` + `scripts/cc0pi_tune_adonis.py` (the **differentiable tune blueprint**).
⇒ "supersede IN PLACE" did not happen: we shipped a **second code path** (`pending` branch) and left it
non-default, the opposite of one-engine.  (Note tension with the memory "always pool, one validated
engine, no branches": there are now two paths in `run_cascade_pool`.)

### C. Validation gaps
- **C1. Refill gated on C only.** Ar (where overflow/queue actually matter) never run through refill.
- **C2. refill + Q together never executed.** Gate + overnight both used `q_cap=0`; the refill loop's
  wait-buffer reset (`wait3` on finished) has never run with Q>0.
- **C3. per_event_cap never exercised** with a value ≠ max_steps.
- **C4. Worker count never benchmarked** — 2-worker overnight choice was inferred, not measured;
  `_engine_workers.sh` written but not run.
- **C5. n_w optimum** from a 3-point sweep (none/1024/4096); n_w<1024 untested.

### D. Design divergences (document, maybe accept)
- **D1. Stage 4** = independent OS processes sharded by SEED (`ADONIS_SEED_LIST`), not the plan's
  in-engine `shard_map`/cursor.  Pragmatic + crash-isolated, but each process still runs a single n_w
  window; not the described design.
- **D2. `n_w=1024` literal** lives only as a gen env default, not a central/derived engine constant.

## Recommended end-state (to make it consistent)
1. Thread `n_w`, `q_cap`, `per_event_cap` through `cascade_nucleus → _cascade_pool → run_cascade_pool`
   AND `pool_fsi.run_fsi`, `generate.py`, `gen_cc_engine_rich.py`, `cc0pi_tune_adonis.py` so refill+queue
   are the DEFAULT for every consumer (true "supersede in place").
2. Enable Q by default, auto-sized (e.g. Q from P / measured max concurrency), so `sofl≡0` by
   construction — not incidental.  Centralize `M_out` likewise and size it from data (max finals/event)
   so `oofl≡0`; or feed escaped finals through the same keep-don't-drop discipline.
3. Re-validate: extend the bit-exact gate to **Ar** and to **refill+Q on**; confirm `sofl=oofl=0`.
4. Benchmark workers (1/2/4/8) × n_w{256,512,1024,2048} to set defaults from measurement, not inference.
5. Produce the actual **cross-section matrix** (`gen_cc_matrix`) with the unified engine.

## Not a regression
Per-event RNG (3a) + refill (3b) are bit-exact to the pre-refill distributions (gates), so none of the
above changed physics — these are wiring/consistency gaps, not silent physics drift.

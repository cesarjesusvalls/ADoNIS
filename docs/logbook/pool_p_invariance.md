# Pool cascade P-invariance (stack width P made a true capacity dial)

## The bug
The pooled cascade's result depended on the stack width **P** (particles processed per global step).
At P below an event's concurrent occupancy, results diverged silently: σ stayed exact and `overflow`
read 0, but the **exclusive final state** was wrong.  Severe for RES (occupancy ~5-10): at P=1,
N(p)=0 was **9× ACHILLES** (5.9e-6 vs 6.6e-7); mild for QE.  Production (P=10/12 ≥ occupancy) was always
correct — the bug only bit below occupancy.  "P is a pure capacity dial by construction" was FALSE
below occupancy.

## Root cause (4 couplings, all from global-step ≠ per-particle-time when M<occupancy via queuing)
1. **RNG keyed by `(evt_id, nstep, slot m)`** — a particle's randoms depended on its slot and the global
   step; queued particles re-keyed differently.
2. **Initial overflow dropped** — `compact(init, M)` discarded primaries beyond M (RES has 2: pion+recoil
   → at M=1 one was lost). The dominant small-P RES catastrophe.
3. **`consumed` background-depletion applied in global-step (slot) order** — order-dependent shared state.
4. **Global `max_steps` cap** — serialized small-P runs need ~occupancy× more global steps → truncated
   by a fixed global cap, P-dependently.

## The fix (one mechanism: per-particle identity + canonical absolute-time processing)
- **Per-particle RNG identity** (`cascade_full.py`): carry `pkey` (lineage RNG key) + `lstep` (own step
  count) as stack fields. Key each step by `fold(pkey, lstep)`; daughter `pkey =
  fold(fold(parent_pkey, channel), parent_lstep)`. RNG now P-invariant.
- **Initial overflow → wait queue**: pack init into M active + Q waiting (mirrors pool_reconcile), so no
  primary is dropped.
- **Canonical processing order** (`consumed` fix): `compact(..., sort_priority=True)` packs the active
  stack by `(gtime, sid)` — `gtime` = ABSOLUTE cascade time (= global step at P≥occupancy; daughters
  inherit `gtime = parent_gtime+1`), `sid` = full `pkey`. `pool_reconcile` + the init pack use it, so
  shared `consumed` claims land in particle-time order at ANY P.
- **Per-particle step cap** (`max_steps` fix): drop a particle when ITS OWN `lstep ≥ cfg.max_steps`; the
  loop runs until no particle is alive (global counter is a non-binding backstop). No P-dependent
  truncation.

## Validation (scripts/test_p_invariance.py — the oracle)
- Per-event exact match vs P=12: **100.00%** at P=1,2,4 for BOTH RES and QE. The engine is bit-exactly
  P-invariant. (Before: P=1 was 2.8%/1.6%.)
- Re-validation vs ACHILLES at the nominal P=1 (240k ev/channel): RES σ ACH/ADO 1.000, QE 1.010,
  multiplicity ~1-2% — i.e. P=1 matches ACHILLES *as well as* the old P=10 engine (no degradation from
  the intentionally-changed RNG/order stream).
- Performance: per-step `lexsort` of ~M+Q≈76 elems is negligible vs the physics; large P still
  early-exits at ~occupancy steps (no extra cost). P=1 nominal validated.

## Commits
2d76ba9 (phase1/2 RNG), 9aba70d (phase3 init-overflow), 7a5a3f0 (phase4 canonical order + per-particle cap).

## Status
P is now a true capacity/perf dial. The pre-existing secondary-neutron multi-knockout transport residual
(QE high-N(n) tail, RES leading-neutron soft tail) is SEPARATE from P (present at all P) and remains a
parked transport item, not a P-invariance issue.

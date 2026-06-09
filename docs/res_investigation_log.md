# RES low-Q² deficit — investigation log

**Single source of truth for this investigation. Update it every step. Before running any
check, look here first — if it's in the LEDGER, do NOT re-run it.**

## Purpose & acceptance
Reproduce ACHILLES CC single-pion σ (ν_μ + ¹²C, T2K flux, no FSI) in ADoNIS `res_xsec`.
**Acceptance = the diagnostic figure `res_WQ2_shapes.png` ratio panels (ADoNIS/ACHILLES vs W and
vs Q²) sit on 1.0 within MC errors**, the way QE already does. Equivalence by *reading* ACHILLES,
not fitting.

## The diagnostic (primary deliverable, regenerated every iteration)
- `paper_figures/diagnostic_WQ2.py` — builds dσ/dW & dσ/dQ² with ratio panels, ADoNIS vs the
  **real ACHILLES hepmc events** (ground truth, not a measured target).
  - RES → `paper_figures/res_WQ2_shapes.png`
  - QE  → `paper_figures/qe_WQ2_shapes_5x.png` (CONTROL — must stay flat at 1.0)
- ACHILLES reference = `Achilles/_resrun_out/nofsi_res_hi.hepmc` (200k events), σ_total read from the
  run's `Total xsec` line. Q² = −(k_nu−k_mu)², k_nu = max-E pid14 (NOT the E=30 status-4
  placeholder), k_mu = pid13 status1. W = M(Nπ) = M(p_N+p_pi) = M(q+p_struck).

## Current state of the figure (baseline, before any fix)
- RES total ADoNIS/ACH = **0.733**. Q² ratio rises **0.65 (Q²→0) → ~1.0 (Q²≈1.5 GeV²)**.
  W ratio ~0.6–0.8, worst at threshold/Δ-peak, →1 in the tail.
- QE total ADoNIS/ACH = **0.984**, Q² ratio **flat at 1.0** including Q²→0.
- **The deficit is Q²-dependent (low-Q²), NOT a flat 0.74× constant.** This is THE feature to explain.

## LEDGER A — ESTABLISHED FACTS (validated; do NOT re-test)
| # | Fact | Evidence | When |
|---|------|----------|------|
| A1 | ACHILLES σ_RES = **1.6947472705414825e-5 nb** (converged) | ran `run_nofsi_res.yml` in image f3600572c01f; `Total xsec` line | this session |
| A2 | ACHILLES per-channel: n→pπ⁰ 3.258e-6, n→nπ⁺ 2.178e-6, p→pπ⁺ 1.1511e-5 | reslog `Process xsec:` lines | this session |
| A3 | At ACHILLES dump points, my **flux, initwgt(pke12p/n), spinavg=0.5** all match **ratio 1.000000** | reconstruct vs RESDUMP | this session |
| A4 | At ACHILLES dump points, my **amps2** matches, weighted 1.004, and per-Q²-bin incl. Q²∈[0,0.1]=1.003 | test_xsec_res_current + broad2.log | prior |
| A5 | RES **psw** reproduced to 1.4e-14 | validate_res_psw | prior |
| A6 | 3-body **Φ₃ volume** correct to 1.0002 (boosted parent) | analytic recursive check | this session |
| A7 | **QE importance** path = 0.994×, Q² ratio flat → struck-sampler+flux+beam+reporting+2body all correct, incl. low Q² | qe_xsec.sample_importance vs ACH 4.313e-5 | this session |
| A8 | QE flat & importance estimators agree when converged (flat 1.009±0.024 @800k×8) | high-stats convergence | this session |

## LEDGER B — RULED-OUT hypotheses (do NOT revisit)
| Hypothesis | Test | Verdict |
|---|---|---|
| Shared process[0]=(m_p,m_π0) masses for all channels (DIVERGENCE 1) | `res_shared_mass_sigma.py`: 1.243e-5 (0.734) vs per-channel 1.247e-5 (0.736) | **<1% effect — NOT the cause** |
| Target number was wrong/measured | ran ACHILLES, Total xsec = 1.6947e-5 exactly | **target confirmed real** |
| amps2 normalization (_NORM) off | A4 (1.004 weighted, flat in Q²) | **not it** |
| 3-body measure total volume wrong | A6 (Φ₃ 1.0002) | **not it (volume); differential in Q² UNTESTED** |
| flat-vs-importance estimator bug | A8 (agree when converged) | **not it** |

## OPEN — divergences to close (from res_chain_full_trace.md) + their expected figure impact
Order: close each, regenerate figure, record impact.
- [ ] **D1d/1e** per-channel masses in s23max/SqLam & amps2 eval point — *expected tiny per A-ledger, but close for cleanliness*
- [ ] **D3** proton channel uses pke12n importance instead of pke12p — proton channel is 68% of σ; close it (build a pke12p importance sampler for the proton channel).
- [ ] **the real one (TBD):** Q²-differential measure or de-Forest-Q²-gate effect that suppresses low Q² in the 3-body, common to res_xsec & res_exact. Φ₆ volume passes but Q²-differential untested.

## DECISIVE pending diagnostic (settles amplitude-vs-measure)
Bin **my exclusive_amps2 evaluated on the real ACHILLES hepmc events** into Q². If it reproduces
ACHILLES dσ/dQ² shape → amplitude right on the true forward distribution → bug is purely my
**sampling/measure**. If low at low Q² → **amplitude/gate**. (Not yet run.)

## ITERATION LOG
### Iter 0 (baseline) — regenerate the diagnostic at high stats vs real hepmc — DONE
- Tooling: `paper_figures/diagnostic_WQ2.py` (canonical, RES+QE), ACHILLES ref = real hepmc
  (`nofsi_res_hi.hepmc`, 200k events), ADoNIS = `res_xsec.generate` 600k, chunked w/ % progress.
- **Speed fix (so iterations are fast):** `exclusive_amps2_batch` was 380 µs/ev (JAX-eager spline =
  72%). Added NumPy spline twin (`spline.py::interp2d_spline_np`) AND NumPy **bilinear**
  (`amplitudes.py::amplitudes_bilinear_np`); switch `dcc_current.BATCH_INTERP` ∈ {"bilinear"(default,
  ~0.3% vs spline, 45×faster),"spline"}. Now **45 µs/ev** → 600k in ~2 min. amps2 regression test
  still passes (scalar path still uses spline).
- **Result (baseline):** total ADoNIS/ACH = **0.726**.
  - Q² ratio: **0.62 (Q²→0) → ~1.0 (Q²≈1.8 GeV²)**, monotonic. Q²<0.2: 0.650; Q²≥0.2: 0.777.
  - W ratio: ~0.65 threshold, ~0.72–0.76 Δ-peak, ~0.6–0.7 tail (roughly flat, low).
  - Figure: `paper_figures/res_WQ2_shapes.png`. Matches the prior diagnostic — baseline confirmed.
- **Learning:** the signature is unchanged & robust at high stats — a **low-Q² deficit** that closes
  by Q²≈1.5–1.8. Next: close OPEN divergences (D1d/1e, D3) and the decisive amps2-on-hepmc test,
  regenerating this figure each time.

### Iter 1 — D3 (proton-channel spectral function) — DONE, ~0 impact
- Change: res_xsec proposes the struck nucleon from pke12n (|p|²S_n) for all channels; the
  PROTON channel's integrand carries S_p. Added importance reweight ×S_p/S_n for ipid==2212
  (res_xsec.py). NOTE: this is the importance-equivalent of what ACHILLES does, NOT a
  transliteration — ACHILLES samples the struck nucleon FLAT and evaluates N·S_p *explicitly*
  (`res_spectral_model.f90:184-188`). Same integral, different operation.
- Result: total ADoNIS/ACH 0.726 → **0.726**; Q²<0.2 0.650 → **0.650**. **Zero change.**
- Learning: confirmed pke12p ≈ pke12n for ¹²C (isoscalar) ⇒ reweight ≈ 1. D3 is bookkeeping
  correctness only; it does NOT touch the low-Q² deficit. (Decision pending: keep reweight vs
  switch to strict explicit-S transliteration — both give this same number.)

### Iter 2 — D1+D1b/c/d/e (faithful ProcessGroup transliteration) — DONE, deficit unchanged
- Change: rewrote `res_xsec.generate` as the ACHILLES ProcessGroup mirror — ONE shared point from
  process[0]=(m_p,m_π0,Smin0), FLAT struck (so initwgt = N·S_channel is EXPLICIT, no S_p/S_n
  reweight), all 3 channels' amps2 on the shared momenta, per-channel flux, summed. Old importance
  path kept as `generate_importance`; **switch via `generate(..., method=)` or `RES_METHOD`**.
- Result: faithful σ/ACH = **0.755 ± 0.023** (200k×5); importance = 0.721. Consistent.
- Learning: strict transliteration (closing D1+D1b/c/d/e + D3 together, no tricks) gives the SAME
  deficit. **D1/D3 are conclusively NOT the cause.** Faithful is ~5× noisier / 3× slower (flat
  struck + 3 amps2/draw); importance is the better tool for clean diagnostic shapes (proven equal).

### Iter 3 — DECISIVE: FREE-NUCLEON bisection (struck at rest) — **LOCALIZED**
Tooling: `paper_figures/free_nucleon_WQ2.py` + ACHILLES hydrogen/free-neutron runs
(`run_nofsi_res_H.yml` → 1H, `run_nofsi_qe_N.yml` → 1N; Coherent mapper = struck at rest,
de Forest shift ≈0, no Fermi/spectral). ADoNIS uses res_xsec's OWN `exclusive_amps2_batch`.
- **RES free proton (ν p→μ⁻pπ⁺):** ACHILLES σ=2.954e-6. ADoNIS/ACH = **0.721**; Q²<0.2=**0.637**,
  Q²≥0.2=0.785. **SAME low-Q² deficit as the nuclear case** (0.726 / 0.65), shape-identical.
- **QE free neutron (ν n→μ⁻p):** ACHILLES σ=8.332e-6. ADoNIS/ACH = **0.991**, Q² ratio **flat**
  (0.990 / 0.991). Control passes → the free-nucleon test setup is sound.
- Figures: `paper_figures/free_res_WQ2.png`, `free_qe_WQ2.png`.

**CONCLUSION (big):** the RES deficit survives with the struck nucleon AT REST. So it is the
**ELEMENTARY DCC current** in `dcc_current.exclusive_amps2_batch` (Q²-dependent: low at low Q²),
**NOT** nuclear. RULED OUT now: de Forest shift, Fermi motion, spectral function, removal-energy
gate, 3-body measure, and everything QE-shared (QE free-nucleon is flat).

**Tension to resolve:** Ledger A4 ("amps2 1.004 per-event vs RESDUMP, flat in Q²") conflicts with
this. Resolution must be that A4 didn't test what the forward dσ/dQ² needs (e.g. `_NORM` absorbs
the average; or the per-bin check used shifted-Q² / a non-forward sample). **A4 is now SUSPECT.**

**Key asset:** ADoNIS has a SECOND DCC amplitude path — `primary/dcc/{channel,differential}.py`
(`amplitudes_*`, build_zmtx_batched) — that made the free-nucleon **total** σ(E_ν) match ACHILLES
(Fig 2). So one path (differential) is right, the other (`dcc_current`) is low at low Q².

### Iter 4 — free-nucleon localization + FITTED-NORMALISATION audit
- **Mono 1 GeV (no beam):** ADoNIS/ACH = 0.725 → deficit is NOT a beam artifact, real at fixed E.
- **Amplitude is faithful** (read amp_dcc_sl.f line-by-line: pion-pole 1/(Q²+m_π²), current
  conservation, vector/axial, angular sum, helicity↔spin, the unitary spin-rotation skip — ALL
  match dcc_current; per-event amps2 1.0003).
- **Isotropic sampler unbiased** (flat-Dalitz ground truth 0.3%, incl. amps2-weighted iso/flat
  0.9975 flat in Q²). Measure (psw) matches 1.0000.
- ⇒ every per-event factor + sampler validated, yet forward = 0.725. CONTRADICTION still open.

**FITTED-NORM AUDIT (user-driven):**
- QE: NO fit (first-principles Llewellyn-Smith). RES: had a FITTED `_NORM=3.7704e-5`.
- **FIXED:** `_NORM` now DERIVED first-principles = `2π/(|FResV|²(2 m_N)²)` = 3.792e-5
  (FResV=Vud·ee/(s_w√2·2); (2mN/ħc)² from currents_pi_dcc.f90:130; 2π from fac). Matches old
  fit to 0.6% → the RES absolute scale was correct, so **the fitted norm is NOT the 28%**.
  amps2 regression still passes. RES production path (res_xsec→dcc_current) is now fit-free.
- **Other fits found (Path B, redundant, Fig-2-only):** `SIGMA_UNIT_NB=2.569e-14` + bridge `c`
  in sigma_enu/channel.py/differential.py. Plan: retire Path B, route Fig 2 through Path A.

### Iter 5 — (next): the 28% is still unexplained with amplitude+measure+sampler all validated
and the scale now derived. Remaining: read the leptonic current + L·H contraction (the one
amplitude piece not yet line-checked) and/or compare the mono dσ/dQ² SHAPE (constant vs Q²-dep).

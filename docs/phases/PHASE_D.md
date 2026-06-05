# Phase D — FSI scaffold & differentiability proof (toy cascade)

Plan ref: `docs/REPRODUCTION_PLAN.md` Phase D — "the single most important early
checkpoint." Prove the differentiable machinery survives a **stochastic FSI transform**
on toy physics before investing in the real cascade (Phases E–H).

## Recovered (this branch)
The Phase-1 toy cascade — removed in commit `f60947a`, recoverable per the plan — is
restored under **`adonis/fsi/toy/`** (from `f60947a^:archive/diffpi_legacy/` +
`diffpi_pkg/kernel.py`):

- `kernel.py` — `score_weight`, parameter bijections, pytree Adam.
- `chain.py` — the compositional driver: `RAW`/`CHOICE`/`SHAPE` weight terms (kinds 1/2/3,
  Strategy §9) + `run_chain` (one global weight, geometry detached centrally).
- `component_{a,b,c_r0,c_r1,d,e}.py` — the toy physics rungs (hard vertex, MB scatter,
  transport, Oset-style absorption, propagating-Δ).
- `integrate_bc.py`, `integrate_full.py` — the integrated chains; `integrate_full` is the
  **5-parameter joint closure**: ν+N→ℓ+(Δ→Nπ) then the pion propagates through a uniform
  sphere (escape / scatter+energy-loss / absorb), histogrammed in (Q², |p_π|) + a CC0π
  (absorbed) bin. Recovers M_A, m_Δ, Γ_Δ, σ_scatter, σ_abs jointly from one observable.

## Verified
- Imports cleanly (`adonis.fsi.toy`), main package unaffected.
- `weighted_histogram` (differentiable) and `sampled_histogram` (hard reference) run.
- **Differentiability through the stochastic cascade**: `jax.grad` of a χ² loss wrt the
  full 5-tuple `(M_A, m_Δ, Γ_Δ, σ_sc, σ_abs)` is finite — gradient flows through the
  vertex reweighting (kind 1), the escape/scatter expected-value deposits (kind 1), and
  the absorb-vs-scatter score weight (kind 2). Gate: `tests/test_toy_cascade.py`.

## Remaining (D1 proper)
- ☐ Re-run the **full 5-param joint closure** end-to-end (synthesize at truth → fit →
  recover within MC error). The fit loop (Adam on the χ² of `weighted_histogram` vs a
  `sampled_histogram` pseudo-dataset) needs wiring from the config's truth/init knobs
  (`ConfigFull.{MA,mDelta,GammaDelta,sigma_scatter,sigma_abs}_*`); the forward-agreement
  check (weighted==sampled in expectation) and gradient-SNR study live here.
- ☐ **Port to a concrete `FSIModel`** acting on the real `EventRecord` (the plan's D1):
  wrap the cascade as `FSIModel.apply(params, EventRecord) -> EventRecord`, carrying the
  one global weight into the existing chain, and revive the joint closure on **real**
  produced final states (toy FSI physics; no oracle). This is the bridge from the toy
  package to the production chain — the actual Phase-D deliverable.

Toy physics ⇒ **no oracle gate** (per the plan); the gates are closure (differentiability)
and the joint-recovery closure.
</content>

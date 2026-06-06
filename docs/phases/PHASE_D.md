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

## D1 — concrete FSIModel on the real EventRecord — DONE
`adonis/fsi/cascade.py` (`ToyCascadeFSI`): the plan's D1 — the toy cascade ported to the
production contract `FSIModel.apply(params, EventRecord) -> EventRecord`. The produced pion
(`event.p_pi`) starts at the nucleus centre along its production direction and propagates
through a uniform sphere (R [fm]); per step it escapes, scatters (deflect + lose `mom_loss`
of |p|), or is absorbed (pion removed → `pid_pi=0`, a CC0π event). Two new differentiable
knobs live in `PhysicsParams`: `fsi_sigma_scatter`, `fsi_sigma_abs` [1/fm].

**Estimator (per event, kind-1 reweighting).** Like ACHILLES' cascade, each event gets a
*definite sampled* final state (so the EventRecord stays a per-event record, not an
expected-value histogram). Every stochastic decision is sampled against a **frozen
proposal** `q` (a detached copy of the params); θ enters only through a likelihood-ratio
weight `p_θ/q` folded into `event.w`. The sampled path is frozen as θ varies → the loss is
smooth and, at a fixed key, **autodiff == finite difference to machine precision**
(no REINFORCE jump variance). In production `proposal=None` defaults to `sg(params)`
(self-normalised: forward weight 1, gradient `d log p`), so it drops into the `Generator`
chain unchanged (`Generator(ch, fsi=ToyCascadeFSI())`).

**Gates** (`tests/test_cascade_fsi.py`):
- **closure** — `d/d(σ_sc, σ_abs)` of a final-state observable, autodiff vs FD with a frozen
  proposal: rel **~9e-7** (machine precision).
- **joint recovery** — synthesize the post-FSI `(Q², |p_π|)+CC0π` histogram on **real**
  DCC-produced events at a known `(M_A, σ_sc, σ_abs)`, then recover all three jointly by
  Adam on χ² (common random numbers + one frozen proposal ⇒ smooth, exact). Recovers
  `(1.150, 0.349, 0.218)` vs truth `(1.15, 0.35, 0.22)` from init `(0.95, 0.18, 0.40)`.
  Production (M_A, lepton side) and FSI (σ, hadron side) knobs are **simultaneously
  identifiable from one observable**, gradient flowing through the whole chain.

Figure: `figures/fsi_cascade_c12.png` (`scripts/make_fsi_figure.py`) — the produced vs
escaped pion spectrum, showing absorption depletion (~CC0π fraction) + low-|p| softening.

## D2 — Oset-shaped, momentum-dependent absorption (F→D) — DONE
`CascadeConfig(oset_shape=True)`: the cascade's σ_abs becomes **momentum-dependent**,
`σ_abs(T_π) = fsi_sigma_abs · absorption_rate_shape(T_π)`, where the shape is the transcribed
Oset absorption self-energy (Phase F, `adonis/fsi/mb/oset.py::absorption_rate_shape`)
normalised to 1 at the Δ (T_π=180 MeV). The rate is recomputed each step from the current
(degrading) pion |p|, so low-energy pions feel the s-wave tail and Δ-region pions are
absorbed preferentially. Stays pure kind-1 (the shape reads the detached T_π; θ enters only
via `fsi_sigma_abs`): **closure rel ~7e-7**, and the cascade absorption fraction tracks the
Oset shape, peaking at the Δ (`test_oset_shape_closure_and_delta_peak`; figure
`figures/oset_fsi_absorption_c12.png`). This is the resonant-FSI-rate item, wired from the
real Oset physics. (The *absolute* σ_abs(T_π) overlay vs ACHILLES is still F2-soft-blocked
on the `AbsCrossSection` source — see PHASE_F.md — but the **shape** drives the cascade here.)

## Remaining (optional)
- ☐ Re-run the toy `integrate_full` **5-param** joint closure end-to-end (adds m_Δ, Γ_Δ,
  which are amplitude-table values in the real DCC, not `PhysicsParams` knobs — so the
  real-chain joint closure fits the 3 it owns: M_A, σ_sc, σ_abs).
- ☐ Expose the Oset coefficient C_A2 as a `PhysicsParams` FSI knob to fit the absorption
  *shape* (not just its strength) jointly — the shape is already differentiable in C_A2.

Toy FSI physics ⇒ **no oracle gate** (per the plan); the gates are closure (differentiability)
and the joint-recovery closure.

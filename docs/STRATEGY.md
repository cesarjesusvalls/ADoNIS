# Differentiable Single-Pion Production — Implementation Strategy

A step-by-step plan to re-implement the ACHILLES single-pion-production + pion-propagation
chain (arXiv:2508.19213) as a **differentiable** Monte-Carlo simulator, so its underlying
physics parameters can be tuned against data by gradient descent.

---

## 0. The core idea, and why it fits

ACHILLES is a stochastic event generator: it samples an initial nucleon, a hard
electroweak vertex, and a cascade of final-state interactions, then bins the surviving
particles into histograms (differential cross sections, TKI variables) that are compared
to data. A histogram is *piecewise-constant* in the sampling parameters, so its naive
derivative is zero almost everywhere — you cannot backprop through `argmax`/binning/discrete
channel choices.

The `differentiable-sampling/` repo solves exactly this with four reusable ideas:

1. **Score-function weight** — `score_weight(p) = p / stop_gradient(p)`. Forward value `1`
   (the simulation is untouched, the histogram stays an honest count); backward value
   `d log p`. Multiply one of these into a trajectory's carried weight for every random
   decision whose probability depends on a parameter θ, deposit the product into a **hard**
   bin, and the bin contents become differentiable with the *exact* gradient
   `d/dθ E[counts]` — the bin index never moves.

2. **Continuous-shape weight** — for a continuous quantity sampled from a θ-dependent
   density (a scattering angle, an invariant mass), detach the sampled value and carry
   `score_weight(density(sample; θ))`. Gradient w.r.t. the shape parameter rides on the
   density evaluated at the (frozen) sample. (`ring_scattering.py`: the Mie `g`.)

3. **Expected-value / implicit-capture deposits** — instead of terminating a particle,
   deposit the fraction that "exits now" (`reach_prob`) into the exit bin and carry the
   remainder (`scatter_prob`) forward. A variance-reduced estimator of the same histogram;
   essential for the deep cascade.

4. **Detach the geometry** — `stop_gradient` on positions/directions of the random walk so
   gradients flow through the carried *weights*, not through the chaotic trajectory.

Plus the plumbing: parameter **bijections** (`softplus`→positive, `sigmoid`→unit,
`tanh`→signed-unit) to optimise constrained physics parameters in an unconstrained space,
and a pytree **Adam**.

`ring_scattering.py` is already a toy intranuclear cascade: a particle leaves the centre of
a disc, scatters through competing processes with mean free paths `L_c` and angular
distributions (Rayleigh/Mie), exits through the boundary, lands in an angular bin; a joint
fit recovers the mean free paths *and* the Mie asymmetry from one histogram. The real INC is
this with realistic cross sections, 3D geometry, absorption, and resonances.

---

## 1. Strategic decisions (recommended defaults)

| Decision | Recommendation | Rationale |
|---|---|---|
| **Framework** | **JAX** | Matches the example verbatim; `vmap`/`jit`/`grad`, `stop_gradient`, pytrees all used already. PyTorch is possible but throws away the working toolkit. |
| **Scope** | **Differentiable surrogate, not a port of the C++** | We re-implement in Python/JAX the *dominant, parameter-bearing* pieces of the physics, validated to reproduce ACHILLES forward outputs — not a line-by-line C++ port. The goal is a tunable model, not a second generator. |
| **DCC amplitudes** | **Treat the tabulated PWAs as the differentiable interface; tune *effective knobs* on them** | Re-solving the DCC coupled-channel integral equations differentiably is out of scope for v1. Instead expose a small set of physically-meaningful knobs that multiply/shift the tabulated τ^{L,±,I}(s) (overall normalisations, resonance peak position/width, axial-current strength). v2 can replace a knob with a differentiable sub-model if a parameter proves important. |
| **First parameters to target** | Cascade cross-section normalisations + Oset absorption coefficients (C_Q, C_A2, C_A3) + density-suppression α; then axial-vertex strength | These are (a) genuinely uncertain, (b) the paper itself flags as tunable (α, the Oset extrapolation beyond 350 MeV), and (c) most directly constrained by the pion-production/absorption data shown. |
| **Initial-state spectral function** | **Keep fixed in v1** | Nuclear structure input, not the tuning target; can be made differentiable later via the same score-weight trick on the (k,E) draw. |
| **Loss** | Binned χ² with experimental covariance (fall back to bin-wise MSE for closure tests) | Matches how the paper compares to data. |

The guiding principle throughout: **build each parameter-bearing physics block in isolation,
prove it reproduces the hard sampler in the forward pass and gives a correct gradient
(finite-difference checked), then compose.**

---

## 2. Phase 0 — Infrastructure

- Vendor `differentiable.py` (`score_weight`, bijections, Adam) as the project's kernel.
- Add a **gradient-validation harness** used by *every* component test:
  - `sampled_histogram` (hard, one count per event) == `weighted_histogram` (differentiable)
    in the forward pass, to within MC error (this is the ring-demo discipline).
  - `jax.value_and_grad` vs central finite differences on each scalar parameter.
  - A **closure-test driver**: synthesise data at known θ*, re-init θ, fit, assert recovery.
- Physics constants/kinematics utilities: Källén λ, Lorentz boosts, CM↔lab, Legendre
  `P_L`/`P_L'`, Wigner small-`d`, Clebsch-Gordan/isospin tables, Breit-Wigner spectral
  function `A_Δ`. All pure-functional and differentiable.
- A thin **tabulated-amplitude loader**: read the DCC τ^{L,±,I}(s) tables, expose them as
  JAX arrays with a differentiable interpolation in `s` (`jnp.interp`/spline) and the knob
  parameters layered on top.

**Exit criterion:** the harness can run a trivial closure test (re-fit a 1-parameter toy)
end to end and the finite-difference check passes.

---

## 3. Phase 1 — Standalone differentiable components

Each component is a separate module + test. Each test = forward-agreement check +
finite-difference gradient check + a closure test (recover synthetic-truth parameters).

### A. Hard vertex: single-pion production kinematics
- **Physics:** 1N1π final state. Sample CM scattering angle of the πN pair from the DCC
  angular distribution `dσ/dΩ` (Eq. for `sigma_dif2`: Legendre `P_L`/`P_L'` weighted by
  `τ^{L,±}`), φ uniform; build (p_π, p_N) via 2→2 phase space; weight by |M|²(θ).
- **Differentiable handles:** angle sampled by inverse-CDF (detached) → carry
  `score_weight(density(cosθ; θ))`; the |M|² normalisation and any reparameterisable
  kinematic Jacobian carry gradient directly.
- **Knobs (θ):** axial-current strength / axial-mass-like parameter, resonance peak
  position & width, overall normalisation.
- **Test A:** synthesise events at θ*, histogram in (W, cosθ*_π, p_π); forget θ; recover.
  Cross-check the *forward* angular distribution against the paper's Fig. of DCC angular
  cross sections.

### B. Single meson-baryon scattering vertex (one INC interaction)
- **Physics:** πN → {πN, ηN, KΛ, KΣ}. Total cross section `σ(W)` from squared PWAs;
  channel chosen with probability ∝ partial σ_if; CM angle from `dσ_if/dΩ`; φ uniform.
- **Differentiable handles:** channel choice (categorical) → `score_weight(channel_prob)`;
  angle (detached) → `score_weight(density(cosθ))`. Direct analog of ring-demo's
  `_channel` + `_shape_weight`, but with realistic PWA angular distributions.
- **Knobs:** per-channel normalisation; a multiplicative/positional knob on a chosen
  partial wave (e.g. the Δ-dominated P33); resonance position.
- **Test B:** fixed incoming πN at fixed s; histogram channel fractions + cosθ; recover a
  knob. Forward cross-check against the DCC total-cross-section figure.

### C. Cascade transport (FSI propagation) — *the centrepiece, closest to the demo*
- **Physics:** propagate particles through a nucleus in steps; interaction probability from
  impact parameter + total cross section (Gaussian/cylinder model) ≈ mean-free-path
  exponential; on interaction, pick channel (Component B) and deflect.
- **Differentiable handles:** *exactly* `ring_scattering.weighted_histogram` generalised to
  3D and real cross sections — expected-value deposits of the "escape now" fraction,
  `score_weight(channel_prob)` for the rate gradient, `score_weight(density)` for the angle,
  geometry detached.
- **Knobs:** total/partial cross-section normalisations (≡ mean free paths).
- **Test C:** toy nucleus (uniform sphere or sampled nucleon positions); inject a pion;
  histogram exit channel (transmitted / charge-exchanged) and exit kinematics; recover
  cross-section normalisations. This is the single most important closure test — it proves
  the whole transport loop is differentiable and low-variance.

### D. Oset pion absorption
- **Physics:** absorption rate from Im Σ_Δ(T_π) = −[C_Q (ρ/ρ₀)^α + C_A2 (ρ/ρ₀)^β +
  C_A3 (ρ/ρ₀)^γ], local-density-dependent; per-step absorption probability.
- **Differentiable handles:** the absorption probability is an *analytic, smooth* function of
  the coefficients → reparameterisable directly; the absorb/survive decision per step is a
  Bernoulli → `score_weight`, or fold into the expected-value deposit as an "absorbed"
  channel weight.
- **Knobs (θ):** C_Q, C_A2, C_A3 (and optionally the exponents α,β,γ).
- **Test D:** pion through a slab/sphere with a density profile; histogram survival fraction
  vs T_π and absorption depth; recover the C coefficients. Stresses the >350 MeV
  extrapolation the paper flags.

### E. Propagating-Δ model (GiBUU mode)
- **Physics:** πN→Δ; Δ propagation; probabilistic decay `P=1−e^{−δt/τ}`, τ=γħc/Γ; two-body
  decay channel by branching ratios; ΔN→NN absorption; NN→NΔ production with the one-pion-
  exchange matrix elements (direct + crossed + interference), Δ spectral function `A_Δ`,
  form factor `F(t)=(Λ²−m_π²)/(Λ²−t)`, `z(t,μ_Δ)`, optional density suppression
  `exp(−α ρ/ρ₀)`.
- **Differentiable handles:** decay-vs-interact Bernoulli per step → `score_weight`; decay
  channel categorical → `score_weight(branching)`; off-shell mass μ_Δ sampled from `A_Δ`
  (detached) → `score_weight(A_Δ)`; the matrix-element weights are analytic in (Λ, κ, Γ, α).
- **Knobs:** Λ (0.63 GeV), κ (0.2 GeV), Γ (112 MeV), density-suppression α, couplings.
- **Test E:** a Δ with given momentum in a uniform medium; histogram decay time and
  NN-vs-πN products; recover Γ and Λ. Forward cross-check against the NN→NNπ cross-section
  figure.

### F. Initial-state spectral function (deferred)
- Sample (k, E) from tabulated S(k,E). Keep fixed in v1; when needed, the (k,E) draw gets a
  `score_weight(S)` and any nuclear knob becomes tunable.

### G. Observable & loss layer
- Build final histograms incl. TKI variables (δp_T, δα_T, δφ_T, δp_TT) from the carried
  weights and *detached* kinematics; flux-fold; χ² against experimental data + covariance.
- Trivially differentiable (arithmetic on detached kinematics × differentiable weights).
- **Test G:** feed known weighted events, confirm the loss and its gradient w.r.t. an
  upstream knob match finite differences.

---

## 4. Phase 2 — Integration into the full chain

Compose A → (C with B, D) or (C with E) → G, carrying **one global weight per event** that
accumulates every `score_weight` factor along the way, with geometry detached throughout.

- **Staged composition:** A→G first (hard vertex only, no FSI); then add transport C; then
  absorption D; then swap in the propagating-Δ mode E. Re-run the closure test after each
  addition so any variance blow-up or gradient bug is localised.
- **End-to-end closure test:** synthesise a full pseudo-dataset at known θ* (a realistic
  observable like dσ/dp_π or a TKI distribution), re-init θ, fit *all* targeted parameters
  jointly, assert recovery within MC/statistical error. This is the headline validation that
  the whole differentiable chain works.
- **Forward fidelity gate:** before any tuning, the composed forward simulator must reproduce
  the corresponding ACHILLES histograms / paper figures at the published parameters (a
  non-gradient regression test). Differentiability is worthless if the forward model is wrong.

---

## 5. Phase 3 — Tuning to real data

- Start with **one observable, one or two parameters** with a clear physical handle (e.g.
  Oset C-coefficients against a pion-absorption measurement; α against the Ar TKI data the
  paper discusses) before joint multi-observable fits.
- Joint fit across e4ν / T2K / MINERvA / MicroBooNE with per-experiment normalisation and the
  proper covariance; add priors/regularisation on physically-constrained parameters.
- Propagate uncertainties: the Hessian (or Laplace/ensemble) at the optimum gives parameter
  errors and correlations — a direct benefit of having gradients.

---

## 6. Cross-cutting concerns

- **Variance control (the main practical risk).** Deep cascades make pure REINFORCE noisy.
  Mitigations, all demonstrated or implied by the demo: expected-value deposits (don't
  terminate particles), detach geometry, `n_model > n_data`, **common random numbers** /
  shared PRNG key between data and model, and per-step weight clipping/`nan_to_num`
  (the demo already `nan_to_num`s grads). Consider control variates / baselines if needed.
- **Forward correctness vs gradient correctness are separate gates.** Every component checks
  *both*: hard==weighted forward agreement, and grad==finite-difference.
- **Identifiability.** Many knobs + limited data ⇒ degeneracies. Map which observable
  constrains which parameter; start low-dimensional; watch the Hessian conditioning.
- **Tabulated-amplitude gradients.** Gradients flow into table *knobs*, not raw table cells,
  unless we deliberately make spline coefficients the parameters. Decide per parameter.
- **Hard kinematic boundaries** (thresholds, energy-momentum δ-functions). Where a boundary
  itself moves with θ, the score-function estimator still applies (the probability mass
  redistributes); reparameterise smooth phase-space maps, score-weight discrete/boundary
  choices.

---

## 7. Risks & mitigations (summary)

| Risk | Mitigation |
|---|---|
| Cascade-depth variance swamps the gradient | Expected-value deposits + geometry detach + CRN + more model events; validate on Test C before integrating |
| Forward model diverges from ACHILLES | Forward regression gate against paper figures/C++ output at every stage |
| DCC amplitudes not differentiable in fundamental params | v1 tunes effective knobs on tabulated PWAs; escalate to sub-models only where a parameter matters |
| Parameter degeneracies | Low-dim first; Hessian/identifiability analysis; priors |
| Threshold/boundary non-smoothness | Score-function handles probability redistribution; reparameterise only smooth maps |

---

## 8. Recommended first build

**Component C (cascade transport) on a toy nucleus**, because it is the closest analog to the
working `ring_scattering.py`, exercises every technique (rate gradient + angle gradient +
expected-value deposit + geometry detach) in one loop, and its closure test de-risks the
hardest part (the deep stochastic transport) before any DCC/phase-space machinery is built.
Then B (real angular distribution inside the same loop), then D (absorption), then A (hard
vertex), then E, then integrate.

---
---

# Part II — Refined technical plan (toy-first path)

This part makes Part I concrete for the chosen approach: **toy-first fidelity** (prove the
differentiable machinery on simplified physics, then raise realism rung by rung). It pins
down (9) the exact weight algebra of the full chain, (10) the reparameterise-vs-score-weight
decision rule, (11) a compositional API that bakes in "one global weight, detached geometry",
(12) the toy→realistic ladder with graduation criteria, (13) the variance analysis and how we
measure it, and (14) how parameter knobs attach to tabulated DCC amplitudes.

## 9. The weight algebra of the whole chain

Every event is a trajectory carrying a scalar **weight** `w` (initialised to 1) and a bundle
of **detached kinematics**. The differentiable histogram is the sum, over events and over
deposits, of `w` dropped into a *hard* bin. Formally, for bin `b`:

```
H_b(θ) = E[ Σ_deposits  w_deposit · 1(bin = b) ]      (forward: honest count; backward: exact dH_b/dθ)
```

`w` accumulates three *kinds* of factor, and the entire skill is knowing which kind each
physics operation is:

1. **Smooth probability factors carried as themselves** (reparameterised; gradient flows
   directly). The expected-value deposits live here. At a transport step with total rate
   `Λ(θ)=Σ_c 1/L_c(θ)` over a straight segment `d`:
   - deposit `w · reach_prob`, with `reach_prob = exp(−d·Λ)` → the "escape now" fraction;
   - carry `w ← w · scatter_prob`, with `scatter_prob = 1 − exp(−d·Λ)`.
   Both factors are differentiable in θ and contribute the *overall-rate* gradient with **zero
   extra variance** (nothing is sampled). Absorption (Component D) enters the same way as an
   "absorbed-now" deposit with `absorb_prob(θ)`.

2. **Categorical choices carried by a score weight** (genuinely sampled, discrete). Choosing
   channel `c` with probability `π_c(θ)`: `w ← w · score_weight(π_c)`. This supplies the
   *relative-split* gradient `d log π_c`. Examples: meson-baryon channel, isospin channel,
   decay channel, absorb-vs-survive when sampled rather than deposited.

3. **Continuous shapes carried by a density score weight** (sampled by detached inverse-CDF).
   A CM angle `cosθ ~ p(·;θ)`: sample it, **detach it**, and `w ← w · score_weight(p(cosθ;θ))`.
   Gradient `d log p` rides on the density evaluated at the frozen sample. Examples: the DCC
   angular distribution, the Δ off-shell mass drawn from the spectral function `A_Δ`.

**Geometry is detached everywhere** (`stop_gradient` on positions/directions/boosts): the
gradient must ride on `w`, never compound through the chaotic random walk. The carried
kinematics are *values*, not *gradient paths*.

A worked composition (transport step, mirroring `ring_scattering.weighted_histogram`):
```
reach  = exp(-d*Λ);  scatter = 1 - reach                 # kind 1
hist   = hist.at[exit_bin].add(w * reach)                # deposit (kind 1 gradient)
c      = sample_channel(π(θ));  φ = sample_angle(p_c(θ)) # detached samples
w      = w * scatter * score_weight(π_c) * score_weight(p_c(φ;θ))   # kinds 1,2,3
pos, dir = stop_gradient(advance(pos, dir, step, φ))     # geometry detached
```

## 10. The reparameterise-vs-score-weight decision rule

For each stochastic operation, prefer the lowest-variance valid option:

| Situation | Treatment | Why |
|---|---|---|
| Probability of a *binary fate* you can afford to split (escape/scatter, absorb/survive) | **Expected-value deposit** (kind 1) | deterministic, zero sampling variance |
| Smooth map from randomness to a momentum, θ enters only through the weight | **Detach kinematics, score-weight the |M|²/density** | matches the demo; robust |
| Smooth map where θ enters the *kinematics* and you need that sensitivity | **Reparameterise** (keep momenta attached, differentiate the map) | exact pathwise gradient; use sparingly |
| High-multiplicity discrete choice (which of N channels) | **Sample + score_weight(π_c)** (kind 2) | can't afford to enumerate; REINFORCE |
| Continuous angle/mass from a θ-shaped density | **Sample + detach + score_weight(p)** (kind 3) | REINFORCE on the shape only |

**v1 rule of thumb:** expected-value-deposit every binary fate; sample-and-score-weight every
high-multiplicity categorical and every continuous shape; reparameterise phase-space maps only
later, and only for parameters whose sensitivity genuinely lives in the kinematics rather than
in `|M|²`. The Höche/Byckling phase-space building blocks (Källén-λ Jacobians) are smooth and
*can* be reparameterised, but in v1 the matrix element is the θ-carrier, so detach-and-reweight
is the default.

## 11. Compositional API (makes the invariants structural)

Each physics block is a pure step with a uniform signature, so the driver — not the physics
code — owns weight accumulation and geometry detachment. This prevents the most likely class
of bug (a forgotten `stop_gradient` or a double-counted score weight).

```python
# A step consumes detached state + params + randomness and returns:
#   new_state   : detached kinematics (positions, directions, particle list)
#   logw_terms  : list of probabilities to fold into the weight, tagged by kind
#   deposits    : list of (bin, fraction_of_weight) expected-value deposits
Step = Callable[[State, Params, Key], tuple[State, list[WeightTerm], list[Deposit]]]

def run_chain(steps, init_state, params, key, n_bins):
    state, w, hist = init_state, ones(n), zeros(n_bins)
    for step in steps:
        key, sub = split(key)
        state, terms, deposits = step(state, params, sub)
        for (b, frac) in deposits:
            hist = hist.at[b].add(w * frac)          # kind-1 gradient flows here
        for t in terms:
            w = w * fold(t)                          # score_weight or raw factor by kind
        state = tree_map(stop_gradient, state)       # geometry detach, enforced centrally
    return hist
```

`WeightTerm` kinds: `RAW(p)` (carried as `p`, kind 1), `CHOICE(π, c)` (→ `score_weight(π_c)`,
kind 2), `SHAPE(p, x)` (→ `score_weight(p(x))`, kind 3). The same driver runs the **hard
reference sampler** (replace deposits with a sampled terminal bin and set all weight terms to
1) so forward-agreement is testable by construction.

## 12. Toy→realistic ladder (per component) with graduation criteria

Each rung must pass the same three gates before promotion: **(F)** forward agreement
hard==weighted within MC error; **(G)** autodiff grad == central finite difference (rel. err
below a set tol, e.g. 1e-2 at fixed key); **(C)** closure — recover synthetic truth within
statistical error. Toy-first means we climb rung 0→3 only as far as each fit needs.

- **Component C (transport)** — *first build*
  - R0: 1D slab, single channel, constant σ → recover one mean free path. (≈ the demo.)
  - R1: 3D uniform-density sphere, 2–3 competing channels, schematic angular shapes →
    recover channel norms + one shape param jointly.
  - R2: sampled (optionally correlated) nucleon positions + the actual ACHILLES
    impact-parameter Gaussian/cylinder interaction-probability model → recover σ norms.
  - R3: real DCC total + angular cross sections plugged in (forward gate vs paper Fig.
    DCC_total_xsecs / DCC_angular_xsecs).
- **Component B (MB vertex)** — R0 fixed-`s` 2-channel toy → R3 full {πN,ηN,KΛ,KΣ} PWAs.
- **Component D (Oset absorption)** — R0 constant-density slab → R3 local-density profile,
  recover C_Q/C_A2/C_A3 and stress the >350 MeV extrapolation.
- **Component A (hard vertex)** — R0 schematic resonance Breit-Wigner angular shape → R3 DCC
  τ^{L,±} with axial-strength knob; forward gate vs published angular distribution.
- **Component E (propagating Δ)** — R0 single Δ, decay-vs-survive Bernoulli, recover Γ → R3
  full NN↔NΔ one-pion-exchange matrix elements, recover Λ, κ, α.

## 13. Variance analysis (the make-or-break property) and how we measure it

Pure REINFORCE over a depth-`K` trajectory carries `Σ_k` score terms; the gradient-estimator
variance grows with `K`. The three controls, in order of impact:

1. **Expected-value deposits** shrink the *effective* depth: most of each bin's mass is
   deposited deterministically (kind 1) within a step or two, so few score terms (kind 2/3)
   actually reach any given bin. This is the single biggest lever.
2. **Geometry detach** removes a second, independent variance source (gradient flowing through
   the chaotic position recursion).
3. **Common random numbers** (shared PRNG key between data and model) + **`n_model > n_data`**
   turn the loss-gradient into a low-noise systematic difference (the demo reuses the key and
   `nan_to_num`s the grads).

**Deliverable:** in the Component-C test, measure gradient **SNR = |E[g]| / std[g]** as a
function of cascade depth `K` and `n_model`, with and without each control, and record the
budget needed for SNR > a threshold (say 5). If SNR is inadequate at realistic `K`, escalate
to: per-step weight clipping, a learned/constant **baseline** (control variate) subtracted
before the score weight, or antithetic randomness. This measurement gates whether the
integrated chain (Phase 2) is viable, so it is done *early*, on the toy.

## 14. Attaching tunable knobs to tabulated DCC amplitudes

The PWAs `τ^{L,±,I}(s)` arrive as tables. The differentiable interface layers knobs on top of
a differentiable interpolation `τ̂(s) = interp(s; table)`:

- **Normalisation knob:** `τ → (1 + a_{L,I}) · τ` — a per-(L,I) multiplicative strength.
- **Resonance position/width knob:** for the resonant partial wave, multiply by a
  differentiable Breit-Wigner ratio `BW(s; m+δm, Γ+δΓ) / BW(s; m, Γ)` so the dominant pole can
  shift/broaden while non-resonant background stays fixed.
- **Axial-strength knob:** an overall factor on the axial-current contribution to the hard
  vertex (the PCAC-constrained piece), the closest differentiable proxy for an axial-mass tune.

Gradients flow into these *knobs*, not into raw table cells (unless we deliberately make spline
coefficients the parameters — an option for a flexible, less physical fit). Each knob has a
natural prior centred at its published value, used as regularisation in Phase 3.

## 15. Sequenced milestones (toy-first)

1. Phase 0 harness + Component C **R0** closure (1 param). — *proves the loop is differentiable.*
2. Component C **R1** + the §13 variance/SNR study. — *proves it scales to a real cascade.*
3. Component B **R0→R1** folded into C. — *real channel + angle gradients.*
4. Component D **R0→R1**. — *absorption coefficients recoverable.*
5. Component A **R0**, then **G** loss layer; integrate A→C→G (no FSI realism yet) and run the
   **end-to-end closure test** on a toy observable. — *the headline "it works" result.*
6. Raise rungs (C/B/D → R2/R3, add E) only where a target fit needs the realism; add the
   forward-fidelity gates against the paper figures.
7. Phase 3 real-data fit, lowest-dimensional first.

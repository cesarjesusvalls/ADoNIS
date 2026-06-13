# Cascade: declared approximations vs ACHILLES — what diverges, the assumptions, and why

This documents where the ADoNIS FSI cascade deliberately departs from a verbatim transcription of
ACHILLES `Cascade.cc`, under what assumptions each departure is expected to hold, and what motivates
it. These are the **non-faithful** parts of the chain and therefore the prime suspects for any
ACHILLES-vs-ADoNIS residual that is FSI-coupled (e.g. the CC1π high-W/pₚᵢ tail).

## Overarching motivation: differentiability forces a fixed-size surrogate

ACHILLES's cascade (`src/Achilles/Cascade.cc`, Evolve/BaseAlgorithm) is a **stochastic, variable-
particle, recursive event loop**: ONE nucleus in which the pion and *every* kicked nucleon co-evolve
against a **shared** background (shared consumed-nucleon mask, shared positions), each collision
spawning new propagating particles that themselves re-cascade until everything exits or is absorbed.

That structure is fundamentally **non-differentiable** and cannot be expressed as a fixed-size,
vectorized JAX computation. ADoNIS's premise is a *differentiable* generator, so the cascade is
re-cast as the **kind-1 blueprint**: a fixed-size, θ-independent "walk" run at nominal, with physics
knobs entering only through differentiable per-event reweights (`pion_branch_reweight`,
`nucleon_scat_reweight`). Making the walk fixed-size and differentiable is what forces the
approximations below — they replace ACHILLES's variable-particle recursion with fixed-size surrogates.

## The declared approximations

### 1. Factorized independent cascades vs ONE shared nucleus
`scripts/cc1pi_fig_tki.py::res_C` runs the **pion** cascade (`DiscreteCascadeFSI`) on one sampled
nuclear background, then the **nucleon** cascade (`DiscreteNucleonFSI`) and the knockout re-cascade on
a **freshly resampled** background (`sample_nucleons(fold_in(kn,1), …)` — different nucleons,
positions, and an independent consumed set). ACHILLES uses ONE nucleus with the *same* background and
a *shared* consumed/positions state for pion and nucleons.
- **Assumption:** correlations between the pion's trajectory and the recoil nucleon's trajectory
  (traversing the same nucleons, depleting the same local Fermi sea) are second-order.
- **Why not faithful:** the shared state couples all particles into one variable-length co-evolution
  — the non-vectorizable, non-differentiable loop. Independent fixed-size walks are differentiable.

### 2. Leading-proton tracking vs the full hadronic final state
Per event ADoNIS tracks only the **single highest-momentum proton candidate** among {RES nucleon,
leading pion-scatter recoil, that recoil's one knockout}; it does not build the full multi-nucleon
final state.
- **Assumption:** the leading-proton observables (p_p, p_N, δp_TT) are dominated by the
  highest-momentum proton; a secondary proton rarely becomes the leading one.
- **Why not faithful:** a full final state has a variable number of nucleons per event (not fixed-size).

### 3. Bounded knockout recursion (~2 generations) vs full recursion
ADoNIS re-cascades the leading pion-scatter recoil **once** (its own knockout is a further candidate),
then stops; ACHILLES recurses every kicked nucleon indefinitely. (`cascade_discrete.py` tracks the
leading PROTON recoil per event; neutron recoils' secondary knockouts are explicitly neglected.)
- **Assumption:** 3rd-and-higher-generation knockouts are too low-momentum to ever be the leading
  in-window (450–1200 MeV/c) proton.
- **Why not faithful:** unbounded recursion is variable-depth (not fixed-size).

### 4. η/K conversion products not propagated
On πN→ηN/KΛ/KΣ (`cascade_discrete.py:362-365`) the pion is removed but the η/K + baryon products are
NOT cascaded.
- **Assumption:** both CC0π and CC1π signal definitions VETO these events anyway (a strange hadron /
  no surviving π⁺), so the products' fate is irrelevant to the signal; the product nucleons sit far
  above k_F near threshold (would escape regardless of Pauli).
- **Why not faithful:** would add particle species/channels with no impact on the pion-counting signals.

### (Not an approximation) fast_xsec
`fast_xsec` evaluates cross sections only for the K nearest in-slab nucleons. Declared **bit-exact**
(rarely >K nucleons in a slab), so a speed optimization, not a physics departure.

## Validation status: NO fitting to ACHILLES

The reweight knobs **`sabs`** (scale oset absorption) and **`sscat`** (scale MB scatter) are the
**fit-to-DATA** parameters of the separate T2K CC0π tune; for the **ADoNIS-vs-ACHILLES** comparison
they sit at **nominal = 1.0**. The cascade's agreement with the ACHILLES π⁺-¹²C transparency oracle
(~1–5%) was achieved by **first-principles physics corrections**, not by tuning — explicitly "no
tuned constants" (`docs/cascade_transport_residual.md`): the escape-radius rule (`Nucleus.cc`), the
PionAbsorption isospin partition 5/6·oset (`PionAbsorption.cc`), and the charge-resolved scatter σ
(`GetCchannel`). The other `DiscreteCascadeConfig` entries (`prob`, `pauli`, `nn_inelastic`) are
model choices set to match the ACHILLES run-card, not fitted numbers.

## Where an FSI-coupled residual would live

Items **1–3** are precisely the path by which an n→nπ⁺ event (no primary proton) acquires a signal
proton via a knockout chain. ACHILLES's shared-state, fully-recursive cascade models that exactly;
ADoNIS approximates it with a factorized, leading-only, depth-bounded surrogate validated against
*transparency* but not W-by-W in the second-resonance region. A separate found-and-fixed item — the
MB pion-scatter **angular table truncated at W=1700** (now extended to the full ANL/ACHILLES grid,
2200; `cascade_mb.py`) — was a genuine unfaithful truncation but affects only W_j≳1950 (rare). The
factorization / leading-proton / recursion-depth approximations remain the open candidates, and they
are hard to remove without sacrificing differentiability — the central tension of the project. (A
shared-state "event cascade" was attempted in logbook #9–#15 and reverted: non-differentiable and
itself not a faithful `Cascade.cc` port.)

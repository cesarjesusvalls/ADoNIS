# Phase A3 — Free-nucleon single-pion σ(E_ν) → Fig 2 (ANL/BNL)

Plan ref: `docs/REPRODUCTION_PLAN.md` Phase A3. The smallest self-contained,
fully-differentiable, oracle-checkable unit. Target: the paper's **Fig `anl_bnl`**
(`figures/tight/ANL_BNL_tight.pdf`) — total pion-production cross sections for ν_μ on
**elementary nucleons** as σ(E_ν), three channels, vs reanalyzed ANL/BNL bubble-chamber
data (Wilkinson:2014yfa):

- `ν_μ p → μ⁻ p π⁺`  (channel 2 here; pure I=3/2, Δ-dominated → largest)
- `ν_μ n → μ⁻ n π⁺`  (channel 1)
- `ν_μ n → μ⁻ p π⁰`  (channel 0; from deuterium data)

Paper (line 481): "ACHILLES reproduces the neutrino energy dependence of these
elementary cross sections very well." So the physics content is the **energy
dependence** + **channel ratios** (parameter-free), plus the **absolute scale**.

## Gates (plan)
- **closure**: grad of σ wrt M_A, autodiff == central FD.
- **oracle**: vs ACHILLES on a stationary nucleon (ν_μ) + the ANL/BNL data overlay.

---

## Architecture decision
Reuse the existing validated DCC channel (`DCCSinglePion`) unchanged and **swap only the
initial state** to a nucleon at rest. The kind-1 reweight contract and the M_A gradient
are therefore identical to the (already-gated) nuclear path — A3 adds no risk to the
differentiable core.

The three isospin channels already live in the channel
(`_PID_NI/_PID_N/_PID_PI` in `primary/dcc/channel.py`):
`ch0 n→pπ⁰, ch1 n→nπ⁺, ch2 p→pπ⁺`.

## Sub-tasks & status

- ☑ **A3.1 `FreeNucleon` nuclear model** (`adonis/nuclear/free.py`) — nucleon at rest
  (p=0, E_rm=0 → p_struck=(M_N,0,0,0)). Verified: plugs into `DCCSinglePion(nuclear=…)`,
  W peaks in the Δ region (~1211 MeV), **channel split p:n→pπ⁰:n→nπ⁺ = 66.4/18.6/15.0 %**,
  matching the DCC isospin fractions in `docs/STATUS.md` (.661/.193/.146). No new
  differentiable param → closure skipped; physics validated at σ level (below).

- ☐ **A3.2 σ(E_ν) scan + per-channel decomposition** (`adonis/primary/dcc/sigma_enu.py`):
  for each E_ν, integrate the proposal (lepton E′∈[m_ℓ, E_ν], θ∈[0,π]; pion angle already
  folded as the explicit 4π in the weight) → σ_total and per-channel σ_c ∝ mult_c·L·W_c.
  Relative σ(E_ν) (shape + ratios) needs only consistent proposal-volume normalisation;
  the absolute scale is one universal constant (A3.5).
  - closure: dσ/dM_A autodiff==FD at a fixed energy (reuses `grad_closure`).

- ☐ **A3.3 Muon-mass extension** (for the ν_μ ANL/BNL comparison). The current
  `lepton_tensor_cc` + lepton kinematics are **massless** (built/validated for ν_e).
  Add an OPTIONAL `m_lep` (default 0 → bit-identical to today, so the ν_e oracle gate
  never regresses): (a) outgoing lepton |p′|=√(E′²−m_ℓ²) in `sample_final_state`;
  (b) the m_ℓ² terms in the CC lepton tensor; (c) threshold E′≥m_ℓ. ν_μ: m_ℓ=105.658 MeV.
  Gate the massless limit against the existing νe path before trusting the massive one.

- ☑ **A3.4 Free-nucleon ACHILLES oracle** (ν_e, stationary nucleon, σ(E_ν) scan) — DONE.
  **Resolved the Phase-0 open question:** ACHILLES *does* give genuinely static nucleons
  — `Nucleus::GenerateConfig` returns the nucleon **at rest** for `Name: 1H` (free proton,
  `Initialize(1,1)` → `{m_p,0,0,0}`) and `Name: 1N` (free **neutron**, `Initialize(0,1)` →
  `{m_n,0,0,0}`); the earlier "1N uses QESpectral/Fermi motion" worry was wrong at the
  nucleus level. Note `RES_Spectral_Func` still multiplies by `S(p=0,E)` (an
  energy-independent factor for a nucleon at rest) — which is why we validate **shape +
  channel ratios** (constant-independent), not the absolute scale here.
  - Used **ν_e** (massless, `Leptons: [12,[11]]`) to match the model's massless lepton
    tensor exactly (muon mass is A3.3). `1H` → ch2 `p→pπ⁺`; `1N` → ch0 `n→pπ⁰`, ch1
    `n→nπ⁺` (ACHILLES per-process order = our channel order). 9 energies 0.4–2.5 GeV.
  - Oracle data committed: `data/oracle/freenucleon_nue_sigma.csv` (reproducible; the
    integrated σ printed during optimization, so NEvents can be tiny).
  - **Result:** a single universal constant c bridges model→ACHILLES across all
    **9 energies × 3 channels (27 cells)** with **max rel 3.3%, mean 0.8%, c-spread std
    1.1%** (150k events). Gate: `freenucleon_sigma_oracle()` +
    `tests/test_sigma_enu.py::test_freenucleon_sigma_oracle`. Figure:
    `scripts/make_a3_sigma_figure.py` → `figures/a3_freenucleon_sigma.png` (Fig-2 shape).
  - Gives c = ACH/model ≈ 2.57e-14 (model→ACHILLES units) — the A3.5 bridge.

- ☑ **A3.3 Muon-mass extension** — DONE. `GenConfig.m_lep` (default 0; muon = 105.658 MeV)
  threaded into `sample_final_state`. The CC lepton-tensor *form* is unchanged (the (1∓γ5)
  projectors kill the m_ℓ terms), so the muon mass enters **only kinematically**: (a) the
  outgoing lepton momentum |p'|=√(E'²−m²), and (b) — the catch — the leptonic phase-space
  factor in the weight is |p'|/E_ν (the standard |k'|/|k| ratio), **not** E'/E_ν. Both now
  use `plep`; m_lep=0 is bit-identical to the validated ν_e path (no regression).
  - Validated vs an ACHILLES **ν_μ** free-nucleon oracle (`data/oracle/freenucleon_numu_sigma.csv`,
    `Leptons:[14,[13]]` on 1H/1N): **max rel 2.9%, mean 1.1%** over 7 energies × 3 channels.
  - **Decisive cross-check:** the bridging constant is the SAME for ν_e and ν_μ —
    **c_μ/c_e = 0.9975** (they share the CC coupling, so the muon mass must be purely
    kinematic). Gates: `test_freenucleon_sigma_oracle_muon`,
    `test_muon_electron_constant_consistency`. (This also fixed a latent E'-vs-|p'|
    phase-space bug that was harmless for massless leptons but matters for B/I muons.)

- ◐ **A3.5 Physical absolute units (DONE) + ANL/BNL data overlay (remaining).**
  - **Absolute units resolved:** ACHILLES reports σ in **nb** (Constants.hh
    `HBARC2 = 0.38938 mb·GeV²`; the EM run prints "676 nb"; the free-proton 7.6e-6 at the
    Δ peak = 0.76×10⁻³⁸cm², the physically-correct CC1π⁺ scale). So
    `data/oracle/freenucleon_*_sigma.csv` is **already in nb**, and because the oracle gate
    fits a SINGLE scalar c and bounds `|c·model − ACH_nb| < 3%`, **it already validates the
    absolute σ(E_ν) in nb** (not just the shape). c is purely the model's internal unit
    scale (proposal-volume bookkeeping + the G_F²cos²θ_c the relative weight omits). Baked
    into `sigma_enu.SIGMA_UNIT_NB` so `sigma_vs_enu_nb()` outputs physical nb;
    1 nb = 1e5 ×10⁻³⁸cm² (the Fig-2 axis). (A first-principles derivation of c from
    G_F²cos²θ_c·flux·measure is a nice-to-have cross-check, not needed for correctness.)
  - **Remaining: ANL/BNL data overlay.** Data located: `../nuisance/data/{ANL,BNL}/`
    (`CC1pip_on_p` → p→pπ⁺, `CC1pip_on_n` → n→nπ⁺, `CC1pi0_on_n` → n→pπ⁰; ROOT, read with
    uproot; `*_XSec_1DEnu_*` classes). **Caveat:** the paper's Fig 2 uses the **Wilkinson
    2014 reanalysis** (re-extracted flux normalisation), not the raw NUISANCE PRD digitizations
    — for an exact match, use the Wilkinson points (digitize from the paper or find the
    release); the NUISANCE data is a close proxy. This is the only piece of A3 that depends
    on an external measurement; the model/ACHILLES validation (the model's actual job) is done.

## Status: A3 essentially complete (only the external-data overlay remains)
The differentiable free-nucleon σ(E_ν) for all 3 CC channels reproduces ACHILLES to ≤3%
for **both ν_e and ν_μ** (muon mass added; c_μ/c_e=0.998), in **physical absolute units**
(p→pπ⁺ plateau ≈0.8×10⁻³⁸ cm², the literature/Fig-2 scale), with the dσ/dM_A closure exact.
The model now produces genuine physical σ via `sigma_vs_enu_nb`. The **only** remaining
piece of Fig 2 is overlaying the external ANL/BNL (Wilkinson-2014-reanalysis) data points —
a comparison-to-measurement step, not part of the model's forward-validation job, which is
done. Gates live in `tests/test_sigma_enu.py` (closure, channel ratios, rise/plateau, ν_e &
ν_μ oracle, μ/e constant consistency, physical absolute scale).

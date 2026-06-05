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

- ☐ **A3.3 Muon-mass extension** (deferred; see above) — needed to switch the oracle/data
  comparison to ν_μ (ANL/BNL is ν_μ). Validate the massless limit against the ν_e gate above.

- ☐ **A3.5 Physical absolute units + ANL/BNL data overlay.** Model↔ACHILLES is done (c
  above). Remaining: ACHILLES-units → physical nb/10⁻³⁸cm² (derive from the C++/Fortran
  xsec assembly: flux 1/(4 m_N E_ν), G_F²cos²θ_c, MC measure — or read ACHILLES's printed
  nb on a known run), then overlay digitized ANL/BNL (Wilkinson:2014yfa — check the
  `../nuisance` clone for a data release; else digitize from the paper). With ν_μ (A3.3).

## Status: A3 core DONE (differentiable vertex validated vs ACHILLES on free nucleons)
The differentiable free-nucleon σ(E_ν) for all 3 CC channels reproduces ACHILLES to ~1%
(closure dσ/dM_A exact). Remaining for the *full* Fig-2 reproduction: muon mass (A3.3) +
physical units & ANL/BNL data overlay (A3.5). These are additive and independent of the
already-validated vertex.

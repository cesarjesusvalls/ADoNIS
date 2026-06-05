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

- ☐ **A3.4 Free-nucleon ACHILLES oracle** (ν_μ, stationary nucleon, σ(E_ν) scan).
  **Open question (from Phase 0 §2):** the `1N` example uses `QESpectral` (Fermi motion),
  not a static nucleon. Need a genuinely free/at-rest nucleon target in ACHILLES — options
  to test: a hydrogen (`1H`) card (free proton, for the p→pπ⁺ channel), a momentum-zeroed
  / delta spectral function, or a nuclear-model setting that disables Fermi motion. Run
  ν_μ (`Processes: Leptons: [14,[13]]`) at a series of E_ν → total σ per channel; parse to
  an oracle npz keyed `nu_mu_1N` (fits the Phase-0 §5 oracle matrix). Generate locally
  (emulated) at modest stats; high-stats regen in CI `oracle.yml`.

- ☐ **A3.5 Absolute normalisation + ANL/BNL overlay.** The model weight is relative
  (shape-correct to ~4e-4 vs the oracle). Absolute σ(E_ν) needs the universal constant
  G_F²cos²θ_c/(flux·measure). Two cross-checked routes: (a) derive analytically from the
  ACHILLES C++/Fortran xsec assembly (flux factor 1/(4 m_N E_ν), coupling, MC measure);
  (b) calibrate the single constant against ACHILLES's printed absolute σ at one energy,
  then verify the *energy dependence* (constant cancels) and *channel ratios* (constant
  common) are reproduced parameter-free. Overlay digitized ANL/BNL (Wilkinson:2014yfa —
  check the `../nuisance` clone for a data release; else digitize from the paper).

## Done = Fig 2 reproduced
σ(E_ν) for the 3 channels matching ACHILLES (shape+ratios at the oracle tolerance) and
overlaying the ANL/BNL data, with the closure (dσ/dM_A) passing. Then A3 is complete.

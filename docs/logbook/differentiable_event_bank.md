# Differentiable event bank — design, build, validation

## Goal
Run the cascade+reweight machinery ONCE at high statistics, persist a per-event bank, then make ANY plot
(forward T2K-style distributions OR per-knob Jacobians, ANY signal definition / observable / binning) at
plot time with no re-running. Decouple compute from plotting permanently.

## Schema (event_bank.py)
Per event (QE + RES concatenated; channel 0=qe,1=res), chunked/streamed so peak mem ~ one chunk:
- kinematics `k_nu, p_struck, k_mu` (4-vectors) — f32
- **full post-FSI final state**, RAGGED/CSR (`fs_off`, `fs_pid`, `fs_chg`, `fs_p4`) — f32. The complete
  escaped-particle buffer `nterms[0]` (all nucleons + surviving pions) → any topology signal at plot time.
- `prim_pi_pid` (pterm pid: 0=absorbed/none, ±211/111=escaped, −1=converted) — recovers the
  model_hist_full CC0π definition (primary pion absorbed) which the final-state list alone can't (primary
  vs cascade-created pions are indistinguishable without an origin tag).
- bare `w0` (nominal weight, NO signal folded in) — f64
- `D1, D2, D3` = 1st + **diagonal** 2nd & 3rd derivatives of w wrt each of the 27 knobs — f64.
  Off-diagonal Hessian is reconstructible at plot time as `Σ_i J_i J_j / w0` (knobs compose
  multiplicatively); only the diagonal needs storing. Derivs via vmap'd nested `jacfwd` along each
  one-hot knob direction (1 compile, 27 dirs × 3 orders).

Knob set/order = `grad_arrows._specs(NOM)` (27 knobs, pw_norm excluded).
Size ≈ **~1.4 KB/event** as built (138 MB for 100k/channel = 187k rows; dominated by D1/D2/D3 f64).

## Plot-time consumer (bank_plot.py)
`load_bank` (concatenate chunks, divide w0/D by n_chunks → Σ = cross section). Reductions over the ragged
final state: `n_pions`, `n_protons(pmin,pmax)`, `leading_proton`. Observables `dpt/dat` + `acceptance`
reuse the validated tune formulas. `signal_cc0pi(topological=False)` = model_hist_full def (prim_pi_pid==0);
`topological=True` = veto ANY surviving pion. `hist_forward` / `hist_gradient` / `hist_diag_curv` →
forward, per-knob gradient, diagonal 2nd/3rd, at any binning (accumulate in f64).

## Known cascade artifact (filtered in the bank, flagged for separate investigation)
~0.1% of "alive" final-state NUCLEONS carry unphysical momenta (up to inf / ~1e38 MeV) — a numerical
pathology in the pool cascade buffer. This is PRE-EXISTING: model_hist_full/grad_all read the same
`nterms[0]` buffer; the T2K proton-momentum acceptance cut rejects these events, which is why the 1M
forward/Jacobian results were clean. The bank now drops alive particles with |p|>1e4 MeV or non-finite
(`compact_finalstate`, logs the count) so the stored bank is finite and safe for ANY signal def. Root
cause in the cascade kinematics is NOT yet diagnosed — TODO (affects grad_all/model_hist_full equally,
masked by acceptance).

## Validation (bank_validate.py)
Reproduces the established results within MC stats (bank N < 1M). Re-renders the 1D arrow grids FROM THE
BANK (`*_BANK.png`) — match grad_all's per-knob shapes/signs.
- **Smoke (743 evt, 141 CC0π):** forward integral ratio bank/1M = 1.036 (dpt) / 0.917 (dat); knob-grad 0.88–1.07.
- **100k/channel (187,276 evt, 33,959 CC0π):** forward integral ratio **0.9997 (dpt) / 1.0063 (dat)**;
  dominant-knob gradient ratios **0.99–1.01 (dpt)**, within ~3% (dat) — residuals are pure 100k-vs-1M
  statistics (one high-δp_T tail bin 1.22). Diagonal 2nd/3rd derivs finite. → bank reproduces
  grad_all/model_hist_full to ~1%.

## Status
- event_bank.py / bank_plot.py / bank_validate.py committed (6e1163b); filter fix + this log: next commit.
- 100k/channel bank built (~19 min) at output/event_bank/.
- Next: 1M (or larger) bank overnight once the 100k validation lands at ~1%.

## CC1π matrix discrepancy — RES recoil Npid hardcoded (2026-06-29, commit 0f44788)
The bank-sourced T2K matrix (`bank_matrix.py`) disagreed with the validated forward matrix on **CC1π RES
only**: bank ACH/ADO 0.817 vs forward 0.996 (~22% bank σ too high). CC0π cells agreed (~1.00).

**Localization (evidence, in order):**
- Schema faithful: same-buffer test — forward `prot` (generate.py) and bank `compact_fs`+`_topk` produce
  bit-identical proton sets (median lead |p| 501.01 both, identical counts). Not a schema/selection bug.
- Pion logic faithful: final-state vs primary-pion CC1π+ selection agree to 0.2%.
- Flux/kinematics physically identical: **weighted** E_nu (mean 1567 vs 1591) and muon (frac pmu>250
  0.705 vs 0.707) match; the raw-spectrum difference was only a BEAM_MODE importance-sampling artifact.
- Staged σ-breakdown: weighted σ agrees through meson_ok→muon→pi_acc (2.81 vs 2.84 e-6) and diverges
  ONLY at the proton step (1.20 vs 1.46 e-6). Bank lead-proton harder (weighted med 522 vs 451 MeV),
  proton present in 98.9% of base events vs 67%.

**Root cause:** `event_bank.py` (and `tune.py`) fed the RES cascade a hardcoded recoil PID
`jnp.full(nr, 2212)` (all protons). `res_xsec.generate` returns `Npid ∈ {2112: ~34%, 2212: ~66%}` —
the n→n π⁺ channel recoils a NEUTRON, and that channel is a dominant CC1π⁺ signal mode. Forcing those
neutrons to protons injected spurious hard in-window protons into the signal → +22% σ. Head-to-head on
the same RES events: frac-events-with-proton 0.989 (Npid=2212) → 0.837 (threaded). The validated forward
generator (`generate.py:run_one_seed`) already threads `a["Npid"]`; the bank/tune shared the latent literal.

**Fix:** thread `res["Npid"]` (single source of truth) in event_bank.py:82 + tune.py:127. Audit of all
other `full(...,2212)` sites: every remaining one is a QE (n→p, recoil always proton) or free-proton
context — physically correct, not bugs.

**Validation (100k bank, new-bank vs forward, identical selections):** cc1pi_res 0.991, cc0pi_res 0.980
(N=2071), cc0pi_qe 1.021 — cc1pi closed from ~0.82 to ~0.99; cc0pi cells within 100k stats. Since forward
matched ACHILLES at 0.996, bank-sourced matrix cc1pi_res now ≈1.00.

**1M regenerated** → output/event_bank (10 chunks). Full matrix on the corrected bank: cc1pi_res_incl
**0.817 → 0.992** (pull −0.5), cc1pi_both_incl 0.987, cc0pi_qe_incl 0.996, cc0pi_both_incl 0.998 — every
physically-meaningful cell ~1–2%, pulls ≤1.3σ (low-stats outliers: cc0pi_res_0p N_ach=98, cc1pi_qe_* via
rare FSI π-creation — noise). All 5 PDFs (t2k matrix + cc0pi/cc1pi/cc1pi-stv/incl variation) regenerated.

**CC0π tune re-validated** (tune.py Npid change): nominal χ²/ndf 1.89 (A=0.874) at 40k/NREP=4 vs pre-fix
2.05 (A=0.881) production — within the documented stats spread (1.44–2.05); fit unperturbed (CC0π is
QE-dominated; RES enters only via pion-absorption).

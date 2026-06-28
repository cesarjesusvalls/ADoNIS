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
Reproduces the established results within MC stats (bank N < 1M). Smoke (743 evt, 141 CC0π): forward
integral ratio bank/1M = 1.036 (dpt) / 0.917 (dat); dominant-knob gradient ratios 0.88–1.07; diagonal
2nd/3rd derivs finite. 100k run: <PENDING — fill ratios>. Re-renders the 1D arrow grids FROM THE BANK
(`*_BANK.png`) to eyeball against grad_all.

## Status
- event_bank.py / bank_plot.py / bank_validate.py committed (6e1163b); filter fix + this log: next commit.
- 100k/channel bank built (~19 min) at output/event_bank/.
- Next: 1M (or larger) bank overnight once the 100k validation lands at ~1%.

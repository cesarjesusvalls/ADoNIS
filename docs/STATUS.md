# ADoNIS status

A Differentiable generatOr of Neutrino Interaction Samples (JAX), mirroring the
ACHILLES physics implementation. Goal: reproduce all ACHILLES predictions in
arXiv-2508.19213v2 to ≤1%, preserving differentiability. The validation metric is
always ADoNIS vs ACHILLES (not vs data).

Orientation: `docs/phases/README.md` is the phase-by-phase record (what was built,
gates, commands). `docs/logbook/` holds long-task investigation logbooks.
`docs/CONTAINER.md` covers running the ACHILLES oracle images.

## Where we are (2026-06-12)

**CC0π for the T2K configuration agrees with ACHILLES.** The full chain — QE + RES
primaries on ¹²C/H, real nucleon+pion cascade FSI, topology signal, TKI observables —
is in place and differentiable (kind-1 sample/reweight contract; closures recover
parameters under MC noise).

Milestones behind that, with the durable record for each:

| area | state | record |
|---|---|---|
| QE primary | bit-exact transliteration of the ACHILLES generator path (amps2/flux/spectral); abs σ ~1.4% | git history (QEDUMP audits) |
| RES primary (DCC 1π) | frame fix (q∥z before amps2) + π⁰-mass convention closed the norm: free p 1.002, ¹²C 0.994 | `res_amps2_frame_fix.md`, `conventions.py` |
| QE FSI | struck-neutron removal (pid through cascade): ACH/ADO 1.040 → 1.000 | git history (`9189390`, `1138307`) |
| π cascade: absorption | isospin partition (π⁺p/π⁻n = 5/6·oset): +5% residual closed | `logbook/cascade_transport_residual.md`, `9124afb` |
| π cascade: reaction | charge-resolved scatter σ: −4% deficit closed; beam-pion escape plane fix: 305/335 MeV closed | same logbook, `1ad6cf0`, `1b938a2` |
| differentiability | cascade differentiable in (σ_abs, σ_scatter); M_A and FSI knobs recovered jointly by gradient descent | tests/ closure gates |

## Parked

**+1.5% pion-cascade reaction overshoot at 245 MeV** (appeared after the escape-plane
fix; decomposed as +0.8% cascade-proper on identical configs + ~0.7% config-ensemble).
Deliberately parked — not being chased further for now. Full hypothesis ledger, evidence,
and next steps in `docs/logbook/cascade_transport_residual.md` (§6–7, entries #13–14).

Known small open items (recorded in `res_amps2_frame_fix.md`): dσ/dW high-W tail
(W > 1600 MeV) ~0.8–0.9× (DCC table coverage / interpolation), one W-threshold bin
from the π-mass convention in the mono figure.

## Next

Repo tidy-up (docs pruned 2026-06-12; architecture refactors: constants unification,
Sample container, xsec legacy isolation, oracle scripts out of the package, logging),
then continue toward full arXiv-2508.19213v2 figure coverage.

## History

The detailed earlier history (toy reweighting machinery, DCC port, hadron-tensor fold,
EW-isospin fix, oracle/CI setup) lives in `docs/phases/` and git history; the previous
revision of this file summarized phases 1–3 and is retained in git history.

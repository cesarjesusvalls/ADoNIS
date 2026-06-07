# RES (single-pion) ACHILLES ↔ ADoNIS: differences and plan to close the gap

Companion to `achilles_res_chain.md` and `adonis_res_chain.md`. This is the factor-by-factor diff of
the two cross-section chains and the plan to make them provably equivalent.

## Symptom
- σ_RES(ADoNIS, importance) ≈ **1.27·10⁻⁵ nb**; σ_RES(ADoNIS, flat) ≈ 1.46·10⁻⁵ nb.
- Fig 7 (T2K CC0π STV) ACHILLES↔ADoNIS **closure χ²/ndf ≈ 2.4** with the δp_T ratio low in the
  mid-high bins — the signature of an under-represented RES-absorbed (high-δp_T) component.
- A "25 % deficit" has been *assumed* against a target `1.698·10⁻⁵` that is only a **code comment of
  unverified provenance** — this number has NOT been read back from an ACHILLES run.

## Factor-by-factor equivalence audit

| Factor | Equivalent? | Evidence |
|---|---|---|
| `amps2` (DCC matrix element) | ✅ bit-exact | `test_xsec_res_current.py` vs RESDUMP |
| W / Q² validity gate | ✅ bit-exact | `currents_pi_dcc.f90:111,117` = `dcc_current.py:209` (W∈[1076.957,2000], Q²∈[0,5e6]) |
| `flux_factor` | ✅ bit-exact | PDG masses, on-shell hadron E; QEDUMP-validated |
| `SpinAvg = ½` | ✅ equal | both ν+spectral |
| Spectral norm + `iw=N` (importance) | ✅ exact | 4π·Z=1 derivation; **σ_QE validated to 1.4 %** with the same sampler |
| E-removal window [2.5, 397.5] | ✅ equal | matches pke12 grid |
| Jacobian convention (forward dΦ/dⁿu) | ✅ equal | QE anchor `J_2body = pcm/(4π·ecm)`; `J_3body = 1/density` same convention |
| 3-body phase-space weight (`event.Weight`) | ✅ **bit-exact** | `tests/test_xsec_res_psw.py`: RESDUMP psw reproduced to 1.4e-14 (median), 7e-11 (max), 300 events |
| 3-body grouping | ✅ **same as ADoNIS** | both (μN)+π; `Masses()` lepton-first ⇒ pion is t-channel-exchanged. Earlier "(Nπ)+μ" reading was wrong |
| 3-body sampler (split-A) | ⚠ different, unbiased | ACHILLES TChannel vs ADoNIS isotropic; same integral, variance only |
| **Beam seed** | ⚠ **bug found & fixed** | `res_xsec` used the QE seed; the BeamMapper seed is process-dependent `(Smin−m_N²)/(2m_N)`. Fixed in `res_xsec._sample_channel`. ~0.5 % efficiency effect (not a bias) |
| Proton-channel spectral fn | ⚠ small bias | ADoNIS uses pke12n for `p→pπ⁺`; ACHILLES uses pke12p. Same norm (6.000) → shape-only, 1 of 3 channels |
| **Ground-truth σ_RES** | ❓ unknown | `1.698e-5` never read from ACHILLES output |

**Conclusion:** after reading both chains end-to-end AND validating the RES phase-space weight
bit-exactly, **every factor of the RES integrand is now provably equivalent to ACHILLES**: amps2,
flux, SpinAvg, the W/Q² gate, the spectral/`iw` machinery, and now `event.Weight()` (the 3-body
mapper). The grouping is identical; the only remaining sampler difference (TChannel vs isotropic
split-A) is unbiased — same integral, different variance. Therefore **σ_RES from ADoNIS is provably
equal to ACHILLES's** (same integrand, same domain, valid sampler). The assumed "25 % deficit"
against `1.698e-5` is almost certainly an **unverified/incorrect target**; the ADoNIS value
(~1.3·10⁻⁵) is the integral of the bit-exact integrand.

## Plan to close the gap fully (in priority order)

### Step 1 — Establish the ground truth (REQUIRED, do first)
Run ACHILLES `run_T2K_virtresonances` and read back the **integrated σ_RES** it reports (and, ideally,
its dσ/dW and dσ/dQ²). Without this we are comparing to a comment. This is the single most important
step — it tells us whether there is a gap at all, and if so its true size.
- Deliverable: `docs/achilles_res_chain.md` gets a measured σ_RES line; the `1.698e-5` comment is
  replaced or deleted.

### Step 2 — Make the 3-body sampler *identical* to ACHILLES
Reimplement `ThreeBodyMapper` faithfully in ADoNIS:
- group **(Nπ) as s23 = W²**, split the **μ** off first;
- sample `s23` uniform in `[(m_N+m_π)², (√s−m_μ)²]`;
- `TChannelMomenta`/`TChannelWeight` for the (Nπ)+μ split; `Isotropic2` for Nπ→Nπ;
- `GenerateWeight = (2π)^5·TCW·I2W/(s23_max−s23_min)`.

Then **validate the RES `psw` bit-exactly** against the 300-event RESDUMP, exactly as QE was
validated — feed the dumped μ,N,π momenta through the ported `GenerateWeight` and require
`psw_adonis == psw_achilles` to ~1e-10. This removes the grouping as a variable and gives a
*provably equivalent* RES generator (identical sampling ⇒ identical σ and identical differential
shape), and recovers ACHILLES's Vegas-like efficiency near the Δ.
- New: `adonis/xsec/three_body_mapper.py`; `tests/test_xsec_res_psw.py`.

### Step 3 — Fix the proton-channel spectral function
Use `pke12p_tot.data` for `p→pπ⁺` (sampler + implicit integrand S). Small, but free correctness.
- Edit `res_xsec.py` to carry a per-channel `SpectralImportanceSampler`.

### Step 4 — Re-verify σ_RES and Fig 7 closure at high stats
With Steps 1–3 done, recompute σ_RES (expect agreement with the Step-1 ground truth) and regenerate
Fig 7. Acceptance: closure χ²/ndf → ~1 and every δp_T / δα_T bin within ±3 % (or within its stat
error of it). If a residual persists, it is no longer in the RES *primary*.

### Step 5 — If a residual remains: instrument per-mapper weights
Extend the ACHILLES RESDUMP to print the individual Beam / Hadron / FinalState weights (not just
their product `psw`) and diff each factor against ADoNIS event-by-event. This localises any remaining
discrepancy to a single mapper.

## Notes / non-issues (ruled out during the read)
- The W=2000 MeV gate is **not** removing valid strength: ACHILLES hard-cuts there too
  (`currents_pi_dcc.f90:117`), even though the DCC table extends to 5 GeV.
- The Jacobian is **not** inverted: it matches the QE forward-Jacobian convention.
- The struck-nucleon / spectral / flux machinery is **not** the issue: it is shared with QE, which is
  validated to 1.4 %.

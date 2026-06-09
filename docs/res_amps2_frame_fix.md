# RES dσ/dQ² sag — root cause, fix, and the debugging process

**Status: RESOLVED (2026-06-09).**

The ADoNIS RES single-pion dσ/dQ² disagreed with ACHILLES by a smooth monotone tilt
(ratio ≈ 0.7 at low Q² → 1.3 at high Q², χ²/ndf ≈ 240), and the total σ_RES sat at ≈ 0.74×
ACHILLES. Both turned out to be the **same** root cause.

## Root cause

`adonis/xsec/dcc_current.py::exclusive_amps2_batch` (the DCC RES matrix element) was **not
rotation-invariant**. amps2 is a Lorentz scalar, so it must be — but it changed by up to **6.6×**
under a pure rotation (median ratio amps2(q∥z)/amps2(lab) = 1.22).

The DCC partial-wave / Wigner-d construction uses the **momentum transfer q as the quantization
(z) axis**. So the amplitude is only correct when **q is along +z**. ACHILLES enforces this with
`NuclearModel::TransformQZ` (`q.AlignZ()`), which rotates every event so q∥z *before* computing the
hadron current — confirmed directly in an `-finstrument-functions` call trace of one event
(`GenerateSingleEvent → … → TransformFrame → TransformQZ → CalcCurrents`).

The ADoNIS RES generators (`res_xsec`, `scripts/free_proton_gen.py`,
`scripts/achilles_mirror_gen.py`) evaluated amps2 in the **lab frame** (beam∥z, q *not* along z).
That frame mismatch *was* the entire sag and the σ_RES deficit.

## Why it hid for months

Every validation of amps2 against ACHILLES used ACHILLES **dumps** (RESDUMP / the injection driver /
the bulk `resbig` dump). Those momenta are all written **after** `TransformQZ`, i.e. already q∥z —
the one frame where the ADoNIS amps2 is correct. So amps2 "matched ACHILLES bit-for-bit" in every
test (`zmtx_compare`, `zj_compare`, `lh_reconcile`, the 2D ratio, the driver). The only code path
that ever fed amps2 lab-frame momenta was the actual event generation, which is exactly the thing
the dumps don't cover.

## The fix

One rotation at the top of `exclusive_amps2_batch` (`ROTATE_QZ = True`, `_rotate_q_to_z`): rotate
all event momenta so q∥z before building the amplitude, making the function frame-invariant. `gw`
(phase-space weight) and `flux` are built from invariants and need no change. All callers are fixed
automatically.

**Verified:**
- mono free proton dσ/dQ² ratio: lab 0.70→1.38 → q∥z **flat**, χ²/ndf 240 → **1.5**
  (`figures/free_proton_WQ.png`).
- flux-averaged `paper_figures/res_WQ2_shapes.png`: dσ/dQ² flat; **total σ_RES 0.74× → 0.969×**.
- sanity on ACHILLES q∥z dump momenta: my/ach = 1.0010 unchanged → all prior bit-exact validations
  still hold (the rotation is identity there).

## How it was found — the substitution bisection

The decisive method (full ACHILLES matches its own hepmc; replace its functions with ADoNIS's one at
a time until the match breaks). Built with ACHILLES-side instrumentation (see "Debug infrastructure"):

1. **Vegas** — ran ACHILLES with a flat grid (`ACHILLES_NO_VEGAS`); hepmc unchanged. Eliminated.
2. **amps2 value** — ACHILLES's own amplitude on injected ADoNIS events == ADoNIS amps2 (driver).
   Eliminated *(but only because injected events get TransformQZ'd — the hidden assumption)*.
3. **`GenerateWeight` (gw)** — with Vegas off, `psw × gw = 1.00000` (cv 0): bit-identical. Eliminated.
4. **flux / initwgt / spinavg** — constant. Eliminated.
5. **PercentileUnweighter** — replicated; provably unbiased, no shape change. Eliminated.
6. **a missing step** — `-finstrument-functions` call trace of one event showed no hidden function.
7. **GeneratePoint** — raw proposal distributions (cosθ*, Q², W, s23) matched to ~3%.

The tell: *"ACHILLES proposal events + ADoNIS recipe = hepmc, but ADoNIS mirror events + the
**identical** recipe = sag."* Same code, same amps2 function, different value — because ACHILLES's
events were already q∥z and the mirror's were lab-frame. Testing amps2 for rotation-invariance then
showed the 6.6× frame dependence directly.

## Debug infrastructure kept (for future use)

ADoNIS (`scripts/`):
- `achilles_mirror_gen.py` — bit-faithful port of ACHILLES's `ThreeBodyMapper` (t-channel
  `GeneratePoint`/`GenerateWeight`) as a standalone free-proton generator; the controlled testbed.
- `free_proton_gen.py` — mono 1 GeV free-proton dσ/dW, dσ/dQ² with ratio + χ² vs the hepmc oracle.
- `test_3body_q2_measure.py` — flat-Dalitz (uniform-phase-space, machinery-free) ground truth.
- `validate_res_psw.py` — bit-exact reconstruction of ACHILLES `event.Weight()` (the mapper port).
- `zmtx_compare.py`, `zj_compare.py`, `lh_reconcile.py`, `weight_reconcile.py` — per-piece reconcilers.

ACHILLES side (instrumentation, env/counter-gated — see the ACHILLES repo working tree):
- `XSecBackend.cc` driver injection (`ACHILLES_DRIVER`): override event momenta to run ACHILLES's own
  amplitude on arbitrary external events. Inject **before** `TransformFrame` (q∥z matters!).
- `Process.cc` `ACHILLES_NO_VEGAS`: skip Vegas optimization (flat grid) to test for Vegas bias.
- `instrument_trace.cc` + `-finstrument-functions`: full ordered call trace of one `GenerateSingleEvent`.
- `XSecBackend.cc` RESDUMP/LHDUMP/QEDUMP/ZMTX/ZJ dumps for per-event reconciliation.

## Residual items (separate, smaller, not the sag)

- dσ/dW high-W tail (W > 1600 MeV) sits ~0.8–0.9× — likely DCC table coverage / bilinear-vs-spline
  interpolation in the second-resonance region. (Use `BATCH_INTERP="spline"` for the faithful number.)
- One W-threshold bin in the mono figure: ADoNIS uses the π⁺ kinematic mass 139.57 where ACHILLES
  uses 134.977 → πN threshold W_min ~4.6 MeV higher → lowest-W bin depleted.

Superseded notes: the earlier `docs/res_WQ2_investigation_findings.md` recorded many dead-end leads
(pion mass, phase-space weight, Vegas, de Forest) — all wrong; kept only as a record of the hunt.

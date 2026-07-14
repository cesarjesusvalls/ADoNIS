# Soft-π⁺ absorption σ is NaN (Oset) — found 2026-07-14, NOT fixed, needs a decision

## What

`oset_xsec.abs_cross_section` returns **NaN** whenever the pion has `E < m_π`:

```
pi+ mass 139.57018
  E=189.570  E-m=+50.000  sigma_abs=14.23
  E=140.570  E-m= +1.000  sigma_abs=7.51
  E=139.570  E-m= +0.000  sigma_abs=7.43
  E=139.070  E-m= -0.500  sigma_abs=nan      <-- 
  E=135.976  E-m= -3.594  sigma_abs=nan
```

Source: `oset_xsec.py:_reduced_half_width`

```python
pion_mom = jnp.sqrt(pionE * pionE - m_pi * m_pi)     # UNCLIPPED -> sqrt(neg) for E < m
```

## Why E < m_π happens in ADoNIS

Two different pion masses, both of them deliberate:

| where | mass | why |
|---|---|---|
| RES generator builds the pion 4-vector | `_pi_kin_mass` = **m_π0 = 134.98** | the validated ACHILLES 1π-kinematics convention (`res_xsec.py:46`; fixing this closed the RES norm deficit) |
| cascade evaluates σ | `_CH_MASS = [M_PIP, M_PI0, M_PIP]` → **139.57** for π± | per-charge physical mass |

For a π⁺: `E² − m_π+² = |p|² + 134.98² − 139.57² = |p|² − 1259.6`, negative for **|p| ≲ 35.5 MeV/c**.
So every soft π⁺ gets σ_abs = NaN.

## Consequence in the walk (silent)

`prob = exp(−π·b²/σ_tot)` → NaN, and `passes = (u < NaN)` is **False**. So a NaN-σ candidate
**never interacts** — the interaction opportunity is silently dropped. Measured: **~1.3% of in-slab
candidate steps** (80 / ~6000 on a 1328-event RES sample). It never surfaced before because the old kind-1
pion record only logged *hits*, and a NaN-σ candidate can never be a hit.

## Is ACHILLES the same?

Yes, as far as the source shows — this is **not** a transcription error:
- `Achilles/src/Achilles/OsetCrossSections.cc :: ReducedHalfWidth` has the identical unclipped
  re-derivation: `auto pion_mom = sqrt(pionE * pionE - pion_mass * pion_mass);`
- `Particle::Mass()` (`Particle.hh:209`) returns `info.Mass()` — the **PID** mass (139.57 for π⁺), not the
  4-vector mass. So ACHILLES has the same E-vs-m inconsistency available to it.
- In C++, `sqrt(negative)` is NaN and `rand < NaN` is false → same silent no-interaction.

**Not yet verified against a running ACHILLES.** The cheap check: instrument `OSETABS` (the fprintf at
`OsetCrossSections.cc:73` already prints Tpi/pmom/vrel/sqrts) and count NaN σ for soft π⁺ in a T2K run.

## Note on internal inconsistency (independent of the mass question)

Within a *single* σ evaluation the two halves disagree about the pion momentum:
- `_kinematics` uses the `pion_mom` it is **passed** (the true `|p|` from the 4-vector),
- `_reduced_half_width` **re-derives** it as `sqrt(E² − m²)`.

They agree only when the 4-vector is on-shell w.r.t. `m_pi`. ACHILLES has the same structure (its
`PXSecCommon` even has the re-derivation commented out at line 210, while `ReducedHalfWidth` still does it).

## Status: forward walk UNCHANGED

Nothing about the forward physics was touched — it stays bit-faithful to ACHILLES.
What *was* done (commit: pion survival reweight) is only that the kind-1 record **excludes non-finite-σ
candidates**. That is not a patch, it is the correct encoding: a NaN-σ candidate cannot interact at ANY θ
(scaling NaN is still NaN), so it carries zero θ-dependence and recording it would inject NaN into the
weight for no physics.

## The decision to make

1. **Leave it** — bit-faithful to ACHILLES, including the pathology. ~1.3% of pion candidate steps cannot
   interact; soft π⁺ are under-absorbed by construction on BOTH sides, so ADoNIS-vs-ACHILLES closure is
   unaffected. (Diverging here would *break* closure.)
2. **Fix it in both** — clip the root (`sqrt(clip(E²−m², 0))`) or, better, thread the true `|p|` through
   (`_reduced_half_width` should not re-derive what the caller already has — CLAUDE.md discipline). Then
   soft π⁺ get a finite σ and can absorb. This CHANGES ADoNIS predictions and must be mirrored in the
   ACHILLES fork to keep the comparison meaningful.
3. **Fix the mass convention instead** — put the cascade pion on-shell w.r.t. the mass used to build it.

Option 2/3 shift pion absorption at low |p|, which is exactly the region the T2K CC0π/CC1π FSI knobs are
sensitive to — so this is not cosmetic. Needs the user's call.

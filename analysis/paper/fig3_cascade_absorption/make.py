"""Paper Fig. 3 (arXiv:2508.19213v2): pi+ on 12C REACTION and ABSORPTION cross section sigma(p_pi),
ADoNIS vs ACHILLES.  This is the pion-nucleus FSI observable the cascade is built to reproduce; both
sides use the SAME untuned in-medium cross sections (Oset absorption + ANL-Osaka DCC scatter), so the
comparison is a pure TRANSPORT validation (docs/logbook/cascade_transport_residual.md: <1% at the Delta).

ADoNIS side = the validated POOL cascade (_pion_step physics via run_cascade_pool), driven in ACHILLES's
EXACT CrossSection-mode beam geometry (RunCascade.cc::InitCrossSection): a pi+ fired along +z at impact
parameter b uniform in a disk of radius R_DISK=10 fm, starting at z = -1.05*R_nuc (5% outside the
surface, status external_test -> no formation zone).  No fitted constants, no duplicated physics -- it
reuses cascade_full.run_cascade_pool + cascade_discrete._pion_step (the same engine the T2K blueprint
uses).  reaction = the primary pion ACTUALLY interacted (prim_fate ABSORB/CONVERT or an actual scatter
nsc>0) -- NOT kind-1 nh, which records the pre-Pauli sampled hit (ACHILLES counts a reaction only when
NOT Pauli-blocked: Cascade.cc:903 `if(hit)`); absorption = reacted with NO surviving pion.

ACHILLES side = configs/achilles/run_cascade_pip_C.yml (achilles:cascade, Mode: CrossSection, VirtRes
interactions), a uniform-momentum beam over KickMomentum=[80,500].  Only REACTED events are written; the
GenCrossSection counter gives n_tried (uniform in p), so sigma(p_b) = piR^2 * n_react,b / (n_tried *
binwidth/range), absorption likewise with "no final-state pion".  ACHILLES SIGSEGVs mid-run -> combine
seed batches (the n_tried estimator is unbiased for partial runs).

Usage:  python -m analysis.paper.fig3_cascade_absorption.make [n_adonis]
  ACHILLES scan first: python -m analysis.utils.run_achilles configs/achilles/run_cascade_pip_C.yml --seeds 8
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp
from adonis.workflow.materials import resolve_targets
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density, sample_nucleons, _CH_MASS
from adonis.fsi import cascade_full as CF

ACH_DIR = ROOT / "output" / "achilles"
OUT = ROOT / "output" / "figures"
R_DISK = 10.0                               # beam impact-parameter disk [fm] (card Cascade/Params/radius)
PIR2_MB = np.pi * R_DISK ** 2 * 10.0        # pi R^2 in mb (1 fm^2 = 10 mb) = 3141.6 mb
P_LO, P_HI = 80.0, 500.0                    # KickMomentum range [MeV]
EDGES = np.linspace(P_LO, P_HI, 22)         # 21 bins of 20 MeV
CEN = 0.5 * (EDGES[:-1] + EDGES[1:])


# --------------------------------------------------------------------------- ADoNIS (pool, beam geometry)
def adonis_transparency(n=200_000, seed=0, target="C", M=12, max_steps=2000):
    """sigma_reaction(p), sigma_abs(p) [mb] from the pool cascade fired in ACHILLES's CrossSection beam."""
    tg = resolve_targets(target)[0][0]
    cfg = DiscreteCascadeConfig(nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs,
                                step=0.04, pauli=True, nn_inelastic=True)
    _rg, _rp, _rn, radius = _load_density(cfg.nucleus, cfg.density_n)
    z0 = -1.05 * float(radius)                          # InitCrossSection: 5% outside the nuclear surface
    key = jax.random.PRNGKey(seed)
    k_mom, k_disk, k_nuc, k_run = jax.random.split(key, 4)
    mom = jax.random.uniform(k_mom, (n,), minval=P_LO, maxval=P_HI)
    u = jax.random.uniform(k_disk, (n, 2))
    br = R_DISK * jnp.sqrt(u[:, 0]); th = 2 * jnp.pi * u[:, 1]
    m_pi = _CH_MASS[0]                                  # pi+ kinematic mass (charge index 0)
    E = jnp.sqrt(mom ** 2 + m_pi ** 2)
    p4 = jnp.stack([E, jnp.zeros(n), jnp.zeros(n), mom], axis=1)        # beam along +z
    pos0 = jnp.stack([br * jnp.cos(th), br * jnp.sin(th), jnp.full((n,), z0)], axis=1)
    # background nucleus (struck vertex unused: beam pion is external) -> build su manually
    kn, kp = jax.random.split(k_nuc, 2)
    npos, nmom, nisp = sample_nucleons(kn, n, cfg)
    A = nisp.shape[1]
    su = dict(npos=npos, nmom=nmom, nisp=nisp, pos0=pos0,
              consumed0=jnp.zeros((n, A), bool), ch0=jnp.zeros(n, jnp.int32), kp=kp)
    _kpi, knuc, _kpi2 = jax.random.split(su["kp"], 3)
    g0 = CF.empty_batch(n, 1)
    g0["alive"] = jnp.ones((n, 1), bool)
    g0["species"] = jnp.full((n, 1), CF.PION, jnp.int32)
    g0["charge"] = jnp.zeros((n, 1), jnp.int32)                          # pi+
    g0["p4"] = p4[:, None, :]; g0["pos"] = pos0[:, None, :]
    g0["origin"] = jnp.full((n, 1), CF._ORIG_PRIM_PI, jnp.int32)
    _base = jax.vmap(lambda e: jax.random.fold_in(knuc, e))(jnp.arange(n))
    g0["pkey"] = jax.vmap(lambda b: jax.random.fold_in(b, 0))(_base)[:, None, :]
    stepper = CF.make_pool_stepper(su, cfg, with_rec=True)
    out, _sofl, _oofl, prim_fate, (_fsi_rec, _rofl) = CF.run_cascade_pool(
        g0, stepper, knuc, su["consumed0"], M=M, max_steps=max_steps, M_out=24,
        prim_origin=CF._ORIG_PRIM_PI, rec_caps=(8, 8))
    # REACTION = the primary pion ACTUALLY interacted (ACHILLES Cascade.cc:903 adds the history vertex
    # only inside `if(hit)` -> a Pauli-BLOCKED interaction is NOT a reaction).  So count the latched
    # primary fate (absorbed/converted) OR an actual scatter (nsc counts is_scat only, excludes blocked),
    # NOT the kind-1 `nh` (which records the pre-Pauli sampled hit and would over-count blocked scatters).
    prim_fate = np.asarray(prim_fate)
    prim_pi = np.asarray((out["species"] == CF.PION) & out["alive"] & (out["origin"] == CF._ORIG_PRIM_PI))
    nsc_prim = (np.asarray(out["nsc"]) * prim_pi).max(axis=1)            # surviving primary pion scatter count
    scattered = nsc_prim > 0
    reacted = (prim_fate == CF.FATE_ABSORB) | (prim_fate == CF.FATE_CONVERT) | scattered
    has_pi = np.asarray(((out["species"] == CF.PION) & out["alive"]).any(axis=1))
    absorbed = reacted & ~has_pi                                         # reacted with no surviving pion
    p = np.asarray(mom)
    idx = np.clip(np.digitize(p, EDGES) - 1, 0, len(CEN) - 1)
    sr = np.zeros(len(CEN)); sa = np.zeros(len(CEN)); cnt = np.zeros(len(CEN))
    for b in range(len(CEN)):
        m = idx == b; nb = int(m.sum()); cnt[b] = nb
        if nb:
            sr[b] = PIR2_MB * reacted[m].mean()
            sa[b] = PIR2_MB * absorbed[m].mean()
    return CEN, sr, sa, cnt


# --------------------------------------------------------------------------- ACHILLES (CrossSection hepmc)
def _events(path):
    """Yield (p_in[MeV], absorbed[bool], evt_id) per WRITTEN (=reacted) event; also return n_tried."""
    p_in = None; has_final_pi = False; evt_id = None; n_tried = 0
    with open(path) as fh:
        for line in fh:
            t = line[:2]
            if t == "E ":
                if p_in is not None:
                    yield p_in, (not has_final_pi), evt_id
                f = line.split(); evt_id = int(f[1]); p_in = None; has_final_pi = False
            elif t == "A " and "GenCrossSection" in line:
                n_tried = max(n_tried, int(line.split()[-1]))            # 4th field = running n_tried
            elif t == "P ":
                f = line.split(); pid = int(f[3]); status = int(f[9])
                if abs(pid) in (211, 111) and status == 29 and p_in is None:
                    p_in = abs(float(f[6]))                              # incoming beam pion |pz|
                if abs(pid) in (211, 111) and status == 1:
                    has_final_pi = True
    if p_in is not None:
        yield p_in, (not has_final_pi), evt_id
    _events.n_tried = n_tried


def achilles_transparency(target="C"):
    """sigma_reaction(p), sigma_abs(p) [mb] combined over all run_cascade_pip_<target>*.hepmc seed batches."""
    paths = sorted(ACH_DIR.glob(f"cascade_pip_{target}*.hepmc"))
    if not paths:
        return None
    nr = np.zeros(len(CEN)); na = np.zeros(len(CEN)); n_tried = 0
    for hp in paths:
        gen = _events(hp)
        for p_in, absorbed, _eid in gen:
            b = int(np.clip(np.digitize([p_in], EDGES)[0] - 1, 0, len(CEN) - 1))
            nr[b] += 1; na[b] += int(absorbed)
        n_tried += getattr(_events, "n_tried", 0)
    ntot = n_tried * (np.diff(EDGES) / (P_HI - P_LO))                    # uniform-in-p attempts per bin
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = PIR2_MB * nr / ntot; sa = PIR2_MB * na / ntot
        er = PIR2_MB * np.sqrt(nr) / ntot; ea = PIR2_MB * np.sqrt(na) / ntot
    return CEN, sr, sa, er, ea, nr, na


# --------------------------------------------------------------------------- figure
def main(argv=None):
    args = argv or sys.argv[1:]
    n_ado = int([a for a in args if a.isdigit()][0]) if any(a.isdigit() for a in args) else 200_000
    cache = OUT / "fig3_adonis_transparency.npz"                       # persist ADoNIS sigma(p) (instant re-render)
    if cache.exists() and "--recompute" not in args:
        d = np.load(cache); cen, sr_d, sa_d, cnt = d["cen"], d["sr"], d["sa"], d["cnt"]
        print(f"ADoNIS transparency: loaded cache {cache.name}", flush=True)
    else:
        print(f"ADoNIS pool transparency: n={n_ado} ...", flush=True)
        cen, sr_d, sa_d, cnt = adonis_transparency(n=n_ado)
        os.makedirs(OUT, exist_ok=True); np.savez(cache, cen=cen, sr=sr_d, sa=sa_d, cnt=cnt)
    ach = achilles_transparency("C")
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), height_ratios=[3, 1], sharex="col")
    PANELS = [("reaction", r"$\pi^+\,^{12}$C reaction $\sigma$", sr_d, "C0", 0),
              ("absorption", r"$\pi^+\,^{12}$C absorption $\sigma$", sa_d, "C3", 1)]
    for key, title, yd, col, j in PANELS:
        a0, a1 = axes[0, j], axes[1, j]
        a0.plot(cen / 1000, yd, "s-", color=col, ms=4, lw=1.2, label="ADoNIS (pool)")
        chi = ""
        if ach:
            xa = ach[0]; ya = ach[2] if key == "absorption" else ach[1]
            ey = ach[4] if key == "absorption" else ach[3]
            a0.errorbar(xa / 1000, ya, yerr=ey, fmt="D--", color="0.35", ms=4, lw=1.0,
                        capsize=2, label="ACHILLES (VirtRes)")
            m = (ya > 0) & (yd > 0) & (ey > 0)
            r = yd[m] / ya[m]
            a1.errorbar(xa[m] / 1000, r, yerr=r * ey[m] / ya[m], fmt="o", color=col, ms=4, capsize=2)
            chi2 = float(np.sum(((yd[m] - ya[m]) / ey[m]) ** 2)); ndf = int(m.sum())
            chi = f"   χ²/ndf={chi2/max(ndf,1):.2f}"
        a1.axhspan(0.9, 1.1, color="green", alpha=0.12); a1.axhline(1.0, ls="--", color="green", lw=0.7)
        a0.set_title(title + chi, fontsize=10); a0.set_ylim(bottom=0); a0.legend(fontsize=8)
        a1.set_ylim(0.5, 1.5); a1.set_xlabel(r"$p_\pi$ [GeV]")
        if j == 0:
            a0.set_ylabel(r"$\sigma$ [mb]"); a1.set_ylabel("ADO/ACH")
    fa = (PIR2_MB * 0)  # absorption fraction annotation
    if ach:
        frac_d = sa_d[cnt > 0].sum() / max(sr_d[cnt > 0].sum(), 1e-9)
        frac_a = ach[6].sum() / max(ach[5].sum(), 1e-9)
        fa = f"abs. fraction  ADoNIS {frac_d:.2f} / ACHILLES {frac_a:.2f}"
    fig.suptitle(r"Fig. 3 — $\pi^+\,^{12}$C reaction & absorption $\sigma(p_\pi)$: ADoNIS vs ACHILLES "
                 f"(same untuned Oset+DCC).  {fa}", fontsize=10)
    os.makedirs(OUT, exist_ok=True); out = OUT / "fig3_cascade_absorption.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); print("wrote", out, flush=True)
    if not ach:
        print("  (no ACHILLES cascade scan yet -- run: python -m analysis.utils.run_achilles "
              "configs/achilles/run_cascade_pip_C.yml --seeds 8)", flush=True)


if __name__ == "__main__":
    main()

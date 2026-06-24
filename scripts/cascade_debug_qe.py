"""ADoNIS QE -> cascade-created-pion dump: run the QE proton through the FULL pool cascade and record
the cascade-created pion multiplicity (npip/npi0/npim from escaped terminals) vs the QE nucleon's
initial |p|.  This probes the NN->NDelta->Npi (nn_inelastic) secondary-pion channel, the statistically
significant CC1pi-QE matrix cell (ACH/ADO=0.688, -4.6sigma: ADoNIS over-produces created pi+).

ACHILLES reference (from /tmp/ach_fatepion_C_gauss.fate, QE-primary events): created-pi+ rate 0.0040
overall; ~0 below p_N~700, 0.0034 at [800,1000), 0.0388 at [1000,1500).

Run: python -u scripts/cascade_debug_qe.py [n_qe_per_seed] [out.npz] [P] [M_out] [max_steps] [n_recoil]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
n_qe = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ado_cascade_qe_C.npz"
P_BUF = int(sys.argv[3]) if len(sys.argv) > 3 else 16
M_OUT = int(sys.argv[4]) if len(sys.argv) > 4 else 48
MAX_STEPS_REQ = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
N_RECOIL = sys.argv[6] if len(sys.argv) > 6 else "8"
os.environ["ADONIS_N_RECOIL"] = N_RECOIL

import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
from adonis.workflow.materials import resolve_targets
from adonis.xsec import qe_xsec
import adonis.xsec.dcc_current as dcc; dcc.BATCH_INTERP = "spline"
from adonis.xsec.spectral import SpectralFunction
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density
from adonis.fsi.cascade_full import (setup_nucleus, empty_batch, make_pool_stepper, run_cascade_pool,
                                      PION, NUCLEON)
SEEDS = 8; CHUNK = 25000
ED = [0, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 3000.0]


def build_cfg():
    tg = resolve_targets("C")[0][0]
    _, _, _, radius = _load_density(tg.density_p, tg.density_n)
    ms = max(MAX_STEPS_REQ, int(np.ceil(3.0 * radius / 0.04)))
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True, pauli=True,
                                nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
    return tg, cfg, ms


def qe_chunk(p_N, Npid, cfg, key):
    """QE pool cascade: gen-0 stack = the struck->proton (1 NUCLEON slot), no primary pion.
    Returns the escaped-terminal batch `out` (with species+charge) + overflow."""
    n = p_N.shape[0]
    su = setup_nucleus(jnp.zeros((n, 4)), jnp.zeros(n, jnp.int32), jnp.full(n, 2112, jnp.int32), cfg, key)
    _kpi, knuc, _ = jax.random.split(su["kp"], 3)
    g0 = empty_batch(n, 1)
    g0["alive"] = jnp.ones((n, 1), bool)
    g0["species"] = jnp.full((n, 1), NUCLEON, jnp.int32)
    g0["charge"] = (Npid == 2212).astype(jnp.int32)[:, None]
    g0["p4"] = p_N[:, None, :]; g0["pos"] = su["pos0"][:, None, :]
    stepper = make_pool_stepper(su, cfg, with_rec=False)
    out, sofl, oofl, _ = run_cascade_pool(g0, stepper, knuc, su["consumed0"], M=P_BUF,
                                          max_steps=cfg.max_steps, M_out=M_OUT, prim_origin=-999)
    return out, int(sofl), int(oofl)


def main():
    tg, cfg, ms = build_cfg()
    print(f"[cfg] C QE  max_steps={ms}  P={P_BUF} M_out={M_OUT} N_RECOIL={N_RECOIL}  SEEDS={SEEDS} n_qe/seed={n_qe}", flush=True)
    sf_n = SpectralFunction(tg.spectral_n)
    cols = {k: [] for k in ("p_init", "npip", "npi0", "npim", "w")}; sofl = oofl = 0
    for sd in range(SEEDS):
        r = qe_xsec.sample_importance(n_qe, seed=sd, sf=sf_n, n_neutron=tg.A - tg.Z)
        pN = np.asarray(r["p_out"]); w = np.asarray(r["w"]) / n_qe; m = len(w)
        Npid = np.full(m, 2212)
        for c0 in range(0, m, CHUNK):
            sl = slice(c0, c0 + CHUNK)
            out, so, oo = qe_chunk(jnp.asarray(pN[sl]), jnp.asarray(Npid[sl], jnp.int32), cfg,
                                   jax.random.PRNGKey(11 + sd))
            sofl += so; oofl += oo
            sp = np.asarray(out["species"]); chg = np.asarray(out["charge"]); al = np.asarray(out["alive"])
            is_pi = (sp == PION) & al
            cols["npip"].append((is_pi & (chg == 0)).sum(1)); cols["npi0"].append((is_pi & (chg == 1)).sum(1))
            cols["npim"].append((is_pi & (chg == 2)).sum(1))
            cols["p_init"].append(np.linalg.norm(pN[sl][:, 1:], axis=1)); cols["w"].append(w[sl])
            print(f"    seed {sd+1}/{SEEDS} chunk @{c0}: +{len(w[sl])} (overflow stack={sofl} out={oofl})", flush=True)
    D = {k: np.concatenate(v) for k, v in cols.items()}; D["w"] = D["w"] / SEEDS
    np.savez(out_path, max_steps=ms, P=P_BUF, M_out=M_OUT, overflow_stack=sofl, overflow_out=oofl, **D)
    w = D["w"]; pin = D["p_init"]; npip = D["npip"]
    print(f"\n=== ADoNIS QE created-pi+ rate vs nucleon |p|  [overflow stack={sofl} out={oofl}] ===")
    print(f"overall created-pi+ P(npip>=1) = {float((w*(npip>=1)).sum()/w.sum()):.4f}  <npip>={float((w*npip).sum()/w.sum()):.4f}")
    print(f"{'p_N bin':>12} {'Nevt':>8} {'created-pi+':>12} {'<npip>':>8}")
    for lo, hi in zip(ED[:-1], ED[1:]):
        b = (pin >= lo) & (pin < hi)
        if not b.any(): continue
        wb = w[b]
        print(f"{f'[{lo:.0f},{hi:.0f})':>12} {int(b.sum()):8d} {float((wb*(npip[b]>=1)).sum()/wb.sum()):12.4f} "
              f"{float((wb*npip[b]).sum()/wb.sum()):8.4f}")
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

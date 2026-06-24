"""Hybrid ACHILLES<->ADoNIS cascade ablation: run ADoNIS's pool cascade on ACHILLES's EXACT per-event
input nucleus (parsed from an ACHILLES_CASCADEDUMP stream) and compare final-state nucleon multiplicity
+ momenta.  Because the INPUT (primary 4-momentum/position + full background config + Fermi momenta) is
identical, any difference is PURE cascade transport -- it isolates the implementation deviation behind the
QE FSI N(n) tension (mean N(n) ACH/ADO ~1.12) from input-generation differences.

Usage: python -u scripts/cascade_ablation.py <cascadedump.txt> [target=C] [--no-pauli]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from adonis.workflow.materials import resolve_targets
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density
from adonis.fsi.cascade_full import cascade_nucleus, NUCLEON
from adonis.data.oracle.parse_cascadedump import parse_cascadedump, to_arrays


def build_cfg(target, pauli=True):
    tg = resolve_targets(target)[0][0]
    _, _, _, r = _load_density(tg.density_p, tg.density_n)
    ms = max(600, int(np.ceil(3.0 * r / 0.04)))
    return DiscreteCascadeConfig(step=0.04, max_steps=ms, seed=1, nn_inelastic=True, pauli=pauli,
                                 nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)


def su_from_achilles(arr, key):
    """Build the ADoNIS setup dict from ACHILLES's per-event nucleus (su_external).  The struck nucleon is
    ALREADY removed by ACHILLES (it is the propagating primary) -> consumed0 is all-False over the A-1
    spectators; pos0 = the primary's start position (the QE vertex)."""
    n = arr["npos"].shape[0]
    npos = jnp.asarray(arr["npos"]); nmom = jnp.asarray(arr["nmom"]); nisp = jnp.asarray(arr["nisp"])
    pos0 = jnp.asarray(arr["prim_pos"])
    consumed0 = jnp.asarray(~arr["nmask"])           # invalid (padding) slots marked consumed; real bg live
    ch0 = jnp.ones(n, jnp.int32)                     # QE: no pion (unused)
    return dict(npos=npos, nmom=nmom, nisp=nisp, pos0=pos0, consumed0=consumed0, ch0=ch0, kp=key)


def ado_finalstate(nterms):
    g0 = nterms[0]
    sp = np.asarray(g0["species"]); ch = np.asarray(g0["charge"]); al = np.asarray(g0["alive"])
    al = al & (np.linalg.norm(np.asarray(g0["p4"])[:, :, 1:], axis=2) > 0.0)
    n_p = ((sp == NUCLEON) & (ch == 1) & al).sum(1)
    n_n = ((sp == NUCLEON) & (ch == 0) & al).sum(1)
    return n_p, n_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("target", nargs="?", default="C")
    ap.add_argument("--no-pauli", action="store_true")
    a = ap.parse_args()
    print(f"parsing {a.dump} ...", flush=True)
    events = parse_cascadedump(a.dump)
    # QE only: events with no primary pion (proc==0)
    events = [e for e in events if e["proc"] == 0]
    arr = to_arrays(events)
    n = arr["npos"].shape[0]
    print(f"  {n} QE events; bg A={arr['A']}  mean bg p/n = {arr['nisp'].sum(1).mean():.2f}/"
          f"{(arr['nmask'].sum(1)-arr['nisp'].sum(1)).mean():.2f}", flush=True)
    cfg = build_cfg(a.target, pauli=not a.no_pauli)
    key = jax.random.PRNGKey(12345)
    su = su_from_achilles(arr, key)
    p_pi = jnp.zeros((n, 4))
    p_N = jnp.asarray(arr["prim_p4"])
    ppid = jnp.zeros(n, jnp.int32)
    ipid = jnp.full(n, 2112, jnp.int32)              # QE struck = neutron
    Npid = jnp.full(n, 2212, jnp.int32)              # outgoing proton
    print("running ADoNIS cascade on ACHILLES input ...", flush=True)
    _, nterms, _, _ = cascade_nucleus(p_pi, p_N, ppid, ipid, Npid, cfg, key, P=2, channel="qe",
                                      su_external=su)
    ado_np, ado_nn = ado_finalstate(nterms)
    w = arr["w"]; W = w.sum()
    ach_np, ach_nn = arr["ach_np"], arr["ach_nn"]

    def mean(x): return float((x * w).sum() / W)
    print("\n=== IDENTICAL INPUT: ADoNIS transport vs ACHILLES transport (weighted means) ===", flush=True)
    print(f"  primary |p| match: ADoNIS==ACHILLES input (fed directly) "
          f"mean={np.linalg.norm(arr['prim_p4'][:, 1:], axis=1).mean():.1f} MeV", flush=True)
    print(f"  mean N(p): ADoNIS {mean(ado_np):.4f}   ACHILLES {mean(ach_np):.4f}   ACH/ADO {mean(ach_np)/mean(ado_np):.3f}", flush=True)
    print(f"  mean N(n): ADoNIS {mean(ado_nn):.4f}   ACHILLES {mean(ach_nn):.4f}   ACH/ADO {mean(ach_nn)/max(mean(ado_nn),1e-9):.3f}", flush=True)
    print(f"  mean secondary nucleons: ADoNIS {mean(ado_np-1)+mean(ado_nn):.4f}   ACHILLES {mean(ach_np-1)+mean(ach_nn):.4f}", flush=True)
    # ACHILLES scatter-trace stats (transport intensity it actually did)
    nsc = np.array([len(e["scat"]) for e in events]); nblk = np.array([sum(s["blocked"] for s in e["scat"]) for e in events])
    print(f"\n  ACHILLES trace: mean NN attempts/event={ (nsc*w).sum()/W:.3f}  Pauli-blocked frac={ (nblk*w).sum()/max((nsc*w).sum(),1e-9):.3f}", flush=True)


if __name__ == "__main__":
    main()
